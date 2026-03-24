import multiprocessing as mp
import os
import queue
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from ..threshold import Threshold
from .ddqn import QNetwork, copy_state_dict_to_cpu, experienceReplayBuffer


def _build_worker_env(args, instance):
    from envs import PVZEnv
    from .adapter import DDQNEnvAdapter

    env = PVZEnv(
        hook_port=instance["port"],
        target_pid=instance["pid"],
        game_speed=args.speed,
        frame_skip=args.frameskip,
        verbose=args.env_verbose,
    )
    return DDQNEnvAdapter(env)


def _drain_latest_weights(weights_queue):
    latest = None
    while True:
        try:
            latest = weights_queue.get_nowait()
        except queue.Empty:
            return latest


def _put_latest_weights(weights_queue, state_dict):
    while True:
        try:
            weights_queue.get_nowait()
        except queue.Empty:
            break
    weights_queue.put(copy_state_dict_to_cpu(state_dict))


def ddqn_worker_main(
    worker_id,
    args,
    instance,
    initial_state_dict,
    transition_queue,
    stats_queue,
    weights_queue,
    stop_event,
):
    env = None
    try:
        env = _build_worker_env(args, instance)
        network = QNetwork(env, learning_rate=args.ddqn_lr, device="cpu")
        network.load_state_dict(initial_state_dict)
        network.eval()

        threshold = Threshold(
            seq_length=100000,
            start_epsilon=1.0,
            interpolation="exponential",
            end_epsilon=0.05,
        )

        state = _reset_env_with_retry(env, worker_id, instance, stats_queue, stop_event)
        episode_reward = 0.0
        local_episode = 0

        while not stop_event.is_set():
            latest_state_dict = _drain_latest_weights(weights_queue)
            if latest_state_dict is not None:
                network.load_state_dict(latest_state_dict)

            mask = np.array(env.mask_available_actions(), dtype=bool)
            epsilon = threshold.epsilon(local_episode)
            action = network.decide_action(state, mask, epsilon=epsilon)

            try:
                next_state, reward, done, info = env.step(action)
            except Exception as exc:
                stats_queue.put(
                    {
                        "type": "warning",
                        "worker_id": worker_id,
                        "port": instance["port"],
                        "pid": instance["pid"],
                        "message": f"step 失败，准备重置: {repr(exc)}",
                    }
                )
                state = _reset_env_with_retry(
                    env, worker_id, instance, stats_queue, stop_event
                )
                episode_reward = 0.0
                continue
            next_mask = np.array(env.mask_available_actions(), dtype=bool)

            while not stop_event.is_set():
                try:
                    transition_queue.put(
                        (
                            state,
                            action,
                            reward,
                            done,
                            next_state,
                            mask,
                            next_mask,
                        ),
                        timeout=1.0,
                    )
                    break
                except queue.Full:
                    latest_state_dict = _drain_latest_weights(weights_queue)
                    if latest_state_dict is not None:
                        network.load_state_dict(latest_state_dict)

            state = next_state.copy()
            episode_reward += reward

            if not done:
                continue

            local_episode += 1
            stats_queue.put(
                {
                    "type": "episode",
                    "worker_id": worker_id,
                    "reward": episode_reward,
                    "iterations": env.steps,
                    "epsilon": epsilon,
                    "port": instance["port"],
                    "pid": instance["pid"],
                }
            )
            episode_reward = 0.0
            state = _reset_env_with_retry(env, worker_id, instance, stats_queue, stop_event)

    except Exception as exc:
        stats_queue.put(
            {
                "type": "error",
                "worker_id": worker_id,
                "port": instance["port"],
                "pid": instance["pid"],
                "message": repr(exc),
            }
        )
    finally:
        if env is not None:
            env.close()


def _reset_env_with_retry(env, worker_id, instance, stats_queue, stop_event):
    consecutive_failures = 0
    while not stop_event.is_set():
        try:
            return env.reset()
        except Exception as exc:
            consecutive_failures += 1
            stats_queue.put(
                {
                    "type": "warning",
                    "worker_id": worker_id,
                    "port": instance["port"],
                    "pid": instance["pid"],
                    "message": (
                        f"reset 失败，第 {consecutive_failures} 次重试: {repr(exc)}"
                    ),
                }
            )
            if consecutive_failures >= 5:
                raise RuntimeError(
                    f"reset 连续失败 {consecutive_failures} 次: {repr(exc)}"
                ) from exc
            import time

            time.sleep(min(2.0 * consecutive_failures, 5.0))
    raise RuntimeError("worker 停止前未能完成 reset")


class AsyncDDQNTrainer:
    def __init__(self, args, instances, network):
        self.args = args
        self.instances = instances
        self.network = network
        self.target_network = deepcopy(network)
        self.buffer = experienceReplayBuffer(
            memory_size=args.ddqn_buffer_size, burn_in=args.ddqn_burn_in
        )
        self.batch_size = args.ddqn_batch_size
        self.gamma = args.ddqn_gamma
        self.window = 100
        self.reward_threshold = 30000

        self.training_rewards = []
        self.training_iterations = []
        self.training_loss = []
        self.mean_training_rewards = []
        self.mean_training_iterations = []
        self.real_rewards = []
        self.real_iterations = []
        self.sync_eps = []

        self.transition_count = 0
        self.episode_count = 0
        self.solved = False

    def train(
        self,
        max_episodes,
        network_update_frequency,
        network_sync_frequency,
        evaluate_frequency,
        evaluate_n_iter,
    ):
        ctx = mp.get_context("spawn")
        transition_queue = ctx.Queue(maxsize=max(2048, self.batch_size * 64))
        stats_queue = ctx.Queue()
        weight_queues = [ctx.Queue(maxsize=1) for _ in self.instances]
        stop_event = ctx.Event()

        initial_state_dict = copy_state_dict_to_cpu(self.network.state_dict())
        workers = []
        for worker_id, instance in enumerate(self.instances):
            process = ctx.Process(
                target=ddqn_worker_main,
                args=(
                    worker_id,
                    self.args,
                    instance,
                    initial_state_dict,
                    transition_queue,
                    stats_queue,
                    weight_queues[worker_id],
                    stop_event,
                ),
            )
            process.start()
            workers.append(process)
            print(
                f"[DDQN] Worker {worker_id} 已启动: pid={instance['pid']} port={instance['port']}"
            )

        try:
            self._run_training_loop(
                transition_queue=transition_queue,
                stats_queue=stats_queue,
                weight_queues=weight_queues,
                stop_event=stop_event,
                max_episodes=max_episodes,
                network_update_frequency=network_update_frequency,
                network_sync_frequency=network_sync_frequency,
                evaluate_frequency=evaluate_frequency,
                evaluate_n_iter=evaluate_n_iter,
            )
        finally:
            stop_event.set()
            for process in workers:
                process.join(timeout=5.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)

    def _run_training_loop(
        self,
        transition_queue,
        stats_queue,
        weight_queues,
        stop_event,
        max_episodes,
        network_update_frequency,
        network_sync_frequency,
        evaluate_frequency,
        evaluate_n_iter,
    ):
        while self.episode_count < max_episodes and not self.solved:
            self._drain_stats_queue(stats_queue, evaluate_frequency, evaluate_n_iter)

            try:
                transition = transition_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self.buffer.append(*transition)
            self.transition_count += 1

            if self.buffer.burn_in_capacity() < 1:
                continue

            if self.transition_count % network_update_frequency == 0:
                loss_value = self.update()
                self.training_loss.append(loss_value)

            if self.transition_count % network_sync_frequency == 0:
                self.target_network.load_state_dict(self.network.state_dict())
                self.sync_eps.append(self.episode_count)
                latest_state_dict = copy_state_dict_to_cpu(self.network.state_dict())
                for weights_queue in weight_queues:
                    _put_latest_weights(weights_queue, latest_state_dict)

        stop_event.set()
        self._drain_stats_queue(stats_queue, evaluate_frequency, evaluate_n_iter)
        if self.solved:
            print(f"\nEnvironment solved in {self.episode_count} episodes.")
        else:
            print("\nEpisode limit reached.")

    def _drain_stats_queue(self, stats_queue, evaluate_frequency, evaluate_n_iter):
        while True:
            try:
                message = stats_queue.get_nowait()
            except queue.Empty:
                return

            if message["type"] == "error":
                raise RuntimeError(
                    f"DDQN worker {message['worker_id']} 失败: {message['message']}"
                )

            if message["type"] == "warning":
                print(
                    f"\n[DDQN][Worker {message['worker_id']}] {message['message']} "
                    f"(pid={message['pid']}, port={message['port']})",
                    flush=True,
                )
                continue

            self.episode_count += 1
            self.training_rewards.append(message["reward"])
            self.training_iterations.append(message["iterations"])

            mean_rewards = np.mean(self.training_rewards[-self.window :])
            mean_iteration = np.mean(self.training_iterations[-self.window :])
            self.mean_training_rewards.append(mean_rewards)
            self.mean_training_iterations.append(mean_iteration)

            progress_line = (
                "Episode {:d} Mean Rewards {:.2f}\t\t Mean Iterations {:.2f}\t\t".format(
                    self.episode_count, mean_rewards, mean_iteration
                )
            )
            print("\r" + progress_line, end="", flush=True)

            if mean_rewards >= self.reward_threshold:
                self.solved = True
                return

            if (
                self.episode_count > 0
                and (self.episode_count % evaluate_frequency) == 0
            ):
                recent_rewards = self.training_rewards[-evaluate_n_iter:]
                recent_iterations = self.training_iterations[-evaluate_n_iter:]
                avg_score = float(np.mean(recent_rewards)) if recent_rewards else 0.0
                avg_iter = (
                    float(np.mean(recent_iterations)) if recent_iterations else 0.0
                )
                self.real_rewards.append(avg_score)
                self.real_iterations.append(avg_iter)
                print(
                    f"\n[Eval] Episode {self.episode_count} | avg_score={avg_score:.2f} | avg_iter={avg_iter:.2f}",
                    flush=True,
                )
                print("\r" + progress_line, end="", flush=True)

    def calculate_loss(self, batch):
        states, actions, rewards, dones, next_states, masks, next_masks = [
            item for item in batch
        ]

        rewards_t = (
            torch.FloatTensor(rewards).to(device=self.network.device).reshape(-1, 1)
        )
        actions_t = (
            torch.LongTensor(np.array(actions))
            .reshape(-1, 1)
            .to(device=self.network.device)
        )
        dones_t = torch.as_tensor(dones, dtype=torch.bool, device=self.network.device)

        qvals = torch.gather(self.network.get_qvals(states), 1, actions_t)

        next_masks = np.array(next_masks, dtype=bool)
        qvals_next_pred = self.network.get_qvals(next_states)
        next_masks_t = torch.as_tensor(
            next_masks, dtype=torch.bool, device=qvals_next_pred.device
        )
        qvals_next_pred = qvals_next_pred.clone()
        qvals_next_pred[~next_masks_t] = qvals_next_pred.min()
        next_actions = torch.max(qvals_next_pred, dim=-1)[1]
        next_actions_t = torch.as_tensor(
            next_actions, dtype=torch.long, device=self.network.device
        ).reshape(-1, 1)

        target_qvals = self.target_network.get_qvals(next_states)
        qvals_next = torch.gather(target_qvals, 1, next_actions_t).detach()
        qvals_next[dones_t] = 0
        expected_qvals = self.gamma * qvals_next + rewards_t
        return nn.MSELoss()(qvals, expected_qvals)

    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch)
        loss.backward()
        self.network.optimizer.step()
        return (
            float(loss.detach().cpu().item())
            if self.network.device == "cuda"
            else float(loss.detach().item())
        )
