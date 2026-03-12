import torch.nn as nn
import torch
import numpy as np
from copy import deepcopy
from collections import namedtuple, deque
from ..threshold import Threshold
from .evaluate_agent import evaluate


class QNetwork(nn.Module):
    def __init__(self, env, learning_rate=1e-3, device="cpu"):
        super(QNetwork, self).__init__()
        self.device = device

        self.rows = env.rows
        self.cols = env.cols
        self.num_cards = env.num_cards
        self.grid_size = self.rows * self.cols
        self.n_inputs = self.grid_size * 2 + self.num_cards + 1
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate

        # Set up network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 50, bias=True),
            nn.LeakyReLU(),
            nn.Linear(50, self.n_outputs, bias=True),
        )

        # Set to GPU if cuda is specified
        if self.device == "cuda":
            self.network.cuda()

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate
        )

    def decide_action(self, state, mask, epsilon):
        # mask = self.env.mask_available_actions()
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions[mask])
        else:
            action = self.get_greedy_action(state, mask)
        return action

    def get_greedy_action(self, state, mask):
        qvals = self.get_qvals(state)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=qvals.device)
        qvals = qvals.clone()
        qvals[~mask_t] = qvals.min()
        return torch.max(qvals, dim=-1)[1].item()

    def get_qvals(self, state):
        if isinstance(state, (list, tuple)):
            state = np.array(state)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return self.network(state_t)


class DDQNAgent:

    def __init__(self, env, network, buffer, n_iter=100000, batch_size=32):

        self.env = env
        self.network = network
        self.target_network = deepcopy(network)
        self.buffer = buffer
        # self.pre_buffer = []
        # self.pre_buffer_rewards = []
        # self.threshold = Threshold(seq_length = 100000, start_epsilon=1.0,
        #                   end_epsilon=0.2,interpolation='sinusoidal',
        #                   periods=np.floor(n_iter/100))
        self.threshold = Threshold(
            seq_length=100000,
            start_epsilon=1.0,
            interpolation="exponential",
            end_epsilon=0.05,
        )
        self.epsilon = 0
        self.batch_size = batch_size
        self.window = 100
        self.reward_threshold = 30000
        self.initialize()
        self.player = PlayerQ(env=env, render=False)

    def take_step(self, mode="train"):
        mask = np.array(self.env.mask_available_actions())
        if mode == "explore":
            if np.random.random() < 0.5:
                action = 0  # Do nothing
            else:
                action = np.random.choice(np.arange(self.env.action_space.n)[mask])
        else:
            action = self.network.decide_action(self.s_0, mask, epsilon=self.epsilon)
            self.step_count += 1
        s_1, r, done, _ = self.env.step(action)
        next_mask = np.array(self.env.mask_available_actions())
        self.rewards += r
        self.buffer.append(self.s_0, action, r, done, s_1, mask, next_mask)
        self.s_0 = s_1.copy()
        if done:
            if mode != "explore":  # We document the end of the play
                self.training_iterations.append(
                    self.env.steps
                )
            self.s_0 = self.env.reset()
        return done

    # def add_play_to_buffer(self):
    #     rewards = self.discount_rewards(np.array(self.pre_buffer_rewards))
    #     for i in range(len(rewards)):
    #         s_0, action, done, s_1 = self.pre_buffer[i]
    #         r = rewards[i]
    #         self.buffer.append(s_0, action, r, done, s_1)
    #     self.pre_buffer_rewards = []
    #     self.pre_buffer = []

    # Implement DQN training algorithm
    def train(
        self,
        gamma=0.99,
        max_episodes=100000,
        network_update_frequency=32,
        network_sync_frequency=2000,
        evaluate_frequency=500,
        evaluate_n_iter=1000,
    ):

        self.gamma = gamma
        # Populate replay buffer
        while self.buffer.burn_in_capacity() < 1:
            done = self.take_step(mode="explore")
            # if done:
            #     self.add_play_to_buffer()
        ep = 0
        training = True
        self.s_0 = self.env.reset()

        while training:
            self.rewards = 0
            done = False
            while done == False:
                self.epsilon = self.threshold.epsilon(ep)
                done = self.take_step(mode="train")
                # Update network
                if self.step_count % network_update_frequency == 0:
                    self.update()
                # Sync networks
                if self.step_count % network_sync_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())
                    self.sync_eps.append(ep)

                if done:
                    ep += 1
                    self.training_rewards.append(self.rewards)
                    if self.update_loss:
                        self.training_loss.append(np.mean(self.update_loss))
                    else:
                        self.training_loss.append(0.0)
                    self.update_loss = []
                    mean_rewards = np.mean(self.training_rewards[-self.window :])
                    self.mean_training_rewards.append(mean_rewards)

                    mean_iteration = np.mean(self.training_iterations[-self.window :])
                    self.mean_training_iterations.append(mean_iteration)
                    progress_line = (
                        "Episode {:d} Mean Rewards {:.2f}\t\t Mean Iterations {:.2f}\t\t".format(
                            ep, mean_rewards, mean_iteration
                        )
                    )
                    print("\r" + progress_line, end="", flush=True)

                    if ep >= max_episodes:
                        training = False
                        print("\nEpisode limit reached.")
                        break
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print("\nEnvironment solved in {} episodes!".format(ep))
                        break
                    if (ep % evaluate_frequency) == evaluate_frequency - 1:
                        avg_score, avg_iter = evaluate(
                            self.player,
                            self.network,
                            n_iter=evaluate_n_iter,
                            verbose=False,
                        )
                        self.real_iterations.append(avg_iter)
                        self.real_rewards.append(avg_score)
                        print(
                            f"\n[Eval] Episode {ep} | avg_score={avg_score:.2f} | avg_iter={avg_iter:.2f}",
                            flush=True,
                        )
                        # Reprint progress line after eval output
                        print("\r" + progress_line, end="", flush=True)

    def calculate_loss(self, batch):
        states, actions, rewards, dones, next_states, masks, next_masks = [
            i for i in batch
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

        qvals = torch.gather(
            self.network.get_qvals(states), 1, actions_t
        )  # The selected action already respects the mask

        #################################################################
        # DDQN Update
        next_masks = np.array(next_masks, dtype=bool)
        qvals_next_pred = self.network.get_qvals(next_states)
        next_masks_t = torch.as_tensor(
            next_masks, dtype=torch.bool, device=qvals_next_pred.device
        )
        qvals_next_pred = qvals_next_pred.clone()
        qvals_next_pred[~next_masks_t] = qvals_next_pred.min()
        next_actions = torch.max(qvals_next_pred, dim=-1)[1]
        # next_actions_t = torch.LongTensor(next_actions).reshape(-1,1).to(
        #     device=self.network.device)
        next_actions_t = torch.as_tensor(
            next_actions, dtype=torch.long, device=self.network.device
        ).reshape(
            -1, 1
        )  # qs modified
        target_qvals = self.target_network.get_qvals(next_states)
        qvals_next = torch.gather(target_qvals, 1, next_actions_t).detach()
        #################################################################
        qvals_next[dones_t] = 0  # Zero-out terminal states
        expected_qvals = self.gamma * qvals_next + rewards_t
        loss = nn.MSELoss()(qvals, expected_qvals)
        return loss

    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch)
        loss.backward()
        self.network.optimizer.step()
        if self.network.device == "cuda":
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())

    def _save_training_data(self, nn_name):
        np.save(nn_name + "_rewards", self.training_rewards)
        np.save(nn_name + "_iterations", self.training_iterations)
        np.save(nn_name + "_real_rewards", self.real_rewards)
        np.save(nn_name + "_real_iterations", self.real_iterations)
        torch.save(self.training_loss, nn_name + "_loss")

    def initialize(self):
        self.training_rewards = []
        self.training_loss = []
        self.training_iterations = []
        self.real_rewards = []
        self.real_iterations = []
        self.update_loss = []
        self.mean_training_rewards = []
        self.mean_training_iterations = []
        self.sync_eps = []
        self.rewards = 0
        self.step_count = 0
        self.s_0 = self.env.reset()


class experienceReplayBuffer:

    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.Buffer = namedtuple(
            "Buffer",
            field_names=[
                "state",
                "action",
                "reward",
                "done",
                "next_state",
                "mask",
                "next_mask",
            ],
        )
        self.replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size, replace=False)
        # Use asterisk operator to unpack deque
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def append(self, state, action, reward, done, next_state, mask, next_mask):
        self.replay_memory.append(
            self.Buffer(state, action, reward, done, next_state, mask, next_mask)
        )

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in


class PlayerQ:
    def __init__(self, env=None, render=True):
        self.env = env
        self.render = render

    def get_actions(self):
        return list(range(self.env.action_space.n))

    def num_observations(self):
        return (
            self.env.rows * self.env.cols * 2 + self.env.num_cards + 1
        )

    def num_actions(self):
        return self.env.action_space.n

    def play(self, agent, epsilon=0):
        """Play one episode and collect observations and rewards"""

        summary = dict()
        summary["rewards"] = list()
        summary["observations"] = list()
        summary["actions"] = list()
        observation = self.env.reset()

        t = 0

        while True:
            if self.render:
                self.env.render()
            # if np.random.random()<epsilon:
            #     # print("exploration")
            #     action = np.random.choice(self.get_actions(), 1)[0]
            # else:
            # action = agent.decide_action(observation, np.full(self.num_actions(), True), epsilon)
            action = agent.decide_action(
                observation, self.env.mask_available_actions(), epsilon
            )
            summary["observations"].append(observation)
            summary["actions"].append(action)
            observation, reward, done, info = self.env.step(action)
            summary["rewards"].append(reward)

            if done:
                break

        summary["observations"] = np.vstack(summary["observations"])
        summary["actions"] = np.vstack(summary["actions"])
        summary["rewards"] = np.vstack(summary["rewards"])
        return summary

    def get_render_info(self):
        base_env = getattr(self.env, "env", self.env)
        scene = getattr(base_env, "_scene", None)
        return getattr(scene, "_render_info", None)
