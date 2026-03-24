import os
import random
import time
from typing import Tuple

import numpy as np
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecFrameStack,
    VecNormalize,
)

from envs import PVZEnv
import train_utils
import models.ddqn.train_entry as ddqn_factory


def mask_fn(env):
    """获取动作掩码的回调函数"""
    return env.unwrapped._get_action_mask(
        env.unwrapped.pvz.get_game_state() if env.unwrapped.pvz else None
    )


class DiversifiedPVZEnv(PVZEnv):
    """
    多样化训练环境

    策略:
    - 随机跳过开局若干帧，让 AI 学会应对各种中途状态
    - 随机初始阳光 (模拟不同经济状况)
    - 随机选择是否预先放置一些植物
    """

    def __init__(
        self,
        diversify_prob: float = 0.3,  # 30% 概率使用多样化
        skip_frames_range: Tuple[int, int] = (0, 500),  # 跳过帧数范围
        random_sun_range: Tuple[int, int] = (50, 500),  # 随机阳光范围
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.diversify_prob = diversify_prob
        self.skip_frames_range = skip_frames_range
        self.random_sun_range = random_sun_range
        self._diversify_enabled = True

    def set_diversify(self, enabled: bool):
        """动态开关多样化训练"""
        self._diversify_enabled = enabled

    def reset(self, seed=None, options=None):
        """重置环境，有概率进行多样化"""
        obs, info = super().reset(seed=seed, options=options)

        # 决定是否多样化
        if self._diversify_enabled and random.random() < self.diversify_prob:
            self._apply_diversification()
            # 重新获取观测
            obs = self._get_observation()

        # 标记需要重置记忆
        info["needs_memory_reset"] = True

        return obs, info

    def _apply_diversification(self):
        """应用多样化策略"""
        if not self.hook_client or not self.hook_client.connected:
            return

        # 策略1: 随机跳过若干帧 (模拟中途状态)
        skip_frames = random.randint(*self.skip_frames_range)
        if skip_frames > 0:
            print(f"  🎲 多样化: 跳过 {skip_frames} 帧，模拟中途状态")
            # 执行若干次等待，让游戏自然推进
            wait_iterations = min(
                skip_frames // max(1, self.frame_skip), 50
            )  # 最多50次
            for i in range(wait_iterations):
                # 自动收集阳光
                if i % 5 == 0:  # 每5次收集一次
                    self.hook_client.collect()
                time.sleep(0.002)  # 2ms延迟，配合高速游戏

            # 再次收集阳光确保资源
            self.hook_client.collect()

            # 更新缓存的游戏状态
            if self.pvz:
                self._cached_game_state = self.pvz.get_game_state()


def make_single_env(args, instance):
    if args.no_diversify:
        env = PVZEnv(
            hook_port=instance["port"],
            target_pid=instance["pid"],
            game_speed=args.speed,
            frame_skip=args.frameskip,
            verbose=args.env_console_log_level,
            log_verbose=args.file_log_level,
        )
    else:
        env = DiversifiedPVZEnv(
            hook_port=instance["port"],
            target_pid=instance["pid"],
            game_speed=args.speed,
            frame_skip=args.frameskip,
            diversify_prob=args.diversify,
            verbose=args.env_console_log_level,
            log_verbose=args.file_log_level,
        )
    env = ActionMasker(env, mask_fn)
    return env


def _make_env_factory(args, instance):
    def _factory():
        return make_single_env(args, instance)

    return _factory


def get_env(args):
    if args.algo == "ddqn":
        env = ddqn_factory._build_ddqn_env(args)
        return env

    load_path = train_utils.resolve_load_path(args)
    instances = getattr(args, "game_instances", None) or train_utils.resolve_game_instances(
        args
    )
    factories = [_make_env_factory(args, instance) for instance in instances]
    if len(factories) == 1:
        env = DummyVecEnv(factories)
    else:
        env = SubprocVecEnv(factories, start_method="spawn")

    # 1. 监控 (记录原始奖励)
    env = VecMonitor(env)

    # 2. 归一化 (关键优化: 稳定 PPO 训练)
    # 归一化观测值和奖励，防止梯度爆炸/消失
    if load_path:
        path_no_ext = os.path.splitext(load_path)[0]
        vec_norm_path = path_no_ext + "_vecnormalize.pkl"
        if os.path.exists(vec_norm_path):
            print(f"加载归一化统计: {vec_norm_path}")
            env = VecNormalize.load(vec_norm_path, env)
            env.training = True
            env.norm_reward = True
            env.norm_obs = True
        else:
            print(f"未找到归一化统计文件，创建新的归一化层")
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 3. 帧堆叠 (赋予时间感知)
    # 堆叠最近4帧，让AI能感知僵尸的移动速度和波次节奏
    env = VecFrameStack(env, n_stack=4, channels_order="last")

    return env
