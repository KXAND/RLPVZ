import os
import time

from collections import deque
from typing import Dict, Any, List

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


# =============================================================================
# 综合速度监控回调
# =============================================================================
class AdvancedSpeedCallback(BaseCallback):
    """高级速度和状态监控 - 实时单行更新"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_time = None
        self.last_time = None
        self.last_steps = 0
        self.speed_history = deque(maxlen=10)
        self.episode_rewards = deque(maxlen=100)
        self.episode_count = 0
        self.win_count = 0

    def _on_training_start(self):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.last_steps = 0
        print("\n" + "=" * 70)
        print("训练开始 (实时监控)")
        print("=" * 70)

    def _on_step(self):
        # 检测 episode 结束
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
                ep_reward = info["episode"]["r"]
                self.episode_rewards.append(ep_reward)
                # 简单判断胜利 (奖励 > 0)
                if ep_reward > 0:
                    self.win_count += 1
        return True

    def _on_rollout_end(self):
        current_time = time.time()
        current_steps = self.num_timesteps

        # 计算速度
        elapsed = current_time - self.last_time
        steps = current_steps - self.last_steps
        if elapsed > 0:
            speed = steps / elapsed
            self.speed_history.append(speed)
            avg_speed = sum(self.speed_history) / len(self.speed_history)
            total_elapsed = current_time - self.start_time

            # 估算剩余时间
            total_ts = (
                self.model.num_timesteps
                if hasattr(self.model, "_total_timesteps")
                else 1000000
            )
            if hasattr(self.model, "_total_timesteps"):
                total_ts = self.model._total_timesteps
            remaining = total_ts - current_steps
            eta_seconds = remaining / avg_speed if avg_speed > 0 else 0

            # 格式化时间
            if eta_seconds > 3600:
                eta_str = f"{int(eta_seconds//3600)}h{int((eta_seconds%3600)//60)}m"
            else:
                eta_str = f"{int(eta_seconds//60)}m{int(eta_seconds%60)}s"

            # 计算胜率
            win_rate = (
                (self.win_count / self.episode_count * 100)
                if self.episode_count > 0
                else 0
            )
            avg_reward = (
                sum(self.episode_rewards) / len(self.episode_rewards)
                if self.episode_rewards
                else 0
            )

            # 获取当前探索系数
            ent_coef = getattr(self.model, "ent_coef", 0.01)

            # 实时单行输出 (使用 \r 覆盖)
            progress = current_steps / total_ts * 100
            status = (
                f"\r🎮 {current_steps:>8,}/{total_ts:,} ({progress:5.1f}%) | "
                f"⚡{speed:>4.0f}it/s | "
                f"🏆{win_rate:>5.1f}% | "
                f"📊{avg_reward:>+6.1f} | "
                f"🔍{ent_coef:.3f} | "
                f"⏳{eta_str:>8}"
            )
            print(status, end="", flush=True)

        self.last_time = current_time
        self.last_steps = current_steps

    def _on_training_end(self):
        total_time = time.time() - self.start_time
        avg_speed = self.num_timesteps / total_time if total_time > 0 else 0
        win_rate = (
            (self.win_count / self.episode_count * 100) if self.episode_count > 0 else 0
        )
        print()  # 换行
        print("\n" + "=" * 70)
        print(f"✅ 训练完成!")
        print(f"  📈 总步数: {self.num_timesteps:,}")
        print(f"  ⏱️ 总时间: {total_time/60:.1f} 分钟")
        print(f"  ⚡ 平均速度: {avg_speed:.0f} it/s")
        print(f"  🎯 总局数: {self.episode_count} | 胜率: {win_rate:.1f}%")
        print("=" * 70)
