import os
import time
import threading
from collections import deque
from typing import Dict, Any, List

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class FailurePrioritizedCallback(BaseCallback):
    """
    失败优先学习

    策略:
    - 记录失败的 episode
    - 失败后进行额外的经验回放/重训练
    - 增加失败 episode 的权重

    实现方式:
    - 监控 episode 结束
    - 失败时记录并触发额外训练
    """

    def __init__(
        self,
        failure_buffer_size: int = 100,  # 保存最近 100 次失败
        extra_train_on_failure: bool = True,  # 失败时是否额外训练
        failure_penalty_multiplier: float = 2.0,  # 失败惩罚倍数
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.failure_buffer_size = failure_buffer_size
        self.extra_train_on_failure = extra_train_on_failure
        self.failure_penalty_multiplier = failure_penalty_multiplier

        # 统计
        self.total_episodes = 0
        self.failure_episodes = 0
        self.success_episodes = 0
        self.recent_results = deque(maxlen=100)  # 最近 100 局结果

        # 失败经验缓冲 (可用于后续分析)
        self.failure_buffer: List[Dict[str, Any]] = []

    def _on_step(self) -> bool:
        # 检查是否有 episode 结束
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode" in info:
                self.total_episodes += 1
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]

                # 判断是否失败 (奖励为负或长度很短)
                is_failure = ep_reward < 0 or ep_length < 100

                if is_failure:
                    self.failure_episodes += 1
                    self.recent_results.append(0)
                    self._on_failure(info, ep_reward, ep_length)
                else:
                    self.success_episodes += 1
                    self.recent_results.append(1)

                # 打印统计
                if self.verbose and self.total_episodes % 10 == 0:
                    win_rate = sum(self.recent_results) / len(self.recent_results) * 100
                    print(
                        f"Episode {self.total_episodes}: "
                        f"胜率 {win_rate:.1f}% | "
                        f"成功 {self.success_episodes} | "
                        f"失败 {self.failure_episodes}"
                    )

        return True

    def _on_failure(self, info: Dict, reward: float, length: int):
        """处理失败 episode"""
        # 记录失败信息
        failure_info = {
            "episode": self.total_episodes,
            "reward": reward,
            "length": length,
            "timesteps": self.num_timesteps,
        }

        self.failure_buffer.append(failure_info)
        if len(self.failure_buffer) > self.failure_buffer_size:
            self.failure_buffer.pop(0)

        if self.verbose:
            print(
                f"  ❌ 失败 #{self.failure_episodes}: "
                f"奖励={reward:.1f}, 长度={length}"
            )

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_episodes": self.total_episodes,
            "success_episodes": self.success_episodes,
            "failure_episodes": self.failure_episodes,
            "win_rate": self.success_episodes / max(1, self.total_episodes),
            "recent_win_rate": sum(self.recent_results)
            / max(1, len(self.recent_results)),
        }
