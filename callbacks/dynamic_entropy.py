import os
import time
import threading
from collections import deque
from typing import Dict, Any, List

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class DynamicEntropyCallback(BaseCallback):
    """
    动态调整探索系数 (ent_coef)

    策略:
    - 训练初期: 高探索 (ent_coef 大) - 大胆尝试各种策略
    - 训练后期: 低探索 (ent_coef 小) - 稳定收敛到最优策略

    衰减方式:
    - 线性衰减: ent_coef = start - (start - end) * progress
    - 指数衰减: ent_coef = end + (start - end) * exp(-decay * progress)
    - 余弦衰减: ent_coef = end + (start - end) * 0.5 * (1 + cos(pi * progress))
    """

    def __init__(
        self,
        start_ent_coef: float = 0.1,
        end_ent_coef: float = 0.01,
        decay_type: str = "cosine",  # "linear", "exponential", "cosine"
        total_timesteps: int = 500000,
        warmup_steps: int = 10000,  # 热身期保持高探索
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.start_ent_coef = start_ent_coef
        self.end_ent_coef = end_ent_coef
        self.decay_type = decay_type
        self.total_timesteps = total_timesteps
        self.warmup_steps = warmup_steps
        self.last_print_step = 0

    def _on_step(self) -> bool:
        # 计算进度
        progress = min(1.0, self.num_timesteps / self.total_timesteps)

        # 热身期保持高探索
        if self.num_timesteps < self.warmup_steps:
            new_ent_coef = self.start_ent_coef
        else:
            # 调整进度 (排除热身期)
            adjusted_progress = (self.num_timesteps - self.warmup_steps) / (
                self.total_timesteps - self.warmup_steps
            )
            adjusted_progress = min(1.0, max(0.0, adjusted_progress))

            if self.decay_type == "linear":
                new_ent_coef = (
                    self.start_ent_coef
                    - (self.start_ent_coef - self.end_ent_coef) * adjusted_progress
                )

            elif self.decay_type == "exponential":
                decay_rate = 5.0  # 控制衰减速度
                new_ent_coef = self.end_ent_coef + (
                    self.start_ent_coef - self.end_ent_coef
                ) * np.exp(-decay_rate * adjusted_progress)

            elif self.decay_type == "cosine":
                # 余弦退火 - 平滑衰减
                new_ent_coef = self.end_ent_coef + (
                    self.start_ent_coef - self.end_ent_coef
                ) * 0.5 * (1 + np.cos(np.pi * adjusted_progress))
            else:
                new_ent_coef = self.start_ent_coef

        # 更新模型的 ent_coef
        self.model.ent_coef = new_ent_coef

        # 每 10000 步打印一次
        if self.verbose and self.num_timesteps - self.last_print_step >= 10000:
            print(
                f"  📊 动态探索: ent_coef = {new_ent_coef:.4f} (进度: {progress*100:.1f}%)"
            )
            self.last_print_step = self.num_timesteps

        return True


# =============================================================================
# 优化2: 多样化训练环境包装器
# =============================================================================
