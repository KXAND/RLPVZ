import os
import time
import threading
from collections import deque
from typing import Dict, Any, List

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

# =============================================================================
# 优化2.5: 记忆重置回调
# =============================================================================
class MemoryResetCallback(BaseCallback):
    """
    记忆重置回调 - 在每个episode开始时重置模型记忆

    功能:
    - 检测episode重置信号
    - 调用特征提取器的reset_memory方法
    - 支持多环境并行
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_starts = None

    def _on_step(self) -> bool:
        # 获取当前的dones标志
        if hasattr(self.locals, "dones"):
            dones = self.locals["dones"]
        elif "done" in self.locals:
            dones = self.locals["done"]
        else:
            return True

        # 检测哪些环境需要重置记忆
        if isinstance(dones, (list, np.ndarray)):
            dones = np.array(dones)
        else:
            dones = np.array([dones])

        # 对于刚完成的环境,下一步将是新episode的开始
        # 我们在这里标记,在下一次rollout开始时重置
        if hasattr(self, "model") and hasattr(self.model.policy, "features_extractor"):
            extractor = self.model.policy.features_extractor
            if hasattr(extractor, "reset_memory") and np.any(dones):
                # 记录哪些环境需要重置
                if self.episode_starts is None:
                    self.episode_starts = dones
                else:
                    self.episode_starts = np.logical_or(self.episode_starts, dones)

        return True

    def _on_rollout_start(self) -> None:
        """在rollout开始时重置标记的环境的记忆"""
        if self.episode_starts is not None and hasattr(
            self.model.policy, "features_extractor"
        ):
            extractor = self.model.policy.features_extractor
            if hasattr(extractor, "reset_memory"):
                # 获取环境数量
                n_envs = (
                    self.training_env.num_envs
                    if hasattr(self.training_env, "num_envs")
                    else 1
                )

                # 重置所有标记的环境的记忆
                if np.any(self.episode_starts):
                    try:
                        extractor.reset_memory(
                            batch_size=n_envs, device=extractor.lstm_h.device
                        )
                        if self.verbose > 0:
                            print(f"已重置 {np.sum(self.episode_starts)} 个环境的记忆")
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"记忆重置失败: {e}")

                # 清空标记
                self.episode_starts = None


# =============================================================================
# 优化3: 失败优先学习回调
# =============================================================================
