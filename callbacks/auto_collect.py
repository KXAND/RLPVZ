import os
import time
import threading
from collections import deque
from typing import Dict, Any, List

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class AutoCollectCallback(BaseCallback):
    """
    使用代码 Patch 方式启用自动收集（类似 PvZ Tools）

    在训练开始时一次性启用 Patch，无需每步手动收集
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.patch_enabled = False

    def _on_training_start(self) -> bool:
        """训练开始时启用 Patch"""
        env = self._get_base_env()

        if env and hasattr(env, "pvz") and env.pvz:
            try:
                success = env.pvz.enable_auto_collect_patch(enable=True)
                if success:
                    self.patch_enabled = True
                    print("[AutoCollect] ✓ 自动收集 Patch 已启用 (代码修改方式)")
                else:
                    print("[AutoCollect] ✗ 启用自动收集 Patch 失败")
            except Exception as e:
                print(f"[AutoCollect] ✗ 启用 Patch 时出错: {e}")

        return True

    def _on_training_end(self) -> bool:
        """训练结束时恢复 Patch"""
        if self.patch_enabled:
            env = self._get_base_env()

            if env and hasattr(env, "pvz") and env.pvz:
                try:
                    env.pvz.enable_auto_collect_patch(enable=False)
                    print("[AutoCollect] 自动收集 Patch 已恢复")
                except Exception as e:
                    print(f"[AutoCollect] 恢复 Patch 时出错: {e}")

        return True

    def _get_base_env(self):
        """获取底层环境"""
        env = self.training_env

        # 处理 VecEnv 包装器
        if hasattr(env, "venv"):
            env = env.venv
        elif hasattr(env, "env"):
            env = env.env

        # 处理 DummyVecEnv / SubprocVecEnv
        if hasattr(env, "envs"):
            env = env.envs[0]  # 获取第一个环境

        # 解包到最底层
        if hasattr(env, "unwrapped"):
            env = env.unwrapped

        return env

    def _on_step(self) -> bool:
        """每步不需要额外操作，Patch 会自动生效"""
        return True


# =============================================================================
# 简洁监控回调 - 关键信息 + 调试
# =============================================================================
