import os
import time
import threading
from collections import deque
from typing import Dict, Any, List

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class SimpleMonitorCallback(BaseCallback):
    """简洁监控 - 失败次数 + 生存时长"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_time = None
        self.episode_count = 0
        self.fail_count = 0  # 失败次数（僵尸进屋）
        self.total_len = 0  # 累计生存步数
        self.recent_lens = deque(maxlen=20)  # 最近20局生存时长
        self.best_len = 0  # 最佳生存时长
        self.last_print_step = 0

    def _on_training_start(self):
        self.start_time = time.time()
        print("\n" + "=" * 60)
        print("训练开始")
        print("=" * 60)

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
                ep_reward = info["episode"]["r"]
                ep_len = int(info["episode"]["l"])

                # 更新统计
                self.total_len += ep_len
                self.recent_lens.append(ep_len)
                self.best_len = max(self.best_len, ep_len)

                # 检查是否失败（僵尸进屋）
                # 优先使用 info 中的 win 标记 (最准确)
                is_win = info.get("win", False)
                game_ended = info.get("game_ended", False)

                # 失败定义: 游戏结束且未胜利 (排除超时截断)
                # 注意: info['game_ended'] 在超时(TimeLimit)时通常为 False，只有真正胜负分出时为 True
                is_fail = game_ended and not is_win

                if is_fail:
                    self.fail_count += 1

                # 计算统计
                avg_len = self.total_len / self.episode_count
                recent_avg = (
                    sum(self.recent_lens) / len(self.recent_lens)
                    if self.recent_lens
                    else 0
                )
                fail_rate = self.fail_count / self.episode_count * 100

                # 状态标记
                if is_win:
                    status = "🏆"  # 胜利
                elif is_fail:
                    status = "💀"  # 失败
                else:
                    status = "✓"  # 存活/超时

                # 简洁输出
                print(
                    f"\n[EP{self.episode_count:>4}] {status} len={ep_len:<5} rew={ep_reward:>+6.1f} | "
                    f"近20局均:{recent_avg:>5.0f} 最佳:{self.best_len:<5} 失败率:{fail_rate:>4.1f}%"
                )

                # 打印奖励详情 (如果有)
                if "reward_details" in info:
                    print(f"[EP 奖励详情] 总分: {ep_reward:.1f}")
                    sorted_stats = sorted(
                        info["reward_details"].items(),
                        key=lambda x: abs(x[1]),
                        reverse=True,
                    )
                    for k, v in sorted_stats:
                        if abs(v) > 0.1:
                            print(f"   - {k}: {v:.1f}")
        return True

    def _on_rollout_end(self):
        """每个 rollout 结束时输出所有关键调试信息"""
        # 每 5000 步输出一次调试信息
        if self.num_timesteps - self.last_print_step >= 5000:
            self.last_print_step = self.num_timesteps

            # 获取训练指标
            logger = self.model.logger
            if hasattr(logger, "name_to_value"):
                metrics = logger.name_to_value

                # PPO 核心指标
                approx_kl = metrics.get("train/approx_kl", 0)
                entropy = metrics.get("train/entropy_loss", 0)
                clip_frac = metrics.get("train/clip_fraction", 0)
                explained_var = metrics.get("train/explained_variance", 0)

                # 损失指标
                policy_loss = metrics.get("train/policy_gradient_loss", 0)
                value_loss = metrics.get("train/value_loss", 0)
                loss = metrics.get("train/loss", 0)

                # 学习率
                learning_rate = metrics.get("train/learning_rate", 0)

                # 判断是否健康
                kl_ok = 0.005 < approx_kl < 0.1
                ent_ok = entropy < -0.5
                clip_ok = clip_frac > 0.01
                ev_ok = explained_var > 0.1

                elapsed = time.time() - self.start_time
                fps = self.num_timesteps / elapsed if elapsed > 0 else 0

                # 健康状态汇总
                health = sum([kl_ok, ent_ok, clip_ok, ev_ok])
                health_icon = "🟢" if health >= 3 else "🟡" if health >= 2 else "🔴"

                print(
                    f"\n  {health_icon} [{self.num_timesteps:,}步 | {fps:.0f}it/s | lr={learning_rate:.2e}]"
                )
                print(
                    f"     kl={approx_kl:.2e}{'✓' if kl_ok else '⚠️'} "
                    f"ent={entropy:.3f}{'✓' if ent_ok else '⚠️'} "
                    f"clip={clip_frac:.3f}{'✓' if clip_ok else '⚠️'} "
                    f"ev={explained_var:.3f}{'✓' if ev_ok else '⚠️'}"
                )
                print(
                    f"     policy_loss={policy_loss:.4f} value_loss={value_loss:.2f} total_loss={loss:.2f}"
                )

    def _on_training_end(self):
        avg_len = self.total_len / max(1, self.episode_count)
        fail_rate = self.fail_count / max(1, self.episode_count) * 100
        print(
            f"\n训练结束: {self.episode_count}局 | 平均生存:{avg_len:.0f}步 | 最佳:{self.best_len}步 | 失败率:{fail_rate:.1f}%"
        )


