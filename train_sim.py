"""
Simulation environment training entry point.

Usage:
    python train_sim.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def plot_training(save_path, rewards, iterations, loss):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_path = save_path.replace(".pt", "_training.png")
    if len(rewards) == 0:
        return

    x_rewards = np.arange(1, len(rewards) + 1)
    x_loss = np.arange(1, len(loss) + 1)
    window = min(100, max(1, len(rewards)))

    def moving_average(values):
        if len(values) < window:
            return values
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="valid")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)

    axes[0].plot(x_rewards, rewards, alpha=0.35, label="episode")
    ma_rewards = moving_average(rewards)
    axes[0].plot(
        np.arange(len(rewards) - len(ma_rewards) + 1, len(rewards) + 1),
        ma_rewards,
        label=f"mean {window}",
    )
    axes[0].set_title("Sim DDQN Reward")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_rewards, iterations, alpha=0.5)
    axes[1].set_title("Survival Frames")
    axes[1].set_ylabel("Frames")
    axes[1].grid(True, alpha=0.3)

    if len(loss) > 0:
        axes[2].plot(x_loss, loss, alpha=0.7)
    axes[2].set_title("DDQN Loss")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Loss")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DDQN agent on SimPVZ")
    parser.parse_args()

    from simenv.trainer import train_sim
    train_sim(
        max_episodes=100000,
        buffer_size=50000,
        burn_in=10000,
        batch_size=200,
        plot_callback=plot_training,
    )
