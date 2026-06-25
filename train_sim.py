"""
Simulation environment training entry point.

Usage:
    python train_sim.py            # DDQN (default)
    python train_sim.py --ppo      # PPO
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RL agent on SimPVZ")
    parser.add_argument("--ppo", action="store_true",
                        help="Use PPO instead of DDQN")
    parser.add_argument("--algo", type=str, default="ddqn",
                        choices=["ddqn", "ppo"],
                        help="Algorithm to use (default: ddqn)")
    args = parser.parse_args()

    algo = "ppo" if args.ppo else args.algo

    if algo == "ppo":
        from simenv.ppo import train_ppo
        train_ppo(
            max_episodes=100000,
            horizon=2048,
            batch_size=64,
            n_epochs=10,
            network_type="mlp",  # cnn / mlp / deepmlp
        )
    else:
        from simenv.trainer import train_sim
        train_sim(
            max_episodes=100000,
            buffer_size=50000,
            burn_in=10000,
            batch_size=200,
            network_type="mlp",  # cnn / mlp / deepmlp
        )
