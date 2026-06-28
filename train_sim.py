"""
Simulation environment training entry point.

Usage:
    python train_sim.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


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
    )
