import os


def add_ddqn_args(parser):
    group = parser.add_argument_group("DDQN")
    group.add_argument("--ddqn_episodes", type=int, default=200000, help="DDQN episodes")
    group.add_argument("--ddqn_gamma", type=float, default=0.99, help="DDQN gamma")
    group.add_argument(
        "--ddqn_batch_size", type=int, default=32, help="DDQN batch size"
    )
    group.add_argument(
        "--ddqn_buffer_size", type=int, default=50000, help="DDQN replay buffer size"
    )
    group.add_argument(
        "--ddqn_burn_in", type=int, default=10000, help="DDQN burn-in steps"
    )
    group.add_argument(
        "--ddqn_lr", type=float, default=1e-3, help="DDQN learning rate"
    )
    group.add_argument(
        "--ddqn_update_freq",
        type=int,
        default=32,
        help="DDQN network update frequency",
    )
    group.add_argument(
        "--ddqn_sync_freq",
        type=int,
        default=2000,
        help="DDQN target sync frequency",
    )
    group.add_argument(
        "--ddqn_eval_freq",
        type=int,
        default=20,
        help="DDQN evaluation frequency (episodes)",
    )
    group.add_argument(
        "--ddqn_eval_iters",
        type=int,
        default=5,
        help="DDQN evaluation iterations",
    )
    group.add_argument(
        "--ddqn_save_path",
        type=str,
        default="models/ddqn_model.pt",
        help="DDQN model save path",
    )
    group.add_argument(
        "--ddqn_load_path",
        type=str,
        default=None,
        help="DDQN model load path",
    )


def _build_ddqn_env(args):
    from envs import PVZEnv
    from .adapter import DDQNEnvAdapter

    env = PVZEnv(
        hook_port=args.port,
        game_speed=args.speed,
        frame_skip=args.frameskip,
    )
    return DDQNEnvAdapter(env)


def train_ddqn(args):
    import torch
    from .ddqn import QNetwork, DDQNAgent, experienceReplayBuffer

    env = _build_ddqn_env(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = QNetwork(env, learning_rate=args.ddqn_lr, device=device)

    if args.ddqn_load_path and os.path.exists(args.ddqn_load_path):
        state_dict = torch.load(args.ddqn_load_path, map_location=device)
        network.load_state_dict(state_dict)

    buffer = experienceReplayBuffer(
        memory_size=args.ddqn_buffer_size, burn_in=args.ddqn_burn_in
    )
    agent = DDQNAgent(env, network, buffer, batch_size=args.ddqn_batch_size)

    try:
        agent.train(
            gamma=args.ddqn_gamma,
            max_episodes=args.ddqn_episodes,
            network_update_frequency=args.ddqn_update_freq,
            network_sync_frequency=args.ddqn_sync_freq,
            evaluate_frequency=args.ddqn_eval_freq,
            evaluate_n_iter=args.ddqn_eval_iters,
        )
    finally:
        os.makedirs(os.path.dirname(args.ddqn_save_path) or ".", exist_ok=True)
        torch.save(network.state_dict(), args.ddqn_save_path)
        env.close()
