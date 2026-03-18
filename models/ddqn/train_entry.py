import os


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
