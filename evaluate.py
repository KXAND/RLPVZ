import argparse
import os
from datetime import datetime

from models.ddqn.evaluate import evaluate_ddqn
from models.ppo.evaluate import evaluate_ppo
from training.args import get_args
from training.evaluation import EvaluationWriter, load_evaluation_config
from training.game_instances import prepare_game_instances
from training.paths import get_cached_model_path
from training.specs import build_base_eval_specs
from utils.train_utils import load_training_config


def main(argv=None):
    eval_args, train_argv = _parse_eval_args(argv)
    args = get_args(train_argv)
    eval_config = _load_eval_config(args)
    _apply_eval_instance_config(args, eval_args, eval_config)

    model_path = eval_args.model or get_cached_model_path(args.algo)
    output_dir = eval_args.eval_output or _default_eval_output(args.algo)

    env_spec, scenario_spec = build_base_eval_specs(args)
    instances = prepare_game_instances(args)
    if instances is None:
        return

    if args.algo == "ddqn":
        result = evaluate_ddqn(
            args=args,
            model_path=model_path,
            instances=instances,
            env_spec=env_spec,
            scenario_spec=scenario_spec,
            episodes=eval_args.eval_episodes or eval_config.episodes,
        )
    elif args.algo == "ppo":
        result = evaluate_ppo(
            args=args,
            model_path=model_path,
            instances=instances,
            env_spec=env_spec,
            scenario_spec=scenario_spec,
            episodes=eval_args.eval_episodes or eval_config.episodes,
            device="auto",
        )
    else:
        raise NotImplementedError(
            f"Offline evaluate is not implemented for algo: {args.algo}"
        )

    writer = EvaluationWriter(
        output_dir,
        save_episode_details=eval_config.save_episode_details,
    )
    writer.write(result)
    print(f"Saved eval summary to {writer.csv_path}")


def _parse_eval_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a trained real-env model")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=None,
        help="Number of independent eval episodes",
    )
    parser.add_argument(
        "--eval_output",
        type=str,
        default=None,
        help="Directory for eval.jsonl/eval.csv/eval_snapshot.json",
    )
    parser.add_argument(
        "--eval_num_envs",
        type=int,
        default=None,
        help="Number of real eval game instances",
    )
    return parser.parse_known_args(argv)


def _load_eval_config(args):
    config = load_training_config(args.training_config)
    return load_evaluation_config(
        config.get("training", {}).get("eval", {})
    )


def _apply_eval_instance_config(args, eval_args, eval_config):
    args.num_envs = eval_args.eval_num_envs or eval_config.real_num_envs
    if eval_config.real_base_port is not None:
        args.base_port = eval_config.real_base_port
        args.port = eval_config.real_base_port


def _default_eval_output(algo):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("eval_output", algo, timestamp)


if __name__ == "__main__":
    main()
