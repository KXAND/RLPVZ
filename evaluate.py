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
    episodes = eval_args.eval_episodes or eval_config.episodes

    env_spec, scenario_spec = build_base_eval_specs(args)
    _print_eval_metadata(
        args=args,
        model_path=model_path,
        output_dir=output_dir,
        episodes=episodes,
        env_spec=env_spec,
        scenario_spec=scenario_spec,
    )
    instances = prepare_game_instances(args)
    if instances is None:
        return
    _print_instances(instances)

    if args.algo == "ddqn":
        result = evaluate_ddqn(
            args=args,
            model_path=model_path,
            instances=instances,
            env_spec=env_spec,
            scenario_spec=scenario_spec,
            episodes=episodes,
        )
    elif args.algo == "ppo":
        result = evaluate_ppo(
            args=args,
            model_path=model_path,
            instances=instances,
            env_spec=env_spec,
            scenario_spec=scenario_spec,
            episodes=episodes,
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
    _print_eval_result(result)
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
    return parser.parse_known_args(argv)


def _load_eval_config(args):
    config = load_training_config(args.training_config)
    return load_evaluation_config(
        config.get("training", {}).get("eval", {})
    )


def _apply_eval_instance_config(args, eval_args, eval_config):
    args.num_envs = 1
    if eval_config.real_base_port is not None:
        args.base_port = eval_config.real_base_port
        args.port = eval_config.real_base_port


def _default_eval_output(algo):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("eval_output", algo, timestamp)


def _print_eval_metadata(args, model_path, output_dir, episodes,
                         env_spec, scenario_spec):
    sep = "-" * 58
    print(f"\n{sep}")
    print("  Evaluation Configuration")
    print(f"{sep}")
    print(f"  {'Algorithm:':24s} {args.algo}")
    print(f"  {'Model path:':24s} {model_path}")
    print(f"  {'Output dir:':24s} {output_dir}")
    print(f"  {'Episodes:':24s} {episodes}")
    print(f"  {'Eval envs:':24s} {args.num_envs}")
    print(f"  {'Base port:':24s} {getattr(args, 'base_port', args.port)}")
    print(f"  {'Grid:':24s} {env_spec.rows}x{env_spec.cols}")
    print(f"  {'Actions:':24s} {env_spec.action_space_size}")
    print(f"  {'Cards:':24s} {list(scenario_spec.cards)}")
    print(f"  {'Game mode:':24s} {scenario_spec.game_mode_id}")
    print(f"  {'Initial sun:':24s} {scenario_spec.initial_sun}")
    print(f"  {'Win condition:':24s} {scenario_spec.win_condition}")
    print(f"  {'Target sublevels:':24s} {scenario_spec.target_sublevels}")
    print(f"{sep}\n")


def _print_instances(instances):
    print("[Eval] Instances: " + ", ".join(
        f"pid={item['pid']} port={item['port']}" for item in instances
    ))


def _print_eval_result(result):
    sep = "-" * 58
    print(f"\n{sep}")
    print("  Evaluation Result")
    print(f"{sep}")
    print(f"  {'Reward:':20s} mean={result.reward_mean:8.2f}  "
          f"std={result.reward_std:8.2f}  min={result.reward_min:8.2f}  "
          f"max={result.reward_max:8.2f}")
    print(f"  {'Survival:':20s} mean={result.survival_mean:8.2f}  "
          f"std={result.survival_std:8.2f}  min={result.survival_min:8.0f}  "
          f"max={result.survival_max:8.0f}")
    print(f"  {'Win rate:':20s} {result.win_count}/{result.episodes} "
          f"({100 * result.win_rate:.1f}%)")
    print(f"  {'Duration:':20s} {result.duration_sec:.2f}s")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
