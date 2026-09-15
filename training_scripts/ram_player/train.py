"""Train Player 1 on structured RAM; no pixel preprocessing is imported."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
from math import isclose
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import nn

from .callbacks import RAMEvalCallback
from .common import (
    DEFAULT_CONFIG,
    DEFAULT_RUN_ROOT,
    evaluation_episode_steps,
    load_config,
    make_ram_env,
    normalize_opponent_league,
    normalize_opponents,
    opponent_state_path,
    prepare_rom,
    require_all_opponent_slots,
    resolve_repo_path,
    training_state_paths,
    validate_model_action_space,
    validate_model_observation_space,
    write_run_metadata,
)


def _opponent(value: str) -> tuple[int, str]:
    try:
        raw_slot, raw_path = value.split("=", 1)
        slot = int(raw_slot)
    except ValueError as error:
        raise argparse.ArgumentTypeError("use SLOT=RAM_MODEL_PATH") from error
    if slot not in (1, 2, 3):
        raise argparse.ArgumentTypeError("opponent slot must be 1, 2, or 3")
    return slot, raw_path


def _league_opponent(value: str) -> tuple[str, str]:
    try:
        label, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("use LABEL=RAM_MODEL_PATH") from error
    if not label.strip() or not raw_path:
        raise argparse.ArgumentTypeError("league label and path must be non-empty")
    return label.strip(), raw_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a fast Player 1 RAM policy on Dino Boneyard multi"
    )
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", help="override config run_name")
    parser.add_argument("--timesteps", type=int, help="override total_timesteps")
    parser.add_argument("--num-envs", type=int, help="override num_envs")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--resume-model", type=Path)
    parser.add_argument(
        "--no-record-video",
        action="store_true",
        help="skip recording the best checkpoint after successful training",
    )
    parser.add_argument(
        "--opponent",
        action="append",
        type=_opponent,
        metavar="SLOT=RAM_MODEL_PATH",
        help="replace configured self-play opponents; repeat for more slots",
    )
    parser.add_argument(
        "--opponent-state",
        type=Path,
        help="fresh Dino state containing four player-class racers",
    )
    parser.add_argument(
        "--league-opponent",
        action="append",
        type=_league_opponent,
        metavar="LABEL=RAM_MODEL_PATH",
        help="replace configured frozen league; provide at least three models",
    )
    parser.add_argument("--device", default="cpu")
    return parser


def _ppo_arguments(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ppo = dict(config["ppo"])
    policy_kwargs = {
        "activation_fn": nn.Tanh,
        "net_arch": {
            "pi": list(ppo.pop("policy_layers")),
            "vf": list(ppo.pop("value_layers")),
        },
    }
    return ppo, policy_kwargs


def _verify_resume_ppo_configuration(model: PPO, ppo: dict[str, Any]) -> None:
    """Fail loudly if a loaded checkpoint retained stale optimizer settings."""

    scheduled = {
        "learning_rate": model.lr_schedule(1.0),
        "clip_range": model.clip_range(1.0),
    }
    scalar_names = (
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
    )
    actual = {name: getattr(model, name) for name in scalar_names}
    actual.update(scheduled)
    mismatches = [
        f"{name}={actual[name]!r} (expected {expected!r})"
        for name, expected in ppo.items()
        if name in actual
        and not isclose(
            float(actual[name]), float(expected), rel_tol=1e-9, abs_tol=1e-12
        )
    ]
    if mismatches:
        raise RuntimeError(
            "resume PPO configuration mismatch: " + ", ".join(mismatches)
        )


def _resume_summary(model: PPO) -> str:
    return (
        f"learning_rate={model.lr_schedule(1.0):g} "
        f"clip_range={model.clip_range(1.0):g} "
        f"n_epochs={model.n_epochs} batch_size={model.batch_size}"
    )


def main() -> None:
    args = _parser().parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    if args.run_name:
        config["run_name"] = args.run_name
    if args.timesteps is not None:
        config["total_timesteps"] = args.timesteps
    if args.num_envs is not None:
        config["num_envs"] = args.num_envs
    if args.opponent is not None:
        if len(dict(args.opponent)) != len(args.opponent):
            raise ValueError("each self-play opponent slot may appear only once")
        config["opponents"] = dict(args.opponent)
        config["opponent_league"] = {}
    if args.league_opponent is not None:
        if len(dict(args.league_opponent)) != len(args.league_opponent):
            raise ValueError("each frozen league label may appear only once")
        config["opponents"] = {}
        config["opponent_league"] = dict(args.league_opponent)
    if args.opponent_state is not None:
        config["opponent_state"] = str(args.opponent_state.expanduser().resolve())
    if int(config["total_timesteps"]) < 1:
        raise ValueError("timesteps must be at least one")
    if int(config["num_envs"]) < 1:
        raise ValueError("num-envs must be at least one")

    opponents = normalize_opponents(config["opponents"])
    opponent_league = normalize_opponent_league(config.get("opponent_league"))
    require_all_opponent_slots(opponents)
    has_model_opponents = bool(opponents or opponent_league)
    states = (
        [opponent_state_path(config)]
        if has_model_opponents
        else training_state_paths(config)
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.run_root.expanduser().resolve() / f"{config['run_name']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    source_rom = args.rom.expanduser().resolve()
    controlled_slots = (1, 2, 3) if opponent_league else tuple(opponents)
    active_rom = prepare_rom(source_rom, controlled_slots, run_dir / "private")
    write_run_metadata(
        run_dir,
        config,
        config_path,
        source_rom,
        active_rom,
        opponents,
        opponent_league,
    )
    opponent_paths = {slot: str(path) for slot, path in opponents.items()}
    league_paths = {label: str(path) for label, path in opponent_league.items()}
    training_state_values = [str(state) if state else None for state in states]
    sample_training_states = bool(config.get("sample_training_states", False))

    env_functions = []
    for index in range(int(config["num_envs"])):
        state = states[index % len(states)]
        env_functions.append(
            partial(
                make_ram_env,
                frame_skip=int(config["frame_skip"]),
                max_episode_steps=int(config["max_episode_steps"]),
                seed=int(config["seed"]) + index,
                opponent_paths=opponent_paths,
                opponent_league=league_paths,
                state_path=(
                    str(states[0])
                    if has_model_opponents
                    else (
                        None if sample_training_states or state is None else str(state)
                    )
                ),
                state_paths=(
                    training_state_values
                    if sample_training_states and not has_model_opponents
                    else None
                ),
                monitor_path=str(run_dir / "monitor" / f"worker_{index}"),
                reward_config=config.get("reward"),
            )
        )

    print(f"Run directory: {run_dir}")
    print(
        f"Training Player 1 with {config['num_envs']} RAM environments for "
        f"{int(config['total_timesteps']):,} timesteps"
    )
    if opponents:
        print("Frozen self-play opponents: " + ", ".join(map(str, opponents)))
    if opponent_league:
        print("Frozen self-play league: " + ", ".join(opponent_league))
    print(f"TensorBoard: tensorboard --logdir {run_dir / 'tensorboard'}")

    ppo, policy_kwargs = _ppo_arguments(config)
    completed = False
    training_env = SubprocVecEnv(env_functions)
    try:
        stock_evaluation_state = (
            resolve_repo_path(
                config.get(
                    "stock_evaluation_state",
                    "training_scripts/data/states/dino_boneyard_multi.state",
                )
            ).resolve()
            if has_model_opponents
            else None
        )
        if stock_evaluation_state is not None and not stock_evaluation_state.is_file():
            raise FileNotFoundError(stock_evaluation_state)
        stock_eval_env_factory = partial(
            make_ram_env,
            frame_skip=int(config["frame_skip"]),
            max_episode_steps=evaluation_episode_steps(config),
            seed=int(config["seed"]) + 10_000,
            state_path=(
                str(stock_evaluation_state)
                if stock_evaluation_state is not None
                else None
            ),
            reward_config=config.get("reward"),
        )
        league_eval_env_factory = (
            partial(
                make_ram_env,
                frame_skip=int(config["frame_skip"]),
                max_episode_steps=evaluation_episode_steps(config),
                seed=int(config["seed"]) + 20_000,
                opponent_paths=opponent_paths,
                opponent_league=league_paths,
                state_path=str(states[0]),
                reward_config=config.get("reward"),
            )
            if has_model_opponents
            else None
        )
        if args.resume_model:
            resume_path = args.resume_model.expanduser().resolve()
            model = PPO.load(
                resume_path,
                env=training_env,
                device=args.device,
                tensorboard_log=str(run_dir / "tensorboard"),
                # SB3 otherwise restores stale training hyperparameters
                # from the checkpoint and silently ignores this run's YAML.
                custom_objects=dict(ppo),
            )
            validate_model_observation_space(model, resume_path)
            validate_model_action_space(model, resume_path)
            _verify_resume_ppo_configuration(model, ppo)
            print(f"Applied resume PPO configuration: {_resume_summary(model)}")
        else:
            model = PPO(
                "MlpPolicy",
                training_env,
                policy_kwargs=policy_kwargs,
                tensorboard_log=str(run_dir / "tensorboard"),
                verbose=1,
                seed=int(config["seed"]),
                device=args.device,
                **ppo,
            )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(
                1,
                int(config["checkpoint_every_timesteps"]) // int(config["num_envs"]),
            ),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="dino_ram_player",
        )
        stock_eval_callback = RAMEvalCallback(
            None,
            eval_env_factory=stock_eval_env_factory,
            eval_every_timesteps=int(config["eval_every_timesteps"]),
            episodes=int(config["eval_episodes"]),
            max_episode_frames=(
                evaluation_episode_steps(config) * int(config["frame_skip"])
            ),
            output_dir=run_dir / "evaluation",
        )
        callbacks = [checkpoint_callback, stock_eval_callback]
        if league_eval_env_factory is not None:
            callbacks.append(
                RAMEvalCallback(
                    None,
                    eval_env_factory=league_eval_env_factory,
                    eval_every_timesteps=int(config["eval_every_timesteps"]),
                    episodes=int(
                        config.get("league_eval_episodes", config["eval_episodes"])
                    ),
                    max_episode_frames=(
                        evaluation_episode_steps(config) * int(config["frame_skip"])
                    ),
                    output_dir=run_dir / "league_evaluation",
                    metric_prefix="league_eval",
                    selection_metric="competitive_selection_score",
                )
            )
        try:
            model.learn(
                total_timesteps=int(config["total_timesteps"]),
                callback=CallbackList(callbacks),
                tb_log_name="ppo",
                reset_num_timesteps=not bool(args.resume_model),
                progress_bar=False,
            )
        except KeyboardInterrupt:
            model.save(run_dir / "interrupted_model")
            print(f"Interrupted model saved to {run_dir / 'interrupted_model.zip'}")
            raise
        model.save(run_dir / "final_model")
        print(f"Final model saved to {run_dir / 'final_model.zip'}")
        print(f"Best model saved to {run_dir / 'evaluation' / 'best_model.zip'}")
        completed = True
    finally:
        training_env.close()

    if (
        completed
        and bool(config.get("record_video", True))
        and not args.no_record_video
    ):
        from .record import record_model

        video_path = run_dir / "best_model.mp4"
        record_model(
            run_dir / "evaluation" / "best_model.zip",
            video_path,
            config,
            opponent_paths={},
            device=args.device,
        )
        print(f"Stock best-model video saved to {video_path}")
        if opponent_league:
            league_video_path = run_dir / "best_model_league.mp4"
            record_model(
                run_dir / "evaluation" / "best_model.zip",
                league_video_path,
                config,
                opponent_paths={
                    slot: league_paths[label]
                    for slot, label in zip((1, 2, 3), opponent_league)
                },
                device=args.device,
            )
            print(f"League best-model video saved to {league_video_path}")


if __name__ == "__main__":
    main()
