"""Train Player 1 on structured RAM; no pixel preprocessing is imported."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
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
    normalize_opponents,
    opponent_state_path,
    prepare_rom,
    require_all_opponent_slots,
    training_state_paths,
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
    if args.opponent_state is not None:
        config["opponent_state"] = str(args.opponent_state.expanduser().resolve())
    if int(config["total_timesteps"]) < 1:
        raise ValueError("timesteps must be at least one")
    if int(config["num_envs"]) < 1:
        raise ValueError("num-envs must be at least one")

    opponents = normalize_opponents(config["opponents"])
    require_all_opponent_slots(opponents)
    states = (
        [opponent_state_path(config)] if opponents else training_state_paths(config)
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.run_root.expanduser().resolve() / f"{config['run_name']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    source_rom = args.rom.expanduser().resolve()
    active_rom = prepare_rom(source_rom, tuple(opponents), run_dir / "private")
    write_run_metadata(
        run_dir,
        config,
        config_path,
        source_rom,
        active_rom,
        opponents,
    )
    opponent_paths = {slot: str(path) for slot, path in opponents.items()}

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
                opponent_max_turn=int(config["opponent_max_turn"]),
                opponent_max_target_speed=int(config["opponent_max_target_speed"]),
                state_path=str(state) if state else None,
                monitor_path=str(run_dir / "monitor" / f"worker_{index}"),
            )
        )

    print(f"Run directory: {run_dir}")
    print(
        f"Training Player 1 with {config['num_envs']} RAM environments for "
        f"{int(config['total_timesteps']):,} timesteps"
    )
    if opponents:
        print("Frozen self-play opponents: " + ", ".join(map(str, opponents)))
    print(f"TensorBoard: tensorboard --logdir {run_dir / 'tensorboard'}")

    ppo, policy_kwargs = _ppo_arguments(config)
    completed = False
    training_env = SubprocVecEnv(env_functions)
    try:
        eval_env = make_ram_env(
            frame_skip=int(config["frame_skip"]),
            max_episode_steps=evaluation_episode_steps(config),
            seed=int(config["seed"]) + 10_000,
            opponent_paths=opponent_paths,
            opponent_max_turn=int(config["opponent_max_turn"]),
            opponent_max_target_speed=int(config["opponent_max_target_speed"]),
            state_path=str(states[0]) if opponents else None,
        )
        try:
            if args.resume_model:
                resume_path = args.resume_model.expanduser().resolve()
                model = PPO.load(
                    resume_path,
                    device=args.device,
                    tensorboard_log=str(run_dir / "tensorboard"),
                )
                validate_model_observation_space(model, resume_path)
                model.set_env(training_env)
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
                    int(config["checkpoint_every_timesteps"])
                    // int(config["num_envs"]),
                ),
                save_path=str(run_dir / "checkpoints"),
                name_prefix="dino_ram_player",
            )
            eval_callback = RAMEvalCallback(
                eval_env,
                eval_every_timesteps=int(config["eval_every_timesteps"]),
                episodes=int(config["eval_episodes"]),
                max_episode_frames=(
                    evaluation_episode_steps(config) * int(config["frame_skip"])
                ),
                output_dir=run_dir / "evaluation",
            )
            try:
                model.learn(
                    total_timesteps=int(config["total_timesteps"]),
                    callback=CallbackList([checkpoint_callback, eval_callback]),
                    tb_log_name="ppo",
                    reset_num_timesteps=not bool(args.resume_model),
                    progress_bar=False,
                )
            except KeyboardInterrupt:
                model.save(run_dir / "interrupted_model")
                print(
                    f"Interrupted model saved to "
                    f"{run_dir / 'interrupted_model.zip'}"
                )
                raise
            model.save(run_dir / "final_model")
            print(f"Final model saved to {run_dir / 'final_model.zip'}")
            print(
                f"Best model saved to " f"{run_dir / 'evaluation' / 'best_model.zip'}"
            )
            completed = True
        finally:
            eval_env.close()
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
            opponent_paths=opponent_paths,
            device=args.device,
        )
        print(f"Best-model video saved to {video_path}")


if __name__ == "__main__":
    main()
