"""Train an isolated Dino pickup/hairpin expert from private savestates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
from torch import nn

from .common import (
    DEFAULT_RUN_ROOT,
    load_config,
    prepare_rom,
    resolve_repo_path,
    write_run_metadata,
)
from .pickup_lab import PickupEvalCallback, make_pickup_lab_env

DEFAULT_CONFIG = Path(__file__).with_name("dino_boneyard_pickup_expert_v11.yml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the short-horizon Dino Boneyard pickup expert"
    )
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--no-behavior-cloning", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser


def _ppo_arguments(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ppo = dict(config["ppo"])
    policy_kwargs = {
        "activation_fn": nn.Tanh,
        "net_arch": {
            "pi": list(ppo.pop("policy_layers")),
            "vf": list(ppo.pop("value_layers")),
        },
    }
    return ppo, policy_kwargs


def _manifest_paths(
    path: Path,
) -> tuple[list[Path], list[Path], Path, dict[str, Any]]:
    with path.open() as handle:
        manifest = json.load(handle)
    if manifest.get("version") != 1:
        raise ValueError(f"unsupported pickup-state manifest: {path}")
    entries = manifest.get("states")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"pickup-state manifest has no states: {path}")
    training: list[Path] = []
    evaluation: list[Path] = []
    for entry in entries:
        state = (path.parent / str(entry["path"])).resolve()
        if not state.is_file():
            raise FileNotFoundError(state)
        if entry.get("split") == "evaluation":
            evaluation.append(state)
        else:
            training.append(state)
    if not training:
        raise ValueError("pickup-state manifest has no training split")
    if not evaluation:
        evaluation = training[: min(5, len(training))]
    demonstrations = (path.parent / str(manifest["demonstrations"])).resolve()
    if not demonstrations.is_file():
        raise FileNotFoundError(demonstrations)
    return training, evaluation, demonstrations, manifest


def behavior_clone(
    model: PPO,
    demonstrations: Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, float]:
    """Warm-start the expert from successful legacy-policy actions."""

    with np.load(demonstrations) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.int64)
    if (
        observations.ndim != 2
        or observations.shape[1:] != model.observation_space.shape
    ):
        raise ValueError(
            f"demonstrations have observation shape {observations.shape}, expected "
            f"(N, {model.observation_space.shape[0]})"
        )
    if actions.shape != (len(observations), 3):
        raise ValueError(
            f"demonstrations have action shape {actions.shape}, expected (N, 3)"
        )
    if not len(observations):
        raise ValueError("demonstration file contains no successful actions")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError(
            "behavior-cloning epochs, batch size, and learning rate must be positive"
        )

    generator = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    device = model.device
    model.policy.set_training_mode(True)
    final_loss = 0.0
    for _ in range(epochs):
        order = generator.permutation(len(observations))
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            observation_batch = torch.as_tensor(observations[indices], device=device)
            action_batch = torch.as_tensor(actions[indices], device=device)
            _, log_probability, _ = model.policy.evaluate_actions(
                observation_batch, action_batch
            )
            loss = -log_probability.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses))
    model.policy.set_training_mode(False)

    correct = 0
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            observation_batch = torch.as_tensor(
                observations[start : start + batch_size], device=device
            )
            predicted = model.policy.get_distribution(observation_batch).get_actions(
                deterministic=True
            )
            expected = torch.as_tensor(
                actions[start : start + batch_size], device=device
            )
            correct += int(torch.all(predicted == expected, dim=1).sum().cpu())
    return {
        "steps": float(len(observations)),
        "loss": final_loss,
        "exact_action_accuracy": correct / len(observations),
    }


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
    if config["opponents"] or config.get("opponent_league"):
        raise ValueError("the pickup lab intentionally trains without model opponents")

    raw_manifest = config.get("pickup_state_manifest")
    if not raw_manifest:
        raise ValueError("pickup_state_manifest is required")
    manifest_path = resolve_repo_path(str(raw_manifest)).resolve()
    training_states, evaluation_states, demonstrations, manifest = _manifest_paths(
        manifest_path
    )
    cloning = config.get("behavior_cloning") or {}
    if cloning.get("demonstrations"):
        demonstrations = resolve_repo_path(
            str(cloning["demonstrations"])
        ).resolve()
        if not demonstrations.is_file():
            raise FileNotFoundError(demonstrations)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.run_root.expanduser().resolve() / f"{config['run_name']}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    source_rom = args.rom.expanduser().resolve()
    active_rom = prepare_rom(source_rom, (), run_dir / "private")
    write_run_metadata(
        run_dir,
        config,
        config_path,
        source_rom,
        active_rom,
        {},
        {},
    )
    with (run_dir / "pickup_state_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    task_config = config.get("pickup_task")
    state_values = [str(path) for path in training_states]
    env_functions = [
        partial(
            make_pickup_lab_env,
            frame_skip=int(config["frame_skip"]),
            max_episode_steps=int(config["max_episode_steps"]),
            seed=int(config["seed"]) + index,
            state_paths=state_values,
            task_config=task_config,
            monitor_path=str(run_dir / "monitor" / f"worker_{index}"),
        )
        for index in range(int(config["num_envs"]))
    ]
    evaluation_values = [str(path) for path in evaluation_states]
    eval_env_factory = partial(
        make_pickup_lab_env,
        frame_skip=int(config["frame_skip"]),
        max_episode_steps=int(config["max_episode_steps"]),
        seed=int(config["seed"]) + 10_000,
        state_paths=evaluation_values,
        state_pool_selection="cycle",
        task_config=task_config,
    )

    print(f"Run directory: {run_dir}")
    print(
        f"Pickup lab: {len(training_states)} training states, "
        f"{len(evaluation_states)} holdout states, "
        f"{config['num_envs']} environments, "
        f"{int(config['total_timesteps']):,} timesteps"
    )
    print(f"TensorBoard: tensorboard --logdir {run_dir / 'tensorboard'}")

    ppo, policy_kwargs = _ppo_arguments(config)
    training_env = SubprocVecEnv(env_functions, start_method="fork")
    completed = False
    try:
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
        if cloning and not args.no_behavior_cloning:
            clone_metrics = behavior_clone(
                model,
                demonstrations,
                epochs=int(cloning.get("epochs", 20)),
                batch_size=int(cloning.get("batch_size", 256)),
                learning_rate=float(cloning.get("learning_rate", 3e-4)),
                seed=int(config["seed"]),
            )
            model.save(run_dir / "behavior_cloned_model")
            with (run_dir / "behavior_cloning.json").open("w") as handle:
                json.dump(clone_metrics, handle, indent=2)
                handle.write("\n")
            print(
                "Behavior cloning: "
                f"steps={clone_metrics['steps']:.0f} "
                f"loss={clone_metrics['loss']:.4f} "
                f"exact_action={clone_metrics['exact_action_accuracy']:.1%}"
            )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(
                1,
                int(config["checkpoint_every_timesteps"]) // int(config["num_envs"]),
            ),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="dino_pickup_expert",
        )
        eval_callback = PickupEvalCallback(
            eval_env_factory=eval_env_factory,
            eval_every_timesteps=int(config["eval_every_timesteps"]),
            episodes=len(evaluation_states),
            output_dir=run_dir / "evaluation",
        )
        try:
            model.learn(
                total_timesteps=int(config["total_timesteps"]),
                callback=CallbackList([checkpoint_callback, eval_callback]),
                tb_log_name="ppo",
                reset_num_timesteps=True,
                progress_bar=False,
            )
        except KeyboardInterrupt:
            model.save(run_dir / "interrupted_model")
            print(f"Interrupted model saved to {run_dir / 'interrupted_model.zip'}")
            raise
        model.save(run_dir / "final_model")
        print(f"Final pickup expert saved to {run_dir / 'final_model.zip'}")
        print(
            f"Best pickup expert saved to "
            f"{run_dir / 'evaluation' / 'best_model.zip'}"
        )
        completed = True
    finally:
        training_env.close()
    if not completed:
        raise RuntimeError("pickup training did not complete")


if __name__ == "__main__":
    main()
