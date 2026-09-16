"""Capture labeled pickup-approach savestates and successful demonstrations."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from HotWheelsGym.ram_opponent_control import DINO_BONEYARD_PROGRESS_COUNT

from .common import (
    file_sha1,
    load_ram_policy,
    make_ram_env,
    prepare_rom,
)


def _policy(value: str) -> tuple[str, Path]:
    try:
        label, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("use LABEL=RAM_MODEL_PATH") from error
    if not label.strip() or not raw_path:
        raise argparse.ArgumentTypeError("policy label and path must be non-empty")
    return label.strip(), Path(raw_path)


def _targets(value: str) -> tuple[int, ...]:
    try:
        targets = tuple(sorted({int(item) for item in value.split(",")}))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "targets must be comma-separated integers"
        ) from error
    if not targets or targets[0] < 0 or targets[-1] >= 43:
        raise argparse.ArgumentTypeError("pickup targets must be in progress 0..42")
    return targets


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture Dino pickup states from old policies and retain only "
            "successful state/action traces as imitation demonstrations"
        )
    )
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument(
        "--policy",
        action="append",
        type=_policy,
        required=True,
        metavar="LABEL=MODEL",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", type=_targets, default=(12, 18, 24, 30, 34))
    parser.add_argument("--stochastic-rollouts", type=int, default=0)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser


def _write_state(path: Path, state: bytes, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"state already exists: {path} (pass --force)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        handle.write(state)


def _capture_rollout(
    *,
    policy: Any,
    label: str,
    rollout: int,
    deterministic: bool,
    env: Any,
    targets: tuple[int, ...],
    output_dir: Path,
    force: bool,
    policy_index: int,
    max_episode_steps: int,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
    reset_policy = getattr(policy, "reset", None)
    if callable(reset_policy):
        reset_policy()
    observation, info = env.reset()
    captured_lap: int | None = None
    candidates: dict[int, tuple[bytes, dict[str, Any]]] = {}
    trace_observations: list[np.ndarray] = []
    trace_actions: list[np.ndarray] = []
    pickup_count = 0
    manifest_states: list[dict[str, Any]] = []
    demo_observations: list[np.ndarray] = []
    demo_actions: list[np.ndarray] = []

    def finish_approach(lap: int) -> None:
        nonlocal candidates, trace_observations, trace_actions, pickup_count
        if not candidates:
            return
        outcome = "success" if pickup_count else "miss"
        for target_index, (target, (state, metadata)) in enumerate(
            sorted(candidates.items())
        ):
            filename = f"{label}_r{rollout:02d}_lap{lap}_p{target:02d}_{outcome}.state"
            relative = Path("states") / outcome / filename
            _write_state(output_dir / relative, state, force=force)
            split = (
                "evaluation"
                if (target_index + lap + rollout + policy_index) % 5 == 0
                else "training"
            )
            manifest_states.append(
                {
                    **metadata,
                    "path": str(relative),
                    "source_policy": label,
                    "rollout": rollout,
                    "deterministic": deterministic,
                    "lap": lap,
                    "target_progress": target,
                    "teacher_outcome": outcome,
                    "split": split,
                }
            )
        if pickup_count:
            demo_observations.extend(trace_observations)
            demo_actions.extend(trace_actions)
        candidates = {}
        trace_observations = []
        trace_actions = []
        pickup_count = 0

    for _ in range(max_episode_steps):
        lap = int(info.get("ram_player_lap", 1))
        progress = (
            int(info.get("ram_player_progress", 0)) % DINO_BONEYARD_PROGRESS_COUNT
        )
        if captured_lap is None:
            captured_lap = lap
        elif lap != captured_lap:
            finish_approach(captured_lap)
            captured_lap = lap

        target_available = bool(
            info.get("ram_player_next_power_up_is_jet_boost", False)
        ) and bool(info.get("ram_player_next_power_up_available", False))
        if progress < 43 and target_available:
            for target in targets:
                if target not in candidates and progress >= target:
                    candidates[target] = (
                        bytes(env.unwrapped.em.get_state()),
                        {
                            "captured_progress": progress,
                            "speed": int(info.get("ram_player_speed", 0)),
                            "lateral_offset": float(
                                info.get("ram_player_lateral_offset", 0.0)
                            ),
                            "heading_alignment": float(
                                info.get("ram_player_heading_alignment", 0.0)
                            ),
                            "pickup_progress_distance": float(
                                info.get(
                                    "ram_player_next_power_up_progress_distance", 0.0
                                )
                            ),
                            "pickup_lateral_error": float(
                                info.get("ram_player_next_power_up_lateral_error", 0.0)
                            ),
                        },
                    )

        action, _ = policy.predict(observation, deterministic=deterministic)
        if candidates and progress < 61:
            trace_observations.append(np.asarray(observation, dtype=np.float32).copy())
            trace_actions.append(np.asarray(action, dtype=np.int64).reshape(3).copy())
        observation, _, terminated, truncated, info = env.step(action)
        pickup_count += int(info.get("ram_decision_jet_boost_pickups", 0))
        next_progress = (
            int(info.get("ram_player_progress", 0)) % DINO_BONEYARD_PROGRESS_COUNT
        )
        if candidates and next_progress >= 61:
            finish_approach(captured_lap)
        if terminated or truncated:
            break

    if captured_lap is not None:
        finish_approach(captured_lap)
    return manifest_states, demo_observations, demo_actions


def main() -> None:
    args = _parser().parse_args()
    if args.stochastic_rollouts < 0:
        raise ValueError("stochastic-rollouts must be nonnegative")
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"output already exists: {manifest_path} (pass --force)")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rom = args.rom.expanduser().resolve()
    prepare_rom(source_rom, (), output_dir / "rom_import")

    policies: list[tuple[str, Path, Any]] = []
    seen_labels: set[str] = set()
    for label, raw_path in args.policy:
        if label in seen_labels:
            raise ValueError(f"duplicate policy label: {label}")
        seen_labels.add(label)
        path = raw_path.expanduser().resolve()
        policies.append((label, path, load_ram_policy(path, device=args.device)))

    all_states: list[dict[str, Any]] = []
    all_demo_observations: list[np.ndarray] = []
    all_demo_actions: list[np.ndarray] = []
    for policy_index, (label, _, policy) in enumerate(policies):
        for rollout in range(1 + args.stochastic_rollouts):
            deterministic = rollout == 0
            rollout_seed = args.seed + policy_index * 1000 + rollout
            np.random.seed(rollout_seed)
            torch.manual_seed(rollout_seed)
            env = make_ram_env(
                frame_skip=args.frame_skip,
                max_episode_steps=args.max_episode_steps,
                seed=rollout_seed,
                reward_config={"mode": "time_trial"},
            )
            try:
                states, observations, actions = _capture_rollout(
                    policy=policy,
                    label=label,
                    rollout=rollout,
                    deterministic=deterministic,
                    env=env,
                    targets=args.targets,
                    output_dir=output_dir,
                    force=args.force,
                    policy_index=policy_index,
                    max_episode_steps=args.max_episode_steps,
                )
            finally:
                env.close()
            all_states.extend(states)
            all_demo_observations.extend(observations)
            all_demo_actions.extend(actions)
            successes = sum(state["teacher_outcome"] == "success" for state in states)
            print(
                f"{label} rollout {rollout}: states={len(states)} "
                f"successful_states={successes} demonstrations={len(observations)}",
                flush=True,
            )

    demonstration_path = output_dir / "successful_demonstrations.npz"
    if all_demo_observations:
        np.savez_compressed(
            demonstration_path,
            observations=np.stack(all_demo_observations),
            actions=np.stack(all_demo_actions),
        )
    else:
        np.savez_compressed(
            demonstration_path,
            observations=np.empty((0, 0), dtype=np.float32),
            actions=np.empty((0, 3), dtype=np.int64),
        )
    manifest = {
        "version": 1,
        "track": "dino_boneyard",
        "pickup_progress": 42.97132782790171,
        "hairpin_exit_progress": 61,
        "targets": list(args.targets),
        "source_rom_sha1": file_sha1(source_rom),
        "policies": {
            label: {"path": str(path), "sha1": file_sha1(path)}
            for label, path, _ in policies
        },
        "states": all_states,
        "demonstrations": str(demonstration_path.relative_to(output_dir)),
        "demonstration_steps": len(all_demo_observations),
    }
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(
        f"wrote {len(all_states)} labeled states and "
        f"{len(all_demo_observations)} successful demonstration steps to "
        f"{output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
