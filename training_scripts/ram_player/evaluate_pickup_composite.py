"""Compare and record a pickup expert gated into a frozen full-race policy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .callbacks import evaluate_ram_policy
from .common import (
    DEFAULT_CONFIG,
    evaluation_episode_steps,
    load_config,
    load_ram_policy,
    make_ram_env,
    prepare_rom,
)
from .media import RGBVideoWriter
from .pickup_lab import PickupExpertCompositePolicy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a frozen race policy with its pickup-expert composite"
    )
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--expert-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser


def _environment(config: Mapping[str, Any], *, seed_offset: int = 0) -> Any:
    return make_ram_env(
        frame_skip=int(config["frame_skip"]),
        max_episode_steps=evaluation_episode_steps(config),
        seed=int(config["seed"]) + seed_offset,
        reward_config=config.get("reward"),
    )


def _evaluate(policy: Any, config: Mapping[str, Any], episodes: int) -> dict[str, Any]:
    env = _environment(config)
    try:
        result = evaluate_ram_policy(
            policy,
            env,
            episodes,
            max_episode_frames=(
                evaluation_episode_steps(config) * int(config["frame_skip"])
            ),
        )
    finally:
        env.close()
    return asdict(result)


def _record(
    policy: Any,
    config: Mapping[str, Any],
    video_path: Path,
) -> dict[str, Any]:
    video_path = video_path.expanduser().resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    env = _environment(config, seed_offset=30_000)
    reset_policy = getattr(policy, "reset", None)
    if callable(reset_policy):
        reset_policy()
    observation, info = env.reset()
    writer = RGBVideoWriter(
        video_path,
        env.render(),
        fps=60 / int(config["frame_skip"]),
    )
    terminated = truncated = False
    decisions = 0
    pickups = 0
    try:
        while not (terminated or truncated):
            action, _ = policy.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            decisions += 1
            pickups += int(info.get("ram_decision_jet_boost_pickups", 0))
            writer.write(env.render())
    finally:
        writer.close()
        env.close()
    finish_frame = info.get("ram_player_finish_frame")
    return {
        "path": str(video_path),
        "decisions": decisions,
        "finished": bool(info.get("ram_player_finished", False)),
        "finish_frame": finish_frame,
        "finish_seconds": (
            float(finish_frame) / 60.0 if finish_frame is not None else None
        ),
        "rank": int(info.get("ram_player_rank", 4)),
        "completion": float(info.get("ram_player_completion", 0.0)),
        "jet_boost_pickups": pickups,
        "lap_split_seconds": [
            int(info.get(f"ram_player_lap_{lap}_frames", 0)) / 60.0 for lap in (1, 2, 3)
        ],
        "hairpin_jet_boost_entries": int(
            info.get("ram_player_hairpin_jet_boost_entries", 0)
        ),
    }


def main() -> None:
    args = _parser().parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepare_rom(
        args.rom.expanduser().resolve(),
        (),
        output_path.parent / "private",
    )
    config = load_config(args.config.expanduser().resolve())
    base = load_ram_policy(args.base_model, device=args.device)
    expert = load_ram_policy(args.expert_model, device=args.device)
    composite = PickupExpertCompositePolicy(base, expert)

    base_result = _evaluate(base, config, args.episodes)
    composite_result = _evaluate(composite, config, args.episodes)
    report: dict[str, Any] = {
        "base_model": str(args.base_model.expanduser().resolve()),
        "expert_model": str(args.expert_model.expanduser().resolve()),
        "episodes": args.episodes,
        "base": base_result,
        "composite": composite_result,
        "delta": {
            "finish_frames": (
                composite_result["mean_finish_frames"]
                - base_result["mean_finish_frames"]
            ),
            "jet_boost_pickups": (
                composite_result["mean_jet_boost_pickups"]
                - base_result["mean_jet_boost_pickups"]
            ),
            "hairpin_seconds": (
                composite_result["mean_hairpin_seconds"]
                - base_result["mean_hairpin_seconds"]
            ),
        },
    }
    if args.video is not None:
        report["video"] = _record(composite, config, args.video)
    with output_path.open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(
        "Base: "
        f"finished={base_result['finish_rate']:.0%} "
        f"frames={base_result['mean_finish_frames']:.0f} "
        f"pickups={base_result['mean_jet_boost_pickups']:.1f}"
    )
    print(
        "Composite: "
        f"finished={composite_result['finish_rate']:.0%} "
        f"frames={composite_result['mean_finish_frames']:.0f} "
        f"pickups={composite_result['mean_jet_boost_pickups']:.1f}"
    )
    print(f"Report saved to {output_path}")
    if args.video is not None:
        print(f"Video saved to {args.video.expanduser().resolve()}")


if __name__ == "__main__":
    main()
