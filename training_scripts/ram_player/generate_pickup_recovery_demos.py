"""Generate corrective pickup-and-recovery demonstrations from saved states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import file_sha1, load_ram_policy, prepare_rom
from .pickup_lab import PickupFeedbackCompositePolicy, make_pickup_lab_env


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay pickup-lab training states with RAM feedback and retain only "
            "trajectories that collect the Jet Boost and recover to a later progress"
        )
    )
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-policy", action="append", default=[])
    parser.add_argument("--exit-progress", type=float, default=100.0)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--device", default="cpu")
    return parser


def _eligible_entries(
    manifest: dict[str, Any],
    source_policies: set[str],
) -> list[dict[str, Any]]:
    entries = manifest.get("states")
    if not isinstance(entries, list):
        raise ValueError("pickup-state manifest has no states list")
    return [
        entry
        for entry in entries
        if entry.get("split") != "evaluation"
        and (
            not source_policies
            or str(entry.get("source_policy")) in source_policies
        )
    ]


def main() -> None:
    args = _parser().parse_args()
    if args.exit_progress <= 43.0:
        raise ValueError("exit-progress must be after the first Jet Boost")
    if args.frame_skip < 1 or args.max_episode_steps < 1:
        raise ValueError("frame-skip and max-episode-steps must be positive")

    manifest_path = args.manifest.expanduser().resolve()
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    entries = _eligible_entries(manifest, set(args.source_policy))
    if not entries:
        raise ValueError("no matching training states in pickup-state manifest")

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_rom = args.rom.expanduser().resolve()
    prepare_rom(source_rom, (), output_path.parent / "rom_import")
    base_model_path = args.base_model.expanduser().resolve()
    base = load_ram_policy(base_model_path, device=args.device)
    policy = PickupFeedbackCompositePolicy(base)

    successful_observations: list[np.ndarray] = []
    successful_actions: list[np.ndarray] = []
    results: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        state_path = (manifest_path.parent / str(entry["path"])).resolve()
        env = make_pickup_lab_env(
            frame_skip=args.frame_skip,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed + index,
            state_path=str(state_path),
            task_config={"exit_progress": args.exit_progress},
        )
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        try:
            policy.reset()
            observation, _ = env.reset()
            terminated = truncated = False
            info: dict[str, Any] = {}
            while not (terminated or truncated):
                action, _ = policy.predict(observation, deterministic=True)
                observations.append(np.asarray(observation, dtype=np.float32).copy())
                actions.append(np.asarray(action, dtype=np.int64).reshape(3).copy())
                observation, _, terminated, truncated, info = env.step(action)
        finally:
            env.close()

        success = bool(info.get("pickup_task_success", False))
        if success:
            successful_observations.extend(observations)
            successful_actions.extend(actions)
        result = {
            "path": entry["path"],
            "source_policy": entry.get("source_policy"),
            "lap": entry.get("lap"),
            "target_progress": entry.get("target_progress"),
            "success": success,
            "pickup": bool(info.get("pickup_task_acquired", False)),
            "frames": int(info.get("pickup_task_frames", 0)),
            "exit_speed": int(info.get("pickup_task_exit_speed", 0)),
            "exit_abs_lateral_offset": float(
                info.get("pickup_task_exit_abs_lateral_offset", 0.0)
            ),
            "exit_heading_alignment": float(
                info.get("pickup_task_exit_heading_alignment", 0.0)
            ),
        }
        results.append(result)
        print(
            f"[{index + 1}/{len(entries)}] {entry['path']}: "
            f"pickup={result['pickup']} success={success} frames={result['frames']}",
            flush=True,
        )

    if not successful_observations:
        raise RuntimeError("feedback controller produced no successful trajectories")
    np.savez_compressed(
        output_path,
        observations=np.stack(successful_observations),
        actions=np.stack(successful_actions),
    )
    report = {
        "version": 1,
        "manifest": str(manifest_path),
        "manifest_sha1": file_sha1(manifest_path),
        "base_model": str(base_model_path),
        "base_model_sha1": file_sha1(base_model_path),
        "source_policies": sorted(set(args.source_policy)),
        "exit_progress": args.exit_progress,
        "states": len(entries),
        "successful_states": sum(int(result["success"]) for result in results),
        "demonstration_steps": len(successful_observations),
        "results": results,
    }
    report_path = output_path.with_suffix(".json")
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(
        f"Wrote {len(successful_observations)} steps from "
        f"{report['successful_states']}/{len(entries)} successful states to "
        f"{output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
