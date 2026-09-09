"""Re-evaluate every RAM checkpoint with a finish-capable race horizon."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter

from .callbacks import evaluate_ram_policy
from .common import load_config, make_ram_env, prepare_rom
from .media import log_video_replay
from .record import record_model

CHECKPOINT_PATTERN = re.compile(r"dino_ram_player_(\d+)_steps\.zip$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate every checkpoint and retain the fastest finisher"
    )
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-record-video",
        action="store_true",
        help="skip the fastest-checkpoint MP4 and TensorBoard replay",
    )
    return parser


def _final_timesteps(run_dir: Path, fallback: int) -> int:
    evaluations = run_dir / "evaluation" / "evaluations.csv"
    if not evaluations.is_file():
        return fallback
    with evaluations.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return int(rows[-1]["timesteps"]) if rows else fallback


def checkpoint_candidates(
    run_dir: Path, configured_timesteps: int
) -> list[tuple[int, str, Path]]:
    """Return periodic checkpoints plus the separately saved final model."""

    candidates: list[tuple[int, str, Path]] = []
    for path in (run_dir / "checkpoints").glob("*.zip"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            steps = int(match.group(1))
            candidates.append((steps, f"checkpoint_{steps}", path))
    final_model = run_dir / "final_model.zip"
    if final_model.is_file():
        steps = _final_timesteps(run_dir, configured_timesteps)
        candidates.append((steps, "final_model", final_model))
    return sorted(candidates, key=lambda candidate: (candidate[0], candidate[1]))


def checkpoint_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    """Prefer reliable finishes, then lower finish time, then race position."""

    finish_rate = float(row["finish_rate"])
    finish_frames = float(row["mean_finish_frames"])
    return (
        float(finish_rate > 0.0),
        finish_rate,
        -finish_frames if finish_rate > 0.0 else float(row["mean_completion"]),
        -float(row["mean_rank"]),
        float(row["mean_reward"]),
    )


def _summary_markdown(rows: list[dict[str, Any]], best: dict[str, Any]) -> str:
    lines = [
        "# Dino RAM checkpoint sweep",
        "",
        f"Selected **{best['label']}** at **{int(best['timesteps']):,} steps**.",
        "",
        "| checkpoint | steps | finish | time | completion | rank |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        finish_time = (
            f"{float(row['mean_finish_frames']) / 60.0:.2f}s"
            if float(row["finish_rate"]) > 0.0
            else "—"
        )
        lines.append(
            f"| {row['label']} | {int(row['timesteps']):,} | "
            f"{float(row['finish_rate']):.0%} | {finish_time} | "
            f"{float(row['mean_completion']):.1%} | "
            f"{float(row['mean_rank']):.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = _parser().parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be at least one")
    if args.max_episode_steps < 1:
        raise ValueError("max-episode-steps must be at least one")
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    config_path = (
        args.config.expanduser().resolve()
        if args.config
        else run_dir / "resolved_config.yml"
    )
    config = load_config(config_path)
    config["evaluation_max_episode_steps"] = args.max_episode_steps
    candidates = checkpoint_candidates(run_dir, int(config["total_timesteps"]))
    if not candidates:
        raise FileNotFoundError(f"no checkpoints found under {run_dir}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = run_dir / f"evaluation_sweep_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    prepare_rom(args.rom, (), output_dir / "private")
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    rows: list[dict[str, Any]] = []
    frame_skip = int(config["frame_skip"])
    env = make_ram_env(
        frame_skip=frame_skip,
        max_episode_steps=args.max_episode_steps,
        seed=int(config["seed"]) + 40_000,
    )
    try:
        for steps, label, model_path in candidates:
            model = PPO.load(model_path, device=args.device)
            result = evaluate_ram_policy(
                model,
                env,
                args.episodes,
                max_episode_frames=args.max_episode_steps * frame_skip,
            )
            row = {
                "label": label,
                "timesteps": steps,
                "model_path": str(model_path.resolve()),
                **asdict(result),
            }
            rows.append(row)
            for name, value in asdict(result).items():
                writer.add_scalar(f"checkpoint_sweep/{name}", value, steps)
            writer.flush()
            time_text = (
                f"{result.mean_finish_frames / 60.0:.2f}s"
                if result.finish_rate
                else "no finish"
            )
            print(
                f"{label}: finish={result.finish_rate:.0%} time={time_text} "
                f"completion={result.mean_completion:.1%} "
                f"rank={result.mean_rank:.2f}",
                flush=True,
            )
    finally:
        env.close()

    best = max(rows, key=checkpoint_sort_key)
    results_csv = output_dir / "checkpoint_results.csv"
    with results_csv.open("w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    fastest_model = output_dir / "fastest_model.zip"
    shutil.copy2(best["model_path"], fastest_model)

    replay: dict[str, Any] | None = None
    replay_gif: Path | None = None
    if not args.no_record_video:
        video_path = output_dir / "fastest_model.mp4"
        replay = record_model(fastest_model, video_path, config, device=args.device)
        replay_gif = log_video_replay(
            writer,
            "evaluation_replays/fastest_ram_checkpoint",
            video_path,
            global_step=int(best["timesteps"]),
        )

    writer.add_text("checkpoint_sweep/results", _summary_markdown(rows, best), 0)
    writer.flush()
    writer.close()
    payload = {
        "source_run": str(run_dir),
        "config": str(config_path),
        "episodes_per_checkpoint": args.episodes,
        "max_episode_steps": args.max_episode_steps,
        "selected": best,
        "fastest_model": str(fastest_model),
        "replay": replay,
        "tensorboard_replay": str(replay_gif) if replay_gif else None,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Selected {best['label']} -> {fastest_model}")
    print(f"Sweep results and TensorBoard replay written to {output_dir}")


if __name__ == "__main__":
    main()
