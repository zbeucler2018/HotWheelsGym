"""Record and measure a RAM Player 1 policy racing three RAM-model opponents."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter

from .common import (
    DEFAULT_CONFIG,
    DEFAULT_RUN_ROOT,
    evaluation_episode_steps,
    load_config,
    prepare_rom,
)
from .media import log_video_replay
from .record import record_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a Player 1 RAM model against three frozen RAM opponents"
    )
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--player-model", type=Path, required=True)
    parser.add_argument("--opponent-model", type=Path)
    parser.add_argument(
        "--state",
        type=Path,
        required=True,
        help="fresh Dino state containing four player-class racers",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--device", default="cpu")
    return parser


def _summary_markdown(result: dict[str, Any]) -> str:
    player_time = (
        f"{float(result['raw_frames']) / 60.0:.2f}s" if result["finished"] else "—"
    )
    rows = [
        "# RAM self-play validation",
        "",
        "| racer | finish | time | completion | rank |",
        "|---|---:|---:|---:|---:|",
        f"| Player 1 | {bool(result['finished'])!s} | {player_time} | "
        f"{float(result['completion']):.1%} | {int(result['rank'])} |",
    ]
    for slot, opponent in result["opponents"].items():
        opponent_time = (
            f"{float(opponent['finish_frame']) / 60.0:.2f}s"
            if opponent["finished"] and opponent["finish_frame"] is not None
            else "—"
        )
        rows.append(
            f"| Model NPC slot {slot} | {bool(opponent['finished'])!s} | "
            f"{opponent_time} | {float(opponent['completion']):.1%} | "
            f"{int(opponent['rank'])} |"
        )
    return "\n".join(rows)


def main() -> None:
    args = _parser().parse_args()
    player_model = args.player_model.expanduser().resolve()
    opponent_model = (
        args.opponent_model.expanduser().resolve()
        if args.opponent_model
        else player_model
    )
    if not player_model.is_file():
        raise FileNotFoundError(player_model)
    if not opponent_model.is_file():
        raise FileNotFoundError(opponent_model)
    config = load_config(args.config.expanduser().resolve())
    config["opponent_state"] = str(args.state.expanduser().resolve())
    max_episode_steps = (
        args.max_episode_steps
        if args.max_episode_steps is not None
        else evaluation_episode_steps(config)
    )
    if max_episode_steps < 1:
        raise ValueError("max-episode-steps must be at least one")
    config["evaluation_max_episode_steps"] = max_episode_steps

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root.expanduser().resolve() / f"self_play_eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    prepare_rom(args.rom, (1, 2, 3), output_dir / "private")
    video_path = output_dir / "ram_self_play.mp4"
    result = record_model(
        player_model,
        video_path,
        config,
        opponent_paths={slot: str(opponent_model) for slot in (1, 2, 3)},
        device=args.device,
    )

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    try:
        player_metrics = {
            "finished": float(result["finished"]),
            "finish_seconds": (
                float(result["raw_frames"]) / 60.0 if result["finished"] else 0.0
            ),
            "completion": float(result["completion"]),
            "rank": float(result["rank"]),
            "final_abs_lateral_offset": abs(float(result["lateral_offset"])),
            "final_heading_alignment": float(result["heading_alignment"]),
        }
        for name, value in player_metrics.items():
            writer.add_scalar(f"self_play/player/{name}", value, 0)
        writer.add_scalar(
            "self_play/system/masked_opponent_respawn_flash_frames",
            float(result.get("masked_opponent_respawn_flash_frames", 0)),
            0,
        )
        for slot, opponent in result["opponents"].items():
            opponent_metrics = {
                "finished": float(opponent["finished"]),
                "finish_seconds": (
                    float(opponent["finish_frame"]) / 60.0
                    if opponent["finished"] and opponent["finish_frame"] is not None
                    else 0.0
                ),
                "completion": float(opponent["completion"]),
                "rank": float(opponent["rank"]),
                "final_abs_lateral_offset": abs(
                    float(opponent["lateral_offset"])
                ),
                "final_heading_alignment": float(opponent["heading_alignment"]),
            }
            for name, value in opponent_metrics.items():
                writer.add_scalar(f"self_play/npc_{slot}/{name}", value, 0)
        writer.add_text("self_play/results", _summary_markdown(result), 0)
        replay = log_video_replay(
            writer, "evaluation_replays/ram_self_play", video_path
        )
    finally:
        writer.close()

    summary = {
        "player_model": str(player_model),
        "opponent_model": str(opponent_model),
        "opponent_slots": [1, 2, 3],
        "max_episode_steps": max_episode_steps,
        "tensorboard_replay": str(replay),
        "result": result,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))
    print(f"Self-play results and TensorBoard replay written to {output_dir}")


if __name__ == "__main__":
    main()
