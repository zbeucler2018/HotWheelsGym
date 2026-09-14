"""Validate a mixed frozen-model league without performing any training."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from torch.utils.tensorboard import SummaryWriter

from .common import (
    DEFAULT_CONFIG,
    DEFAULT_RUN_ROOT,
    evaluation_episode_steps,
    load_config,
    prepare_rom,
)
from .record import record_model


def _named_model(value: str) -> tuple[str, Path]:
    try:
        label, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("use LABEL=RAM_MODEL_PATH") from error
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("label and model path must be non-empty")
    return label, Path(raw_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Race one current-contract Player 1 policy against three frozen "
            "RAM policies, rotating each opponent through every native slot"
        )
    )
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--player-model", type=Path, required=True)
    parser.add_argument(
        "--opponent",
        action="append",
        type=_named_model,
        required=True,
        metavar="LABEL=RAM_MODEL_PATH",
        help="provide exactly three uniquely labelled v5-or-newer RAM policies",
    )
    parser.add_argument(
        "--state",
        type=Path,
        required=True,
        help="private Dino starting state containing four player-class racers",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--device", default="cpu")
    return parser


def _rotations(models: list[tuple[str, Path]]) -> list[dict[int, tuple[str, Path]]]:
    return [
        {slot: models[(slot - 1 + offset) % len(models)] for slot in (1, 2, 3)}
        for offset in range(len(models))
    ]


def _markdown(summary: dict[str, Any]) -> str:
    rows = [
        "# Phase 0 mixed-league validation",
        "",
        "| race | slot 1 | slot 2 | slot 3 | P1 time | P1 rank | P1 pickup |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for race in summary["races"]:
        result = race["result"]
        finish = (
            f"{float(result['raw_frames']) / 60.0:.2f}s" if result["finished"] else "—"
        )
        assignments = race["assignments"]
        rows.append(
            f"| {race['rotation']} | {assignments['1']} | {assignments['2']} | "
            f"{assignments['3']} | {finish} | {result['rank']} | "
            f"{result['jet_boost_pickups']} |"
        )
    rows.extend(
        (
            "",
            f"Player win rate: **{summary['player_win_rate']:.0%}**",
            "",
            "NPC Jet Boost pickups: "
            + ", ".join(
                f"{label}={pickups}"
                for label, pickups in summary["npc_jet_boost_pickups"].items()
            ),
        )
    )
    return "\n".join(rows) + "\n"


def main() -> None:
    args = _parser().parse_args()
    if len(args.opponent) != 3:
        raise ValueError("Phase 0 requires exactly three opponent models")
    labels = [label for label, _ in args.opponent]
    if len(labels) != len(set(labels)):
        raise ValueError("Phase 0 opponent labels must be unique")

    source_rom = args.rom.expanduser().resolve()
    player_model = args.player_model.expanduser().resolve()
    state = args.state.expanduser().resolve()
    models = [(label, path.expanduser().resolve()) for label, path in args.opponent]
    for path in (player_model, state, *(path for _, path in models)):
        if not path.is_file():
            raise FileNotFoundError(path)

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    config["opponent_state"] = str(state)
    max_episode_steps = int(
        args.max_episode_steps
        if args.max_episode_steps is not None
        else evaluation_episode_steps(config)
    )
    if max_episode_steps < 1:
        raise ValueError("max-episode-steps must be at least one")
    config["evaluation_max_episode_steps"] = max_episode_steps

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_root.expanduser().resolve() / f"phase0_league_eval_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    prepare_rom(source_rom, (1, 2, 3), output_dir / "private")

    races: list[dict[str, Any]] = []
    for rotation, assignment in enumerate(_rotations(models), start=1):
        labels_by_slot = {str(slot): label for slot, (label, _) in assignment.items()}
        paths_by_slot = {slot: str(path) for slot, (_, path) in assignment.items()}
        print(
            f"Phase 0 race {rotation}/3: "
            + ", ".join(
                f"slot {slot}={labels_by_slot[str(slot)]}" for slot in (1, 2, 3)
            ),
            flush=True,
        )
        result = record_model(
            player_model,
            output_dir / f"rotation_{rotation}.mp4",
            config,
            opponent_paths=paths_by_slot,
            device=args.device,
        )
        races.append(
            {
                "rotation": rotation,
                "assignments": labels_by_slot,
                "result": result,
            }
        )
        print(
            f"  Player 1 rank={result['rank']} "
            f"time={float(result['raw_frames']) / 60.0:.2f}s "
            f"pickups={result['jet_boost_pickups']}",
            flush=True,
        )

    wins = [
        float(race["result"]["finished"] and race["result"]["rank"] == 1)
        for race in races
    ]
    finish_seconds = [
        float(race["result"]["raw_frames"]) / 60.0
        for race in races
        if race["result"]["finished"]
    ]
    npc_pickups = {label: 0 for label in labels}
    npc_completions = {label: [] for label in labels}
    npc_ranks = {label: [] for label in labels}
    for race in races:
        for raw_slot, label in race["assignments"].items():
            opponent = race["result"]["opponents"][raw_slot]
            npc_pickups[label] += int(opponent["jet_boost_pickups"])
            npc_completions[label].append(float(opponent["completion"]))
            npc_ranks[label].append(float(opponent["rank"]))

    summary = {
        "player_model": str(player_model),
        "state": str(state),
        "opponent_models": {label: str(path) for label, path in models},
        "player_win_rate": fmean(wins),
        "player_finish_rate": fmean(
            float(race["result"]["finished"]) for race in races
        ),
        "player_mean_rank": fmean(float(race["result"]["rank"]) for race in races),
        "player_mean_finish_seconds": fmean(finish_seconds) if finish_seconds else 0.0,
        "npc_jet_boost_pickups": npc_pickups,
        "npc_mean_completion": {
            label: fmean(values) for label, values in npc_completions.items()
        },
        "npc_mean_rank": {label: fmean(values) for label, values in npc_ranks.items()},
        "races": races,
    }
    markdown = _markdown(summary)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    with (output_dir / "summary.md").open("w") as handle:
        handle.write(markdown)

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    try:
        writer.add_scalar("phase0/player/win_rate", summary["player_win_rate"], 0)
        writer.add_scalar(
            "phase0/player/mean_finish_seconds",
            summary["player_mean_finish_seconds"],
            0,
        )
        for label in labels:
            writer.add_scalar(
                f"phase0/npc_{label}/jet_boost_pickups",
                npc_pickups[label],
                0,
            )
            writer.add_scalar(
                f"phase0/npc_{label}/mean_completion",
                summary["npc_mean_completion"][label],
                0,
            )
        writer.add_text("phase0/results", markdown, 0)
    finally:
        writer.close()

    print(markdown)
    print(f"Phase 0 results and videos written to {output_dir}")


if __name__ == "__main__":
    main()
