"""Evaluate matched 67-D and 68-D continuous-progress ablation checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter

from .callbacks import evaluate_ram_policy, evaluation_values
from .common import (
    DEFAULT_RUN_ROOT,
    evaluation_episode_steps,
    evaluation_state_paths,
    load_config,
    make_ram_env,
    normalize_opponent_league,
    opponent_state_path,
    prepare_rom,
    validate_model_action_space,
    validate_model_observation_space,
)
from .legacy_policy import adapt_legacy_policy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare matched Dino continuous-progress ablation checkpoints"
    )
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--control-model", type=Path, required=True)
    parser.add_argument("--experiment-model", type=Path, required=True)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--device", default="cpu")
    return parser


def _matched_configs(control: Mapping[str, Any], experiment: Mapping[str, Any]) -> None:
    ignored = {"run_name", "include_continuous_progress_rate"}
    control_fields = {key: value for key, value in control.items() if key not in ignored}
    experiment_fields = {
        key: value for key, value in experiment.items() if key not in ignored
    }
    if control_fields != experiment_fields:
        raise ValueError("control and experiment configs differ outside the feature flag")
    if bool(control.get("include_continuous_progress_rate", False)):
        raise ValueError("control config must retain the 67-D observation")
    if not bool(experiment.get("include_continuous_progress_rate", False)):
        raise ValueError("experiment config must enable the 68th observation")


def _load_policy(path: Path, *, include_continuous_progress_rate: bool, device: str) -> Any:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    loaded = PPO.load(resolved, device=device)
    policy = adapt_legacy_policy(
        loaded, include_continuous_progress_rate=include_continuous_progress_rate
    )
    if policy is loaded:
        validate_model_observation_space(
            policy,
            resolved,
            include_continuous_progress_rate=include_continuous_progress_rate,
        )
        validate_model_action_space(policy, resolved)
    return policy


def _evaluate_stock_suite(
    label: str,
    policy: Any,
    config: Mapping[str, Any],
    states: list[Path],
) -> list[dict[str, Any]]:
    include_feature = bool(config.get("include_continuous_progress_rate", False))
    rows: list[dict[str, Any]] = []
    for index, state in enumerate(states):
        env = make_ram_env(
            frame_skip=int(config["frame_skip"]),
            max_episode_steps=evaluation_episode_steps(config),
            seed=int(config["seed"]) + 30_000 + index,
            state_path=str(state),
            reward_config=config.get("reward"),
            include_continuous_progress_rate=include_feature,
        )
        try:
            result = evaluate_ram_policy(
                policy,
                env,
                1,
                max_episode_frames=(
                    evaluation_episode_steps(config) * int(config["frame_skip"])
                ),
            )
        finally:
            env.close()
        rows.append({"model": label, "state": state.name, **evaluation_values(result)})
    return rows


def _evaluate_league(
    label: str,
    policy: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    league = normalize_opponent_league(config.get("opponent_league"))
    if not league:
        raise ValueError("ablation config must provide a frozen opponent league")
    include_feature = bool(config.get("include_continuous_progress_rate", False))
    env = make_ram_env(
        frame_skip=int(config["frame_skip"]),
        max_episode_steps=evaluation_episode_steps(config),
        seed=int(config["seed"]) + 40_000,
        opponent_league={name: str(path) for name, path in league.items()},
        state_path=str(opponent_state_path(config)),
        reward_config=config.get("reward"),
        include_continuous_progress_rate=include_feature,
    )
    try:
        result = evaluate_ram_policy(
            policy,
            env,
            int(config.get("league_eval_episodes", 3)),
            max_episode_frames=(
                evaluation_episode_steps(config) * int(config["frame_skip"])
            ),
        )
    finally:
        env.close()
    return {"model": label, **evaluation_values(result)}


def _seconds(frames: float) -> str:
    return f"{frames / 60.0:.2f}" if frames else "—"


def _markdown(rows: list[dict[str, Any]], league_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Continuous-progress observation ablation",
        "",
        "## Fixed stock-traffic state suite",
        "",
        "| model | state | finish | median finish s | progress/frame | respawns | wall |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['state']} | {row['finish_rate']:.0%} | "
            f"{_seconds(float(row['median_finish_frames']))} | "
            f"{row['mean_continuous_progress_per_frame']:.4f} | "
            f"{row['mean_respawns']:.2f} | {row['wall_contact_rate']:.2%} |"
        )
    lines.extend(
        (
            "",
            "## Same frozen-opponent league",
            "",
            "| model | finish | win | median finish s | progress/frame | respawns | wall |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in league_rows:
        lines.append(
            f"| {row['model']} | {row['finish_rate']:.0%} | {row['win_rate']:.0%} | "
            f"{_seconds(float(row['median_finish_frames']))} | "
            f"{row['mean_continuous_progress_per_frame']:.4f} | "
            f"{row['mean_respawns']:.2f} | {row['wall_contact_rate']:.2%} |"
        )
    lines.extend(
        (
            "",
            "The checked-in suite has no recovery/crash savestate. No state was "
            "fabricated for this controlled comparison.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    args = _parser().parse_args()
    control_config = load_config(args.control_config.expanduser().resolve())
    experiment_config = load_config(args.experiment_config.expanduser().resolve())
    _matched_configs(control_config, experiment_config)
    control = _load_policy(
        args.control_model,
        include_continuous_progress_rate=False,
        device=args.device,
    )
    experiment = _load_policy(
        args.experiment_model,
        include_continuous_progress_rate=True,
        device=args.device,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root.expanduser().resolve() / f"continuous_progress_eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    prepare_rom(args.rom, (1, 2, 3), output_dir / "private")

    states = evaluation_state_paths(control_config)
    stock_rows = _evaluate_stock_suite("control", control, control_config, states)
    stock_rows.extend(
        _evaluate_stock_suite("experiment", experiment, experiment_config, states)
    )
    league_rows = [
        _evaluate_league("control", control, control_config),
        _evaluate_league("experiment", experiment, experiment_config),
    ]
    markdown = _markdown(stock_rows, league_rows)
    with (output_dir / "stock_suite.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(stock_rows[0]))
        writer.writeheader()
        writer.writerows(stock_rows)
    with (output_dir / "league.json").open("w") as handle:
        json.dump(league_rows, handle, indent=2)
        handle.write("\n")
    with (output_dir / "summary.md").open("w") as handle:
        handle.write(markdown)
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(
            {"stock_suite": stock_rows, "league": league_rows}, handle, indent=2
        )
        handle.write("\n")
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    try:
        for row in stock_rows:
            prefix = f"stock/{row['model']}/{row['state']}"
            for key in (
                "finish_rate",
                "median_finish_frames",
                "mean_continuous_progress_per_frame",
                "mean_respawns",
                "wall_contact_rate",
            ):
                writer.add_scalar(f"{prefix}/{key}", float(row[key]), 0)
        for row in league_rows:
            prefix = f"league/{row['model']}"
            for key in (
                "finish_rate",
                "win_rate",
                "median_finish_frames",
                "mean_continuous_progress_per_frame",
                "mean_respawns",
                "wall_contact_rate",
            ):
                writer.add_scalar(f"{prefix}/{key}", float(row[key]), 0)
        writer.add_text("comparison/results", markdown, 0)
    finally:
        writer.close()
    print(markdown)
    print(f"Comparison CSV, JSON, Markdown, and TensorBoard written to {output_dir}")


if __name__ == "__main__":
    main()
