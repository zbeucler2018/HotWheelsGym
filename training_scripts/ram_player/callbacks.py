"""Evaluation callback dedicated to the Player 1 RAM model."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback

from HotWheelsGym.ram_opponent_control import (
    DINO_SECTOR_COUNT,
    RAM_ACTION_METRIC_NAMES,
)

SECTOR_TOTAL_FIELDS = (
    "entries",
    "completed",
    "frames",
    "completed_frames",
    "entry_speed_total",
    "minimum_speed_total",
    "exit_speed_total",
    "wall_frames",
    "skid_frames",
    "boost_frames",
    "jet_boost_frames",
    "jet_boost_pickups",
) + tuple(f"action_{name}_frames" for name in RAM_ACTION_METRIC_NAMES)
SECTOR_METRIC_SUFFIXES = (
    "completions_per_episode",
    "seconds",
    "entry_speed",
    "minimum_speed",
    "exit_speed",
    "wall_contact_rate",
    "skid_active_rate",
    "boost_active_rate",
    "jet_boost_active_rate",
    "jet_boost_pickup_rate",
) + tuple(f"action_{name}_rate" for name in RAM_ACTION_METRIC_NAMES)

HAIRPIN_ACTION_GROUPS = ("all", "jet", "nojet")
HAIRPIN_INFO_PREFIXES = {"all": "", "jet": "jet_boost_", "nojet": "no_jet_boost_"}
ACTION_METRIC_NAMES = tuple(
    [f"race_{name}_rate" for name in RAM_ACTION_METRIC_NAMES]
    + [
        f"hairpin_{group}_{name}_rate"
        for group in HAIRPIN_ACTION_GROUPS
        for name in RAM_ACTION_METRIC_NAMES
    ]
)


@dataclass(frozen=True)
class RAMEvaluation:
    mean_reward: float
    mean_length: float
    mean_completion: float
    finish_rate: float
    win_rate: float
    mean_finish_frames: float
    mean_speed: float
    mean_rank: float
    mean_abs_lateral_offset: float
    mean_heading_alignment: float
    wall_contact_rate: float
    mean_respawns: float
    mean_boost_charge: float
    mean_boost_spent: float
    mean_boost_gained: float
    boost_active_rate: float
    mean_jet_boost_remaining: float
    jet_boost_active_rate: float
    mean_jet_boost_pickups: float
    skid_active_rate: float
    mean_hairpin_completions: float
    mean_hairpin_seconds: float
    mean_hairpin_entry_speed: float
    mean_hairpin_minimum_speed: float
    mean_hairpin_exit_speed: float
    hairpin_wall_contact_rate: float
    hairpin_skid_active_rate: float
    hairpin_jet_boost_entry_rate: float
    mean_lap_1_seconds: float
    mean_lap_2_seconds: float
    mean_lap_3_seconds: float
    selection_score: float
    competitive_selection_score: float
    action_metrics: dict[str, float]
    sector_metrics: dict[str, float]


def _empty_sector_totals() -> dict[str, list[float]]:
    return {name: [0.0] * DINO_SECTOR_COUNT for name in SECTOR_TOTAL_FIELDS}


def _accumulate_sector_info(
    totals: dict[str, list[float]], info: dict[str, Any]
) -> None:
    for sector in range(DINO_SECTOR_COUNT):
        for name in SECTOR_TOTAL_FIELDS:
            totals[name][sector] += float(
                info.get(f"ram_player_sector_{sector:02d}_{name}", 0)
            )


def _sector_metrics(totals: dict[str, list[float]], episodes: int) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for sector in range(DINO_SECTOR_COUNT):
        prefix = f"sector_{sector:02d}_"
        entries = totals["entries"][sector]
        completed = totals["completed"][sector]
        frames = totals["frames"][sector]
        metrics.update(
            {
                f"{prefix}completions_per_episode": completed / max(1, episodes),
                f"{prefix}seconds": (
                    totals["completed_frames"][sector] / max(1.0, completed) / 60.0
                ),
                f"{prefix}entry_speed": (
                    totals["entry_speed_total"][sector] / max(1.0, entries)
                ),
                f"{prefix}minimum_speed": (
                    totals["minimum_speed_total"][sector] / max(1.0, completed)
                ),
                f"{prefix}exit_speed": (
                    totals["exit_speed_total"][sector] / max(1.0, completed)
                ),
                f"{prefix}wall_contact_rate": (
                    totals["wall_frames"][sector] / max(1.0, frames)
                ),
                f"{prefix}skid_active_rate": (
                    totals["skid_frames"][sector] / max(1.0, frames)
                ),
                f"{prefix}boost_active_rate": (
                    totals["boost_frames"][sector] / max(1.0, frames)
                ),
                f"{prefix}jet_boost_active_rate": (
                    totals["jet_boost_frames"][sector] / max(1.0, frames)
                ),
                f"{prefix}jet_boost_pickup_rate": (
                    totals["jet_boost_pickups"][sector] / max(1.0, entries)
                ),
            }
        )
        for name in RAM_ACTION_METRIC_NAMES:
            metrics[f"{prefix}action_{name}_rate"] = totals[f"action_{name}_frames"][
                sector
            ] / max(1.0, frames)
    return metrics


def _empty_lap_sector_values() -> dict[int, list[float]]:
    return {lap: [0.0] * DINO_SECTOR_COUNT for lap in (1, 2, 3)}


def _accumulate_lap_sector_info(
    frame_totals: dict[int, list[float]],
    samples: dict[int, list[float]],
    info: dict[str, Any],
) -> None:
    for lap in frame_totals:
        for sector in range(DINO_SECTOR_COUNT):
            frames = float(
                info.get(f"ram_player_lap_{lap}_sector_{sector:02d}_frames", 0)
            )
            if frames > 0:
                frame_totals[lap][sector] += frames
                samples[lap][sector] += 1


def _lap_sector_metrics(
    frame_totals: dict[int, list[float]], samples: dict[int, list[float]]
) -> dict[str, float]:
    return {
        f"lap_{lap}_sector_{sector:02d}_seconds": (
            frame_totals[lap][sector] / max(1.0, samples[lap][sector]) / 60.0
        )
        for lap in frame_totals
        for sector in range(DINO_SECTOR_COUNT)
    }


def sector_metrics_from_info(info: dict[str, Any]) -> dict[str, float]:
    """Return derived full-track sector metrics for one completed episode."""

    totals = _empty_sector_totals()
    _accumulate_sector_info(totals, info)
    frame_totals = _empty_lap_sector_values()
    samples = _empty_lap_sector_values()
    _accumulate_lap_sector_info(frame_totals, samples, info)
    return {
        **_sector_metrics(totals, 1),
        **_lap_sector_metrics(frame_totals, samples),
    }


def evaluation_metric_names() -> tuple[str, ...]:
    scalar_names = tuple(
        item.name
        for item in fields(RAMEvaluation)
        if item.name not in {"action_metrics", "sector_metrics"}
    )
    sector_names = tuple(
        f"sector_{sector:02d}_{suffix}"
        for sector in range(DINO_SECTOR_COUNT)
        for suffix in SECTOR_METRIC_SUFFIXES
    )
    lap_sector_names = tuple(
        f"lap_{lap}_sector_{sector:02d}_seconds"
        for lap in (1, 2, 3)
        for sector in range(DINO_SECTOR_COUNT)
    )
    return scalar_names + ACTION_METRIC_NAMES + sector_names + lap_sector_names


def evaluation_values(result: RAMEvaluation) -> dict[str, float]:
    values = asdict(result)
    action_metrics = values.pop("action_metrics")
    sector_metrics = values.pop("sector_metrics")
    values.update(action_metrics)
    values.update(sector_metrics)
    return values


def evaluate_ram_policy(
    model: Any,
    env: gym.Env,
    episodes: int,
    *,
    max_episode_frames: int,
) -> RAMEvaluation:
    """Evaluate deterministic races, ranking finishers by emulator frames."""

    if episodes < 1:
        raise ValueError("evaluation requires at least one episode")
    rewards: list[float] = []
    lengths: list[int] = []
    completions: list[float] = []
    finishes: list[float] = []
    wins: list[float] = []
    finish_frames: list[float] = []
    mean_speeds: list[float] = []
    ranks: list[float] = []
    lateral_offsets: list[float] = []
    heading_alignments: list[float] = []
    wall_contact_rates: list[float] = []
    respawn_counts: list[float] = []
    boost_charges: list[float] = []
    boost_spent_totals: list[float] = []
    boost_gained_totals: list[float] = []
    boost_active_rates: list[float] = []
    jet_boost_remaining_values: list[float] = []
    jet_boost_active_rates: list[float] = []
    jet_boost_pickup_counts: list[float] = []
    skid_active_rates: list[float] = []
    hairpin_completions: list[float] = []
    hairpin_entries = 0
    hairpin_completed = 0
    hairpin_jet_boost_entries = 0
    hairpin_frames = 0
    hairpin_completed_frames = 0
    hairpin_entry_speed_total = 0.0
    hairpin_minimum_speed_total = 0.0
    hairpin_exit_speed_total = 0.0
    hairpin_wall_frames = 0
    hairpin_skid_frames = 0
    lap_splits: dict[int, list[float]] = {lap: [] for lap in (1, 2, 3)}
    sector_totals = _empty_sector_totals()
    lap_sector_frame_totals = _empty_lap_sector_values()
    lap_sector_samples = _empty_lap_sector_values()
    scores: list[float] = []
    competitive_scores: list[float] = []
    hairpin_action_totals = {
        group: dict.fromkeys(RAM_ACTION_METRIC_NAMES, 0.0)
        for group in HAIRPIN_ACTION_GROUPS
    }
    hairpin_condition_frames = dict.fromkeys(HAIRPIN_ACTION_GROUPS, 0.0)

    for _ in range(episodes):
        observation, _ = env.reset()
        terminated = truncated = False
        episode_reward = 0.0
        episode_length = 0
        speeds: list[float] = []
        lateral_total = 0.0
        heading_total = 0.0
        wall_frames = 0
        observed_frames = 0
        respawns = 0
        boost_charge_total = 0.0
        boost_spent = 0
        boost_gained = 0
        boost_frames = 0
        jet_boost_remaining_total = 0.0
        jet_boost_frames = 0
        jet_boost_pickups = 0
        skid_frames = 0
        info: dict[str, Any] = {}
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            episode_length += 1
            speeds.append(float(info["ram_player_speed"]))
            frames = int(info.get("ram_action_repeat_frames", 1))
            observed_frames += frames
            lateral_total += (
                float(info["ram_decision_mean_abs_lateral_offset"]) * frames
            )
            heading_total += float(info["ram_decision_mean_heading_alignment"]) * frames
            wall_frames += int(info["ram_decision_wall_frames"])
            respawns += int(info["ram_decision_respawns"])
            boost_charge_total += (
                float(info.get("ram_decision_mean_boost_charge", 0.0)) * frames
            )
            boost_spent += int(info.get("ram_decision_boost_spent", 0))
            boost_gained += int(info.get("ram_decision_boost_gained", 0))
            boost_frames += int(info.get("ram_decision_boost_frames", 0))
            jet_boost_remaining_total += (
                float(info.get("ram_decision_mean_jet_boost_remaining", 0.0)) * frames
            )
            jet_boost_frames += int(info.get("ram_decision_jet_boost_frames", 0))
            jet_boost_pickups += int(info.get("ram_decision_jet_boost_pickups", 0))
            skid_frames += int(info.get("ram_decision_skid_frames", 0))

        finished = bool(info.get("ram_player_finished", False))
        rank = int(info.get("ram_player_rank", 4))
        won = finished and rank == 1
        completion = float(info.get("ram_player_completion", 0.0))
        finish_frame = info.get("ram_player_finish_frame")
        normalized_time = (
            min(1.0, float(finish_frame) / max(1, max_episode_frames))
            if finished and finish_frame is not None
            else 1.0
        )
        # Finish rate dominates partial progress; among finishers, time wins.
        score = 11.0 - normalized_time if finished else completion
        competitive_score = (
            21.0 - normalized_time
            if won
            else (11.0 - normalized_time if finished else completion)
        )

        rewards.append(episode_reward)
        lengths.append(episode_length)
        completions.append(completion)
        finishes.append(float(finished))
        wins.append(float(won))
        if finished and finish_frame is not None:
            finish_frames.append(float(finish_frame))
        mean_speeds.append(fmean(speeds) if speeds else 0.0)
        ranks.append(float(rank))
        lateral_offsets.append(lateral_total / max(1, observed_frames))
        heading_alignments.append(heading_total / max(1, observed_frames))
        wall_contact_rates.append(wall_frames / max(1, observed_frames))
        respawn_counts.append(float(respawns))
        boost_charges.append(boost_charge_total / max(1, observed_frames))
        boost_spent_totals.append(float(boost_spent))
        boost_gained_totals.append(float(boost_gained))
        boost_active_rates.append(boost_frames / max(1, observed_frames))
        jet_boost_remaining_values.append(
            jet_boost_remaining_total / max(1, observed_frames)
        )
        jet_boost_active_rates.append(jet_boost_frames / max(1, observed_frames))
        jet_boost_pickup_counts.append(float(jet_boost_pickups))
        skid_active_rates.append(skid_frames / max(1, observed_frames))
        episode_hairpin_completed = int(info.get("ram_player_hairpin_completed", 0))
        hairpin_completions.append(float(episode_hairpin_completed))
        hairpin_entries += int(info.get("ram_player_hairpin_entries", 0))
        hairpin_completed += episode_hairpin_completed
        hairpin_jet_boost_entries += int(
            info.get("ram_player_hairpin_jet_boost_entries", 0)
        )
        hairpin_frames += int(info.get("ram_player_hairpin_frames", 0))
        hairpin_completed_frames += int(
            info.get("ram_player_hairpin_completed_frames", 0)
        )
        hairpin_entry_speed_total += float(
            info.get("ram_player_hairpin_entry_speed_total", 0)
        )
        hairpin_minimum_speed_total += float(
            info.get("ram_player_hairpin_minimum_speed_total", 0)
        )
        hairpin_exit_speed_total += float(
            info.get("ram_player_hairpin_exit_speed_total", 0)
        )
        hairpin_wall_frames += int(info.get("ram_player_hairpin_wall_frames", 0))
        hairpin_skid_frames += int(info.get("ram_player_hairpin_skid_frames", 0))
        hairpin_condition_frames["all"] += float(
            info.get("ram_player_hairpin_frames", 0)
        )
        hairpin_condition_frames["jet"] += float(
            info.get("ram_player_hairpin_condition_jet_boost_frames", 0)
        )
        hairpin_condition_frames["nojet"] += float(
            info.get("ram_player_hairpin_condition_no_jet_boost_frames", 0)
        )
        for group in HAIRPIN_ACTION_GROUPS:
            prefix = HAIRPIN_INFO_PREFIXES[group]
            for name in RAM_ACTION_METRIC_NAMES:
                hairpin_action_totals[group][name] += float(
                    info.get(
                        f"ram_player_hairpin_{prefix}action_{name}_frames",
                        0,
                    )
                )
        for lap in lap_splits:
            split_frames = int(info.get(f"ram_player_lap_{lap}_frames", 0))
            if split_frames > 0:
                lap_splits[lap].append(split_frames / 60.0)
        _accumulate_sector_info(sector_totals, info)
        _accumulate_lap_sector_info(lap_sector_frame_totals, lap_sector_samples, info)
        scores.append(score)
        competitive_scores.append(competitive_score)

    return RAMEvaluation(
        mean_reward=fmean(rewards),
        mean_length=fmean(lengths),
        mean_completion=fmean(completions),
        finish_rate=fmean(finishes),
        win_rate=fmean(wins),
        mean_finish_frames=fmean(finish_frames) if finish_frames else 0.0,
        mean_speed=fmean(mean_speeds),
        mean_rank=fmean(ranks),
        mean_abs_lateral_offset=fmean(lateral_offsets),
        mean_heading_alignment=fmean(heading_alignments),
        wall_contact_rate=fmean(wall_contact_rates),
        mean_respawns=fmean(respawn_counts),
        mean_boost_charge=fmean(boost_charges),
        mean_boost_spent=fmean(boost_spent_totals),
        mean_boost_gained=fmean(boost_gained_totals),
        boost_active_rate=fmean(boost_active_rates),
        mean_jet_boost_remaining=fmean(jet_boost_remaining_values),
        jet_boost_active_rate=fmean(jet_boost_active_rates),
        mean_jet_boost_pickups=fmean(jet_boost_pickup_counts),
        skid_active_rate=fmean(skid_active_rates),
        mean_hairpin_completions=fmean(hairpin_completions),
        mean_hairpin_seconds=(
            hairpin_completed_frames / max(1, hairpin_completed) / 60.0
        ),
        mean_hairpin_entry_speed=(hairpin_entry_speed_total / max(1, hairpin_entries)),
        mean_hairpin_minimum_speed=(
            hairpin_minimum_speed_total / max(1, hairpin_completed)
        ),
        mean_hairpin_exit_speed=(hairpin_exit_speed_total / max(1, hairpin_completed)),
        hairpin_wall_contact_rate=hairpin_wall_frames / max(1, hairpin_frames),
        hairpin_skid_active_rate=hairpin_skid_frames / max(1, hairpin_frames),
        hairpin_jet_boost_entry_rate=(
            hairpin_jet_boost_entries / max(1, hairpin_entries)
        ),
        mean_lap_1_seconds=fmean(lap_splits[1]) if lap_splits[1] else 0.0,
        mean_lap_2_seconds=fmean(lap_splits[2]) if lap_splits[2] else 0.0,
        mean_lap_3_seconds=fmean(lap_splits[3]) if lap_splits[3] else 0.0,
        selection_score=fmean(scores),
        competitive_selection_score=fmean(competitive_scores),
        action_metrics={
            **{
                f"race_{name}_rate": (
                    sum(sector_totals[f"action_{name}_frames"])
                    / max(1.0, sum(sector_totals["frames"]))
                )
                for name in RAM_ACTION_METRIC_NAMES
            },
            **{
                f"hairpin_{group}_{name}_rate": (
                    hairpin_action_totals[group][name]
                    / max(1.0, hairpin_condition_frames[group])
                )
                for group in HAIRPIN_ACTION_GROUPS
                for name in RAM_ACTION_METRIC_NAMES
            },
        },
        sector_metrics={
            **_sector_metrics(sector_totals, episodes),
            **_lap_sector_metrics(lap_sector_frame_totals, lap_sector_samples),
        },
    )


class RAMEvalCallback(BaseCallback):
    """Log evaluations to TensorBoard/CSV and save the fastest Player 1 model."""

    def __init__(
        self,
        eval_env: gym.Env | None,
        *,
        eval_env_factory: Callable[[], gym.Env] | None = None,
        eval_every_timesteps: int,
        episodes: int,
        max_episode_frames: int,
        output_dir: Path,
        metric_prefix: str = "eval",
        selection_metric: str = "selection_score",
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        if (eval_env is None) == (eval_env_factory is None):
            raise ValueError("provide eval_env or eval_env_factory, not both")
        self.eval_env = eval_env
        self.eval_env_factory = eval_env_factory
        self.eval_every_timesteps = int(eval_every_timesteps)
        self.episodes = int(episodes)
        self.max_episode_frames = int(max_episode_frames)
        self.output_dir = output_dir
        self.metric_prefix = metric_prefix
        self.selection_metric = selection_metric
        self.best_score = float("-inf")
        self.next_evaluation = self.eval_every_timesteps
        self.last_evaluation = -1
        self.csv_path = output_dir / "evaluations.csv"

    def _init_callback(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as handle:
                csv.writer(handle).writerow(("timesteps", *evaluation_metric_names()))

    def _record_evaluation(self) -> None:
        eval_env = self.eval_env_factory() if self.eval_env_factory else self.eval_env
        assert eval_env is not None
        try:
            result = evaluate_ram_policy(
                self.model,
                eval_env,
                self.episodes,
                max_episode_frames=self.max_episode_frames,
            )
        finally:
            if self.eval_env_factory is not None:
                eval_env.close()
        values = evaluation_values(result)
        for name, value in values.items():
            self.logger.record(f"{self.metric_prefix}/{name}", value)
        self.logger.dump(self.num_timesteps)
        with self.csv_path.open("a", newline="") as handle:
            csv.writer(handle).writerow((self.num_timesteps, *values.values()))
        self.last_evaluation = self.num_timesteps

        if self.verbose:
            print(
                "RAM Player 1 eval "
                f"[{self.metric_prefix}] steps={self.num_timesteps} "
                f"score={getattr(result, self.selection_metric):.4f} "
                f"finish={result.finish_rate:.0%} "
                f"win={result.win_rate:.0%} "
                f"completion={result.mean_completion:.1%} "
                f"finish_frames={result.mean_finish_frames:.1f} "
                f"rank={result.mean_rank:.2f} "
                f"lateral={result.mean_abs_lateral_offset:.3f} "
                f"wall={result.wall_contact_rate:.1%} "
                f"boost_spent={result.mean_boost_spent:.0f} "
                f"jet_boost_pickups={result.mean_jet_boost_pickups:.1f} "
                f"skid={result.skid_active_rate:.1%} "
                f"hairpin={result.mean_hairpin_seconds:.2f}s "
                f"hairpin_boost={result.hairpin_jet_boost_entry_rate:.0%} "
                f"laps={result.mean_lap_1_seconds:.2f}/"
                f"{result.mean_lap_2_seconds:.2f}/"
                f"{result.mean_lap_3_seconds:.2f}s"
            )
        score = float(getattr(result, self.selection_metric))
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.output_dir / "best_model")
            if self.verbose:
                print(f"Saved new best model to {self.output_dir / 'best_model.zip'}")

    def _on_training_start(self) -> None:
        if self.eval_every_timesteps > 0:
            self.next_evaluation = (
                self.num_timesteps // self.eval_every_timesteps + 1
            ) * self.eval_every_timesteps
        self._record_evaluation()

    def _on_step(self) -> bool:
        if self.eval_every_timesteps <= 0:
            return True
        while self.num_timesteps >= self.next_evaluation:
            self._record_evaluation()
            self.next_evaluation += self.eval_every_timesteps
        return True

    def _on_training_end(self) -> None:
        if self.num_timesteps and self.num_timesteps != self.last_evaluation:
            self._record_evaluation()
