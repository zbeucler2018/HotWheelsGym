"""Evaluation callback dedicated to the Player 1 RAM model."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback


@dataclass(frozen=True)
class RAMEvaluation:
    mean_reward: float
    mean_length: float
    mean_completion: float
    finish_rate: float
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
    mean_lap_1_seconds: float
    mean_lap_2_seconds: float
    mean_lap_3_seconds: float
    selection_score: float


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
    lap_splits: dict[int, list[float]] = {lap: [] for lap in (1, 2, 3)}
    scores: list[float] = []

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

        finished = bool(info.get("ram_player_finished", False))
        completion = float(info.get("ram_player_completion", 0.0))
        finish_frame = info.get("ram_player_finish_frame")
        normalized_time = (
            min(1.0, float(finish_frame) / max(1, max_episode_frames))
            if finished and finish_frame is not None
            else 1.0
        )
        # Finish rate dominates partial progress; among finishers, time wins.
        score = 11.0 - normalized_time if finished else completion

        rewards.append(episode_reward)
        lengths.append(episode_length)
        completions.append(completion)
        finishes.append(float(finished))
        if finished and finish_frame is not None:
            finish_frames.append(float(finish_frame))
        mean_speeds.append(fmean(speeds) if speeds else 0.0)
        ranks.append(float(info.get("ram_player_rank", 4)))
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
        for lap in lap_splits:
            split_frames = int(info.get(f"ram_player_lap_{lap}_frames", 0))
            if split_frames > 0:
                lap_splits[lap].append(split_frames / 60.0)
        scores.append(score)

    return RAMEvaluation(
        mean_reward=fmean(rewards),
        mean_length=fmean(lengths),
        mean_completion=fmean(completions),
        finish_rate=fmean(finishes),
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
        mean_lap_1_seconds=fmean(lap_splits[1]) if lap_splits[1] else 0.0,
        mean_lap_2_seconds=fmean(lap_splits[2]) if lap_splits[2] else 0.0,
        mean_lap_3_seconds=fmean(lap_splits[3]) if lap_splits[3] else 0.0,
        selection_score=fmean(scores),
    )


class RAMEvalCallback(BaseCallback):
    """Log evaluations to TensorBoard/CSV and save the fastest Player 1 model."""

    def __init__(
        self,
        eval_env: gym.Env,
        *,
        eval_every_timesteps: int,
        episodes: int,
        max_episode_frames: int,
        output_dir: Path,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_every_timesteps = int(eval_every_timesteps)
        self.episodes = int(episodes)
        self.max_episode_frames = int(max_episode_frames)
        self.output_dir = output_dir
        self.best_score = float("-inf")
        self.next_evaluation = self.eval_every_timesteps
        self.last_evaluation = -1
        self.csv_path = output_dir / "evaluations.csv"

    def _init_callback(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as handle:
                csv.writer(handle).writerow(
                    (
                        "timesteps",
                        "mean_reward",
                        "mean_length",
                        "mean_completion",
                        "finish_rate",
                        "mean_finish_frames",
                        "mean_speed",
                        "mean_rank",
                        "mean_abs_lateral_offset",
                        "mean_heading_alignment",
                        "wall_contact_rate",
                        "mean_respawns",
                        "mean_boost_charge",
                        "mean_boost_spent",
                        "mean_boost_gained",
                        "boost_active_rate",
                        "mean_jet_boost_remaining",
                        "jet_boost_active_rate",
                        "mean_jet_boost_pickups",
                        "mean_lap_1_seconds",
                        "mean_lap_2_seconds",
                        "mean_lap_3_seconds",
                        "selection_score",
                    )
                )

    def _record_evaluation(self) -> None:
        result = evaluate_ram_policy(
            self.model,
            self.eval_env,
            self.episodes,
            max_episode_frames=self.max_episode_frames,
        )
        values = {
            "mean_reward": result.mean_reward,
            "mean_length": result.mean_length,
            "mean_completion": result.mean_completion,
            "finish_rate": result.finish_rate,
            "mean_finish_frames": result.mean_finish_frames,
            "mean_speed": result.mean_speed,
            "mean_rank": result.mean_rank,
            "mean_abs_lateral_offset": result.mean_abs_lateral_offset,
            "mean_heading_alignment": result.mean_heading_alignment,
            "wall_contact_rate": result.wall_contact_rate,
            "mean_respawns": result.mean_respawns,
            "mean_boost_charge": result.mean_boost_charge,
            "mean_boost_spent": result.mean_boost_spent,
            "mean_boost_gained": result.mean_boost_gained,
            "boost_active_rate": result.boost_active_rate,
            "mean_jet_boost_remaining": result.mean_jet_boost_remaining,
            "jet_boost_active_rate": result.jet_boost_active_rate,
            "mean_jet_boost_pickups": result.mean_jet_boost_pickups,
            "mean_lap_1_seconds": result.mean_lap_1_seconds,
            "mean_lap_2_seconds": result.mean_lap_2_seconds,
            "mean_lap_3_seconds": result.mean_lap_3_seconds,
            "selection_score": result.selection_score,
        }
        for name, value in values.items():
            self.logger.record(f"eval/{name}", value)
        self.logger.dump(self.num_timesteps)
        with self.csv_path.open("a", newline="") as handle:
            csv.writer(handle).writerow((self.num_timesteps, *values.values()))
        self.last_evaluation = self.num_timesteps

        if self.verbose:
            print(
                "RAM Player 1 eval "
                f"steps={self.num_timesteps} score={result.selection_score:.4f} "
                f"finish={result.finish_rate:.0%} "
                f"completion={result.mean_completion:.1%} "
                f"finish_frames={result.mean_finish_frames:.1f} "
                f"rank={result.mean_rank:.2f} "
                f"lateral={result.mean_abs_lateral_offset:.3f} "
                f"wall={result.wall_contact_rate:.1%} "
                f"boost_spent={result.mean_boost_spent:.0f} "
                f"jet_boost_pickups={result.mean_jet_boost_pickups:.1f} "
                f"laps={result.mean_lap_1_seconds:.2f}/"
                f"{result.mean_lap_2_seconds:.2f}/"
                f"{result.mean_lap_3_seconds:.2f}s"
            )
        if result.selection_score > self.best_score:
            self.best_score = result.selection_score
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
