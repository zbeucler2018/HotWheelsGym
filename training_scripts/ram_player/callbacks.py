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
    scores: list[float] = []

    for _ in range(episodes):
        observation, _ = env.reset()
        terminated = truncated = False
        episode_reward = 0.0
        episode_length = 0
        speeds: list[float] = []
        info: dict[str, Any] = {}
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            episode_length += 1
            speeds.append(float(info["ram_player_speed"]))

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
        scores.append(score)

    return RAMEvaluation(
        mean_reward=fmean(rewards),
        mean_length=fmean(lengths),
        mean_completion=fmean(completions),
        finish_rate=fmean(finishes),
        mean_finish_frames=fmean(finish_frames) if finish_frames else 0.0,
        mean_speed=fmean(mean_speeds),
        mean_rank=fmean(ranks),
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
                f"rank={result.mean_rank:.2f}"
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
