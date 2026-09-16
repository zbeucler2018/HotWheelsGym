"""Short-horizon Dino Boneyard pickup and hairpin training task."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from math import atan2, pi
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from HotWheelsGym.dino_boneyard_track import (
    DINO_BONEYARD_POWER_UPS,
    DINO_POWER_UP_JET_BOOST_TYPE,
)
from HotWheelsGym.ram_opponent_control import (
    DINO_BONEYARD_PROGRESS_COUNT,
    DINO_HAIRPIN_END_PROGRESS,
    RAM_OBSERVATION_NAMES,
    RAM_SPEED_SCALE,
)

from .common import make_ram_env

HAIRPIN_PICKUP = next(
    pickup
    for pickup in DINO_BONEYARD_POWER_UPS
    if pickup.kind == DINO_POWER_UP_JET_BOOST_TYPE
)

TRACK_PHASE_SIN_INDEX = RAM_OBSERVATION_NAMES.index("track_phase_sin")
TRACK_PHASE_COS_INDEX = RAM_OBSERVATION_NAMES.index("track_phase_cos")
PICKUP_DISTANCE_INDEX = RAM_OBSERVATION_NAMES.index("next_power_up_progress_distance")
PICKUP_AVAILABLE_INDEX = RAM_OBSERVATION_NAMES.index("next_power_up_available")
PICKUP_TYPE_INDEX = RAM_OBSERVATION_NAMES.index("next_power_up_is_jet_boost")
JET_BOOST_REMAINING_INDEX = RAM_OBSERVATION_NAMES.index("self_jet_boost_remaining")

PICKUP_MONITOR_INFO_KEYS = (
    "pickup_task_success",
    "pickup_task_acquired",
    "pickup_task_missed",
    "pickup_task_respawned",
    "pickup_task_timed_out",
    "pickup_task_frames",
    "pickup_task_frames_to_pickup",
    "pickup_task_exit_speed",
    "pickup_task_mean_abs_lateral_error",
    "pickup_task_score_gained",
)


@dataclass(frozen=True)
class PickupTaskConfig:
    """Reward and terminal conditions for the isolated pickup maneuver."""

    max_start_distance: float = 34.0
    miss_margin: float = 2.0
    progress_scale: float = 0.25
    frame_cost: float = 0.05
    lateral_error_penalty: float = 0.05
    wall_penalty: float = 0.25
    respawn_penalty: float = 100.0
    miss_penalty: float = 50.0
    timeout_penalty: float = 75.0
    pickup_bonus: float = 100.0
    exit_bonus: float = 25.0
    exit_speed_scale: float = 10.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object] | None) -> "PickupTaskConfig":
        if raw is None:
            return cls()
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                "unknown pickup-task fields: " + ", ".join(sorted(unknown))
            )
        config = cls(**{name: float(value) for name, value in raw.items()})
        for field in fields(config):
            if getattr(config, field.name) < 0:
                raise ValueError(f"pickup-task field {field.name} must be nonnegative")
        return config


class DinoPickupHairpinTask(gym.Wrapper):
    """Turn the first pickup and hairpin into a seconds-long episodic task.

    A miss ends immediately after the car passes the pickup. A collection keeps
    running through progress 61 so the expert must also leave the hairpin at
    useful speed. The full-race reward is deliberately discarded.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        config: PickupTaskConfig | Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(env)
        self.config = (
            config
            if isinstance(config, PickupTaskConfig)
            else PickupTaskConfig.from_mapping(config)
        )
        self._start_distance = 0.0
        self._exit_distance = 0.0
        self._progress = 0.0
        self._frames = 0
        self._frames_to_pickup = 0
        self._acquired = False
        self._success = False
        self._missed = False
        self._respawned = False
        self._timed_out = False
        self._lateral_error_total = 0.0
        self._lateral_error_frames = 0
        self._episode_reward = 0.0
        self._score_gained = 0

    def _task_info(self, info: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "pickup_task_success": self._success,
            "pickup_task_acquired": self._acquired,
            "pickup_task_missed": self._missed,
            "pickup_task_respawned": self._respawned,
            "pickup_task_timed_out": self._timed_out,
            "pickup_task_frames": self._frames,
            "pickup_task_frames_to_pickup": self._frames_to_pickup,
            "pickup_task_exit_speed": (
                int(info.get("ram_player_speed", 0)) if self._success else 0
            ),
            "pickup_task_mean_abs_lateral_error": (
                self._lateral_error_total / max(1, self._lateral_error_frames)
            ),
            "pickup_task_score_gained": self._score_gained,
            "pickup_task_progress": self._progress,
            "pickup_task_episode_reward": self._episode_reward,
        }

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        is_target = bool(info.get("ram_player_next_power_up_is_jet_boost", False))
        available = bool(info.get("ram_player_next_power_up_available", False))
        distance = float(info.get("ram_player_next_power_up_progress_distance", -1))
        if (
            not is_target
            or not available
            or not 0 < distance <= self.config.max_start_distance
        ):
            state = info.get("ram_initial_state", "configured initial state")
            raise ValueError(
                f"pickup-lab state {state} must begin 0.."
                f"{self.config.max_start_distance:g} progress units before the "
                "available first Jet Boost"
            )
        self._start_distance = distance
        self._exit_distance = distance + (
            DINO_HAIRPIN_END_PROGRESS - HAIRPIN_PICKUP.progress
        )
        self._progress = 0.0
        self._frames = 0
        self._frames_to_pickup = 0
        self._acquired = False
        self._success = False
        self._missed = False
        self._respawned = False
        self._timed_out = False
        self._lateral_error_total = 0.0
        self._lateral_error_frames = 0
        self._episode_reward = 0.0
        self._score_gained = 0
        info.update(self._task_info(info))
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, _, terminated, truncated, info = self.env.step(action)
        frames = int(info.get("ram_action_repeat_frames", 1))
        advance = float(info.get("ram_decision_continuous_progress_delta", 0.0))
        self._frames += frames
        self._progress += advance
        self._score_gained += int(info.get("ram_decision_score_delta", 0))

        reward = self.config.progress_scale * advance - self.config.frame_cost * frames
        wall_frames = int(info.get("ram_decision_wall_frames", 0))
        reward -= self.config.wall_penalty * wall_frames

        if not self._acquired and bool(
            info.get("ram_player_next_power_up_is_jet_boost", False)
        ):
            lateral_error = abs(
                float(info.get("ram_player_next_power_up_lateral_error", 0.0))
            )
            self._lateral_error_total += lateral_error * frames
            self._lateral_error_frames += frames
            reward -= self.config.lateral_error_penalty * lateral_error * frames

        pickups = int(info.get("ram_decision_jet_boost_pickups", 0))
        if pickups and not self._acquired:
            self._acquired = True
            self._frames_to_pickup = self._frames
            reward += self.config.pickup_bonus

        if int(info.get("ram_decision_respawns", 0)):
            self._respawned = True
            reward -= self.config.respawn_penalty
            terminated = True

        passed_pickup = self._progress >= self._start_distance + self.config.miss_margin
        target_changed = not bool(
            info.get("ram_player_next_power_up_is_jet_boost", False)
        )
        target_unavailable = not bool(
            info.get("ram_player_next_power_up_available", True)
        )
        if (
            not self._acquired
            and not self._respawned
            and (passed_pickup or target_changed or target_unavailable)
        ):
            self._missed = True
            reward -= self.config.miss_penalty
            terminated = True

        if self._acquired and self._progress >= self._exit_distance:
            self._success = True
            speed_ratio = min(
                1.0,
                max(0.0, float(info.get("ram_player_speed", 0)) / RAM_SPEED_SCALE),
            )
            reward += (
                self.config.exit_bonus + self.config.exit_speed_scale * speed_ratio
            )
            terminated = True

        if truncated and not self._success:
            self._timed_out = True
            reward -= self.config.timeout_penalty
        if terminated and not (self._success or self._missed or self._respawned):
            reward -= self.config.timeout_penalty

        self._episode_reward += reward
        info.update(self._task_info(info))
        return observation, float(reward), terminated, truncated, info


class PickupExpertCompositePolicy:
    """Gate a local pickup expert into an otherwise frozen full-race policy."""

    def __init__(self, base_policy: Any, pickup_expert: Any) -> None:
        self.base_policy = base_policy
        self.pickup_expert = pickup_expert
        self._expert_active = False
        self._expert_acquired = False

    @staticmethod
    def _progress(observation: Any) -> float:
        phase = atan2(
            float(observation[TRACK_PHASE_SIN_INDEX]),
            float(observation[TRACK_PHASE_COS_INDEX]),
        )
        return (phase % (2.0 * pi)) / (2.0 * pi) * DINO_BONEYARD_PROGRESS_COUNT

    def reset(self) -> None:
        self._expert_active = False
        self._expert_acquired = False
        for policy in (self.base_policy, self.pickup_expert):
            reset_policy = getattr(policy, "reset", None)
            if callable(reset_policy):
                reset_policy()

    def predict(
        self,
        observation: Any,
        *,
        deterministic: bool = True,
    ) -> Any:
        if getattr(observation, "ndim", 1) != 1:
            raise ValueError("pickup composite currently expects one observation")
        progress = self._progress(observation)
        if self._expert_active:
            self._expert_acquired = self._expert_acquired or (
                float(observation[JET_BOOST_REMAINING_INDEX]) > 0.0
            )
            target_lost = (
                float(observation[PICKUP_TYPE_INDEX]) <= 0.5
                or float(observation[PICKUP_AVAILABLE_INDEX]) <= 0.5
            )
            if progress >= DINO_HAIRPIN_END_PROGRESS or (
                target_lost and not self._expert_acquired
            ):
                self._expert_active = False
                self._expert_acquired = False
        if (
            not self._expert_active
            and progress < DINO_HAIRPIN_END_PROGRESS
            and float(observation[PICKUP_TYPE_INDEX]) > 0.5
            and float(observation[PICKUP_AVAILABLE_INDEX]) > 0.5
            and float(observation[PICKUP_DISTANCE_INDEX]) <= 24.0 / 64.0
        ):
            self._expert_active = True
            self._expert_acquired = False
        policy = self.pickup_expert if self._expert_active else self.base_policy
        return policy.predict(observation, deterministic=deterministic)


def make_pickup_lab_env(
    *,
    frame_skip: int,
    max_episode_steps: int,
    seed: int,
    state_path: str | None = None,
    state_paths: Sequence[str | None] | None = None,
    state_pool_selection: str = "random",
    task_config: Mapping[str, object] | PickupTaskConfig | None = None,
    monitor_path: str | None = None,
) -> gym.Env:
    """Construct the isolated task without changing the full-race trainer."""

    env = make_ram_env(
        frame_skip=frame_skip,
        max_episode_steps=max_episode_steps,
        seed=seed,
        state_path=state_path,
        state_paths=state_paths,
        state_pool_selection=state_pool_selection,
        reward_config={
            "mode": "time_trial",
            "progress_scale": 0.0,
            "frame_cost": 0.0,
            "wall_penalty": 0.0,
            "wrong_way_penalty": 0.0,
            "respawn_penalty": 0.0,
            "rank_gain": 0.0,
            "lap_bonus": 0.0,
            "finish_bonus": 0.0,
        },
    )
    env = DinoPickupHairpinTask(env, config=task_config)
    if monitor_path:
        Path(monitor_path).parent.mkdir(parents=True, exist_ok=True)
        env = Monitor(
            env,
            filename=monitor_path,
            info_keywords=PICKUP_MONITOR_INFO_KEYS,
        )
    return env


@dataclass(frozen=True)
class PickupEvaluation:
    mean_reward: float
    success_rate: float
    pickup_rate: float
    miss_rate: float
    respawn_rate: float
    timeout_rate: float
    mean_frames: float
    mean_success_frames: float
    mean_frames_to_pickup: float
    mean_exit_speed: float
    mean_abs_lateral_error: float
    mean_score_gained: float
    selection_score: float


def evaluate_pickup_policy(
    model: Any,
    env: gym.Env,
    episodes: int,
) -> PickupEvaluation:
    """Evaluate every short maneuver deterministically."""

    if episodes < 1:
        raise ValueError("pickup evaluation requires at least one episode")
    rewards: list[float] = []
    successes: list[float] = []
    pickups: list[float] = []
    misses: list[float] = []
    respawns: list[float] = []
    timeouts: list[float] = []
    frames: list[float] = []
    success_frames: list[float] = []
    pickup_frames: list[float] = []
    exit_speeds: list[float] = []
    lateral_errors: list[float] = []
    scores: list[float] = []

    for _ in range(episodes):
        reset_policy = getattr(model, "reset", None)
        if callable(reset_policy):
            reset_policy()
        observation, _ = env.reset()
        terminated = truncated = False
        episode_reward = 0.0
        info: dict[str, Any] = {}
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
        success = bool(info.get("pickup_task_success", False))
        acquired = bool(info.get("pickup_task_acquired", False))
        episode_frames = float(info.get("pickup_task_frames", 0))
        rewards.append(episode_reward)
        successes.append(float(success))
        pickups.append(float(acquired))
        misses.append(float(bool(info.get("pickup_task_missed", False))))
        respawns.append(float(bool(info.get("pickup_task_respawned", False))))
        timeouts.append(float(bool(info.get("pickup_task_timed_out", False))))
        frames.append(episode_frames)
        if success:
            success_frames.append(episode_frames)
            exit_speeds.append(float(info.get("pickup_task_exit_speed", 0)))
        if acquired:
            pickup_frames.append(float(info.get("pickup_task_frames_to_pickup", 0)))
        lateral_errors.append(
            float(info.get("pickup_task_mean_abs_lateral_error", 0.0))
        )
        scores.append(float(info.get("pickup_task_score_gained", 0)))

    success_rate = fmean(successes)
    pickup_rate = fmean(pickups)
    mean_success_frames = fmean(success_frames) if success_frames else fmean(frames)
    mean_exit_speed = fmean(exit_speeds) if exit_speeds else 0.0
    # Success dominates collection; collection dominates speed. This prevents a
    # fast miss from replacing a slower policy that actually solves the skill.
    selection_score = (
        success_rate * 1_000.0
        + pickup_rate * 100.0
        - mean_success_frames / 1_000.0
        + mean_exit_speed / 1_000_000.0
    )
    return PickupEvaluation(
        mean_reward=fmean(rewards),
        success_rate=success_rate,
        pickup_rate=pickup_rate,
        miss_rate=fmean(misses),
        respawn_rate=fmean(respawns),
        timeout_rate=fmean(timeouts),
        mean_frames=fmean(frames),
        mean_success_frames=mean_success_frames,
        mean_frames_to_pickup=(fmean(pickup_frames) if pickup_frames else 0.0),
        mean_exit_speed=mean_exit_speed,
        mean_abs_lateral_error=fmean(lateral_errors),
        mean_score_gained=fmean(scores),
        selection_score=selection_score,
    )


class PickupEvalCallback(BaseCallback):
    """Save the most reliable, then fastest, isolated pickup expert."""

    def __init__(
        self,
        *,
        eval_env_factory: Callable[[], gym.Env],
        eval_every_timesteps: int,
        episodes: int,
        output_dir: Path,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env_factory = eval_env_factory
        self.eval_every_timesteps = int(eval_every_timesteps)
        self.episodes = int(episodes)
        self.output_dir = output_dir
        self.csv_path = output_dir / "evaluations.csv"
        self.best_score = float("-inf")
        self.next_evaluation = self.eval_every_timesteps
        self.last_evaluation = -1

    def _init_callback(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as handle:
                csv.writer(handle).writerow(
                    ("timesteps", *(field.name for field in fields(PickupEvaluation)))
                )

    def _record_evaluation(self) -> None:
        env = self.eval_env_factory()
        try:
            result = evaluate_pickup_policy(self.model, env, self.episodes)
        finally:
            env.close()
        values = asdict(result)
        for name, value in values.items():
            self.logger.record(f"pickup_eval/{name}", value)
        self.logger.dump(self.num_timesteps)
        with self.csv_path.open("a", newline="") as handle:
            csv.writer(handle).writerow((self.num_timesteps, *values.values()))
        self.last_evaluation = self.num_timesteps
        if self.verbose:
            print(
                f"Pickup eval steps={self.num_timesteps} "
                f"success={result.success_rate:.0%} "
                f"pickup={result.pickup_rate:.0%} "
                f"miss={result.miss_rate:.0%} "
                f"timeout={result.timeout_rate:.0%} "
                f"frames={result.mean_success_frames:.1f} "
                f"exit_speed={result.mean_exit_speed:.0f} "
                f"score={result.selection_score:.3f}"
            )
        if result.selection_score > self.best_score:
            self.best_score = result.selection_score
            self.model.save(self.output_dir / "best_model")
            if self.verbose:
                print(
                    f"Saved new best pickup expert to "
                    f"{self.output_dir / 'best_model.zip'}"
                )

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
