"""Gymnasium wrappers for Player 1 RAM training and RAM-model opponents."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from .enums import RaceMode, Tracks
from .npc_control import (
    DEFAULT_MAX_TARGET_SPEED,
    DEFAULT_MAX_TURN,
    NPCCommand,
    RaceMemory,
    RacerState,
    controlled_vehicle_indices_from_rom,
    progress_delta,
)
from .ram_opponent_control import (
    DINO_BONEYARD_PROGRESS_COUNT,
    DINO_RAM_OBSERVATION_SIZE,
    RAM_ACTION_SIZE,
    RaceProgressTracker,
    build_dino_ram_observation,
    player_buttons_from_action,
    race_reward,
    read_racer_states,
    write_cpu_command,
)


class PredictPolicy(Protocol):
    def predict(
        self, observation: np.ndarray, *, deterministic: bool = True
    ) -> Any: ...


def _bare_env(env: gym.Env) -> Any:
    return env.unwrapped


def _validate_dino_multi(env: gym.Env) -> None:
    base = _bare_env(env)
    if base.mode != RaceMode.MULTI or base.track != Tracks.Dino_Boneyard:
        raise ValueError(
            "the RAM prototype is intentionally limited to Dino Boneyard multi mode"
        )


def _validate_slots(env: gym.Env, race: RaceMemory, slots: tuple[int, ...]) -> None:
    if any(slot not in (1, 2, 3) for slot in slots):
        raise ValueError("RAM opponent slots must be 1, 2, or 3")
    controlled_vehicles = controlled_vehicle_indices_from_rom(
        Path(_bare_env(env).rom_path)
    )
    if not controlled_vehicles:
        raise RuntimeError("the active ROM does not have the selected-NPC control patch")
    missing = [
        race.layout.cpu(slot).vehicle_index
        for slot in slots
        if race.layout.cpu(slot).vehicle_index not in controlled_vehicles
    ]
    if missing:
        raise RuntimeError(
            "the active ROM does not expose CPU vehicle index/indices "
            + ", ".join(str(index) for index in sorted(set(missing)))
        )


def _state_info(
    prefix: str,
    slot: int,
    states: Mapping[int, RacerState],
    progress: RaceProgressTracker,
    total_laps: int,
    *,
    finished: bool,
    finish_frame: int | None,
) -> dict[str, Any]:
    state = states[slot]
    return {
        f"{prefix}slot": slot,
        f"{prefix}vehicle_index": state.vehicle_index,
        f"{prefix}progress": state.progress,
        f"{prefix}lap": min(progress.laps[slot], total_laps),
        f"{prefix}rank": progress.rank(slot, states),
        f"{prefix}speed": state.speed,
        f"{prefix}heading": state.current_heading,
        f"{prefix}completion": progress.completion(slot, state, total_laps),
        f"{prefix}finished": finished,
        f"{prefix}finish_frame": finish_frame,
    }


class DinoRAMPlayerEnv(gym.Wrapper):
    """Train Player 1 from racer-centric RAM instead of framebuffer pixels."""

    def __init__(self, env: gym.Env) -> None:
        _validate_dino_multi(env)
        super().__init__(env)
        self._race: RaceMemory | None = None
        self._progress: RaceProgressTracker | None = None
        self._states: dict[int, RacerState] | None = None
        self._previous_states: dict[int, RacerState] | None = None
        self._previous_action = 0
        self._raw_frame = 0
        self._finished = False
        self._finish_frame: int | None = None
        self.action_space = gym.spaces.Discrete(RAM_ACTION_SIZE)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(DINO_RAM_OBSERVATION_SIZE,),
            dtype=np.float32,
        )

    @property
    def progress_tracker(self) -> RaceProgressTracker:
        if self._progress is None:
            raise RuntimeError("call reset() before accessing race progress")
        return self._progress

    def _observation(self) -> np.ndarray:
        if self._states is None or self._previous_states is None:
            raise RuntimeError("call reset() before requesting an observation")
        return np.asarray(
            build_dino_ram_observation(
                self._states,
                self._previous_states,
                self.progress_tracker,
                0,
                self._previous_action,
                total_laps=int(_bare_env(self.env).total_laps),
            ),
            dtype=np.float32,
        )

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        _, info = self.env.reset(**kwargs)
        self._race = RaceMemory(_bare_env(self.env).data.memory)
        self._states = read_racer_states(self._race)
        self._previous_states = dict(self._states)
        self._progress = RaceProgressTracker.from_states(
            self._states, DINO_BONEYARD_PROGRESS_COUNT
        )
        self._previous_action = 0
        self._raw_frame = 0
        self._finished = False
        self._finish_frame = None
        info.update(
            _state_info(
                "ram_player_",
                0,
                self._states,
                self.progress_tracker,
                int(_bare_env(self.env).total_laps),
                finished=False,
                finish_frame=None,
            )
        )
        return self._observation(), info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._race is None or self._states is None:
            raise RuntimeError("call reset() before step()")
        flat_action = np.asarray(action).reshape(-1)
        if flat_action.size != 1:
            raise ValueError(
                f"RAM Player 1 policy returned {flat_action.size} action values"
            )
        action_index = int(flat_action[0])
        previous_states = self._states
        previous = previous_states[0]
        previous_rank = self.progress_tracker.rank(0, previous_states)
        previous_lap = self.progress_tracker.laps[0]
        native_action = player_buttons_from_action(
            action_index, tuple(_bare_env(self.env).buttons)
        )
        _, _, terminated, truncated, info = self.env.step(native_action)
        self._raw_frame += 1

        current_states = read_racer_states(self._race)
        self.progress_tracker.update(current_states)
        current = current_states[0]
        current_rank = self.progress_tracker.rank(0, current_states)
        current_lap = self.progress_tracker.laps[0]
        total_laps = int(_bare_env(self.env).total_laps)
        game_reports_finish = bool(
            terminated and int(info.get("lap", 0)) >= total_laps
        )
        finished_now = not self._finished and (
            current_lap > total_laps or game_reports_finish
        )
        if finished_now:
            self._finished = True
            self._finish_frame = self._raw_frame

        advance = progress_delta(
            previous.progress,
            current.progress,
            DINO_BONEYARD_PROGRESS_COUNT,
        )
        reward = race_reward(
            advance,
            current.speed,
            DEFAULT_MAX_TARGET_SPEED,
            previous_rank=previous_rank,
            current_rank=current_rank,
            completed_laps=max(0, current_lap - previous_lap),
            finished_now=finished_now,
        )
        self._previous_states = previous_states
        self._states = current_states
        self._previous_action = action_index
        info.update(
            _state_info(
                "ram_player_",
                0,
                current_states,
                self.progress_tracker,
                total_laps,
                finished=self._finished,
                finish_frame=self._finish_frame,
            )
        )
        info["ram_player_progress_delta"] = advance
        info["ram_player_frame_reward"] = reward
        info["ram_player_raw_frame"] = self._raw_frame
        return self._observation(), reward, terminated, truncated, info


class RAMActionRepeat(gym.Wrapper):
    """Repeat one shared RAM-policy action for several emulator frames."""

    def __init__(self, env: gym.Env, repeat: int = 4) -> None:
        if repeat < 1:
            raise ValueError("action repeat must be at least one frame")
        super().__init__(env)
        self.repeat = repeat

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        observation: Any = None
        info: dict[str, Any] = {}
        terminated = truncated = False
        frames = 0
        for _ in range(self.repeat):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            frames += 1
            if terminated or truncated:
                break
        info["ram_action_repeat_frames"] = frames
        info["ram_decision_reward"] = total_reward
        return observation, total_reward, terminated, truncated, info


class DinoRAMModelOpponentEnv(gym.Wrapper):
    """Put frozen Player 1 RAM checkpoints into patched native CPU slots."""

    def __init__(
        self,
        env: gym.Env,
        opponents: Mapping[int, PredictPolicy],
        *,
        deterministic: bool = True,
        action_repeat: int = 4,
        max_turn: int = DEFAULT_MAX_TURN,
        max_target_speed: int = DEFAULT_MAX_TARGET_SPEED,
    ) -> None:
        _validate_dino_multi(env)
        slots = tuple(sorted(opponents))
        if not slots:
            raise ValueError("provide at least one RAM opponent policy")
        if any(slot not in (1, 2, 3) for slot in slots):
            raise ValueError("RAM opponent slots must be 1, 2, or 3")
        if action_repeat < 1:
            raise ValueError("action_repeat must be at least one")
        super().__init__(env)
        self.opponents = dict(opponents)
        self.deterministic = deterministic
        self.action_repeat = action_repeat
        self.max_turn = max_turn
        self.max_target_speed = max_target_speed
        self._race: RaceMemory | None = None
        self._progress: RaceProgressTracker | None = None
        self._states: dict[int, RacerState] | None = None
        self._previous_states: dict[int, RacerState] | None = None
        self._actions: dict[int, int] = {}
        self._frames_until_action: dict[int, int] = {}
        self._finished: dict[int, bool] = {}
        self._finish_frames: dict[int, int | None] = {}
        self._raw_frame = 0

    def _observation(self, slot: int) -> np.ndarray:
        if self._states is None or self._previous_states is None or self._progress is None:
            raise RuntimeError("call reset() before requesting opponent observations")
        return np.asarray(
            build_dino_ram_observation(
                self._states,
                self._previous_states,
                self._progress,
                slot,
                self._actions[slot],
                total_laps=int(_bare_env(self.env).total_laps),
                max_target_speed=self.max_target_speed,
            ),
            dtype=np.float32,
        )

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._race = RaceMemory(_bare_env(self.env).data.memory)
        slots = tuple(sorted(self.opponents))
        _validate_slots(self.env, self._race, slots)
        self._states = read_racer_states(self._race)
        self._previous_states = dict(self._states)
        self._progress = RaceProgressTracker.from_states(
            self._states, DINO_BONEYARD_PROGRESS_COUNT
        )
        self._actions = {slot: 0 for slot in slots}
        self._frames_until_action = {slot: 0 for slot in slots}
        self._finished = {slot: False for slot in slots}
        self._finish_frames = {slot: None for slot in slots}
        self._raw_frame = 0
        total_laps = int(_bare_env(self.env).total_laps)
        for slot in slots:
            info.update(
                _state_info(
                    f"ram_npc_{slot}_",
                    slot,
                    self._states,
                    self._progress,
                    total_laps,
                    finished=False,
                    finish_frame=None,
                )
            )
        return observation, info

    def step(self, player_action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self._race is None or self._states is None:
            raise RuntimeError("call reset() before step()")

        commands: dict[int, NPCCommand] = {}
        for slot, policy in self.opponents.items():
            if self._frames_until_action[slot] == 0:
                prediction = policy.predict(
                    self._observation(slot), deterministic=self.deterministic
                )
                raw_action = prediction[0] if isinstance(prediction, tuple) else prediction
                flat_action = np.asarray(raw_action).reshape(-1)
                if flat_action.size != 1:
                    raise ValueError(
                        f"RAM opponent in slot {slot} returned "
                        f"{flat_action.size} action values"
                    )
                self._actions[slot] = int(flat_action[0])
                self._frames_until_action[slot] = self.action_repeat
            commands[slot] = write_cpu_command(
                self._race,
                slot,
                self._actions[slot],
                max_turn=self.max_turn,
                max_target_speed=self.max_target_speed,
            )
            self._frames_until_action[slot] -= 1

        observation, reward, terminated, truncated, info = self.env.step(player_action)
        self._raw_frame += 1
        previous_states = self._states
        current_states = read_racer_states(self._race)
        assert self._progress is not None
        self._progress.update(current_states)
        self._previous_states = previous_states
        self._states = current_states
        total_laps = int(_bare_env(self.env).total_laps)

        for slot, command in commands.items():
            if not self._finished[slot] and self._progress.laps[slot] > total_laps:
                self._finished[slot] = True
                self._finish_frames[slot] = self._raw_frame
            prefix = f"ram_npc_{slot}_"
            info.update(
                _state_info(
                    prefix,
                    slot,
                    current_states,
                    self._progress,
                    total_laps,
                    finished=self._finished[slot],
                    finish_frame=self._finish_frames[slot],
                )
            )
            info[f"{prefix}action"] = self._actions[slot]
            info[f"{prefix}command_heading"] = command.desired_heading
            info[f"{prefix}command_speed"] = command.target_speed
        return observation, reward, terminated, truncated, info
