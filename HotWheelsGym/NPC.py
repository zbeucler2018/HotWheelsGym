"""Gymnasium wrappers for training and racing native CPU-slot policies."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from .enums import RaceMode
from .npc_control import (
    DEFAULT_MAX_TARGET_SPEED,
    DEFAULT_MAX_TURN,
    OBSERVATION_SIZE,
    TRACK_PROGRESS_COUNTS,
    NPCCommand,
    RaceMemory,
    RacerState,
    controlled_vehicle_indices_from_rom,
    progress_delta,
)


class PredictPolicy(Protocol):
    """The Stable-Baselines3 ``predict`` shape used by model opponents."""

    def predict(
        self, observation: np.ndarray, *, deterministic: bool = True
    ) -> Any: ...


def _bare_env(env: gym.Env) -> Any:
    return env.unwrapped


def _track_progress_count(env: gym.Env) -> int:
    track = _bare_env(env).track.value
    try:
        return TRACK_PROGRESS_COUNTS[track]
    except KeyError as error:
        raise ValueError(f"no progress count is known for track {track!r}") from error


def _ensure_multi_mode(env: gym.Env) -> None:
    if _bare_env(env).mode != RaceMode.MULTI:
        raise ValueError("native NPC control requires the game's multi race mode")


def _validate_configuration(
    slots: tuple[int, ...], max_turn: int, max_target_speed: int
) -> None:
    if any(slot not in (1, 2, 3) for slot in slots):
        raise ValueError("native CPU racer slots must be 1, 2, or 3")
    if not 0 <= max_turn <= 0x800:
        raise ValueError("max_turn must be between 0 and 0x800")
    if not 0 < max_target_speed <= 0xFFFFFFFF:
        raise ValueError("max_target_speed must fit in an unsigned 32-bit value")


def _new_race_memory(env: gym.Env) -> RaceMemory:
    return RaceMemory(_bare_env(env).data.memory)


def _validate_controlled_slots(
    env: gym.Env, race: RaceMemory, slots: tuple[int, ...]
) -> None:
    rom_path = Path(_bare_env(env).rom_path)
    patched_vehicles = controlled_vehicle_indices_from_rom(rom_path)
    if not patched_vehicles:
        raise RuntimeError(
            "the active ROM has no selected-NPC control patch; generate it with "
            "`hwre patch-npc-control` and import that generated ROM"
        )
    missing = [
        race.layout.cpu(slot).vehicle_index
        for slot in slots
        if race.layout.cpu(slot).vehicle_index not in patched_vehicles
    ]
    if missing:
        raise RuntimeError(
            "the active ROM does not expose CPU vehicle index/indices "
            + ", ".join(str(index) for index in sorted(set(missing)))
        )


def _zero_action(space: gym.Space) -> Any:
    if getattr(space, "shape", None) is None:
        return 0
    return np.zeros(space.shape, dtype=space.dtype)


def _npc_info(prefix: str, state: RacerState, lap: int | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        f"{prefix}slot": state.slot,
        f"{prefix}vehicle_index": state.vehicle_index,
        f"{prefix}progress": state.progress,
        f"{prefix}rank": state.rank,
        f"{prefix}speed": state.speed,
        f"{prefix}heading": state.current_heading,
        f"{prefix}desired_heading": state.desired_heading,
        f"{prefix}stock_heading": state.stock_heading,
    }
    if lap is not None:
        info[f"{prefix}lap"] = lap
    return info


class HotWheelsNPCEnv(gym.Wrapper):
    """Train a policy as one of the game's native CPU racers.

    Actions are ``[steering, throttle]`` where steering is in ``[-1, 1]`` and
    throttle is in ``[0, 1]``. Observations are a compact 12-value telemetry
    vector, including the original AI's waypoint heading from the ROM hook.
    """

    def __init__(
        self,
        env: gym.Env,
        npc_slot: int = 1,
        *,
        max_turn: int = DEFAULT_MAX_TURN,
        max_target_speed: int = DEFAULT_MAX_TARGET_SPEED,
        player_action: Any | None = None,
    ) -> None:
        _ensure_multi_mode(env)
        _validate_configuration((npc_slot,), max_turn, max_target_speed)
        super().__init__(env)
        self.npc_slot = npc_slot
        self.max_turn = max_turn
        self.max_target_speed = max_target_speed
        self._progress_count = _track_progress_count(env)
        self._player_action = (
            _zero_action(env.action_space) if player_action is None else player_action
        )
        self._race: RaceMemory | None = None
        self._previous_state: RacerState | None = None
        self._npc_lap = 1
        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, 0.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )

    @property
    def race_memory(self) -> RaceMemory:
        if self._race is None:
            raise RuntimeError("call reset() before accessing NPC race memory")
        return self._race

    def _observation(self) -> np.ndarray:
        return np.asarray(
            self.race_memory.observation(
                self.npc_slot,
                self._progress_count,
                max_target_speed=self.max_target_speed,
            ),
            dtype=np.float32,
        )

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        _, info = self.env.reset(**kwargs)
        self._race = _new_race_memory(self.env)
        _validate_controlled_slots(self.env, self._race, (self.npc_slot,))
        self._race.prime_stock_heading(self.npc_slot)
        self._previous_state = self._race.state(self.npc_slot)
        self._npc_lap = 1
        info.update(_npc_info("npc_", self._previous_state, self._npc_lap))
        return self._observation(), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        previous = self._previous_state
        if previous is None:
            raise RuntimeError("call reset() before step()")
        command = self.race_memory.command(
            self.npc_slot,
            action,
            max_turn=self.max_turn,
            max_target_speed=self.max_target_speed,
        )
        _, _, base_terminated, truncated, info = self.env.step(self._player_action)
        current = self.race_memory.state(self.npc_slot)

        advance = progress_delta(
            previous.progress, current.progress, self._progress_count
        )
        if advance > 0 and current.progress < previous.progress:
            self._npc_lap += 1
        speed_bonus = 0.01 * min(1.0, current.speed / self.max_target_speed)
        reward = float(advance) + speed_bonus
        total_laps = int(_bare_env(self.env).total_laps)
        terminated = bool(base_terminated or self._npc_lap > total_laps)
        self._previous_state = current
        info.update(_npc_info("npc_", current, self._npc_lap))
        info["npc_progress_delta"] = advance
        info["npc_command_heading"] = command.desired_heading
        info["npc_command_speed"] = command.target_speed
        return self._observation(), reward, terminated, truncated, info


class ModelOpponentEnv(gym.Wrapper):
    """Run trained NPC policies while the wrapped environment controls Player 1.

    The wrapped action/observation spaces are unchanged, so an interactive
    keyboard/controller loop can keep sending normal Player 1 button actions.
    """

    def __init__(
        self,
        env: gym.Env,
        opponents: Mapping[int, PredictPolicy],
        *,
        deterministic: bool = True,
        max_turn: int = DEFAULT_MAX_TURN,
        max_target_speed: int = DEFAULT_MAX_TARGET_SPEED,
    ) -> None:
        _ensure_multi_mode(env)
        if not opponents:
            raise ValueError("provide at least one NPC slot and policy")
        _validate_configuration(tuple(opponents), max_turn, max_target_speed)
        super().__init__(env)
        self.opponents = dict(opponents)
        self.deterministic = deterministic
        self.max_turn = max_turn
        self.max_target_speed = max_target_speed
        self._progress_count = _track_progress_count(env)
        self._race: RaceMemory | None = None

    @property
    def race_memory(self) -> RaceMemory:
        if self._race is None:
            raise RuntimeError("call reset() before accessing opponent race memory")
        return self._race

    def _model_observation(self, slot: int) -> np.ndarray:
        return np.asarray(
            self.race_memory.observation(
                slot,
                self._progress_count,
                max_target_speed=self.max_target_speed,
            ),
            dtype=np.float32,
        )

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._race = _new_race_memory(self.env)
        slots = tuple(sorted(self.opponents))
        _validate_controlled_slots(self.env, self._race, slots)
        for slot in slots:
            self._race.prime_stock_heading(slot)
            info.update(_npc_info(f"npc_{slot}_", self._race.state(slot)))
        return observation, info

    def step(self, player_action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        commands: dict[int, NPCCommand] = {}
        for slot, policy in self.opponents.items():
            prediction = policy.predict(
                self._model_observation(slot), deterministic=self.deterministic
            )
            action = prediction[0] if isinstance(prediction, tuple) else prediction
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            commands[slot] = self.race_memory.command(
                slot,
                action,
                max_turn=self.max_turn,
                max_target_speed=self.max_target_speed,
            )

        observation, reward, terminated, truncated, info = self.env.step(player_action)
        for slot, command in commands.items():
            state = self.race_memory.state(slot)
            info.update(_npc_info(f"npc_{slot}_", state))
            info[f"npc_{slot}_command_heading"] = command.desired_heading
            info[f"npc_{slot}_command_speed"] = command.target_speed
        return observation, reward, terminated, truncated, info
