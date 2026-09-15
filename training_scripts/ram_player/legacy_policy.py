"""Compatibility adapters for pre-factorized RAM policies."""

from __future__ import annotations

from typing import Any

import numpy as np

from HotWheelsGym.dino_boneyard_track import DINO_TRACK_OBSERVATION_SIZE
from HotWheelsGym.ram_opponent_control import (
    DINO_RAM_BASE_OBSERVATION_SIZE,
    DINO_RAM_OBSERVATION_SIZE,
    RAM_ACTION_COMPONENT_SIZES,
)

LEGACY_V5_OBSERVATION_SIZE = 60
LEGACY_V5_ACTION_SIZE = 7
LEGACY_V6_OBSERVATION_SIZE = DINO_RAM_BASE_OBSERVATION_SIZE
LEGACY_V5_ACTIONS = (
    (0, 0, 0),  # coast
    (1, 0, 0),  # accelerate
    (1, 1, 0),  # accelerate left
    (1, 2, 0),  # accelerate right
    (2, 0, 0),  # brake
    (3, 0, 0),  # accelerate + up
    (1, 0, 1),  # accelerate + L+R boost
)
V6_ACTION_HISTORY_START = 15 + DINO_TRACK_OBSERVATION_SIZE
V6_OTHER_RACERS_START = V6_ACTION_HISTORY_START + sum(RAM_ACTION_COMPONENT_SIZES)


def is_legacy_v5_policy(model: Any) -> bool:
    shape = tuple(getattr(model.observation_space, "shape", ()) or ())
    return shape == (LEGACY_V5_OBSERVATION_SIZE,) and getattr(
        model.action_space, "n", None
    ) == len(LEGACY_V5_ACTIONS)


def is_legacy_v6_policy(model: Any) -> bool:
    """Return whether a model uses the factorized pre-radar contract."""

    shape = tuple(getattr(model.observation_space, "shape", ()) or ())
    nvec = tuple(int(value) for value in getattr(model.action_space, "nvec", ()))
    return shape == (LEGACY_V6_OBSERVATION_SIZE,) and nvec == tuple(
        RAM_ACTION_COMPONENT_SIZES
    )


def legacy_v5_observation(observation: np.ndarray, previous_action: int) -> np.ndarray:
    """Translate the shared v6 observation into v5's seven-action history."""

    current = np.asarray(observation, dtype=np.float32).reshape(-1)
    if current.shape != (DINO_RAM_OBSERVATION_SIZE,):
        raise ValueError(
            f"active RAM observation expected {DINO_RAM_OBSERVATION_SIZE} values, "
            f"got {current.shape}"
        )
    action_history = np.zeros(LEGACY_V5_ACTION_SIZE, dtype=np.float32)
    action_history[previous_action] = 1.0
    legacy = np.concatenate(
        (
            current[:V6_ACTION_HISTORY_START],
            action_history,
            current[V6_OTHER_RACERS_START:DINO_RAM_BASE_OBSERVATION_SIZE],
        )
    ).astype(np.float32, copy=False)
    if legacy.shape != (LEGACY_V5_OBSERVATION_SIZE,):
        raise ValueError(
            f"legacy v5 adapter expected {LEGACY_V5_OBSERVATION_SIZE} values, "
            f"got {legacy.shape}"
        )
    return legacy


class LegacyV5PolicyAdapter:
    """Expose a legacy Discrete(7) PPO policy as a factorized racer policy."""

    def __init__(self, model: Any) -> None:
        if not is_legacy_v5_policy(model):
            raise ValueError("model does not use the legacy v5 RAM contract")
        self.model = model
        self.previous_action = 0

    def reset(self) -> None:
        self.previous_action = 0

    def predict(
        self, observation: np.ndarray, *, deterministic: bool = True
    ) -> tuple[np.ndarray, Any]:
        prediction = self.model.predict(
            legacy_v5_observation(observation, self.previous_action),
            deterministic=deterministic,
        )
        raw_action = prediction[0] if isinstance(prediction, tuple) else prediction
        action_index = int(np.asarray(raw_action).item())
        if not 0 <= action_index < len(LEGACY_V5_ACTIONS):
            raise ValueError(f"legacy v5 policy returned action {action_index}")
        self.previous_action = action_index
        state = prediction[1] if isinstance(prediction, tuple) else None
        return np.asarray(LEGACY_V5_ACTIONS[action_index], dtype=np.int64), state


class LegacyV6PolicyAdapter:
    """Let a 62-input factorized policy ignore the appended pickup radar."""

    def __init__(self, model: Any) -> None:
        if not is_legacy_v6_policy(model):
            raise ValueError("model does not use the legacy v6 RAM contract")
        self.model = model

    def reset(self) -> None:
        return None

    def predict(
        self, observation: np.ndarray, *, deterministic: bool = True
    ) -> tuple[np.ndarray, Any]:
        current = np.asarray(observation, dtype=np.float32).reshape(-1)
        if current.shape != (DINO_RAM_OBSERVATION_SIZE,):
            raise ValueError(
                f"active RAM observation expected {DINO_RAM_OBSERVATION_SIZE} "
                f"values, got {current.shape}"
            )
        return self.model.predict(
            current[:LEGACY_V6_OBSERVATION_SIZE], deterministic=deterministic
        )


def adapt_legacy_policy(model: Any) -> Any:
    """Adapt known pre-radar policies; return current/unknown models unchanged."""

    if is_legacy_v5_policy(model):
        return LegacyV5PolicyAdapter(model)
    if is_legacy_v6_policy(model):
        return LegacyV6PolicyAdapter(model)
    return model
