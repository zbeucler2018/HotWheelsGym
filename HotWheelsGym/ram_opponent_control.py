"""Pure contracts shared by the Player 1 RAM policy and model opponents.

The policy sees the same racer-relative observation and emits the same discrete
driving intent whether it occupies Player 1 or a patched native CPU slot. This
module deliberately has no Gymnasium, Stable-Retro, NumPy, or PyTorch imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, pi, sin
from operator import index as integer_index
from typing import Mapping

from .npc_control import (
    DEFAULT_MAX_TARGET_SPEED,
    DEFAULT_MAX_TURN,
    HEADING_PERIOD,
    NPCCommand,
    RACER_HELD_OFFSET,
    RACER_PRESSED_OFFSET,
    RACER_RELEASED_OFFSET,
    RACER_DESIRED_HEADING_OFFSET,
    RACER_TARGET_SPEED_OFFSET,
    RaceMemory,
    RacerState,
    heading_delta,
    progress_delta,
)

DINO_BONEYARD_PROGRESS_COUNT = 342
DINO_POSITION_CENTER = 1 << 24
DINO_POSITION_SCALE = 1 << 24
RELATIVE_POSITION_SCALE = 1 << 23
SPEED_DELTA_SCALE = 1 << 13

GBA_BUTTON_BITS = {
    "A": 0x001,
    "B": 0x002,
    "SELECT": 0x004,
    "START": 0x008,
    "RIGHT": 0x010,
    "LEFT": 0x020,
    "UP": 0x040,
    "DOWN": 0x080,
    "R": 0x100,
    "L": 0x200,
}


@dataclass(frozen=True)
class RacerAction:
    """One shared policy action and its Player/CPU interpretations."""

    name: str
    buttons: tuple[str, ...]
    steering: int
    throttle: int


@dataclass(frozen=True)
class RacerButtonState:
    """Native GBA transition masks written to a converted NPC racer."""

    pressed: int
    released: int
    held: int


RAM_ACTIONS = (
    RacerAction("coast", (), 0, 0),
    RacerAction("accelerate", ("A",), 0, 1),
    RacerAction("accelerate_left", ("A", "LEFT"), -1, 1),
    RacerAction("accelerate_right", ("A", "RIGHT"), 1, 1),
    RacerAction("brake", ("B",), 0, 0),
    RacerAction("accelerate_up", ("A", "UP"), 0, 1),
    RacerAction("boost", ("A", "L", "R"), 0, 1),
)
RAM_ACTION_SIZE = len(RAM_ACTIONS)

SELF_OBSERVATION_NAMES = (
    "self_heading_sin",
    "self_heading_cos",
    "self_x",
    "self_z",
    "self_speed",
    "track_phase_sin",
    "track_phase_cos",
    "self_lap",
    "self_rank",
    "self_acceleration",
    "self_turn_rate",
    "self_progress_rate",
) + tuple(f"previous_action_{action.name}" for action in RAM_ACTIONS)

OTHER_OBSERVATION_NAMES = (
    "forward",
    "right",
    "distance",
    "relative_progress",
    "relative_speed",
    "relative_heading_sin",
    "relative_heading_cos",
    "relative_lap",
)

RAM_OBSERVATION_NAMES = SELF_OBSERVATION_NAMES + tuple(
    f"nearby_racer_{position}_{name}"
    for position in (1, 2, 3)
    for name in OTHER_OBSERVATION_NAMES
)

SELF_OBSERVATION_SIZE = 12 + RAM_ACTION_SIZE
OTHER_OBSERVATION_SIZE = 8
DINO_RAM_OBSERVATION_SIZE = SELF_OBSERVATION_SIZE + 3 * OTHER_OBSERVATION_SIZE

if len(RAM_OBSERVATION_NAMES) != DINO_RAM_OBSERVATION_SIZE:
    raise RuntimeError("RAM observation name and value counts differ")


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _angle(heading: int) -> float:
    return 2.0 * pi * (heading & (HEADING_PERIOD - 1)) / HEADING_PERIOD


def _heading_pair(heading: int) -> tuple[float, float]:
    angle = _angle(heading)
    return sin(angle), cos(angle)


def _action_index(action: object) -> int:
    try:
        result = integer_index(action)
    except TypeError as error:
        raise ValueError("RAM racer action must be an integer") from error
    if not 0 <= result < RAM_ACTION_SIZE:
        raise ValueError(f"RAM racer action must be in 0..{RAM_ACTION_SIZE - 1}")
    return result


def player_buttons_from_action(
    action: object, button_names: tuple[str, ...] | list[str]
) -> tuple[bool, ...]:
    """Convert shared driving intent into the native Player 1 button vector."""

    selected = set(RAM_ACTIONS[_action_index(action)].buttons)
    missing = selected - set(button_names)
    if missing:
        raise ValueError(
            "environment is missing buttons: " + ", ".join(sorted(missing))
        )
    return tuple(name in selected for name in button_names)


def gba_button_mask_from_action(action: object) -> int:
    """Convert one shared action to the game's native 10-bit button mask."""

    return sum(
        GBA_BUTTON_BITS[name] for name in RAM_ACTIONS[_action_index(action)].buttons
    )


def write_racer_buttons(
    race: RaceMemory,
    slot: int,
    action: object,
    previous_held: int,
) -> RacerButtonState:
    """Give a converted NPC exactly the player-class button transition fields."""

    racer = race.layout.opponent(slot)
    if racer.kind != "player":
        raise ValueError(
            f"racer slot {slot} is {racer.kind}, not a button-controlled player class"
        )
    held = gba_button_mask_from_action(action)
    previous_held &= 0x3FF
    state = RacerButtonState(
        pressed=held & ~previous_held,
        released=previous_held & ~held,
        held=held,
    )
    race.memory.assign(racer.address + RACER_PRESSED_OFFSET, "<u2", state.pressed)
    race.memory.assign(racer.address + RACER_RELEASED_OFFSET, "<u2", state.released)
    race.memory.assign(racer.address + RACER_HELD_OFFSET, "<u2", state.held)
    return state


def cpu_command_from_action(
    state: RacerState,
    action: object,
    *,
    max_turn: int = DEFAULT_MAX_TURN,
    max_target_speed: int = DEFAULT_MAX_TARGET_SPEED,
) -> NPCCommand:
    """Convert the same intent into the patched native CPU control boundary."""

    if not 0 <= max_turn <= 0x800:
        raise ValueError("max_turn must be between 0 and 0x800")
    if not 0 < max_target_speed <= 0xFFFFFFFF:
        raise ValueError("max_target_speed must fit in an unsigned 32-bit value")
    selected = RAM_ACTIONS[_action_index(action)]
    desired_heading = (state.current_heading + selected.steering * max_turn) & (
        HEADING_PERIOD - 1
    )
    return NPCCommand(
        desired_heading=desired_heading,
        target_speed=selected.throttle * max_target_speed,
    )


def write_cpu_command(
    race: RaceMemory,
    slot: int,
    action: object,
    *,
    max_turn: int = DEFAULT_MAX_TURN,
    max_target_speed: int = DEFAULT_MAX_TARGET_SPEED,
) -> NPCCommand:
    racer = race.layout.cpu(slot)
    command = cpu_command_from_action(
        race.state(slot),
        action,
        max_turn=max_turn,
        max_target_speed=max_target_speed,
    )
    race.memory.assign(
        racer.address + RACER_DESIRED_HEADING_OFFSET,
        "<u2",
        command.desired_heading,
    )
    race.memory.assign(
        racer.address + RACER_TARGET_SPEED_OFFSET,
        "<u4",
        command.target_speed,
    )
    return command


@dataclass
class RaceProgressTracker:
    """Track per-racer laps and race order across checkpoint wraparound."""

    progress_count: int
    laps: dict[int, int]
    previous_progress: dict[int, int]

    @classmethod
    def from_states(
        cls, states: Mapping[int, RacerState], progress_count: int
    ) -> "RaceProgressTracker":
        if progress_count <= 0:
            raise ValueError("progress_count must be positive")
        return cls(
            progress_count=progress_count,
            laps={slot: 1 for slot in states},
            previous_progress={slot: state.progress for slot, state in states.items()},
        )

    def update(self, states: Mapping[int, RacerState]) -> None:
        if set(states) != set(self.previous_progress):
            raise ValueError("racer slots changed after progress tracking started")
        for slot, state in states.items():
            previous = self.previous_progress[slot]
            delta = progress_delta(previous, state.progress, self.progress_count)
            if delta > 0 and state.progress < previous:
                self.laps[slot] += 1
            elif delta < 0 and state.progress > previous and self.laps[slot] > 1:
                self.laps[slot] -= 1
            self.previous_progress[slot] = state.progress

    def total_progress(self, slot: int, state: RacerState) -> int:
        return (self.laps[slot] - 1) * self.progress_count + state.progress

    def current_lap(self, slot: int, state: RacerState) -> int:
        """Return a one-based lap for wrapped or monotonic progress counters."""

        return max(1, self.total_progress(slot, state) // self.progress_count + 1)

    def rank(self, slot: int, states: Mapping[int, RacerState]) -> int:
        own = self.total_progress(slot, states[slot])
        return 1 + sum(
            self.total_progress(other_slot, state) > own
            for other_slot, state in states.items()
            if other_slot != slot
        )

    def completion(self, slot: int, state: RacerState, total_laps: int) -> float:
        denominator = max(1, total_laps * self.progress_count)
        return _clip(self.total_progress(slot, state) / denominator, 0.0, 1.0)


def read_racer_states(race: RaceMemory) -> dict[int, RacerState]:
    return {racer.slot: race.state(racer.slot) for racer in race.layout.racers}


def _ordered_other_slots(
    controlled_slot: int,
    states: Mapping[int, RacerState],
    progress: RaceProgressTracker,
) -> tuple[int, ...]:
    if controlled_slot not in states:
        raise ValueError(f"controlled racer slot {controlled_slot} is missing")
    if len(states) != 4:
        raise ValueError(f"expected four racers, got {len(states)}")
    own_total = progress.total_progress(controlled_slot, states[controlled_slot])
    return tuple(
        sorted(
            (slot for slot in states if slot != controlled_slot),
            key=lambda slot: (
                abs(progress.total_progress(slot, states[slot]) - own_total),
                slot,
            ),
        )
    )


def build_dino_ram_observation(
    states: Mapping[int, RacerState],
    previous_states: Mapping[int, RacerState],
    progress: RaceProgressTracker,
    controlled_slot: int,
    previous_action: object,
    *,
    total_laps: int = 3,
    max_target_speed: int = DEFAULT_MAX_TARGET_SPEED,
) -> tuple[float, ...]:
    """Build the same Dino racer-centric observation for Player 1 or a CPU."""

    if set(states) != set(previous_states):
        raise ValueError("current and previous racer slots differ")
    if total_laps <= 0 or max_target_speed <= 0:
        raise ValueError("normalization scales must be positive")

    action_index = _action_index(previous_action)
    state = states[controlled_slot]
    previous = previous_states[controlled_slot]
    heading = _heading_pair(state.current_heading)
    track_phase = _heading_pair(
        round((state.progress / progress.progress_count) * HEADING_PERIOD)
    )
    lap_value = _clip(
        (progress.current_lap(controlled_slot, state) - 1) / max(1, total_laps - 1),
        0.0,
        1.0,
    )
    rank_value = _clip((progress.rank(controlled_slot, states) - 1) / 3.0, 0.0, 1.0)
    advance = progress_delta(previous.progress, state.progress, progress.progress_count)

    features: list[float] = [
        *heading,
        _clip((state.x - DINO_POSITION_CENTER) / DINO_POSITION_SCALE),
        _clip((state.z - DINO_POSITION_CENTER) / DINO_POSITION_SCALE),
        _clip(state.speed / max_target_speed, 0.0, 1.0),
        *track_phase,
        lap_value,
        rank_value,
        _clip((state.speed - previous.speed) / SPEED_DELTA_SCALE),
        _clip(heading_delta(state.current_heading, previous.current_heading) / 0x200),
        _clip(advance / 4.0),
        *(1.0 if index == action_index else 0.0 for index in range(RAM_ACTION_SIZE)),
    ]

    angle = _angle(state.current_heading)
    forward_x, forward_z = sin(angle), cos(angle)
    right_x, right_z = cos(angle), -sin(angle)
    own_total = progress.total_progress(controlled_slot, state)

    for other_slot in _ordered_other_slots(controlled_slot, states, progress):
        other = states[other_slot]
        dx = other.x - state.x
        dz = other.z - state.z
        relative_heading = _heading_pair(
            heading_delta(other.current_heading, state.current_heading)
        )
        relative_progress = progress.total_progress(other_slot, other) - own_total
        features.extend(
            (
                _clip((dx * forward_x + dz * forward_z) / RELATIVE_POSITION_SCALE),
                _clip((dx * right_x + dz * right_z) / RELATIVE_POSITION_SCALE),
                _clip(hypot(dx, dz) / RELATIVE_POSITION_SCALE, 0.0, 1.0),
                _clip(relative_progress / max(1.0, progress.progress_count / 2.0)),
                _clip((other.speed - state.speed) / max_target_speed),
                *relative_heading,
                _clip(
                    (
                        progress.current_lap(other_slot, other)
                        - progress.current_lap(controlled_slot, state)
                    )
                    / max(1, total_laps)
                ),
            )
        )

    if len(features) != DINO_RAM_OBSERVATION_SIZE:
        raise RuntimeError(
            f"RAM observation has {len(features)} values, "
            f"expected {DINO_RAM_OBSERVATION_SIZE}"
        )
    return tuple(features)


def race_reward(
    advance: int,
    speed: int,
    max_target_speed: int,
    *,
    previous_rank: int,
    current_rank: int,
    completed_laps: int = 0,
    finished_now: bool = False,
) -> float:
    """Progress-dominant reward with pressure toward quick race completion."""

    speed_ratio = _clip(speed / max(1, max_target_speed), 0.0, 1.0)
    return float(
        advance
        + 0.002 * speed_ratio
        - 0.005
        + 0.25 * (previous_rank - current_rank)
        + 5.0 * completed_laps
        + (50.0 if finished_now else 0.0)
    )
