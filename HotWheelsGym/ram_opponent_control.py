"""Pure contracts shared by the Player 1 RAM policy and model opponents.

The policy sees the same racer-relative observation and emits the same factorized
button action whether it occupies Player 1 or a converted player-class slot. This
module deliberately has no Gymnasium, Stable-Retro, NumPy, or PyTorch imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, hypot, pi, sin
from operator import index as integer_index
from typing import Mapping

from .dino_boneyard_track import (
    DINO_TRACK_OBSERVATION_NAMES,
    DINO_TRACK_OBSERVATION_SIZE,
    DINO_TRACK_POINT_COUNT,
    dino_track_pose,
)
from .npc_control import (
    HEADING_PERIOD,
    MAX_JET_BOOST_TIMER,
    MAX_BOOST_CHARGE,
    RACER_HELD_OFFSET,
    RACER_PRESSED_OFFSET,
    RACER_RELEASED_OFFSET,
    RaceMemory,
    RacerState,
    heading_delta,
    progress_delta,
)

DINO_BONEYARD_PROGRESS_COUNT = 342
DINO_HAIRPIN_START_PROGRESS = 45
DINO_HAIRPIN_END_PROGRESS = 61
DINO_RAM_OBSERVATION_VERSION = 6
DINO_SECTOR_BOUNDARIES = (0, 32, 45, 61, 92, 124, 156, 188, 220, 252, 284, 316, 342)
DINO_SECTOR_COUNT = len(DINO_SECTOR_BOUNDARIES) - 1
DINO_POSITION_CENTER = 1 << 24
DINO_POSITION_SCALE = 1 << 24
RELATIVE_POSITION_SCALE = 1 << 23
SPEED_DELTA_SCALE = 1 << 13
RAM_SPEED_SCALE = 0x12000

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

if DINO_TRACK_POINT_COUNT != DINO_BONEYARD_PROGRESS_COUNT:
    raise RuntimeError("Dino track reference and progress counts differ")


@dataclass(frozen=True)
class RacerAction:
    """One shared policy action and its native GBA buttons."""

    name: str
    buttons: tuple[str, ...]


@dataclass(frozen=True)
class RacerButtonState:
    """Native GBA transition masks written to a converted NPC racer."""

    pressed: int
    released: int
    held: int


@dataclass(frozen=True)
class BoostTelemetry:
    """One racer boost-meter transition, excluding non-boost charge loss."""

    delta: int
    spent: int
    gained: int
    active: bool


@dataclass
class DinoHairpinTelemetry:
    """Accumulate raw-frame diagnostics for Dino Boneyard's first hairpin."""

    entries: int = 0
    completed: int = 0
    entries_with_jet_boost: int = 0
    frames: int = 0
    completed_frames: int = 0
    speed_total: int = 0
    entry_speed_total: int = 0
    exit_speed_total: int = 0
    minimum_speed_total: int = 0
    wall_frames: int = 0
    skid_frames: int = 0
    active: bool = False
    _active_frames: int = 0
    _active_minimum_speed: int | None = None

    @staticmethod
    def contains(progress: int) -> bool:
        index = progress % DINO_BONEYARD_PROGRESS_COUNT
        return DINO_HAIRPIN_START_PROGRESS <= index < DINO_HAIRPIN_END_PROGRESS

    def update(
        self,
        previous: RacerState,
        current: RacerState,
        *,
        advance: int,
        hit_wall: bool,
    ) -> None:
        """Record one emulator-frame transition without affecting game state."""

        was_inside = self.contains(previous.progress)
        is_inside = self.contains(current.progress)
        if not self.active and is_inside and not was_inside and advance > 0:
            self.active = True
            self.entries += 1
            self.entries_with_jet_boost += int(current.jet_boost_remaining > 0)
            self.entry_speed_total += current.speed
            self._active_frames = 0
            self._active_minimum_speed = None

        if self.active and is_inside:
            self.frames += 1
            self.speed_total += current.speed
            self.wall_frames += int(hit_wall)
            self.skid_frames += int(current.skid_active)
            self._active_frames += 1
            if self._active_minimum_speed is None:
                self._active_minimum_speed = current.speed
            else:
                self._active_minimum_speed = min(
                    self._active_minimum_speed, current.speed
                )

        if self.active and was_inside and not is_inside:
            current_index = current.progress % DINO_BONEYARD_PROGRESS_COUNT
            if advance > 0 and current_index >= DINO_HAIRPIN_END_PROGRESS:
                self.completed += 1
                self.completed_frames += self._active_frames
                self.exit_speed_total += current.speed
                self.minimum_speed_total += self._active_minimum_speed or 0
            self.active = False
            self._active_frames = 0
            self._active_minimum_speed = None


def dino_sector_index(progress: int) -> int:
    """Map native track progress to one of the full-lap diagnostic sectors."""

    index = progress % DINO_BONEYARD_PROGRESS_COUNT
    for sector, (start, end) in enumerate(
        zip(DINO_SECTOR_BOUNDARIES, DINO_SECTOR_BOUNDARIES[1:])
    ):
        if start <= index < end:
            return sector
    raise RuntimeError(f"Dino progress {index} is outside the sector map")


@dataclass
class DinoSectorTelemetry:
    """Accumulate raw-frame timing and handling diagnostics across the whole lap."""

    entries: list[int] = field(default_factory=lambda: [0] * DINO_SECTOR_COUNT)
    completed: list[int] = field(default_factory=lambda: [0] * DINO_SECTOR_COUNT)
    frames: list[int] = field(default_factory=lambda: [0] * DINO_SECTOR_COUNT)
    completed_frames: list[int] = field(default_factory=lambda: [0] * DINO_SECTOR_COUNT)
    entry_speed_total: list[int] = field(
        default_factory=lambda: [0] * DINO_SECTOR_COUNT
    )
    minimum_speed_total: list[int] = field(
        default_factory=lambda: [0] * DINO_SECTOR_COUNT
    )
    exit_speed_total: list[int] = field(default_factory=lambda: [0] * DINO_SECTOR_COUNT)
    wall_frames: list[int] = field(default_factory=lambda: [0] * DINO_SECTOR_COUNT)
    skid_frames: list[int] = field(default_factory=lambda: [0] * DINO_SECTOR_COUNT)
    boost_frames: list[int] = field(default_factory=lambda: [0] * DINO_SECTOR_COUNT)
    jet_boost_frames: list[int] = field(default_factory=lambda: [0] * DINO_SECTOR_COUNT)
    jet_boost_pickups: list[int] = field(
        default_factory=lambda: [0] * DINO_SECTOR_COUNT
    )
    lap_completed_frames: dict[tuple[int, int], int] = field(default_factory=dict)
    active_sector: int | None = None
    _active_lap: int = 1
    _active_frames: int = 0
    _active_minimum_speed: int | None = None

    @classmethod
    def start(cls, state: RacerState, current_lap: int = 1) -> "DinoSectorTelemetry":
        telemetry = cls()
        telemetry._start_sector(
            dino_sector_index(state.progress), state.speed, current_lap
        )
        return telemetry

    def _start_sector(self, sector: int, speed: int, current_lap: int) -> None:
        self.active_sector = sector
        self._active_lap = current_lap
        self.entries[sector] += 1
        self.entry_speed_total[sector] += speed
        self._active_frames = 0
        self._active_minimum_speed = None

    def update(
        self,
        previous: RacerState,
        current: RacerState,
        *,
        advance: int,
        hit_wall: bool,
        boost_active: bool,
        jet_boost_acquired: bool,
        current_lap: int = 1,
    ) -> None:
        """Record one emulator frame without affecting the policy or game state."""

        current_sector = dino_sector_index(current.progress)
        if self.active_sector is None:
            self._start_sector(current_sector, current.speed, current_lap)
        elif current_sector != self.active_sector:
            expected_sector = (self.active_sector + 1) % DINO_SECTOR_COUNT
            if advance > 0 and current_sector == expected_sector:
                completed_sector = self.active_sector
                self.completed[completed_sector] += 1
                self.completed_frames[completed_sector] += self._active_frames
                key = (self._active_lap, completed_sector)
                self.lap_completed_frames[key] = (
                    self.lap_completed_frames.get(key, 0) + self._active_frames
                )
                self.exit_speed_total[completed_sector] += current.speed
                self.minimum_speed_total[completed_sector] += (
                    self._active_minimum_speed or 0
                )
            self._start_sector(current_sector, current.speed, current_lap)

        sector = current_sector
        self.frames[sector] += 1
        self.wall_frames[sector] += int(hit_wall)
        self.skid_frames[sector] += int(current.skid_active)
        self.boost_frames[sector] += int(boost_active)
        self.jet_boost_frames[sector] += int(current.jet_boost_remaining > 0)
        self.jet_boost_pickups[sector] += int(jet_boost_acquired)
        self._active_frames += 1
        if self._active_minimum_speed is None:
            self._active_minimum_speed = current.speed
        else:
            self._active_minimum_speed = min(self._active_minimum_speed, current.speed)


RAM_DRIVE_ACTIONS = (
    RacerAction("coast", ()),
    RacerAction("accelerate", ("A",)),
    RacerAction("brake", ("B",)),
    RacerAction("accelerate_up", ("A", "UP")),
)
RAM_STEERING_ACTIONS = (
    RacerAction("straight", ()),
    RacerAction("left", ("LEFT",)),
    RacerAction("right", ("RIGHT",)),
)
RAM_BOOST_ACTIONS = (
    RacerAction("off", ()),
    RacerAction("on", ("L", "R")),
)
RAM_ACTION_COMPONENTS = (
    ("drive", RAM_DRIVE_ACTIONS),
    ("steering", RAM_STEERING_ACTIONS),
    ("boost", RAM_BOOST_ACTIONS),
)
RAM_ACTION_COMPONENT_SIZES = tuple(len(actions) for _, actions in RAM_ACTION_COMPONENTS)
RAM_DEFAULT_ACTION = (0, 0, 0)
RAM_BOOST_COMPONENT = 2
RAM_BOOST_ON = 1

SELF_OBSERVATION_NAMES = (
    (
        "self_heading_sin",
        "self_heading_cos",
        "self_x",
        "self_z",
        "self_speed",
        "self_boost_charge",
        "self_jet_boost_remaining",
        "self_skid_active",
        "track_phase_sin",
        "track_phase_cos",
        "self_lap",
        "self_rank",
        "self_acceleration",
        "self_turn_rate",
        "self_progress_rate",
    )
    + DINO_TRACK_OBSERVATION_NAMES
    + tuple(
        f"previous_{component}_{action.name}"
        for component, actions in RAM_ACTION_COMPONENTS
        for action in actions
    )
)

OTHER_OBSERVATION_NAMES = (
    "forward",
    "right",
    "distance",
    "relative_progress",
    "relative_speed",
    "boost_charge",
    "relative_heading_sin",
    "relative_heading_cos",
    "relative_lap",
)

RAM_OBSERVATION_NAMES = SELF_OBSERVATION_NAMES + tuple(
    f"nearby_racer_{position}_{name}"
    for position in (1, 2, 3)
    for name in OTHER_OBSERVATION_NAMES
)

SELF_OBSERVATION_SIZE = (
    15 + DINO_TRACK_OBSERVATION_SIZE + sum(RAM_ACTION_COMPONENT_SIZES)
)
OTHER_OBSERVATION_SIZE = 9
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


def normalize_racer_action(action: object) -> tuple[int, int, int]:
    """Validate one factorized ``drive, steering, boost`` policy action."""

    try:
        raw_components = tuple(action)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(
            "RAM racer action must contain drive, steering, and boost values"
        ) from error
    if len(raw_components) != len(RAM_ACTION_COMPONENTS):
        raise ValueError(
            "RAM racer action must contain exactly drive, steering, and boost values"
        )
    components: list[int] = []
    for raw_value, (name, choices) in zip(raw_components, RAM_ACTION_COMPONENTS):
        try:
            value = integer_index(raw_value)
        except TypeError as error:
            raise ValueError(f"RAM racer {name} action must be an integer") from error
        if not 0 <= value < len(choices):
            raise ValueError(
                f"RAM racer {name} action must be in 0..{len(choices) - 1}"
            )
        components.append(value)
    return tuple(components)  # type: ignore[return-value]


def racer_action_buttons(action: object) -> tuple[str, ...]:
    """Return the native buttons represented by a factorized policy action."""

    components = normalize_racer_action(action)
    return tuple(
        button
        for component, (_, choices) in zip(components, RAM_ACTION_COMPONENTS)
        for button in choices[component].buttons
    )


def player_buttons_from_action(
    action: object, button_names: tuple[str, ...] | list[str]
) -> tuple[bool, ...]:
    """Convert shared driving intent into the native Player 1 button vector."""

    selected = set(racer_action_buttons(action))
    missing = selected - set(button_names)
    if missing:
        raise ValueError(
            "environment is missing buttons: " + ", ".join(sorted(missing))
        )
    return tuple(name in selected for name in button_names)


def gba_button_mask_from_action(action: object) -> int:
    """Convert one shared action to the game's native 10-bit button mask."""

    return sum(GBA_BUTTON_BITS[name] for name in racer_action_buttons(action))


def boost_telemetry(
    previous_charge: int, current_charge: int, action: object
) -> BoostTelemetry:
    """Describe charge movement caused while a racer is requesting boost."""

    components = normalize_racer_action(action)
    delta = current_charge - previous_charge
    active = components[RAM_BOOST_COMPONENT] == RAM_BOOST_ON and previous_charge > 0
    return BoostTelemetry(
        delta=delta,
        spent=max(0, -delta) if active else 0,
        gained=max(0, delta),
        active=active,
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


@dataclass
class LapSplitTracker:
    """Measure completed laps in raw emulator frames."""

    total_laps: int
    completed_laps: int
    last_crossing_frame: int
    split_frames: list[int]

    @classmethod
    def start(cls, current_lap: int, total_laps: int) -> "LapSplitTracker":
        if total_laps < 1:
            raise ValueError("total_laps must be positive")
        completed = max(0, min(total_laps, current_lap - 1))
        return cls(total_laps, completed, 0, [0] * completed)

    def update(
        self, current_lap: int, raw_frame: int, *, finished_now: bool = False
    ) -> None:
        if raw_frame < self.last_crossing_frame:
            raise ValueError("raw_frame cannot move backward")
        target = max(0, min(self.total_laps, current_lap - 1))
        if finished_now:
            target = self.total_laps
        while self.completed_laps < target:
            self.split_frames.append(raw_frame - self.last_crossing_frame)
            self.last_crossing_frame = raw_frame
            self.completed_laps += 1

    def padded_splits(self) -> tuple[int, ...]:
        return tuple(self.split_frames) + (0,) * (
            self.total_laps - len(self.split_frames)
        )


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
) -> tuple[float, ...]:
    """Build the same Dino racer-centric observation for any racer slot."""

    if set(states) != set(previous_states):
        raise ValueError("current and previous racer slots differ")
    if total_laps <= 0:
        raise ValueError("total_laps must be positive")

    action_components = normalize_racer_action(previous_action)
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
    track = dino_track_pose(state)

    features: list[float] = [
        *heading,
        _clip((state.x - DINO_POSITION_CENTER) / DINO_POSITION_SCALE),
        _clip((state.z - DINO_POSITION_CENTER) / DINO_POSITION_SCALE),
        _clip(state.speed / RAM_SPEED_SCALE, 0.0, 1.0),
        _clip(state.boost / MAX_BOOST_CHARGE, 0.0, 1.0),
        _clip(state.jet_boost_remaining / MAX_JET_BOOST_TIMER, 0.0, 1.0),
        float(state.skid_active),
        *track_phase,
        lap_value,
        rank_value,
        _clip((state.speed - previous.speed) / SPEED_DELTA_SCALE),
        _clip(heading_delta(state.current_heading, previous.current_heading) / 0x200),
        _clip(advance / 4.0),
        *track.features,
        *(
            1.0 if index == selected else 0.0
            for selected, size in zip(action_components, RAM_ACTION_COMPONENT_SIZES)
            for index in range(size)
        ),
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
                _clip((other.speed - state.speed) / RAM_SPEED_SCALE),
                _clip(other.boost / MAX_BOOST_CHARGE, 0.0, 1.0),
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
    *,
    previous_rank: int,
    current_rank: int,
    completed_laps: int = 0,
    finished_now: bool = False,
    hit_wall: bool = False,
    lateral_offset: float = 0.0,
    heading_alignment: float = 1.0,
    respawned_now: bool = False,
) -> float:
    """Progress-dominant reward with modest road-following safety shaping."""

    speed_ratio = _clip(speed / RAM_SPEED_SCALE, 0.0, 1.0)
    lateral_penalty = 0.003 * abs(_clip(lateral_offset))
    heading_penalty = 0.002 * (1.0 - _clip(heading_alignment)) / 2.0
    return float(
        advance
        + 0.002 * speed_ratio
        - 0.005
        - lateral_penalty
        - heading_penalty
        - (0.05 if hit_wall else 0.0)
        - (2.0 if respawned_now else 0.0)
        + 0.25 * (previous_rank - current_rank)
        + 5.0 * completed_laps
        + (50.0 if finished_now else 0.0)
    )
