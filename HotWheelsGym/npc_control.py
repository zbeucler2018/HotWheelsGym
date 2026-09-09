"""ROM-independent memory model for controlling native CPU racers."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, sin
from pathlib import Path
import struct
from typing import Mapping, Protocol, Sequence

EWRAM_BASE = 0x02000000
EWRAM_SIZE = 0x40000

PLAYER_RACER_VTABLE = 0x0817C430
CPU_RACER_VTABLE = 0x0817B6D4
RACER_MANAGER_OFFSET = 0x50
RACER_VEHICLE_INDEX_OFFSET = 0xDC
RACER_CURRENT_HEADING_OFFSET = 0xDE
RACER_STEERING_HEADING_OFFSET = 0xD8
RACER_DESIRED_HEADING_OFFSET = 0xE0
RACER_SPEED_OFFSET = 0xE8
RACER_X_OFFSET = 0xF8
RACER_Z_OFFSET = 0x100
RACER_PROGRESS_OFFSET = 0x148
RACER_RESPAWN_PENDING_OFFSET = 0x26B
RACER_STOCK_HEADING_SHADOW_OFFSET = 0x2EE
RACER_TARGET_SPEED_OFFSET = 0x2F0
RACER_PRESSED_OFFSET = 0x302
RACER_RELEASED_OFFSET = 0x304
RACER_HELD_OFFSET = 0x306

MANAGER_PLAYER_COUNT_OFFSET = 0x448
MANAGER_CPU_COUNT_OFFSET = 0x449
MANAGER_TOTAL_COUNT_OFFSET = 0x44A
MANAGER_POINTER_LIST_OFFSET = 0x450

NPC_CONTROL_MARKER_OFFSET = 0x79BF08
NPC_CONTROL_MARKER = b"HWNP"
NPC_CONTROL_PATCH_VERSION = 1
NPC_BUTTON_CONTROL_MARKER_OFFSET = 0x79BEF4
NPC_BUTTON_CONTROL_MARKER = b"HWBT"
NPC_BUTTON_CONTROL_PATCH_VERSION = 3

HEADING_PERIOD = 0x1000
DEFAULT_MAX_TURN = 0x200
DEFAULT_MAX_TARGET_SPEED = 0x12000
OBSERVATION_SIZE = 12

TRACK_PROGRESS_COUNTS = {
    "trex_valley": 316,
    "dino_boneyard": 342,
    "black_widows_nest": 395,
    "insect_hive": 380,
    "monsters_of_the_deep": 342,
    "whiteskull_cliffs": 340,
    "jungle_snakepit": 465,
    "gator_forest": 512,
    "satellite_mission": 376,
    "solar_strip": 325,
    "fire_mountain": 465,
    "volcano_battle": 495,
}


class MemoryView(Protocol):
    """Subset of Stable-Retro's memory API used by the control wrappers."""

    @property
    def blocks(self) -> Mapping[int, bytes]: ...

    def extract(self, address: int, data_type: str) -> int: ...

    def assign(self, address: int, data_type: str, value: int) -> None: ...


@dataclass(frozen=True)
class RacerRef:
    slot: int
    kind: str
    address: int
    manager: int
    vehicle_index: int


@dataclass(frozen=True)
class RaceLayout:
    manager: int
    pointer_list: int
    racers: tuple[RacerRef, ...]

    def racer(self, slot: int) -> RacerRef:
        try:
            racer = self.racers[slot]
        except IndexError as error:
            raise ValueError(
                f"racer slot {slot} is outside the {len(self.racers)}-racer table"
            ) from error
        if racer.slot != slot:
            raise RuntimeError("racer table slot ordering is inconsistent")
        return racer

    def cpu(self, slot: int) -> RacerRef:
        racer = self.racer(slot)
        if racer.kind != "cpu":
            raise ValueError(f"racer slot {slot} is {racer.kind}, not a native CPU")
        return racer

    def opponent(self, slot: int) -> RacerRef:
        """Return a non-Player-1 racer, including a converted player-class NPC."""

        racer = self.racer(slot)
        if slot == 0:
            raise ValueError("racer slot 0 is Player 1, not an opponent")
        return racer


@dataclass(frozen=True)
class RacerState:
    slot: int
    vehicle_index: int
    current_heading: int
    desired_heading: int
    stock_heading: int
    speed: int
    target_speed: int
    progress: int
    x: int
    z: int
    rank: int


@dataclass(frozen=True)
class NPCCommand:
    desired_heading: int
    target_speed: int


def _read_u8(memory: MemoryView, address: int) -> int:
    return int(memory.extract(address, "|u1"))


def _read_u16(memory: MemoryView, address: int) -> int:
    return int(memory.extract(address, "<u2"))


def _read_u32(memory: MemoryView, address: int) -> int:
    return int(memory.extract(address, "<u4"))


def _signed_u32(value: int) -> int:
    return value - (1 << 32) if value & (1 << 31) else value


def _aligned_hits(blocks: Mapping[int, bytes], value: int) -> set[int]:
    needle = struct.pack("<I", value)
    hits: set[int] = set()
    for raw_base, raw_block in blocks.items():
        base = int(raw_base)
        block = bytes(raw_block)
        position = 0
        while True:
            position = block.find(needle, position)
            if position < 0:
                break
            if (base + position) % 4 == 0:
                hits.add(base + position)
            position += 1
    return hits


def discover_race_layout(memory: MemoryView) -> RaceLayout:
    """Discover the active race manager and its ordered racer pointer table."""

    known_vtables = {
        PLAYER_RACER_VTABLE: "player",
        CPU_RACER_VTABLE: "cpu",
    }
    candidates: dict[int, str] = {}
    for vtable, kind in known_vtables.items():
        for address in _aligned_hits(memory.blocks, vtable):
            if EWRAM_BASE <= address < EWRAM_BASE + EWRAM_SIZE:
                candidates[address] = kind

    layouts: dict[int, RaceLayout] = {}
    for racer_address in candidates:
        manager = _read_u32(memory, racer_address + RACER_MANAGER_OFFSET)
        if not EWRAM_BASE <= manager < EWRAM_BASE + EWRAM_SIZE:
            continue
        try:
            players = _read_u8(memory, manager + MANAGER_PLAYER_COUNT_OFFSET)
            cpus = _read_u8(memory, manager + MANAGER_CPU_COUNT_OFFSET)
            total = _read_u8(memory, manager + MANAGER_TOTAL_COUNT_OFFSET)
            pointer_list = _read_u32(memory, manager + MANAGER_POINTER_LIST_OFFSET)
            if not (total and total <= 8 and players + cpus == total):
                continue
            if not EWRAM_BASE <= pointer_list < EWRAM_BASE + EWRAM_SIZE:
                continue
            racers: list[RacerRef] = []
            for slot in range(total):
                address = _read_u32(memory, pointer_list + slot * 4)
                vtable = _read_u32(memory, address)
                kind = known_vtables[vtable]
                if _read_u32(memory, address + RACER_MANAGER_OFFSET) != manager:
                    raise ValueError("racer points at a different manager")
                racers.append(
                    RacerRef(
                        slot=slot,
                        kind=kind,
                        address=address,
                        manager=manager,
                        vehicle_index=_read_u8(
                            memory, address + RACER_VEHICLE_INDEX_OFFSET
                        ),
                    )
                )
            if sum(racer.kind == "player" for racer in racers) != players:
                continue
            if sum(racer.kind == "cpu" for racer in racers) != cpus:
                continue
        except (KeyError, ValueError):
            continue
        layouts[manager] = RaceLayout(manager, pointer_list, tuple(racers))

    if not layouts:
        raise RuntimeError("could not discover an active Hot Wheels race manager")
    if len(layouts) != 1:
        addresses = ", ".join(f"{address:#010x}" for address in sorted(layouts))
        raise RuntimeError(f"found multiple plausible race managers: {addresses}")
    return next(iter(layouts.values()))


class RaceMemory:
    """Read observations and write commands through a discovered race layout."""

    def __init__(self, memory: MemoryView):
        self.memory = memory
        self.layout = discover_race_layout(memory)

    def state(self, slot: int) -> RacerState:
        racer = self.layout.racer(slot)
        states = []
        for other in self.layout.racers:
            states.append(
                (
                    other,
                    _read_u16(self.memory, other.address + RACER_PROGRESS_OFFSET),
                )
            )
        progress = next(value for other, value in states if other.slot == slot)
        rank = 1 + sum(value > progress for _, value in states)
        return RacerState(
            slot=slot,
            vehicle_index=racer.vehicle_index,
            current_heading=_read_u16(
                self.memory, racer.address + RACER_CURRENT_HEADING_OFFSET
            )
            & (HEADING_PERIOD - 1),
            desired_heading=_read_u16(
                self.memory, racer.address + RACER_DESIRED_HEADING_OFFSET
            )
            & (HEADING_PERIOD - 1),
            stock_heading=_read_u16(
                self.memory, racer.address + RACER_STOCK_HEADING_SHADOW_OFFSET
            )
            & (HEADING_PERIOD - 1),
            speed=_read_u32(self.memory, racer.address + RACER_SPEED_OFFSET),
            target_speed=_read_u32(
                self.memory, racer.address + RACER_TARGET_SPEED_OFFSET
            ),
            progress=progress,
            x=_signed_u32(_read_u32(self.memory, racer.address + RACER_X_OFFSET)),
            z=_signed_u32(_read_u32(self.memory, racer.address + RACER_Z_OFFSET)),
            rank=rank,
        )

    def respawn_pending(self, slot: int) -> bool:
        """Return whether a player-class racer has entered its respawn sequence."""

        racer = self.layout.racer(slot)
        if racer.kind != "player":
            return False
        return bool(_read_u8(self.memory, racer.address + RACER_RESPAWN_PENDING_OFFSET))

    def command(
        self,
        slot: int,
        action: Sequence[float],
        *,
        max_turn: int = DEFAULT_MAX_TURN,
        max_target_speed: int = DEFAULT_MAX_TARGET_SPEED,
    ) -> NPCCommand:
        racer = self.layout.cpu(slot)
        state = self.state(slot)
        command = command_from_action(
            state,
            action,
            max_turn=max_turn,
            max_target_speed=max_target_speed,
        )
        self.memory.assign(
            racer.address + RACER_DESIRED_HEADING_OFFSET,
            "<u2",
            command.desired_heading,
        )
        self.memory.assign(
            racer.address + RACER_TARGET_SPEED_OFFSET,
            "<u4",
            command.target_speed,
        )
        return command

    def prime_stock_heading(self, slot: int) -> None:
        """Seed the patch-owned observation field from the savestate heading."""

        racer = self.layout.cpu(slot)
        desired = _read_u16(self.memory, racer.address + RACER_DESIRED_HEADING_OFFSET)
        self.memory.assign(
            racer.address + RACER_STOCK_HEADING_SHADOW_OFFSET,
            "<u2",
            desired & (HEADING_PERIOD - 1),
        )

    def observation(
        self,
        slot: int,
        progress_count: int,
        *,
        max_target_speed: int = DEFAULT_MAX_TARGET_SPEED,
    ) -> tuple[float, ...]:
        state = self.state(slot)
        player = self.state(0)
        angle = 2.0 * pi * state.current_heading / HEADING_PERIOD
        stock_angle = 2.0 * pi * state.stock_heading / HEADING_PERIOD
        scale = float(max(1, max_target_speed))
        count = float(max(1, progress_count))
        features = (
            sin(angle),
            cos(angle),
            sin(stock_angle),
            cos(stock_angle),
            _clip(state.speed / scale, -1.0, 1.0),
            _clip(state.target_speed / scale, 0.0, 1.0),
            _clip(state.progress / count, 0.0, 1.0),
            _clip((player.progress - state.progress) / count, -1.0, 1.0),
            _clip((player.x - state.x) / (1 << 22), -1.0, 1.0),
            _clip((player.z - state.z) / (1 << 22), -1.0, 1.0),
            _clip((state.rank - 1) / max(1, len(self.layout.racers) - 1), 0.0, 1.0),
            _clip(
                heading_delta(state.desired_heading, state.current_heading) / 0x800,
                -1.0,
                1.0,
            ),
        )
        return features


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def heading_delta(target: int, current: int) -> int:
    """Return the shortest signed distance on the game's 12-bit heading circle."""

    return ((target - current + 0x800) & (HEADING_PERIOD - 1)) - 0x800


def command_from_action(
    state: RacerState,
    action: Sequence[float],
    *,
    max_turn: int = DEFAULT_MAX_TURN,
    max_target_speed: int = DEFAULT_MAX_TARGET_SPEED,
) -> NPCCommand:
    if not 0 <= max_turn <= 0x800:
        raise ValueError("max_turn must be between 0 and 0x800")
    if not 0 < max_target_speed <= 0xFFFFFFFF:
        raise ValueError("max_target_speed must fit in an unsigned 32-bit value")
    if len(action) != 2:
        raise ValueError("NPC action must contain [steering, throttle]")
    steering, throttle = float(action[0]), float(action[1])
    if not isfinite(steering) or not isfinite(throttle):
        raise ValueError("NPC action values must be finite")
    steering = _clip(steering, -1.0, 1.0)
    throttle = _clip(throttle, 0.0, 1.0)
    desired_heading = (state.current_heading + round(steering * max_turn)) & (
        HEADING_PERIOD - 1
    )
    return NPCCommand(desired_heading, round(throttle * max_target_speed))


def progress_delta(previous: int, current: int, progress_count: int) -> int:
    """Calculate signed checkpoint motion while accounting for lap wraparound."""

    delta = current - previous
    halfway = max(1, progress_count // 2)
    if delta < -halfway:
        delta += progress_count
    elif delta > halfway:
        delta -= progress_count
    return delta


def controlled_vehicle_indices_from_rom(path: Path) -> tuple[int, ...]:
    """Read the selected vehicle mask embedded by ``patch-npc-control``."""

    with path.open("rb") as handle:
        handle.seek(NPC_CONTROL_MARKER_OFFSET)
        marker = handle.read(8)
    if marker[:4] != NPC_CONTROL_MARKER:
        return ()
    if len(marker) != 8 or marker[4] != NPC_CONTROL_PATCH_VERSION:
        raise ValueError("unsupported NPC-control ROM patch marker")
    mask = marker[5]
    if mask & ~0x0E:
        raise ValueError(f"invalid NPC-control vehicle mask: {mask:#x}")
    return tuple(index for index in (1, 2, 3) if mask & (1 << index))


def button_controlled_vehicle_indices_from_rom(path: Path) -> tuple[int, ...]:
    """Read the native-button vehicle mask embedded by ``patch-npc-buttons``."""

    with path.open("rb") as handle:
        handle.seek(NPC_BUTTON_CONTROL_MARKER_OFFSET)
        marker = handle.read(8)
    if marker[:4] != NPC_BUTTON_CONTROL_MARKER:
        return ()
    if len(marker) != 8 or marker[4] != NPC_BUTTON_CONTROL_PATCH_VERSION:
        raise ValueError("unsupported NPC button-control ROM patch marker")
    mask = marker[5]
    if mask != 0x0E:
        raise ValueError(f"invalid NPC button-control vehicle mask: {mask:#x}")
    return tuple(index for index in (1, 2, 3) if mask & (1 << index))
