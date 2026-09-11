"""ROM-independent discovery and telemetry for active race objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Mapping, Protocol

EWRAM_BASE = 0x02000000
EWRAM_SIZE = 0x40000

PLAYER_RACER_VTABLE = 0x0817C430
CPU_RACER_VTABLE = 0x0817B6D4
RACER_MANAGER_OFFSET = 0x50
RACER_VEHICLE_INDEX_OFFSET = 0xDC
RACER_CURRENT_HEADING_OFFSET = 0xDE
RACER_SPEED_OFFSET = 0xE8
RACER_BOOST_OFFSET = 0xF0
RACER_X_OFFSET = 0xF8
RACER_Z_OFFSET = 0x100
RACER_PROGRESS_OFFSET = 0x148
RACER_POWER_UP_TYPE_OFFSET = 0x14D
RACER_POWER_UP_TIMER_OFFSET = 0x14E
RACER_RESPAWN_PENDING_OFFSET = 0x26B
RACER_PRESSED_OFFSET = 0x302
RACER_RELEASED_OFFSET = 0x304
RACER_HELD_OFFSET = 0x306

MANAGER_PLAYER_COUNT_OFFSET = 0x448
MANAGER_CPU_COUNT_OFFSET = 0x449
MANAGER_TOTAL_COUNT_OFFSET = 0x44A
MANAGER_POINTER_LIST_OFFSET = 0x450

NPC_BUTTON_CONTROL_MARKER_OFFSET = 0x79BEF4
NPC_BUTTON_CONTROL_MARKER = b"HWBT"
NPC_BUTTON_CONTROL_PATCH_VERSION = 3

HEADING_PERIOD = 0x1000
MAX_BOOST_CHARGE = 980
JET_BOOST_POWER_UP_TYPE = 3
MAX_JET_BOOST_TIMER = 150


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
    speed: int
    progress: int
    x: int
    z: int
    boost: int = 0
    power_up_type: int = 0xFF
    power_up_timer: int = 0

    @property
    def jet_boost_remaining(self) -> int:
        """Return the active Jet Boost countdown, or zero for another power-up."""

        if self.power_up_type != JET_BOOST_POWER_UP_TYPE:
            return 0
        return min(self.power_up_timer, MAX_JET_BOOST_TIMER)


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
    """Read racer telemetry through a discovered race layout."""

    def __init__(self, memory: MemoryView):
        self.memory = memory
        self.layout = discover_race_layout(memory)

    def state(self, slot: int) -> RacerState:
        racer = self.layout.racer(slot)
        return RacerState(
            slot=slot,
            vehicle_index=racer.vehicle_index,
            current_heading=_read_u16(
                self.memory, racer.address + RACER_CURRENT_HEADING_OFFSET
            )
            & (HEADING_PERIOD - 1),
            speed=_read_u32(self.memory, racer.address + RACER_SPEED_OFFSET),
            boost=_read_u32(self.memory, racer.address + RACER_BOOST_OFFSET),
            progress=_read_u16(self.memory, racer.address + RACER_PROGRESS_OFFSET),
            x=_signed_u32(_read_u32(self.memory, racer.address + RACER_X_OFFSET)),
            z=_signed_u32(_read_u32(self.memory, racer.address + RACER_Z_OFFSET)),
            power_up_type=_read_u8(
                self.memory, racer.address + RACER_POWER_UP_TYPE_OFFSET
            ),
            power_up_timer=_read_u8(
                self.memory, racer.address + RACER_POWER_UP_TIMER_OFFSET
            ),
        )

    def respawn_pending(self, slot: int) -> bool:
        """Return whether a player-class racer has entered its respawn sequence."""

        racer = self.layout.racer(slot)
        if racer.kind != "player":
            return False
        return bool(_read_u8(self.memory, racer.address + RACER_RESPAWN_PENDING_OFFSET))


def heading_delta(target: int, current: int) -> int:
    """Return the shortest signed distance on the game's 12-bit heading circle."""

    return ((target - current + 0x800) & (HEADING_PERIOD - 1)) - 0x800


def progress_delta(previous: int, current: int, progress_count: int) -> int:
    """Calculate signed checkpoint motion while accounting for lap wraparound."""

    delta = current - previous
    halfway = max(1, progress_count // 2)
    if delta < -halfway:
        delta += progress_count
    elif delta > halfway:
        delta -= progress_count
    return delta


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
