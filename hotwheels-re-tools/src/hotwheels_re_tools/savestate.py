"""Parse the fixed-size gzip savestates used by the historical integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
from pathlib import Path
import struct
from typing import Any


SAVESTATE_SIZE = 0x61000
EWRAM_OFFSET = 0x21000
EWRAM_SIZE = 0x40000
EWRAM_BASE = 0x02000000

MANAGER_POINTER_LIST_OFFSET = 0x450
RACER_MANAGER_OFFSET = 0x50


@dataclass(frozen=True)
class RacerClass:
    name: str
    vtable: int
    allocation_size: int
    constructor: int


PLAYER_RACER = RacerClass("player", 0x0817C430, 0x368, 0x08101618)
CPU_RACER = RacerClass("cpu", 0x0817B6D4, 0x300, 0x080F5888)
RACER_CLASSES = (PLAYER_RACER, CPU_RACER)


@dataclass(frozen=True)
class Racer:
    slot: int
    kind: str
    address: int
    allocation_size: int
    manager: int
    vehicle_index: int
    progress: int
    speed: int
    cpu_score_be: int | None
    pressed: int | None
    released: int | None
    held: int | None


@dataclass(frozen=True)
class Inspection:
    state_path: str
    payload_size: int
    ewram_payload_offset: int
    pointer_list: int | None
    manager: int | None
    player_count: int | None
    cpu_count: int | None
    total_count: int | None
    racers: tuple[Racer, ...]
    historical_npc_score_owner: str | None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["racers"] = [asdict(racer) for racer in self.racers]
        return result


def load_payload(path: Path) -> bytes:
    with gzip.open(path, "rb") as handle:
        payload = handle.read()
    if len(payload) != SAVESTATE_SIZE:
        raise ValueError(
            f"unexpected savestate payload size for {path}: "
            f"0x{len(payload):x}, expected 0x{SAVESTATE_SIZE:x}"
        )
    return payload


def extract_ewram(payload: bytes) -> bytes:
    if len(payload) != SAVESTATE_SIZE:
        raise ValueError(
            f"unexpected savestate payload size: 0x{len(payload):x}, "
            f"expected 0x{SAVESTATE_SIZE:x}"
        )
    return payload[EWRAM_OFFSET : EWRAM_OFFSET + EWRAM_SIZE]


def _index(address: int, width: int = 1) -> int:
    index = address - EWRAM_BASE
    if index < 0 or index + width > EWRAM_SIZE:
        raise ValueError(f"address outside EWRAM: {address:#010x}")
    return index


def read_u8(ewram: bytes, address: int) -> int:
    return ewram[_index(address)]


def read_u16(ewram: bytes, address: int) -> int:
    return struct.unpack_from("<H", ewram, _index(address, 2))[0]


def read_u32(ewram: bytes, address: int) -> int:
    return struct.unpack_from("<I", ewram, _index(address, 4))[0]


def read_be_u32(ewram: bytes, address: int) -> int:
    return struct.unpack_from(">I", ewram, _index(address, 4))[0]


def find_u32(ewram: bytes, value: int, *, aligned: bool = True) -> list[int]:
    needle = struct.pack("<I", value)
    hits: list[int] = []
    position = 0
    while True:
        position = ewram.find(needle, position)
        if position < 0:
            return hits
        if not aligned or position % 4 == 0:
            hits.append(EWRAM_BASE + position)
        position += 1


def _discover_objects(ewram: bytes) -> list[tuple[int, RacerClass]]:
    objects: list[tuple[int, RacerClass]] = []
    for racer_class in RACER_CLASSES:
        for address in find_u32(ewram, racer_class.vtable):
            objects.append((address, racer_class))
    return sorted(objects)


def _discover_manager(
    ewram: bytes, objects: list[tuple[int, RacerClass]]
) -> tuple[int | None, int | None]:
    if not objects:
        return None, None
    pointer_bytes = b"".join(struct.pack("<I", address) for address, _ in objects)
    pointer_index = ewram.find(pointer_bytes)
    if pointer_index < 0:
        return None, None
    pointer_list = EWRAM_BASE + pointer_index
    for reference in find_u32(ewram, pointer_list):
        manager = reference - MANAGER_POINTER_LIST_OFFSET
        if manager < EWRAM_BASE:
            continue
        try:
            if all(
                read_u32(ewram, address + RACER_MANAGER_OFFSET) == manager
                for address, _ in objects
            ):
                return manager, pointer_list
        except ValueError:
            continue
    return None, pointer_list


def inspect_state(path: Path) -> Inspection:
    payload = load_payload(path)
    ewram = extract_ewram(payload)
    objects = _discover_objects(ewram)
    manager, pointer_list = _discover_manager(ewram, objects)
    racers: list[Racer] = []
    for slot, (address, racer_class) in enumerate(objects):
        is_player = racer_class == PLAYER_RACER
        racers.append(
            Racer(
                slot=slot,
                kind=racer_class.name,
                address=address,
                allocation_size=racer_class.allocation_size,
                manager=read_u32(ewram, address + RACER_MANAGER_OFFSET),
                vehicle_index=read_u8(ewram, address + 0xDC),
                progress=read_u16(ewram, address + 0x148),
                speed=read_u16(ewram, address + 0xE9),
                cpu_score_be=(read_be_u32(ewram, address + 0xEE) if not is_player else None),
                pressed=(read_u16(ewram, address + 0x302) if is_player else None),
                released=(read_u16(ewram, address + 0x304) if is_player else None),
                held=(read_u16(ewram, address + 0x306) if is_player else None),
            )
        )

    historical_address = 0x02007C16
    owner = None
    for racer in racers:
        if racer.address <= historical_address < racer.address + racer.allocation_size:
            owner = (
                f"slot {racer.slot} ({racer.kind}) +"
                f"0x{historical_address - racer.address:X}"
            )
            break

    return Inspection(
        state_path=str(path),
        payload_size=len(payload),
        ewram_payload_offset=EWRAM_OFFSET,
        pointer_list=pointer_list,
        manager=manager,
        player_count=(read_u8(ewram, manager + 0x448) if manager else None),
        cpu_count=(read_u8(ewram, manager + 0x449) if manager else None),
        total_count=(read_u8(ewram, manager + 0x44A) if manager else None),
        racers=tuple(racers),
        historical_npc_score_owner=owner,
    )

