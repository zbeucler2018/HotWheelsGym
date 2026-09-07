"""ROM identity checks and narrowly scoped experimental patches."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
import struct


ROM_BASE = 0x08000000
EXPECTED_ROM_SIZE = 0x800000
EXPECTED_ROM_SHA1 = "3a950fa7f2979ac8b1e2e2eb628267b3146f1da1"

PLAYER_RACER_CONSTRUCTOR = 0x08101618
CPU_RACER_CONSTRUCTOR = 0x080F5888


@dataclass(frozen=True)
class Patch:
    """A checked ROM byte replacement at a GBA bus address."""

    name: str
    address: int
    expected: bytes
    replacement: bytes

    @property
    def offset(self) -> int:
        return self.address - ROM_BASE


def sha1_bytes(data: bytes) -> str:
    return sha1(data).hexdigest()


def sha1_file(path: Path) -> str:
    digest = sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_rom(path: Path, expected_sha1: str = EXPECTED_ROM_SHA1) -> str:
    size = path.stat().st_size
    if size != EXPECTED_ROM_SIZE:
        raise ValueError(
            f"unexpected ROM size for {path}: 0x{size:x}, expected 0x{EXPECTED_ROM_SIZE:x}"
        )
    actual = sha1_file(path)
    if actual.lower() != expected_sha1.strip().lower():
        raise ValueError(
            f"unexpected ROM SHA-1 for {path}: {actual}, expected {expected_sha1}"
        )
    return actual


def decode_thumb_bl(encoded: bytes, source_address: int) -> int:
    """Decode an ARMv4T two-halfword Thumb BL instruction."""

    if len(encoded) != 4:
        raise ValueError("a Thumb BL is exactly four bytes")
    high, low = struct.unpack("<HH", encoded)
    if high & 0xF800 != 0xF000 or low & 0xF800 != 0xF800:
        raise ValueError(f"not an ARMv4T Thumb BL: {encoded.hex()}")
    raw = ((high & 0x07FF) << 11) | (low & 0x07FF)
    if raw & (1 << 21):
        raw -= 1 << 22
    return source_address + 4 + (raw << 1)


def encode_thumb_bl(source_address: int, target_address: int) -> bytes:
    """Encode an ARMv4T two-halfword Thumb BL instruction."""

    delta = target_address - (source_address + 4)
    if delta & 1:
        raise ValueError("Thumb BL target must be halfword aligned")
    if not -(1 << 22) <= delta <= (1 << 22) - 2:
        raise ValueError("Thumb BL target is out of range")
    raw = (delta >> 1) & ((1 << 22) - 1)
    high = 0xF000 | ((raw >> 11) & 0x07FF)
    low = 0xF800 | (raw & 0x07FF)
    return struct.pack("<HH", high, low)


MIRROR_CPU_PATCHES = (
    Patch(
        "CPU allocation size, path 1",
        0x080FBAA2,
        bytes.fromhex("c021"),  # movs r1, #0xc0; followed by lsls r1, #2
        bytes.fromhex("da21"),  # movs r1, #0xda -> 0x368 bytes after shift
    ),
    Patch(
        "CPU constructor call, path 1",
        0x080FBAB8,
        bytes.fromhex("f9f7e6fe"),
        encode_thumb_bl(0x080FBAB8, PLAYER_RACER_CONSTRUCTOR),
    ),
    Patch(
        "CPU allocation size, path 2",
        0x080FC03A,
        bytes.fromhex("c021"),
        bytes.fromhex("da21"),
    ),
    Patch(
        "CPU constructor call, path 2",
        0x080FC050,
        bytes.fromhex("f9f71afc"),
        encode_thumb_bl(0x080FC050, PLAYER_RACER_CONSTRUCTOR),
    ),
)


def apply_checked_patches(data: bytes, patches: tuple[Patch, ...]) -> bytes:
    output = bytearray(data)
    for patch in patches:
        start = patch.offset
        end = start + len(patch.expected)
        actual = bytes(output[start:end])
        if actual != patch.expected:
            raise ValueError(
                f"{patch.name} mismatch at {patch.address:#010x}: "
                f"found {actual.hex()}, expected {patch.expected.hex()}"
            )
        if len(patch.expected) != len(patch.replacement):
            raise ValueError(f"{patch.name} changes ROM length")
        output[start:end] = patch.replacement
    return bytes(output)


def create_mirror_cpu_rom(source: Path, output: Path, *, force: bool = False) -> str:
    """Create the experimental CPU-to-player-racer mirror-control ROM."""

    if source.resolve() == output.resolve():
        raise ValueError("refusing to overwrite the source ROM")
    if output.exists() and not force:
        raise FileExistsError(f"output already exists: {output} (pass --force to replace it)")
    validate_rom(source)
    patched = apply_checked_patches(source.read_bytes(), MIRROR_CPU_PATCHES)
    output.write_bytes(patched)
    return sha1_bytes(patched)

