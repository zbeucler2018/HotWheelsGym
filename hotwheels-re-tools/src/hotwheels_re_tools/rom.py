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

NPC_CONTROL_SITE = 0x080F7116
NPC_CONTROL_TRAMPOLINE = 0x080EC538
NPC_CONTROL_HOOK = 0x0879BEE0
NPC_CONTROL_RETURN = 0x080F6DDE
NPC_STOCK_HEADING_SHADOW_OFFSET = 0x2EE
NPC_CONTROL_MARKER = b"HWNP"
NPC_CONTROL_PATCH_VERSION = 1
NPC_CONTROL_ROM_SHA1S = {
    (1,): "c41ce6f267acad9320df0cf3b6bf3a8c43e3e595",
    (2,): "52a1c6b07219081a2939ef6e9a97d7656245930c",
    (3,): "88cc62fba58b6e01092265d1964d555a0312d35a",
    (1, 2): "8d52a2c33e1ac9bb73e799aa400ab371ae70ece1",
    (1, 3): "567bca6472aa1b88d4041674ef8e42833a43b992",
    (2, 3): "ca8ee8d5033d4b654c84e49db8b265592808f565",
    (1, 2, 3): "2a05fd722c8f77b4d4b9ea3e7ff16c527d638cec",
}


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


EXTERNAL_CPU_HEADING_PATCHES = (
    Patch(
        "stock CPU desired-heading write",
        0x080F7116,
        bytes.fromhex("2880"),  # strh r0, [r5] where r5 == racer + 0xe0
        bytes.fromhex("c046"),  # nop; an external controller owns racer + 0xe0
    ),
)


def _halfwords(*values: int) -> bytes:
    return struct.pack(f"<{len(values)}H", *values)


def _npc_control_hook(control_mask: int) -> bytes:
    """Build the Thumb hook that filters the stock heading write by vehicle index."""

    # r0 contains the stock AI heading, r5 points at racer + 0xE0, and r7 is the
    # CPU racer object. Registers r1-r3 are overwritten at the return target.
    code = _halfwords(
        0x1C3B,  # adds r3, r7, #0
        0x33DC,  # adds r3, #0xdc
        0x781B,  # ldrb r3, [r3]
        0x2201,  # movs r2, #1
        0x409A,  # lsls r2, r2, r3
        0x2100 | control_mask,  # movs r1, #control_mask
        0x400A,  # ands r2, r1
        0xD005,  # beq stock_write
        0x1C3B,  # adds r3, r7, #0
        0x33FF,  # adds r3, #0xff
        0x33FF,  # adds r3, #0xff
        0x33F0,  # adds r3, #0xf0 -> racer + 0x2ee
        0x8018,  # strh r0, [r3] (stock-heading observation shadow)
        0xE000,  # b continue
        0x8028,  # stock_write: strh r0, [r5]
        0x4B01,  # continue: ldr r3, [pc, #4]
        0x4718,  # bx r3
        0x46C0,  # nop (align literal)
    )
    marker = NPC_CONTROL_MARKER + bytes(
        (NPC_CONTROL_PATCH_VERSION, control_mask, 0, 0)
    )
    return code + struct.pack("<I", NPC_CONTROL_RETURN | 1) + marker


def npc_control_patches(vehicle_indices: tuple[int, ...]) -> tuple[Patch, ...]:
    """Return checked patches for externally controlled CPU vehicle indices."""

    selected = tuple(sorted(set(vehicle_indices)))
    if not selected:
        raise ValueError("select at least one CPU vehicle index")
    if any(index not in (1, 2, 3) for index in selected):
        raise ValueError("CPU vehicle indices must be in the range 1..3")
    control_mask = sum(1 << index for index in selected)
    hook = _npc_control_hook(control_mask)
    return (
        Patch(
            "CPU heading store and branch hook",
            NPC_CONTROL_SITE,
            bytes.fromhex("288061e6"),  # strh r0, [r5]; b 0x080f6dde
            encode_thumb_bl(NPC_CONTROL_SITE, NPC_CONTROL_TRAMPOLINE),
        ),
        Patch(
            "NPC-control Thumb trampoline",
            NPC_CONTROL_TRAMPOLINE,
            bytes(8),
            _halfwords(0x4B00, 0x4718)
            + struct.pack("<I", NPC_CONTROL_HOOK | 1),
        ),
        Patch(
            "selected-NPC control hook",
            NPC_CONTROL_HOOK,
            bytes(len(hook)),
            hook,
        ),
    )


def controlled_vehicle_indices(data: bytes) -> tuple[int, ...]:
    """Read the selected-NPC marker from a generated ROM, if present."""

    hook = _npc_control_hook(0)
    marker_offset = NPC_CONTROL_HOOK - ROM_BASE + len(hook) - 8
    marker = data[marker_offset : marker_offset + 8]
    if marker[:4] != NPC_CONTROL_MARKER:
        return ()
    if len(marker) != 8 or marker[4] != NPC_CONTROL_PATCH_VERSION:
        raise ValueError("unsupported NPC-control ROM patch marker")
    mask = marker[5]
    if mask & ~0x0E:
        raise ValueError(f"invalid NPC-control vehicle mask: {mask:#x}")
    return tuple(index for index in (1, 2, 3) if mask & (1 << index))


def validate_supported_rom(path: Path) -> tuple[str, tuple[int, ...]]:
    """Validate either the original ROM or a known selected-NPC output."""

    size = path.stat().st_size
    if size != EXPECTED_ROM_SIZE:
        raise ValueError(
            f"unexpected ROM size for {path}: 0x{size:x}, expected 0x{EXPECTED_ROM_SIZE:x}"
        )
    data = path.read_bytes()
    digest = sha1_bytes(data)
    if digest == EXPECTED_ROM_SHA1:
        return digest, ()
    selected = controlled_vehicle_indices(data)
    expected = NPC_CONTROL_ROM_SHA1S.get(selected)
    if expected != digest:
        raise ValueError(f"unsupported ROM SHA-1 for {path}: {digest}")
    return digest, selected


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


def create_external_cpu_heading_rom(
    source: Path, output: Path, *, force: bool = False
) -> str:
    """Create a ROM where an external controller owns CPU racer ``+0xE0``."""

    if source.resolve() == output.resolve():
        raise ValueError("refusing to overwrite the source ROM")
    if output.exists() and not force:
        raise FileExistsError(f"output already exists: {output} (pass --force to replace it)")
    validate_rom(source)
    patched = apply_checked_patches(source.read_bytes(), EXTERNAL_CPU_HEADING_PATCHES)
    output.write_bytes(patched)
    return sha1_bytes(patched)


def create_npc_control_rom(
    source: Path,
    output: Path,
    vehicle_indices: tuple[int, ...],
    *,
    force: bool = False,
) -> str:
    """Create a ROM where selected native CPU racers accept external commands."""

    if source.resolve() == output.resolve():
        raise ValueError("refusing to overwrite the source ROM")
    if output.exists() and not force:
        raise FileExistsError(f"output already exists: {output} (pass --force to replace it)")
    validate_rom(source)
    patched = apply_checked_patches(source.read_bytes(), npc_control_patches(vehicle_indices))
    output.write_bytes(patched)
    return sha1_bytes(patched)
