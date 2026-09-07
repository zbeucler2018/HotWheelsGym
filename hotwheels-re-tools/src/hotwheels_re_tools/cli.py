"""Command-line interface for the reverse-engineering utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .disassemble import disassemble_thumb
from .rom import (
    EXTERNAL_CPU_HEADING_PATCHES,
    EXPECTED_ROM_SHA1,
    MIRROR_CPU_PATCHES,
    Patch,
    create_external_cpu_heading_rom,
    create_mirror_cpu_rom,
    decode_thumb_bl,
    validate_rom,
)
from .savestate import inspect_state


def _integer(value: str) -> int:
    return int(value, 0)


def _hex_or_none(value: int | None) -> str:
    return "unknown" if value is None else f"{value:#010x}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hwre")
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify-rom", help="validate ROM size and SHA-1")
    verify.add_argument("rom", type=Path)

    state = commands.add_parser("inspect-state", help="inspect racers in a gzip state")
    state.add_argument("state", type=Path)
    state.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    disassemble = commands.add_parser("disassemble", help="disassemble a Thumb ROM range")
    disassemble.add_argument("rom", type=Path)
    disassemble.add_argument("start", type=_integer)
    disassemble.add_argument("stop", type=_integer)
    disassemble.add_argument("--objdump", default="arm-none-eabi-objdump")

    patch = commands.add_parser(
        "patch-mirror-cpu", help="create the experimental mirror-control ROM"
    )
    patch.add_argument("rom", type=Path)
    patch.add_argument("output", type=Path)
    patch.add_argument("--force", action="store_true")
    patch.add_argument(
        "--dry-run", action="store_true", help="verify and display patches without writing"
    )

    heading_patch = commands.add_parser(
        "patch-external-cpu-heading",
        help="let an external controller own each CPU racer's desired heading",
    )
    heading_patch.add_argument("rom", type=Path)
    heading_patch.add_argument("output", type=Path)
    heading_patch.add_argument("--force", action="store_true")
    heading_patch.add_argument(
        "--dry-run", action="store_true", help="verify and display patches without writing"
    )
    return parser


def _print_state(path: Path, as_json: bool) -> None:
    result = inspect_state(path)
    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
        return
    print(f"state: {result.state_path}")
    print(f"payload: {result.payload_size:#x}")
    print(f"EWRAM payload offset: {result.ewram_payload_offset:#x}")
    print(
        "header: "
        f"version={result.version_magic:#010x} bios={result.bios_checksum:#010x} "
        f"rom_crc32={result.rom_crc32:#010x}"
    )
    print(
        f"cpu: pc={result.pc:#010x} cpsr={result.cpsr:#010x} "
        f"spsr={result.spsr:#010x}"
    )
    print(f"manager: {_hex_or_none(result.manager)}")
    print(f"racer pointer list: {_hex_or_none(result.pointer_list)}")
    print(
        "counts: "
        f"player={result.player_count} cpu={result.cpu_count} total={result.total_count}"
    )
    for racer in result.racers:
        details = (
            f"slot {racer.slot}: {racer.kind:6} address={racer.address:#010x} "
            f"vehicle={racer.vehicle_index} progress={racer.progress} speed={racer.speed}"
        )
        if racer.cpu_score_be is not None:
            details += (
                f" score_be={racer.cpu_score_be} heading={racer.current_heading:#05x}"
                f" desired={racer.desired_heading:#05x} speed_fixed={racer.speed_fixed}"
                f" target_speed={racer.target_speed}"
            )
        else:
            details += (
                f" pressed={racer.pressed:#05x} released={racer.released:#05x}"
                f" held={racer.held:#05x}"
            )
        print(details)
    print(f"historical 0x02007C16 owner: {result.historical_npc_score_owner}")


def _print_patch_plan(patches: tuple[Patch, ...]) -> None:
    for patch in patches:
        detail = ""
        if len(patch.replacement) == 4 and patch.replacement[1] & 0xF8 == 0xF0:
            detail = f" -> {decode_thumb_bl(patch.replacement, patch.address):#010x}"
        print(
            f"{patch.address:#010x}: {patch.expected.hex()} -> "
            f"{patch.replacement.hex()}  {patch.name}{detail}"
        )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "verify-rom":
            digest = validate_rom(args.rom)
            print(f"OK {args.rom}: sha1={digest} size=0x{args.rom.stat().st_size:x}")
            print(f"expected sha1={EXPECTED_ROM_SHA1}")
        elif args.command == "inspect-state":
            _print_state(args.state, args.json)
        elif args.command == "disassemble":
            print(
                disassemble_thumb(
                    args.rom, args.start, args.stop, objdump=args.objdump
                ),
                end="",
            )
        elif args.command == "patch-mirror-cpu":
            validate_rom(args.rom)
            _print_patch_plan(MIRROR_CPU_PATCHES)
            if args.dry_run:
                print("dry run: ROM validated; no output written")
            else:
                digest = create_mirror_cpu_rom(args.rom, args.output, force=args.force)
                print(f"wrote {args.output} (sha1={digest})")
                print("keep this generated ROM ignored and uncommitted")
        elif args.command == "patch-external-cpu-heading":
            validate_rom(args.rom)
            _print_patch_plan(EXTERNAL_CPU_HEADING_PATCHES)
            if args.dry_run:
                print("dry run: ROM validated; no output written")
            else:
                digest = create_external_cpu_heading_rom(
                    args.rom, args.output, force=args.force
                )
                print(f"wrote {args.output} (sha1={digest})")
                print("keep this generated ROM ignored and uncommitted")
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
