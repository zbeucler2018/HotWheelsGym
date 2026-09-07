"""Small wrapper around GNU ARM objdump for raw GBA ROM ranges."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from .rom import ROM_BASE, validate_rom


def disassemble_thumb(
    rom: Path,
    start: int,
    stop: int,
    *,
    objdump: str = "arm-none-eabi-objdump",
) -> str:
    if start < ROM_BASE or stop <= start:
        raise ValueError("use an increasing GBA ROM address range starting at 0x08000000")
    validate_rom(rom)
    executable = shutil.which(objdump)
    if executable is None:
        raise FileNotFoundError(
            f"{objdump!r} was not found; install binutils-arm-none-eabi or pass --objdump"
        )
    command = [
        executable,
        "-D",
        "-b",
        "binary",
        "-marm",
        "-Mforce-thumb",
        f"--adjust-vma={ROM_BASE:#x}",
        f"--start-address={start:#x}",
        f"--stop-address={stop:#x}",
        str(rom),
    ]
    return subprocess.check_output(command, text=True)

