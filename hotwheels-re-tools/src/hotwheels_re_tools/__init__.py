"""Utilities for Hot Wheels Stunt Track Challenge reverse engineering."""

from .rom import EXPECTED_ROM_SHA1, EXPECTED_ROM_SIZE
from .savestate import EWRAM_OFFSET, EWRAM_SIZE, SAVESTATE_SIZE

__all__ = [
    "EXPECTED_ROM_SHA1",
    "EXPECTED_ROM_SIZE",
    "EWRAM_OFFSET",
    "EWRAM_SIZE",
    "SAVESTATE_SIZE",
]

__version__ = "0.1.0"

