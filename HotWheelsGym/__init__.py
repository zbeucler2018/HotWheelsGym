import os
import shutil

import retro

from .enums import RaceMode, Tracks
from .HotWheels import HotWheelsEnv

GAME_NAME = "HotWheelsStuntTrackChallenge-GbAdvance"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

__all__ = ["import_rom", "HotWheelsEnv", GAME_NAME]


def import_rom(path_to_rom) -> None:
    """
    Copy the ROM file into the custom game integration folder
    """
    _custom_integration_path = os.path.join(SCRIPT_DIR, GAME_NAME, "rom.gba")

    abs_rom_path = os.path.abspath(path_to_rom)
    assert abs_rom_path.endswith("rom.gba"), "ROM import path must end with rom.gba"

    if not os.path.isfile(_custom_integration_path):
        print(f"Copying {abs_rom_path} to {_custom_integration_path}")
        shutil.copy(abs_rom_path, _custom_integration_path)
