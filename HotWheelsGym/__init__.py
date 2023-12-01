import os
import shutil

import retro

from .enums import RaceMode, Tracks
from .HotWheels import HotWheelsEnv

GAME_NAME = "HotWheelsStuntTrackChallenge-GBAdvance"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

__all__ = ["import_rom", "make", "HotWheelsEnv"]


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


def make(track: Tracks, mode: RaceMode, total_laps: int = 3, **kwargs):
    """
    Creates a HotWheelsEnv for the given track and race mode.
    """

    # 1. Integrate the game
    retro.data.Integrations.add_custom_path(SCRIPT_DIR)
    _inttype = retro.data.Integrations.ALL

    # 2. Ensure valid integration
    if not GAME_NAME in retro.data.list_games(_inttype):
        raise Exception(f"The game was not successfully integrated into retro")

    # 3. Check the ROM is valid
    try:
        retro.data.get_romfile_path(GAME_NAME, _inttype)
    except FileNotFoundError:
        if not retro.data.get_file_path(GAME_NAME, "rom.sha", _inttype):
            raise

    # 4. Return the HotWheelsEnv
    return HotWheelsEnv(track, mode, total_laps, **kwargs)
