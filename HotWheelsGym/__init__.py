import os

import retro

from .enums import RaceMode, Tracks
from .HotWheels import HotWheelsEnv

GAME_NAME = "HotWheelsStuntTrackChallenge-GBAdvance"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

__all__ = ["import_rom", "make", "HotWheelsEnv"]


def import_rom(rom_file_path) -> None:
    """
    Copy the ROM into retro
    """
    # os.system(f"{python_executable} -m retro.import {rom_file_path}")
    return


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
