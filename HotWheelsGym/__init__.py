import os
import shutil

from .enums import RaceMode, Tracks
from .HotWheels import HotWheelsEnv

GAME_NAME = "HotWheelsStuntTrackChallenge-GbAdvance"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

__all__ = ["import_rom", "make", "HotWheelsEnv", GAME_NAME]


def import_rom(path_to_rom) -> None:
    """
    Copy the ROM file into the custom game integration folder
    """
    abs_rom_path = os.path.abspath(path_to_rom)
    assert abs_rom_path.endswith("rom.gba"), "ROM import path must end with rom.gba"

    custom_integration_path = os.path.join(SCRIPT_DIR, GAME_NAME, "rom.gba")

    if not os.path.isfile(custom_integration_path):
        print(f"Copying {abs_rom_path} to {custom_integration_path}")
        shutil.copy(abs_rom_path, custom_integration_path)


def make(id: str, **kwargs) -> HotWheelsEnv:
    """Make a HotWheelsEnv from an ID"""
    try:
        _, track, mode, laps = id.split("-")
    except ValueError:
        print(
            f"Got ID with invalid format {id}. The correct format is HWSTC-<track>-<mode>-<laps>"
        )
        raise

    if Tracks(track) not in Tracks:
        print(f"ERROR! Got invalid track {track}")
        return

    if RaceMode(mode) not in RaceMode:
        print(f"ERROR! Got invalid mode {mode}")
        return

    if int(laps) not in {1, 2, 3}:
        print(f"ERROR! Got invalid lap {laps}")
        return

    return HotWheelsEnv(
        track=Tracks(track),
        mode=RaceMode(mode),
        total_laps=int(laps),
        **kwargs,
    )
