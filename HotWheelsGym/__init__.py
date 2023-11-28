from .HotWheelsEnv import HotWheelsEnv
import os
import retro
from .enums import Tracks, RaceMode

def make(track: Tracks, mode: RaceMode, total_laps=3, **kwargs):
	"""
	1. integrate the rom
	2. return the HotWheelsEnv
	"""
	SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
	retro.data.Integrations.add_custom_path(SCRIPT_DIR)
	print("Game exists: ", "HotWheelsStuntTrackChallenge-GBAdvance" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
	
	return HotWheelsEnv(track, mode, total_laps, **kwargs)