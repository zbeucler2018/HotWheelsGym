import os
from statistics import mean

import retro

from .enums import RaceMode, Tracks

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class HotWheelsEnv(retro.RetroEnv):
    """
    RL enviroment for the GBA game 'Hot Wheels Stunt Track Challenge'
    """

    def __init__(
        self,
        track: Tracks = Tracks.Dino_Boneyard,
        mode: RaceMode = RaceMode.MULTI,
        total_laps: int = 3,
        **retro_kwargs,
    ) -> None:
        self.GAME_NAME = "HotWheelsStuntTrackChallenge-GbAdvance"
        self.track = track
        self.mode = mode
        self.total_laps = total_laps
        self._inttype = retro.data.Integrations.ALL
        self._total_steps = 0
        self._target_fps = 60
        self._episode_speeds = []


        # Integrate custom game into stable-retro.
        # retro has a bug that it will only look for
        # the rom in the custom integrations folder
        # immedately after integration and will not after
        retro.data.Integrations.add_custom_path(SCRIPT_DIR)

        if not self.GAME_NAME in retro.data.list_games(self._inttype):
            raise Exception(f"The game was not successfully integrated into retro")

        # Check retro can find the ROM
        try:
            retro.data.get_romfile_path(self.GAME_NAME, self._inttype)
        except FileNotFoundError:
            if not retro.data.get_file_path(self.GAME_NAME, "rom.sha", self._inttype):
                raise

        # init the RetroEnv parent
        # with the correct state and info
        super().__init__(
            game=self.GAME_NAME,
            state=f"{self.track.value}_{self.mode.value}.state",
            info=retro.data.get_file_path(
                self.GAME_NAME,
                f"{self.track.value}_{self.mode.value}.json",
                self._inttype,
            ),
            inttype=self._inttype,
            **retro_kwargs,
        )

    def step(self, action):
        _obs, _rew, _term, _trun, _info = super().step(action)

        self._total_steps += 1
        
        _info["speed"] = int(_info["speed"])

        # Integrated value is 0 for False and 1 for True
        _info["hit_wall"] = bool(_info["hit_wall"])

        # The integrated 'rank' variable maps
        # first place to 0, second to 1, etc.
        if self.mode == RaceMode.MULTI:
            _info["rank"] += 1

        # Terminate the env if the agent reaches the
        # specified lap limit
        if _info["lap"] > self.total_laps:
            _term = True

        # NOTE: Sometimes, stable-retro's code isn't in sync
        # with the emulator. This means that the lua done condition 
        # might not detect immedately
        if _info["lap"] == 4:
            _info["lap"] = self.total_laps

        # Calculate some statistics
        # Average speed
        self._episode_speeds.append(_info["speed"])
        _info["average_speed"] = mean(self._episode_speeds)

        # Race duration
        _info["race_duration_s"] = self._total_steps / self._target_fps

        return _obs, _rew, _term, _trun, _info


    def reset(self, **kwargs):
        self._total_steps = 0
        self._episode_speeds = []
        return super().reset(**kwargs)