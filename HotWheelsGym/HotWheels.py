from pprint import pprint

import retro

from .enums import RaceMode, Tracks


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

        if track == Tracks.TRex_Valley and mode == RaceMode.MULTI:
            raise NotImplementedError(f"Can only play SINGLE on the TRex_Valley track")

        # init the RetroEnv parent
        # with the correct state and info
        super().__init__(
            game=self.GAME_NAME,
            state=f"{self.track.value}_{self.mode.value}.state",
            info=retro.data.get_file_path(
                self.GAME_NAME,
                f"{self.track.value}_{self.mode.value}.json",
                retro.data.Integrations.ALL,
            ),
            inttype=retro.data.Integrations.ALL,
            **retro_kwargs,
        )

    def step(self, action):
        _obs, _rew, _term, _trun, _info = super().step(action)

        # Fix the raw integrated speed
        # TODO: Make this much better lol
        _info["speed"] = int(_info["speed"] * 0.702)

        # pprint(_info)

        # Terminate the env if the agent reaches the
        # specified lap limit
        if int(_info["lap"]) > self.total_laps:
            _term = True

        # NOTE: sometimes, stable-retro's code isn't in sync
        # with the emulator. this means that
        # the lua done condition might not detect immedately
        if int(_info["lap"]) == 4:
            _info["lap"] = self.total_laps

        return _obs, _rew, _term, _trun, _info
