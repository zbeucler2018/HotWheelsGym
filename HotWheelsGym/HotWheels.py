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

        self.prev_progress = 0
        self.prev_lap = 0
        self.prev_score = 0

        if track == Tracks.TRex_Valley and mode == RaceMode.MULTI:
            raise NotImplementedError(f"Can only play SINGLE on the TRex_Valley track")

        # # Specific info file (?)
        # if not retro_kwargs["info"]:
        #     retro_kwargs["info"] = retro.data.get_file_path(
        #         game=self.GAME_NAME,
        #         file=f"{self.track}_{self.mode}.json",
        #         inttype=retro.data.Integrations.ALL,
        #     )

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

        # Fix the integrated speed
        # TODO: Make this much better lol
        _info["speed"] *= 0.702

        # After completeing a full race (3 laps), the integrated
        # progress, lap, and score variables get set to obsure
        # values. Set those values to their last good values
        # when the full race is done
        if self.total_laps == 11:  # end of race
            _info["progress"] = self.prev_progress
            _info["lap"] = 3
            _info["score"] = self.prev_score

        else:
            self.prev_progress = _info["progress"]
            self.prev_lap = _info["lap"]
            self.prev_score = _info["score"]

            # Terminate the env if the agent reaches the
            # specified lap limit
            if int(_info["lap"]) > self.total_laps:
                _term = True

        return _obs, _rew, _term, _trun, _info

    def reset(self, **kwargs):
        self.prev_progress = 0
        self.prev_lap = 0
        self.prev_score = 0
        return super().reset(**kwargs)
