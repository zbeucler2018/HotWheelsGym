import os
import random
import gzip
from typing import Any, Tuple, List
from pathlib import Path

import retro
from HotWheelsGym import Tracks, RaceMode, GAME_NAME
import numpy as np
import gymnasium as gym
import pygame
from pygame.locals import K_ESCAPE, KEYDOWN, QUIT




def discover_state_names(
    state_dirs: list[str | Path],
    *,
    track: str | None = None,
    mode: str = "multi",
) -> list[str]:
    """Finds .state files and returns stable-retro state names without .state suffix"""
    state_names: list[str] = []

    for state_dir in state_dirs:
        state_dir = Path(state_dir)
        if track is None:
            pattern = f"*_{mode}_*.state"
        else:
            pattern = f"{track}_{mode}_*.state"

        for path in sorted(state_dir.glob(pattern)):
            state_names.append(path.stem)

    if not state_names:
        raise FileNotFoundError(
            f"No states found for track={track!r}, mode={mode!r}, dirs={state_dirs}"
        )

    return state_names



def parse_hotwheels_state_filename(path: str | Path) -> tuple[Tracks, RaceMode]:
    """
    Parse filenames like: trex_valley_single_130.state and gator_forest_multi_358.state

    :param path: the .state file path
    :return: the track and mode encoded in the filename
    :raises ValueError: if the filename is not in the expected format or contains invalid track or mode`
    """
    path = Path(path)
    parts = path.stem.split("_")

    if len(parts) < 3:
        raise ValueError(f"Invalid Hot Wheels state filename: {path}")

    mode_str = parts[-2]
    track_str = "_".join(parts[:-2])

    try:
        track = Tracks(track_str)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse track from state file {path}. "
            f"Parsed track={track_str!r}."
        ) from exc

    try:
        mode = RaceMode(mode_str)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse mode from state file {path}. "
            f"Parsed mode={mode_str!r}."
        ) from exc

    return track, mode


class RandomHotWheelsStateOnReset(gym.Wrapper):
    """
    Randomly loads a Hot Wheels stable-retro .state file after env.reset().

    Safety rule:
        All state files must match the env's fixed track and mode.

    This wrapper intentionally does NOT call env.unwrapped.data.load(...).
    The base env must already have been created with the correct integration,
    e.g.:
        HWSTC-trex_valley-single-1
    or:
        HWSTC-trex_valley-multi-1
    because switching track/mode after construction can make the Lua RAM
    lookup code read the wrong addresses and crash the emulator.
    """

    def __init__(
        self,
        env: gym.Env,
        state_paths: list[str | Path],
        *,
        track: Tracks,
        mode: RaceMode,
        seed: int | None = None,
        load_on_first_reset: bool = True,
    ):
        super().__init__(env)

        if not state_paths:
            raise ValueError("RandomHotWheelsStateOnReset requires at least one state.")

        self.track = track
        self.mode = mode
        self.load_on_first_reset = load_on_first_reset
        self._has_reset_once = False
        self._rng = random.Random(seed)

        self.state_paths = [Path(p).resolve() for p in state_paths]
        self.current_state_path: Path | None = None

        self._validate_state_paths()

    def _validate_state_paths(self) -> None:
        bad_states: list[tuple[Path, Tracks, RaceMode]] = []

        for state_path in self.state_paths:
            parsed_track, parsed_mode = parse_hotwheels_state_filename(state_path)

            if parsed_track != self.track or parsed_mode != self.mode:
                bad_states.append((state_path, parsed_track, parsed_mode))

        if bad_states:
            details = "\n".join(
                f"  {path} -> track={track.value}, mode={mode.value}"
                for path, track, mode in bad_states[:20]
            )

            extra = ""
            if len(bad_states) > 20:
                extra = f"\n  ... and {len(bad_states) - 20} more"

            raise ValueError(
                "RandomHotWheelsStateOnReset received states that do not match "
                f"the env track/mode: expected track={self.track.value}, "
                f"mode={self.mode.value}.\n"
                f"Bad states:\n{details}{extra}"
            )

    def _load_state(self, state_path: Path) -> None:
        self.env.unwrapped.load_state(
            statename=str(state_path),
            inttype=retro.data.Integrations.ALL,
        )

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)

        if not self.load_on_first_reset and not self._has_reset_once:
            self._has_reset_once = True
            return obs, info

        state_path = self._rng.choice(self.state_paths)
        self._load_state(state_path)

        self.current_state_path = state_path
        self._has_reset_once = True

        info["state_path"] = str(state_path)
        info["track"] = self.track.value
        info["mode"] = self.mode.value
        info["random_state_loaded"] = True

        # TODO: Get more recent obs from em after loading state
        return obs, info



def parse_state_filename(fname: str) -> Tuple[Tracks, RaceMode]:
    """Parse a filename `fname` of the format `track_mode_laps.state` and return the track and mode as enums.
    
    :param fname: Filename to parse. Should be in the format of `<track_name>_<mode>_<n_laps>.state`.
    :returns: A tuple of (track, mode) where track is a Tracks enum and mode is a RaceMode enum.
    :raises ValueError: If the filename cannot be parsed or if the track/mode does not exist in the respective enums.
    """
    fname = Path(fname)
    stem = fname.stem
    parts = stem.split("_")
    # laps = parts[-1]
    mode_str = parts[-2]
    track_str = "_".join(parts[:-2])
    try:
        mode = RaceMode(mode_str)
    except ValueError:
        raise ValueError(f"Could not parse mode from {fname}. {mode_str} does not exist")
    try:
        track = Tracks(track_str)
    except ValueError:
        raise ValueError(f"Could not parse track from {fname}. {track_str} does not exist")
    return track, mode


class Legacy_LoadRandomTrainingStateWrapper(gym.Wrapper):
    """Gym wrapper to load a """

    def __init__(self, env: gym.Env, training_state_paths: List[str]):
        super().__init__(env)
        self._init = False
        self._t_states = training_state_paths


    def _load_new_emulator_state(self, state_file_path: str, track: Tracks, mode: RaceMode):
        # print(f"load_state(statename={state_file_path})")
		# load game state into emulator
        self.env.unwrapped.load_state(
			statename=str(state_file_path),
			inttype=retro.data.Integrations.ALL,
		)
        # print("load(file=", f"{track.value.lower()}_{mode.value.lower()}.json", ")", sep="")
		# load the correct track variable file
        self.env.unwrapped.data.load(
			retro.data.get_file_path(
				game=GAME_NAME,
				file=f"{track.value.lower()}_{mode.value.lower()}.json",
				inttype=retro.data.Integrations.ALL
			)
		)

    def reset(self, **kwargs):
        # NOTE: We skip the first reset bc it doesnt happen during training
        if not self._init:
            self._init = True
            return super().reset(**kwargs)
        self._init = True
        # select next random state
        rand_indx = random.randrange(0, len(self._t_states))
        new_state_file = self._t_states[rand_indx]
        track, mode = parse_state_filename(new_state_file) #NOTE: new_state_file must be an absolute path!!!
        # print(f"Randomly reset state to {track} {mode} indx={rand_indx}")
        self._load_new_emulator_state(new_state_file, track, mode)
        return super().reset(**kwargs)


class SaveRandomStates(gym.Wrapper):

    def __init__(self, env, save_dir_path: str, rnd_threshold=0.01):
        super().__init__(env)
        self._rnd_threshold = rnd_threshold
        self._save_dir_path = save_dir_path
        assert os.path.exists(self._save_dir_path), f"{self._save_dir_path} does not exist!"

    def step(self, action):
        ob, rew, terminated, truncated, info = self.env.step(action)

        if random.random() < self._rnd_threshold:
            filepath = f"{self.env.unwrapped.track.value}_{self.env.unwrapped.mode.value}_{info.get('checkpoint', -1)}.state"
            with gzip.open(f"{self._save_dir_path}{filepath}", "wb") as f:
                f.write(self.env.unwrapped.em.get_state())
                print(f"Saving {filepath}")

        return ob, rew, terminated, truncated, info


class InteractiveWrapper(gym.Wrapper):

    def __init__(self, env, fps=60):
        super().__init__(env)
        self.fps = fps
        self.clock = pygame.time.Clock()
        self.buttons = env.unwrapped.buttons if hasattr(env.unwrapped, "buttons") else []
        self.key_map = {
            pygame.K_z: "A",
            pygame.K_x: "B",
            pygame.K_c: "C",
            pygame.K_a: "X",
            pygame.K_s: "Y",
            pygame.K_d: "Z",
            pygame.K_q: "L",
            pygame.K_w: "R",
            pygame.K_UP: "UP",
            pygame.K_DOWN: "DOWN",
            pygame.K_LEFT: "LEFT",
            pygame.K_RIGHT: "RIGHT",
            pygame.K_TAB: "SELECT",
            pygame.K_RETURN: "START",
        }

        self.screen = None
        self.obs_shape = self.observation_space.shape

        pygame.init()
        if len(self.obs_shape) == 3:  # (H, W, C)
            self.screen = pygame.display.set_mode((self.obs_shape[1], self.obs_shape[0]))
        else:
            raise ValueError("InteractiveWrapper only supports visual environments with 3D observations.")

    def keys_to_action(self, pressed_keys):
        keys_down = set()
        for key_code in self.key_map:
            if pressed_keys[key_code]:
                keys_down.add(pygame.key.name(key_code).upper())

        inputs = {
            None: False,
            "BUTTON": "Z" in keys_down,
            "A": "Z" in keys_down,
            "B": "X" in keys_down,
            "C": "C" in keys_down,
            "X": "A" in keys_down,
            "Y": "S" in keys_down,
            "Z": "D" in keys_down,
            "L": "Q" in keys_down,
            "R": "W" in keys_down,
            "UP": "UP" in keys_down,
            "DOWN": "DOWN" in keys_down,
            "LEFT": "LEFT" in keys_down,
            "RIGHT": "RIGHT" in keys_down,
            "MODE": "TAB" in keys_down,
            "SELECT": "TAB" in keys_down,
            "RESET": "ENTER" in keys_down,
            "START": "ENTER" in keys_down,
        }

        return np.array([inputs.get(button, False) for button in self.buttons], dtype=np.int8)

    def play(self) -> bool:
        """Interactively play in the environment"""
        obs, _ = self.reset()
        done = False

        while not done:
            self.clock.tick(self.fps)
            keys = pygame.key.get_pressed()
            action = self.keys_to_action(keys)

            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    pygame.quit()
                    return True

            obs, reward, terminated, truncated, info = self.step(action)
            done = terminated or truncated

            # Convert and render frame
            frame = np.transpose(obs, (1, 0, 2))
            surface = pygame.surfarray.make_surface(frame)
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()

        pygame.quit()
        self.env.close()
        return False