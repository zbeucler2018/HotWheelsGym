from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import yaml
import HotWheelsGym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

from HotWheelsGym import RaceMode, Tracks

from tools.wrappers import (
    HotWheelsWrapper,
    RandomHotWheelsStateOnReset,
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def parse_state_filename(path: str | Path) -> tuple[Tracks, RaceMode]:
    path = Path(path)
    parts = path.stem.split("_")

    if len(parts) < 3:
        raise ValueError(f"Invalid Hot Wheels state filename: {path}")

    mode_str = parts[-2]
    track_str = "_".join(parts[:-2])

    return Tracks(track_str), RaceMode(mode_str)


def discover_states_by_track(
    *,
    state_dir: Path,
    tracks: list[Tracks],
    mode: RaceMode,
) -> dict[Tracks, list[Path]]:
    state_dir = state_dir.expanduser().resolve()

    states_by_track: dict[Tracks, list[Path]] = defaultdict(list)

    for state_path in sorted(state_dir.glob(f"*_{mode.value}_*.state")):
        track, parsed_mode = parse_state_filename(state_path)

        if parsed_mode != mode:
            continue

        if track not in tracks:
            continue

        states_by_track[track].append(state_path.resolve())

    missing = [track.value for track in tracks if not states_by_track.get(track)]

    if missing:
        raise FileNotFoundError(
            f"Missing {mode.value} states for tracks={missing} in {state_dir}"
        )

    return dict(states_by_track)


class ForceTruncateAfterNSteps(gym.Wrapper):
    def __init__(self, env: gym.Env, max_episode_steps: int):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self._steps = 0

    def reset(self, **kwargs):
        self._steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1

        if self._steps >= self.max_episode_steps:
            truncated = True
            info = dict(info)
            info["TimeLimit.truncated"] = True

        return obs, reward, terminated, truncated, info


def make_env(
    *,
    rank: int,
    track: Tracks,
    mode: RaceMode,
    laps: int,
    state_paths: list[Path],
    cfg: dict[str, Any],
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = HotWheelsGym.make(
            id=f"HWSTC-{track.value}-{mode.value}-{laps}",
            render_mode="rgb_array",
        )

        env = RandomHotWheelsStateOnReset(
            env,
            state_paths=state_paths,
            track=track,
            mode=mode,
            seed=int(cfg.get("seed", 0)) + rank,
            load_on_first_reset=True,
        )

        env = HotWheelsWrapper(
            env,
            action_space=cfg["action_space"],
            crash_reward=cfg["crash_reward"],
            wall_crash_reward=cfg["wall_crash_reward"],
            terminate_on_crash=cfg["terminate_on_crash"],
            terminate_on_wall_crash=cfg["terminate_on_wall_crash"],
            nature_env=cfg["nature_env"],
            frame_skip=cfg.get("frame_skip", 4),
            frame_skip_prob=cfg.get("frame_skip_prob", 0.25),
            trim_obs=cfg.get("trim_obs", False),
            minimap_obs=cfg.get("minimap_obs", False),
        )

        env = ForceTruncateAfterNSteps(
            env,
            max_episode_steps=cfg["max_episode_steps"],
        )

        env = Monitor(env)
        return env

    return _init


def build_vec_env(cfg: dict[str, Any]):
    tracks = [Tracks(t) for t in cfg["tracks"]]
    mode = RaceMode(cfg["mode"])
    laps = int(cfg["laps"])
    num_envs = int(cfg["num_envs"])

    states_by_track = discover_states_by_track(
        state_dir=Path(cfg["state_dir"]),
        tracks=tracks,
        mode=mode,
    )

    env_fns = []

    for rank in range(num_envs):
        track = tracks[rank % len(tracks)]

        env_fns.append(
            make_env(
                rank=rank,
                track=track,
                mode=mode,
                laps=laps,
                state_paths=states_by_track[track],
                cfg=cfg,
            )
        )

    venv = SubprocVecEnv(
        env_fns,
        start_method="spawn",
    )

    if int(cfg.get("frame_stack", 1)) > 1:
        venv = VecFrameStack(
            venv,
            n_stack=int(cfg["frame_stack"]),
        )

    return venv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    run_id = cfg.get("run_id") or "generalized_multi"
    run_dir = Path("data/runs") / run_id
    model_dir = run_dir / "models"
    checkpoint_dir = run_dir / "checkpoints"

    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(args.config, run_dir / "config.yml")

    with (run_dir / "resolved_config.json").open("w") as f:
        json.dump(cfg, f, indent=2)

    venv = build_vec_env(cfg)

    checkpoint_callback = CheckpointCallback(
        save_freq=int(cfg["model_save_freq"]),
        save_path=str(checkpoint_dir),
        name_prefix="ppo_generalized_hotwheels",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    load_model_path = cfg.get("load_model_path")

    if load_model_path:
        model = PPO.load(
            load_model_path,
            env=venv,
            tensorboard_log=str(run_dir / "tb"),
        )
    else:
        model = PPO(
            cfg["policy"],
            venv,
            verbose=1,
            tensorboard_log=str(run_dir / "tb"),
            learning_rate=lambda f: f * float(cfg["learning_rate"]),
            n_steps=int(cfg["n_steps"]),
            batch_size=int(cfg["batch_size"]),
            n_epochs=int(cfg["n_epochs"]),
            gamma=float(cfg["gamma"]),
            gae_lambda=float(cfg["gae_lambda"]),
            clip_range=float(cfg["clip_range"]),
            ent_coef=float(cfg["ent_coef"]),
        )

    try:
        model.learn(
            total_timesteps=int(cfg["total_training_steps"]),
            callback=checkpoint_callback,
            progress_bar=True,
        )

        model.save(model_dir / "final_model")

    finally:
        venv.close()


if __name__ == "__main__":
    main()