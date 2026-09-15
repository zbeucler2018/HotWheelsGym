"""Shared setup for isolated Player 1 RAM training and self-play."""

from __future__ import annotations

import gzip
import json
import os
import platform
import random
import subprocess
import sys
from hashlib import sha1
from pathlib import Path
from typing import Any, Mapping, Sequence

import gymnasium as gym
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import HotWheelsGym
from HotWheelsGym import DinoRAMModelOpponentEnv, DinoRAMPlayerEnv, RAMActionRepeat
from HotWheelsGym.npc_control import button_controlled_vehicle_indices_from_rom
from HotWheelsGym.ram_opponent_control import (
    DINO_RAM_OBSERVATION_SIZE,
    DINO_RAM_OBSERVATION_VERSION,
    DINO_SECTOR_BOUNDARIES,
    RAM_ACTION_COMPONENTS,
    RAM_ACTION_COMPONENT_SIZES,
    RAM_OBSERVATION_NAMES,
)

from .legacy_policy import LegacyV5PolicyAdapter, is_legacy_v5_policy

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("dino_boneyard.yml")
DEFAULT_RUN_ROOT = REPO_ROOT / "training_scripts" / "ram_runs"
ENV_ID = "HWSTC-dino_boneyard-multi-3"

MONITOR_INFO_KEYS = (
    "ram_player_completion",
    "ram_player_finished",
    "ram_player_finish_frame",
    "ram_player_lap",
    "ram_player_rank",
    "ram_player_won_now",
    "ram_player_speed",
    "ram_player_boost",
    "ram_player_power_up_type",
    "ram_player_jet_boost_remaining",
    "ram_player_jet_boost_pickups",
    "ram_player_skid_active",
    "ram_player_hairpin_entries",
    "ram_player_hairpin_completed",
    "ram_player_hairpin_jet_boost_entries",
    "ram_player_hairpin_frames",
    "ram_player_hairpin_completed_frames",
    "ram_player_hairpin_speed_total",
    "ram_player_hairpin_entry_speed_total",
    "ram_player_hairpin_exit_speed_total",
    "ram_player_hairpin_minimum_speed_total",
    "ram_player_hairpin_wall_frames",
    "ram_player_hairpin_skid_frames",
    "ram_player_lap_1_frames",
    "ram_player_lap_2_frames",
    "ram_player_lap_3_frames",
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    required = {
        "run_name",
        "total_timesteps",
        "num_envs",
        "environment",
        "frame_skip",
        "max_episode_steps",
        "opponents",
        "eval_every_timesteps",
        "eval_episodes",
        "checkpoint_every_timesteps",
        "seed",
        "ppo",
    }
    missing = required - set(config)
    if missing:
        raise ValueError("missing configuration fields: " + ", ".join(sorted(missing)))
    if int(config["num_envs"]) < 1:
        raise ValueError("num_envs must be at least one")
    if config["environment"] != ENV_ID:
        raise ValueError(f"the prototype environment must be {ENV_ID}")
    if int(config["total_timesteps"]) < 1:
        raise ValueError("total_timesteps must be at least one")
    if int(config["frame_skip"]) < 1:
        raise ValueError("frame_skip must be at least one")
    if int(config["max_episode_steps"]) < 1:
        raise ValueError("max_episode_steps must be at least one")
    if int(config.get("evaluation_max_episode_steps", config["max_episode_steps"])) < 1:
        raise ValueError("evaluation_max_episode_steps must be at least one")
    if int(config["eval_episodes"]) < 1:
        raise ValueError("eval_episodes must be at least one")
    if int(config.get("league_eval_episodes", config["eval_episodes"])) < 1:
        raise ValueError("league_eval_episodes must be at least one")
    if int(config["eval_every_timesteps"]) < 1:
        raise ValueError("eval_every_timesteps must be at least one")
    if int(config["checkpoint_every_timesteps"]) < 1:
        raise ValueError("checkpoint_every_timesteps must be at least one")
    if not isinstance(config["ppo"], dict):
        raise ValueError("ppo configuration must be a mapping")
    if not isinstance(config["opponents"], dict):
        raise ValueError("opponents must map racer slots to RAM model paths")
    if not isinstance(config.get("opponent_league", {}), dict):
        raise ValueError("opponent_league must map model labels to RAM model paths")
    if config["opponents"] and config.get("opponent_league"):
        raise ValueError("configure fixed opponents or opponent_league, not both")
    if "reward" in config and not isinstance(config["reward"], dict):
        raise ValueError("reward configuration must be a mapping")
    return config


def evaluation_episode_steps(config: Mapping[str, Any]) -> int:
    """Return the evaluation horizon, with legacy-config compatibility."""

    return int(config.get("evaluation_max_episode_steps", config["max_episode_steps"]))


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def file_sha1(path: Path) -> str:
    digest = sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_opponents(raw: Mapping[Any, Any]) -> dict[int, Path]:
    opponents: dict[int, Path] = {}
    for raw_slot, raw_path in raw.items():
        slot = int(raw_slot)
        if slot not in (1, 2, 3):
            raise ValueError("opponent slots must be 1, 2, or 3")
        path = resolve_repo_path(str(raw_path)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        opponents[slot] = path
    return opponents


def normalize_opponent_league(raw: Mapping[Any, Any] | None) -> dict[str, Path]:
    """Resolve a named frozen-policy pool used for balanced slot rotation."""

    league: dict[str, Path] = {}
    for raw_label, raw_path in (raw or {}).items():
        label = str(raw_label).strip()
        if not label:
            raise ValueError("opponent league labels must be non-empty")
        path = resolve_repo_path(str(raw_path)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        league[label] = path
    if league and len(league) < 3:
        raise ValueError("an opponent league requires at least three models")
    return league


def require_all_opponent_slots(opponents: Mapping[int, Any]) -> None:
    """Prevent converted but uncontrolled opponent cars from blocking the track."""

    slots = set(opponents)
    if slots and slots != {1, 2, 3}:
        raise ValueError(
            "native-button self-play must control opponent slots 1, 2, and 3; "
            "reuse the same checkpoint for multiple slots if desired"
        )


def validate_model_observation_space(model: Any, path: str | Path) -> None:
    """Reject older RAM checkpoints with a useful migration message."""

    shape = tuple(getattr(model.observation_space, "shape", ()) or ())
    expected = (DINO_RAM_OBSERVATION_SIZE,)
    if shape != expected:
        legacy_note = ""
        if shape == (43,):
            legacy_note = (
                " This is an observation-v1 checkpoint; v2 added Dino Boneyard "
                "centerline cues, v3 added symmetric boost charge, and v4 adds "
                "the symmetric Jet Boost countdown. V5 added native skid state; "
                "v6 adds factorized action context."
            )
        elif shape == (54,):
            legacy_note = (
                " This is an observation-v2 checkpoint; v3 added each racer's "
                "normalized boost charge and v4 adds the symmetric Jet Boost "
                "countdown. V5 added native skid state and v6 adds factorized "
                "action context. Train v6 from scratch."
            )
        elif shape == (58,):
            legacy_note = (
                " This is an observation-v3 checkpoint; v4 adds each controlled "
                "racer's normalized Jet Boost countdown, v5 added native skid "
                "state, and v6 adds factorized action context. Train v6 from scratch."
            )
        elif shape == (59,):
            legacy_note = (
                " This is an observation-v4 checkpoint; v5 adds each controlled "
                "racer's native skid state and v6 adds factorized action context. "
                "Train v6 from scratch."
            )
        elif shape == (60,):
            legacy_note = (
                " This is an observation-v5 checkpoint; v6 factorizes drive, "
                "steering, and boost so the policy can steer while boosting, and "
                "encodes those action components in the observation. Train v6 "
                "from scratch."
            )
        raise ValueError(
            f"RAM model {path} expects observation shape {shape}, but the active "
            f"contract is {expected}.{legacy_note}"
        )


def validate_model_action_space(model: Any, path: str | Path) -> None:
    """Reject models that do not emit the v6 factorized driving action."""

    nvec = tuple(int(value) for value in getattr(model.action_space, "nvec", ()))
    if nvec != RAM_ACTION_COMPONENT_SIZES:
        legacy_note = ""
        if getattr(model.action_space, "n", None) == 7:
            legacy_note = (
                " This is a v5 seven-action checkpoint; v6 uses independent "
                "drive, steering, and boost components."
            )
        raise ValueError(
            f"RAM model {path} expects action space {model.action_space}, but the "
            f"active contract is MultiDiscrete{RAM_ACTION_COMPONENT_SIZES}."
            f"{legacy_note}"
        )


def prepare_rom(
    source_rom: Path, opponent_slots: tuple[int, ...], private_dir: Path
) -> Path:
    """Import the original ROM or privately patch requested opponent slots."""

    source_rom = source_rom.expanduser().resolve()
    if not source_rom.is_file():
        raise FileNotFoundError(source_rom)
    requested = tuple(sorted(set(opponent_slots)))
    button_controlled = button_controlled_vehicle_indices_from_rom(source_rom)
    if button_controlled and not set(requested).issubset(button_controlled):
        raise ValueError(
            f"{source_rom} controls vehicle indices {button_controlled}, but this run "
            f"requests {requested}"
        )
    if requested and not button_controlled:
        private_dir.mkdir(parents=True, exist_ok=True)
        active_rom = private_dir / "dino-native-buttons.gba"
        tool_source = REPO_ROOT / "hotwheels-re-tools" / "src"
        process_environment = os.environ.copy()
        current_pythonpath = process_environment.get("PYTHONPATH")
        process_environment["PYTHONPATH"] = str(tool_source) + (
            os.pathsep + current_pythonpath if current_pythonpath else ""
        )
        command = [
            sys.executable,
            "-m",
            "hotwheels_re_tools",
            "patch-npc-buttons",
            str(source_rom),
            str(active_rom),
            "--force",
        ]
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=process_environment,
            check=True,
        )
    else:
        active_rom = source_rom
    HotWheelsGym.import_rom(active_rom, force=True)
    return active_rom


def training_state_paths(config: dict[str, Any]) -> list[Path | None]:
    raw_states = config.get("training_states") or [None]
    states: list[Path | None] = []
    for raw_state in raw_states:
        if raw_state is None:
            states.append(None)
            continue
        state = resolve_repo_path(raw_state)
        if not state.is_file():
            raise FileNotFoundError(state)
        states.append(state)
    return states


def opponent_state_path(config: Mapping[str, Any]) -> Path:
    """Resolve the fresh all-player state required by native-button opponents."""

    raw_state = config.get("opponent_state")
    if not raw_state:
        raise ValueError(
            "RAM model opponents require opponent_state: a Dino Boneyard state "
            "created after patch-npc-buttons constructed four player-class racers"
        )
    state = resolve_repo_path(str(raw_state)).resolve()
    if not state.is_file():
        raise FileNotFoundError(state)
    return state


class InitialStatePool(gym.Wrapper):
    """Choose a deterministic per-worker sequence of training states on reset."""

    def __init__(
        self,
        env: gym.Env,
        state_paths: Sequence[str | None],
        *,
        seed: int,
    ) -> None:
        super().__init__(env)
        if not state_paths:
            raise ValueError("initial state pool must not be empty")
        base = env.unwrapped
        default_state = bytes(base.initial_state)
        default_name = str(base.statename)
        self._states: list[tuple[bytes, str]] = []
        for raw_path in state_paths:
            if raw_path is None:
                self._states.append((default_state, default_name))
                continue
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            with gzip.open(path, "rb") as handle:
                self._states.append((handle.read(), str(path)))
        self._random = random.Random(seed)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        if kwargs.get("seed") is not None:
            self._random.seed(int(kwargs["seed"]))
        state, name = self._random.choice(self._states)
        base = self.env.unwrapped
        base.initial_state = state
        base.statename = name
        observation, info = self.env.reset(**kwargs)
        info["ram_initial_state"] = name
        return observation, info


def make_ram_env(
    *,
    frame_skip: int,
    max_episode_steps: int,
    seed: int,
    opponent_paths: Mapping[int, str] | None = None,
    opponent_league: Mapping[str, str] | None = None,
    state_path: str | None = None,
    state_paths: Sequence[str | None] | None = None,
    monitor_path: str | None = None,
    reward_config: Mapping[str, object] | None = None,
) -> gym.Env:
    base = HotWheelsGym.make(ENV_ID, render_mode="rgb_array")
    if opponent_paths and opponent_league:
        raise ValueError("provide fixed opponents or an opponent league, not both")
    if state_path and state_paths:
        raise ValueError("provide state_path or state_paths, not both")
    if state_path:
        state = Path(state_path).expanduser().resolve()
        if not state.is_file():
            raise FileNotFoundError(state)
        with gzip.open(state, "rb") as handle:
            base.unwrapped.initial_state = handle.read()
        base.unwrapped.statename = str(state)
    env: gym.Env = base
    if state_paths:
        env = InitialStatePool(env, state_paths, seed=seed)
    if opponent_paths or opponent_league:
        models: dict[int, Any] = {}
        league_models: dict[str, Any] = {}

        def load_policy(path: str) -> Any:
            model = PPO.load(path, device="cpu")
            if is_legacy_v5_policy(model):
                return LegacyV5PolicyAdapter(model)
            validate_model_observation_space(model, path)
            validate_model_action_space(model, path)
            return model

        for slot, path in (opponent_paths or {}).items():
            models[int(slot)] = load_policy(path)
        for label, path in (opponent_league or {}).items():
            league_models[str(label)] = load_policy(path)
        env = DinoRAMModelOpponentEnv(
            env,
            models,
            opponent_league=league_models,
            seed=seed,
            action_repeat=frame_skip,
        )
    env = DinoRAMPlayerEnv(env, reward_config=reward_config)
    env = RAMActionRepeat(env, repeat=frame_skip)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if monitor_path:
        Path(monitor_path).parent.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=monitor_path, info_keywords=MONITOR_INFO_KEYS)
    return env


def write_run_metadata(
    run_dir: Path,
    config: dict[str, Any],
    config_path: Path,
    source_rom: Path,
    active_rom: Path,
    opponents: Mapping[int, Path],
    opponent_league: Mapping[str, Path] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "resolved_config.yml").open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    metadata = {
        "environment": ENV_ID,
        "config_path": str(config_path.resolve()),
        "git_commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "source_rom_sha1": file_sha1(source_rom),
        "active_rom_sha1": file_sha1(active_rom),
        "active_rom_path": str(active_rom.resolve()),
        "observation_version": DINO_RAM_OBSERVATION_VERSION,
        "observation_names": RAM_OBSERVATION_NAMES,
        "sector_progress_boundaries": DINO_SECTOR_BOUNDARIES,
        "action_space": {
            "type": "MultiDiscrete",
            "nvec": RAM_ACTION_COMPONENT_SIZES,
            "components": {
                component: [
                    {"index": index, "name": action.name, "buttons": action.buttons}
                    for index, action in enumerate(actions)
                ]
                for component, actions in RAM_ACTION_COMPONENTS
            },
        },
        "opponents": {
            str(slot): {"path": str(path), "sha1": file_sha1(path)}
            for slot, path in opponents.items()
        },
        "opponent_league": {
            label: {"path": str(path), "sha1": file_sha1(path)}
            for label, path in (opponent_league or {}).items()
        },
    }
    with (run_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
