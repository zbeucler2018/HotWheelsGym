"""Shared setup for isolated Player 1 RAM training and self-play."""

from __future__ import annotations

import gzip
import json
import os
import platform
import subprocess
import sys
from hashlib import sha1
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import HotWheelsGym
from HotWheelsGym import DinoRAMModelOpponentEnv, DinoRAMPlayerEnv, RAMActionRepeat
from HotWheelsGym.npc_control import (
    button_controlled_vehicle_indices_from_rom,
    controlled_vehicle_indices_from_rom,
)
from HotWheelsGym.ram_opponent_control import (
    DINO_RAM_OBSERVATION_SIZE,
    DINO_RAM_OBSERVATION_VERSION,
    RAM_ACTIONS,
    RAM_OBSERVATION_NAMES,
)

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
    "ram_player_speed",
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
        "opponent_max_turn",
        "opponent_max_target_speed",
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
    if int(config["eval_every_timesteps"]) < 1:
        raise ValueError("eval_every_timesteps must be at least one")
    if int(config["checkpoint_every_timesteps"]) < 1:
        raise ValueError("checkpoint_every_timesteps must be at least one")
    if not 0 <= int(config["opponent_max_turn"]) <= 0x800:
        raise ValueError("opponent_max_turn must be between 0 and 0x800")
    if not 0 < int(config["opponent_max_target_speed"]) <= 0xFFFFFFFF:
        raise ValueError("opponent_max_target_speed must fit in unsigned 32 bits")
    if not isinstance(config["ppo"], dict):
        raise ValueError("ppo configuration must be a mapping")
    if not isinstance(config["opponents"], dict):
        raise ValueError("opponents must map CPU slots to RAM model paths")
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


def require_all_opponent_slots(opponents: Mapping[int, Any]) -> None:
    """Prevent converted but uncontrolled opponent cars from blocking the track."""

    slots = set(opponents)
    if slots and slots != {1, 2, 3}:
        raise ValueError(
            "native-button self-play must control opponent slots 1, 2, and 3; "
            "reuse the same checkpoint for multiple slots if desired"
        )


def validate_model_observation_space(model: Any, path: str | Path) -> None:
    """Reject legacy 43-input checkpoints with a useful migration message."""

    shape = tuple(getattr(model.observation_space, "shape", ()) or ())
    expected = (DINO_RAM_OBSERVATION_SIZE,)
    if shape != expected:
        legacy_note = (
            " This appears to be a legacy 43-input checkpoint; observation v2 "
            "adds Dino Boneyard centerline cues and must be trained from scratch."
            if shape == (43,)
            else ""
        )
        raise ValueError(
            f"RAM model {path} expects observation shape {shape}, but the active "
            f"contract is {expected}.{legacy_note}"
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
    controlled = controlled_vehicle_indices_from_rom(source_rom)
    if button_controlled and not set(requested).issubset(button_controlled):
        raise ValueError(
            f"{source_rom} controls vehicle indices {button_controlled}, but this run "
            f"requests {requested}"
        )
    if controlled and controlled != requested:
        raise ValueError(
            f"{source_rom} controls vehicle indices {controlled}, but this run needs "
            f"exactly {requested}; use the original ROM and let the runner patch it"
        )
    if requested and not button_controlled and not controlled:
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


def make_ram_env(
    *,
    frame_skip: int,
    max_episode_steps: int,
    seed: int,
    opponent_paths: Mapping[int, str] | None = None,
    opponent_max_turn: int = 0x200,
    opponent_max_target_speed: int = 0x12000,
    state_path: str | None = None,
    monitor_path: str | None = None,
) -> gym.Env:
    base = HotWheelsGym.make(ENV_ID, render_mode="rgb_array")
    if state_path:
        state = Path(state_path).expanduser().resolve()
        if not state.is_file():
            raise FileNotFoundError(state)
        with gzip.open(state, "rb") as handle:
            base.unwrapped.initial_state = handle.read()
        base.unwrapped.statename = str(state)
    env: gym.Env = base
    if opponent_paths:
        models = {}
        for slot, path in opponent_paths.items():
            model = PPO.load(path, device="cpu")
            validate_model_observation_space(model, path)
            models[int(slot)] = model
        env = DinoRAMModelOpponentEnv(
            env,
            models,
            action_repeat=frame_skip,
            max_turn=opponent_max_turn,
            max_target_speed=opponent_max_target_speed,
        )
    env = DinoRAMPlayerEnv(env)
    env = RAMActionRepeat(env, repeat=frame_skip)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if monitor_path:
        Path(monitor_path).parent.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=monitor_path, info_keywords=MONITOR_INFO_KEYS)
    env.reset(seed=seed)
    return env


def write_run_metadata(
    run_dir: Path,
    config: dict[str, Any],
    config_path: Path,
    source_rom: Path,
    active_rom: Path,
    opponents: Mapping[int, Path],
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
        "actions": [
            {"index": index, "name": action.name, "buttons": action.buttons}
            for index, action in enumerate(RAM_ACTIONS)
        ],
        "opponents": {
            str(slot): {"path": str(path), "sha1": file_sha1(path)}
            for slot, path in opponents.items()
        },
    }
    with (run_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
