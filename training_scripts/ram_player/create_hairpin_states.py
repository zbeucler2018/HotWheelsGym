"""Capture private native Dino hairpin states with and without Jet Boost."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from HotWheelsGym.dino_boneyard_track import DINO_TRACK_OBSERVATION_SIZE
from HotWheelsGym.ram_opponent_control import (
    DINO_BONEYARD_PROGRESS_COUNT,
    DINO_HAIRPIN_START_PROGRESS,
    DINO_RAM_OBSERVATION_SIZE,
    RAM_ACTION_COMPONENT_SIZES,
)

from .common import (
    DEFAULT_CONFIG,
    evaluation_episode_steps,
    load_config,
    make_ram_env,
    prepare_rom,
    validate_model_action_space,
    validate_model_observation_space,
)

LEGACY_V5_OBSERVATION_SIZE = 60
LEGACY_V5_ACTION_SIZE = 7
LEGACY_V5_ACTIONS = (
    (0, 0, 0),  # coast
    (1, 0, 0),  # accelerate
    (1, 1, 0),  # accelerate left
    (1, 2, 0),  # accelerate right
    (2, 0, 0),  # brake
    (3, 0, 0),  # accelerate + up
    (1, 0, 1),  # accelerate + L+R boost
)
V6_ACTION_HISTORY_START = 15 + DINO_TRACK_OBSERVATION_SIZE
V6_OTHER_RACERS_START = V6_ACTION_HISTORY_START + sum(RAM_ACTION_COMPONENT_SIZES)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a RAM policy and capture native pre-hairpin states from one "
            "attempt with Jet Boost and one without it"
        )
    )
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--player-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--target-progress", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser


def _legacy_v5_observation(observation: np.ndarray, previous_action: int) -> np.ndarray:
    action_history = np.zeros(LEGACY_V5_ACTION_SIZE, dtype=np.float32)
    action_history[previous_action] = 1.0
    return np.concatenate(
        (
            observation[:V6_ACTION_HISTORY_START],
            action_history,
            observation[V6_OTHER_RACERS_START:],
        )
    ).astype(np.float32, copy=False)


def _model_kind(model: Any, path: Path) -> str:
    shape = tuple(getattr(model.observation_space, "shape", ()) or ())
    if shape == (DINO_RAM_OBSERVATION_SIZE,):
        validate_model_observation_space(model, path)
        validate_model_action_space(model, path)
        return "v6"
    if shape == (LEGACY_V5_OBSERVATION_SIZE,) and getattr(
        model.action_space, "n", None
    ) == len(LEGACY_V5_ACTIONS):
        return "v5"
    raise ValueError(
        f"hairpin state capture requires a v5 or v6 RAM model, got "
        f"observation {shape} and action space {model.action_space}"
    )


def _write_state(path: Path, state: bytes, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"output already exists: {path} (pass --force)")
    with gzip.open(path, "wb") as handle:
        handle.write(state)


def main() -> None:
    args = _parser().parse_args()
    source_rom = args.rom.expanduser().resolve()
    model_path = args.player_model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    target = int(args.target_progress) % DINO_BONEYARD_PROGRESS_COUNT
    if not 0 <= target < DINO_HAIRPIN_START_PROGRESS:
        raise ValueError(
            f"target progress must precede the hairpin at "
            f"{DINO_HAIRPIN_START_PROGRESS}"
        )
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_rom(source_rom, (), output_dir / "rom_import")
    config = load_config(args.config.expanduser().resolve())
    model = PPO.load(model_path, device=args.device)
    model_kind = _model_kind(model, model_path)
    env = make_ram_env(
        frame_skip=int(config["frame_skip"]),
        max_episode_steps=evaluation_episode_steps(config),
        seed=int(config["seed"]) + 50_000,
        reward_config=config.get("reward"),
    )
    captures: dict[str, tuple[bytes, int, int]] = {}
    previous_legacy_action = 0
    try:
        observation, info = env.reset(seed=int(config["seed"]) + 50_000)
        for _ in range(evaluation_episode_steps(config)):
            if model_kind == "v5":
                legacy_observation = _legacy_v5_observation(
                    observation, previous_legacy_action
                )
                legacy_action, _ = model.predict(legacy_observation, deterministic=True)
                previous_legacy_action = int(np.asarray(legacy_action).item())
                action = LEGACY_V5_ACTIONS[previous_legacy_action]
            else:
                action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            progress = int(info["ram_player_progress"]) % DINO_BONEYARD_PROGRESS_COUNT
            timer = int(info["ram_player_jet_boost_remaining"])
            if target <= progress < DINO_HAIRPIN_START_PROGRESS:
                condition = "with_jet" if timer > 0 else "no_jet"
                captures.setdefault(
                    condition,
                    (bytes(env.unwrapped.em.get_state()), progress, timer),
                )
            if len(captures) == 2:
                break
            if terminated or truncated:
                break
    finally:
        env.close()

    missing = {"no_jet", "with_jet"} - set(captures)
    if missing:
        raise RuntimeError(
            f"model did not naturally produce: {', '.join(sorted(missing))}; "
            "use a checkpoint known to both collect and miss Jet Boost"
        )
    no_jet_state, no_jet_progress, _ = captures["no_jet"]
    with_jet_state, with_jet_progress, with_jet_timer = captures["with_jet"]
    no_jet_path = output_dir / "dino_hairpin_no_jet.state"
    with_jet_path = output_dir / "dino_hairpin_with_jet.state"
    _write_state(no_jet_path, no_jet_state, force=args.force)
    _write_state(with_jet_path, with_jet_state, force=args.force)
    print(
        f"wrote native curriculum states from {model_kind}:\n"
        f"  no Jet Boost: {no_jet_path} (progress {no_jet_progress})\n"
        f"  with Jet Boost: {with_jet_path} "
        f"(progress {with_jet_progress}, timer {with_jet_timer})",
        flush=True,
    )


if __name__ == "__main__":
    main()
