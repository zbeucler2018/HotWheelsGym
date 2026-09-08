"""ROM/state/environment preflight for RAM training; performs no learning."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import (
    DEFAULT_CONFIG,
    load_config,
    make_ram_env,
    normalize_opponents,
    prepare_rom,
    training_state_paths,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight the Dino RAM environment")
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be at least one")

    config = load_config(args.config.expanduser().resolve())
    opponents = normalize_opponents(config["opponents"])
    source_rom = args.rom.expanduser().resolve()
    active_rom = prepare_rom(
        source_rom,
        tuple(opponents),
        Path("training_scripts/ram_runs/preflight_private").resolve(),
    )
    states = training_state_paths(config)
    env = make_ram_env(
        frame_skip=int(config["frame_skip"]),
        max_episode_steps=int(config["max_episode_steps"]),
        seed=int(config["seed"]),
        opponent_paths={slot: str(path) for slot, path in opponents.items()},
        opponent_max_turn=int(config["opponent_max_turn"]),
        opponent_max_target_speed=int(config["opponent_max_target_speed"]),
        state_path=str(states[0]) if states[0] else None,
    )
    try:
        observation, info = env.reset(seed=int(config["seed"]))
        for step in range(args.steps):
            action = (1, 2, 1, 3)[step % 4]
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        if not env.observation_space.contains(observation):
            raise RuntimeError("RAM observation is outside its declared space")
        print(
            f"OK: ROM={active_rom.name} observation={observation.shape} "
            f"actions={env.action_space.n} progress={info['ram_player_progress']} "
            f"rank={info['ram_player_rank']}"
        )
        print("Preflight only: no model was created and no training was started.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
