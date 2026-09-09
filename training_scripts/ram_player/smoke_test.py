"""ROM/state/environment preflight for RAM training; performs no learning."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import (
    DEFAULT_CONFIG,
    load_config,
    make_ram_env,
    normalize_opponents,
    opponent_state_path,
    prepare_rom,
    require_all_opponent_slots,
    training_state_paths,
)


def _opponent(value: str) -> tuple[int, str]:
    try:
        raw_slot, raw_path = value.split("=", 1)
        slot = int(raw_slot)
    except ValueError as error:
        raise argparse.ArgumentTypeError("use SLOT=RAM_MODEL_PATH") from error
    if slot not in (1, 2, 3):
        raise argparse.ArgumentTypeError("opponent slot must be 1, 2, or 3")
    return slot, raw_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight the Dino RAM environment")
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument(
        "--opponent",
        action="append",
        type=_opponent,
        metavar="SLOT=RAM_MODEL_PATH",
        help="replace configured self-play opponents; repeat for more slots",
    )
    parser.add_argument(
        "--opponent-state",
        type=Path,
        help="fresh Dino state containing four player-class racers",
    )
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be at least one")

    config = load_config(args.config.expanduser().resolve())
    if args.opponent is not None:
        if len(dict(args.opponent)) != len(args.opponent):
            raise ValueError("each self-play opponent slot may appear only once")
        config["opponents"] = dict(args.opponent)
    if args.opponent_state is not None:
        config["opponent_state"] = str(args.opponent_state.expanduser().resolve())
    opponents = normalize_opponents(config["opponents"])
    require_all_opponent_slots(opponents)
    source_rom = args.rom.expanduser().resolve()
    active_rom = prepare_rom(
        source_rom,
        tuple(opponents),
        Path("training_scripts/ram_runs/preflight_private").resolve(),
    )
    states = (
        [opponent_state_path(config)] if opponents else training_state_paths(config)
    )
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
        for slot in opponents:
            print(
                f"NPC slot {slot}: "
                f"completion={float(info[f'ram_npc_{slot}_completion']):.1%} "
                f"rank={int(info[f'ram_npc_{slot}_rank'])} "
                f"held={int(info[f'ram_npc_{slot}_buttons_held']):#05x}"
            )
        print("Preflight only: no model was created and no training was started.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
