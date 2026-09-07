"""Race Player 1 from the keyboard against trained NPC policies."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

import HotWheelsGym
from tools.wrappers.interactive import RetroInteractive


def _opponent(value: str) -> tuple[int, Path]:
    try:
        raw_slot, raw_path = value.split("=", 1)
        slot = int(raw_slot)
    except ValueError as error:
        raise argparse.ArgumentTypeError("use SLOT=MODEL_PATH") from error
    if slot not in (1, 2, 3):
        raise argparse.ArgumentTypeError("NPC slot must be 1, 2, or 3")
    return slot, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="HWSTC-dino_boneyard-multi-3")
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument(
        "--opponent",
        type=_opponent,
        action="append",
        required=True,
        metavar="SLOT=MODEL_PATH",
    )
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    opponents = dict(args.opponent)
    if len(opponents) != len(args.opponent):
        parser.error("each NPC slot may be specified only once")

    HotWheelsGym.import_rom(args.rom, force=True)
    base_env = HotWheelsGym.make(args.id, render_mode="rgb_array")
    models = {slot: PPO.load(path) for slot, path in opponents.items()}
    env = HotWheelsGym.ModelOpponentEnv(
        base_env, models, deterministic=not args.stochastic
    )
    RetroInteractive(None, None, None, env=env).run()


if __name__ == "__main__":
    main()
