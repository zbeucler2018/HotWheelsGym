"""Train PPO against one native CPU racer slot."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import HotWheelsGym
from tools.wrappers.action import StochasticFrameSkip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="HWSTC-dino_boneyard-multi-3")
    parser.add_argument("--slot", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--rom", type=Path, help="generated selected-NPC ROM to import")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("npc_ppo"))
    args = parser.parse_args()

    if args.rom:
        HotWheelsGym.import_rom(args.rom, force=True)
    env = HotWheelsGym.make_npc(args.id, npc_slot=args.slot, render_mode="rgb_array")
    if args.frame_skip > 1:
        env = StochasticFrameSkip(env, args.frame_skip, 0.25)
    env = Monitor(env)
    try:
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=args.timesteps)
        model.save(args.output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
