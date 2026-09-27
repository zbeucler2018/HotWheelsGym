"""CLI entry point for ``python -m HotWheelsGym.visualizer``."""

from __future__ import annotations

import argparse
from pathlib import Path

import HotWheelsGym

from ..enums import Tracks
from .adapter import ObservationVisualizerAdapter
from .server import serve


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect the canonical 67-D HotWheelsGym RAM observation"
    )
    parser.add_argument(
        "--track",
        choices=[track.value for track in Tracks],
        default=Tracks.Dino_Boneyard.value,
    )
    parser.add_argument(
        "--state",
        type=Path,
        help="optional committed gzip .state; defaults to the track's multi state",
    )
    parser.add_argument("--controlled-slot", type=int, choices=range(4), default=0)
    parser.add_argument(
        "--model",
        type=Path,
        help="optional Stable-Baselines3 PPO checkpoint that drives Player 1",
    )
    parser.add_argument(
        "--model-action-repeat",
        type=int,
        default=4,
        help="raw frames per model decision (default: 4, matching RAM training)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main() -> None:
    args = _parser().parse_args()
    policy = None
    model_path = None
    if args.model is not None:
        from stable_baselines3 import PPO

        model_path = args.model.expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        policy = PPO.load(model_path, device="cpu")
        if tuple(policy.observation_space.shape) != (67,):
            raise ValueError(
                f"model observation shape is {policy.observation_space.shape}; expected (67,)"
            )
        nvec = tuple(int(value) for value in policy.action_space.nvec)
        if nvec != (4, 3, 2):
            raise ValueError(f"model action space is {nvec}; expected (4, 3, 2)")
    env = HotWheelsGym.make(
        f"HWSTC-{args.track}-multi-3", render_mode="rgb_array"
    )
    adapter = ObservationVisualizerAdapter(
        env,
        controlled_slot=args.controlled_slot,
        state_path=args.state,
        policy=policy,
        policy_name=str(model_path) if model_path is not None else None,
        policy_action_repeat=args.model_action_repeat,
    )
    adapter.reset()
    serve(adapter, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
