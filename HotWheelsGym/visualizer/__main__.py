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
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main() -> None:
    args = _parser().parse_args()
    env = HotWheelsGym.make(
        f"HWSTC-{args.track}-multi-3", render_mode="rgb_array"
    )
    adapter = ObservationVisualizerAdapter(
        env,
        controlled_slot=args.controlled_slot,
        state_path=args.state,
    )
    adapter.reset()
    serve(adapter, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
