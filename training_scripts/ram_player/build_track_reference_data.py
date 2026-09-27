"""Build static non-Dino reference profiles from validated multi-race traces.

The input is deliberately kept outside the repository: it contains only temporary
RAM-coordinate captures.  The generated module contains the median X/Z point for
each documented modulo-checkpoint index plus the native start-state pickup layout.
Missing short runs are linearly interpolated, matching the existing Dino profile
construction convention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


TRACK_PROGRESS_COUNTS = {
    "trex_valley": 316,
    "black_widows_nest": 395,
    "insect_hive": 380,
    "monsters_of_the_deep": 342,
    "whiteskull_cliffs": 340,
    "jungle_snakepit": 465,
    "gator_forest": 512,
    "satellite_mission": 376,
    "solar_strip": 325,
    "fire_mountain": 465,
    "volcano_battle": 495,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--chunk-root", type=Path, required=True)
    parser.add_argument("--pickups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _read_samples(
    track: str, trace_root: Path, chunk_root: Path
) -> dict[str, list[list[int]]]:
    chunk_path = chunk_root / track / "metadata.json"
    if chunk_path.exists():
        return json.loads(chunk_path.read_text())["samples"]
    trace_path = trace_root / f"{track}.json"
    trace = json.loads(trace_path.read_text())
    if "samples" not in trace:
        raise ValueError(
            f"{track} has no full racer trace; recapture it with the resumable tracer"
        )
    return trace["samples"]


def _interpolate(points: list[tuple[int, int] | None]) -> tuple[list[tuple[int, int]], list[int]]:
    count = len(points)
    missing = [index for index, point in enumerate(points) if point is None]
    for index in missing:
        before = next(
            offset
            for offset in range(1, count)
            if points[(index - offset) % count] is not None
        )
        after = next(
            offset
            for offset in range(1, count)
            if points[(index + offset) % count] is not None
        )
        start = points[(index - before) % count]
        end = points[(index + after) % count]
        assert start is not None and end is not None
        fraction = before / (before + after)
        points[index] = (
            round(start[0] + (end[0] - start[0]) * fraction),
            round(start[1] + (end[1] - start[1]) * fraction),
        )
    return [point for point in points if point is not None], missing


def _profile(
    track: str, count: int, samples: dict[str, list[list[int]]], pickups: list[list[int]]
) -> dict[str, object]:
    buckets: list[list[tuple[int, int]]] = [[] for _ in range(count)]
    for racer_samples in samples.values():
        for _frame, progress, x, z, *_ in racer_samples:
            buckets[int(progress) % count].append((int(x), int(z)))
    points: list[tuple[int, int] | None] = [
        (
            round(median(point[0] for point in bucket)),
            round(median(point[1] for point in bucket)),
        )
        if bucket
        else None
        for bucket in buckets
    ]
    centerline, interpolated = _interpolate(points)
    if len(centerline) != count:
        raise RuntimeError(f"{track} centerline has wrong length")
    return {
        "progress_count": count,
        "centerline": centerline,
        "power_up_objects": [tuple(int(value) for value in pickup) for pickup in pickups],
        "interpolated_indices": interpolated,
    }


def main() -> None:
    args = _parser().parse_args()
    pickups = json.loads(args.pickups.read_text())
    profiles = {
        track: _profile(
            track,
            count,
            _read_samples(track, args.trace_root, args.chunk_root),
            pickups[track],
        )
        for track, count in TRACK_PROGRESS_COUNTS.items()
    }
    rendered = (
        '"""Generated non-Dino multi-race reference geometry.\n\n'
        "Created from CPU-racer RAM traces using the documented checkpoints-per-lap.\n"
        "Do not edit by hand; rebuild with build_track_reference_data.py.\n"
        '\"\"\"\n\nfrom __future__ import annotations\n\n'
        f"TRACK_REFERENCE_DATA = {profiles!r}\n"
    )
    args.output.write_text(rendered)


if __name__ == "__main__":
    main()
