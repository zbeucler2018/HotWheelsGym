"""Record a deterministic Player 1 RAM-policy race to MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import imageio_ffmpeg
import numpy as np
from stable_baselines3 import PPO

from .common import (
    DEFAULT_CONFIG,
    ENV_ID,
    load_config,
    make_ram_env,
    normalize_opponents,
    prepare_rom,
)


def record_model(
    model_path: Path,
    video_path: Path,
    config: dict[str, Any],
    *,
    opponent_paths: Mapping[int, str] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Record one start-line race at real-time playback speed."""

    model_path = model_path.expanduser().resolve()
    video_path = video_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    frame_skip = int(config["frame_skip"])
    env = make_ram_env(
        frame_skip=frame_skip,
        max_episode_steps=int(config["max_episode_steps"]),
        seed=int(config["seed"]) + 30_000,
        opponent_paths=opponent_paths,
        opponent_max_turn=int(config["opponent_max_turn"]),
        opponent_max_target_speed=int(config["opponent_max_target_speed"]),
    )
    model = PPO.load(model_path, device=device)
    observation, info = env.reset(seed=int(config["seed"]) + 30_000)
    first_frame = np.asarray(env.render(), dtype=np.uint8)
    if first_frame.ndim != 3 or first_frame.shape[2] != 3:
        env.close()
        raise RuntimeError(f"unexpected RGB frame shape: {first_frame.shape}")
    height, width = first_frame.shape[:2]
    encoder = imageio_ffmpeg.write_frames(
        str(video_path),
        (width, height),
        fps=60 / frame_skip,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="warning",
    )
    encoder.send(None)
    encoder.send(np.ascontiguousarray(first_frame))
    terminated = truncated = False
    reward_sum = 0.0
    decision_steps = 0
    try:
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            decision_steps += 1
            encoder.send(
                np.ascontiguousarray(np.asarray(env.render(), dtype=np.uint8))
            )
    finally:
        encoder.close()
        env.close()

    result = {
        "environment": ENV_ID,
        "model": str(model_path),
        "video": str(video_path),
        "decision_steps": decision_steps,
        "raw_frames": int(info.get("ram_player_raw_frame", decision_steps * frame_skip)),
        "reward": reward_sum,
        "finished": bool(info.get("ram_player_finished", False)),
        "completion": float(info.get("ram_player_completion", 0.0)),
        "rank": int(info.get("ram_player_rank", 4)),
    }
    with video_path.with_suffix(".json").open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a Dino RAM model to MP4")
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config(args.config.expanduser().resolve())
    opponents = normalize_opponents(config["opponents"])
    prepare_rom(args.rom, tuple(opponents), args.output.parent / "private")
    result = record_model(
        args.model,
        args.output,
        config,
        opponent_paths={slot: str(path) for slot, path in opponents.items()},
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
