"""Compare a Player 1 RAM model with the best Player 1 pixel model."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from torch.utils.tensorboard import SummaryWriter

import HotWheelsGym
from training_scripts.tools.wrappers import HotWheelsDiscretizer, HotWheelsWrapper

from .common import (
    DEFAULT_CONFIG,
    DEFAULT_RUN_ROOT,
    ENV_ID,
    evaluation_episode_steps,
    load_config,
    make_ram_env,
    prepare_rom,
    resolve_repo_path,
    validate_model_action_space,
    validate_model_observation_space,
)
from .media import RGBVideoWriter
from .record import record_model

DEFAULT_PIXEL_MODEL = resolve_repo_path("zoo/dbm_basic/best_model.zip")
DEFAULT_PIXEL_CONFIG = resolve_repo_path("zoo/dbm_basic/config.yml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate RAM and pixel Player 1 models on matched Dino races"
    )
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--ram-model", type=Path, required=True)
    parser.add_argument("--ram-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pixel-model", type=Path, default=DEFAULT_PIXEL_MODEL)
    parser.add_argument("--pixel-config", type=Path, default=DEFAULT_PIXEL_CONFIG)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-record-video",
        action="store_true",
        help="skip MP4 recordings",
    )
    return parser


def _episode_row(
    model_kind: str,
    episode: int,
    reward: float,
    steps: int,
    info: dict[str, Any],
    *,
    is_ram: bool,
    frame_skip: int,
) -> dict[str, Any]:
    if is_ram:
        finished = bool(info.get("ram_player_finished", False))
        completion = float(info.get("ram_player_completion", 0.0))
        raw_frames = int(
            info.get("ram_player_finish_frame")
            or info.get("ram_player_raw_frame")
            or steps * frame_skip
        )
        rank = int(info.get("ram_player_rank", 4))
        speed = float(info.get("ram_player_speed", 0.0))
    else:
        finished = bool(
            int(info.get("lap", 0)) >= 3 and not info.get("TimeLimit.truncated", False)
        )
        lap = int(info.get("lap", 1))
        checkpoint = int(info.get("checkpoint", 0))
        completion = (
            1.0
            if finished
            else min(1.0, max(0.0, ((lap - 1) * 342 + checkpoint) / (3 * 342)))
        )
        raw_frames = round(
            float(info.get("race_duration_s", steps * frame_skip / 60)) * 60
        )
        rank = int(info.get("rank", 4))
        speed = float(info.get("speed", 0.0))
    return {
        "model": model_kind,
        "episode": episode,
        "finished": finished,
        "raw_frames": raw_frames,
        "seconds": raw_frames / 60.0,
        "completion": completion,
        "rank": rank,
        "final_speed": speed,
        "episode_reward": reward,
        "decision_steps": steps,
    }


def _evaluate_ram(
    model_path: Path, config: dict[str, Any], episodes: int, device: str
) -> list[dict[str, Any]]:
    frame_skip = int(config["frame_skip"])
    env = make_ram_env(
        frame_skip=frame_skip,
        max_episode_steps=evaluation_episode_steps(config),
        seed=int(config["seed"]) + 20_000,
        reward_config=config.get("reward"),
    )
    model = PPO.load(model_path, device=device)
    validate_model_observation_space(model, model_path)
    validate_model_action_space(model, model_path)
    rows: list[dict[str, Any]] = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=int(config["seed"]) + episode)
            terminated = truncated = False
            reward_sum = 0.0
            steps = 0
            info: dict[str, Any] = {}
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                reward_sum += float(reward)
                steps += 1
            rows.append(
                _episode_row(
                    "ram",
                    episode,
                    reward_sum,
                    steps,
                    info,
                    is_ram=True,
                    frame_skip=frame_skip,
                )
            )
            row = rows[-1]
            print(
                f"RAM episode {episode + 1}/{episodes}: "
                f"finish={row['finished']} time={row['seconds']:.2f}s "
                f"rank={row['rank']}",
                flush=True,
            )
    finally:
        env.close()
    return rows


def _pixel_env(pixel_config: dict[str, Any], max_episode_steps: int) -> Any:
    def factory() -> Any:
        env = HotWheelsGym.make(ENV_ID, render_mode="rgb_array")
        env = HotWheelsDiscretizer(env, combos=pixel_config["action_space"])
        return HotWheelsWrapper(
            env,
            frame_skip=int(pixel_config["frame_skip"]),
            frame_skip_prob=float(pixel_config["frame_skip_prob"]),
            use_nature_cnn=bool(pixel_config["nature_env"]),
            clip_reward=False,
            crash_reward=0,
            wall_crash_reward=0,
            terminate_on_crash=False,
            terminate_on_wall_crash=False,
            max_episode_steps=max_episode_steps,
        )

    env = VecFrameStack(
        DummyVecEnv([factory]), n_stack=int(pixel_config["frame_stack"])
    )
    return VecTransposeImage(env) if bool(pixel_config["nature_env"]) else env


def _evaluate_pixel(
    model_path: Path,
    config: dict[str, Any],
    episodes: int,
    device: str,
    max_episode_steps: int,
) -> list[dict[str, Any]]:
    frame_skip = int(config["frame_skip"])
    env = _pixel_env(config, max_episode_steps)
    model = PPO.load(model_path, env=env, device=device)
    rows: list[dict[str, Any]] = []
    try:
        for episode in range(episodes):
            observation = env.reset()
            done = False
            reward_sum = 0.0
            steps = 0
            info: dict[str, Any] = {}
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, dones, infos = env.step(action)
                reward_sum += float(reward[0])
                steps += 1
                done = bool(dones[0])
                info = infos[0]
            rows.append(
                _episode_row(
                    "pixel",
                    episode,
                    reward_sum,
                    steps,
                    info,
                    is_ram=False,
                    frame_skip=frame_skip,
                )
            )
            row = rows[-1]
            print(
                f"Pixel episode {episode + 1}/{episodes}: "
                f"finish={row['finished']} time={row['seconds']:.2f}s "
                f"rank={row['rank']}",
                flush=True,
            )
    finally:
        env.close()
    return rows


def _record_pixel(
    model_path: Path,
    config: dict[str, Any],
    device: str,
    max_episode_steps: int,
    video_path: Path,
) -> dict[str, Any]:
    """Record one deterministic-policy pixel race from the start line."""

    frame_skip = int(config["frame_skip"])
    env = _pixel_env(config, max_episode_steps)
    model = PPO.load(model_path, env=env, device=device)
    recorder: RGBVideoWriter | None = None
    try:
        observation = env.reset()
        recorder = RGBVideoWriter(video_path, env.render(), fps=60 / frame_skip)
        done = False
        reward_sum = 0.0
        steps = 0
        info: dict[str, Any] = {}
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, dones, infos = env.step(action)
            reward_sum += float(reward[0])
            steps += 1
            done = bool(dones[0])
            info = infos[0]
            recorder.write(env.render())
    finally:
        if recorder is not None:
            recorder.close()
        env.close()
    row = _episode_row(
        "pixel",
        0,
        reward_sum,
        steps,
        info,
        is_ram=False,
        frame_skip=frame_skip,
    )
    payload = {"model_path": str(model_path), "video": str(video_path), **row}
    with video_path.with_suffix(".json").open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload


def _summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for model_kind in ("ram", "pixel"):
        selected = [row for row in rows if row["model"] == model_kind]
        finishes = [float(row["raw_frames"]) for row in selected if row["finished"]]
        result[model_kind] = {
            "finish_rate": fmean(float(row["finished"]) for row in selected),
            "mean_finish_frames": fmean(finishes) if finishes else 0.0,
            "mean_finish_seconds": fmean(finishes) / 60.0 if finishes else 0.0,
            "mean_completion": fmean(float(row["completion"]) for row in selected),
            "mean_rank": fmean(float(row["rank"]) for row in selected),
        }
    return result


def _summary_markdown(summary: dict[str, dict[str, float]]) -> str:
    lines = [
        "# Matched start-line comparison",
        "",
        "| model | finish rate | mean finish | completion | mean rank |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_kind, metrics in summary.items():
        finish_time = (
            f"{metrics['mean_finish_seconds']:.2f}s" if metrics["finish_rate"] else "—"
        )
        lines.append(
            f"| {model_kind} | {metrics['finish_rate']:.0%} | {finish_time} | "
            f"{metrics['mean_completion']:.1%} | {metrics['mean_rank']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = _parser().parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be at least one")
    ram_model = args.ram_model.expanduser().resolve()
    pixel_model = args.pixel_model.expanduser().resolve()
    if not ram_model.is_file():
        raise FileNotFoundError(ram_model)
    if not pixel_model.is_file():
        raise FileNotFoundError(pixel_model)
    ram_config = load_config(args.ram_config.expanduser().resolve())
    max_episode_steps = (
        args.max_episode_steps
        if args.max_episode_steps is not None
        else evaluation_episode_steps(ram_config)
    )
    if max_episode_steps < 1:
        raise ValueError("max-episode-steps must be at least one")
    ram_config["evaluation_max_episode_steps"] = max_episode_steps
    with args.pixel_config.expanduser().resolve().open() as handle:
        pixel_config = yaml.safe_load(handle)
    if not isinstance(pixel_config, dict):
        raise ValueError("pixel configuration must be a mapping")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root.expanduser().resolve() / f"comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    prepare_rom(args.rom, (), output_dir / "private")

    rows = _evaluate_ram(ram_model, ram_config, args.episodes, args.device)
    rows.extend(
        _evaluate_pixel(
            pixel_model,
            pixel_config,
            args.episodes,
            args.device,
            max_episode_steps,
        )
    )
    summary = _summary(rows)
    with (output_dir / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    videos: dict[str, Any] = {}
    try:
        for model_kind, metrics in summary.items():
            for name, value in metrics.items():
                writer.add_scalar(f"comparison/{model_kind}/{name}", value, 0)
        writer.add_text("comparison/results", _summary_markdown(summary), 0)
        if not args.no_record_video:
            ram_video = output_dir / "ram.mp4"
            videos["ram"] = record_model(
                ram_model, ram_video, ram_config, device=args.device
            )
            pixel_video = output_dir / "pixel.mp4"
            videos["pixel"] = _record_pixel(
                pixel_model,
                pixel_config,
                args.device,
                max_episode_steps,
                pixel_video,
            )
    finally:
        writer.close()

    with (output_dir / "summary.json").open("w") as handle:
        json.dump(
            {
                "ram_model": str(ram_model),
                "pixel_model": str(pixel_model),
                "episodes_per_model": args.episodes,
                "max_episode_steps": max_episode_steps,
                "results": summary,
                "videos": videos,
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    print(f"Comparison written to {output_dir}")
    for model_kind, metrics in summary.items():
        print(
            f"{model_kind:5} finish={metrics['finish_rate']:.0%} "
            f"time={metrics['mean_finish_seconds']:.2f}s "
            f"completion={metrics['mean_completion']:.1%} "
            f"rank={metrics['mean_rank']:.2f}"
        )


if __name__ == "__main__":
    main()
