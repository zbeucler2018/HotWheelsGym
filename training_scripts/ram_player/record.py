"""Record a deterministic Player 1 RAM-policy race to MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from stable_baselines3 import PPO

from .common import (
    DEFAULT_CONFIG,
    ENV_ID,
    evaluation_episode_steps,
    load_config,
    make_ram_env,
    normalize_opponents,
    opponent_state_path,
    prepare_rom,
    require_all_opponent_slots,
    validate_model_observation_space,
)
from .media import RGBVideoWriter


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
    require_all_opponent_slots(opponent_paths or {})
    video_path.parent.mkdir(parents=True, exist_ok=True)
    frame_skip = int(config["frame_skip"])
    env = make_ram_env(
        frame_skip=frame_skip,
        max_episode_steps=evaluation_episode_steps(config),
        seed=int(config["seed"]) + 30_000,
        opponent_paths=opponent_paths,
        state_path=(str(opponent_state_path(config)) if opponent_paths else None),
    )
    model = PPO.load(model_path, device=device)
    validate_model_observation_space(model, model_path)
    observation, info = env.reset(seed=int(config["seed"]) + 30_000)
    encoder = RGBVideoWriter(video_path, env.render(), fps=60 / frame_skip)
    terminated = truncated = False
    reward_sum = 0.0
    decision_steps = 0
    observed_frames = 0
    boost_charge_total = 0.0
    boost_spent = 0
    boost_gained = 0
    boost_active_frames = 0
    jet_boost_remaining_total = 0.0
    jet_boost_active_frames = 0
    jet_boost_pickups = 0
    skid_active_frames = 0
    try:
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            decision_steps += 1
            frames = int(info.get("ram_action_repeat_frames", frame_skip))
            observed_frames += frames
            boost_charge_total += (
                float(info.get("ram_decision_mean_boost_charge", 0.0)) * frames
            )
            boost_spent += int(info.get("ram_decision_boost_spent", 0))
            boost_gained += int(info.get("ram_decision_boost_gained", 0))
            boost_active_frames += int(info.get("ram_decision_boost_frames", 0))
            jet_boost_remaining_total += (
                float(info.get("ram_decision_mean_jet_boost_remaining", 0.0)) * frames
            )
            jet_boost_active_frames += int(info.get("ram_decision_jet_boost_frames", 0))
            jet_boost_pickups += int(info.get("ram_decision_jet_boost_pickups", 0))
            skid_active_frames += int(info.get("ram_decision_skid_frames", 0))
            encoder.write(env.render())
    finally:
        encoder.close()
        env.close()

    result = {
        "environment": ENV_ID,
        "model": str(model_path),
        "video": str(video_path),
        "decision_steps": decision_steps,
        "raw_frames": int(
            info.get("ram_player_raw_frame", decision_steps * frame_skip)
        ),
        "reward": reward_sum,
        "finished": bool(info.get("ram_player_finished", False)),
        "completion": float(info.get("ram_player_completion", 0.0)),
        "rank": int(info.get("ram_player_rank", 4)),
        "final_boost": int(info.get("ram_player_boost", 0)),
        "mean_boost_charge": boost_charge_total / max(1, observed_frames),
        "boost_spent": boost_spent,
        "boost_gained": boost_gained,
        "boost_active_frames": boost_active_frames,
        "boost_active_rate": boost_active_frames / max(1, observed_frames),
        "final_power_up_type": int(info.get("ram_player_power_up_type", 0xFF)),
        "final_jet_boost_remaining": int(info.get("ram_player_jet_boost_remaining", 0)),
        "mean_jet_boost_remaining": (
            jet_boost_remaining_total / max(1, observed_frames)
        ),
        "jet_boost_active_frames": jet_boost_active_frames,
        "jet_boost_active_rate": (jet_boost_active_frames / max(1, observed_frames)),
        "jet_boost_pickups": jet_boost_pickups,
        "skid_active_frames": skid_active_frames,
        "skid_active_rate": skid_active_frames / max(1, observed_frames),
        "hairpin": {
            "entries": int(info.get("ram_player_hairpin_entries", 0)),
            "completed": int(info.get("ram_player_hairpin_completed", 0)),
            "jet_boost_entries": int(
                info.get("ram_player_hairpin_jet_boost_entries", 0)
            ),
            "frames": int(info.get("ram_player_hairpin_frames", 0)),
            "completed_frames": int(info.get("ram_player_hairpin_completed_frames", 0)),
            "speed_total": int(info.get("ram_player_hairpin_speed_total", 0)),
            "entry_speed_total": int(
                info.get("ram_player_hairpin_entry_speed_total", 0)
            ),
            "exit_speed_total": int(info.get("ram_player_hairpin_exit_speed_total", 0)),
            "minimum_speed_total": int(
                info.get("ram_player_hairpin_minimum_speed_total", 0)
            ),
            "wall_frames": int(info.get("ram_player_hairpin_wall_frames", 0)),
            "skid_frames": int(info.get("ram_player_hairpin_skid_frames", 0)),
        },
        "lap_split_frames": [
            int(info.get(f"ram_player_lap_{lap}_frames", 0)) for lap in (1, 2, 3)
        ],
        "lap_split_seconds": [
            int(info.get(f"ram_player_lap_{lap}_frames", 0)) / 60.0 for lap in (1, 2, 3)
        ],
        "lateral_offset": float(info.get("ram_player_lateral_offset", 0.0)),
        "heading_alignment": float(info.get("ram_player_heading_alignment", 0.0)),
    }
    if opponent_paths:
        result["masked_opponent_respawn_flash_frames"] = int(
            info.get("ram_opponent_respawn_flash_frames", 0)
        )
        result["opponents"] = {
            str(slot): {
                "finished": bool(info.get(f"ram_npc_{slot}_finished", False)),
                "finish_frame": info.get(f"ram_npc_{slot}_finish_frame"),
                "completion": float(info.get(f"ram_npc_{slot}_completion", 0.0)),
                "lap": int(info.get(f"ram_npc_{slot}_lap", 1)),
                "rank": int(info.get(f"ram_npc_{slot}_rank", 4)),
                "lateral_offset": float(
                    info.get(f"ram_npc_{slot}_lateral_offset", 0.0)
                ),
                "heading_alignment": float(
                    info.get(f"ram_npc_{slot}_heading_alignment", 0.0)
                ),
                "speed": int(info.get(f"ram_npc_{slot}_speed", 0)),
                "boost": int(info.get(f"ram_npc_{slot}_boost", 0)),
                "power_up_type": int(info.get(f"ram_npc_{slot}_power_up_type", 0xFF)),
                "jet_boost_remaining": int(
                    info.get(f"ram_npc_{slot}_jet_boost_remaining", 0)
                ),
                "jet_boost_pickups": int(
                    info.get(f"ram_npc_{slot}_jet_boost_pickups", 0)
                ),
                "skid_active": bool(
                    info.get(f"ram_npc_{slot}_skid_active", False)
                ),
                "lap_split_frames": [
                    int(info.get(f"ram_npc_{slot}_lap_{lap}_frames", 0))
                    for lap in (1, 2, 3)
                ],
                "action": int(info.get(f"ram_npc_{slot}_action", 0)),
            }
            for slot in sorted(opponent_paths)
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
    parser.add_argument(
        "--opponent-state",
        type=Path,
        help="fresh Dino state containing four player-class racers",
    )
    args = parser.parse_args()
    config = load_config(args.config.expanduser().resolve())
    if args.opponent_state is not None:
        config["opponent_state"] = str(args.opponent_state.expanduser().resolve())
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
