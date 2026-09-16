"""Counterfactual audit of score/boost-producing button combinations."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

import HotWheelsGym
from HotWheelsGym.npc_control import RaceMemory
from HotWheelsGym.ram_opponent_control import racer_action_buttons

from .common import load_ram_policy, make_ram_env, prepare_rom

BUTTON_TRIALS = (
    ("coast", ()),
    ("accelerate", ("A",)),
    ("brake", ("B",)),
    ("up", ("UP",)),
    ("down", ("DOWN",)),
    ("left", ("LEFT",)),
    ("right", ("RIGHT",)),
    ("left_shoulder", ("L",)),
    ("right_shoulder", ("R",)),
    ("boost", ("L", "R")),
    ("accelerate_up", ("A", "UP")),
    ("accelerate_down", ("A", "DOWN")),
    ("accelerate_left", ("A", "LEFT")),
    ("accelerate_right", ("A", "RIGHT")),
    ("accelerate_left_shoulder", ("A", "L")),
    ("accelerate_right_shoulder", ("A", "R")),
    ("accelerate_boost", ("A", "L", "R")),
    ("up_left_shoulder", ("UP", "L")),
    ("up_right_shoulder", ("UP", "R")),
    ("down_left_shoulder", ("DOWN", "L")),
    ("down_right_shoulder", ("DOWN", "R")),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find a state shortly before the policy gains score, then replay "
            "candidate raw button pulses from the identical emulator state"
        )
    )
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--events", type=int, default=1)
    parser.add_argument("--lookback-decisions", type=int, default=30)
    parser.add_argument("--pulse-frames", type=int, default=30)
    parser.add_argument("--horizon-frames", type=int, default=240)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser


def _button_action(
    button_names: tuple[str, ...], selected: tuple[str, ...]
) -> list[bool]:
    missing = set(selected) - set(button_names)
    if missing:
        raise ValueError(
            "environment is missing buttons: " + ", ".join(sorted(missing))
        )
    enabled = set(selected)
    return [name in enabled for name in button_names]


def _capture_score_event_states(
    *,
    model: Any,
    frame_skip: int,
    max_episode_steps: int,
    seed: int,
    lookback_decisions: int,
    events: int,
) -> list[tuple[bytes, dict[str, Any], list[tuple[int, int, int]]]]:
    env = make_ram_env(
        frame_skip=frame_skip,
        max_episode_steps=max_episode_steps,
        seed=seed,
        reward_config={"mode": "time_trial"},
    )
    ring: deque[tuple[bytes, dict[str, Any], tuple[int, int, int]]] = deque(
        maxlen=lookback_decisions + 1
    )
    captured: list[tuple[bytes, dict[str, Any], list[tuple[int, int, int]]]] = []
    try:
        reset_policy = getattr(model, "reset", None)
        if callable(reset_policy):
            reset_policy()
        observation, info = env.reset(seed=seed)
        cooldown = 0
        for _ in range(max_episode_steps):
            action, _ = model.predict(observation, deterministic=True)
            ring.append(
                (
                    bytes(env.unwrapped.em.get_state()),
                    {
                        "progress": int(info.get("ram_player_progress", 0)),
                        "lap": int(info.get("ram_player_lap", 1)),
                        "speed": int(info.get("ram_player_speed", 0)),
                        "score": int(info.get("ram_player_score", 0)),
                        "boost": int(info.get("ram_player_boost", 0)),
                        "y": int(info.get("ram_player_y", 0)),
                    },
                    tuple(int(value) for value in np.asarray(action).reshape(3)),
                )
            )
            observation, _, terminated, truncated, info = env.step(action)
            score_delta = int(info.get("ram_decision_score_delta", 0))
            if cooldown:
                cooldown -= 1
            elif score_delta > 0 and len(ring) > lookback_decisions:
                state, metadata, _ = ring[0]
                captured.append(
                    (
                        state,
                        {
                            **metadata,
                            "observed_score_delta": score_delta,
                            "event_progress": int(info.get("ram_player_progress", 0)),
                        },
                        [entry[2] for entry in ring],
                    )
                )
                cooldown = lookback_decisions * 2
                if len(captured) >= events:
                    break
            if terminated or truncated:
                break
    finally:
        env.close()
    return captured


def _trial(
    *,
    state: bytes,
    selected: tuple[str, ...],
    pulse_frames: int,
    horizon_frames: int,
    decision_actions: list[tuple[int, int, int]] | None = None,
    frame_skip: int = 4,
) -> dict[str, Any]:
    env = HotWheelsGym.make("HWSTC-dino_boneyard-multi-3", render_mode="rgb_array")
    base = env.unwrapped
    base.initial_state = state
    base.statename = "mechanics-audit.state"
    try:
        observation, info = env.reset()
        race = RaceMemory(base.data.memory)
        initial = race.state(0)
        initial_score = int(info.get("score", 0))
        initial_boost = initial.boost
        min_y = max_y = initial.y
        white_frames = 0
        for frame in range(horizon_frames):
            decision_index = frame // frame_skip
            if decision_actions is not None and decision_index < len(decision_actions):
                buttons = racer_action_buttons(decision_actions[decision_index])
            else:
                buttons = selected if frame < pulse_frames else ("A",)
            action = _button_action(tuple(base.buttons), buttons)
            observation, _, terminated, truncated, info = env.step(action)
            current = race.state(0)
            min_y = min(min_y, current.y)
            max_y = max(max_y, current.y)
            frame_array = np.asarray(observation)
            if frame_array.ndim == 3 and float(frame_array[..., :3].mean()) > 238:
                white_frames += 1
            if terminated or truncated:
                break
        final = race.state(0)
        return {
            "buttons": list(selected),
            "source_decisions": (
                [list(action) for action in decision_actions]
                if decision_actions is not None
                else None
            ),
            "score_delta": int(info.get("score", initial_score)) - initial_score,
            "boost_delta": final.boost - initial_boost,
            "progress_delta": final.progress - initial.progress,
            "speed_delta": final.speed - initial.speed,
            "final_speed": final.speed,
            "y_range": max_y - min_y,
            "respawn_pending": race.respawn_pending(0),
            "white_frames": white_frames,
        }
    finally:
        env.close()


def main() -> None:
    args = _parser().parse_args()
    if args.events < 1 or args.lookback_decisions < 1:
        raise ValueError("events and lookback-decisions must be positive")
    if not 0 < args.pulse_frames <= args.horizon_frames:
        raise ValueError("pulse-frames must be within the audit horizon")
    output_dir = args.output_dir.expanduser().resolve()
    report_path = output_dir / "mechanics_audit.json"
    if report_path.exists() and not args.force:
        raise FileExistsError(f"output already exists: {report_path} (pass --force)")
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_rom(args.rom.expanduser().resolve(), (), output_dir / "rom_import")
    model = load_ram_policy(args.model.expanduser().resolve(), device=args.device)
    events = _capture_score_event_states(
        model=model,
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        lookback_decisions=args.lookback_decisions,
        events=args.events,
    )
    if not events:
        raise RuntimeError("the policy produced no score-increase event to audit")

    report_events: list[dict[str, Any]] = []
    for event_index, (state, metadata, source_actions) in enumerate(events):
        state_path = output_dir / f"score_event_{event_index:02d}.state"
        if state_path.exists() and not args.force:
            raise FileExistsError(f"state already exists: {state_path}")
        with gzip.open(state_path, "wb") as handle:
            handle.write(state)
        trials = {
            "source_policy": _trial(
                state=state,
                selected=(),
                pulse_frames=args.pulse_frames,
                horizon_frames=args.horizon_frames,
                decision_actions=source_actions,
                frame_skip=args.frame_skip,
            ),
            **{
                name: _trial(
                    state=state,
                    selected=buttons,
                    pulse_frames=args.pulse_frames,
                    horizon_frames=args.horizon_frames,
                )
                for name, buttons in BUTTON_TRIALS
            },
        }
        report_events.append(
            {
                "state": str(state_path),
                "metadata": metadata,
                "trials": trials,
            }
        )
    report = {
        "model": str(args.model.expanduser().resolve()),
        "lookback_decisions": args.lookback_decisions,
        "pulse_frames": args.pulse_frames,
        "horizon_frames": args.horizon_frames,
        "events": report_events,
    }
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"mechanics audit saved to {report_path}")
    for event_index, event in enumerate(report_events):
        ranked = sorted(
            event["trials"].items(),
            key=lambda item: (
                item[1]["score_delta"],
                item[1]["boost_delta"],
                item[1]["progress_delta"],
            ),
            reverse=True,
        )
        print(f"event {event_index} top button trials:")
        for name, values in ranked[:5]:
            print(
                f"  {name}: score={values['score_delta']:+d} "
                f"boost={values['boost_delta']:+d} "
                f"progress={values['progress_delta']:+d} "
                f"respawn={values['respawn_pending']}"
            )


if __name__ == "__main__":
    main()
