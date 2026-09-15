"""Create a private Dino start-line state containing four player-class racers."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

from stable_baselines3 import PPO

from HotWheelsGym.npc_control import RaceMemory

from .common import (
    DEFAULT_CONFIG,
    evaluation_episode_steps,
    load_config,
    make_ram_env,
    prepare_rom,
    validate_model_action_space,
    validate_model_observation_space,
)
from .legacy_policy import adapt_legacy_policy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Finish the stock Dino race, retry it under the native-button ROM, "
            "and save a private four-player starting-line state"
        )
    )
    parser.add_argument("--rom", type=Path, required=True)
    parser.add_argument("--player-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    source_rom = args.rom.expanduser().resolve()
    model_path = args.player_model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if output.exists() and not args.force:
        raise FileExistsError(f"output already exists: {output} (pass --force)")

    config = load_config(args.config.expanduser().resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    active_rom = prepare_rom(source_rom, (1, 2, 3), output.parent)
    loaded_model = PPO.load(model_path, device=args.device)
    model = adapt_legacy_policy(loaded_model)
    if model is loaded_model:
        validate_model_observation_space(model, model_path)
        validate_model_action_space(model, model_path)
    frame_skip = int(config["frame_skip"])
    env = make_ram_env(
        frame_skip=frame_skip,
        max_episode_steps=evaluation_episode_steps(config),
        seed=int(config["seed"]) + 40_000,
        reward_config=config.get("reward"),
    )
    try:
        observation, info = env.reset(seed=int(config["seed"]) + 40_000)
        for decision in range(evaluation_episode_steps(config)):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            if (decision + 1) % 1_000 == 0:
                print(
                    f"decision={decision + 1} "
                    f"completion={float(info.get('ram_player_completion', 0.0)):.1%}",
                    flush=True,
                )
            if terminated or truncated:
                if not bool(info.get("ram_player_finished", False)):
                    raise RuntimeError(
                        "Player model did not finish the source Dino race"
                    )
                break
        else:
            raise RuntimeError("Player model did not finish the source Dino race")

        # The post-race flow accepts A to advance and eventually selects Retry.
        # Pulses are separated so one press cannot carry across menu transitions.
        base = env.unwrapped
        button_index = {name: index for index, name in enumerate(base.buttons) if name}
        advance_pulses = set(range(180, 1_621, 180))
        for frame in range(1, 1_801):
            native_action = [False] * len(base.buttons)
            if frame in advance_pulses:
                native_action[button_index["A"]] = True
            base.step(native_action)
            try:
                race = RaceMemory(base.data.memory)
                states = [race.state(slot) for slot in range(4)]
            except (IndexError, KeyError, RuntimeError, ValueError):
                continue
            four_players = all(racer.kind == "player" for racer in race.layout.racers)
            coordinates_ready = all(
                0 < state.x < 100_000_000 and 0 < state.z < 100_000_000
                for state in states
            )
            if four_players and coordinates_ready:
                with gzip.open(output, "wb") as handle:
                    handle.write(bytes(base.em.get_state()))
                print(
                    f"wrote {output} at retry frame {frame}; "
                    f"ROM={active_rom.name}; racers=4 player,0 CPU",
                    flush=True,
                )
                return
        raise RuntimeError("retry flow did not reach a ready four-player Dino grid")
    finally:
        env.close()


if __name__ == "__main__":
    main()
