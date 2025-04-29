import argparse

import HotWheelsGym as hw
from wrappers import SaveRandomStates, InteractiveWrapper


def run(save_dir_path: str, rnd_threshold: float):
    tracks = [t for t in hw.Tracks]
    mode = hw.RaceMode.MULTI
    t_laps = 1

    for track in tracks:
        env = hw.HotWheelsEnv(track=track, mode=mode, total_laps=t_laps, render_mode="rgb_array")
        env = SaveRandomStates(env, save_dir_path=save_dir_path, rnd_threshold=rnd_threshold)
        env = InteractiveWrapper(env)
        stopped_early = env.play()
        if stopped_early:
            del env
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play through all tracks and randomly save states")
    parser.add_argument(
        "--save_dir_path", help="Directory to save the random states. (Default: HotWheelsGym/training_scripts/data/states/generated/)", type=str, default="./data/states/generated/"
    )
    parser.add_argument(
        "--rnd_threshold", help="Random threshold to save a state. (Default: 0.01)", type=float, default=0.01
    )
    args = parser.parse_args()
    
    run(save_dir_path=args.save_dir_path, rnd_threshold=args.rnd_threshold)