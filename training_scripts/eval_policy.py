from pprint import pprint
from dataclasses import dataclass
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from stable_baselines3 import PPO
#from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import pandas as pd

import HotWheelsGym

from tools import Config
from tools.evaluation import evaluate_policy
from tools.wrappers import HotWheelsWrapper, HotWheelsDiscretizer


# def evaluate_agent(
#         model,
#         env,
#         n_episodes: int = 5,
#         render: bool = False,

# ):
#     """
#     Evaluates a model in an env

#     :model      | The model to evaluate
#     :env        | The env to use
#     :n_episodes | Number of episodes to evaluate
#     :render     | Whether to render the evaluations
#     """

    


@dataclass
class Run:
    model_path: str
    config_path: str
    data_path: str

def main(run: Run) -> None:

    model_path = run.model_path
    config_path = run.config_path

    cfg = Config(config_path)
    cfg.env_id = "HWSTC-trex_valley-multi-3"
    #cfg.terminate_on_wall_crash = False
    cfg.training_states = []

    def make_env():
        _env = HotWheelsGym.make(cfg.env_id, render_mode="rgb_array")
        _env = HotWheelsDiscretizer(_env, cfg.action_space)
        _env = HotWheelsWrapper(
            _env,
            frame_skip=cfg.frame_skip,
            frame_skip_prob=cfg.frame_skip_prob,
            use_nature_cnn=cfg.nature_env,
            clip_reward=cfg.nature_env,
            crash_reward=cfg.crash_reward,
            wall_crash_reward=cfg.wall_crash_reward,
            terminate_on_crash=cfg.terminate_on_crash,
            terminate_on_wall_crash=cfg.terminate_on_wall_crash,
            max_episode_steps=cfg.max_episode_steps,
        )
        return Monitor(_env)
    
    venv = VecTransposeImage(
        VecFrameStack(
            DummyVecEnv([make_env]),
            # SubprocVecEnv([make_env] * 10),
            n_stack=cfg.frame_stack)
    )


    model = PPO.load(path=model_path, env=venv)

    num_eps = 2
    output = evaluate_policy(
        model=model,
        env=venv,
        n_eval_episodes=num_eps,
        deterministic=False,
        render=True,
    )

    pprint(output)


    # with open(run.data_path, "w") as f:
    #     f.write("indx,episode_duration,episode_reward\n")
    #     for indx, el in enumerate(ep_lengths):
    #         f.write(f"{indx}, {el}, {rew[indx]}\n")

    venv.close()


if __name__ == "__main__":
    r1 = Run(
        f"{SCRIPT_DIR}/../zoo/dbm_basic/model.zip",
        f"{SCRIPT_DIR}/../zoo/dbm_basic/config.yml",
        f"{SCRIPT_DIR}/short.csv"
    )
    # r2 = Run(
    #     "../zoo/dbm_basic/best_model.zip",
    #     "../zoo/dbm_basic/config.yml",
    #     "./best.csv"
    # )
    runs = [r1]

    for r in runs:
        print(r)
        main(r)

    # for r in runs:
    #     df = pd.read_csv(r.data_path)
    #     print(df.describe())
