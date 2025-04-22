from pprint import pprint
from dataclasses import dataclass
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

import HotWheelsGym

from tools import Config
from tools.evaluation import evaluate_policy
from tools.wrappers import HotWheelsWrapper, HotWheelsDiscretizer, SpeedReward



@dataclass
class Run:
    model_path: str
    config_path: str

def main(run: Run) -> None:

    model_path = run.model_path
    config_path = run.config_path

    cfg = Config(config_path)


    def make_env() -> gym.Env:
        """Util to create envs for vec envs"""
        _env = HotWheelsGym.make(
            id=cfg.env_id,
            render_mode="rgb_array",
        )

        if cfg.action_space:  # apply custom action space
            _env = HotWheelsDiscretizer(_env, combos=cfg.action_space)
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
        # if cfg.use_speed_reward:
        #     _env = SpeedReward(_env)
        _env = Monitor(_env)
        return _env
    
    venv = VecTransposeImage(
        VecFrameStack(SubprocVecEnv([make_env]*6), n_stack=cfg.frame_stack)
    )

    model = PPO.load(path=model_path, env=venv)

    num_eps = 6
    output = evaluate_policy(
        model=model,
        env=venv,
        n_eval_episodes=num_eps,
        deterministic=True,
        render=True,
    )

    pprint(output)

    venv.close()

if __name__ == "__main__":
    r1 = Run(
        f"{SCRIPT_DIR}/../zoo/tvm_basic/best_model.zip",
        f"{SCRIPT_DIR}/../zoo/tvm_basic/config.yml",
    )
    main(r1)
