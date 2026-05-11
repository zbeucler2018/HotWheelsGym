import HotWheelsGym as hwg
from tools.wrappers import StochasticFrameSkip, LoadRandomTrainingStateWrapper, HotWheelsDiscretizer
from tools.evaluation import evaluate_policy as hw_eval_policy

import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, TimeLimit
from stable_baselines3.ppo import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy as masked_evaluate_policy
import wandb
from wandb.integration.sb3 import WandbCallback
import numpy as np

import os
from dataclasses import dataclass
from typing import List
from pathlib import Path


action_mask = True
logs_dir = f"{os.path.dirname(os.path.abspath(__file__))}/data/logs"
models_dir = f"{os.path.dirname(os.path.abspath(__file__))}/data/models"
best_models_dir = f"{os.path.dirname(os.path.abspath(__file__))}/data/best_models"
gen_t_states = [str(f.absolute()) for f in Path("./data/states/generated/").iterdir() if f.suffix == ".state"]
spec_t_states = [str(f.absolute()) for f in Path("./data/states/specific/").iterdir() if f.suffix == ".state"]
all_t_states = [*gen_t_states, *spec_t_states]
button_combos = [
    [],                     # 0: Coast
    ["A"],                  # 1: Accelerate
    ["B"],                  # 2: Brake
    ["LEFT"],               # 3: Turn left
    ["RIGHT"],              # 4: Turn right
    ["A", "LEFT"],           # 5: Accelerate + left
    ["A", "RIGHT"],          # 6: Accelerate + right
    ["B", "LEFT"],           # 7: Brake + left
    ["B", "RIGHT"],          # 8: Brake + right
    ["L", "R"],              # 9: Boost
    ["L", "R", "LEFT"],      # 10: Boost + left
    ["L", "R", "RIGHT"],     # 11: Boost + right
]

class ActionMaskEnv(gym.Env):
    ...
    def get_action_mask(self):
        boost_ready = self.last_info.get("boost", 0.0) >= 970
        mask = np.ones(self.action_space.n, dtype=bool)
        if not boost_ready:
            mask[9] = mask[10] = mask[11] = False
        return mask


class ActionMasker(gym.Wrapper):
    def __init__(self, env):
        self.BOOST_REQUIRED_ACTIONS = [9, 10, 11]
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_info = info
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_info = info
        return obs, reward, terminated, truncated, info

    def get_action_mask(self):
        boost_ready = self._last_info.get("boost", 0.0) >= 970
        mask = np.ones(self.action_space.n, dtype=bool)
        if not boost_ready:
            mask[self.BOOST_REQUIRED_ACTIONS] = False
        return mask


class RacingRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_progress = None
        self.prev_boost = None
        self.track_checkpoint_map = {
            "trex_valley": 316,
            "dino_boneyard": 342,
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
        self.idle_steps = 0


    def reset(self, **kwargs):
        self._init = False
        obs, info = self.env.reset(**kwargs)
        self.prev_progress = info.get('checkpoint', 0.0) / self.track_checkpoint_map.get(self.env.unwrapped.track.value)
        self.prev_boost = info.get('boost', 0.0) >= 970
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Read necessary info
        progress = info.get('checkpoint', 0.0) / self.track_checkpoint_map.get(self.env.unwrapped.track.value)
        speed = info.get('speed', 0.0)
        crashed = info.get('hit_wall', False)
        rank = info.get('rank', 3) + 1 # 1-4
        can_boost = info.get('boost', 0.0) >= 970

        # Delta progress
        delta_progress = progress - self.prev_progress
        self.prev_progress = progress

        # Normalize speed
        norm_speed = speed / 300.0

        # Rank bonus
        rank_bonus = 0.0
        if rank == 1:
            rank_bonus = 3.0
        elif rank == 2:
            rank_bonus = 1.0

        # Boost usage bonus
        used_boost = 0.0
        if self.prev_boost and not can_boost:
            used_boost = 1.0
        self.prev_boost = can_boost

        # Detect if accelerating (assuming discrete actions mapped by HotWheelsDiscretizer)
        # Update this set to match any actions that involve acceleration
        ACCEL_ACTIONS = {1, 5, 6}  # indexes from button_combos that include "A"
        is_accelerating = action in ACCEL_ACTIONS

        if is_accelerating:
            self.idle_steps = 0
            idle_penalty = 0.0
        else:
            self.idle_steps += 1
            idle_penalty = -0.1 * min(self.idle_steps, 100)


        progress_reward = 5.0 * delta_progress
        speed_reward = 0.05 * norm_speed
        crash_penalty = -25 * (1 if crashed else 0)
        rank_reward = 1 * rank_bonus
        boost_reward = 0.2 * used_boost

        custom_reward = (
            progress_reward + 
            speed_reward +
            crash_penalty + 
            rank_reward + 
            boost_reward + 
            idle_penalty
        )

        print(f"progress_rew: {progress_reward}", f"speed_rew: {speed_reward}", f"crash_penalty: {crash_penalty}", f"rank_reward: {rank_reward}",
             f"boost_reward: {boost_reward}", f"idle_penalty: {idle_penalty}", f"custom_reward: {custom_reward}", f"action: {action} {button_combos[action]}", "", sep="\n")

        # custom_reward = (
        #     5.0 * delta_progress +
        #     0.05 * norm_speed -
        #     15.0 * crash_penalty +
        #     rank_bonus +
        #     2.0 * used_boost +
        #     idle_penalty
        # )

        return obs, custom_reward, terminated, truncated, info


@dataclass
class Config:
    random_training_states: List[str]
    nature_cnn: bool
    max_step_limit: int



def make_env(seed=None):
    def _init():
        env = hwg.make("HWSTC-dino_boneyard-multi-3", render_mode="rgb_array")
        env = LoadRandomTrainingStateWrapper(env, gen_t_states)
        env = HotWheelsDiscretizer(env, combos=button_combos)
        env = StochasticFrameSkip(env, n=4, stickprob=0.25)
        env = ResizeObservation(env, (84, 84))
        env = GrayscaleObservation(env, keep_dim=True)
        env = RacingRewardWrapper(env)
        env = TimeLimit(env, max_episode_steps=5_100)
        env = Monitor(env, info_keywords=("score", "speed", "checkpoint", "rank", "lap", "boost"))
        # env = ActionMaskEnv(env)
        return env
    return _init


def eval():

    venv = SubprocVecEnv([make_env()]*1)
    venv = VecFrameStack(venv, n_stack=4)
    venv = VecTransposeImage(venv)

    model_path = "./model-old-old.zip"
    model = PPO.load(model_path)

    _ = venv.reset()
    
    mean_rwd, std_rew = hw_eval_policy(model, venv, render=True, n_eval_episodes=5)
    print(f"rwd: {mean_rwd} std: {std_rew}")
    del venv



def train():

    wandb_run = wandb.init(
        project="HotWheelsGym",
        #config=cfg,
        sync_tensorboard=True,
        resume=False,
        dir=f"{logs_dir}/wandb",
        mode="disabled"
    )

    num_envs = 32
    venv = SubprocVecEnv([make_env() for i in range(num_envs)])
    venv = VecFrameStack(venv, n_stack=4)
    venv = VecTransposeImage(venv)


    # callbacks = CallbackList([
    #     EvalCallback(
    #         venv,
    #         best_model_save_path=best_models_dir,
    #         log_path=logs_dir,
    #         eval_freq=100_000,  # Evaluate every 250k steps
    #         deterministic=True,
    #         render=False,
    #     ),
    #     WandbCallback(
    #         verbose=1,
    #         model_save_path=models_dir,
    #         model_save_freq=500_000,
    #     )
    # ])

    callbacks = [
        WandbCallback(
            verbose=1,
            model_save_path=models_dir,
            model_save_freq=500_000,
        )
    ]

    if action_mask:
        model = MaskablePPO(
            "CnnPolicy",
            venv,  # SubprocVecEnv
            verbose=1,
            n_steps=128,
            batch_size=512,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            learning_rate=lambda f: 2.5e-4 * f,
            tensorboard_log=logs_dir,
            device="auto"
        )
        callbacks.append(
            MaskableEvalCallback(
                venv,
                use_masking=True,
                best_model_save_path=best_models_dir,
                log_path=logs_dir,
                eval_freq=100_000,  # Evaluate every 250k steps
                deterministic=True,
                render=False,
            )
        )
    else:
        model = PPO(
            "CnnPolicy",
            venv,
            verbose=1,
            n_steps=128,
            batch_size=512, # 32 envs × 128 steps = 4096 total; batch_size=512 is good
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            learning_rate=lambda f: 2.5e-4 * f,
            tensorboard_log=logs_dir,
            device="auto",
        )
        callbacks.append(
            EvalCallback(
                venv,
                best_model_save_path=best_models_dir,
                log_path=logs_dir,
                eval_freq=100_000,  # Evaluate every 250k steps
                deterministic=True,
                render=False,
            ),
        )


    total_timesteps = 30_000_000

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks)
        )
    finally:
        wandb_run.finish()
        venv.close()


if __name__ == "__main__":
    eval()
