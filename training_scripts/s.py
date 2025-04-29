import HotWheelsGym as hw
from tools.wrappers import HotWheelsWrapper, HotWheelsDiscretizer, SpeedReward, SaveRandomStates, InteractiveWrapper, LoadRandomTrainingStateWrapper

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
import retro

import random
from pathlib import Path

from time import sleep
from typing import List
from dataclasses import dataclass

import os

from z import RacingRewardWrapper

gen_t_states = [str(f.absolute()) for f in Path("./data/states/generated/").iterdir() if f.suffix == ".state"]
spec_t_states = [str(f.absolute()) for f in Path("./data/states/specific/").iterdir() if f.suffix == ".state"]

def main():
    env = hw.make("HWSTC-trex_valley-single-3", render_mode="human", use_restricted_actions=retro.Actions.DISCRETE)
    env = LoadRandomTrainingStateWrapper(env, spec_t_states)
    env = RacingRewardWrapper(env)


    observation, info = env.reset(seed=42)
    for i in range(2000):
        # this is where you would insert your policy
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)
        print(i, reward, action, info, env.unwrapped)

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated or i % 125 == 0:
            observation, info = env.reset()
            sleep(5)
            print()

    del env

def vec_main():
    n_envs = 1

    print(os.path.dirname(os.path.abspath(__file__)))
    
    def make_env() -> gym.Env:
        _env = hw.make("HWSTC-trex_valley-single-3", render_mode="human")
        _env = RandomTrainingStateWrapper(_env, training_states=t_states)
        return _env
    
    venv = DummyVecEnv([make_env] * n_envs)

    obs = venv.reset()
    for i in range(120 * 10):

        obs, rewards, dones, info = venv.step([venv.action_space.sample()])
        print(i, info, dones)

        if any(dones) or i % 60 == 0:
            obs = venv.reset()
            sleep(5)
            print()

    del venv

if __name__ == "__main__":
    main()