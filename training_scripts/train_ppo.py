import argparse
import os
from pathlib import Path

import gymnasium as gym
import HotWheelsGym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (SubprocVecEnv, VecFrameStack,
                                              VecTransposeImage)

from tools import Config, get_random_name
from tools.evaluation import EvalCallback
from tools.wrappers import HotWheelsDiscretizer, HotWheelsWrapper

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = f"{SCRIPT_DIR}/data"
LOG_DIR = f"{DATA_DIR}/logs"
MODEl_SAVE_PATH = f"{DATA_DIR}/models"
BEST_MODEL_SAVE_PATH = f"{DATA_DIR}/best_models"

INFO_VARS = ("score", "speed", "progress")


def train(cfg: Config) -> None:
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
        _env = Monitor(_env, info_keywords=INFO_VARS)
        return _env

    # Vectorize env
    venv = VecTransposeImage(
        VecFrameStack(SubprocVecEnv([make_env] * cfg.num_envs), n_stack=cfg.frame_stack)
    )

    # Apply training states
    if cfg.training_states:
        # Need to change state AFTER adding SubProcVec because
        # stable-retro will throw "1 Emulator per process only" exception
        # if applied before
        for indx, t_state in enumerate(cfg.training_states):
            _ = venv.env_method(
                method_name="load_state",
                statename=str(Path(t_state).absolute()),
                indices=indx,
            )
            _ = venv.env_method(method_name="reset_emulator_data", indices=indx)
        _ = venv.reset()

    # create / load model
    if cfg.load_model_path:
        model = PPO.load(path=cfg.load_model_path, env=venv)
    else:
        model = PPO(
            policy=cfg.policy,
            env=venv,
            learning_rate=lambda f: f * 2.5e-4,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            verbose=1,
            tensorboard_log=f"{LOG_DIR}/tf/{cfg.run_id}",
        )

    # Callbacks
    _callbacks = []

    eval_callback = EvalCallback(
        venv,
        best_model_save_path=f"{BEST_MODEL_SAVE_PATH}/{cfg.run_id}",
        log_path=f"{DATA_DIR}/evaluation/{cfg.run_id}",
        eval_freq=cfg.eval_freq,
        eval_state_path=str(Path(cfg.eval_state_path).absolute()),
        deterministic=True,
        render=cfg.render_eval,
    )
    _callbacks.append(eval_callback)

    if cfg.use_wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        _run = wandb.init(
            project="HotWheelsGym",
            config=cfg,
            sync_tensorboard=True,
            resume=True if cfg.load_model_path else None,
            id=cfg.run_id,
            dir=f"{LOG_DIR}/wandb",
        )

        wandb_callback = WandbCallback(
            # gradient_save_freq=50_000,
            model_save_path=f"{MODEl_SAVE_PATH}/{cfg.run_id}",
            model_save_freq=cfg.model_save_freq,
            verbose=1,
        )
        _callbacks.append(wandb_callback)

    try:
        model.learn(
            total_timesteps=cfg.total_training_steps,
            log_interval=1,
            callback=CallbackList(_callbacks),
            reset_num_timesteps=False if cfg.load_model_path else True,
        )
    finally:
        venv.close()

        if cfg.use_wandb:
            _run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PPO model for the HotWheelsEnv"
    )
    parser.add_argument(
        "--config", help="Path to config YAML file", type=str, required=False
    )
    args = parser.parse_args()

    cfg = Config(file_path=args.config)

    # Generate a new run ID if training new model
    if not cfg.run_id:
        cfg.run_id = get_random_name()
        print(f"New training run {cfg.run_id}")

    print(cfg)

    train(cfg)
