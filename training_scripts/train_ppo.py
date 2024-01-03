import argparse

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import (SubprocVecEnv, VecFrameStack,
                                              VecTransposeImage)
from tools import Config, get_random_name

from HotWheelsGym import HotWheelsEnv

# from tools.wrappers import HotWheelsDiscretizer



def train(cfg: Config) -> None:
    def make_env() -> gym.Env:
        """Util to create envs for vec envs"""
        _env = HotWheelsEnv.make(
            id=cfg.env_id,
            render_mode="rgb_array",
        )

        # if cfg.action_space: # apply custom action space
        #     _env = HotWheelsDiscretizer(_env, action_space=cfg.action_space)
        # _env = HotWheelsWrapper(_env)
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
                method_name="load_state", statename=t_state, indices=indx
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
            tensorboard_log="./data/logs/tf/",
        )

    # Callbacks
    _callbacks = []

    eval_callback = EvalCallback(
        venv,
        best_model_save_path=f"./data/best_models/{cfg.run_id}",
        log_path=f"./data/evaluation/{cfg.run_id}",
        eval_freq=cfg.eval_freq,
        eval_statename=cfg.evaluation_statename,
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
            dir="./data/logs/wandb/",
        )

        wandb_callback = WandbCallback(
            # gradient_save_freq=50_000,
            model_save_path=cfg.model_save_path,
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

    train(cfg)
