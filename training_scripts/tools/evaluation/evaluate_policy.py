from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import (DummyVecEnv, VecEnv, VecMonitor,
                                              is_vecenv_wrapped)



def evaluate_policy(
    model,
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    evaluation_state_file: Optional[str] = None
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or VecEnv environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param evaluation_state_file: Filepath to the evaluation state file to use for
        evaluation instead of the training state. Will be returned to original training state.

    Returns metrics and statistics from evaluation
        episode_info = {
            "episode_rewards": [],
            "episode_lengths": [],
            "episode_checkpoints": [],
            "episode_scores": [],
            "episode_laps": [],
            "episode_rank": [],
            "episode_avg_speed": [],
            "episode_duration_s": []
        }
    """



    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = (
        is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    )

    if evaluation_state_file is not None:
        training_states = env.unwrapped.get_attr("statename")
        # load the state
        _ = env.env_method(method_name="load_state", statename=evaluation_state_file)
        # reset RAM and variables
        _ = env.env_method(method_name="reset_emulator_data")


    n_envs = env.unwrapped.num_envs
    episode_info = {
        "episode_rewards": [],
        "episode_lengths": [],
        "episode_checkpoints": [],
        "episode_scores": [],
        "episode_laps": [],
        "episode_rank": [],
        "episode_avg_speed": [],
        "episode_duration_s": []
    }

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array(
        [(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int"
    )

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.unwrapped.num_envs,), dtype=bool)
    total_steps = 0
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        total_steps += 1
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_info["episode_rewards"].append(info["episode"]["r"])
                            episode_info["episode_lengths"].append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_info["episode_lengths"].append(current_lengths[i])
                        episode_info["episode_rewards"].append(current_rewards[i])
                        episode_counts[i] += 1
                    
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    episode_info["episode_checkpoints"].append(info["checkpoint"])
                    episode_info["episode_laps"].append(info["lap"])
                    episode_info["episode_scores"].append(info["score"])
                    episode_info["episode_rank"].append(info.get("rank", 0))
                    episode_info["episode_avg_speed"].append(info["average_speed"])
                    episode_info["episode_duration_s"].append(info["race_duration_s"])

        observations = new_observations

        if render:
            env.render("human")

    if evaluation_state_file is not None:
        training_states = env.unwrapped.get_attr("statename")
        # set back the original training states
        for indx, t_state in enumerate(training_states):
            _ = env.env_method(method_name="load_state", indices=indx, statename=t_state)
        # reset RAM and variables
        _ = env.env_method(method_name="reset_emulator_data")

    return episode_info
