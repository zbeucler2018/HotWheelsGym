import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from gymnasium.wrappers import TimeLimit

from .action import StochasticFrameSkip
from .reward import ClipReward, RewardOnCrash, RewardOnWallCrash


class HotWheelsWrapper(gym.Wrapper):
    """
    HotWheels preprocessings

    Specifically:

    * Adds a Monitor wrapper
    * Frame skipping: 4 by default
    * Sticky actions: 25% chance to repeat a action by default
    * Use the Nature CNN (grayscale and 84x84 obs) by default
    * Clip reward to {-1, 0, 1} by default
    * Terminate the episode when the agent crashes after failing a trick, by default
    * Terminate the episode when the agent crashes into a wall, by default
    * Sets the maximum steps per episode to 5100 by default

    """

    def __init__(
        self,
        env: gym.Env,
        frame_skip: int = 4,
        frame_skip_prob: float = 0.25,
        use_nature_cnn: bool = True,
        clip_reward: bool = True,
        crash_reward: float = -5,
        wall_crash_reward: float = -5,
        terminate_on_crash: bool = True,
        terminate_on_wall_crash: bool = True,
        max_episode_steps: int = 5_100,
    ) -> None:
        if frame_skip > 1:
            env = StochasticFrameSkip(env, frame_skip, frame_skip_prob)
        if terminate_on_crash:
            env = TerminateOnCrash(env)
        if terminate_on_wall_crash:
            env = TerminateOnWallCrash(env)
        if crash_reward:
            env = RewardOnCrash(env, crash_reward)
        if wall_crash_reward:
            env = RewardOnWallCrash(env, wall_crash_reward)
        if max_episode_steps:
            env = TimeLimit(env, max_episode_steps)
        if use_nature_cnn:
            env = ResizeObservation(env, (84, 84))
            env = GrayscaleObservation(env, keep_dim=True)
        if clip_reward:
            env = ClipReward(env)

        super().__init__(env)

    def reset_emulator_data(self) -> None:
        """
        Resets the emulator and the integrated variables
        """
        try:  # Bare RetroEnv
            retro_data = self.env.unwrapped.data
        except AttributeError:  # Find RetroEnv.data recursively
            retro_data = self.get_wrapper_attr(name="data")
        retro_data.reset()
        retro_data.update_ram()


class TerminateOnCrash(gym.Wrapper):
    """
    A wrapper that ends the episode if the agent
    fails a trick.

    TODO: check this still works w grayscale obs
    """

    def __init__(self, env):
        super().__init__(env)
        self.crash_restart_obs_threshold = 238

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if observation.mean() >= self.crash_restart_obs_threshold:
            terminated = True

        return observation, reward, terminated, truncated, info


class TerminateOnWallCrash(gym.Wrapper):
    """
    A wrapper that ends the episode if the agent
    crashes into a wall. Also applies a penality.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if info["hit_wall"] == 1:
            terminated = True

        return observation, reward, terminated, truncated, info
