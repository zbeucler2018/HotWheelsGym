import pprint
from pathlib import Path

import yaml
from HotWheelsGym import RaceMode, Tracks


class Config:
    """Configuration to train a PPO model"""

    def __init__(self, file_path: str = "") -> None:
        # Blank for new run
        self.run_id: str
        """ID of the experiment. Leave blank for new run. Will also be used as the WandB run ID"""

        # Env
        self.env_id: str
        self.track: Tracks
        self.mode: RaceMode
        self.total_laps: int

        self.num_envs: int
        """The total amount of logical cores in your CPU to train with"""
        self.action_space: list[list[str]]
        """Button combos for the for the agent to use"""

        # Env termination / truncation
        self.max_episode_steps: int
        """Maximum amount of steps before terminating a episode"""
        self.terminate_on_crash: bool
        """Terminate the episode when failing a trick"""
        self.terminate_on_wall_crash: bool
        """Terminate the episode when colliding with a wall"""

        self.frame_stack: int

        # Frame skipping
        self.frame_skip: int
        self.frame_skip_prob: float

        # Env obs
        self.trim_obs: bool
        """Remove the timers, minimap, and speed dial from the observation"""
        self.minimap_obs: bool
        """Reduce the observation to just the minimap"""

        # Reward
        self.crash_reward: float
        """Reward amount given each time the agent crashes when failing a trick"""
        self.wall_crash_reward: float
        """Reward amount given each time the agent crashes into a wall"""
        self.use_speed_reward: bool
        """Use the speed wrapper or not"""

        # Training states
        self.training_states: list[str]
        """Filenames of states the envs will use to train. Can be used to train on multiple parts of the track at once"""

        # Resume training
        self.load_model_path: str
        """Filepath of the model to load and resume training"""

        # Model storage
        self.model_save_freq: int
        """Step freq in which to save the model"""
        # self.model_save_path: str
        # """filepath in which to save the model during training"""
        # self.best_model_save_path: str
        # """filepath in which the best model is save with the EvalCallback"""

        self.total_training_steps: int
        """Total steps for the model to train"""
        self.training_reward_threshold: int
        """Reward threshold to stop training"""

        # model parameters
        self.policy: str
        self.learning_rate: float
        self.n_steps: int
        self.batch_size: int
        self.n_epochs: int
        self.gamma: float
        self.gae_lambda: float
        self.clip_range: float
        self.ent_coef: float

        # Evaluation
        self.eval_freq: int  # max(value // self.num_envs, 1)
        """Step freq in which to evaluate the model"""
        self.render_eval: bool
        """Whether to display the evaluation"""
        self.eval_state_path: str
        """Filepath to evaluation state"""

        # Misc
        self.use_wandb: bool
        """Use WandB monitoring"""
        self.config_filepath: str
        """Filepath to the config file to load"""
        self.nature_env: bool
        """Convert to grayscale and warp frames to 84x84 (default) as done in the Nature paper and clip reward to {-1, 0 +1} by sign"""

        if file_path:
            self.load_file_config(file_path)

        self.validate_config()

    def load_file_config(self, file_path):
        """
        loads a given config yaml file
        """
        with open(file_path, "r") as file:
            config_data = yaml.safe_load(file)

        # Create class variables for each field in the config file
        for key, value in config_data.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        # For a nicer print
        properties = vars(self)
        formatted_properties = pprint.pformat(properties, sort_dicts=False)
        return formatted_properties

    def validate_config(self) -> None:
        """
        Throws exception if the config doesn't meet
        the rules below
        """

        # Ensure only 1 obs wrapper can be used at once
        if self.trim_obs and self.minimap_obs:
            raise ValueError(
                "Cannot use both 'trim_obs' and 'minimap_obs'. You can only use one obs wrapper at a time."
            )

        # Ensure that the amount of training states is equal
        # to the amount of envs
        if self.training_states and len(self.training_states) != self.num_envs:
            raise ValueError(
                f"The amount of training states ({len(self.training_states)}) must be the same as the amount of envs ({self.num_envs}) used for training."
            )

        # Ensure training state file paths are valid
        if self.training_states:
            for state in self.training_states:
                if not Path(state).absolute().exists():
                    raise ValueError(
                        f"The training state {state} ({Path(state).absolute()}) does not exist"
                    )

        # Ensure evaluation state file paths are valid
        if self.eval_state_path:
            if not Path(self.eval_state_path).absolute().exists():
                raise ValueError(
                    f"The evaluation state {self.eval_state_path} ({Path(self.eval_state_path).absolute()}) does not exist"
                )
            
        if self.eval_freq:
            self.eval_freq = max(self.eval_freq // self.num_envs, 1)
