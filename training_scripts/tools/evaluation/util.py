from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path


class RotateStateCallback(BaseCallback):
    """
    Change the training state every `rotate_freq` steps to the next state in `state_list`.
    NOTE: Will set ALL envs to a single state

    rotate_freq: step frequency to change to the next state
    state_list: list of state names to rotate through
    """

    def __init__(
        self, rotate_freq: int, training_states: list, verbose: int = 0, **kwargs
    ):
        self.rotate_freq = rotate_freq
        self.training_states = training_states
        self.current_state_index = 0
        super().__init__(verbose)

    def _on_step(self):
        if self.rotate_freq > 0 and self.n_calls % self.rotate_freq == 0:
            new_state = self.training_states[self.current_state_index]
            self.logger.record(
                f"[total-steps-{self.n_calls}] Setting new training state", new_state
            )
            for indx in range(self.training_env.n_envs):
                _ = self.training_env.env_method(
                    method_name="load_state",
                    statename=str(Path(new_state).absolute()),
                    indices=indx,
                )
                _ = self.training_env.env_method(
                    method_name="reset_emulator_data", indices=indx
                )
                self.logger.record(f"[env-index-{indx}] Set training state", new_state)
            # _ = venv.reset()
        return True
