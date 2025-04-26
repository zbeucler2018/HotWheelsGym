import os
import random
import gzip

import numpy as np
import gymnasium as gym
import pygame
from pygame.locals import K_ESCAPE, KEYDOWN, QUIT


class SaveRandomStates(gym.Wrapper):

    def __init__(self, env, save_dir_path: str, rnd_threshold=0.05):
        super().__init__(env)
        self._rnd_threshold = rnd_threshold
        self._save_dir_path = save_dir_path
        assert os.path.exists(self._save_dir_path), f"{self._save_dir_path} does not exist!"

    def step(self, action):
        ob, rew, terminated, truncated, info = self.env.step(action)

        if random.random() < self._rnd_threshold:
            filepath = f"{self.env.unwrapped.track.value}_{self.env.unwrapped.mode.value}_{info.get('checkpoint', -1)}.state"
            with gzip.open(f"{self._save_dir_path}{filepath}", "wb") as f:
                f.write(self.env.unwrapped.em.get_state())
                print(f"saving {filepath}")

        return ob, rew, terminated, truncated, info

class InteractiveWrapper(gym.Wrapper):
    def __init__(self, env, fps=60):
        super().__init__(env)
        self.fps = fps
        self.clock = pygame.time.Clock()
        self.buttons = env.unwrapped.buttons if hasattr(env.unwrapped, "buttons") else []
        self.key_map = {
            pygame.K_z: "A",
            pygame.K_x: "B",
            pygame.K_c: "C",
            pygame.K_a: "X",
            pygame.K_s: "Y",
            pygame.K_d: "Z",
            pygame.K_q: "L",
            pygame.K_w: "R",
            pygame.K_UP: "UP",
            pygame.K_DOWN: "DOWN",
            pygame.K_LEFT: "LEFT",
            pygame.K_RIGHT: "RIGHT",
            pygame.K_TAB: "SELECT",
            pygame.K_RETURN: "START",
        }

        self.screen = None
        self.obs_shape = self.observation_space.shape

        pygame.init()
        if len(self.obs_shape) == 3:  # (H, W, C)
            self.screen = pygame.display.set_mode((self.obs_shape[1], self.obs_shape[0]))
        else:
            raise ValueError("InteractiveWrapper only supports visual environments with 3D observations.")

    def keys_to_action(self, pressed_keys):
        keys_down = set()
        for key_code in self.key_map:
            if pressed_keys[key_code]:
                keys_down.add(pygame.key.name(key_code).upper())

        inputs = {
            None: False,
            "BUTTON": "Z" in keys_down,
            "A": "Z" in keys_down,
            "B": "X" in keys_down,
            "C": "C" in keys_down,
            "X": "A" in keys_down,
            "Y": "S" in keys_down,
            "Z": "D" in keys_down,
            "L": "Q" in keys_down,
            "R": "W" in keys_down,
            "UP": "UP" in keys_down,
            "DOWN": "DOWN" in keys_down,
            "LEFT": "LEFT" in keys_down,
            "RIGHT": "RIGHT" in keys_down,
            "MODE": "TAB" in keys_down,
            "SELECT": "TAB" in keys_down,
            "RESET": "ENTER" in keys_down,
            "START": "ENTER" in keys_down,
        }

        return np.array([inputs.get(button, False) for button in self.buttons], dtype=np.int8)

    def play(self) -> bool:
        obs, _ = self.reset()
        done = False

        while not done:
            self.clock.tick(self.fps)
            keys = pygame.key.get_pressed()
            action = self.keys_to_action(keys)

            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    pygame.quit()
                    return True

            obs, reward, terminated, truncated, info = self.step(action)
            done = terminated or truncated

            # Convert and render frame
            frame = np.transpose(obs, (1, 0, 2))
            surface = pygame.surfarray.make_surface(frame)
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()

        pygame.quit()
        self.env.close()
        return False
