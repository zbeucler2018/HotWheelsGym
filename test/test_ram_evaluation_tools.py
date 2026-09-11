import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np

from HotWheelsGym.RAMOpponent import _is_white_respawn_frame
from training_scripts.ram_player.callbacks import evaluate_ram_policy
from training_scripts.ram_player.common import (
    require_all_opponent_slots,
    validate_model_observation_space,
)
from training_scripts.ram_player.media import RGBVideoWriter
from training_scripts.ram_player.sweep import checkpoint_sort_key


class RAMEvaluationToolTests(unittest.TestCase):
    def test_evaluation_aggregates_track_quality_metrics(self):
        class OneStepEnv(gym.Env):
            observation_space = gym.spaces.Box(-1.0, 1.0, (60,), np.float32)
            action_space = gym.spaces.Discrete(7)

            def reset(self, *, seed=None, options=None):
                return np.zeros(60, dtype=np.float32), {}

            def step(self, action):
                return (
                    np.zeros(60, dtype=np.float32),
                    1.0,
                    False,
                    True,
                    {
                        "ram_player_speed": 20_000,
                        "ram_player_finished": False,
                        "ram_player_completion": 0.25,
                        "ram_player_rank": 3,
                        "ram_action_repeat_frames": 4,
                        "ram_decision_mean_abs_lateral_offset": 0.125,
                        "ram_decision_mean_heading_alignment": 0.75,
                        "ram_decision_wall_frames": 1,
                        "ram_decision_respawns": 1,
                        "ram_decision_mean_boost_charge": 0.5,
                        "ram_decision_boost_spent": 8,
                        "ram_decision_boost_gained": 16,
                        "ram_decision_boost_frames": 1,
                        "ram_decision_mean_jet_boost_remaining": 0.5,
                        "ram_decision_jet_boost_frames": 2,
                        "ram_decision_jet_boost_pickups": 1,
                        "ram_decision_skid_frames": 3,
                        "ram_player_lap_1_frames": 4800,
                        "ram_player_lap_2_frames": 4500,
                        "ram_player_lap_3_frames": 4200,
                    },
                )

        class CoastModel:
            def predict(self, observation, deterministic=True):
                return 0, None

        result = evaluate_ram_policy(
            CoastModel(), OneStepEnv(), 1, max_episode_frames=4
        )
        self.assertEqual(result.mean_abs_lateral_offset, 0.125)
        self.assertEqual(result.mean_heading_alignment, 0.75)
        self.assertEqual(result.wall_contact_rate, 0.25)
        self.assertEqual(result.mean_respawns, 1.0)
        self.assertEqual(result.mean_boost_charge, 0.5)
        self.assertEqual(result.mean_boost_spent, 8.0)
        self.assertEqual(result.mean_boost_gained, 16.0)
        self.assertEqual(result.boost_active_rate, 0.25)
        self.assertEqual(result.mean_jet_boost_remaining, 0.5)
        self.assertEqual(result.jet_boost_active_rate, 0.5)
        self.assertEqual(result.mean_jet_boost_pickups, 1.0)
        self.assertEqual(result.skid_active_rate, 0.75)
        self.assertEqual(result.mean_lap_1_seconds, 80.0)
        self.assertEqual(result.mean_lap_2_seconds, 75.0)
        self.assertEqual(result.mean_lap_3_seconds, 70.0)

    def test_legacy_ram_checkpoint_gets_clear_observation_error(self):
        class LegacyModel:
            observation_space = type("Space", (), {"shape": (43,)})()

        with self.assertRaisesRegex(ValueError, "observation-v1 checkpoint"):
            validate_model_observation_space(LegacyModel(), "old_model.zip")

    def test_v2_ram_checkpoint_gets_clear_boost_migration_error(self):
        class V2Model:
            observation_space = type("Space", (), {"shape": (54,)})()

        with self.assertRaisesRegex(ValueError, "observation-v2 checkpoint"):
            validate_model_observation_space(V2Model(), "v2_model.zip")

    def test_v3_ram_checkpoint_gets_clear_jet_boost_migration_error(self):
        class V3Model:
            observation_space = type("Space", (), {"shape": (58,)})()

        with self.assertRaisesRegex(ValueError, "observation-v3 checkpoint"):
            validate_model_observation_space(V3Model(), "v3_model.zip")

    def test_v4_ram_checkpoint_gets_clear_skid_state_migration_error(self):
        class V4Model:
            observation_space = type("Space", (), {"shape": (59,)})()

        with self.assertRaisesRegex(ValueError, "observation-v4 checkpoint"):
            validate_model_observation_space(V4Model(), "v4_model.zip")

    def test_white_respawn_frame_detection_rejects_normal_frames(self):
        normal = np.zeros((16, 16, 3), dtype=np.uint8)
        white = np.full((16, 16, 3), 255, dtype=np.uint8)
        white[:1, :1] = 0

        self.assertFalse(_is_white_respawn_frame(normal))
        self.assertTrue(_is_white_respawn_frame(white))

    def test_native_button_runs_require_every_opponent_slot(self):
        require_all_opponent_slots({})
        require_all_opponent_slots({1: "model", 2: "model", 3: "model"})
        with self.assertRaisesRegex(ValueError, "slots 1, 2, and 3"):
            require_all_opponent_slots({1: "model"})

    def test_checkpoint_selection_prefers_finish_then_speed(self):
        partial = {
            "finish_rate": 0.0,
            "mean_finish_frames": 0.0,
            "mean_completion": 0.99,
            "mean_rank": 1.0,
            "mean_reward": 999.0,
        }
        slow_finish = {
            "finish_rate": 1.0,
            "mean_finish_frames": 20_000.0,
            "mean_completion": 1.0,
            "mean_rank": 1.0,
            "mean_reward": 900.0,
        }
        fast_finish = dict(slow_finish, mean_finish_frames=18_000.0)
        self.assertIs(
            max((partial, slow_finish, fast_finish), key=checkpoint_sort_key),
            fast_finish,
        )

    def test_rgb_video_writer_keeps_evaluation_video_on_disk(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "race.mp4"
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            recorder = RGBVideoWriter(video, frame, fps=4)
            for red in (32, 64, 128, 255):
                next_frame = frame.copy()
                next_frame[:, :, 0] = red
                recorder.write(next_frame)
            recorder.close()

            self.assertTrue(video.is_file())
            self.assertGreater(video.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
