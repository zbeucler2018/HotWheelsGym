import gzip
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from HotWheelsGym.RAMOpponent import _is_white_respawn_frame
from training_scripts.ram_player.callbacks import (
    evaluate_ram_policy,
    evaluation_metric_names,
)
from training_scripts.ram_player.common import (
    InitialStatePool,
    load_config,
    require_all_opponent_slots,
    validate_model_action_space,
    validate_model_observation_space,
)
from training_scripts.ram_player.media import RGBVideoWriter
from training_scripts.ram_player.sweep import checkpoint_sort_key
from training_scripts.ram_player.train import _verify_resume_ppo_configuration


class RAMEvaluationToolTests(unittest.TestCase):
    def test_resume_ppo_configuration_rejects_stale_checkpoint_values(self):
        model = SimpleNamespace(
            lr_schedule=lambda _: 3e-4,
            clip_range=lambda _: 0.2,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
        with self.assertRaisesRegex(RuntimeError, "learning_rate"):
            _verify_resume_ppo_configuration(
                model,
                {
                    "learning_rate": 5e-5,
                    "clip_range": 0.1,
                    "n_steps": 1024,
                    "batch_size": 256,
                    "n_epochs": 5,
                    "gamma": 0.995,
                    "gae_lambda": 0.95,
                    "ent_coef": 0.002,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                },
            )

    def test_evaluation_metric_names_survive_console_truncation(self):
        names = evaluation_metric_names()
        truncated = [("   " + name)[:36] for name in names]
        self.assertEqual(len(truncated), len(set(truncated)))

    def test_initial_state_pool_loads_selected_gzip_state(self):
        class FakeStateEnv(gym.Env):
            observation_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
            action_space = gym.spaces.Discrete(1)

            def __init__(self):
                self.initial_state = b"default"
                self.statename = "default.state"

            def reset(self, *, seed=None, options=None):
                return np.zeros(1, dtype=np.float32), {}

        with TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "alternate.state"
            with gzip.open(state_path, "wb") as handle:
                handle.write(b"alternate")
            base = FakeStateEnv()
            env = InitialStatePool(base, [str(state_path)], seed=7)
            _, info = env.reset(seed=7)

            self.assertEqual(base.initial_state, b"alternate")
            self.assertEqual(base.statename, str(state_path))
            self.assertEqual(info["ram_initial_state"], str(state_path))

    def test_ablation_configs_use_equal_raw_frame_budgets(self):
        root = Path(__file__).resolve().parents[1] / "training_scripts" / "ram_player"
        slow = load_config(root / "ablation_15hz.yml")
        fast = load_config(root / "ablation_30hz.yml")

        self.assertEqual(
            int(slow["total_timesteps"]) * int(slow["frame_skip"]),
            int(fast["total_timesteps"]) * int(fast["frame_skip"]),
        )
        self.assertEqual(
            int(slow["evaluation_max_episode_steps"]) * int(slow["frame_skip"]),
            int(fast["evaluation_max_episode_steps"]) * int(fast["frame_skip"]),
        )
        self.assertEqual(
            int(slow["ppo"]["n_steps"]) * int(slow["frame_skip"]),
            int(fast["ppo"]["n_steps"]) * int(fast["frame_skip"]),
        )

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
                        "ram_player_hairpin_entries": 3,
                        "ram_player_hairpin_completed": 3,
                        "ram_player_hairpin_jet_boost_entries": 2,
                        "ram_player_hairpin_frames": 120,
                        "ram_player_hairpin_completed_frames": 120,
                        "ram_player_hairpin_entry_speed_total": 150_000,
                        "ram_player_hairpin_minimum_speed_total": 90_000,
                        "ram_player_hairpin_exit_speed_total": 135_000,
                        "ram_player_hairpin_wall_frames": 12,
                        "ram_player_hairpin_skid_frames": 30,
                        "ram_player_hairpin_condition_jet_boost_frames": 80,
                        "ram_player_hairpin_condition_no_jet_boost_frames": 40,
                        "ram_player_hairpin_action_drive_brake_frames": 30,
                        "ram_player_hairpin_jet_boost_action_drive_brake_frames": 10,
                        "ram_player_hairpin_no_jet_boost_action_drive_brake_frames": 20,
                        "ram_player_lap_1_frames": 4800,
                        "ram_player_lap_2_frames": 4500,
                        "ram_player_lap_3_frames": 4200,
                        "ram_player_sector_02_entries": 3,
                        "ram_player_sector_02_completed": 3,
                        "ram_player_sector_02_frames": 120,
                        "ram_player_sector_02_completed_frames": 120,
                        "ram_player_sector_02_entry_speed_total": 150_000,
                        "ram_player_sector_02_minimum_speed_total": 90_000,
                        "ram_player_sector_02_exit_speed_total": 135_000,
                        "ram_player_sector_02_wall_frames": 12,
                        "ram_player_sector_02_skid_frames": 30,
                        "ram_player_sector_02_boost_frames": 60,
                        "ram_player_sector_02_jet_boost_frames": 90,
                        "ram_player_sector_02_jet_boost_pickups": 2,
                        "ram_player_sector_02_action_drive_brake_frames": 30,
                        "ram_player_lap_1_sector_02_frames": 42,
                        "ram_player_lap_2_sector_02_frames": 39,
                        "ram_player_lap_3_sector_02_frames": 36,
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
        self.assertEqual(result.mean_hairpin_completions, 3.0)
        self.assertAlmostEqual(result.mean_hairpin_seconds, 2 / 3)
        self.assertEqual(result.mean_hairpin_entry_speed, 50_000)
        self.assertEqual(result.mean_hairpin_minimum_speed, 30_000)
        self.assertEqual(result.mean_hairpin_exit_speed, 45_000)
        self.assertEqual(result.hairpin_wall_contact_rate, 0.1)
        self.assertEqual(result.hairpin_skid_active_rate, 0.25)
        self.assertAlmostEqual(result.hairpin_jet_boost_entry_rate, 2 / 3)
        self.assertEqual(result.action_metrics["race_drive_brake_rate"], 0.25)
        self.assertEqual(result.action_metrics["hairpin_all_drive_brake_rate"], 0.25)
        self.assertEqual(
            result.action_metrics["hairpin_jet_drive_brake_rate"],
            0.125,
        )
        self.assertEqual(
            result.action_metrics["hairpin_nojet_drive_brake_rate"],
            0.5,
        )
        self.assertEqual(result.mean_lap_1_seconds, 80.0)
        self.assertEqual(result.mean_lap_2_seconds, 75.0)
        self.assertEqual(result.mean_lap_3_seconds, 70.0)
        self.assertAlmostEqual(result.sector_metrics["sector_02_seconds"], 2 / 3)
        self.assertEqual(result.sector_metrics["sector_02_entry_speed"], 50_000)
        self.assertEqual(result.sector_metrics["sector_02_minimum_speed"], 30_000)
        self.assertEqual(result.sector_metrics["sector_02_exit_speed"], 45_000)
        self.assertEqual(result.sector_metrics["sector_02_wall_contact_rate"], 0.1)
        self.assertEqual(result.sector_metrics["sector_02_skid_active_rate"], 0.25)
        self.assertEqual(result.sector_metrics["sector_02_boost_active_rate"], 0.5)
        self.assertEqual(
            result.sector_metrics["sector_02_action_drive_brake_rate"], 0.25
        )
        self.assertEqual(result.sector_metrics["sector_02_jet_boost_active_rate"], 0.75)
        self.assertAlmostEqual(
            result.sector_metrics["sector_02_jet_boost_pickup_rate"], 2 / 3
        )
        self.assertEqual(result.sector_metrics["lap_1_sector_02_seconds"], 42 / 60)
        self.assertEqual(result.sector_metrics["lap_2_sector_02_seconds"], 39 / 60)
        self.assertEqual(result.sector_metrics["lap_3_sector_02_seconds"], 36 / 60)

    def test_legacy_ram_checkpoint_gets_clear_observation_error(self):
        class LegacyModel:
            observation_space = type("Space", (), {"shape": (43,)})()

        with self.assertRaisesRegex(ValueError, "observation-v1 checkpoint"):
            validate_model_observation_space(LegacyModel(), "old_model.zip")

    def test_hairpin_means_ignore_episodes_without_a_completed_pass(self):
        class TwoEpisodeEnv(gym.Env):
            observation_space = gym.spaces.Box(-1.0, 1.0, (60,), np.float32)
            action_space = gym.spaces.Discrete(7)

            def __init__(self):
                self.episode = 0

            def reset(self, *, seed=None, options=None):
                self.episode += 1
                return np.zeros(60, dtype=np.float32), {}

            def step(self, action):
                completed = int(self.episode == 1)
                return (
                    np.zeros(60, dtype=np.float32),
                    0.0,
                    False,
                    True,
                    {
                        "ram_player_speed": 0,
                        "ram_player_finished": False,
                        "ram_player_completion": 0.0,
                        "ram_player_rank": 4,
                        "ram_action_repeat_frames": 1,
                        "ram_decision_mean_abs_lateral_offset": 0.0,
                        "ram_decision_mean_heading_alignment": 1.0,
                        "ram_decision_wall_frames": 0,
                        "ram_decision_respawns": 0,
                        "ram_player_hairpin_entries": completed,
                        "ram_player_hairpin_completed": completed,
                        "ram_player_hairpin_frames": 60 * completed,
                        "ram_player_hairpin_completed_frames": 60 * completed,
                    },
                )

        class CoastModel:
            def predict(self, observation, deterministic=True):
                return 0, None

        result = evaluate_ram_policy(
            CoastModel(), TwoEpisodeEnv(), 2, max_episode_frames=1
        )
        self.assertEqual(result.mean_hairpin_completions, 0.5)
        self.assertEqual(result.mean_hairpin_seconds, 1.0)

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

    def test_v5_ram_checkpoint_gets_clear_factorized_action_migration_error(self):
        class V5Model:
            observation_space = type("Space", (), {"shape": (60,)})()

        with self.assertRaisesRegex(ValueError, "observation-v5 checkpoint"):
            validate_model_observation_space(V5Model(), "v5_model.zip")

    def test_v5_action_space_gets_clear_factorized_action_error(self):
        class V5Model:
            action_space = gym.spaces.Discrete(7)

        with self.assertRaisesRegex(ValueError, "v5 seven-action checkpoint"):
            validate_model_action_space(V5Model(), "v5_model.zip")

    def test_v6_action_space_is_accepted(self):
        class V6Model:
            action_space = gym.spaces.MultiDiscrete((4, 3, 2))

        validate_model_action_space(V6Model(), "v6_model.zip")

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
