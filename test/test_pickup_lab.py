import gzip
from math import cos, pi, sin
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np

from HotWheelsGym.ram_opponent_control import DINO_RAM_OBSERVATION_SIZE
from training_scripts.ram_player.common import InitialStatePool, load_config
from training_scripts.ram_player.pickup_lab import (
    PICKUP_AVAILABLE_INDEX,
    PICKUP_DISTANCE_INDEX,
    PICKUP_LATERAL_ERROR_INDEX,
    PICKUP_TYPE_INDEX,
    SELF_LAP_INDEX,
    JET_BOOST_REMAINING_INDEX,
    TRACK_PHASE_COS_INDEX,
    TRACK_PHASE_SIN_INDEX,
    DinoPickupHairpinTask,
    PickupExpertCompositePolicy,
    PickupFeedbackCompositePolicy,
)


class ScriptedPickupEnv(gym.Env):
    observation_space = gym.spaces.Box(
        -1.0,
        1.0,
        (DINO_RAM_OBSERVATION_SIZE,),
        np.float32,
    )
    action_space = gym.spaces.MultiDiscrete((4, 3, 2))

    def __init__(self, steps):
        self.steps = list(steps)
        self.index = 0

    def reset(self, *, seed=None, options=None):
        self.index = 0
        return np.zeros(DINO_RAM_OBSERVATION_SIZE, dtype=np.float32), {
            "ram_initial_state": "test.state",
            "ram_player_next_power_up_is_jet_boost": True,
            "ram_player_next_power_up_available": True,
            "ram_player_next_power_up_progress_distance": 10.0,
        }

    def step(self, action):
        values = dict(self.steps[self.index])
        self.index += 1
        terminated = bool(values.pop("_terminated", False))
        truncated = bool(values.pop("_truncated", False))
        info = {
            "ram_action_repeat_frames": 4,
            "ram_decision_continuous_progress_delta": 0.0,
            "ram_decision_score_delta": 0,
            "ram_decision_wall_frames": 0,
            "ram_decision_jet_boost_pickups": 0,
            "ram_decision_respawns": 0,
            "ram_player_next_power_up_is_jet_boost": True,
            "ram_player_next_power_up_available": True,
            "ram_player_next_power_up_lateral_error": 0.0,
            "ram_player_speed": 0,
            **values,
        }
        return (
            np.zeros(DINO_RAM_OBSERVATION_SIZE, dtype=np.float32),
            999.0,
            terminated,
            truncated,
            info,
        )


class PickupLabTests(unittest.TestCase):
    def test_feedback_composite_corrects_v9_line_then_returns_control(self):
        class BasePolicy:
            def predict(self, observation, deterministic=True):
                return np.asarray((3, 2, 1)), None

        policy = PickupFeedbackCompositePolicy(BasePolicy())
        observation = np.zeros(DINO_RAM_OBSERVATION_SIZE, dtype=np.float32)
        angle = 2.0 * pi * 34.0 / 342.0
        observation[TRACK_PHASE_SIN_INDEX] = sin(angle)
        observation[TRACK_PHASE_COS_INDEX] = cos(angle)
        observation[PICKUP_TYPE_INDEX] = 1.0
        observation[PICKUP_AVAILABLE_INDEX] = 1.0
        observation[PICKUP_DISTANCE_INDEX] = 9.0 / 64.0
        observation[PICKUP_LATERAL_ERROR_INDEX] = -0.15

        action, _ = policy.predict(observation)
        self.assertEqual(action, (1, 1, 0))

        observation[JET_BOOST_REMAINING_INDEX] = 0.5
        action, _ = policy.predict(observation)
        np.testing.assert_array_equal(action, (3, 2, 1))

        policy.reset()
        observation[JET_BOOST_REMAINING_INDEX] = 0.0
        observation[SELF_LAP_INDEX] = 1.0
        observation[PICKUP_LATERAL_ERROR_INDEX] = -0.15
        action, _ = policy.predict(observation)
        self.assertEqual(action, (1, 0, 0))

    def test_v11_config_pretrains_strongly_and_cannot_reward_stalling(self):
        config = load_config(
            Path(__file__).resolve().parents[1]
            / "training_scripts"
            / "ram_player"
            / "dino_boneyard_pickup_expert_v11.yml"
        )

        self.assertEqual(config["behavior_cloning"]["epochs"], 200)
        self.assertGreater(
            config["pickup_task"]["timeout_penalty"],
            config["pickup_task"]["miss_penalty"],
        )
        self.assertGreater(config["pickup_task"]["frame_cost"], 0.0)

    def test_v12_config_extends_objective_through_recovery(self):
        config = load_config(
            Path(__file__).resolve().parents[1]
            / "training_scripts"
            / "ram_player"
            / "dino_boneyard_pickup_recovery_v12.yml"
        )

        self.assertEqual(config["pickup_task"]["exit_progress"], 100.0)
        self.assertGreater(config["pickup_task"]["exit_lateral_penalty"], 0.0)
        self.assertGreater(config["pickup_task"]["exit_heading_scale"], 0.0)
        self.assertIn("demonstrations", config["behavior_cloning"])

    def test_composite_gates_expert_only_through_pickup_hairpin(self):
        class NamedPolicy:
            def __init__(self, action):
                self.action = np.asarray(action)

            def predict(self, observation, deterministic=True):
                return self.action, None

        base = NamedPolicy((1, 0, 0))
        expert = NamedPolicy((1, 1, 1))
        composite = PickupExpertCompositePolicy(base, expert)

        def observation(
            progress,
            *,
            pickup=True,
            distance=10.0,
            jet_boost_remaining=0.0,
        ):
            values = np.zeros(DINO_RAM_OBSERVATION_SIZE, dtype=np.float32)
            angle = 2.0 * pi * progress / 342.0
            values[TRACK_PHASE_SIN_INDEX] = sin(angle)
            values[TRACK_PHASE_COS_INDEX] = cos(angle)
            values[PICKUP_TYPE_INDEX] = float(pickup)
            values[PICKUP_AVAILABLE_INDEX] = float(pickup)
            values[PICKUP_DISTANCE_INDEX] = distance / 64.0
            values[JET_BOOST_REMAINING_INDEX] = jet_boost_remaining
            return values

        action, _ = composite.predict(observation(20.0))
        np.testing.assert_array_equal(action, expert.action)
        action, _ = composite.predict(
            observation(50.0, pickup=False, jet_boost_remaining=0.5)
        )
        np.testing.assert_array_equal(action, expert.action)
        action, _ = composite.predict(observation(62.0, pickup=False))
        np.testing.assert_array_equal(action, base.action)

        composite.reset()
        composite.predict(observation(20.0))
        action, _ = composite.predict(observation(45.0, pickup=False))
        np.testing.assert_array_equal(action, base.action)

        recovery_composite = PickupExpertCompositePolicy(
            base,
            expert,
            handoff_progress=100.0,
        )
        recovery_composite.predict(observation(20.0))
        action, _ = recovery_composite.predict(
            observation(62.0, pickup=False, jet_boost_remaining=0.5)
        )
        np.testing.assert_array_equal(action, expert.action)
        action, _ = recovery_composite.predict(observation(101.0, pickup=False))
        np.testing.assert_array_equal(action, base.action)

        final_lap_composite = PickupExpertCompositePolicy(
            base,
            expert,
            handoff_progress=100.0,
            target_lap=3,
        )
        action, _ = final_lap_composite.predict(observation(20.0))
        np.testing.assert_array_equal(action, base.action)
        final_lap_observation = observation(20.0)
        final_lap_observation[SELF_LAP_INDEX] = 1.0
        action, _ = final_lap_composite.predict(final_lap_observation)
        np.testing.assert_array_equal(action, expert.action)

        late_composite = PickupExpertCompositePolicy(
            base,
            expert,
            activation_distance=10.0,
        )
        action, _ = late_composite.predict(observation(20.0, distance=20.0))
        np.testing.assert_array_equal(action, base.action)
        action, _ = late_composite.predict(observation(34.0, distance=9.0))
        np.testing.assert_array_equal(action, expert.action)

    def test_task_rewards_pickup_and_terminates_after_hairpin_exit(self):
        env = DinoPickupHairpinTask(
            ScriptedPickupEnv(
                [
                    {
                        "ram_decision_continuous_progress_delta": 11.0,
                        "ram_decision_jet_boost_pickups": 1,
                        "ram_decision_score_delta": 250,
                        "ram_player_next_power_up_is_jet_boost": False,
                        "ram_player_next_power_up_available": False,
                    },
                    {
                        "ram_decision_continuous_progress_delta": 18.0,
                        "ram_player_next_power_up_is_jet_boost": False,
                        "ram_player_next_power_up_available": False,
                        "ram_player_speed": 48_000,
                    },
                ]
            )
        )
        env.reset()
        _, first_reward, terminated, truncated, info = env.step((1, 0, 0))
        self.assertGreater(first_reward, 100.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["pickup_task_acquired"])

        _, _, terminated, truncated, info = env.step((1, 1, 1))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["pickup_task_success"])
        self.assertEqual(info["pickup_task_frames_to_pickup"], 4)
        self.assertEqual(info["pickup_task_exit_speed"], 48_000)
        self.assertEqual(info["pickup_task_score_gained"], 250)

    def test_recovery_task_waits_for_later_handoff_and_reports_pose(self):
        env = DinoPickupHairpinTask(
            ScriptedPickupEnv(
                [
                    {
                        "ram_decision_continuous_progress_delta": 11.0,
                        "ram_decision_jet_boost_pickups": 1,
                        "ram_player_next_power_up_is_jet_boost": False,
                        "ram_player_next_power_up_available": False,
                    },
                    {
                        "ram_decision_continuous_progress_delta": 30.0,
                        "ram_player_next_power_up_is_jet_boost": False,
                        "ram_player_next_power_up_available": False,
                    },
                    {
                        "ram_decision_continuous_progress_delta": 28.1,
                        "ram_player_next_power_up_is_jet_boost": False,
                        "ram_player_next_power_up_available": False,
                        "ram_player_speed": 60_000,
                        "ram_player_lateral_offset": -0.2,
                        "ram_player_heading_alignment": 0.9,
                    },
                ]
            ),
            config={
                "exit_progress": 100.0,
                "exit_lateral_penalty": 5.0,
                "exit_heading_scale": 5.0,
            },
        )
        env.reset()
        env.step((1, 0, 0))
        _, _, terminated, _, _ = env.step((1, 0, 0))
        self.assertFalse(terminated)
        _, _, terminated, _, info = env.step((1, 0, 0))
        self.assertTrue(terminated)
        self.assertTrue(info["pickup_task_success"])
        self.assertAlmostEqual(info["pickup_task_exit_abs_lateral_offset"], 0.2)
        self.assertAlmostEqual(info["pickup_task_exit_heading_alignment"], 0.9)

    def test_task_ends_immediately_after_missed_pickup(self):
        env = DinoPickupHairpinTask(
            ScriptedPickupEnv([{"ram_decision_continuous_progress_delta": 12.1}])
        )
        env.reset()
        _, reward, terminated, truncated, info = env.step((1, 0, 0))

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["pickup_task_missed"])
        self.assertFalse(info["pickup_task_acquired"])
        self.assertFalse(info["pickup_task_timed_out"])
        self.assertLess(reward, 0.0)

    def test_task_marks_timeout_and_makes_it_worse_than_a_miss(self):
        env = DinoPickupHairpinTask(ScriptedPickupEnv([{"_truncated": True}]))
        env.reset()
        _, timeout_reward, terminated, truncated, info = env.step((0, 0, 0))

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(info["pickup_task_timed_out"])
        self.assertLess(timeout_reward, -50.0)

    def test_initial_state_pool_can_cycle_holdout_states(self):
        class FakeStateEnv(gym.Env):
            observation_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
            action_space = gym.spaces.Discrete(1)

            def __init__(self):
                self.initial_state = b"default"
                self.statename = "default.state"

            def reset(self, *, seed=None, options=None):
                return np.zeros(1, dtype=np.float32), {}

        with TemporaryDirectory() as temporary:
            paths = []
            for name in ("first", "second"):
                path = Path(temporary) / f"{name}.state"
                with gzip.open(path, "wb") as handle:
                    handle.write(name.encode())
                paths.append(str(path))
            base = FakeStateEnv()
            env = InitialStatePool(base, paths, seed=3, selection="cycle")

            names = [Path(env.reset()[1]["ram_initial_state"]).stem for _ in range(3)]
            self.assertEqual(names, ["first", "second", "first"])


if __name__ == "__main__":
    unittest.main()
