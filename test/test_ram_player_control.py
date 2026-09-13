import gzip
import importlib.util
from itertools import product
from pathlib import Path
import struct
import sys
import types
import unittest
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "hotwheels_ram_test_package"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(ROOT / "HotWheelsGym")]
sys.modules[PACKAGE_NAME] = package


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE_NAME}.{name}", ROOT / "HotWheelsGym" / filename
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


npc = load_module("npc_control", "npc_control.py")
ram = load_module("ram_opponent_control", "ram_opponent_control.py")
track = sys.modules[f"{PACKAGE_NAME}.dino_boneyard_track"]


class FakeMemory:
    def __init__(self, ewram: bytes):
        self._ewram = bytearray(ewram)

    @property
    def blocks(self):
        return {npc.EWRAM_BASE: bytes(self._ewram)}

    def extract(self, address, data_type):
        offset = address - npc.EWRAM_BASE
        formats = {"|u1": "<B", "<u2": "<H", "<u4": "<I"}
        return struct.unpack_from(formats[data_type], self._ewram, offset)[0]

    def assign(self, address, data_type, value):
        offset = address - npc.EWRAM_BASE
        formats = {"|u1": "<B", "<u2": "<H", "<u4": "<I"}
        struct.pack_into(formats[data_type], self._ewram, offset, value)


def state_memory(path: Path) -> FakeMemory:
    with gzip.open(path, "rb") as handle:
        payload = handle.read()
    return FakeMemory(payload[0x21000:0x61000])


class RAMPlayerControlTests(unittest.TestCase):
    state_path = (
        ROOT
        / "HotWheelsGym"
        / "HotWheelsStuntTrackChallenge-GbAdvance"
        / "dino_boneyard_multi.state"
    )

    def test_shared_observation_is_bounded_for_player_and_cpu(self):
        race = npc.RaceMemory(state_memory(self.state_path))
        states = ram.read_racer_states(race)
        tracker = ram.RaceProgressTracker.from_states(
            states, ram.DINO_BONEYARD_PROGRESS_COUNT
        )
        for slot in (0, 1, 2, 3):
            with self.subTest(slot=slot):
                observation = ram.build_dino_ram_observation(
                    states, states, tracker, slot, ram.RAM_DEFAULT_ACTION
                )
                self.assertEqual(len(observation), ram.DINO_RAM_OBSERVATION_SIZE)
                self.assertEqual(ram.DINO_RAM_OBSERVATION_SIZE, 62)
                self.assertEqual(len(observation), len(ram.RAM_OBSERVATION_NAMES))
                self.assertTrue(all(-1.0 <= value <= 1.0 for value in observation))

    def test_centerline_pose_is_zero_offset_at_reference_point(self):
        race = npc.RaceMemory(state_memory(self.state_path))
        state = race.state(0)
        index = 100
        x, z = track.DINO_BONEYARD_CENTERLINE[index]
        pose = ram.dino_track_pose(replace(state, progress=index, x=x, z=z))

        self.assertEqual(len(pose.features), 11)
        self.assertAlmostEqual(pose.lateral_offset, 0.0, places=6)
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in pose.features))

    def test_track_observation_names_are_part_of_shared_contract(self):
        expected = {
            "track_lateral_offset",
            "track_heading_error_sin",
            "track_heading_error_cos",
            "track_target_short_forward",
            "track_target_long_right",
            "track_curvature_long",
        }
        self.assertTrue(expected.issubset(set(ram.RAM_OBSERVATION_NAMES)))
        self.assertEqual(ram.DINO_RAM_OBSERVATION_VERSION, 6)

    def test_native_skid_state_is_symmetric_for_player_and_opponents(self):
        memory = state_memory(self.state_path)
        stock = npc.RaceMemory(memory)
        active_slots = {0, 2}
        for slot in range(4):
            racer = stock.layout.racer(slot)
            memory.assign(
                racer.address + npc.RACER_SKID_ACTIVE_OFFSET,
                "|u1",
                int(slot in active_slots),
            )
        race = npc.RaceMemory(memory)
        states = ram.read_racer_states(race)
        tracker = ram.RaceProgressTracker.from_states(
            states, ram.DINO_BONEYARD_PROGRESS_COUNT
        )
        skid_index = ram.RAM_OBSERVATION_NAMES.index("self_skid_active")

        for slot in range(4):
            with self.subTest(slot=slot):
                observation = ram.build_dino_ram_observation(
                    states, states, tracker, slot, ram.RAM_DEFAULT_ACTION
                )
                self.assertEqual(observation[skid_index], float(slot in active_slots))

    def test_jet_boost_countdown_is_symmetric_for_player_and_opponents(self):
        memory = state_memory(self.state_path)
        stock = npc.RaceMemory(memory)
        timers = {0: 150, 1: 120, 2: 75, 3: 30}
        for slot, timer in timers.items():
            racer = stock.layout.racer(slot)
            memory.assign(
                racer.address + npc.RACER_POWER_UP_TYPE_OFFSET,
                "|u1",
                npc.JET_BOOST_POWER_UP_TYPE,
            )
            memory.assign(
                racer.address + npc.RACER_POWER_UP_TIMER_OFFSET,
                "|u1",
                timer,
            )
        race = npc.RaceMemory(memory)
        states = ram.read_racer_states(race)
        tracker = ram.RaceProgressTracker.from_states(
            states, ram.DINO_BONEYARD_PROGRESS_COUNT
        )
        jet_boost_index = ram.RAM_OBSERVATION_NAMES.index("self_jet_boost_remaining")

        for slot, timer in timers.items():
            with self.subTest(slot=slot):
                observation = ram.build_dino_ram_observation(
                    states, states, tracker, slot, ram.RAM_DEFAULT_ACTION
                )
                self.assertAlmostEqual(observation[jet_boost_index], timer / 150)

        other_power_up = replace(states[0], power_up_type=2, power_up_timer=150)
        self.assertEqual(other_power_up.jet_boost_remaining, 0)

    def test_boost_charge_is_symmetric_for_player_and_opponents(self):
        state_path = (
            ROOT
            / "training_scripts"
            / "data"
            / "states"
            / "dino_boneyard_multi_180.state"
        )
        race = npc.RaceMemory(state_memory(state_path))
        states = ram.read_racer_states(race)
        tracker = ram.RaceProgressTracker.from_states(
            states, ram.DINO_BONEYARD_PROGRESS_COUNT
        )
        expected = {0: 1.0, 1: 600 / 980, 2: 450 / 980, 3: 1.0}
        boost_index = ram.RAM_OBSERVATION_NAMES.index("self_boost_charge")

        for slot, charge in expected.items():
            with self.subTest(slot=slot):
                observation = ram.build_dino_ram_observation(
                    states, states, tracker, slot, ram.RAM_DEFAULT_ACTION
                )
                self.assertAlmostEqual(observation[boost_index], charge)
                other_slots = ram._ordered_other_slots(slot, states, tracker)
                for position, other_slot in enumerate(other_slots, start=1):
                    other_index = ram.RAM_OBSERVATION_NAMES.index(
                        f"nearby_racer_{position}_boost_charge"
                    )
                    self.assertAlmostEqual(
                        observation[other_index], expected[other_slot]
                    )

    def test_track_features_cover_every_configured_training_state(self):
        states_dir = ROOT / "training_scripts" / "data" / "states"
        for suffix in ("71", "180", "290"):
            race = npc.RaceMemory(
                state_memory(states_dir / f"dino_boneyard_multi_{suffix}.state")
            )
            for slot in (0, 1, 2, 3):
                with self.subTest(state=suffix, slot=slot):
                    pose = ram.dino_track_pose(race.state(slot))
                    self.assertTrue(
                        all(-1.0 <= value <= 1.0 for value in pose.features)
                    )

    def test_player_action_maps_to_accelerate_and_left_buttons(self):
        buttons = ("B", "A", "LEFT", "RIGHT", "L", "R", "UP")
        mapped = ram.player_buttons_from_action((1, 1, 0), buttons)
        self.assertEqual(
            {button for button, enabled in zip(buttons, mapped) if enabled},
            {"A", "LEFT"},
        )

    def test_factorized_action_supports_pure_and_steerable_boost(self):
        self.assertEqual(ram.gba_button_mask_from_action((0, 0, 1)), 0x300)
        self.assertEqual(ram.gba_button_mask_from_action((1, 1, 1)), 0x321)
        self.assertEqual(ram.gba_button_mask_from_action((1, 2, 1)), 0x311)

    def test_previous_factorized_action_is_one_hot_in_observation(self):
        race = npc.RaceMemory(state_memory(self.state_path))
        states = ram.read_racer_states(race)
        tracker = ram.RaceProgressTracker.from_states(
            states, ram.DINO_BONEYARD_PROGRESS_COUNT
        )
        observation = ram.build_dino_ram_observation(
            states, states, tracker, 0, (2, 1, 1)
        )

        for component, selected in (
            ("drive_brake", 1.0),
            ("steering_left", 1.0),
            ("boost_on", 1.0),
        ):
            index = ram.RAM_OBSERVATION_NAMES.index(f"previous_{component}")
            self.assertEqual(observation[index], selected)
        previous_values = [
            value
            for name, value in zip(ram.RAM_OBSERVATION_NAMES, observation)
            if name.startswith("previous_")
        ]
        self.assertEqual(sum(previous_values), 3.0)

    def test_boost_telemetry_excludes_charge_lost_without_boost_action(self):
        spent = ram.boost_telemetry(980, 972, (1, 0, 1))
        reset = ram.boost_telemetry(980, 0, (1, 0, 0))
        gained = ram.boost_telemetry(0, 980, (1, 0, 0))

        self.assertEqual(
            (spent.delta, spent.spent, spent.gained, spent.active),
            (-8, 8, 0, True),
        )
        self.assertEqual((reset.delta, reset.spent, reset.active), (-980, 0, False))
        self.assertEqual(
            (gained.delta, gained.gained, gained.active), (980, 980, False)
        )

    def test_player_and_converted_npc_actions_have_identical_buttons(self):
        buttons = ("A", "B", "SELECT", "START", "RIGHT", "LEFT", "UP", "DOWN", "R", "L")
        for action in product(
            *(range(size) for size in ram.RAM_ACTION_COMPONENT_SIZES)
        ):
            with self.subTest(action=action):
                player = ram.player_buttons_from_action(action, buttons)
                player_names = {
                    button for button, enabled in zip(buttons, player) if enabled
                }
                mask = ram.gba_button_mask_from_action(action)
                npc_names = {
                    name for name, bit in ram.GBA_BUTTON_BITS.items() if mask & bit
                }
                self.assertEqual(player_names, npc_names)

    def test_converted_npc_button_writer_tracks_transitions(self):
        memory = state_memory(self.state_path)
        stock = npc.RaceMemory(memory)
        for slot in (1, 2, 3):
            memory.assign(
                stock.layout.racer(slot).address, "<u4", npc.PLAYER_RACER_VTABLE
            )
        memory.assign(stock.layout.manager + npc.MANAGER_PLAYER_COUNT_OFFSET, "|u1", 4)
        memory.assign(stock.layout.manager + npc.MANAGER_CPU_COUNT_OFFSET, "|u1", 0)
        race = npc.RaceMemory(memory)
        pressed = ram.write_racer_buttons(race, 1, (1, 1, 0), 0)
        held = ram.write_racer_buttons(race, 1, (1, 1, 0), pressed.held)
        released = ram.write_racer_buttons(race, 1, ram.RAM_DEFAULT_ACTION, held.held)
        self.assertEqual(
            (pressed.pressed, pressed.released, pressed.held), (0x21, 0, 0x21)
        )
        self.assertEqual((held.pressed, held.released, held.held), (0, 0, 0x21))
        self.assertEqual(
            (released.pressed, released.released, released.held), (0, 0x21, 0)
        )

    def test_player_class_respawn_pending_is_per_racer(self):
        memory = state_memory(self.state_path)
        stock = npc.RaceMemory(memory)
        for slot in (0, 1, 2, 3):
            memory.assign(
                stock.layout.racer(slot).address, "<u4", npc.PLAYER_RACER_VTABLE
            )
        memory.assign(stock.layout.manager + npc.MANAGER_PLAYER_COUNT_OFFSET, "|u1", 4)
        memory.assign(stock.layout.manager + npc.MANAGER_CPU_COUNT_OFFSET, "|u1", 0)
        race = npc.RaceMemory(memory)

        self.assertFalse(race.respawn_pending(0))
        memory.assign(
            race.layout.racer(3).address + npc.RACER_RESPAWN_PENDING_OFFSET,
            "|u1",
            1,
        )
        self.assertFalse(race.respawn_pending(0))
        self.assertTrue(race.respawn_pending(3))

    def test_progress_tracker_counts_wrap_and_ranks_across_laps(self):
        def state(slot, progress):
            return npc.RacerState(
                slot=slot,
                vehicle_index=slot,
                current_heading=0,
                speed=0,
                progress=progress,
                x=0,
                z=0,
            )

        before = {slot: state(slot, 340 - slot) for slot in range(4)}
        tracker = ram.RaceProgressTracker.from_states(before, 342)
        after = dict(before)
        after[0] = state(0, 2)
        tracker.update(after)
        self.assertEqual(tracker.laps[0], 2)
        self.assertEqual(tracker.current_lap(0, after[0]), 2)
        self.assertEqual(tracker.rank(0, after), 1)

    def test_progress_tracker_derives_laps_from_monotonic_progress(self):
        state = npc.RacerState(
            slot=0,
            vehicle_index=0,
            current_heading=0,
            speed=0,
            progress=700,
            x=0,
            z=0,
        )
        tracker = ram.RaceProgressTracker.from_states({0: state}, 342)
        self.assertEqual(tracker.current_lap(0, state), 3)
        self.assertAlmostEqual(tracker.completion(0, state, 3), 700 / 1026)

    def test_lap_split_tracker_records_each_raw_frame_interval(self):
        timing = ram.LapSplitTracker.start(current_lap=1, total_laps=3)
        timing.update(current_lap=2, raw_frame=5300)
        timing.update(current_lap=3, raw_frame=10400)
        timing.update(current_lap=3, raw_frame=15300, finished_now=True)

        self.assertEqual(timing.padded_splits(), (5300, 5100, 4900))

    def test_hairpin_tracker_records_raw_frame_speed_and_handling(self):
        def state(progress, speed, *, jet_boost=0, skid=False):
            return npc.RacerState(
                slot=0,
                vehicle_index=0,
                current_heading=0,
                speed=speed,
                progress=progress,
                x=0,
                z=0,
                power_up_type=npc.JET_BOOST_POWER_UP_TYPE if jet_boost else 0xFF,
                power_up_timer=jet_boost,
                skid_active=skid,
            )

        telemetry = ram.DinoHairpinTelemetry()
        transitions = (
            (state(44, 110), state(45, 100, jet_boost=145), False),
            (state(45, 100), state(50, 80, jet_boost=140, skid=True), True),
            (state(50, 80), state(60, 70, jet_boost=135), False),
            (state(60, 70), state(61, 90, jet_boost=134), False),
        )
        for previous, current, hit_wall in transitions:
            telemetry.update(previous, current, advance=1, hit_wall=hit_wall)

        self.assertEqual(telemetry.entries, 1)
        self.assertEqual(telemetry.completed, 1)
        self.assertEqual(telemetry.entries_with_jet_boost, 1)
        self.assertEqual(telemetry.frames, 3)
        self.assertEqual(telemetry.completed_frames, 3)
        self.assertEqual(telemetry.speed_total, 250)
        self.assertEqual(telemetry.entry_speed_total, 100)
        self.assertEqual(telemetry.minimum_speed_total, 70)
        self.assertEqual(telemetry.exit_speed_total, 90)
        self.assertEqual(telemetry.wall_frames, 1)
        self.assertEqual(telemetry.skid_frames, 1)
        self.assertFalse(telemetry.active)

    def test_hairpin_tracker_ignores_wrong_way_boundary_crossing(self):
        def state(progress):
            return npc.RacerState(0, 0, 0, 100, progress, 0, 0)

        telemetry = ram.DinoHairpinTelemetry()
        telemetry.update(state(61), state(60), advance=-1, hit_wall=False)
        telemetry.update(state(60), state(61), advance=1, hit_wall=False)

        self.assertEqual(telemetry.entries, 0)
        self.assertEqual(telemetry.completed, 0)
        self.assertEqual(telemetry.frames, 0)

    def test_sector_map_is_contiguous_and_isolates_hairpin(self):
        self.assertEqual(ram.DINO_SECTOR_BOUNDARIES[0], 0)
        self.assertEqual(
            ram.DINO_SECTOR_BOUNDARIES[-1], ram.DINO_BONEYARD_PROGRESS_COUNT
        )
        self.assertEqual(ram.dino_sector_index(44), 1)
        self.assertEqual(ram.dino_sector_index(45), 2)
        self.assertEqual(ram.dino_sector_index(60), 2)
        self.assertEqual(ram.dino_sector_index(61), 3)
        self.assertEqual(ram.dino_sector_index(342), 0)

    def test_sector_tracker_records_timing_handling_boost_and_wrap(self):
        def state(progress, speed, *, skid=False, jet_boost=0):
            return npc.RacerState(
                slot=0,
                vehicle_index=0,
                current_heading=0,
                speed=speed,
                progress=progress,
                x=0,
                z=0,
                power_up_type=npc.JET_BOOST_POWER_UP_TYPE if jet_boost else 0xFF,
                power_up_timer=jet_boost,
                skid_active=skid,
            )

        telemetry = ram.DinoSectorTelemetry.start(state(44, 100))
        telemetry.update(
            state(44, 100),
            state(45, 90, skid=True, jet_boost=150),
            advance=1,
            hit_wall=True,
            boost_active=True,
            jet_boost_acquired=True,
        )
        telemetry.update(
            state(45, 90),
            state(60, 70, jet_boost=140),
            advance=1,
            hit_wall=False,
            boost_active=False,
            jet_boost_acquired=False,
        )
        telemetry.update(
            state(60, 70),
            state(61, 80),
            advance=1,
            hit_wall=False,
            boost_active=False,
            jet_boost_acquired=False,
        )

        self.assertEqual(telemetry.completed[1], 1)
        self.assertEqual(telemetry.completed[2], 1)
        self.assertEqual(telemetry.completed_frames[2], 2)
        self.assertEqual(telemetry.entry_speed_total[2], 90)
        self.assertEqual(telemetry.minimum_speed_total[2], 70)
        self.assertEqual(telemetry.exit_speed_total[2], 80)
        self.assertEqual(telemetry.wall_frames[2], 1)
        self.assertEqual(telemetry.skid_frames[2], 1)
        self.assertEqual(telemetry.boost_frames[2], 1)
        self.assertEqual(telemetry.jet_boost_pickups[2], 1)
        self.assertEqual(telemetry.lap_completed_frames[(1, 2)], 2)

    def test_finished_race_reward_beats_partial_progress(self):
        partial = ram.race_reward(1, 60_000, previous_rank=2, current_rank=1)
        finished = ram.race_reward(
            1,
            60_000,
            previous_rank=2,
            current_rank=1,
            completed_laps=1,
            finished_now=True,
        )
        self.assertGreater(finished, partial + 50)

    def test_reward_penalizes_wall_contact_bad_alignment_and_respawn(self):
        safe = ram.race_reward(
            0,
            30_000,
            previous_rank=2,
            current_rank=2,
        )
        unsafe = ram.race_reward(
            0,
            30_000,
            previous_rank=2,
            current_rank=2,
            hit_wall=True,
            lateral_offset=1.0,
            heading_alignment=-1.0,
            respawned_now=True,
        )
        self.assertAlmostEqual(safe - unsafe, 2.055, places=6)


if __name__ == "__main__":
    unittest.main()
