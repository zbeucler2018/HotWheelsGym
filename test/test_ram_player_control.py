import gzip
import importlib.util
from pathlib import Path
import struct
import sys
import types
import unittest

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
                    states, states, tracker, slot, 1
                )
                self.assertEqual(len(observation), ram.DINO_RAM_OBSERVATION_SIZE)
                self.assertEqual(len(observation), len(ram.RAM_OBSERVATION_NAMES))
                self.assertTrue(all(-1.0 <= value <= 1.0 for value in observation))

    def test_player_action_maps_to_accelerate_and_left_buttons(self):
        buttons = ("B", "A", "LEFT", "RIGHT", "L", "R", "UP")
        mapped = ram.player_buttons_from_action(2, buttons)
        self.assertEqual(
            {button for button, enabled in zip(buttons, mapped) if enabled},
            {"A", "LEFT"},
        )

    def test_player_and_converted_npc_actions_have_identical_buttons(self):
        buttons = ("A", "B", "SELECT", "START", "RIGHT", "LEFT", "UP", "DOWN", "R", "L")
        for action_index, action in enumerate(ram.RAM_ACTIONS):
            with self.subTest(action=action.name):
                player = ram.player_buttons_from_action(action_index, buttons)
                player_names = {
                    button for button, enabled in zip(buttons, player) if enabled
                }
                mask = ram.gba_button_mask_from_action(action_index)
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
        pressed = ram.write_racer_buttons(race, 1, 2, 0)
        held = ram.write_racer_buttons(race, 1, 2, pressed.held)
        released = ram.write_racer_buttons(race, 1, 0, held.held)
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

    def test_cpu_uses_same_discrete_intent(self):
        race = npc.RaceMemory(state_memory(self.state_path))
        state = race.state(1)
        left = ram.cpu_command_from_action(state, 2)
        right = ram.cpu_command_from_action(state, 3)
        coast = ram.cpu_command_from_action(state, 0)
        self.assertEqual(
            left.desired_heading,
            (state.current_heading - npc.DEFAULT_MAX_TURN) & 0xFFF,
        )
        self.assertEqual(
            right.desired_heading,
            (state.current_heading + npc.DEFAULT_MAX_TURN) & 0xFFF,
        )
        self.assertEqual(left.target_speed, npc.DEFAULT_MAX_TARGET_SPEED)
        self.assertEqual(coast.target_speed, 0)

    def test_progress_tracker_counts_wrap_and_ranks_across_laps(self):
        def state(slot, progress):
            return npc.RacerState(
                slot=slot,
                vehicle_index=slot,
                current_heading=0,
                desired_heading=0,
                stock_heading=0,
                speed=0,
                target_speed=0,
                progress=progress,
                x=0,
                z=0,
                rank=1,
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
            desired_heading=0,
            stock_heading=0,
            speed=0,
            target_speed=0,
            progress=700,
            x=0,
            z=0,
            rank=1,
        )
        tracker = ram.RaceProgressTracker.from_states({0: state}, 342)
        self.assertEqual(tracker.current_lap(0, state), 3)
        self.assertAlmostEqual(tracker.completion(0, state, 3), 700 / 1026)

    def test_finished_race_reward_beats_partial_progress(self):
        partial = ram.race_reward(1, 60_000, 73_728, previous_rank=2, current_rank=1)
        finished = ram.race_reward(
            1,
            60_000,
            73_728,
            previous_rank=2,
            current_rank=1,
            completed_laps=1,
            finished_now=True,
        )
        self.assertGreater(finished, partial + 50)


if __name__ == "__main__":
    unittest.main()
