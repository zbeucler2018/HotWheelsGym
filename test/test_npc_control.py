import gzip
import importlib.util
from pathlib import Path
import struct
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "HotWheelsGym" / "npc_control.py"
SPEC = importlib.util.spec_from_file_location("hotwheels_npc_control", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
npc = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = npc
SPEC.loader.exec_module(npc)


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
    assert len(payload) == 0x61000
    return FakeMemory(payload[0x21000 : 0x61000])


class NPCControlTests(unittest.TestCase):
    integration = ROOT / "HotWheelsGym" / "HotWheelsStuntTrackChallenge-GbAdvance"

    def test_discovers_ordered_racers_in_every_multiplayer_state(self):
        states = sorted(self.integration.glob("*_multi.state"))
        self.assertGreater(len(states), 0)
        for path in states:
            with self.subTest(state=path.name):
                layout = npc.discover_race_layout(state_memory(path))
                self.assertEqual([racer.kind for racer in layout.racers], [
                    "player", "cpu", "cpu", "cpu"
                ])
                self.assertEqual(
                    [racer.vehicle_index for racer in layout.racers], [0, 1, 2, 3]
                )

    def test_maps_normalized_action_to_native_heading_and_speed(self):
        path = self.integration / "dino_boneyard_multi.state"
        memory = state_memory(path)
        race = npc.RaceMemory(memory)
        before = race.state(1)
        command = race.command(1, [1.0, 0.5])
        after = race.state(1)
        self.assertEqual(
            command.desired_heading,
            (before.current_heading + npc.DEFAULT_MAX_TURN) & 0xFFF,
        )
        self.assertEqual(command.target_speed, npc.DEFAULT_MAX_TARGET_SPEED // 2)
        self.assertEqual(after.desired_heading, command.desired_heading)
        self.assertEqual(after.target_speed, command.target_speed)

    def test_structured_observation_is_bounded_and_seeds_stock_heading(self):
        path = self.integration / "dino_boneyard_multi.state"
        race = npc.RaceMemory(state_memory(path))
        desired = race.state(1).desired_heading
        race.prime_stock_heading(1)
        state = race.state(1)
        observation = race.observation(1, npc.TRACK_PROGRESS_COUNTS["dino_boneyard"])
        self.assertEqual(state.stock_heading, desired)
        self.assertEqual(len(observation), npc.OBSERVATION_SIZE)
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in observation))

    def test_progress_delta_handles_forward_wrap_and_backward_motion(self):
        self.assertEqual(npc.progress_delta(340, 2, 342), 4)
        self.assertEqual(npc.progress_delta(100, 97, 342), -3)


if __name__ == "__main__":
    unittest.main()
