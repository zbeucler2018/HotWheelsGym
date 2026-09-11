import gzip
import importlib.util
import json
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
    return FakeMemory(payload[0x21000:0x61000])


class NPCControlTests(unittest.TestCase):
    integration = ROOT / "HotWheelsGym" / "HotWheelsStuntTrackChallenge-GbAdvance"

    def test_discovers_ordered_racers_in_every_multiplayer_state(self):
        states = sorted(self.integration.glob("*_multi.state"))
        self.assertGreater(len(states), 0)
        for path in states:
            with self.subTest(state=path.name):
                layout = npc.discover_race_layout(state_memory(path))
                self.assertEqual(
                    [racer.kind for racer in layout.racers],
                    ["player", "cpu", "cpu", "cpu"],
                )
                self.assertEqual(
                    [racer.vehicle_index for racer in layout.racers], [0, 1, 2, 3]
                )

    def test_boost_is_the_same_racer_local_field_on_every_track(self):
        for path in sorted(self.integration.glob("*_multi.state")):
            with self.subTest(state=path.name):
                memory = state_memory(path)
                race = npc.RaceMemory(memory)
                player = race.layout.racer(0)
                with path.with_suffix(".json").open() as handle:
                    configured = json.load(handle)["info"]["boost"]["address"]
                self.assertEqual(
                    player.address + npc.RACER_BOOST_OFFSET,
                    configured,
                )
                self.assertEqual(
                    race.state(0).boost,
                    memory.extract(configured, "<u4"),
                )

    def test_progress_delta_handles_forward_wrap_and_backward_motion(self):
        self.assertEqual(npc.progress_delta(340, 2, 342), 4)
        self.assertEqual(npc.progress_delta(100, 97, 342), -3)


if __name__ == "__main__":
    unittest.main()
