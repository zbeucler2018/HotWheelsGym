import gzip
import json
from pathlib import Path
import struct
import subprocess
import sys
import unittest

import numpy as np

from HotWheelsGym.enums import RaceMode, Tracks
from HotWheelsGym.npc_control import (
    EWRAM_BASE,
    RaceMemory,
    discover_power_up_addresses,
    read_power_up_states,
)
from HotWheelsGym.ram_opponent_control import (
    RAM_DEFAULT_ACTION,
    RAM_OBSERVATION_NAMES,
    RaceProgressTracker,
    build_dino_ram_observation,
    build_track_ram_observation,
    ordered_other_slots,
    read_racer_states,
)
from HotWheelsGym.track_reference import track_reference_profile
from HotWheelsGym.visualizer import ObservationVisualizerAdapter
from HotWheelsGym.visualizer.server import encode_png

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "HotWheelsGym" / "HotWheelsStuntTrackChallenge-GbAdvance"


class FakeMemory:
    def __init__(self, ewram: bytes):
        self._ewram = bytearray(ewram)

    @property
    def blocks(self):
        return {EWRAM_BASE: bytes(self._ewram)}

    def extract(self, address, data_type):
        offset = address - EWRAM_BASE
        formats = {"|u1": "<B", "<u2": "<H", "<u4": "<I"}
        return struct.unpack_from(formats[data_type], self._ewram, offset)[0]


def state_memory(track: Tracks) -> FakeMemory:
    path = INTEGRATION / f"{track.value}_multi.state"
    with gzip.open(path, "rb") as handle:
        payload = handle.read()
    return FakeMemory(payload[0x21000:0x61000])


class FakeData:
    def __init__(self, memory):
        self.memory = memory


class FakeMultiEnv:
    def __init__(self, track: Tracks):
        self.unwrapped = self
        self.track = track
        self.mode = RaceMode.MULTI
        self.total_laps = 3
        self.statename = str(INTEGRATION / f"{track.value}_multi.state")
        self.initial_state = b""
        self.data = FakeData(state_memory(track))
        self.buttons = ["A", "B", "SELECT", "START", "RIGHT", "LEFT", "UP", "DOWN", "R", "L"]
        self.frame = np.zeros((160, 240, 3), dtype=np.uint8)

    def reset(self):
        return self.frame.copy(), {}

    def close(self):
        pass


class VisualizerAdapterTests(unittest.TestCase):
    def _expected(self, adapter, slot):
        race = RaceMemory(adapter.base.data.memory)
        states = read_racer_states(race)
        power_ups = read_power_up_states(
            race.memory, discover_power_up_addresses(race.memory)
        )
        tracker = RaceProgressTracker.from_states(states, adapter.progress_count)
        if adapter.track is Tracks.Dino_Boneyard:
            values = build_dino_ram_observation(
                states,
                states,
                tracker,
                slot,
                RAM_DEFAULT_ACTION,
                power_ups=power_ups,
            )
        else:
            values = build_track_ram_observation(
                states,
                states,
                tracker,
                slot,
                RAM_DEFAULT_ACTION,
                profile=track_reference_profile(adapter.track),
                power_ups=power_ups,
            )
        return values, states, tracker

    def test_snapshot_exactly_matches_policy_observation_for_every_track_and_slot(self):
        for track in Tracks:
            adapter = ObservationVisualizerAdapter(FakeMultiEnv(track))
            adapter.reset()
            for slot in range(4):
                with self.subTest(track=track.value, slot=slot):
                    snapshot = adapter.set_controlled_slot(slot)
                    expected, states, tracker = self._expected(adapter, slot)
                    self.assertEqual(snapshot.observation_names, RAM_OBSERVATION_NAMES)
                    self.assertEqual(snapshot.observation_values, expected)
                    self.assertEqual(len(snapshot.observation_values), 67)
                    self.assertEqual(
                        snapshot.nearby_order,
                        ordered_other_slots(slot, states, tracker),
                    )

    def test_snapshot_geometry_is_the_geometry_in_the_observation(self):
        for track in (Tracks.Dino_Boneyard, Tracks.TRex_Valley):
            with self.subTest(track=track.value):
                snapshot = ObservationVisualizerAdapter(FakeMultiEnv(track)).reset()
                named = snapshot.named_observation
                geometry = snapshot.geometry
                self.assertEqual(
                    geometry["lateral_offset"], named["track_lateral_offset"]
                )
                self.assertEqual(
                    geometry["heading_error"]["sin"],
                    named["track_heading_error_sin"],
                )
                self.assertEqual(
                    geometry["heading_error"]["cos"],
                    named["track_heading_error_cos"],
                )
                for target in geometry["lookahead_targets"]:
                    label = target["label"]
                    self.assertEqual(
                        target["observation"]["forward"],
                        named[f"track_target_{label}_forward"],
                    )
                    self.assertEqual(
                        target["observation"]["right"],
                        named[f"track_target_{label}_right"],
                    )

    def test_dino_snapshot_does_not_change_dino_policy_semantics(self):
        adapter = ObservationVisualizerAdapter(FakeMultiEnv(Tracks.Dino_Boneyard))
        snapshot = adapter.reset()
        expected, _, _ = self._expected(adapter, 0)
        self.assertEqual(snapshot.observation_values, expected)

    def test_imports_are_isolated_from_normal_runtime(self):
        code = """
import json, sys
import HotWheelsGym
print(json.dumps({
    'visualizer': 'HotWheelsGym.visualizer' in sys.modules,
    'server': 'HotWheelsGym.visualizer.server' in sys.modules,
    'fastapi': 'fastapi' in sys.modules,
}))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        loaded = json.loads(result.stdout)
        self.assertEqual(
            loaded, {"visualizer": False, "server": False, "fastapi": False}
        )

    def test_visualizer_public_import_does_not_import_server(self):
        code = """
import json, sys
import HotWheelsGym.visualizer
print(json.dumps({
    'server': 'HotWheelsGym.visualizer.server' in sys.modules,
    'fastapi': 'fastapi' in sys.modules,
}))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        loaded = json.loads(result.stdout)
        self.assertEqual(loaded, {"server": False, "fastapi": False})

    def test_framebuffer_png_encoder_is_dependency_free_and_valid(self):
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        frame[0, 0] = (255, 64, 32)
        encoded = encode_png(frame)
        self.assertTrue(encoded.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn(b"IHDR", encoded)
        self.assertTrue(encoded.endswith(b"IEND\xaeB`\x82"))


if __name__ == "__main__":
    unittest.main()
