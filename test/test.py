import pytest

import HotWheelsGym
from HotWheelsGym.enums import RaceMode, Tracks

MAX_LAPS = 3
ALL_TRACKS = []

for tr in Tracks:
    for rm in RaceMode:
        for l in range(1, MAX_LAPS + 1):
            temp = f"HWSTC-{tr.value}-{rm.value}-{l}"
            ALL_TRACKS.append(temp)


@pytest.mark.parametrize("test_id", [*ALL_TRACKS])
def test_creation(test_id):
    _env = HotWheelsGym.make(test_id)
    _env.close()

@pytest.mark.parametrize("test_id", [*ALL_TRACKS])
def test_step(test_id):
    _env = HotWheelsGym.make(test_id, render_mode="rgb_array")
    _env.reset()
    for _ in range(5):
        _env.step(_env.action_space.sample())
    _env.close()