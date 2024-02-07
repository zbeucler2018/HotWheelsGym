import pytest

import HotWheelsGym
from HotWheelsGym.enums import RaceMode, Tracks

MAX_LAPS = 3
@pytest.fixture(autouse=True)
def generate_env_ids():
    ALL_TRACKS = []

    for tr in Tracks:
        for rc in RaceMode:
            for l in range(1, MAX_LAPS + 1):
                temp = f"{tr.value}-{rc.value}-{l}"
                ALL_TRACKS.append(temp)
    return ALL_TRACKS


# @pytest.fixture(autouse=True)
# def create_env()


def test_env_creation(generate_env_ids):
    for id in generate_env_ids:
        env = HotWheelsGym.make(id=id)
        env.close()

def test_env_step(generate_env_ids):
    for id in generate_env_ids:
        env = HotWheelsGym.make(id=id, render_mode="rgb_array")

        _ = env.reset()
        _ = env.step(env.action_space.sample())
        env.close()
