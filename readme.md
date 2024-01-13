# Hot Wheels Gymnasium Environment
By: Zack Beucler
<p align="left">
  <img width="100%" src="./media/6-dbm_slow_20m_flawless.gif"></img>
</p>

**HotWheelsGym** is a [gymnasium](https://github.com/Farama-Foundation/Gymnasium) environment for the 2005 GBA game [Hot Wheels Stunt Track Challenge.](https://tcrf.net/Hot_Wheels:_Stunt_Track_Challenge_(Game_Boy_Advance))

# Notable features

- **Multiple Tracks:** _TRex Valley_, _Dino Boneyard_, _Black Widows Nest_, and _Insect Hive_. More to come.
- **Two game modes:** _single_  where you race alone and _multi_ where you race against 3 NPCs.

# Installation

```bash
git clone https://github.com/zbeucler2018/HotWheelsGym.git
cd HotWheelsGym
pip install -e . # [train,dev]
```

# Basic usage

```python
import HotWheelsGym
from HotWheelsGym import HotWheelsEnv, Tracks, RaceModes

# only need to import the ROM once
HotWheelsGym.import_rom("path/to/rom.gba")

env = HotWheelsEnv(
  track=Tracks.Dino_Boneyard,
  mode=RaceModes.MULTI,
  laps=3
)

# OR

env = HotWheelsGym.make("HWSTC-dino_boneyard-multi-3")
```

# Environment

Use the template
```
HWSTC-<track>-<mode>-<laps>
```
where:

- `<track>` is the name of the track
- `<mode>` is the game mode {`single`, `multi`}
- `<laps>` is the total amount of laps {`1`, `2`, `3`}

## Track Varients

| Name | Map |
| ---- | ----- |
| `trex_valley` | <img height="200px" width="200px" src="media/trex_valley_minimap.png"/> |
| `dino_boneyard` | <img height="200px" width="200px" src="media/dino_boneyard_minimap.png"/> |
| `black_widows_nest` | <img height="200px" width="200px" src="media/black_widows_nest_minimap.png" /> |
| `insect_hive` | <img height="200px" width="200px" src="media/insect_hive_minimap.png" /> |
| `monsters_of_the_deep` | <img height="200px" width="200px" src="media/monsters_of_the_deep_minimap.png" /> |
| `whiteskull_cliffs` | <img height="200px" width="200px" src="media/whiteskull_cliffs_minimap.png" /> |
| `jungle_snakepit` | <img height="200px" width="200px" src="media/jungle_snakepit_minimap.png" /> |
| `gator_forest` | <img height="200px" width="200px" src="media/gator_forest_minimap.png" /> |
| `satellite_mission` | <img height="200px" width="200px" src="media/satellite_mission_minimap.png" /> |
| `solar_strip` | <img height="200px" width="200px" src="media/solar_strip_minimap.png" /> |
| `fire_mountain` | <img height="200px" width="200px" src="media/fire_mountain_minimap.png" /> |
| `volcano_battle` | <img height="200px" width="200px" src="media/volcano_battle_minimap.png" /> |


## Game Mode Varients

| Mode | Description |
| ------ | ---- |
| `single` | Single player. Race by yourself. |
| `multi`  | Multi player. Race against 3 NPCs. |

## Information returned from the environment

- The `info` dict returned by the `step` function contains the following keys:

| Key | Type | Description |
| --- | ---- | ----------- |
| `boost` | `int` | The current amount of boost the agent has. `980` is the max and will allow the agent to use the boost. |
| `hit_wall` | `int` | If the agent currently collided with a wall, the value is `1`, otherwise it is `0`. |
| `lap` | `int` | The current lap the agent is on. |
| `progress` | `int` | The agent's current progress through the track. |
| `rank` | `int` | The current rank of the agent in the race. `multi` mode only. |
| `score` | `int` | The current score of the agent. |
| `speed` | `int` | The agent's current (estimated) speed. |

### Progress per lap

- The progress value at each lap

| Lap # | `trex_valley` | `dino_boneyard` | `black_widows_nest` | `insect_hive` | `monsters_of_the_deep` | `whiteskull_cliffs` | `jungle_snakepit` | `gator_forest` | `satellite_mission` | `solar_strip` | `fire_mountain` | `volcano_battle` |
| ----- | ------------- | --------------- | ------------------- | ------------- | ---------------------- | ------------------- | ----------------- | -------------- | --- | --- | --- | --- |
| `1` | 317 | 342 | 395 | 380 | 343 | 340 | 465 | 512 | 376 | 325 | 465 | 495 |
| `2` | 633 | 684 | 789 | 759 | 685 | 679 | 930 | 1024 | 752 | 650 | 930 | 990 |
| `3` | 949 | 1024 | 1183 | 1138 | 1027 | 1018 | 1395 | 1536 | 1128 | 975 | 1395 | 1485 |
