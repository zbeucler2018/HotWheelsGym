# Hot Wheels Gymnasium Environment
By: Zack Beucler
<p align="left">
  <img width="100%" src="./media/6-dbm_slow_20m_flawless.gif"></img>
</p>

**HotWheelsGym** is a [gymnasium](https://github.com/Farama-Foundation/Gymnasium) enviroment for the 2005 GBA game [Hot Wheels Stunt Track Challenge.](https://tcrf.net/Hot_Wheels:_Stunt_Track_Challenge_(Game_Boy_Advance))

# Notable features

- **Multiple Tracks:** _TRex Valley_ and _Dinosaur Boneyard_. More to come.
- **Two game modes:** _single_  where you race alone and _multi_ where you race 3 NPCs.

# Installation

- tbd

# Basic usage

```python3
import HotWheelsGym
from HotWheelsGym.enums import Tracks, RaceModes

# only need to import the ROM once
HotWheelsGym.import_rom("path/to/rom.gba")

env = HotWheelsGym.make(
  track=Tracks.Dino_Boneyard,
  mode=RaceModes.MULTI,
  laps=3
)
```