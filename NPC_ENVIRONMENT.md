# Native NPC training and model opponents

> This document describes the earlier low-level CPU heading/speed interface.
> For the proven symmetric path—identical 58-float RAM observations and native
> GBA button controls for Player 1 and opponents—use [RAM_PLAYER.md](RAM_PLAYER.md).

HotWheelsGym can expose one or more of the game's native CPU racers to Python.
The game still owns physics, collision, turning, animation, and race management;
the policy supplies only a desired heading and target speed.

No ROM is stored in this repository. Generate a local patched ROM from the
matching original ROM:

```bash
PYTHONPATH=hotwheels-re-tools/src \
  python3 -m hotwheels_re_tools patch-npc-control \
  rom.gba /tmp/hotwheels-npc.gba \
  --vehicle-index 1
```

Repeat `--vehicle-index` to expose more opponents. The bundled multiplayer
states consistently map racer slots 1, 2, and 3 to vehicle indices 1, 2, and 3.
Unselected CPU racers retain the original AI.

Import the generated ROM explicitly. `force=True` replaces a previously
imported original or generated ROM in the local custom integration directory.

```python
import HotWheelsGym

HotWheelsGym.import_rom("/tmp/hotwheels-npc.gba", force=True)
```

## Train an NPC

```python
import HotWheelsGym

env = HotWheelsGym.make_npc(
    "HWSTC-dino_boneyard-multi-3",
    npc_slot=1,
    render_mode="rgb_array",
)
observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step([0.0, 1.0])
```

The action is `[steering, throttle]`:

- steering is `-1..1`, mapped relative to the racer's current heading with a
  default maximum request of 45 degrees per environment step;
- throttle is `0..1`, mapped to the native fixed-point target speed.

The observation contains 12 normalized values:

1. sine and cosine of current heading;
2. sine and cosine of the original AI's waypoint heading;
3. current speed and target speed;
4. track progress and progress relative to Player 1;
5. Player 1's relative X/Z position;
6. race rank and current steering error.

`info` adds `npc_*` telemetry including progress, lap, rank, speed, current
heading, requested heading, and the stock waypoint heading. Reward is signed
checkpoint progress with a small speed bonus.

The ready-to-run PPO example uses an MLP policy:

```bash
python training_scripts/train_npc_ppo.py \
  --rom /tmp/hotwheels-npc.gba \
  --slot 1 \
  --timesteps 1000000 \
  --output zoo/my_npc
```

Frame skipping should wrap `HotWheelsNPCEnv`, as the example does, so the
command is refreshed on every emulator subframe.

## Race against trained models

Generate a ROM exposing every slot that will have a model:

```bash
PYTHONPATH=hotwheels-re-tools/src \
  python3 -m hotwheels_re_tools patch-npc-control \
  rom.gba /tmp/hotwheels-opponents.gba \
  --vehicle-index 1 --vehicle-index 2
```

Then run the interactive Player 1 frontend:

```bash
python training_scripts/play_against_models.py \
  --rom /tmp/hotwheels-opponents.gba \
  --id HWSTC-dino_boneyard-multi-3 \
  --opponent 1=zoo/my_npc.zip \
  --opponent 2=zoo/another_npc.zip
```

Player 1 keeps the existing keyboard controls; each model is evaluated once
per emulator frame and writes only its assigned native CPU slot. You can also
construct `ModelOpponentEnv(base_env, {1: model, 2: model})` directly around a
bare `HotWheelsEnv` for another controller or UI.

The existing models in `zoo/` were trained as Player 1 pixel/button policies.
They are not directly compatible with the NPC telemetry/action interface and
must be retrained or adapted.

## Safety checks and compatibility

The wrappers discover the active manager and racer objects after every reset;
they do not hardcode track-specific RAM addresses. They also verify the ROM's
embedded selected-slot marker, which catches an unpatched or incorrectly
selected ROM before training begins.

The checked-in states use an older mGBA/HLE savestate format. If a current
Stable-Retro build cannot resume them faithfully, use the project's compatible
Stable-Retro/core version before spending time on training. The native probe's
`--resume-old-hle` option is a research workaround, not the production path.
