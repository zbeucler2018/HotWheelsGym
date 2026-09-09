# Dino Boneyard Player 1 RAM model

This is a separate training path from `training_scripts/train_ppo.py`. It trains
Player 1 from compact racer RAM, not pixels, and produces checkpoints that can
also replace selected native CPU controls for self-play.

The first run races the stock CPU opponents. Later runs can load any prior RAM
Player 1 checkpoint into CPU slots 1–3 as frozen opponents. The game still owns
physics, collision, rendering, and race management.

## Install and preflight

From the repository root:

```bash
python3 -m pip install -e '.[train,dev]'
python3 -m training_scripts.ram_player.smoke_test --rom rom.gba
```

The smoke test loads a configured historical state and takes a few fixed
actions. It does not train or save a model. Run it before an unattended job,
because the checked-in states use an older mGBA/HLE format and compatibility
depends on the installed Stable-Retro core.

The source ROM must remain local and uncommitted. `*.gba` and every RAM run
directory are ignored. For self-play, the runner privately generates the exact
selected-slot ROM under the ignored run directory.

## First overnight run

```bash
python3 -m training_scripts.ram_player.train --rom rom.gba
```

The default configuration is
`training_scripts/ram_player/dino_boneyard.yml`: 5 million PPO timesteps, three
emulator workers, four emulator frames per policy decision, and a 128×128 MLP.
This leaves one core available for PPO updates on the Intel N150.

Training episodes remain capped at 4,500 decisions, while start-line
evaluation and recording use the separate 6,000-decision horizon. This allows
slower early policies to finish without lengthening every training rollout.

Watch it with:

```bash
tensorboard --logdir training_scripts/ram_runs
```

Each timestamped run contains:

- TensorBoard PPO and `eval/*` metrics;
- per-worker Monitor CSV files;
- `evaluation/evaluations.csv`;
- `evaluation/best_model.zip`, selected by finish rate and then race time;
- periodic checkpoints and `final_model.zip`;
- `best_model.mp4` plus a JSON sidecar after successful training;
- resolved configuration, Git revision, ROM hashes, action/observation schema,
  and frozen-opponent hashes.

Evaluation runs at startup, every 250,000 timesteps, and at shutdown. It uses a
separate deterministic start-line race and records completion, finish rate,
raw finish frames, mean speed, and rank.

After the environments close, the trainer automatically records one
deterministic start-line race using `evaluation/best_model.zip`. The MP4 plays
at real-time speed (15 encoded frames per second for four-frame action repeat).
Pass `--no-record-video` only when this final recording is not wanted.

## Re-evaluate a completed run

The checkpoint sweep evaluates every periodic checkpoint plus `final_model.zip`
with a finish-capable horizon. It copies the fastest finisher, records a fresh
start-line MP4, and logs every checkpoint metric plus an accelerated replay to
TensorBoard:

```bash
python3 -m training_scripts.ram_player.sweep \
  --rom rom.gba \
  --run-dir training_scripts/ram_runs/RUN
```

Use TensorBoard's **Scalars** tab for checkpoint curves, **Text** for the result
table, and **Images** for the animated race replay. The full-quality MP4 and a
CSV/JSON report remain in the sweep directory.

## Observation and action contract

Every controlled racer—Player 1 or a CPU—gets the same 43 normalized floats:

- heading sine/cosine, absolute Dino X/Z, speed, and checkpoint phase;
- tracked lap and race rank;
- acceleration, turn rate, and progress rate;
- a seven-value one-hot encoding of the previous action;
- for the three nearest racers: ego-frame forward/right/distance, relative
  total progress, speed, heading sine/cosine, and relative lap.

There are seven discrete actions:

| Index | Intent | Player 1 buttons | Patched CPU interpretation |
| ---: | --- | --- | --- |
| 0 | coast | none | straight, target speed 0 |
| 1 | accelerate | A | straight, maximum target speed |
| 2 | accelerate left | A + Left | left heading request, maximum speed |
| 3 | accelerate right | A + Right | right heading request, maximum speed |
| 4 | brake | B | straight, target speed 0 |
| 5 | accelerate/up | A + Up | straight, maximum target speed |
| 6 | boost | A + L + R | straight, maximum target speed |

The CPU patch currently exposes heading and target speed, not its boost/trick
state. Actions 5 and 6 therefore degrade to straight acceleration when the same
checkpoint controls a CPU. Steering is relative to the racer's current heading,
so the policy—not the original waypoint AI—owns the CPU's line.

Loading a Player 1 checkpoint into a CPU slot proves the shared observation and
control path, but does not guarantee equal driving performance. Native Player 1
button steering and the patched CPU heading/target-speed boundary have different
dynamics. Always run `self_play_eval` before using a checkpoint as a curriculum
opponent; a model that is fast as Player 1 may still need a more symmetric
control translation or opponent-aware training before it is a strong NPC.

## Self-play against prior Player 1 models

Use the original `rom.gba`; the command creates and imports an ignored private
patch containing exactly the selected CPU slots:

```bash
python3 -m training_scripts.ram_player.train \
  --rom rom.gba \
  --run-name dino_ram_selfplay_1 \
  --opponent 1=training_scripts/ram_runs/PRIOR_RUN/evaluation/best_model.zip
```

Repeat `--opponent` to fill more slots. Each worker loads frozen copies. Player
1 continues learning while the opponent checkpoints remain unchanged.

You can also configure the mapping under `opponents` in the YAML file. The same
wrapper is reusable in code:

```python
base = HotWheelsGym.make("HWSTC-dino_boneyard-multi-3", render_mode="rgb_array")
env = HotWheelsGym.DinoRAMModelOpponentEnv(base, {1: frozen_ram_model})
env = HotWheelsGym.DinoRAMPlayerEnv(env)
```

## Compare with the best pixel model

After training, evaluate both models as Player 1 on separate start-line races:

```bash
python3 -m training_scripts.ram_player.compare \
  --rom rom.gba \
  --ram-model training_scripts/ram_runs/RUN/evaluation/best_model.zip
```

The comparison defaults to `zoo/dbm_basic/best_model.zip`, runs five races per
model, and writes episode CSV, summary JSON, and TensorBoard scalars under
`training_scripts/ram_runs/comparison_*`. Rewards are retained for debugging,
but the meaningful comparison is finish rate, raw frames/time, completion, and
rank because the pixel and RAM reward functions differ. It also records one
MP4 per model and embeds accelerated replays in TensorBoard's **Images** tab.

## Validate a checkpoint as an NPC

This records the selected RAM model driving both Player 1 and one patched CPU
slot, with separate Player/NPC telemetry and a TensorBoard replay:

```bash
python3 -m training_scripts.ram_player.self_play_eval \
  --rom rom.gba \
  --player-model training_scripts/ram_runs/RUN/evaluation_sweep_RUN/fastest_model.zip
```

The generated patched ROM stays inside the ignored output directory. The
source ROM and generated ROM are never added to Git.
