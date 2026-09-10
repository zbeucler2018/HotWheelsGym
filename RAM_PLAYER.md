# Dino Boneyard Player 1 RAM model

This is a separate training path from `training_scripts/train_ppo.py`. It trains
Player 1 from compact racer RAM, not pixels, and produces checkpoints that can
also replace the stock opponents with model-controlled player-class racers.

The first run races the stock CPU opponents. Later runs can load any prior RAM
Player 1 checkpoint into slots 1–3 as frozen opponents. The game still owns
physics, collision, rendering, and race management.

## Install and preflight

From the repository root:

```bash
uv venv
uv pip install -e '.[train,dev]'
uv run --no-project python -m training_scripts.ram_player.smoke_test --rom rom.gba
```

The smoke test loads a configured historical state and takes a few fixed
actions. It does not train or save a model. Run it before an unattended job,
because the checked-in states use an older mGBA/HLE format and compatibility
depends on the installed Stable-Retro core.

The source ROM must remain local and uncommitted. `*.gba` and every RAM run
directory are ignored. For self-play, the runner privately generates the exact
native-button ROM under the ignored run directory.

## First overnight run

```bash
uv run --no-project python -m training_scripts.ram_player.train --rom rom.gba
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
raw finish frames, mean speed, rank, centerline error, heading alignment,
wall-contact rate, respawn count, mean boost charge, boost gained/spent, and
the fraction of emulator frames on which boost was requested while charge was
available. Completed-lap times appear as `mean_lap_1_seconds`,
`mean_lap_2_seconds`, and `mean_lap_3_seconds` in TensorBoard and the evaluation
CSV.

After the environments close, the trainer automatically records one
deterministic start-line race using `evaluation/best_model.zip`. The MP4 plays
at real-time speed (15 encoded frames per second for four-frame action repeat).
Its JSON sidecar records final/mean boost charge, charge gained/spent, and boost
active frames in addition to exact lap-split frames/seconds, race outcome, and
track-quality metrics.
Pass `--no-record-video` only when this final recording is not wanted.

## Validated project checkpoint (2026-09-10)

The first boost-aware observation-v3 run completed from scratch against the
stock opponents at 5 million PPO timesteps. A deterministic start-line sweep
then evaluated all 20 periodic checkpoints plus `final_model.zip` with the
6,000-decision finish horizon. The retained winner was the 2,999,988-step
checkpoint:

- three-lap time: **4:32.52** (`16,351` raw frames at 60 Hz);
- lap splits: **1:32.37**, **1:28.73**, and **1:31.42**;
- result: first place with one Player 1 respawn;
- mean boost charge: `0.484` of capacity;
- boost requested while charged on `7.22%` of emulator frames;
- boost charge gained/spent: `6,396` / `1,472`.

The 4.25-million-step checkpoint produced the fastest individual evaluated lap
at **1:27.35**, but its **4:34.40** total was slower. The sweep reproduced the
same 3-million-step winner selected during training and confirmed substantial
late-training PPO regression: the separately saved final model ran **4:50.80**
and finished third.

For context, the previous observation-v2 RAM model ran **4:27.98**, the best
known pixel agent ran about **4:24**, and the supplied console reference was
**4:22.15**. Observation v3 therefore succeeded at making boost state visible
and produced intentional boost use, but did not improve total race time. Its
near-reference single-lap pace suggests that consistency, wall avoidance, and
eliminating the respawn are more valuable next targets than merely increasing
boost use.

The local ignored artifacts are under:

```text
training_scripts/ram_runs/dino_ram_player_boost_v3_20260910T031248Z/
  evaluation_sweep_20260910T181225Z/
    checkpoint_results.csv
    fastest_model.zip
    fastest_model.mp4
    fastest_model.json
    summary.json
```

At this checkpoint, symmetric native-button control is proven for Player 1 and
all three model-controlled opponent slots. They receive the same racer-centric
58-float observation and seven-action contract, while the original game owns
physics, collision, race state, and rendering. Opponent-only respawn flashes
are isolated so they no longer reset or blank Player 1's observation. A frozen
model can already occupy all three opponent slots for evaluation; a learning
self-play training run has not yet been performed.

Recommended continuation:

1. add sector timing and record the progress location of walls, stalls, and
   respawns;
2. resume the 3-million-step winner with smaller PPO updates and oversample the
   slow/failure sectors before full-race fine-tuning;
3. rank checkpoints over multiple start-line races, preferring reliable
   zero-respawn finishes and then median finish time;
4. perform a small portability audit on one track with an additional power-up,
   moving centerline, progress, state, and power-up differences into per-track
   configuration rather than duplicating the environment;
5. introduce a frozen-checkpoint opponent pool only after the solo policy can
   repeat clean races near or below the pixel baseline.

## Re-evaluate a completed run

The checkpoint sweep evaluates every periodic checkpoint plus `final_model.zip`
with a finish-capable horizon. It copies the fastest finisher, records a fresh
start-line MP4, and logs every checkpoint metric to TensorBoard:

```bash
python3 -m training_scripts.ram_player.sweep \
  --rom rom.gba \
  --run-dir training_scripts/ram_runs/RUN
```

Use TensorBoard's **Scalars** tab for checkpoint curves and **Text** for the
result table. The full-quality MP4 and a CSV/JSON report remain in the sweep
directory; videos are intentionally not encoded as TensorBoard images.

## Observation and action contract

Every controlled racer—Player 1 or an opponent—gets the same 58 normalized
floats, rotated so that racer is always the observation's ego:

- heading sine/cosine, absolute Dino X/Z, speed, boost charge, and checkpoint phase;
- tracked lap and race rank;
- acceleration, turn rate, and progress rate;
- signed centerline offset and heading error relative to the road;
- ego-relative directions to short, medium, and long lookahead points;
- medium- and long-range signed track curvature;
- a seven-value one-hot encoding of the previous action;
- for the three nearest racers: ego-frame forward/right/distance, relative
  total progress, speed, boost charge, heading sine/cosine, and relative lap.

The Dino reference line contains one X/Z point per modulo-342 progress unit. It
was derived from median native-racer telemetry over repeated laps; no ROM bytes
or images are stored. Each live racer is projected onto nearby reference-line
segments, so Player 1 and every model opponent get the same local geometry even
when they occupy different parts of the track.

This is observation contract version 3. The racer-local boost meter at `+0xF0`
is normalized from 0 to its maximum of 980. The same offset was verified against
the configured Player 1 boost address on every bundled multiplayer track and
against all four Dino racer objects. The neighboring `+0xEC` field is only a
lagging display value and is intentionally excluded.

Previous 43-input v1 and 54-input v2 checkpoints are intentionally incompatible:
their neural-network input layers cannot accept the new features. The tools
detect both formats and report a clear migration error. Train the first v3 model
from scratch; subsequent v3 checkpoints can be used symmetrically for self-play.

Dino Boneyard has boost but no alternate power-up inventory, so v3 does not
invent a power-up type field. Tracks with additional power-ups will need their
racer-local inventory/type fields mapped and appended in a later track-specific
observation contract.

The reward remains progress-dominant and finish-aware. Small shaping penalties
now discourage wall contact, large centerline error, wrong-way alignment, and a
new Player 1 respawn. Evaluation logs mean lateral error, heading alignment,
wall-contact rate, and respawns alongside finish time, completion, and rank.

There are seven discrete actions:

| Index | Intent | Player 1 and opponent buttons |
| ---: | --- | --- | --- |
| 0 | coast | none |
| 1 | accelerate | A |
| 2 | accelerate left | A + Left |
| 3 | accelerate right | A + Right |
| 4 | brake | B |
| 5 | accelerate/up | A + Up |
| 6 | boost while accelerating | A + L + R |

The game's boost chord itself is **L+R**. The shared controller action also
holds **A** so choosing boost does not release the accelerator; this is why the
policy metadata records `A + L + R`.

The native-button ROM makes all four race slots genuine player-class racers.
Player 1 continues sampling the hardware keypad; slots 1–3 instead retain the
independent pressed/released/held masks written by Python. This means steering,
braking, boost, physics, collisions, and lap handling all use the exact same
game code for Player 1 and model opponents.

### What the local ROM patch changes

`prepare_rom()` never edits the supplied `rom.gba` in place. When model
opponents are requested, it copies the source into the ignored run-private
directory and applies three narrowly scoped changes to that copy:

1. both race-manager construction paths create four player-class racers and
   zero CPU-class racers;
2. the Player input-preparation call is redirected through a small Thumb hook;
3. vehicle index 0 tail-calls the original hardware-keypad routine, while
   indices 1–3 return without overwriting their racer-local button fields.

Python writes each model action as the same pressed/released/held GBA masks at
racer offsets `+0x302`, `+0x304`, and `+0x306`. From there, the original player
update, physics, collisions, tricks, lap logic, and rendering run normally. A
marker and expected SHA-1 identify the generated patch. The original and
generated ROMs are ignored by Git and must never be committed.

## Self-play against prior Player 1 models

Stable-Retro savestates restore racer objects after construction. Therefore an
old checked-in multiplayer state still contains stock CPU objects even with the
new ROM. First create one private Dino Boneyard start-line state under the
patched ROM. This drives a known Player 1 checkpoint through the source race,
selects Retry, and saves only after all four player-class racers have valid grid
coordinates; it performs no training:

```bash
uv run --no-project python -m training_scripts.ram_player.create_opponent_state \
  --rom rom.gba \
  --player-model training_scripts/ram_runs/PRIOR_RUN/evaluation/best_model.zip \
  --output training_scripts/ram_runs/private/dino_four_players.state
```

Both the generated ROM and state stay under the ignored `ram_runs` tree. Use
the original `rom.gba` for the trainer; it privately generates and imports the
native-button ROM, then loads the four-player state:

```bash
uv run --no-project python -m training_scripts.ram_player.train \
  --rom rom.gba \
  --run-name dino_ram_selfplay_1 \
  --opponent 1=training_scripts/ram_runs/PRIOR_RUN/evaluation/best_model.zip \
  --opponent 2=training_scripts/ram_runs/PRIOR_RUN/evaluation/best_model.zip \
  --opponent 3=training_scripts/ram_runs/PRIOR_RUN/evaluation/best_model.zip \
  --opponent-state training_scripts/ram_runs/private/dino_four_players.state
```

All three opponent slots must be assigned because the symmetric ROM converts
all of them to externally controlled player-class racers. Reusing one checkpoint
for all three is supported, as shown above. Each worker loads frozen copies;
Player 1 continues learning while the opponent checkpoints remain unchanged.

You can also configure the mapping under `opponents` in the YAML file. The same
wrapper is reusable in code:

```python
base = HotWheelsGym.make("HWSTC-dino_boneyard-multi-3", render_mode="rgb_array")
env = HotWheelsGym.DinoRAMModelOpponentEnv(
    base,
    {1: frozen_ram_model, 2: frozen_ram_model, 3: frozen_ram_model},
)
env = HotWheelsGym.DinoRAMPlayerEnv(env)
```

Player-class respawns normally flash the shared GBA framebuffer white. The
opponent wrapper identifies the game's per-racer respawn-pending flag and, for
an opponent-only respawn, holds the last visible frame until the flash ends.
This keeps a pixel-based Player 1 policy from receiving blank observations and
keeps evaluation video readable without changing racer physics or positions.
A genuine Player 1 respawn is not masked. Set
`mask_opponent_respawn_flashes=False` on `DinoRAMModelOpponentEnv` to expose the
original framebuffer behavior. The info dictionary reports the current mask
and cumulative masked-frame count under `ram_opponent_respawn_flash_masked` and
`ram_opponent_respawn_flash_frames`.

## Compare with the best pixel model

After training, evaluate both models as Player 1 on separate start-line races:

```bash
uv run --no-project python -m training_scripts.ram_player.compare \
  --rom rom.gba \
  --ram-model training_scripts/ram_runs/RUN/evaluation/best_model.zip
```

The comparison defaults to `zoo/dbm_basic/best_model.zip`, runs five races per
model, and writes episode CSV, summary JSON, and TensorBoard scalars under
`training_scripts/ram_runs/comparison_*`. Rewards are retained for debugging,
but the meaningful comparison is finish rate, raw frames/time, completion, and
rank because the pixel and RAM reward functions differ. It also records one
MP4 per model on disk without embedding video frames in TensorBoard.

## Validate a checkpoint as an NPC

This records the selected RAM model driving Player 1 and all three opponent
slots, with per-racer TensorBoard telemetry and an MP4 on disk:

```bash
uv run --no-project python -m training_scripts.ram_player.self_play_eval \
  --rom rom.gba \
  --player-model training_scripts/ram_runs/RUN/evaluation_sweep_RUN/fastest_model.zip \
  --state training_scripts/ram_runs/private/dino_four_players.state \
  --max-episode-steps 7000
```

The generated patched ROM stays inside the ignored output directory. The
source ROM and generated ROM are never added to Git.
