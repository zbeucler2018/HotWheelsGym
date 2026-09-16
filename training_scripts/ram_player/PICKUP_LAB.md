# Dino Boneyard pickup lab

The pickup lab isolates the first Jet Boost and the following hairpin into a
short Gymnasium task. It is deliberately separate from the full-race trainer:
the current race champion stays frozen while a small expert learns the one
maneuver that the full-race policies consistently miss.

## Pipeline

1. `capture_pickup_states.py` replays known policies and captures gzip emulator
   states at several distances before the first Jet Boost. Each state is
   labeled by whether its teacher collected the pickup. Successful action
   traces are saved for behavior cloning.
2. `train_pickup.py` behavior-clones those successful traces, then runs PPO on
   `DinoPickupHairpinTask`. A miss, Player 1 respawn, or timeout ends the short
episode. A successful episode must collect the pickup and reach progress 61
with useful exit speed. Timeout is intentionally more expensive than a miss,
and every raw frame has a cost, so stopping before the pickup is not a viable
reward shortcut.
3. Six fixed holdout states are evaluated deterministically every 25,000
   timesteps. Checkpoint selection prioritizes success rate, then pickup rate,
   then maneuver time and exit speed.
4. `PickupExpertCompositePolicy` can gate a successful expert into a frozen
   full-race policy shortly before the pickup and return control after the
   hairpin. A composite must beat the base policy in full-race evaluation
   before it can replace any existing champion.

Generated states, demonstrations, models, logs, and copied ROM data live under
`training_scripts/ram_runs/`, which is gitignored. They are derived from the
user-supplied ROM and must not be committed.

## Commands

Capture the private curriculum from the pickup-capable v5 policy and the fast
v9 policy:

```bash
uv run --no-sync python -m training_scripts.ram_player.capture_pickup_states \
  --rom rom.gba \
  --policy v5=training_scripts/ram_runs/dino_ram_player_jet_boost_skid_v5_20260911T025018Z/final_model.zip \
  --policy v9=training_scripts/ram_runs/dino_ram_player_self_play_v9_20260915T021134Z/evaluation_sweep_20260915T110125Z/fastest_model.zip \
  --output-dir training_scripts/ram_runs/private/dino_pickup_lab
```

Train the isolated expert:

```bash
uv run --no-sync python -m training_scripts.ram_player.train_pickup \
  --rom rom.gba \
  --config training_scripts/ram_player/dino_boneyard_pickup_expert_v11.yml
```

Compare the best expert against the unchanged full-race base and record the
composite race:

```bash
uv run --no-sync python -m training_scripts.ram_player.evaluate_pickup_composite \
  --rom rom.gba \
  --base-model path/to/base_model.zip \
  --expert-model path/to/pickup_expert.zip \
  --output training_scripts/ram_runs/private/composite_evaluation.json \
  --video training_scripts/ram_runs/private/composite_race.mp4
```

Audit which raw button sequences produce score/boost changes from identical
emulator states:

```bash
uv run --no-sync python -m training_scripts.ram_player.audit_mechanics \
  --rom rom.gba \
  --model path/to/model.zip \
  --output-dir training_scripts/ram_runs/private/dino_mechanics_audit \
  --lookback-decisions 30 \
  --pulse-frames 30 \
  --horizon-frames 240
```

The boost component is `L+R`; acceleration is a separate drive component. The
audit therefore tests pure `L+R` and `A+L+R` separately.

## Observation compatibility

The expert uses the existing 67-value observation-v7 contract. Racer Y and
score deltas are currently diagnostic `info` fields only, so this work does not
invalidate v7 checkpoints. The mechanics audit showed that a timed sequence of
existing factorized actions can earn trick score and boost while constant
button pulses cannot. A future observation version should consider normalized
height/vertical motion before optimizing tricks, but that is intentionally
outside this pickup-focused run.
