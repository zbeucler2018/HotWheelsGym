# Hot Wheels NPC Reverse-Engineering Handoff

Branch: `npc-reverse-engineering`

## Goal

Make a trained HotWheelsGym policy occupy one of the stock CPU-racer slots in **Hot Wheels Stunt Track Challenge (GBA)** so a human player can race the model using the original game engine, physics, collisions, rendering, lap logic, and race management.

Desired architecture:

```text
Human controller -> Player 1 control state --\
                                         -> native racer update -> native physics/race
PPO action ------> CPU racer control ----/
```

Avoid direct position/velocity manipulation unless no cleaner control interface exists.

---

## ROM

The user supplied a local ROM named `rom.gba` for analysis. **Do not commit or redistribute it.** Keep it outside the repo or under an ignored private path.

The uploaded ROM was validated against the historical `rom.sha` used by the project. Treat the old Stable-Retro integrations and save states as belonging to this exact build.

Historical ROM/repo context:

- Current repo: `zbeucler2018/HotWheelsGym`
- Older repo with useful RAM archaeology: `zbeucler2018/HotWheelsRL`
- Older `HotWheelsRL` save states include Dino Boneyard multiplayer snapshots such as progress `71`, `156`, `180`, `290`.
- `generalized-agent` in HotWheelsGym contains a larger state pool under:
  `training_scripts/data/states`
- For generated state filenames, the numeric suffix is the **car progress value for that save on that track**. This is useful ground truth when decoding savestates.

---

## Confirmed historical RAM anchors

Dino Boneyard multiplayer historical integration:

| Meaning | Decimal | Hex | Historical type |
|---|---:|---:|---|
| boost | 33583272 | `0x020070A8` | `<=i4` |
| speed | 33583265 | `0x020070A1` | `<u2` |
| hit_wall / packed race state | 33583344 | `0x020070F0` | `|i1` historically |
| rank | 33583345 | `0x020070F1` | `><n1` |
| lap | 33583346 | `0x020070F2` | `|u1` |
| progress / checkpoint | 33583360 | `0x02007100` | `=i2` |
| player score | 33583624 | `0x02007208` | `<=i4` |
| NPC score | 33586198 | `0x02007C16` | `>=u4` |
| older alternate rank candidate | 33590624 | `0x02008D60` | `|i1` |

Important: Zack explicitly remembers that `npc_score` at `0x02007C16` tracked an **actual NPC's score**. The old notes said `"seems to be all of theirs??"`, so its exact semantic identity still needs to be resolved (specific NPC slot vs manager/current-opponent aggregation).

Historical note from HotWheelsRL also recorded a relationship involving the constant **1100** in the packed rank/lap/wall state, e.g. `(rank * 1100) + lap` under one condition. This constant later appeared in static ROM-code searches and may help identify race-state logic.

---

## Repository archaeology already done

HotWheelsGym commit:

- `740684fd2a61942f869c45a983908b6943af3093`
- message: `remove npc_score var from dino boneyard`
- removed:

```json
"npc_score": {
  "address": 33586198,
  "type": ">=u4"
}
```

Parent:

- `f4b6379eb4c69c83a44f4c984b45e4b401811f35`

Old HotWheelsRL notes explicitly contain:

```text
npc_score: 2007c16, >=u4 (seems to be all of theirs??)
```

and note the same address on another Dino multiplayer state/track configuration.

Historical labels changed during reverse engineering. For example, early files temporarily called `0x02007208` progress and `0x02007100` score; later mappings established these as player score and progress/checkpoint respectively. Do not assume old variable names are semantically perfect without dynamic validation.

---

## Static ROM findings from this investigation

These are the most important findings to revalidate immediately in Codex with a real ARM/Thumb disassembler/debugger. They came from the current binary analysis and are strong leads, but should be treated as reverse-engineering hypotheses until reproduced locally.

### 1. Hardware input path

The game reads the GBA keypad register:

- `REG_KEYINPUT = 0x04000130`
- observed ROM routine around: `0x080F27C8`
- active-low key bits are inverted/normalized and then passed into an input-state updater around: `0x080F25B8`

This gives a concrete way to trace **human buttons forward** into racer-local state.

### 2. Race manager / racer allocation

A race-manager initialization cluster was identified around `0x080FBxxx`.

Observed structure hypothesis:

- racer count around manager `+0x448`
- racer pointer array around manager `+0x450`
- approximately `0x368` bytes allocated per racer
- per-racer constructor/init call around:
  `0x08101618(racer, racer_index, race_manager)`

This is a major structural lead: racers appear to be separate runtime objects, which explains why grepping the ROM for absolute `0x02007xxx` telemetry addresses was not productive.

### 3. Racer vtable / methods

Candidate racer vtable:

- around ROM `0x0817C430`

Candidate methods observed from that table:

- `0x0810304C`
- `0x08101D40`
- `0x08101E28`
- `0x08104300`
- `0x08103E8C`
- `0x08102580`

The large recurring racer update appears to be around `0x0810304C`.

### 4. Racer-local control fields

A control/input-preparation path around `0x081025D4` appears to write/copy button/control masks into racer-local fields near:

- `racer + 0x302`
- `racer + 0x304`
- `racer + 0x306`

Working hypothesis: these are three related button-state masks such as held / newly pressed / released (or equivalent current/transition masks).

The important architectural observation is that the human input path and another indexed/per-racer input path appeared to converge on these same racer-local fields before the broader racer update.

**This is currently the leading injection candidate.**

The next job is to determine exactly:

1. what each of `+0x302/+0x304/+0x306` means,
2. whether stock NPC AI writes an upstream per-racer control record that feeds these fields,
3. whether overwriting these fields each frame is sufficient to drive an NPC while preserving native physics.

---

## Why savestates are valuable

The `.state` files fetched from GitHub are real compressed binary savestates; one inspected Dino state began with base64 `H4sI...`, i.e. gzip data rather than a Git LFS pointer.

Best route in a local Codex environment:

1. clone both repos with full history,
2. checkout `npc-reverse-engineering`,
3. collect Dino Boneyard multiplayer states from `HotWheelsRL` plus the larger `generalized-agent` pool,
4. decompress/parse savestates,
5. recover the 256 KiB GBA EWRAM (`0x02000000-0x0203FFFF`),
6. verify extraction using known player values, especially progress at `0x02007100`, where state filename suffixes provide expected values,
7. diff snapshots at different progress positions,
8. identify racer objects / racer-pointer arrays / repeated `~0x368`-byte layouts,
9. correlate the object containing player telemetry with the race-manager pointer array,
10. inspect the object corresponding to the NPC whose score appears at `0x02007C16`.

Do not assume a `.state` is raw RAM. First identify Stable-Retro/libretro serialization format or load it in the matching emulator/core and dump memory programmatically.

---

## Recommended dynamic reverse-engineering path

Prefer mGBA (debugger) for live reverse engineering while retaining Stable-Retro for training.

### A. Reproduce racer objects

At race start:

- break around `0x08101618`
- log each constructor call's racer pointer, racer index, and manager pointer
- verify number of racers
- dump all `0x368` bytes for every racer object
- identify which object belongs to the human player

Use known telemetry values to map fields inside the object.

### B. Trace human input forward

Break on/trace:

- `0x080F27C8`
- `0x080F25B8`
- candidate input preparation around `0x081025D4`

Press one button at a time:

- A only
- B only
- LEFT only
- RIGHT only
- L only
- R only

Observe `racer + 0x302/+0x304/+0x306` and any adjacent fields.

The GBA keypad bit layout is standard:

```text
bit 0 A
bit 1 B
bit 2 Select
bit 3 Start
bit 4 Right
bit 5 Left
bit 6 Up
bit 7 Down
bit 8 R
bit 9 L
```

Remember hardware `KEYINPUT` is active-low; game-local masks may be inverted to active-high.

### C. Find stock AI writer

Once player control masks are understood:

- set write watchpoints on the NPC's `+0x302/+0x304/+0x306`
- let the CPU race normally
- capture every writer PC
- walk backward to find the AI routine or upstream control record

Ideal result:

```text
NPC AI decision routine
  -> per-racer command record
  -> racer +0x302/+0x304/+0x306
  -> common racer update
  -> physics
```

If there is a compact upstream AI command record, hook that instead of racer-local transition state.

### D. Deterministic proof-of-control before PPO

Do **not** integrate PPO first.

Try, in order:

1. Force NPC A/accelerate continuously.
2. Force NPC LEFT continuously.
3. Force NPC RIGHT continuously.
4. Mirror human button mask into one NPC.

If the CPU car visibly responds with native physics/collision behavior, the control boundary is confirmed.

Only then build PPO IPC.

---

## PPO integration caveat: observation problem

The existing policy is pixel-based and was trained from the **human player's camera/framebuffer**.

If the model occupies an NPC slot while the game camera follows the human player, feeding the human camera to the NPC policy is semantically wrong once they separate on track.

So there are two separate milestones:

1. **control injection** into an NPC,
2. **NPC-correct observation** for the policy.

Potential solutions, ranked conceptually:

### Best long-term: structured/RAM racer policy

Train a policy on racer-centric structured state such as:

- racer speed
- heading/yaw
- track segment / progress
- lateral track position
- vector to next waypoint / target direction
- wall/collision state
- boost availability
- lap/rank
- relative position of other racers

Then the same policy can control any racer object and potentially support true multi-agent racing.

### Alternative: camera redirection

Temporarily redirect the game's camera to the NPC for observation capture, then restore it for the human render. More invasive and may introduce timing/render issues.

### Alternative: synchronized second emulator

Maintain another synchronized emulator whose camera follows the model racer. Much more complex.

Do not let the observation problem block the **control-injection proof of concept**. First make an NPC respond to externally supplied commands.

---

## Stable-Retro training context that matters for control translation

Existing action semantics:

- A = accelerate
- B = brake
- LEFT/RIGHT = steering
- UP/DOWN = other directional/trick inputs depending context
- L/R = trick/boost-related controls
- boost is triggered with L+R when available in the existing environment/action design

Training used stochastic frame skip around 4 frames, so eventual inference/control injection around ~15 Hz may be a natural starting point, while the game itself continues at native frame rate.

The successful policy architecture often embedded/forced acceleration into the action space. Preserve that training assumption when translating policy actions to button masks.

---

## Suggested local workspace

```text
~/projects/hotwheels-re/
├── HotWheelsGym/             # checkout npc-reverse-engineering
├── HotWheelsRL/              # full-history archaeology
├── private/
│   └── rom.gba               # NEVER COMMIT
└── work/
    ├── states/
    ├── ewram/
    ├── scripts/
    ├── disasm/
    └── notes/
```

Suggested tools:

- `git`
- Python 3
- `capstone`
- ARM GNU binutils (`arm-none-eabi-objdump`, etc.)
- mGBA + debugger if practical
- gzip/zlib tooling
- Stable-Retro / matching core as needed to load old states

Useful first commands:

```bash
git -C HotWheelsGym checkout npc-reverse-engineering
git -C HotWheelsGym log --all --oneline --decorate --graph
git -C HotWheelsRL log --all --name-status --oneline
```

Search the full object/history database for deleted/renamed state-related material as useful:

```bash
git -C HotWheelsRL rev-list --objects --all > /tmp/hwrl-objects.txt
grep -Ei 'state|rom|dino|multi|ram|npc' /tmp/hwrl-objects.txt
```

---

## Immediate next tasks for Codex

Work these in this order:

1. **Revalidate the ROM SHA** against repo `rom.sha`.
2. **Reproduce the static addresses/functions above** in a proper ARM/Thumb disassembler.
3. **Identify the exact semantics of racer `+0x302/+0x304/+0x306`.**
4. **Find which racer object corresponds to player vs each NPC.**
5. **Write-watch an NPC control field to identify its writer.**
6. **Find the stock AI -> common control/physics boundary.**
7. **Build a deterministic injection test** (`always-left` and/or `mirror-player`).
8. Commit scripts/notes/patches only to `npc-reverse-engineering`.
9. Keep ROM and derived copyrighted ROM dumps out of Git.
10. Once deterministic control works, design the external command IPC and only then connect PPO.

---

## Definition of success for the current phase

A successful reverse-engineering phase ends with a repeatable experiment like:

> Load Dino Boneyard multiplayer, select NPC racer N, overwrite a documented control field/buffer each frame, and visibly make that NPC accelerate/steer according to an external command while all movement, collision, lap, and race behavior remains native.

That proof is more important than immediately wiring in the neural network.

---

## Confidence / caveats

High confidence:

- historical player RAM anchors,
- `npc_score = 0x02007C16` existed and tracked an actual NPC according to the original reverse engineer/user,
- savestate suffixes represent track progress values,
- pixel-observation mismatch must be addressed before a pixel PPO can correctly control a spatially separated NPC.

Strong leads requiring local reproduction:

- race-manager offsets `+0x448/+0x450`,
- racer size about `0x368`,
- constructor `0x08101618`,
- vtable around `0x0817C430`,
- recurring update around `0x0810304C`,
- control preparation around `0x081025D4`,
- racer control fields around `+0x302/+0x304/+0x306`,
- hardware-input routines around `0x080F27C8` and `0x080F25B8`.

Treat those as the starting map for Codex, not as final symbol names.
