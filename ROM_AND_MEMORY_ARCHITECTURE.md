# ROM patching and runtime memory architecture

This document describes how HotWheelsGym changes opponent control without
changing the supplied game file, and how it reads racer state from the running
emulator. The implementation is split between the repeatable binary utilities
in [`hotwheels-re-tools`](hotwheels-re-tools/README.md), the runtime discovery
code in [`HotWheelsGym/npc_control.py`](HotWheelsGym/npc_control.py), and the
Gymnasium wrappers in [`HotWheelsGym/RAMOpponent.py`](HotWheelsGym/RAMOpponent.py).

## Two ROMs with different roles

The **source ROM** is the user's private `rom.gba`. It is an 8 MiB image with
SHA-1:

```text
3a950fa7f2979ac8b1e2e2eb628267b3146f1da1
```

The project validates that size and hash before patching. It never edits this
file in place. `create_npc_button_control_rom()` explicitly refuses to use the
same path for its input and output.

The **active patched ROM** is a generated, private copy used only when model
opponents are requested. `prepare_rom()` writes it below the ignored run
directory, normally as `private/dino-native-buttons.gba`, then asks
`HotWheelsGym.import_rom(..., force=True)` to copy that active image into the
local Stable-Retro custom integration. A run with stock opponents imports the
source ROM instead. Both supported hashes are listed in the integration's
`rom.sha`, so `import_rom()` rejects an unrelated or accidentally modified
image.

The known native-button output has SHA-1:

```text
e88db49ffc5ce3464d1ea5d941971e1ffe54c01a
```

This generated ROM is a private derivative artifact. It is not source code and
must not be committed or distributed.

## What `patch-npc-buttons` changes

GBA ROM addresses start at `0x08000000`; a patch's file offset is its bus
address minus that base. The patcher checks the original bytes at every site
before replacing them and never changes the ROM length. A mismatch aborts the
operation instead of applying a patch to an unknown build.

The version-3 native-button patch has three parts:

| ROM address | Change |
| ---: | --- |
| `0x080FB93A` | First construction path: set CPU-racer count to zero |
| `0x080FB946` | First construction path: set player/link-racer count to four |
| `0x080FBED2` | Second construction path: set CPU-racer count to zero |
| `0x080FBEDE` | Second construction path: set player/link-racer count to four |
| `0x08103EAE` | Redirect player-class input preparation to the trampoline |
| `0x080EC538` | Trampoline to the private hook |
| `0x0879BEE0` | Hook that separates Player 1 hardware input from external opponent input |

The count changes matter because the original game constructs one `0x368`-byte
player/link racer and three `0x300`-byte CPU racers. CPU racers use another
vtable and do not expose the player button fields. The patch instead constructs
four real player-class racers, so all four cars use the same native vehicle
update, physics, collision, trick, boost, respawn, and lap code.

The original player input-preparation routine is at `0x081025D4`. At the hook,
vehicle index 0 tail-calls that routine and therefore continues to sample the
GBA keypad normally. Vehicle indices 1, 2, and 3 return without running input
preparation. This preserves the button-transition masks that Python wrote into
each opponent object immediately before advancing the emulator.

The hook embeds an eight-byte identity record at GBA address `0x0879BEF4`
(ROM file offset `0x79BEF4`): the ASCII marker `HWBT`, patch version `3`, and
vehicle mask `0x0E` for indices 1–3. Runtime code checks this marker before it
allows opponent writes. The patch command reports the complete output SHA-1;
the later inspection and Stable-Retro import paths require the expected hash,
so the marker is an identification aid rather than the only integrity check.

The patch does **not** inject movement, teleport cars, overwrite coordinates,
or replace physics. It changes racer construction and ownership of native
button fields only.

## Stable-Retro integration and frame flow

`HotWheelsGym.import_rom()` copies the chosen active image to the custom game
integration at
`HotWheelsGym/HotWheelsStuntTrackChallenge-GbAdvance/rom.gba`. `HotWheelsEnv`
registers `HotWheelsGym/` as a Stable-Retro custom integration and asks RetroEnv
to load that game's ROM, JSON data description, and selected savestate.

For Player 1, a shared discrete RAM-policy action is translated to the normal
Stable-Retro Boolean controller vector. Stable-Retro presents those buttons to
the emulated keypad, and the unmodified game input path fills Player 1's racer
button fields.

For each model opponent and each raw emulator frame, the opponent wrapper:

1. builds the same ego-centric RAM observation used by Player 1;
2. requests a new policy action when its action-repeat counter expires;
3. converts that action to an active-high 10-bit GBA button mask;
4. derives newly pressed, newly released, and held masks from the prior frame;
5. writes those masks into the opponent's player-class racer object; and
6. advances the emulator, allowing the game's common update and physics to
   consume them.

The supported actions are coast, A, A+Left, A+Right, B, A+Up, and A+L+R.
The game's boost chord is L+R; the policy action also holds A so activating
boost does not release the accelerator.

## ROM addresses versus runtime addresses

Addresses beginning with `0x08` above refer to immutable code in the GBA ROM.
Live race objects are allocations in the GBA's 256 KiB external work RAM
(EWRAM), `0x02000000` through `0x0203FFFF`. Their absolute addresses can move
between boots and savestates, so the runtime does not depend on a single old
absolute address such as `0x02006FB8`.

`RaceMemory` receives Stable-Retro's live memory object. It uses
`memory.blocks` to scan EWRAM for aligned little-endian references to the known
racer vtables:

| Racer kind | Vtable | Allocation size |
| --- | ---: | ---: |
| Player/link racer | `0x0817C430` | `0x368` bytes |
| Stock CPU racer | `0x0817B6D4` | `0x300` bytes |

For every candidate racer, object offset `+0x50` must point to a plausible race
manager in EWRAM. The manager is then validated using its player count at
`+0x448`, CPU count at `+0x449`, total count at `+0x44A`, and ordered racer
pointer-list address at `+0x450`. Every pointer-list entry must have a known
vtable and point back to the same manager. The player/CPU object counts must
agree with the manager. Discovery fails closed if it finds zero or multiple
plausible managers.

The pointer-list order defines slots: slot 0 is Player 1 and slots 1–3 are the
opponents. On the patched ROM plus a fresh all-player savestate, every entry is
a player-class object while retaining its distinct vehicle index.

## Racer-local state and controls

After discovery, all reads and writes use `racer_address + offset`.
Stable-Retro's `memory.extract()` and `memory.assign()` perform the live
access. `|u1` means an unsigned byte, while `<u2` and `<u4` mean little-endian
unsigned 16- and 32-bit values. X and Z are read as 32-bit words and converted
to signed integers.

Current player-class offsets are:

| Offset | Width (bits) | Meaning and use |
| ---: | ---: | --- |
| `+0xDC` | 8 | Vehicle index; used by the input hook and patch validation |
| `+0xDE` | 16 | Current heading, masked to the game's 12-bit heading circle |
| `+0xE8` | 32 | Fixed-point speed |
| `+0xF0` | 32 | Boost charge, normalized against the observed maximum 980 |
| `+0xF8` | 32 signed | World X position |
| `+0x100` | 32 signed | World Z position |
| `+0x148` | 16 | Track progress/checkpoint index |
| `+0x14D` | 8 | Current power-up type; `3` is Dino Boneyard Jet Boost |
| `+0x14E` | 8 | Power-up countdown; Jet Boost begins near 150 |
| `+0x26B` | 8 | Player-class respawn-pending state |
| `+0x27D` | 8 | Native skid/loss-of-grip state |
| `+0x302` | 16 | Newly pressed button bits |
| `+0x304` | 16 | Newly released button bits |
| `+0x306` | 16 | Held/down button bits |

The standard active-high GBA bits are A `0x001`, B `0x002`, Select `0x004`,
Start `0x008`, Right `0x010`, Left `0x020`, Up `0x040`, Down `0x080`, R
`0x100`, and L `0x200`. Only the three button fields are written during model
opponent control. Telemetry, position, progress, and physics state are read,
not imposed.

Jet Boost and skid were verified dynamically, not assigned names from a single
correlation. Moving the Jet Boost pickup into an otherwise fixed trajectory
changed type/timer, and clearing either field from the acquired state removed
the hairpin handling benefit. Scripted steering plus forced-value branches
showed that `+0x27D` participates causally in native traction and heading logic.

## Why Player 1 and model opponents are symmetric

Control symmetry comes from constructing every slot as the same native
player/link racer class and feeding the same pressed/released/held fields. The
only difference is the input source: emulated keypad for slot 0, Python masks
for slots 1–3.

Observation symmetry is separate and is implemented in
`build_dino_ram_observation()`. The function accepts a `controlled_slot` and
rotates the same 60-value version-5 contract around that racer. Its own
heading, position, speed, boost, Jet Boost countdown, skid, lap/rank,
short-term deltas, and Dino track-relative geometry always occupy the self
fields. The other three racers are sorted by progress distance and represented
relative to that ego racer. The same function is called by `DinoRAMPlayerEnv`
for slot 0 and `DinoRAMModelOpponentEnv` for each controlled opponent.

This symmetry means a version-5 Player 1 RAM checkpoint can occupy an opponent
slot without redefining its inputs or outputs. It does not make old pixel
models NPC-correct, and older RAM observations with 43, 54, 58, or 59 inputs
remain shape-incompatible.

## Savestates and offline memory inspection

The historical `.state` files are gzip-compressed emulator snapshots, not raw
RAM. For the known historical format, `hotwheels-re-tools inspect-state`
requires a decompressed payload of `0x61000` bytes and extracts the `0x40000`
EWRAM bytes beginning at payload offset `0x21000`. The parser then uses the
same vtable, manager, pointer-list, and object-offset relationships to inspect
racer state offline. Known progress values in state filenames were used to
validate this EWRAM mapping.

A state restores its serialized objects after the ROM's construction code has
run. Consequently, an old state containing three CPU objects remains a
one-player/three-CPU state even when loaded with the patched ROM. Native-button
self-play requires a fresh, private start-line state captured after the patched
ROM constructed four player-class racers. That state belongs below the ignored
run directory and must not be committed.

The checked-in historical states use an older mGBA/HLE serialization format.
They are useful evidence and may load through the matching Stable-Retro core,
but the offline parser is not a general savestate converter. Newer standalone
mGBA versions can reject or mishandle the pending interrupt context in these
snapshots.

## Safety and reproducibility rules

- Never stage or commit the source ROM, an imported integration ROM, a patched
  ROM, a newly generated savestate, a memory dump, a model checkpoint, or a run
  directory. Existing historical integration assets are repository history;
  they are not permission to add new private artifacts.
- Keep source and generated `.gba` files private. `*.gba` and
  `training_scripts/ram_runs/` are ignored, but ignore rules are only a guard;
  inspect `git status` and stage explicit source/document paths.
- Validate the source as exactly 8 MiB with SHA-1 `3a950fa7...` before analysis
  or patching. Accept the generated patch only with SHA-1 `e88db49...`, marker
  `HWBT`, version 3, and vehicle mask `0x0E`.
- Use checked patches: verify every expected source byte, preserve ROM length,
  and abort on a mismatch. Do not weaken the hash or expected-byte checks to
  support another dump.
- Record source/active ROM hashes, Git revision, observation version, action
  schema, and opponent-model hashes in run metadata. This is already done by
  `write_run_metadata()`.
- After any diagnostic that temporarily imports a patched image, re-import and
  revalidate the source image before returning to stock-ROM work.
- Treat runtime addresses as untrusted until vtable, manager, pointer-list,
  counts, back-pointers, and slot identity all validate. Never write opponent
  controls into a stock CPU object or an undiscovered address.

Useful read-only checks are:

```bash
PYTHONPATH=hotwheels-re-tools/src \
  python3 -m hotwheels_re_tools verify-rom rom.gba

PYTHONPATH=hotwheels-re-tools/src \
  python3 -m hotwheels_re_tools patch-npc-buttons \
  rom.gba /tmp/unused.gba --dry-run

PYTHONPATH=hotwheels-re-tools/src \
  python3 -m hotwheels_re_tools inspect-npc-control-rom \
  training_scripts/ram_runs/RUN/private/dino-native-buttons.gba
```

The dry run validates the source and prints the checked patch plan without
writing an output ROM.

## Current limitations

- All ROM addresses, expected bytes, hashes, and vtables are specific to the
  one validated North American GBA image. Another region or revision is not
  supported.
- The native-button construction hook has been dynamically validated on Dino
  Boneyard. Other tracks still need cold-start soak tests.
- The 60-value geometry/observation contract is Dino Boneyard-specific. Other
  tracks need their own reference geometry and power-up portability audit.
- Runtime discovery expects one active race manager, no more than eight racers,
  known racer vtables, and internally consistent EWRAM pointers. It is intended
  for an active race, not menus or arbitrary game states.
- Player-class opponent control requires all converted slots to be driven. The
  training runner therefore requires slots 1, 2, and 3 together so an
  uncontrolled car cannot block the course.
- A pre-patch savestate cannot be upgraded in place. A private four-player state
  must be captured with the generated ROM.
- Racer fields are reverse-engineered interfaces, not official symbols. Any
  future field must be validated dynamically and, where practical, causally
  before it enters an observation or write path.
