# Reverse-engineering findings

Status: static, savestate, and headless mGBA analysis reproduced beginning on
2026-09-07. The desired-heading boundary, selected-CPU hook, and symmetric
native-button opponent path have been dynamically validated. The older
mirror-control experiment remains separate from the production-facing patch.

## ROM identity

- Size: `0x800000` bytes (8 MiB)
- SHA-1: `3a950fa7f2979ac8b1e2e2eb628267b3146f1da1`
- The SHA-1 exactly matches the repository's `rom.sha`.

## Hardware input path

The Thumb routine at `0x080F27C8` loads the active-low GBA keypad register at
`0x04000130`, inverts it, and calls `0x080F25B8` with the input-state object at
`0x03001B5C`.

The input-state transition fields are:

| Offset | Width | Meaning |
|---:|---:|---|
| `+0x0` | 16 bits | raw/current normalized mask |
| `+0x2` | 16 bits | held/down mask |
| `+0x4` | 16 bits | newly pressed mask |
| `+0x6` | 16 bits | newly released mask |

## Racer classes and race manager

The initialization loops prove that the game uses two distinct concrete racer
classes, not one uniform `0x368`-byte object.

| Kind | Allocation | Constructor | Vtable |
|---|---:|---:|---:|
| player/link racer | `0x368` | `0x08101618` | `0x0817C430` |
| stock CPU racer | `0x300` | `0x080F5888` | `0x0817B6D4` |

Relevant race-manager fields during construction:

| Offset | Meaning |
|---:|---|
| `+0x448` | player/link-racer count |
| `+0x449` | CPU-racer count |
| `+0x44A` | total racer count |
| `+0x450` | pointer to the racer-pointer array |

Both object-allocation paths occur twice in the ROM. The player constructor is
called at `0x080FBA08` and `0x080FBFA0`; the CPU constructor is called at
`0x080FBAB8` and `0x080FC050`.

In the bundled Dino Boneyard multiplayer savestate, the layout is:

```text
race manager       0x020068F8
racer pointer list 0x02006F18
  [0] player       0x02006FB8
  [1] CPU          0x020074E8
  [2] CPU          0x02007808
  [3] CPU          0x02007B28
```

Every racer has a manager pointer at object `+0x50`. The pointer-list address
is stored at manager `+0x450`, which distinguishes the pointer field from the
separately allocated pointer list.

## Player-racer controls

`0x08103E8C`, the player-racer vtable slot-four update, calls control preparation
at `0x081025D4`. The resulting player-racer fields are:

| Offset | Width | Meaning |
|---:|---:|---|
| `+0x302` | 16 bits | newly pressed |
| `+0x304` | 16 bits | newly released |
| `+0x306` | 16 bits | held/down |
| `+0x308` | 16 bits | previous raw mask for the alternate indexed-input path |

In the normal local-input branch, `0x081025D4` copies the global input-state
fields `+0x4`, `+0x6`, and `+0x2` to racer `+0x302`, `+0x304`, and `+0x306`.
The alternate branch obtains a mask from a 14-byte indexed record and derives
the same transitions. This is a player/link-racer control path; stock CPU
objects are smaller and use a different vtable.

## Symmetric native-button opponent patch

`patch-npc-buttons` makes a private patched ROM copy; it never edits the source
ROM in place. At both race-manager construction paths it changes the counts to
four player/link racers and zero CPU racers. It then redirects the call at
`0x08103EAE` through a Thumb hook at `0x0879BEE0`:

- vehicle index 0 tail-calls the original `0x081025D4` input preparation, so
  Player 1 still reads the hardware keypad;
- vehicle indices 1–3 return immediately, preserving independent external
  writes to `+0x302/+0x304/+0x306`.

The emulator-side wrapper converts each of the same seven policy actions used
by Player 1 into native 10-bit GBA pressed/released/held masks and writes them
to those three racer-local fields every frame. All four vehicles then pass
through the same player-class update and native physics. Generated copies carry
the `HWBT` version-3 marker and expected SHA-1
`e88db49ffc5ce3464d1ea5d941971e1ffe54c01a`; ROM files remain ignored.

## Dino Boneyard track-relative RAM contract

Observation version 2 adds a 342-point reference line derived from median X/Z
telemetry for all three stock CPU racers over repeated laps. Live racers are
projected onto nearby segments. The shared 54-float Player/NPC observation now
includes signed lateral offset, heading error, three ego-relative lookahead
directions, and medium/long signed curvature.

A 12,000-frame native-racer replay produced a median normalized lateral error
of 0.0023 (95th percentile 0.112), mean heading alignment of 0.938, and a local
projection within two progress units for 95% of 35,974 racer-frame samples.
The reference table stores only coordinates—no ROM bytes or imagery.

## Stock CPU command boundary

The CPU slot-four update at `0x080F6C2C` has a compact steering boundary:

| Offset | Width | Meaning |
|---:|---:|---|
| `+0xDE` | 16 bits | current heading (12-bit circle) |
| `+0xE0` | 16 bits | desired heading (12-bit circle) |
| `+0xE8` | 32 bits | current fixed-point speed |
| `+0x2F0` | 32 bits | target fixed-point speed |

At `0x080F7112`, the stock AI calculates a heading to its current waypoint. The
store at `0x080F7116` writes that result to racer `+0xE0`. Later in the same
update, `0x08108E94` calculates the bounded turn from `+0xDE` toward `+0xE0`;
the updated current heading drives the velocity-vector calculation.

The `patch-external-cpu-heading` probe NOPs only the store at `0x080F7116`, so
an emulator-side controller can own `+0xE0`. From generated Dino Boneyard state
104, a 120-frame headless mGBA comparison produced:

| Run | CPU 1 progress | heading | x | z |
|---|---:|---:|---:|---:|
| stock AI | 107 -> 113 | `0x641` -> `0x6EF` | 5,456,220 -> 6,251,826 | 6,970,483 -> 5,928,424 |
| forced `+0xE0 = 0x680` | 107 -> 110 | `0x641` -> `0x673` | 5,456,220 -> 6,015,519 | 6,970,483 -> 6,199,021 |

This proves that `+0xE0` is an effective steering-command input rather than
merely telemetry. The one-instruction probe suppresses the desired-heading
write for every CPU, so it is not yet the final selected-slot hook.

The production-facing `patch-npc-control` hook checks the CPU racer's vehicle
index and suppresses the store only for a generated bitmask of indices 1..3.
For controlled racers it writes the stock computed heading to the otherwise
zero padding field at `+0x2EE`, making the native waypoint target available to
the Gym observation. In a 120-frame mGBA test with vehicle 1 selected and held
at heading `0x680`, slot 1 followed the external heading while slots 2 and 3
continued to update their stock desired headings and progress normally.

`+0x2F0` is also a live speed command and does not need the heading-store patch.
In the same 120-frame test, CPU 1's stock target of 59,904 yielded current speed
59,889 and progress 113. Holding the target at 30,000 yielded current speed
33,689 and progress 110; holding it at zero yielded current speed 3,630 and
progress 109. The native update smoothly approaches the supplied target while
retaining the rest of the CPU racer's physics.

## Player-class opponent respawns

The native-button patch makes each opponent a player-class racer, so a failed
trick uses the player-class respawn path. Frame-level Stable-Retro telemetry on
the Dino self-play recording reproduced three independent opponent teleports:
slot 2 at frame 3,266, slot 1 at frame 3,331, and slot 3 at frame 4,465. Player
1 did not teleport in any case. At the visible frame-4,465 event, Player 1 kept
progress 172 and speed 32,918 with unchanged coordinates while only slot 3
jumped 481,995 coordinate units and advanced from progress 162 to 164.

The player-class path does whiten the shared framebuffer during an opponent
respawn. Racer byte `+0x26B` identifies the per-racer pending transition in the
reproduced sequence. `DinoRAMModelOpponentEnv` masks only an opponent-only
white transition by returning and rendering its last visible frame; it leaves
the emulator state untouched and does not mask a real Player 1 respawn.

## Savestate layout and NPC score identity

The gzip payload is `0x61000` bytes. EWRAM is the `0x40000`-byte region at
payload offset `0x21000`. This was established by checking `0x02007100` against
the progress suffix in five generated Dino Boneyard states; all yielded the
same EWRAM base.

The common per-racer progress field is object `+0x148`. Generated savestates
show each CPU's independent progress near the player's filename-labelled
progress.

Historical `npc_score` address `0x02007C16` resolves exactly to racer-pointer
entry 3 (the third CPU object at `0x02007B28`) plus `0xEE`. Equivalent `+0xEE`
fields exist on all CPU racers and vary independently. It is therefore a
specific NPC slot's field, not a race-manager aggregation.

The historical state header is mGBA format `0x01000002` with HLE BIOS checksum
`0x590E6155`. Current mGBA 0.10.2 expects format `0x01000007` and HLE checksum
`0x5A262303`. These states were captured inside the old HLE BIOS IRQ trampoline
(`PC=0x0000001C`, IRQ-mode CPSR). Letting the new HLE BIOS resume that old
trampoline corrupts execution. For reverse-engineering experiments, the native
probe's `--resume-old-hle` option skips the one pending IRQ return by restoring
`CPSR=SPSR` and `PC=LR-4`; the race then advances and renders correctly. This is
a compatibility workaround, not a general savestate converter.

## Experimental deterministic patch

The `patch-mirror-cpu` command changes both CPU construction paths:

| Address | Original | Patched | Purpose |
|---:|---|---|---|
| `0x080FBAA2` | `movs r1, #0xC0` | `movs r1, #0xDA` | allocation: `0x300` to `0x368` |
| `0x080FBAB8` | `bl 0x080F5888` | `bl 0x08101618` | use player constructor |
| `0x080FC03A` | `movs r1, #0xC0` | `movs r1, #0xDA` | second allocation path |
| `0x080FC050` | `bl 0x080F5888` | `bl 0x08101618` | second constructor path |

Expected behavior: CPU slots become player-racer instances, remain in the same
manager pointer array, and consume Player 1's normal control state. This should
produce a visible mirror-control proof while retaining the native racer update,
physics, collision, and race manager. Dynamic testing is required; code that
assumes CPU-specific fields or vtable behavior may expose an incompatibility.

## Open issues

- The CPU-to-player-class mirror patch has not yet been validated from a cold
  race start in mGBA or Stable-Retro.
- The selected-CPU hook has been tested from a resumed historical race state,
  but still needs a cold-start Stable-Retro soak test on every track.
- The current NPC observation/reward is a first training contract and may need
  additional collision or lap telemetry after initial PPO experiments.
- The old-HLE IRQ recovery intentionally skips one interrupt handler invocation;
  use the original Stable-Retro 0.9.2-era core for production-faithful playback.
- A pixel policy still sees the human camera, so model-correct observation is a
  separate milestone after deterministic control.
