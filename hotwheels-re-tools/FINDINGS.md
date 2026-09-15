# Reverse-engineering findings

Status: static, savestate, and headless mGBA analysis reproduced beginning on
2026-09-07. The symmetric native-button opponent path has been dynamically
validated and is the only supported model-opponent control interface.

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

The emulator-side wrapper converts the same factorized drive/steering/boost
action used by Player 1 into native 10-bit GBA pressed/released/held masks and
writes them to those three racer-local fields every frame. All four vehicles
then pass through the same player-class update and native physics. Generated
copies carry the `HWBT` version-3 marker and expected SHA-1
`e88db49ffc5ce3464d1ea5d941971e1ffe54c01a`; ROM files remain ignored.

## Dino Boneyard track-relative RAM contract

Observation version 2 adds a 342-point reference line derived from median X/Z
telemetry for all three stock CPU racers over repeated laps. Live racers are
projected onto nearby segments. Observation version 3 gives Player 1 and every
NPC the same 58 floats: the v2 centerline features plus each racer's normalized
boost charge. Observation version 4 adds a normalized handling/Jet Boost
countdown for 59 symmetric floats. Boost is racer-local at `+0xF0` on every
bundled multiplayer track. Jet Boost is type 3 at racer `+0x14D`; its countdown
at `+0x14E` starts near 150 and decrements while active.

Observation version 5 adds the native binary skid/loss-of-grip state at racer
`+0x27D`, producing 60 symmetric floats for Player 1 and player-class model
opponents.

Observation version 6 factorizes drive, steering, and boost, expands prior-
action context to nine one-hot values, and produces 62 symmetric floats.
Observation version 7 appends a five-value next-pickup radar for 67 floats. The
v6 values remain an exact prefix, permitting frozen-policy adaptation and
lossless first-layer expansion with zero-initialized pickup columns.

The Jet Boost mapping is causal, not just correlated: moving the pickup into an
otherwise fixed Player 1 trajectory acquired type 3 and its timer, while clearing
either field from the same acquired state removed the handling benefit through
the hairpin.

Racer byte `+0x27D` is a causal native skid/loss-of-grip state. Scripted trials
made it activate under sustained high-speed left or right steering at about 40
heading units per physics update, while straight acceleration, coast, and braking
left it clear. The player-physics routine clears it at `0x08103916`, sets it after
a lateral-slip threshold at `0x081039BC`, and reads it in traction and heading
logic at `0x0810285E` and `0x08102D2E`. With an identical 328-frame hairpin input
sequence, the native state reached progress 66 with no wall frames; forced zero
reached 62 with 14 wall frames, and forced one reached 56 with 26 wall frames.
This rules out a display-only flag and justifies exposing it to the policy.

## Dino Boneyard pickup objects

Track pickups use vtable `0x0817C598`. A controlled 12,000-frame replay changed
the target object's byte `+0x05` from state `0` to `2` on the exact Jet Boost
acquisition frame. Continued replay observed `2 -> 3 -> 0`, establishing
available, inactive, and respawn-transition states. The object layout also has
signed X/Y/Z at `+0x08/+0x0C/+0x10`, type at `+0x16`, and a respawn detail byte
at `+0x18`. A type-6 acquisition increased boost by 990 units; type 3 is the
causally verified Jet Boost/handling pickup.

The complete Dino Boneyard layout contains two type-3 Jet Boost objects and five
type-6 boost refills:

| Track progress | Type | X | Y | Z | Lateral offset |
|---:|---|---:|---:|---:|---:|
| 42.971 | Jet Boost | 10,038,491 | -417,570 | 7,568,627 | -0.1889 |
| 94.448 | Boost refill | 4,016,744 | -533,618 | 9,880,310 | -0.1583 |
| 148.983 | Boost refill | 13,140,339 | -410,564 | 1,754,933 | 0.0795 |
| 170.519 | Jet Boost | 17,067,310 | -306,085 | 6,564,462 | 0.0740 |
| 216.000 | Boost refill | 11,839,550 | -179,079 | 14,393,595 | -0.2045 |
| 252.411 | Boost refill | 15,433,710 | -409,600 | 20,757,852 | 0.0930 |
| 307.182 | Boost refill | 3,127,951 | -446,294 | 19,227,290 | -0.1457 |

Runtime code scans by vtable because allocation addresses move, then validates
the seven type/X/Z identities against this map. Availability is global and the
same live objects feed Player 1 and every model-controlled opponent observation.

A 12,000-frame native-racer replay produced a median normalized lateral error
of 0.0023 (95th percentile 0.112), mean heading alignment of 0.938, and a local
projection within two progress units for 95% of 35,974 racer-frame samples.
The reference table stores only coordinates—no ROM bytes or imagery.

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
probe skipped the one pending IRQ return by restoring `CPSR=SPSR` and
`PC=LR-4`; the race then advanced and rendered correctly. This was a diagnostic
compatibility workaround, not a general savestate converter or supported tool.

## Open issues

- The native-button patch still needs a cold-start Stable-Retro soak test on
  tracks beyond Dino Boneyard.
- Additional tracks need track-relative geometry and power-up observations.
