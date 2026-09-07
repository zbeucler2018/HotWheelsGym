# Reverse-engineering findings

Status: static and savestate analysis reproduced on 2026-09-07. The deterministic
control patch has not yet been run in an emulator.

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

- The experimental patch has not yet been validated in mGBA or Stable-Retro.
- A final hook should control one selected CPU slot rather than replacing all
  CPU objects with player-racer objects.
- The stock CPU vtable update (`0x080F6C2C`) computes steering/physics internally;
  its clean command boundary is still unidentified.
- A pixel policy still sees the human camera, so model-correct observation is a
  separate milestone after deterministic control.

