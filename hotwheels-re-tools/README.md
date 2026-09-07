# HotWheels reverse-engineering tools

This project makes the current ROM and savestate findings repeatable. It does
not contain a ROM, decompressed ROM data, or other copyrighted game assets.

The tools target the ROM whose SHA-1 is recorded by HotWheelsGym:

```text
3a950fa7f2979ac8b1e2e2eb628267b3146f1da1
```

## Run without installing

From the repository root:

```bash
PYTHONPATH=hotwheels-re-tools/src \
  python3 -m hotwheels_re_tools verify-rom rom.gba

PYTHONPATH=hotwheels-re-tools/src \
  python3 -m hotwheels_re_tools inspect-state \
  HotWheelsGym/HotWheelsStuntTrackChallenge-GbAdvance/dino_boneyard_multi.state
```

To disassemble a known Thumb range, install `binutils-arm-none-eabi` and run:

```bash
PYTHONPATH=hotwheels-re-tools/src \
  python3 -m hotwheels_re_tools disassemble rom.gba 0x080f27c8 0x080f27dc
```

An editable install provides the shorter `hwre` command:

```bash
python3 -m pip install -e ./hotwheels-re-tools
hwre verify-rom rom.gba
```

## Experimental mirror-control probe

`patch-mirror-cpu` creates a new, ignored `.gba` file. It never overwrites the
source ROM. The patch changes the two CPU allocation paths to allocate the
larger player-racer object and call the player-racer constructor. The expected
result is that CPU slots use the player control path and mirror Player 1 input.

```bash
PYTHONPATH=hotwheels-re-tools/src \
  python3 -m hotwheels_re_tools patch-mirror-cpu \
  rom.gba rom.mirror_cpu.gba
```

This is an architecture probe, not the final PPO integration. It still needs
to be validated in an emulator. See [FINDINGS.md](FINDINGS.md) for the evidence,
limitations, and exact patch sites.

## External CPU-heading probe

The narrower `patch-external-cpu-heading` command disables the stock AI's write
to CPU racer `+0xE0` (desired heading). A debugger, emulator harness, or eventual
policy bridge can then write a 12-bit heading directly while the native turning,
physics, collisions, and race management continue to run.

```bash
PYTHONPATH=hotwheels-re-tools/src \
  python3 -m hotwheels_re_tools patch-external-cpu-heading \
  rom.gba rom.external_heading.gba
```

The source-only headless harness in `native/mgba_probe.c` reproduces the dynamic
test with libmGBA development headers installed:

```bash
cc -O2 -o /tmp/hwre-mgba-probe \
  hotwheels-re-tools/native/mgba_probe.c -lmgba

# Historical states are gzip files; the harness intentionally accepts raw state.
gzip -dc path/to/dino_boneyard_multi_104.state > /tmp/dino104.state

/tmp/hwre-mgba-probe \
  rom.external_heading.gba /tmp/dino104.state 120 /tmp/result.ppm \
  --resume-old-hle --cpu-heading 1:0x680 --cpu-target-speed 1:30000
```

Slot 0 is the player and slot 1 is the first CPU in this state. The old-HLE
recovery is narrowly guarded and is only for reverse-engineering old Stable-Retro
states under current mGBA. The heading patch affects all stock CPUs, so a long
running external controller must supply headings for every CPU. Use the
selected-slot patch below for Gym training. ROM outputs remain ignored and must
not be committed.

`--cpu-target-speed SLOT:VALUE` writes the CPU's fixed-point target speed at
`+0x2F0`. Unlike desired heading, stock AI does not overwrite that field during
the tested race segment, so the speed experiment works with the original ROM as
well as the heading-patched one.

## Selected native-NPC control ROM

`patch-npc-control` is the Gym-facing patch. It suppresses the stock desired-
heading write only for selected CPU vehicle indices, so every other CPU keeps
the original AI. It also exposes the stock AI's computed waypoint heading at
CPU racer `+0x2EE` for structured observations.

```bash
PYTHONPATH=hotwheels-re-tools/src \
  python3 -m hotwheels_re_tools patch-npc-control \
  rom.gba /tmp/hotwheels-npc.gba \
  --vehicle-index 1 --vehicle-index 2

PYTHONPATH=hotwheels-re-tools/src \
  python3 -m hotwheels_re_tools inspect-npc-control-rom \
  /tmp/hotwheels-npc.gba
```

The hook branches through an eight-byte Thumb trampoline in an alignment gap
at `0x080EC538`, then runs from the ROM's unused trailing zero padding at
`0x0879BEE0`. Generated ROMs retain the original 8 MiB size and embed a small
versioned marker used by the Gym wrappers. Keep every generated ROM ignored
and uncommitted.

## Tests

```bash
PYTHONPATH=hotwheels-re-tools/src \
  python3 -m unittest discover -s hotwheels-re-tools/tests -v
```
