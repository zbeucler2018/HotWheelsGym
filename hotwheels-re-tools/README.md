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

## Tests

```bash
PYTHONPATH=hotwheels-re-tools/src \
  python3 -m unittest discover -s hotwheels-re-tools/tests -v
```

