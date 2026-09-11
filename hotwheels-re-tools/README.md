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

## Native player-button opponents

`patch-npc-buttons` is the symmetric self-play patch. In both race-manager
setup paths it changes the requested population from one player plus three CPUs
to four player-class racers plus zero CPUs. It also hooks player input
preparation so vehicle index 0 continues sampling the real GBA keypad while
indices 1–3 retain independent racer-local button transition masks supplied by
Python.

```bash
PYTHONPATH=hotwheels-re-tools/src \
  python3 -m hotwheels_re_tools patch-npc-buttons \
  rom.gba /tmp/hotwheels-native-buttons.gba
```

The seven RAM-policy actions therefore have identical button semantics and run
through identical native control/physics code for Player 1 and model opponents.
A savestate created before this patch still restores stock CPU objects and
cannot be used for native-button self-play; see
[RAM_PLAYER.md](../RAM_PLAYER.md) for the private four-player state generator.

## Tests

```bash
PYTHONPATH=hotwheels-re-tools/src \
  python3 -m unittest discover -s hotwheels-re-tools/tests -v
```
