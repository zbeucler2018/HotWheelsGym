# RAM observation visualizer

This optional package is a local debugger for the canonical 67-D multiplayer
RAM observation. It does not change rewards, actions, training, model loading,
or environment creation. The server uses only Python's standard library and
the browser uses vanilla HTML/CSS/JavaScript.

Launch the default Dino Boneyard multi state:

```bash
python -m HotWheelsGym.visualizer \
  --track dino_boneyard \
  --host 0.0.0.0 \
  --port 8765
```

Load another committed state while retaining the same multi/3-NPC mode:

```bash
python -m HotWheelsGym.visualizer \
  --track trex_valley \
  --state HotWheelsGym/HotWheelsStuntTrackChallenge-GbAdvance/trex_valley_multi.state
```

Let a compatible 67-D PPO checkpoint drive Player 1 while you inspect it:

```bash
python -m HotWheelsGym.visualizer \
  --track dino_boneyard \
  --model training_scripts/ram_runs/dino_ram_player_power_up_radar_v10_20260915T132334Z/evaluation/best_model.zip \
  --model-action-repeat 4 \
  --host 0.0.0.0 \
  --port 8765
```

The model always drives native racer slot 0. The controlled-slot selector only
changes which racer's semantically equivalent observation and geometry are
displayed. Model loading remains optional and Stable-Baselines3 is imported
only when `--model` is supplied. The session header displays both the raw
factorized policy action and its exact native button combination.

Open `http://<device-tailscale-ip>:8765/` from another device on the same
tailnet. The service has no authentication of its own, so keep it on a trusted
local or Tailscale interface and do not expose the port through a public tunnel.

The public Python interface is:

```python
from HotWheelsGym.visualizer import ObservationVisualizerAdapter

adapter = ObservationVisualizerAdapter(env, controlled_slot=0)
snapshot = adapter.reset()
snapshot = adapter.step(frames=4)
```

`snapshot` contains the framebuffer, canonical names and values, actual track
pose/radar intermediates, racer-local geometry, dynamic nearby-racer ordering,
and rolling history. The HTTP layer consumes this snapshot and does not import
or reconstruct observation math.

Static/reset/step/playback and racer-slot selection are supported. Playback is
clocked by the Python server at 60 raw frames/second for 1x speed. A same-origin
WebSocket streams framebuffer updates at up to the game's 30 FPS while sending
the heavier semantic observation/geometry payload at 2 Hz, so Tailnet latency
does not clock the emulator or force a full diagnostic redraw for every frame.
Geometry layers are toggled by tapping their colored legend entries, and the
Rolling history heading collapses that section. Those choices are retained
locally across reloads.
Attaching to
an independently running environment is not currently implemented.
