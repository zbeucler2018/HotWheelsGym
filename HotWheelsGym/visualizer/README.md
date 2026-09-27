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

Static/reset/step/playback and racer-slot selection are supported. Attaching to
an independently running environment is not currently implemented.
