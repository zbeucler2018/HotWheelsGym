"""Small adapter from HotWheelsGym runtime state to semantic debug snapshots."""

from __future__ import annotations

from collections import deque
import gzip
from math import cos, pi, sin
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..dino_boneyard_track import (
    DINO_BONEYARD_CENTERLINE,
    DINO_TRACK_LOOKAHEADS,
    dino_next_power_up_radar,
    dino_track_pose,
)
from ..enums import RaceMode, Tracks
from ..npc_control import (
    HEADING_PERIOD,
    PowerUpState,
    RaceMemory,
    RacerState,
    discover_power_up_addresses,
    read_power_up_states,
)
from ..ram_opponent_control import (
    DINO_BONEYARD_PROGRESS_COUNT,
    RAM_DEFAULT_ACTION,
    RAM_OBSERVATION_NAMES,
    RaceProgressTracker,
    build_dino_ram_observation,
    build_track_ram_observation,
    normalize_racer_action,
    ordered_other_slots,
    player_buttons_from_action,
    read_racer_states,
)
from ..track_reference import (
    TRACK_LATERAL_SCALE,
    TRACK_LOOKAHEADS,
    next_power_up_radar,
    track_pose,
    track_reference_profile,
)
from .models import VisualizerSnapshot

HISTORY_LENGTH = 500
LOCAL_COORDINATE_SCALE = float(TRACK_LATERAL_SCALE)
HISTORY_NAMES = (
    "self_speed",
    "self_progress_rate",
    "track_lateral_offset",
    "track_heading_error_cos",
    "track_curvature_medium",
    "track_curvature_long",
    "next_power_up_progress_distance",
    "self_rank",
    "nearby_racer_1_relative_progress",
    "nearby_racer_2_relative_progress",
    "nearby_racer_3_relative_progress",
)
TRACK_RELATIVE_NAMES = {
    "track_lateral_offset",
    "track_heading_error_sin",
    "track_heading_error_cos",
    "track_target_short_forward",
    "track_target_short_right",
    "track_target_medium_forward",
    "track_target_medium_right",
    "track_target_long_forward",
    "track_target_long_right",
    "track_curvature_medium",
    "track_curvature_long",
}
PICKUP_NAMES = {
    "next_power_up_progress_distance",
    "next_power_up_target_lateral",
    "next_power_up_lateral_error",
    "next_power_up_available",
    "next_power_up_is_jet_boost",
}


def _base_env(env: Any) -> Any:
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _heading_vectors(heading: int) -> tuple[tuple[float, float], tuple[float, float]]:
    angle = 2.0 * pi * (heading & (HEADING_PERIOD - 1)) / HEADING_PERIOD
    return (sin(angle), cos(angle)), (cos(angle), -sin(angle))


def _observation_groups(named: Mapping[str, float]) -> dict[str, tuple[dict[str, Any], ...]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        "Self / context": [],
        "Track-relative": [],
        "Nearby racer 1": [],
        "Nearby racer 2": [],
        "Nearby racer 3": [],
        "Pickup radar": [],
    }
    for name in RAM_OBSERVATION_NAMES:
        entry = {"name": name, "value": float(named[name])}
        if name in TRACK_RELATIVE_NAMES:
            grouped["Track-relative"].append(entry)
        elif name in PICKUP_NAMES:
            grouped["Pickup radar"].append(entry)
        elif name.startswith("nearby_racer_1_"):
            grouped["Nearby racer 1"].append(entry)
        elif name.startswith("nearby_racer_2_"):
            grouped["Nearby racer 2"].append(entry)
        elif name.startswith("nearby_racer_3_"):
            grouped["Nearby racer 3"].append(entry)
        else:
            grouped["Self / context"].append(entry)
    return {name: tuple(entries) for name, entries in grouped.items()}


def _track_components(track: Tracks, state: RacerState, power_ups: tuple[PowerUpState, ...]):
    if track is Tracks.Dino_Boneyard:
        pose = dino_track_pose(state)
        radar = dino_next_power_up_radar(pose, power_ups)
        return (
            DINO_BONEYARD_PROGRESS_COUNT,
            DINO_BONEYARD_CENTERLINE,
            DINO_TRACK_LOOKAHEADS,
            pose,
            radar,
            None,
        )
    profile = track_reference_profile(track)
    pose = track_pose(profile, state)
    radar = next_power_up_radar(profile, pose, power_ups)
    return (
        profile.progress_count,
        profile.centerline,
        TRACK_LOOKAHEADS,
        pose,
        radar,
        profile,
    )


def _build_semantic_snapshot(
    *,
    framebuffer: np.ndarray,
    track: Tracks,
    controlled_slot: int,
    states: Mapping[int, RacerState],
    previous_states: Mapping[int, RacerState],
    progress: RaceProgressTracker,
    power_ups: tuple[PowerUpState, ...],
    previous_action: tuple[int, int, int],
    total_laps: int,
    raw_frame: int,
    revision: int,
    state_name: str,
    history: Mapping[str, tuple[float, ...]],
    runtime_metadata: Mapping[str, Any] | None = None,
) -> VisualizerSnapshot:
    state = states[controlled_slot]
    progress_count, centerline, lookaheads, pose, radar, profile = _track_components(
        track, state, power_ups
    )
    if progress.progress_count != progress_count:
        raise ValueError("progress tracker does not match selected track")
    if track is Tracks.Dino_Boneyard:
        observation = build_dino_ram_observation(
            states,
            previous_states,
            progress,
            controlled_slot,
            previous_action,
            total_laps=total_laps,
            track_pose=pose,
            power_ups=power_ups,
        )
    else:
        observation = build_track_ram_observation(
            states,
            previous_states,
            progress,
            controlled_slot,
            previous_action,
            profile=profile,
            total_laps=total_laps,
            track_pose=pose,
            power_ups=power_ups,
        )
    if len(observation) != 67 or len(RAM_OBSERVATION_NAMES) != 67:
        raise RuntimeError("visualizer requires the canonical 67-D observation")
    named = dict(zip(RAM_OBSERVATION_NAMES, observation))
    nearby = ordered_other_slots(controlled_slot, states, progress)
    (forward_x, forward_z), (right_x, right_z) = _heading_vectors(
        state.current_heading
    )

    def local_point(x: float, z: float) -> dict[str, float]:
        dx, dz = x - state.x, z - state.z
        return {
            "forward": (dx * forward_x + dz * forward_z) / LOCAL_COORDINATE_SCALE,
            "right": (dx * right_x + dz * right_z) / LOCAL_COORDINATE_SCALE,
        }

    def local_vector(x: float, z: float) -> dict[str, float]:
        return {
            "forward": x * forward_x + z * forward_z,
            "right": x * right_x + z * right_z,
        }

    line_points = []
    for offset in range(-8, 33):
        index = (pose.progress_index + offset) % progress_count
        x, z = centerline[index]
        line_points.append({"index": index, **local_point(x, z)})

    target_names = ("short", "medium", "long")
    targets = []
    for label, lookahead in zip(target_names, lookaheads):
        index = (pose.progress_index + lookahead) % progress_count
        x, z = centerline[index]
        targets.append(
            {
                "label": label,
                "index": index,
                "point": local_point(x, z),
                "observation": {
                    "forward": named[f"track_target_{label}_forward"],
                    "right": named[f"track_target_{label}_right"],
                },
            }
        )

    racer_views = []
    nearby_position = {slot: position for position, slot in enumerate(nearby, 1)}
    for slot, racer in sorted(states.items()):
        racer_forward, _ = _heading_vectors(racer.current_heading)
        local_heading = local_vector(*racer_forward)
        view: dict[str, Any] = {
            "slot": slot,
            "controlled": slot == controlled_slot,
            "label": (
                "controlled_racer"
                if slot == controlled_slot
                else f"nearby_racer_{nearby_position[slot]}"
            ),
            "world": {"x": racer.x, "z": racer.z, "heading": racer.current_heading},
            "local": local_point(racer.x, racer.z),
            "heading": local_heading,
            "progress": racer.progress,
            "lap": progress.current_lap(slot, racer),
            "rank": progress.rank(slot, states),
        }
        if slot != controlled_slot:
            prefix = f"nearby_racer_{nearby_position[slot]}_"
            view["observation"] = {
                name.removeprefix(prefix): named[name]
                for name in RAM_OBSERVATION_NAMES
                if name.startswith(prefix)
            }
        racer_views.append(view)

    geometry = {
        "coordinate_units": "world delta / 1048576",
        "progress_index": pose.progress_index,
        "segment_fraction": pose.segment_fraction,
        "centerline": line_points,
        "projected_point": local_point(pose.center_x, pose.center_z),
        "track_tangent": local_vector(pose.tangent_x, pose.tangent_z),
        "controlled_forward": {"forward": 1.0, "right": 0.0},
        "controlled_right": {"forward": 0.0, "right": 1.0},
        "lateral_offset": named["track_lateral_offset"],
        "heading_error": {
            "sin": named["track_heading_error_sin"],
            "cos": named["track_heading_error_cos"],
        },
        "lookahead_targets": targets,
        "curvature": {
            "medium": named["track_curvature_medium"],
            "long": named["track_curvature_long"],
        },
        "pickup": {
            "point": local_point(radar.placement.x, radar.placement.z),
            "kind": radar.placement.kind,
            "available": radar.available,
            "is_jet_boost": radar.is_jet_boost,
            "progress": radar.placement.progress,
            "progress_distance": named["next_power_up_progress_distance"],
            "target_lateral": named["next_power_up_target_lateral"],
            "lateral_error": named["next_power_up_lateral_error"],
        },
    }
    metadata = {
        "progress_count": progress_count,
        "total_laps": total_laps,
        "nearby_order_semantics": "absolute race-progress distance, then slot",
        "frame_shape": list(framebuffer.shape),
        **(runtime_metadata or {}),
    }
    return VisualizerSnapshot(
        revision=revision,
        framebuffer=np.asarray(framebuffer, dtype=np.uint8).copy(),
        track=track.value,
        controlled_slot=controlled_slot,
        raw_frame=raw_frame,
        state_name=state_name,
        observation_names=tuple(RAM_OBSERVATION_NAMES),
        observation_values=tuple(float(value) for value in observation),
        observation_groups=_observation_groups(named),
        geometry=geometry,
        racers=tuple(racer_views),
        nearby_order=nearby,
        history=history,
        metadata=metadata,
    )


class ObservationVisualizerAdapter:
    """Own a multi-race env and expose coherent reset/step/debug snapshots."""

    def __init__(
        self,
        env: Any,
        *,
        controlled_slot: int = 0,
        state_path: str | Path | None = None,
        history_length: int = HISTORY_LENGTH,
        policy: Any | None = None,
        policy_name: str | None = None,
        policy_action_repeat: int = 4,
    ) -> None:
        self.env = env
        self.base = _base_env(env)
        self.track = Tracks(self.base.track)
        if RaceMode(self.base.mode) is not RaceMode.MULTI:
            raise ValueError("RAM observation visualizer requires multi mode")
        if controlled_slot not in range(4):
            raise ValueError("controlled_slot must be 0, 1, 2, or 3")
        self.controlled_slot = controlled_slot
        self.state_path = Path(state_path).expanduser().resolve() if state_path else None
        if self.state_path and not self.state_path.is_file():
            raise FileNotFoundError(self.state_path)
        if policy_action_repeat < 1:
            raise ValueError("policy_action_repeat must be at least one")
        self.policy = policy
        self.policy_name = policy_name or ("loaded policy" if policy is not None else None)
        self.policy_action_repeat = policy_action_repeat
        self._history = {
            name: deque(maxlen=history_length) for name in HISTORY_NAMES
        }
        self._race: RaceMemory | None = None
        self._power_up_addresses: tuple[int, ...] = ()
        self._power_ups: tuple[PowerUpState, ...] = ()
        self._states: dict[int, RacerState] | None = None
        self._previous_states: dict[int, RacerState] | None = None
        self._progress: RaceProgressTracker | None = None
        self._framebuffer: np.ndarray | None = None
        self._snapshot: VisualizerSnapshot | None = None
        self._raw_frame = 0
        self._revision = 0
        self._previous_action = RAM_DEFAULT_ACTION
        self._current_action = RAM_DEFAULT_ACTION
        self._policy_frames_remaining = 0

    @property
    def progress_count(self) -> int:
        if self.track is Tracks.Dino_Boneyard:
            return DINO_BONEYARD_PROGRESS_COUNT
        return track_reference_profile(self.track).progress_count

    def _install_state(self) -> None:
        if self.state_path is None:
            return
        with gzip.open(self.state_path, "rb") as handle:
            self.base.initial_state = handle.read()
        self.base.statename = str(self.state_path)

    def reset(self) -> VisualizerSnapshot:
        self._install_state()
        frame, _ = self.env.reset()
        self._race = RaceMemory(self.base.data.memory)
        if len(self._race.layout.racers) != 4:
            raise ValueError("visualizer requires Player 1 plus exactly three NPCs")
        self._power_up_addresses = discover_power_up_addresses(self.base.data.memory)
        self._power_ups = read_power_up_states(
            self.base.data.memory, self._power_up_addresses
        )
        self._states = read_racer_states(self._race)
        self._previous_states = dict(self._states)
        self._progress = RaceProgressTracker.from_states(
            self._states, self.progress_count
        )
        self._framebuffer = np.asarray(frame)
        self._raw_frame = 0
        self._previous_action = RAM_DEFAULT_ACTION
        self._current_action = RAM_DEFAULT_ACTION
        self._policy_frames_remaining = 0
        reset_policy = getattr(self.policy, "reset", None)
        if callable(reset_policy):
            reset_policy()
        for values in self._history.values():
            values.clear()
        return self._refresh()

    def set_controlled_slot(self, slot: int) -> VisualizerSnapshot:
        if slot not in range(4):
            raise ValueError("controlled_slot must be 0, 1, 2, or 3")
        self.controlled_slot = slot
        for values in self._history.values():
            values.clear()
        return self._refresh()

    def step(self, frames: int = 1) -> VisualizerSnapshot:
        if self._race is None or self._states is None or self._progress is None:
            raise RuntimeError("call reset() before step()")
        if not 1 <= frames <= 600:
            raise ValueError("frames must be in 1..600")
        for _ in range(frames):
            if self.policy is not None and self._policy_frames_remaining == 0:
                self._current_action = self._predict_policy_action()
                self._policy_frames_remaining = self.policy_action_repeat
            if self.policy is None:
                buttons = tuple(False for _ in self.base.buttons)
            else:
                buttons = player_buttons_from_action(
                    self._current_action, self.base.buttons
                )
            previous = self._states
            frame, _, terminated, truncated, _ = self.env.step(buttons)
            current = read_racer_states(self._race)
            self._progress.update(current)
            self._previous_states = previous
            self._states = current
            self._power_ups = read_power_up_states(
                self.base.data.memory, self._power_up_addresses
            )
            self._framebuffer = np.asarray(frame)
            self._raw_frame += 1
            if self.policy is not None:
                self._previous_action = self._current_action
                self._policy_frames_remaining -= 1
            self._refresh()
            if terminated or truncated:
                break
        assert self._snapshot is not None
        return self._snapshot

    def _predict_policy_action(self) -> tuple[int, int, int]:
        """Predict one Player 1 action from the exact canonical observation."""

        if (
            self.policy is None
            or self._states is None
            or self._previous_states is None
            or self._progress is None
            or self._framebuffer is None
        ):
            raise RuntimeError("call reset() before requesting a policy action")
        policy_snapshot = _build_semantic_snapshot(
            framebuffer=self._framebuffer,
            track=self.track,
            controlled_slot=0,
            states=self._states,
            previous_states=self._previous_states,
            progress=self._progress,
            power_ups=self._power_ups,
            previous_action=self._previous_action,
            total_laps=int(self.base.total_laps),
            raw_frame=self._raw_frame,
            revision=self._revision,
            state_name=str(self.base.statename),
            history={},
        )
        action, _ = self.policy.predict(
            np.asarray(policy_snapshot.observation_values, dtype=np.float32),
            deterministic=True,
        )
        return normalize_racer_action(action)

    def _refresh(self) -> VisualizerSnapshot:
        if (
            self._states is None
            or self._previous_states is None
            or self._progress is None
            or self._framebuffer is None
        ):
            raise RuntimeError("call reset() before requesting a snapshot")
        self._revision += 1
        snapshot = _build_semantic_snapshot(
            framebuffer=self._framebuffer,
            track=self.track,
            controlled_slot=self.controlled_slot,
            states=self._states,
            previous_states=self._previous_states,
            progress=self._progress,
            power_ups=self._power_ups,
            previous_action=self._previous_action,
            total_laps=int(self.base.total_laps),
            raw_frame=self._raw_frame,
            revision=self._revision,
            state_name=str(self.base.statename),
            history={name: tuple(values) for name, values in self._history.items()},
            runtime_metadata=self._runtime_metadata(),
        )
        named = snapshot.named_observation
        for name, values in self._history.items():
            values.append(named[name])
        self._snapshot = _build_semantic_snapshot(
            framebuffer=self._framebuffer,
            track=self.track,
            controlled_slot=self.controlled_slot,
            states=self._states,
            previous_states=self._previous_states,
            progress=self._progress,
            power_ups=self._power_ups,
            previous_action=self._previous_action,
            total_laps=int(self.base.total_laps),
            raw_frame=self._raw_frame,
            revision=self._revision,
            state_name=str(self.base.statename),
            history={name: tuple(values) for name, values in self._history.items()},
            runtime_metadata=self._runtime_metadata(),
        )
        return self._snapshot

    def _runtime_metadata(self) -> dict[str, Any]:
        return {
            "policy_enabled": self.policy is not None,
            "policy_name": self.policy_name,
            "policy_slot": 0 if self.policy is not None else None,
            "policy_action_repeat": (
                self.policy_action_repeat if self.policy is not None else None
            ),
            "policy_action": list(self._current_action),
        }

    def snapshot(self) -> VisualizerSnapshot:
        if self._snapshot is None:
            return self.reset()
        return self._snapshot

    def close(self) -> None:
        self.env.close()


def build_visualizer_snapshot(
    *, env: Any, controlled_slot: int = 0, state_path: str | Path | None = None
) -> VisualizerSnapshot:
    """Reset ``env`` and return one canonical visualizer snapshot."""

    adapter = ObservationVisualizerAdapter(
        env, controlled_slot=controlled_slot, state_path=state_path
    )
    return adapter.reset()
