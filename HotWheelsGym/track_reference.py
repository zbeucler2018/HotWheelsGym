"""Track-relative RAM geometry for the non-Dino multi-race courses.

The 67-value observation contract keeps the Dino field names and ordering.  This
module supplies the same semantic geometry and pickup-radar values for the other
courses, using a per-track modulo-checkpoint reference line.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, pi, sin

from .enums import Tracks
from .npc_control import HEADING_PERIOD, PowerUpState, RacerState
from .track_reference_data import TRACK_REFERENCE_DATA

TRACK_LATERAL_SCALE = 1 << 20
TRACK_LOOKAHEADS = (4, 12, 24)
JET_BOOST_POWER_UP_TYPE = 3
POWER_UP_PROGRESS_DISTANCE_SCALE = 64.0


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _unit(dx: float, dz: float) -> tuple[float, float]:
    length = hypot(dx, dz)
    return (0.0, 1.0) if length <= 1e-9 else (dx / length, dz / length)


def _segment_projection(
    x: float, z: float, start: tuple[int, int], end: tuple[int, int]
) -> tuple[float, float, float, float, float, float]:
    dx = float(end[0] - start[0])
    dz = float(end[1] - start[1])
    length_squared = dx * dx + dz * dz
    if length_squared <= 1e-9:
        fraction = 0.0
        center_x, center_z = float(start[0]), float(start[1])
        tangent_x, tangent_z = 0.0, 1.0
    else:
        fraction = _clip(
            ((x - start[0]) * dx + (z - start[1]) * dz) / length_squared
        )
        fraction = max(0.0, fraction)
        center_x = start[0] + fraction * dx
        center_z = start[1] + fraction * dz
        tangent_x, tangent_z = _unit(dx, dz)
    distance_squared = (x - center_x) ** 2 + (z - center_z) ** 2
    return distance_squared, fraction, center_x, center_z, tangent_x, tangent_z


@dataclass(frozen=True)
class TrackPowerUpPlacement:
    kind: int
    x: int
    y: int
    z: int
    progress: float
    lateral_offset: float


@dataclass(frozen=True)
class TrackReferenceProfile:
    """Static geometry for one supported non-Dino course."""

    track: Tracks
    progress_count: int
    centerline: tuple[tuple[int, int], ...]
    power_up_objects: tuple[tuple[int, int, int, int], ...]
    interpolated_indices: tuple[int, ...]
    power_ups: tuple[TrackPowerUpPlacement, ...]


@dataclass(frozen=True)
class TrackPose:
    progress_index: int
    segment_fraction: float
    center_x: float
    center_z: float
    tangent_x: float
    tangent_z: float
    lateral_offset: float
    heading_error_sin: float
    heading_error_cos: float
    features: tuple[float, ...]


@dataclass(frozen=True)
class TrackPowerUpRadar:
    placement: TrackPowerUpPlacement
    progress_distance: float
    lateral_error: float
    available: bool

    @property
    def is_jet_boost(self) -> bool:
        return self.placement.kind == JET_BOOST_POWER_UP_TYPE

    @property
    def features(self) -> tuple[float, ...]:
        return (
            _clip(self.progress_distance / POWER_UP_PROGRESS_DISTANCE_SCALE),
            _clip(self.placement.lateral_offset),
            _clip(self.lateral_error),
            float(self.available),
            float(self.is_jet_boost),
        )


def _placement(
    profile: TrackReferenceProfile, kind: int, x: int, y: int, z: int
) -> TrackPowerUpPlacement:
    best: tuple[float, int, float, float, float, float, float] | None = None
    for index, start in enumerate(profile.centerline):
        end = profile.centerline[(index + 1) % profile.progress_count]
        projection = _segment_projection(x, z, start, end)
        candidate = (projection[0], index, *projection[1:])
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    _, index, fraction, center_x, center_z, tangent_x, tangent_z = best
    lateral = ((x - center_x) * tangent_z - (z - center_z) * tangent_x) / TRACK_LATERAL_SCALE
    return TrackPowerUpPlacement(
        kind=kind,
        x=x,
        y=y,
        z=z,
        progress=index + fraction,
        lateral_offset=_clip(lateral),
    )


def _make_profile(track_name: str, raw: dict[str, object]) -> TrackReferenceProfile:
    track = Tracks(track_name)
    centerline = tuple(tuple(point) for point in raw["centerline"])
    progress_count = int(raw["progress_count"])
    if len(centerline) != progress_count:
        raise RuntimeError(f"{track.value} centerline length does not match progress count")
    initial = TrackReferenceProfile(
        track=track,
        progress_count=progress_count,
        centerline=centerline,
        power_up_objects=tuple(tuple(values) for values in raw["power_up_objects"]),
        interpolated_indices=tuple(raw["interpolated_indices"]),
        power_ups=(),
    )
    return TrackReferenceProfile(
        track=initial.track,
        progress_count=initial.progress_count,
        centerline=initial.centerline,
        power_up_objects=initial.power_up_objects,
        interpolated_indices=initial.interpolated_indices,
        power_ups=tuple(
            sorted(
                (_placement(initial, *values) for values in initial.power_up_objects),
                key=lambda pickup: pickup.progress,
            )
        ),
    )


TRACK_REFERENCE_PROFILES = {
    track_name: _make_profile(track_name, raw)
    for track_name, raw in TRACK_REFERENCE_DATA.items()
}


def track_reference_profile(track: Tracks | str) -> TrackReferenceProfile:
    """Return a validated non-Dino profile by enum or integration track name."""

    key = track.value if isinstance(track, Tracks) else str(track)
    try:
        return TRACK_REFERENCE_PROFILES[key]
    except KeyError as error:
        raise ValueError(f"no non-Dino track reference profile for {key!r}") from error


def _tangent(profile: TrackReferenceProfile, index: int, radius: int = 2) -> tuple[float, float]:
    before = profile.centerline[(index - radius) % profile.progress_count]
    after = profile.centerline[(index + radius) % profile.progress_count]
    return _unit(after[0] - before[0], after[1] - before[1])


def _target_direction(
    profile: TrackReferenceProfile,
    x: float,
    z: float,
    forward_x: float,
    forward_z: float,
    right_x: float,
    right_z: float,
    index: int,
) -> tuple[float, float]:
    target = profile.centerline[index % profile.progress_count]
    direction_x, direction_z = _unit(target[0] - x, target[1] - z)
    return (
        _clip(direction_x * forward_x + direction_z * forward_z),
        _clip(direction_x * right_x + direction_z * right_z),
    )


def track_pose(profile: TrackReferenceProfile, state: RacerState, search_radius: int = 4) -> TrackPose:
    """Project a racer on ``profile`` into the existing 11 track features."""

    nominal = state.progress % profile.progress_count
    best: tuple[float, int, float, float, float, float, float] | None = None
    for offset in range(-search_radius, search_radius + 1):
        index = (nominal + offset) % profile.progress_count
        start = profile.centerline[index]
        end = profile.centerline[(index + 1) % profile.progress_count]
        projection = _segment_projection(state.x, state.z, start, end)
        candidate = (projection[0], index, *projection[1:])
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    _, index, fraction, center_x, center_z, tangent_x, tangent_z = best
    angle = 2.0 * pi * (state.current_heading & (HEADING_PERIOD - 1)) / HEADING_PERIOD
    forward_x, forward_z = sin(angle), cos(angle)
    right_x, right_z = cos(angle), -sin(angle)
    lateral = _clip(((state.x - center_x) * tangent_z - (state.z - center_z) * tangent_x) / TRACK_LATERAL_SCALE)
    heading_error_sin = _clip(tangent_x * right_x + tangent_z * right_z)
    heading_error_cos = _clip(tangent_x * forward_x + tangent_z * forward_z)
    targets: list[float] = []
    for lookahead in TRACK_LOOKAHEADS:
        targets.extend(_target_direction(profile, state.x, state.z, forward_x, forward_z, right_x, right_z, index + lookahead))
    medium_tangent = _tangent(profile, index + TRACK_LOOKAHEADS[1])
    long_tangent = _tangent(profile, index + TRACK_LOOKAHEADS[2])
    return TrackPose(
        progress_index=index,
        segment_fraction=fraction,
        center_x=center_x,
        center_z=center_z,
        tangent_x=tangent_x,
        tangent_z=tangent_z,
        lateral_offset=lateral,
        heading_error_sin=heading_error_sin,
        heading_error_cos=heading_error_cos,
        features=(
            lateral,
            heading_error_sin,
            heading_error_cos,
            *targets,
            _clip(tangent_x * medium_tangent[1] - tangent_z * medium_tangent[0]),
            _clip(tangent_x * long_tangent[1] - tangent_z * long_tangent[0]),
        ),
    )


def next_power_up_radar(
    profile: TrackReferenceProfile,
    pose: TrackPose,
    power_ups: tuple[PowerUpState, ...] | None = None,
) -> TrackPowerUpRadar:
    availability = {(pickup.kind, pickup.x, pickup.z): pickup.available for pickup in power_ups or ()}
    expected = {(pickup.kind, pickup.x, pickup.z) for pickup in profile.power_ups}
    if power_ups is not None and expected - set(availability):
        raise RuntimeError(f"{profile.track.value} power-up layout is incomplete")
    current_progress = pose.progress_index + pose.segment_fraction
    placement = min(profile.power_ups, key=lambda pickup: (pickup.progress - current_progress) % profile.progress_count)
    distance = (placement.progress - current_progress) % profile.progress_count
    return TrackPowerUpRadar(
        placement=placement,
        progress_distance=distance,
        lateral_error=placement.lateral_offset - pose.lateral_offset,
        available=availability.get((placement.kind, placement.x, placement.z), True),
    )


def continuous_progress_delta(
    profile: TrackReferenceProfile,
    previous: TrackPose,
    current: TrackPose,
    *,
    native_advance: int,
    maximum_per_frame: float = 2.0,
) -> float:
    delta = (current.progress_index + current.segment_fraction) - (previous.progress_index + previous.segment_fraction)
    half_lap = profile.progress_count / 2
    if delta > half_lap:
        delta -= profile.progress_count
    elif delta < -half_lap:
        delta += profile.progress_count
    if native_advance > 0 and delta < 0:
        delta = float(native_advance)
    elif native_advance < 0 and delta > 0:
        delta = float(native_advance)
    return max(-maximum_per_frame, min(maximum_per_frame, delta))
