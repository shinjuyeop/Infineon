"""Physical walking Slip/Sink observables and contact-episode bookkeeping.

This module deliberately contains no learned-model, detector-threshold, or
terrain-name label logic.  It converts MuJoCo geometry/contact measurements
into auditable physical signals that a later train-only calibration may use.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


SENSOR_RATE_HZ = 1000
LOAD_THRESHOLD_N = 5.0
TOUCHDOWN_TRANSIENT_SAMPLES = 10
MIN_STABLE_POST_TOUCHDOWN_SAMPLES = 20


def contact_penetration_m(contact_distance_m: float) -> float:
    """Convert MuJoCo's signed contact distance to positive penetration."""
    return max(0.0, -float(contact_distance_m))


def box_surface_top_z(
    model: mujoco.MjModel, data: mujoco.MjData, geom_id: int
) -> float:
    """Return the highest world-z point of a box geometry.

    The pilot boxes are fixed and axis aligned, for which this is exactly
    ``world_center_z + half_size_z``.  The full oriented-box expression keeps
    the coordinate convention explicit and makes the helper safe in tests.
    """
    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        raise ValueError("surface-top geometry must be a box")
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    world_z_extent = float(np.abs(rotation[2]) @ model.geom_size[geom_id])
    return float(data.geom_xpos[geom_id, 2] + world_z_extent)


def sole_sphere_lowest_point_z(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sole_geom_ids: tuple[int, ...] | frozenset[int],
) -> float:
    """Return the minimum world-z point of the named sole spheres."""
    if not sole_geom_ids:
        raise ValueError("at least one sole sphere is required")
    bottoms = []
    for geom_id in sole_geom_ids:
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_SPHERE):
            raise ValueError(f"sole geometry {geom_id} is not a sphere")
        bottoms.append(float(data.geom_xpos[geom_id, 2] - model.geom_size[geom_id, 0]))
    return min(bottoms)


def max_left_foot_contact_penetration_m(
    data: mujoco.MjData,
    sole_geom_ids: tuple[int, ...] | frozenset[int],
    ground_geom_ids: tuple[int, ...] | frozenset[int],
) -> float:
    """Return max(0, -contact.dist) for left-sole/ground contacts."""
    soles = frozenset(int(value) for value in sole_geom_ids)
    grounds = frozenset(int(value) for value in ground_geom_ids)
    maximum = 0.0
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        pair = frozenset((int(contact.geom1), int(contact.geom2)))
        if pair & soles and pair & grounds:
            maximum = max(maximum, contact_penetration_m(float(contact.dist)))
    return maximum


def calibration_mask(sample_count: int, first_fall_sample: int | None) -> np.ndarray:
    """Mask calibration evidence at and after the first sampled fall endpoint."""
    result = np.ones(sample_count, dtype=bool)
    if first_fall_sample is not None:
        if not 0 <= first_fall_sample < sample_count:
            raise ValueError("first_fall_sample is outside the trace")
        result[first_fall_sample:] = False
    return result


def contact_episodes(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return all half-open physical-contact episodes, including short ones."""
    values = np.asarray(mask, dtype=bool)
    edges = np.diff(np.r_[False, values, False].astype(np.int8))
    return [
        (int(start), int(end))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
    ]


def gait_phase(loaded_contact: np.ndarray) -> np.ndarray:
    """Assign the existing deterministic 10/20/final-10-ms loaded phases."""
    loaded = np.asarray(loaded_contact, dtype=bool)
    result = np.full(len(loaded), "AIR", dtype="<U10")
    for start, end in contact_episodes(loaded):
        result[start:min(start + 10, end)] = "TOUCHDOWN"
        result[min(start + 10, end):min(start + 30, end)] = "LOADING"
        result[max(start + 30, end - 10):end] = "PUSH_OFF"
        if end - start > 40:
            result[start + 30:end - 10] = "MID_STANCE"
    return result


@dataclass(frozen=True)
class DerivedContactSignals:
    """Per-endpoint physical signals derived without hazard thresholds."""

    contact_episode_id: np.ndarray
    anchor_xy_m: np.ndarray
    anchor_relative_xy_m: np.ndarray
    tangential_anchor_drift_m: np.ndarray
    tangential_velocity_mps: np.ndarray
    touchdown_transient: np.ndarray
    pre_fall_valid: np.ndarray
    slip_calibration_valid: np.ndarray
    sink_calibration_valid: np.ndarray
    loaded_reference_penetration_m: np.ndarray
    loaded_penetration_change_m: np.ndarray


def derive_contact_signals(
    left_contact: np.ndarray,
    loaded_contact: np.ndarray,
    foot_xyz: np.ndarray,
    foot_velocity_xyz: np.ndarray,
    contact_penetration: np.ndarray,
    first_fall_sample: int | None,
    touchdown_transient_samples: int = TOUCHDOWN_TRANSIENT_SAMPLES,
) -> DerivedContactSignals:
    """Build anchor-relative/contact-relative signals for one trace.

    AIR coordinates and penetration references are NaN and both calibration
    masks are false.  A new raw physical-contact episode always resets the
    anchor, even when force loading begins several samples later.
    """
    contact = np.asarray(left_contact, dtype=bool)
    loaded = np.asarray(loaded_contact, dtype=bool)
    xyz = np.asarray(foot_xyz, dtype=float)
    velocity = np.asarray(foot_velocity_xyz, dtype=float)
    penetration = np.asarray(contact_penetration, dtype=float)
    sample_count = len(contact)
    if (
        loaded.shape != (sample_count,)
        or xyz.shape != (sample_count, 3)
        or velocity.shape != (sample_count, 3)
        or penetration.shape != (sample_count,)
    ):
        raise ValueError("contact signal arrays have inconsistent shapes")
    if np.any(loaded & ~contact):
        raise ValueError("loaded contact cannot be true in AIR")
    if touchdown_transient_samples < 0:
        raise ValueError("touchdown transient must be nonnegative")

    episode_id = np.full(sample_count, -1, dtype=np.int32)
    anchor = np.full((sample_count, 2), np.nan, dtype=float)
    relative = np.full((sample_count, 2), np.nan, dtype=float)
    drift = np.full(sample_count, np.nan, dtype=float)
    tangential_velocity = np.full(sample_count, np.nan, dtype=float)
    transient = np.zeros(sample_count, dtype=bool)
    loaded_reference = np.full(sample_count, np.nan, dtype=float)
    penetration_change = np.full(sample_count, np.nan, dtype=float)

    for current_id, (start, end) in enumerate(contact_episodes(contact)):
        episode_id[start:end] = current_id
        touchdown_anchor = xyz[start, :2].copy()
        anchor[start:end] = touchdown_anchor
        relative[start:end] = xyz[start:end, :2] - touchdown_anchor
        drift[start:end] = np.linalg.norm(relative[start:end], axis=1)
        tangential_velocity[start:end] = np.linalg.norm(velocity[start:end, :2], axis=1)
        transient[start:min(start + touchdown_transient_samples, end)] = True
        loaded_indices = np.flatnonzero(loaded[start:end])
        if loaded_indices.size:
            first_loaded = start + int(loaded_indices[0])
            reference = float(penetration[first_loaded])
            selected = np.arange(first_loaded, end)
            selected = selected[loaded[selected]]
            loaded_reference[selected] = reference
            penetration_change[selected] = penetration[selected] - reference

    pre_fall = calibration_mask(sample_count, first_fall_sample)
    slip_valid = loaded & ~transient & pre_fall
    sink_valid = slip_valid & np.isfinite(penetration_change)
    return DerivedContactSignals(
        contact_episode_id=episode_id,
        anchor_xy_m=anchor,
        anchor_relative_xy_m=relative,
        tangential_anchor_drift_m=drift,
        tangential_velocity_mps=tangential_velocity,
        touchdown_transient=transient,
        pre_fall_valid=pre_fall,
        slip_calibration_valid=slip_valid,
        sink_calibration_valid=sink_valid,
        loaded_reference_penetration_m=loaded_reference,
        loaded_penetration_change_m=penetration_change,
    )


def episode_metric_rows(
    *,
    run_id: str,
    terrain_name: str,
    profile_name: str,
    acquisition_role: str,
    speed_mps: float,
    left_contact: np.ndarray,
    loaded_contact: np.ndarray,
    surface_relative_sole_depth_m: np.ndarray,
    contact_penetration_m_values: np.ndarray,
    signals: DerivedContactSignals,
) -> list[dict[str, object]]:
    """Summarize every physical-contact episode in a trace."""
    contact = np.asarray(left_contact, dtype=bool)
    loaded = np.asarray(loaded_contact, dtype=bool)
    depth = np.asarray(surface_relative_sole_depth_m, dtype=float)
    penetration = np.asarray(contact_penetration_m_values, dtype=float)
    rows: list[dict[str, object]] = []
    for episode_id, (start, end) in enumerate(contact_episodes(contact)):
        episode = np.arange(start, end)
        pre_fall = episode[signals.pre_fall_valid[episode]]
        slip_valid = episode[signals.slip_calibration_valid[episode]]
        sink_valid = episode[signals.sink_calibration_valid[episode]]
        loaded_pre_fall = pre_fall[loaded[pre_fall]]
        last_valid = int(pre_fall[-1]) if pre_fall.size else start
        rows.append(
            {
                "run_id": run_id,
                "terrain_name": terrain_name,
                "profile_name": profile_name,
                "acquisition_role": acquisition_role,
                "walking_speed_mps": speed_mps,
                "contact_episode_id": episode_id,
                "start_sample": start,
                "end_sample_exclusive": end,
                "duration_samples": end - start,
                "duration_ms": end - start,
                "loaded_samples_pre_fall": int(loaded_pre_fall.size),
                "post_touchdown_slip_samples_pre_fall": int(slip_valid.size),
                "post_touchdown_sink_samples_pre_fall": int(sink_valid.size),
                "max_anchor_drift_m": (
                    float(np.nanmax(signals.tangential_anchor_drift_m[pre_fall]))
                    if pre_fall.size else ""
                ),
                "net_anchor_drift_m": (
                    float(np.linalg.norm(signals.anchor_relative_xy_m[last_valid]))
                    if pre_fall.size else ""
                ),
                "max_anchor_drift_post_touchdown_m": (
                    float(np.nanmax(signals.tangential_anchor_drift_m[slip_valid]))
                    if slip_valid.size else ""
                ),
                "max_tangential_velocity_post_touchdown_mps": (
                    float(np.nanmax(signals.tangential_velocity_mps[slip_valid]))
                    if slip_valid.size else ""
                ),
                "max_contact_penetration_m": (
                    float(np.max(penetration[loaded_pre_fall]))
                    if loaded_pre_fall.size else ""
                ),
                "max_contact_penetration_post_touchdown_m": (
                    float(np.max(penetration[sink_valid])) if sink_valid.size else ""
                ),
                "max_surface_relative_sole_depth_m": (
                    float(np.max(depth[loaded_pre_fall])) if loaded_pre_fall.size else ""
                ),
                "max_surface_relative_sole_depth_post_touchdown_m": (
                    float(np.max(depth[sink_valid])) if sink_valid.size else ""
                ),
                "max_loaded_penetration_change_m": (
                    float(np.nanmax(signals.loaded_penetration_change_m[sink_valid]))
                    if sink_valid.size else ""
                ),
                "stable_loaded_contact_pre_fall": int(
                    sink_valid.size >= MIN_STABLE_POST_TOUCHDOWN_SAMPLES
                ),
                "entire_episode_pre_fall": int(
                    bool(pre_fall.size) and pre_fall.size == end - start
                ),
                "has_pre_fall_calibration_samples": int(bool(slip_valid.size)),
            }
        )
    return rows
