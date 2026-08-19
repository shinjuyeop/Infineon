"""Record physical walking Slip/Sink GT observables and run a bounded pilot.

The output is train-only acquisition evidence.  No model, normalization,
detector threshold, persistence rule, INT8 artifact, or System-v1 behavior is
trained, tuned, exported, or changed by this runner.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from controlled_excitation import find_allowed_foot_geom_ids, has_nonfoot_floor_contact
from g1_upstream_locomotion import (
    TESTED_POLICY_SHA256,
    UPSTREAM_REVISION,
    UnitreeG1PretrainedController,
)
from hil_sensor import G1HilSensorReader, HIL_SENSOR_CHANNELS, LEFT_FOOT_CONTACT_GEOM_NAMES
from terrain_profiles import TERRAIN_PROFILES, TerrainProfile, apply_terrain_profile
from walking_hazard_ground_truth_v1 import (
    LOAD_THRESHOLD_N,
    MIN_STABLE_POST_TOUCHDOWN_SAMPLES,
    SENSOR_RATE_HZ,
    TOUCHDOWN_TRANSIENT_SAMPLES,
    box_surface_top_z,
    derive_contact_signals,
    episode_metric_rows,
    gait_phase,
    max_left_foot_contact_penetration_m,
    sole_sphere_lowest_point_z,
)


SIMULATION_DIR = Path(__file__).resolve().parents[2]
SCENE_PATH = (
    SIMULATION_DIR
    / "unitree_mujoco"
    / "unitree_robots"
    / "g1"
    / "scene_walking_terrain_transition.xml"
)
DEFAULT_POLICY = (
    SIMULATION_DIR
    / "unitree_rl_mjlab"
    / "deploy"
    / "robots"
    / "g1"
    / "config"
    / "policy"
    / "velocity"
    / "v0"
    / "exported"
    / "policy.onnx"
)
DEFAULT_OUTPUT = SIMULATION_DIR / "outputs" / "walking_hazard_ground_truth_v1_pilot"
PHYSICS_TIMESTEP_S = 0.0005
PHYSICS_STEPS_PER_SAMPLE = 2
DEFAULT_SPEEDS_MPS = (0.10, 0.15, 0.20)
DEFAULT_DURATION_S = 3.0
GROUND_NAMES = ("ground_source", "ground_target")
FOOT_BODY_NAME = "left_ankle_roll_link"
PELVIS_BODY_NAME = "pelvis"


@dataclass(frozen=True)
class AcquisitionCondition:
    """A planned physical condition; never a terrain-derived hazard label."""

    condition_name: str
    terrain_name: str
    profile: TerrainProfile
    acquisition_role: str
    walking_speed_mps: float
    replicate_index: int

    @property
    def run_id(self) -> str:
        speed = f"{self.walking_speed_mps:.2f}".replace(".", "p")
        return f"{self.condition_name}_{speed}_r{self.replicate_index:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--speeds", nargs="+", type=float, default=list(DEFAULT_SPEEDS_MPS))
    parser.add_argument("--runs-per-condition", type=int, default=1)
    parser.add_argument(
        "--sink-speed",
        type=float,
        default=0.10,
        help="fixed gait speed for the predeclared compliance candidates",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="collect one concrete trace only; readiness gates remain evidence based",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _sand_profile(
    name: str,
    *,
    solref: tuple[float, float] | None = None,
    solimp: tuple[float, float, float, float, float] | None = None,
    description: str,
) -> TerrainProfile:
    native = TERRAIN_PROFILES["sand"]
    return replace(
        native,
        name=name,
        solref=native.solref if solref is None else solref,
        solimp=native.solimp if solimp is None else solimp,
        description=description,
    )


def sink_candidate_profiles() -> tuple[TerrainProfile, ...]:
    """Return the fixed physics-design sweep declared before measurement."""
    hard = np.asarray(TERRAIN_PROFILES["concrete"].solref, dtype=float)
    native = np.asarray(TERRAIN_PROFILES["sand"].solref, dtype=float)
    one_third = tuple((hard + (native - hard) / 3.0).tolist())
    two_thirds = tuple((hard + 2.0 * (native - hard) / 3.0).tolist())
    return (
        _sand_profile(
            "sand_solref_interpolation_1of3",
            solref=one_third,
            description="one-third interpolation from hardened to native Sand solref",
        ),
        _sand_profile(
            "sand_solref_interpolation_2of3",
            solref=two_thirds,
            description="two-thirds interpolation from hardened to native Sand solref",
        ),
        _sand_profile(
            "sand_native",
            description="native Sand contact parameters",
        ),
        replace(
            TERRAIN_PROFILES["sand_slightly_compliant"],
            description="pre-existing bounded slightly-compliant Sand candidate",
        ),
        replace(
            TERRAIN_PROFILES["sand_moderately_compliant"],
            description="pre-existing bounded moderately-compliant Sand candidate",
        ),
    )


def acquisition_conditions(args: argparse.Namespace) -> list[AcquisitionCondition]:
    if args.runs_per_condition <= 0:
        raise ValueError("runs-per-condition must be positive")
    if args.duration_s <= 0.0:
        raise ValueError("duration must be positive")
    if any(not 0.1 <= speed <= 0.5 for speed in (*args.speeds, args.sink_speed)):
        raise ValueError("walking speeds must be within policy gait range [0.1, 0.5]")
    if args.smoke:
        return [
            AcquisitionCondition(
                "concrete_native", "concrete", TERRAIN_PROFILES["concrete"],
                "hard_negative", float(args.speeds[0]), 0,
            )
        ]

    hardened_sand = _sand_profile(
        "sand_walking_support_v1_hardened",
        solref=TERRAIN_PROFILES["concrete"].solref,
        description="walking-support-v1: native Sand except Concrete solref",
    )
    conditions: list[AcquisitionCondition] = []
    for replicate in range(args.runs_per_condition):
        for speed in args.speeds:
            for name, terrain, profile in (
                ("marble_native", "marble", TERRAIN_PROFILES["marble"]),
                ("concrete_native", "concrete", TERRAIN_PROFILES["concrete"]),
                ("sand_hardened", "sand", hardened_sand),
            ):
                conditions.append(
                    AcquisitionCondition(
                        name, terrain, profile, "hard_negative", float(speed), replicate
                    )
                )
            conditions.append(
                AcquisitionCondition(
                    "ice_native",
                    "ice",
                    TERRAIN_PROFILES["ice"],
                    "slip_candidate",
                    float(speed),
                    replicate,
                )
            )
        for profile in sink_candidate_profiles():
            conditions.append(
                AcquisitionCondition(
                    profile.name,
                    "sand",
                    profile,
                    "sink_candidate",
                    float(args.sink_speed),
                    replicate,
                )
            )
    return conditions


def protocol(args: argparse.Namespace, conditions: list[AcquisitionCondition]) -> dict[str, object]:
    return {
        "dataset": "walking_hazard_ground_truth_v1_pilot",
        "purpose": "train-only positive-acquisition evidence before label/threshold calibration",
        "split": "train_only",
        "test_or_final_data_used": False,
        "no_retraining_or_threshold_freeze": True,
        "terrain_name_never_labels_hazard": True,
        "air_samples_are_always_hazard_negative": True,
        "rate_hz": SENSOR_RATE_HZ,
        "physics_timestep_s": PHYSICS_TIMESTEP_S,
        "physics_steps_per_endpoint": PHYSICS_STEPS_PER_SAMPLE,
        "duration_s": args.duration_s,
        "contact_model": {
            "mode": "foot-spheres-only",
            "reason": "reuse the qualified walking foundation runtime isolation without editing source XML",
            "source_xml_unchanged": True,
            "nonfoot_contact_recorded": True,
            "note": "nonfoot terrain collisions are disabled by this inherited runtime mode; height/orientation fall gates remain active",
        },
        "coordinates": {
            "frame": "MuJoCo world frame; z is world up",
            "foot_xyz": f"world position of body {FOOT_BODY_NAME}; ankle z is diagnostic only",
            "foot_velocity_xyz": f"world-frame linear velocity of body {FOOT_BODY_NAME}",
            "pelvis_xyz": f"world position of body {PELVIS_BODY_NAME}",
            "sole_geoms": list(LEFT_FOOT_CONTACT_GEOM_NAMES),
            "sole_lowest_point_z": "minimum over geom_xpos[z] - sphere radius for the four named sole spheres",
            "ground_geoms": list(GROUND_NAMES),
            "surface_top_z": "fixed box world center z plus oriented world-z half extent; both pilot boxes have top z=0 m",
            "sole_to_surface_relative_depth": "surface_top_z - sole_lowest_point_z; positive means below the fixed top plane",
            "contact_penetration": "maximum max(0, -contact.dist) over named left-sole sphere/ground contacts",
        },
        "contact_episode": {
            "definition": "contiguous raw MuJoCo left-sole/ground contact",
            "anchor": "world x/y of left_ankle_roll_link at the first contact endpoint",
            "reset": "every AIR-to-contact transition",
            "loaded_contact": f"raw contact and FSR sum >= {LOAD_THRESHOLD_N:g} N",
            "touchdown_transient_samples": TOUCHDOWN_TRANSIENT_SAMPLES,
            "stable_loaded_post_touchdown_samples": MIN_STABLE_POST_TOUCHDOWN_SAMPLES,
            "loaded_penetration_reference": "first loaded endpoint in the current raw contact episode",
        },
        "fall_censor": {
            "first_fall_sample": "first 1-kHz endpoint with base z<0.55 m, pelvis upright-z<0.55, nonfoot surface contact, or nonfinite state",
            "calibration_mask": "sample index strictly less than first_fall_sample",
            "post_fall_trace_retained_for_diagnosis": True,
        },
        "separation_gates": {
            "slip": "minimum pre-fall Ice run maximum post-touchdown anchor drift exceeds maximum normal-control run value",
            "sink": "a predeclared candidate has stable pre-fall loaded contact and its run maximum contact penetration or positive surface-relative depth exceeds the normal run maximum",
            "threshold_frozen": False,
        },
        "controller": {
            "name": "unitreerobotics/unitree_rl_mjlab pretrained G1-29DOF velocity policy",
            "upstream_revision": UPSTREAM_REVISION,
            "tested_policy_sha256": TESTED_POLICY_SHA256,
        },
        "channels": list(HIL_SENSOR_CHANNELS),
        "trace_layout": {
            "endpoint_arrays": "shape [runs, samples, ...] in traces.npz",
            "run_constant_metadata_arrays": [
                "run_id", "terrain_name", "profile_name", "acquisition_role",
                "walking_speed_mps",
            ],
            "metadata_association": "run-constant arrays at index r apply to every endpoint in endpoint arrays[r]",
        },
        "conditions": [
            {
                "run_id": condition.run_id,
                "condition_name": condition.condition_name,
                "terrain_name": condition.terrain_name,
                "profile_name": condition.profile.name,
                "acquisition_role": condition.acquisition_role,
                "walking_speed_mps": condition.walking_speed_mps,
                "friction": condition.profile.friction,
                "solref": condition.profile.solref,
                "solimp": condition.profile.solimp,
            }
            for condition in conditions
        ],
        "overwrite_policy": "refuse any non-empty output directory",
    }


def _disable_nonfoot_surface_collisions(
    model: mujoco.MjModel, ground_ids: frozenset[int]
) -> frozenset[int]:
    allowed_feet = find_allowed_foot_geom_ids(model)
    for geom_id in range(model.ngeom):
        if geom_id in ground_ids or geom_id in allowed_feet:
            continue
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0
    return allowed_feet


def _fall_reasons(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pelvis_id: int,
    ground_ids: frozenset[int],
    allowed_feet: frozenset[int],
) -> tuple[list[str], bool]:
    nonfoot = any(
        has_nonfoot_floor_contact(data, ground_id, allowed_feet)
        for ground_id in ground_ids
    )
    reasons = []
    if float(data.qpos[2]) < 0.55:
        reasons.append("fallen_base_height")
    if float(data.xmat[pelvis_id, 8]) < 0.55:
        reasons.append("fallen_orientation")
    if nonfoot:
        reasons.append("nonfoot_surface_contact")
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
        reasons.append("nan_or_inf")
    return reasons, nonfoot


def collect_run(
    condition: AcquisitionCondition,
    policy_path: Path,
    duration_s: float,
) -> tuple[dict[str, np.ndarray], dict[str, object], list[dict[str, object]]]:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    model.opt.timestep = PHYSICS_TIMESTEP_S
    ground_ids = frozenset(model.geom(name).id for name in GROUND_NAMES)
    for ground_name in GROUND_NAMES:
        apply_terrain_profile(model, condition.profile, ground_name)
    allowed_feet = _disable_nonfoot_surface_collisions(model, ground_ids)
    data = mujoco.MjData(model)
    controller = UnitreeG1PretrainedController(
        model, data, policy_path, condition.walking_speed_mps
    )
    reader = G1HilSensorReader(model, data)
    sole_ids = tuple(reader.left_foot_geom_ids)
    foot_body_id = model.body(FOOT_BODY_NAME).id
    pelvis_id = model.body(PELVIS_BODY_NAME).id
    velocity = np.zeros(6, dtype=float)
    series: dict[str, list[object]] = {
        key: []
        for key in (
            "time_s",
            "fusion10",
            "left_contact",
            "loaded_contact",
            "foot_xyz",
            "foot_velocity_xyz",
            "pelvis_xyz",
            "pelvis_velocity_xyz",
            "local_surface_top_z_m",
            "sole_lowest_point_z_m",
            "sole_to_surface_relative_depth_m",
            "surface_relative_sole_depth_m",
            "max_contact_penetration_m",
            "ankle_body_z_diagnostic_m",
            "nonfoot_contact",
        )
    }
    first_fall_sample: int | None = None
    first_fall_time_s: float | None = None
    fall_reason = ""
    start_x = float(data.qpos[0])
    total_steps = int(round(duration_s / PHYSICS_TIMESTEP_S))
    for physics_step in range(1, total_steps + 1):
        controller.apply()
        mujoco.mj_step(model, data)
        controller.update_after_step()
        if physics_step % PHYSICS_STEPS_PER_SAMPLE:
            continue
        sensor = reader.read_vector()
        left_contact = reader.has_left_foot_contact(ground_ids)
        loaded_contact = bool(left_contact and sensor[:4].sum() >= LOAD_THRESHOLD_N)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, foot_body_id, velocity, 0
        )
        foot_velocity = velocity[3:].copy()
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, pelvis_id, velocity, 0
        )
        pelvis_velocity = velocity[3:].copy()
        surface_top = max(box_surface_top_z(model, data, value) for value in ground_ids)
        sole_lowest = sole_sphere_lowest_point_z(model, data, sole_ids)
        signed_depth = surface_top - sole_lowest
        penetration = max_left_foot_contact_penetration_m(data, sole_ids, ground_ids)
        reasons, nonfoot = _fall_reasons(
            model, data, pelvis_id, ground_ids, allowed_feet
        )
        if reasons and first_fall_sample is None:
            first_fall_sample = len(series["time_s"])
            first_fall_time_s = float(data.time)
            fall_reason = "|".join(reasons)
        values = {
            "time_s": float(data.time),
            "fusion10": sensor,
            "left_contact": left_contact,
            "loaded_contact": loaded_contact,
            "foot_xyz": data.xpos[foot_body_id].copy(),
            "foot_velocity_xyz": foot_velocity,
            "pelvis_xyz": data.xpos[pelvis_id].copy(),
            "pelvis_velocity_xyz": pelvis_velocity,
            "local_surface_top_z_m": surface_top,
            "sole_lowest_point_z_m": sole_lowest,
            "sole_to_surface_relative_depth_m": signed_depth,
            "surface_relative_sole_depth_m": max(0.0, signed_depth),
            "max_contact_penetration_m": penetration,
            "ankle_body_z_diagnostic_m": float(data.xpos[foot_body_id, 2]),
            "nonfoot_contact": nonfoot,
        }
        for key, value in values.items():
            series[key].append(value)

    trace = {key: np.asarray(value) for key, value in series.items()}
    signals = derive_contact_signals(
        trace["left_contact"],
        trace["loaded_contact"],
        trace["foot_xyz"],
        trace["foot_velocity_xyz"],
        trace["max_contact_penetration_m"],
        first_fall_sample,
    )
    trace.update(
        {
            "gait_phase": gait_phase(trace["loaded_contact"]),
            "contact_episode_id": signals.contact_episode_id,
            "touchdown_anchor_xy_m": signals.anchor_xy_m,
            "anchor_relative_xy_m": signals.anchor_relative_xy_m,
            "tangential_anchor_drift_m": signals.tangential_anchor_drift_m,
            "tangential_velocity_mps": signals.tangential_velocity_mps,
            "touchdown_transient": signals.touchdown_transient,
            "pre_fall_valid": signals.pre_fall_valid,
            "slip_calibration_valid": signals.slip_calibration_valid,
            "sink_calibration_valid": signals.sink_calibration_valid,
            "loaded_reference_penetration_m": signals.loaded_reference_penetration_m,
            "loaded_penetration_change_m": signals.loaded_penetration_change_m,
        }
    )
    episodes = episode_metric_rows(
        run_id=condition.run_id,
        terrain_name=condition.terrain_name,
        profile_name=condition.profile.name,
        acquisition_role=condition.acquisition_role,
        speed_mps=condition.walking_speed_mps,
        left_contact=trace["left_contact"],
        loaded_contact=trace["loaded_contact"],
        surface_relative_sole_depth_m=trace["surface_relative_sole_depth_m"],
        contact_penetration_m_values=trace["max_contact_penetration_m"],
        signals=signals,
    )
    stable_episode_count = sum(int(row["stable_loaded_contact_pre_fall"]) for row in episodes)
    metadata: dict[str, object] = {
        "run_id": condition.run_id,
        "condition_name": condition.condition_name,
        "terrain_name": condition.terrain_name,
        "profile_name": condition.profile.name,
        "acquisition_role": condition.acquisition_role,
        "walking_speed_mps": condition.walking_speed_mps,
        "replicate_index": condition.replicate_index,
        "duration_s": duration_s,
        "sample_count": len(trace["time_s"]),
        "effective_friction": "|".join(f"{value:.9g}" for value in condition.profile.friction),
        "effective_solref": "|".join(f"{value:.9g}" for value in condition.profile.solref),
        "effective_solimp": "|".join(f"{value:.9g}" for value in condition.profile.solimp),
        "contact_episode_count": len(episodes),
        "stable_loaded_episode_count_pre_fall": stable_episode_count,
        "stable_loaded_contact_pre_fall": bool(stable_episode_count),
        "fall_occurred": first_fall_sample is not None,
        "first_fall_sample": first_fall_sample,
        "first_fall_time_s": first_fall_time_s,
        "fall_reason": fall_reason,
        "nonfoot_contact_observed": bool(np.any(trace["nonfoot_contact"])),
        "forward_displacement_m": float(data.qpos[0] - start_x),
        "finite_state": bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))),
    }
    return trace, metadata, episodes


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _numeric_max(rows: list[dict[str, object]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field, "") != ""]
    return max(values) if values else None


def run_metrics(
    manifests: list[dict[str, object]], episodes: list[dict[str, object]]
) -> list[dict[str, object]]:
    result = []
    for manifest in manifests:
        selected = [row for row in episodes if row["run_id"] == manifest["run_id"]]
        result.append(
            {
                **manifest,
                "max_anchor_drift_post_touchdown_m": _numeric_max(
                    selected, "max_anchor_drift_post_touchdown_m"
                ),
                "max_contact_penetration_post_touchdown_m": _numeric_max(
                    selected, "max_contact_penetration_post_touchdown_m"
                ),
                "max_surface_relative_sole_depth_post_touchdown_m": _numeric_max(
                    selected, "max_surface_relative_sole_depth_post_touchdown_m"
                ),
                "max_loaded_penetration_change_m": _numeric_max(
                    selected, "max_loaded_penetration_change_m"
                ),
            }
        )
    return result


def analyze(
    manifests: list[dict[str, object]],
    episodes: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    runs = run_metrics(manifests, episodes)
    normal = [row for row in runs if row["acquisition_role"] == "hard_negative"]
    ice = [row for row in runs if row["acquisition_role"] == "slip_candidate"]
    sink = [row for row in runs if row["acquisition_role"] == "sink_candidate"]
    normal_drift_values = [
        float(row["max_anchor_drift_post_touchdown_m"])
        for row in normal if row["max_anchor_drift_post_touchdown_m"] is not None
    ]
    normal_penetration_values = [
        float(row["max_contact_penetration_post_touchdown_m"])
        for row in normal if row["max_contact_penetration_post_touchdown_m"] is not None
    ]
    normal_depth_values = [
        float(row["max_surface_relative_sole_depth_post_touchdown_m"])
        for row in normal
        if row["max_surface_relative_sole_depth_post_touchdown_m"] is not None
    ]
    normal_drift_max = max(normal_drift_values) if normal_drift_values else None
    normal_penetration_max = max(normal_penetration_values) if normal_penetration_values else None
    normal_depth_max = max(normal_depth_values) if normal_depth_values else None

    slip_rows = []
    for row in normal + ice:
        maximum = row["max_anchor_drift_post_touchdown_m"]
        slip_rows.append(
            {
                "run_id": row["run_id"],
                "acquisition_role": row["acquisition_role"],
                "terrain_name": row["terrain_name"],
                "profile_name": row["profile_name"],
                "walking_speed_mps": row["walking_speed_mps"],
                "fall_occurred": row["fall_occurred"],
                "first_fall_sample": row["first_fall_sample"],
                "stable_loaded_contact_pre_fall": row["stable_loaded_contact_pre_fall"],
                "max_anchor_drift_post_touchdown_m": maximum,
                "normal_run_max_envelope_m": normal_drift_max,
                "physically_separated_from_normal": bool(
                    row["acquisition_role"] == "slip_candidate"
                    and row["stable_loaded_contact_pre_fall"]
                    and maximum is not None
                    and normal_drift_max is not None
                    and float(maximum) > normal_drift_max
                ),
            }
        )

    sink_rows = []
    candidate_matrix = []
    for row in sink:
        penetration = row["max_contact_penetration_post_touchdown_m"]
        depth = row["max_surface_relative_sole_depth_post_touchdown_m"]
        penetration_separated = bool(
            penetration is not None
            and normal_penetration_max is not None
            and float(penetration) > normal_penetration_max
        )
        depth_separated = bool(
            depth is not None
            and normal_depth_max is not None
            and float(depth) > normal_depth_max
        )
        positive_evidence = bool(
            row["stable_loaded_contact_pre_fall"]
            and (penetration_separated or depth_separated)
        )
        comparison = {
            "run_id": row["run_id"],
            "terrain_name": row["terrain_name"],
            "profile_name": row["profile_name"],
            "walking_speed_mps": row["walking_speed_mps"],
            "fall_occurred": row["fall_occurred"],
            "first_fall_sample": row["first_fall_sample"],
            "fall_reason": row["fall_reason"],
            "stable_loaded_contact_pre_fall": row["stable_loaded_contact_pre_fall"],
            "max_contact_penetration_post_touchdown_m": penetration,
            "normal_contact_penetration_run_max_m": normal_penetration_max,
            "contact_penetration_separated_from_normal": penetration_separated,
            "max_surface_relative_sole_depth_post_touchdown_m": depth,
            "normal_surface_relative_sole_depth_run_max_m": normal_depth_max,
            "surface_relative_depth_separated_from_normal": depth_separated,
            "physical_sink_positive_evidence": positive_evidence,
        }
        sink_rows.append(comparison)
        candidate_matrix.append(
            {
                **comparison,
                "effective_friction": row["effective_friction"],
                "effective_solref": row["effective_solref"],
                "effective_solimp": row["effective_solimp"],
                "contact_episode_count": row["contact_episode_count"],
                "stable_loaded_episode_count_pre_fall": row[
                    "stable_loaded_episode_count_pre_fall"
                ],
                "max_loaded_penetration_change_m": row[
                    "max_loaded_penetration_change_m"
                ],
                "classification": (
                    "PHYSICAL_SINK_EVIDENCE"
                    if positive_evidence
                    else "FAILED_NO_STABLE_PRE_FALL_CONTACT"
                    if not row["stable_loaded_contact_pre_fall"]
                    else "NOT_SEPARATED_FROM_NORMAL"
                ),
            }
        )

    required_hard_negative_profiles = {
        "marble", "concrete", "sand_walking_support_v1_hardened"
    }
    hard_negative_ready = required_hard_negative_profiles.issubset(
        {str(row["profile_name"]) for row in normal}
    ) and all(
        row["stable_loaded_contact_pre_fall"]
        and not row["fall_occurred"]
        and float(row["forward_displacement_m"])
        >= 0.5 * float(row["walking_speed_mps"]) * float(row["duration_s"])
        for row in normal
    )
    ice_drift = [
        float(row["max_anchor_drift_post_touchdown_m"])
        for row in ice
        if row["stable_loaded_contact_pre_fall"]
        and row["max_anchor_drift_post_touchdown_m"] is not None
    ]
    slip_ready = bool(normal_drift_values and len(ice_drift) == len(ice)) and bool(ice) and (
        min(ice_drift) > max(normal_drift_values)
    )
    sink_ready = any(bool(row["physical_sink_positive_evidence"]) for row in sink_rows)
    recorder_ready = bool(traces) and all(
        len(trace["time_s"]) == len(trace["fusion10"])
        and trace["fusion10"].shape[1:] == (10,)
        for trace in traces
    )
    censor_ready = all(
        not np.any(trace["pre_fall_valid"][int(meta["first_fall_sample"]):])
        if meta["first_fall_sample"] is not None else np.all(trace["pre_fall_valid"])
        for trace, meta in zip(traces, manifests)
    )
    anchor_ready = all(
        np.all(trace["contact_episode_id"][~trace["left_contact"]] == -1)
        and not np.any(trace["slip_calibration_valid"][~trace["loaded_contact"]])
        for trace in traces
    )
    penetration_ready = all(
        np.all(trace["max_contact_penetration_m"] >= 0.0)
        and np.allclose(trace["local_surface_top_z_m"], 0.0, atol=1e-12)
        for trace in traces
    )
    gates = {
        "WALKING_HAZARD_GT_RECORDER_READY": recorder_ready,
        "WALKING_FIRST_FALL_CENSOR_READY": censor_ready,
        "WALKING_CONTACT_ANCHOR_DRIFT_READY": anchor_ready,
        "WALKING_TERRAIN_RELATIVE_PENETRATION_READY": penetration_ready,
        "WALKING_SLIP_POSITIVE_ACQUISITION_READY": slip_ready,
        "WALKING_SINK_POSITIVE_ACQUISITION_READY": sink_ready,
        "WALKING_HARD_NEGATIVE_SOURCE_READY": hard_negative_ready,
        "WALKING_BOUNDED_RETRAINING_AUTHORIZED": bool(
            recorder_ready
            and censor_ready
            and anchor_ready
            and penetration_ready
            and slip_ready
            and sink_ready
            and hard_negative_ready
        ),
    }
    summary: dict[str, object] = {
        "pilot_runs": len(runs),
        "run_counts": {
            "hard_negative": len(normal),
            "slip_candidate": len(ice),
            "sink_candidate": len(sink),
        },
        "fall_runs": sum(bool(row["fall_occurred"]) for row in runs),
        "first_fall_censored_samples": sum(
            int(row["sample_count"]) - int(row["first_fall_sample"])
            for row in runs if row["first_fall_sample"] is not None
        ),
        "normal_contact_anchor_drift_run_max_m": normal_drift_max,
        "ice_contact_anchor_drift_run_min_m": min(ice_drift) if ice_drift else None,
        "slip_motion_separated_from_normal": slip_ready,
        "normal_contact_penetration_run_max_m": normal_penetration_max,
        "normal_surface_relative_sole_depth_run_max_m": normal_depth_max,
        "sink_candidate_positive_evidence_runs": sum(
            bool(row["physical_sink_positive_evidence"]) for row in sink_rows
        ),
        "sink_candidate_positive_evidence_profiles": [
            row["profile_name"]
            for row in sink_rows if row["physical_sink_positive_evidence"]
        ],
        "gates": gates,
        **gates,
    }
    return slip_rows, sink_rows, candidate_matrix, summary


def representative_plots(
    output: Path,
    traces: list[dict[str, np.ndarray]],
    manifests: list[dict[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output / "plots"
    plot_dir.mkdir()
    normal_index = next(
        (index for index, row in enumerate(manifests) if row["acquisition_role"] == "hard_negative"),
        None,
    )
    ice_index = next(
        (index for index, row in enumerate(manifests) if row["acquisition_role"] == "slip_candidate"),
        None,
    )
    if normal_index is not None or ice_index is not None:
        figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
        for index in (normal_index, ice_index):
            if index is None:
                continue
            trace, metadata = traces[index], manifests[index]
            valid = trace["slip_calibration_valid"]
            axes[0].plot(
                trace["time_s"],
                trace["tangential_anchor_drift_m"],
                label=str(metadata["run_id"]),
            )
            axes[1].plot(
                trace["time_s"][valid],
                trace["tangential_velocity_mps"][valid],
                label=str(metadata["run_id"]),
            )
        axes[0].set_ylabel("anchor drift [m]")
        axes[1].set_ylabel("tangential velocity [m/s]")
        axes[1].set_xlabel("simulation time [s]")
        axes[0].legend()
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(plot_dir / "normal_vs_slip_anchor_drift.png", dpi=150)
        plt.close(figure)

    sink_indices = [
        index for index, row in enumerate(manifests)
        if row["acquisition_role"] == "sink_candidate"
    ]
    if normal_index is not None and sink_indices:
        figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for index in [normal_index, *sink_indices]:
            trace, metadata = traces[index], manifests[index]
            valid = trace["sink_calibration_valid"]
            axes[0].plot(
                trace["time_s"][valid],
                trace["max_contact_penetration_m"][valid],
                label=str(metadata["profile_name"]),
            )
            axes[1].plot(
                trace["time_s"][valid],
                trace["surface_relative_sole_depth_m"][valid],
                label=str(metadata["profile_name"]),
            )
        axes[0].set_ylabel("contact penetration [m]")
        axes[1].set_ylabel("sole depth [m]")
        axes[1].set_xlabel("simulation time [s]")
        axes[0].legend(fontsize=7)
        axes[1].legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(plot_dir / "normal_vs_sink_candidate_penetration.png", dpi=150)
        plt.close(figure)

    fall_index = next(
        (index for index, row in enumerate(manifests) if row["fall_occurred"]), None
    )
    if fall_index is not None:
        trace, metadata = traces[fall_index], manifests[fall_index]
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.plot(trace["time_s"], trace["pelvis_xyz"][:, 2], label="pelvis z")
        axis.step(
            trace["time_s"], trace["pre_fall_valid"].astype(float),
            where="post", label="pre-fall calibration mask",
        )
        fall_sample = int(metadata["first_fall_sample"])
        axis.axvline(trace["time_s"][fall_sample], color="red", linestyle="--")
        axis.set_xlabel("simulation time [s]")
        axis.legend()
        figure.tight_layout()
        figure.savefig(plot_dir / "representative_first_fall_censor.png", dpi=150)
        plt.close(figure)


def audit_markdown(
    summary: dict[str, object],
    slip_rows: list[dict[str, object]],
    sink_rows: list[dict[str, object]],
) -> str:
    gates = summary["gates"]
    slip_min = summary["ice_contact_anchor_drift_run_min_m"]
    normal_drift = summary["normal_contact_anchor_drift_run_max_m"]
    positive_sink = [
        row["profile_name"] for row in sink_rows if row["physical_sink_positive_evidence"]
    ]
    fall_rows = sum(bool(row["fall_occurred"]) for row in slip_rows + sink_rows)
    return f"""# Walking physical hazard ground-truth pilot v1

This checkpoint records physical observables only.  Terrain identity was not
used as a Slip or Sink label, AIR was excluded, all evidence at and after the
first sampled fall was censored, and no model or detector threshold was
trained or frozen.

## Acquisition

- Runs: {summary['pilot_runs']} ({summary['run_counts']['hard_negative']} normal hard-negative,
  {summary['run_counts']['slip_candidate']} Slip candidate, {summary['run_counts']['sink_candidate']} Sink candidate)
- Candidate runs with a recorded fall: {fall_rows}
- Samples censored from first fall onward: {summary['first_fall_censored_samples']}

## Slip evidence

The maximum normal pre-fall, post-touchdown contact-anchor drift was
{normal_drift if normal_drift is not None else 'unavailable'} m.  The minimum
Ice candidate run maximum was {slip_min if slip_min is not None else 'unavailable'} m.
Separation is therefore **{'observed' if gates['WALKING_SLIP_POSITIVE_ACQUISITION_READY'] else 'not established'}**.
This is an envelope comparison, not a frozen detector threshold.

## Sink evidence

The normal maximum post-touchdown contact penetration was
{summary['normal_contact_penetration_run_max_m']} m and the normal maximum
surface-relative sole depth was {summary['normal_surface_relative_sole_depth_run_max_m']} m.
Profiles with stable, pre-fall measurements separated from either normal
envelope: {', '.join(positive_sink) if positive_sink else 'none'}.

## Readiness gates

""" + "\n".join(f"- {name}={str(value).lower()}" for name, value in gates.items()) + """

Even if the final authorization gate is true, retraining is intentionally not
part of this checkpoint.
"""


def main() -> None:
    args = parse_args()
    conditions = acquisition_conditions(args)
    planned_protocol = protocol(args, conditions)
    if not args.execute:
        print(json.dumps(planned_protocol, indent=2))
        return
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    if not args.policy_path.is_file():
        raise FileNotFoundError(args.policy_path)
    policy_path = args.policy_path.resolve()
    policy_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    if policy_hash != TESTED_POLICY_SHA256:
        raise ValueError(f"policy hash mismatch: {policy_hash}")
    output.mkdir(parents=True, exist_ok=True)
    planned_protocol["controller"]["policy_path"] = str(policy_path)
    planned_protocol["controller"]["policy_sha256"] = policy_hash

    traces: list[dict[str, np.ndarray]] = []
    manifests: list[dict[str, object]] = []
    episodes: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, condition in enumerate(conditions):
        trace, metadata, current_episodes = collect_run(
            condition, policy_path, args.duration_s
        )
        traces.append(trace)
        manifests.append(metadata)
        episodes.extend(current_episodes)
        elapsed = time.perf_counter() - started
        mean = elapsed / (index + 1)
        eta = mean * (len(conditions) - index - 1)
        print(
            f"[{index + 1}/{len(conditions)}] {condition.run_id} "
            f"episodes={metadata['contact_episode_count']} "
            f"stable={metadata['stable_loaded_episode_count_pre_fall']} "
            f"fall={metadata['fall_occurred']} elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )

    lengths = {len(trace["time_s"]) for trace in traces}
    if len(lengths) != 1:
        raise ValueError(f"trace lengths differ: {sorted(lengths)}")
    packed = {key: np.asarray([trace[key] for trace in traces]) for key in traces[0]}
    packed.update(
        {
            "run_id": np.asarray([row["run_id"] for row in manifests]),
            "terrain_name": np.asarray([row["terrain_name"] for row in manifests]),
            "profile_name": np.asarray([row["profile_name"] for row in manifests]),
            "acquisition_role": np.asarray([row["acquisition_role"] for row in manifests]),
            "walking_speed_mps": np.asarray(
                [row["walking_speed_mps"] for row in manifests], dtype=float
            ),
        }
    )
    np.savez_compressed(output / "traces.npz", **packed)
    slip_rows, sink_rows, candidate_matrix, summary = analyze(
        manifests, episodes, traces
    )
    summary["wall_time_s"] = time.perf_counter() - started
    (output / "protocol.json").write_text(
        json.dumps(planned_protocol, indent=2) + "\n", encoding="utf-8"
    )
    (output / "manifest.json").write_text(
        json.dumps(manifests, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output / "contact_episode_metrics.csv", episodes)
    write_csv(output / "fall_audit.csv", manifests)
    write_csv(output / "normal_vs_slip_metrics.csv", slip_rows)
    write_csv(output / "normal_vs_sink_candidate_metrics.csv", sink_rows)
    write_csv(output / "candidate_physics_matrix.csv", candidate_matrix)
    representative_plots(output, traces, manifests)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output / "audit.md").write_text(
        audit_markdown(summary, slip_rows, sink_rows), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
