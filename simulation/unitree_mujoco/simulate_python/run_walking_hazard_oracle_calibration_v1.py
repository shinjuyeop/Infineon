"""Acquire diverse walking replicates and calibrate physical oracle candidates.

This is a train/calibration-validation-only robustness checkpoint.  Selected
thresholds remain audit candidates: no production/frozen threshold, model,
INT8 artifact, E84 deployment, or System-v1 runtime is changed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from g1_upstream_locomotion import (
    TESTED_POLICY_SHA256,
    UPSTREAM_REVISION,
    UnitreeG1PretrainedController,
)
from hil_sensor import G1HilSensorReader, HIL_SENSOR_CHANNELS
from terrain_profiles import TERRAIN_PROFILES, TerrainProfile, apply_terrain_profile
from walking_hazard_ground_truth_v1 import (
    LOAD_THRESHOLD_N,
    SENSOR_RATE_HZ,
    box_surface_top_z,
    derive_contact_signals,
    episode_metric_rows,
    gait_phase,
    max_left_foot_contact_penetration_m,
    sole_sphere_lowest_point_z,
)
from walking_hazard_oracle_calibration_v1 import (
    PERSISTENCE_GRID_MS,
    SINK_THRESHOLD_GRID_M,
    SLIP_THRESHOLD_GRID_M,
    assign_calibration_splits,
    duplicate_trace_audit,
    persistent_oracle,
    split_integrity,
)
from run_walking_hazard_ground_truth_v1 import (
    DEFAULT_POLICY,
    FOOT_BODY_NAME,
    GROUND_NAMES,
    PELVIS_BODY_NAME,
    PHYSICS_STEPS_PER_SAMPLE,
    PHYSICS_TIMESTEP_S,
    SCENE_PATH,
    _disable_nonfoot_surface_collisions,
    _fall_reasons,
    sink_candidate_profiles,
    write_csv,
)


SIMULATION_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = SIMULATION_DIR / "outputs" / "walking_hazard_oracle_calibration_v1"
DEFAULT_SPEEDS_MPS = (0.10, 0.15, 0.20)
DEFAULT_DURATION_S = 3.0
VARIATION_SEED_BASE = 202608190
TARGET_SPEEDS = frozenset(DEFAULT_SPEEDS_MPS)


@dataclass(frozen=True)
class Variation:
    index: int
    seed: int
    initial_locomotion_phase_fraction: float
    command_onset_delay_s: float


VARIATIONS = (
    Variation(0, VARIATION_SEED_BASE + 0, 0.0, 0.000),
    Variation(1, VARIATION_SEED_BASE + 1, 1.0 / 3.0, 0.020),
    Variation(2, VARIATION_SEED_BASE + 2, 2.0 / 3.0, 0.040),
)


@dataclass(frozen=True)
class RobustnessCondition:
    condition_name: str
    terrain_name: str
    profile: TerrainProfile
    acquisition_role: str
    walking_speed_mps: float
    variation: Variation

    @property
    def run_id(self) -> str:
        speed = f"{self.walking_speed_mps:.2f}".replace(".", "p")
        return f"{self.condition_name}_{speed}_v{self.variation.index:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--speeds", nargs="+", type=float, default=list(DEFAULT_SPEEDS_MPS))
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="collect three varied Concrete traces without performing calibration",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def hardened_sand_profile() -> TerrainProfile:
    return replace(
        TERRAIN_PROFILES["sand"],
        name="sand_walking_support_v1_hardened",
        solref=TERRAIN_PROFILES["concrete"].solref,
        description="walking-support-v1: native Sand except Concrete solref",
    )


def acquisition_conditions(args: argparse.Namespace) -> list[RobustnessCondition]:
    if args.replicates != len(VARIATIONS):
        raise ValueError("this bounded protocol requires exactly three declared variations")
    if args.duration_s <= 0.0:
        raise ValueError("duration must be positive")
    if any(not 0.1 <= speed <= 0.5 for speed in args.speeds):
        raise ValueError("walking speeds must be within [0.1,0.5] m/s")
    if args.smoke:
        return [
            RobustnessCondition(
                "concrete_native", "concrete", TERRAIN_PROFILES["concrete"],
                "hard_negative", float(args.speeds[0]), variation,
            )
            for variation in VARIATIONS
        ]
    if frozenset(float(speed) for speed in args.speeds) != TARGET_SPEEDS:
        raise ValueError("the robustness protocol requires speeds 0.10, 0.15, 0.20 m/s")
    primary_sink = {
        profile.name: profile for profile in sink_candidate_profiles()
        if profile.name in {
            "sand_solref_interpolation_1of3",
            "sand_solref_interpolation_2of3",
        }
    }
    conditions: list[RobustnessCondition] = []
    for variation in VARIATIONS:
        for speed in args.speeds:
            for name, terrain, profile, role in (
                ("marble_native", "marble", TERRAIN_PROFILES["marble"], "hard_negative"),
                ("concrete_native", "concrete", TERRAIN_PROFILES["concrete"], "hard_negative"),
                ("sand_hardened", "sand", hardened_sand_profile(), "hard_negative"),
                ("ice_native", "ice", TERRAIN_PROFILES["ice"], "slip_candidate"),
                (
                    "sand_solref_interpolation_1of3", "sand",
                    primary_sink["sand_solref_interpolation_1of3"], "sink_candidate",
                ),
                (
                    "sand_solref_interpolation_2of3", "sand",
                    primary_sink["sand_solref_interpolation_2of3"], "sink_candidate",
                ),
            ):
                conditions.append(RobustnessCondition(
                    name, terrain, profile, role, float(speed), variation
                ))
    return conditions


def protocol(args: argparse.Namespace, conditions: list[RobustnessCondition]) -> dict[str, object]:
    return {
        "dataset": "walking_hazard_oracle_calibration_v1",
        "purpose": "physical-oracle replicate robustness and train-only calibration",
        "starting_checkpoint": "40aa6afaa364040b0b2ce52883cd846cc437dab7",
        "existing_pilot_overwritten": False,
        "partitions": ["calibration_train", "calibration_validation"],
        "test_or_final_data_used": False,
        "model_retraining_performed": False,
        "production_threshold_frozen_or_changed": False,
        "terrain_identity_is_not_a_hazard_label": True,
        "sample_rate_hz": SENSOR_RATE_HZ,
        "duration_s": args.duration_s,
        "physics_timestep_s": PHYSICS_TIMESTEP_S,
        "variation_design": {
            "kind": "deterministic bounded initial locomotion-state variation",
            "rationale": "phase-stratified initial policy state plus at most two 50-Hz command ticks of onset delay; no sensor, actuator, contact, or label noise",
            "seed_semantics": "stable protocol identifier; actual phase and onset values are explicitly listed",
            "split": "variation indices 0 and 1 train; index 2 held out for calibration validation",
            "variations": [vars(variation) for variation in VARIATIONS],
        },
        "matrix": {
            "normal_hard_negative": "Marble/Concrete/hardened Sand x 0.10/0.15/0.20 m/s x 3 variations",
            "slip_candidate": "Ice x 0.10/0.15/0.20 m/s x 3 variations",
            "sink_primary_candidate": "Sand solref interpolation 1of3/2of3 x 0.10/0.15/0.20 m/s x 3 variations",
            "native_and_more_compliant_sand": "not reacquired; retained as fall diagnostics in the immutable 40aa6af pilot",
            "run_count": len(conditions),
        },
        "physical_oracles": {
            "slip": {
                "primary": "contact-anchor-relative tangential drift",
                "valid": "loaded and post-touchdown and pre-fall",
                "auxiliary_only": "tangential velocity",
                "threshold_grid_m": SLIP_THRESHOLD_GRID_M,
            },
            "sink": {
                "primary": "loaded_penetration_change_m from first loaded contact in episode",
                "valid": "loaded and post-touchdown and pre-fall",
                "audit_only": ["absolute contact penetration", "surface-relative sole depth"],
                "threshold_grid_m": SINK_THRESHOLD_GRID_M,
            },
            "persistence_grid_ms": PERSISTENCE_GRID_MS,
            "selection": "among train-passing candidates, maximize complete positive profiles, then physical envelope margin, then persistence, then minimize latency",
            "selected_threshold_is_production_frozen": False,
        },
        "acceptance": {
            "normal_validation_false_positive_runs": 0,
            "slip_validation": "positive detection at every target speed",
            "sink_validation": "one primary profile has no-fall stable physical evidence and detection at every target speed",
            "air_post_fall_touchdown_transient_violations": 0,
            "duplicate_replicates": 0,
            "split_leakage": 0,
        },
        "controller": {
            "upstream_revision": UPSTREAM_REVISION,
            "tested_policy_sha256": TESTED_POLICY_SHA256,
        },
        "channels": list(HIL_SENSOR_CHANNELS),
        "conditions": [{
            "run_id": item.run_id,
            "condition_name": item.condition_name,
            "terrain_name": item.terrain_name,
            "profile_name": item.profile.name,
            "acquisition_role": item.acquisition_role,
            "walking_speed_mps": item.walking_speed_mps,
            "variation_index": item.variation.index,
            "variation_seed": item.variation.seed,
            "initial_locomotion_phase_fraction": item.variation.initial_locomotion_phase_fraction,
            "command_onset_delay_s": item.variation.command_onset_delay_s,
            "friction": item.profile.friction,
            "solref": item.profile.solref,
            "solimp": item.profile.solimp,
        } for item in conditions],
        "overwrite_policy": "refuse any non-empty output directory",
    }


def collect_run(
    condition: RobustnessCondition,
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
    controller.global_phase = condition.variation.initial_locomotion_phase_fraction
    nominal_command = controller.command.copy()
    if condition.variation.command_onset_delay_s > 0.0:
        controller.command[:] = 0.0
    reader = G1HilSensorReader(model, data)
    sole_ids = tuple(reader.left_foot_geom_ids)
    foot_body_id = model.body(FOOT_BODY_NAME).id
    pelvis_id = model.body(PELVIS_BODY_NAME).id
    velocity = np.zeros(6, dtype=float)
    keys = (
        "time_s", "fusion10", "left_contact", "loaded_contact", "foot_xyz",
        "foot_velocity_xyz", "pelvis_xyz", "pelvis_velocity_xyz",
        "local_surface_top_z_m", "sole_lowest_point_z_m",
        "sole_to_surface_relative_depth_m", "surface_relative_sole_depth_m",
        "max_contact_penetration_m", "ankle_body_z_diagnostic_m",
        "nonfoot_contact",
    )
    series: dict[str, list[object]] = {key: [] for key in keys}
    first_fall_sample: int | None = None
    first_fall_time_s: float | None = None
    first_nonzero_policy_command_time_s: float | None = None
    fall_reason = ""
    start_x = float(data.qpos[0])
    total_steps = int(round(duration_s / PHYSICS_TIMESTEP_S))
    for physics_step in range(1, total_steps + 1):
        if data.time + 1e-12 >= condition.variation.command_onset_delay_s:
            controller.command[:] = nominal_command
        controller.apply()
        mujoco.mj_step(model, data)
        controller.update_after_step()
        if (
            first_nonzero_policy_command_time_s is None
            and controller.step_count % controller.control_decimation == 0
            and float(controller.command[0]) > 0.0
        ):
            first_nonzero_policy_command_time_s = float(data.time)
        if physics_step % PHYSICS_STEPS_PER_SAMPLE:
            continue
        sensor = reader.read_vector()
        left_contact = reader.has_left_foot_contact(ground_ids)
        loaded = bool(left_contact and sensor[:4].sum() >= LOAD_THRESHOLD_N)
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
        reasons, nonfoot = _fall_reasons(model, data, pelvis_id, ground_ids, allowed_feet)
        if reasons and first_fall_sample is None:
            first_fall_sample = len(series["time_s"])
            first_fall_time_s = float(data.time)
            fall_reason = "|".join(reasons)
        values = {
            "time_s": float(data.time),
            "fusion10": sensor,
            "left_contact": left_contact,
            "loaded_contact": loaded,
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
        trace["left_contact"], trace["loaded_contact"], trace["foot_xyz"],
        trace["foot_velocity_xyz"], trace["max_contact_penetration_m"],
        first_fall_sample,
    )
    trace.update({
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
    })
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
    stable_count = sum(int(row["stable_loaded_contact_pre_fall"]) for row in episodes)
    metadata: dict[str, object] = {
        "run_id": condition.run_id,
        "condition_name": condition.condition_name,
        "terrain_name": condition.terrain_name,
        "profile_name": condition.profile.name,
        "acquisition_role": condition.acquisition_role,
        "walking_speed_mps": condition.walking_speed_mps,
        "variation_index": condition.variation.index,
        "variation_seed": condition.variation.seed,
        "initial_locomotion_phase_fraction": condition.variation.initial_locomotion_phase_fraction,
        "command_onset_delay_s": condition.variation.command_onset_delay_s,
        "first_nonzero_policy_command_time_s": first_nonzero_policy_command_time_s,
        "duration_s": duration_s,
        "sample_count": len(trace["time_s"]),
        "effective_friction": "|".join(f"{v:.9g}" for v in condition.profile.friction),
        "effective_solref": "|".join(f"{v:.9g}" for v in condition.profile.solref),
        "effective_solimp": "|".join(f"{v:.9g}" for v in condition.profile.solimp),
        "contact_episode_count": len(episodes),
        "stable_loaded_episode_count_pre_fall": stable_count,
        "stable_loaded_contact_pre_fall": bool(stable_count),
        "fall_occurred": first_fall_sample is not None,
        "first_fall_sample": first_fall_sample,
        "first_fall_time_s": first_fall_time_s,
        "fall_reason": fall_reason,
        "nonfoot_contact_observed": bool(np.any(trace["nonfoot_contact"])),
        "forward_displacement_m": float(data.qpos[0] - start_x),
        "finite_state": bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))),
    }
    return trace, metadata, episodes


def _run_max(trace: dict[str, np.ndarray], hazard: str) -> float | None:
    if hazard == "slip":
        values = trace["tangential_anchor_drift_m"]
        valid = trace["slip_calibration_valid"]
    else:
        values = trace["loaded_penetration_change_m"]
        valid = trace["sink_calibration_valid"]
    return float(np.nanmax(values[valid])) if np.any(valid) else None


def annotate_physical_sources(
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
    split_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    split_by_run = {str(row["run_id"]): str(row["split"]) for row in split_rows}
    metrics: dict[str, dict[str, object]] = {}
    for metadata, trace in zip(manifests, traces):
        run_id = str(metadata["run_id"])
        metrics[run_id] = {
            "split": split_by_run[run_id],
            "slip_run_max_m": _run_max(trace, "slip"),
            "sink_run_max_m": _run_max(trace, "sink"),
        }
    for split in ("calibration_train", "calibration_validation"):
        selected = [row for row in manifests if split_by_run[str(row["run_id"])] == split]
        normal_ids = [str(row["run_id"]) for row in selected if row["acquisition_role"] == "hard_negative"]
        slip_normal = [metrics[run]["slip_run_max_m"] for run in normal_ids if metrics[run]["slip_run_max_m"] is not None]
        sink_normal = [metrics[run]["sink_run_max_m"] for run in normal_ids if metrics[run]["sink_run_max_m"] is not None]
        slip_normal_max = max(slip_normal) if slip_normal else None
        sink_normal_max = max(sink_normal) if sink_normal else None
        for metadata in selected:
            run_id = str(metadata["run_id"])
            role = str(metadata["acquisition_role"])
            slip_max = metrics[run_id]["slip_run_max_m"]
            sink_max = metrics[run_id]["sink_run_max_m"]
            metrics[run_id].update({
                "slip_normal_envelope_max_m": slip_normal_max,
                "sink_normal_envelope_max_m": sink_normal_max,
                "slip_physical_source_valid": bool(
                    role == "slip_candidate"
                    and metadata["stable_loaded_contact_pre_fall"]
                    and slip_max is not None and slip_normal_max is not None
                    and float(slip_max) > slip_normal_max
                ),
                "sink_physical_source_valid": bool(
                    role == "sink_candidate"
                    and metadata["stable_loaded_contact_pre_fall"]
                    and not metadata["fall_occurred"]
                    and sink_max is not None and sink_normal_max is not None
                    and float(sink_max) > sink_normal_max
                ),
            })
    return metrics


def oracle_fire(trace: dict[str, np.ndarray], hazard: str, threshold: float, persistence: int) -> np.ndarray:
    if hazard == "slip":
        observable = trace["tangential_anchor_drift_m"]
        valid = trace["slip_calibration_valid"]
    else:
        observable = trace["loaded_penetration_change_m"]
        valid = trace["sink_calibration_valid"]
    return persistent_oracle(
        observable, valid, trace["contact_episode_id"], threshold, persistence
    )


def _detection_latency_ms(trace: dict[str, np.ndarray], fire: np.ndarray, hazard: str) -> int | None:
    indices = np.flatnonzero(fire)
    if not indices.size:
        return None
    first = int(indices[0])
    episode = int(trace["contact_episode_id"][first])
    valid = trace[f"{hazard}_calibration_valid"]
    starts = np.flatnonzero(valid & (trace["contact_episode_id"] == episode))
    return first - int(starts[0])


def evaluate_candidates(
    hazard: str,
    thresholds: tuple[float, ...],
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
    physical: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    train_indices = [
        index for index, row in enumerate(manifests)
        if physical[str(row["run_id"])]["split"] == "calibration_train"
    ]
    normal_indices = [i for i in train_indices if manifests[i]["acquisition_role"] == "hard_negative"]
    role = f"{hazard}_candidate"
    positive_indices = [i for i in train_indices if manifests[i]["acquisition_role"] == role]
    rows: list[dict[str, object]] = []
    normal_max = max(
        float(physical[str(manifests[i]["run_id"])][f"{hazard}_run_max_m"])
        for i in normal_indices
    )
    for threshold in thresholds:
        for persistence in PERSISTENCE_GRID_MS:
            fires = {
                i: oracle_fire(traces[i], hazard, float(threshold), int(persistence))
                for i in train_indices
            }
            normal_fp = [i for i in normal_indices if np.any(fires[i])]
            valid_positive = [
                i for i in positive_indices
                if physical[str(manifests[i]["run_id"])][f"{hazard}_physical_source_valid"]
            ]
            detected_positive = [i for i in valid_positive if np.any(fires[i])]
            latencies = [
                _detection_latency_ms(traces[i], fires[i], hazard)
                for i in detected_positive
            ]
            latencies = [value for value in latencies if value is not None]
            complete_profiles = []
            if hazard == "sink":
                profiles = sorted({str(manifests[i]["profile_name"]) for i in positive_indices})
                for profile in profiles:
                    group = [i for i in positive_indices if manifests[i]["profile_name"] == profile]
                    if (
                        {float(manifests[i]["walking_speed_mps"]) for i in group} == TARGET_SPEEDS
                        and all(i in valid_positive and i in detected_positive for i in group)
                    ):
                        complete_profiles.append(profile)
            else:
                if (
                    {float(manifests[i]["walking_speed_mps"]) for i in positive_indices} == TARGET_SPEEDS
                    and len(valid_positive) == len(positive_indices)
                    and len(detected_positive) == len(positive_indices)
                ):
                    complete_profiles = ["slip_all_speeds"]
            positive_maxima = [
                float(physical[str(manifests[i]["run_id"])][f"{hazard}_run_max_m"])
                for i in valid_positive
                if hazard == "slip" or str(manifests[i]["profile_name"]) in complete_profiles
            ]
            positive_min = min(positive_maxima) if positive_maxima else None
            air_violations = sum(int(np.count_nonzero(fires[i] & ~traces[i]["left_contact"])) for i in train_indices)
            fall_violations = sum(int(np.count_nonzero(fires[i] & ~traces[i]["pre_fall_valid"])) for i in train_indices)
            transient_violations = sum(int(np.count_nonzero(fires[i] & traces[i]["touchdown_transient"])) for i in train_indices)
            lower_margin = float(threshold) - normal_max
            upper_margin = None if positive_min is None else positive_min - float(threshold)
            passing = bool(
                not normal_fp and complete_profiles
                and air_violations == 0 and fall_violations == 0 and transient_violations == 0
            )
            rows.append({
                "hazard": hazard,
                "threshold_m": float(threshold),
                "persistence_ms": int(persistence),
                "train_normal_runs": len(normal_indices),
                "train_normal_false_positive_runs": len(normal_fp),
                "train_normal_false_positive_samples": sum(int(np.count_nonzero(fires[i])) for i in normal_indices),
                "train_physical_positive_runs": len(valid_positive),
                "train_detected_positive_runs": len(detected_positive),
                "complete_positive_profiles_all_speeds": "|".join(complete_profiles),
                "complete_positive_profile_count": len(complete_profiles),
                "air_positive_count": air_violations,
                "post_fall_positive_count": fall_violations,
                "touchdown_transient_positive_count": transient_violations,
                "normal_train_envelope_max_m": normal_max,
                "positive_train_run_min_max_m": positive_min,
                "normal_threshold_margin_m": lower_margin,
                "positive_threshold_margin_m": upper_margin,
                "minimum_threshold_margin_m": (
                    None if upper_margin is None else min(lower_margin, upper_margin)
                ),
                "mean_detection_latency_ms": (
                    None if not latencies else float(np.mean(latencies))
                ),
                "max_detection_latency_ms": None if not latencies else max(latencies),
                "candidate_pass": passing,
                "selected": False,
            })
    return rows


def select_candidate(rows: list[dict[str, object]]) -> dict[str, object] | None:
    passing = [row for row in rows if row["candidate_pass"]]
    if not passing:
        return None
    selected = max(
        passing,
        key=lambda row: (
            int(row["complete_positive_profile_count"]),
            float(row["minimum_threshold_margin_m"]),
            int(row["persistence_ms"]),
            -float(row["mean_detection_latency_ms"]),
        ),
    )
    selected["selected"] = True
    return selected


def validation_rows(
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
    physical: dict[str, dict[str, object]],
    selected_by_hazard: dict[str, dict[str, object] | None],
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    selected_outputs: dict[str, list[np.ndarray]] = {"slip": [], "sink": []}
    for hazard in ("slip", "sink"):
        selected = selected_by_hazard[hazard]
        for trace in traces:
            selected_outputs[hazard].append(
                np.zeros(len(trace["time_s"]), dtype=bool)
                if selected is None else oracle_fire(
                    trace, hazard, float(selected["threshold_m"]), int(selected["persistence_ms"])
                )
            )
        if selected is None:
            continue
        for index, metadata in enumerate(manifests):
            run_id = str(metadata["run_id"])
            role = str(metadata["acquisition_role"])
            if physical[run_id]["split"] != "calibration_validation":
                continue
            if role not in ("hard_negative", f"{hazard}_candidate"):
                continue
            trace = traces[index]
            fire = selected_outputs[hazard][index]
            expected = "negative" if role == "hard_negative" else "physical_positive_candidate"
            physical_valid = bool(
                role == "hard_negative"
                or physical[run_id][f"{hazard}_physical_source_valid"]
            )
            detected = bool(np.any(fire))
            first = next((int(value) for value in np.flatnonzero(fire)), None)
            rows.append({
                "hazard": hazard,
                "run_id": run_id,
                "condition_name": metadata["condition_name"],
                "profile_name": metadata["profile_name"],
                "walking_speed_mps": metadata["walking_speed_mps"],
                "variation_index": metadata["variation_index"],
                "split": physical[run_id]["split"],
                "expected_role": expected,
                "physical_source_valid": physical_valid,
                "observable_run_max_m": physical[run_id][f"{hazard}_run_max_m"],
                "normal_validation_envelope_max_m": physical[run_id][f"{hazard}_normal_envelope_max_m"],
                "threshold_m": selected["threshold_m"],
                "persistence_ms": selected["persistence_ms"],
                "detected": detected,
                "false_positive": bool(role == "hard_negative" and detected),
                "positive_missed": bool(role != "hard_negative" and physical_valid and not detected),
                "positive_samples": int(np.count_nonzero(fire)),
                "first_detection_sample": first,
                "first_detection_time_s": None if first is None else float(trace["time_s"][first]),
                "detection_latency_ms": _detection_latency_ms(trace, fire, hazard),
                "fall_occurred": metadata["fall_occurred"],
                "first_fall_sample": metadata["first_fall_sample"],
                "stable_loaded_contact_pre_fall": metadata["stable_loaded_contact_pre_fall"],
                "air_positive_count": int(np.count_nonzero(fire & ~trace["left_contact"])),
                "post_fall_positive_count": int(np.count_nonzero(fire & ~trace["pre_fall_valid"])),
                "touchdown_transient_positive_count": int(np.count_nonzero(fire & trace["touchdown_transient"])),
            })
    return rows, {key: np.asarray(value) for key, value in selected_outputs.items()}


def calibrated_episode_rows(
    episodes: list[dict[str, object]],
    split_by_run: dict[str, str],
    manifests: list[dict[str, object]],
    outputs: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    index_by_run = {str(row["run_id"]): index for index, row in enumerate(manifests)}
    rows = []
    for episode in episodes:
        run_id = str(episode["run_id"])
        index = index_by_run[run_id]
        start, end = int(episode["start_sample"]), int(episode["end_sample_exclusive"])
        rows.append({
            **episode,
            "calibration_split": split_by_run[run_id],
            "selected_slip_positive_samples": int(np.count_nonzero(outputs["slip"][index, start:end])),
            "selected_slip_detected": bool(np.any(outputs["slip"][index, start:end])),
            "selected_sink_positive_samples": int(np.count_nonzero(outputs["sink"][index, start:end])),
            "selected_sink_detected": bool(np.any(outputs["sink"][index, start:end])),
        })
    return rows


def fall_censor_rows(
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
    outputs: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows = []
    for index, (metadata, trace) in enumerate(zip(manifests, traces)):
        rows.append({
            "run_id": metadata["run_id"],
            "acquisition_role": metadata["acquisition_role"],
            "profile_name": metadata["profile_name"],
            "walking_speed_mps": metadata["walking_speed_mps"],
            "variation_index": metadata["variation_index"],
            "fall_occurred": metadata["fall_occurred"],
            "first_fall_sample": metadata["first_fall_sample"],
            "first_fall_time_s": metadata["first_fall_time_s"],
            "fall_reason": metadata["fall_reason"],
            "pre_fall_valid_samples": int(np.count_nonzero(trace["pre_fall_valid"])),
            "post_fall_censored_samples": int(np.count_nonzero(~trace["pre_fall_valid"])),
            "air_samples": int(np.count_nonzero(~trace["left_contact"])),
            "touchdown_transient_samples": int(np.count_nonzero(trace["touchdown_transient"])),
            "selected_slip_air_positives": int(np.count_nonzero(outputs["slip"][index] & ~trace["left_contact"])),
            "selected_slip_post_fall_positives": int(np.count_nonzero(outputs["slip"][index] & ~trace["pre_fall_valid"])),
            "selected_slip_touchdown_positives": int(np.count_nonzero(outputs["slip"][index] & trace["touchdown_transient"])),
            "selected_sink_air_positives": int(np.count_nonzero(outputs["sink"][index] & ~trace["left_contact"])),
            "selected_sink_post_fall_positives": int(np.count_nonzero(outputs["sink"][index] & ~trace["pre_fall_valid"])),
            "selected_sink_touchdown_positives": int(np.count_nonzero(outputs["sink"][index] & trace["touchdown_transient"])),
        })
    return rows


def readiness_summary(
    manifests: list[dict[str, object]],
    physical: dict[str, dict[str, object]],
    duplicate_rows: list[dict[str, object]],
    split_audit: dict[str, object],
    selected: dict[str, dict[str, object] | None],
    validation: list[dict[str, object]],
    outputs: dict[str, np.ndarray],
) -> dict[str, object]:
    duplicate_count = sum(bool(row["duplicate"]) for row in duplicate_rows)
    groups: dict[tuple[str, str, float], set[int]] = {}
    for row in manifests:
        key = (str(row["condition_name"]), str(row["profile_name"]), float(row["walking_speed_mps"]))
        groups.setdefault(key, set()).add(int(row["variation_index"]))
    diversity_ready = bool(groups) and all(values == {0, 1, 2} for values in groups.values()) and duplicate_count == 0
    validation_normal_fp = [row for row in validation if row["expected_role"] == "negative" and row["false_positive"]]
    violation_count = sum(
        int(row["air_positive_count"])
        + int(row["post_fall_positive_count"])
        + int(row["touchdown_transient_positive_count"])
        for row in validation
    )
    slip_positive = [row for row in validation if row["hazard"] == "slip" and row["expected_role"] != "negative"]
    slip_speeds = {
        float(row["walking_speed_mps"]) for row in slip_positive
        if row["physical_source_valid"] and row["detected"]
    }
    missing_slip_speeds = sorted(TARGET_SPEEDS - slip_speeds)
    sink_positive = [row for row in validation if row["hazard"] == "sink" and row["expected_role"] != "negative"]
    sink_profiles_all_speeds = []
    for profile in sorted({str(row["profile_name"]) for row in sink_positive}):
        group = [row for row in sink_positive if row["profile_name"] == profile]
        if (
            {float(row["walking_speed_mps"]) for row in group} == TARGET_SPEEDS
            and all(
                row["physical_source_valid"] and row["detected"]
                and row["stable_loaded_contact_pre_fall"] and not row["fall_occurred"]
                for row in group
            )
        ):
            sink_profiles_all_speeds.append(profile)
    split_ready = int(split_audit["split_leakage_count"]) == 0
    slip_ready = bool(
        selected["slip"] is not None and not validation_normal_fp
        and slip_speeds == TARGET_SPEEDS and violation_count == 0
    )
    sink_ready = bool(
        selected["sink"] is not None and not validation_normal_fp
        and sink_profiles_all_speeds and violation_count == 0
    )
    gates = {
        "WALKING_GT_REPLICATE_DIVERSITY_READY": diversity_ready,
        "WALKING_SLIP_ORACLE_CALIBRATION_READY": slip_ready,
        "WALKING_SINK_ORACLE_CALIBRATION_READY": sink_ready,
        "WALKING_ORACLE_SPLIT_INTEGRITY_READY": split_ready,
        "WALKING_ORACLE_ROBUSTNESS_READY": bool(
            diversity_ready and slip_ready and sink_ready and split_ready and violation_count == 0
        ),
        "WALKING_BOUNDED_RETRAINING_READY": bool(
            diversity_ready and slip_ready and sink_ready and split_ready and violation_count == 0
        ),
    }
    failure_reasons = []
    if missing_slip_speeds:
        failure_reasons.append(
            "held-out Slip validation had no detection at speeds "
            + ",".join(f"{speed:.2f}" for speed in missing_slip_speeds)
            + " m/s"
        )
    if not sink_profiles_all_speeds:
        failure_reasons.append(
            "no Sink profile supplied stable no-fall validation detection at every speed"
        )
    if duplicate_count:
        failure_reasons.append(f"{duplicate_count} duplicate replicate pairs")
    if not split_ready:
        failure_reasons.append(f"{split_audit['split_leakage_count']} split leaks")
    if violation_count:
        failure_reasons.append(f"{violation_count} invalid-sample label violations")
    envelopes = {}
    for hazard in ("slip", "sink"):
        for split in ("calibration_train", "calibration_validation"):
            normal = [
                physical[str(row["run_id"])][f"{hazard}_run_max_m"] for row in manifests
                if physical[str(row["run_id"])]["split"] == split
                and row["acquisition_role"] == "hard_negative"
            ]
            positive = [
                physical[str(row["run_id"])][f"{hazard}_run_max_m"] for row in manifests
                if physical[str(row["run_id"])]["split"] == split
                and row["acquisition_role"] == f"{hazard}_candidate"
                and physical[str(row["run_id"])][f"{hazard}_physical_source_valid"]
            ]
            normal_values = [float(value) for value in normal if value is not None]
            positive_values = [float(value) for value in positive if value is not None]
            envelopes[f"{hazard}_{split}"] = {
                "normal_run_max_m": max(normal_values) if normal_values else None,
                "positive_run_min_max_m": min(positive_values) if positive_values else None,
            }
    return {
        "runs": len(manifests),
        "run_counts": {
            role: sum(row["acquisition_role"] == role for row in manifests)
            for role in ("hard_negative", "slip_candidate", "sink_candidate")
        },
        "variation_count": len(VARIATIONS),
        "duplicate_pair_count": duplicate_count,
        "split_integrity": split_audit,
        "fall_runs": sum(bool(row["fall_occurred"]) for row in manifests),
        "selected_candidates": {
            hazard: None if value is None else {
                "threshold_m": value["threshold_m"],
                "persistence_ms": value["persistence_ms"],
                "production_frozen": False,
            }
            for hazard, value in selected.items()
        },
        "envelopes": envelopes,
        "validation_normal_false_positive_runs": len(validation_normal_fp),
        "validation_slip_detected_speeds_mps": sorted(slip_speeds),
        "validation_slip_missing_speeds_mps": missing_slip_speeds,
        "validation_sink_profiles_all_speeds": sink_profiles_all_speeds,
        "validation_label_violation_count": violation_count,
        "air_positive_count": sum(int(row["air_positive_count"]) for row in validation),
        "post_fall_positive_count": sum(int(row["post_fall_positive_count"]) for row in validation),
        "touchdown_transient_positive_count": sum(
            int(row["touchdown_transient_positive_count"]) for row in validation
        ),
        "thresholds_frozen": False,
        "models_retrained": False,
        "failure_reasons": failure_reasons,
        "gates": gates,
        **gates,
    }


def plots(
    output: Path,
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
    physical: dict[str, dict[str, object]],
    selected: dict[str, dict[str, object] | None],
    outputs: dict[str, np.ndarray],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = output / "plots"
    directory.mkdir()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, hazard in zip(axes, ("slip", "sink")):
        for split, marker in (("calibration_train", "o"), ("calibration_validation", "x")):
            normal = [
                physical[str(row["run_id"])][f"{hazard}_run_max_m"] for row in manifests
                if physical[str(row["run_id"])]["split"] == split and row["acquisition_role"] == "hard_negative"
            ]
            positive = [
                physical[str(row["run_id"])][f"{hazard}_run_max_m"] for row in manifests
                if physical[str(row["run_id"])]["split"] == split and row["acquisition_role"] == f"{hazard}_candidate"
            ]
            axis.scatter(np.zeros(len(normal)), np.asarray(normal) * 1000, marker=marker, label=f"{split} normal")
            axis.scatter(np.ones(len(positive)), np.asarray(positive) * 1000, marker=marker, label=f"{split} candidate")
        if selected[hazard] is not None:
            axis.axhline(float(selected[hazard]["threshold_m"]) * 1000, color="black", linestyle="--")
        axis.set_xticks([0, 1], ["normal", "candidate"])
        axis.set_ylabel(f"{hazard} primary run max [mm]")
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(directory / "train_validation_envelopes.png", dpi=150)
    plt.close(figure)

    group = [i for i, row in enumerate(manifests) if row["condition_name"] == "concrete_native" and np.isclose(row["walking_speed_mps"], 0.10)]
    figure, axis = plt.subplots(figsize=(10, 4))
    for index in group:
        axis.plot(traces[index]["time_s"], traces[index]["fusion10"][:, :4].sum(1), label=str(manifests[index]["run_id"]))
    axis.set_xlabel("simulation time [s]")
    axis.set_ylabel("left FSR sum [N]")
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "replicate_diversity_concrete_0p10.png", dpi=150)
    plt.close(figure)

    for hazard in ("slip", "sink"):
        index = next(
            i for i, row in enumerate(manifests)
            if row["variation_index"] == 2 and row["acquisition_role"] == f"{hazard}_candidate"
        )
        trace = traces[index]
        observable = trace["tangential_anchor_drift_m"] if hazard == "slip" else trace["loaded_penetration_change_m"]
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.plot(trace["time_s"], observable * 1000, label="primary observable")
        axis.step(trace["time_s"], outputs[hazard][index] * np.nanmax(observable * 1000), where="post", label="selected candidate firing")
        if selected[hazard] is not None:
            axis.axhline(float(selected[hazard]["threshold_m"]) * 1000, color="black", linestyle="--")
        axis.set_xlabel("simulation time [s]")
        axis.set_ylabel("mm")
        axis.legend()
        figure.tight_layout()
        figure.savefig(directory / f"{hazard}_validation_timeline.png", dpi=150)
        plt.close(figure)


def audit_markdown(summary: dict[str, object]) -> str:
    slip = summary["selected_candidates"]["slip"]
    sink = summary["selected_candidates"]["sink"]
    return f"""# Walking Slip/Sink physical oracle robustness + calibration v1

Fifty-four 3-second, 1-kHz train-only runs used three deterministic but
physically distinct initial policy conditions.  No test/final data, model
training, production threshold change, INT8/E84 export, or System-v1 change
was performed.

Replicate variation combined gait phase fractions 0, 1/3 and 2/3 with command
onset delays 0, 20 and 40 ms.  This adds no sensor or label noise.  Pairwise
full endpoint hashes found {summary['duplicate_pair_count']} duplicate traces.

The selected Slip audit candidate is {json.dumps(slip, sort_keys=True)}; the
selected Sink audit candidate is {json.dumps(sink, sort_keys=True)}.  These are
**not frozen production thresholds**.

Train/validation envelopes:

```json
{json.dumps(summary['envelopes'], indent=2)}
```

Validation normal false-positive runs: {summary['validation_normal_false_positive_runs']}.
Slip detected speeds: {summary['validation_slip_detected_speeds_mps']}.
Slip missing speeds: {summary['validation_slip_missing_speeds_mps']}.
Sink all-speed profiles: {summary['validation_sink_profiles_all_speeds']}.
AIR/post-fall/touchdown-transient violations: {summary['validation_label_violation_count']}.
Split leakage: {summary['split_integrity']['split_leakage_count']}.
Failure reasons: {summary['failure_reasons']}.

## Readiness gates

""" + "\n".join(
        f"- {name}={str(value).lower()}" for name, value in summary["gates"].items()
    ) + """

Threshold and persistence values remain bounded calibration candidates only.
"""


def save_packed(
    path: Path,
    traces: list[dict[str, np.ndarray]],
    manifests: list[dict[str, object]],
    outputs: dict[str, np.ndarray] | None = None,
) -> None:
    packed = {key: np.asarray([trace[key] for trace in traces]) for key in traces[0]}
    packed.update({
        "run_id": np.asarray([row["run_id"] for row in manifests]),
        "terrain_name": np.asarray([row["terrain_name"] for row in manifests]),
        "profile_name": np.asarray([row["profile_name"] for row in manifests]),
        "acquisition_role": np.asarray([row["acquisition_role"] for row in manifests]),
        "walking_speed_mps": np.asarray([row["walking_speed_mps"] for row in manifests], float),
        "variation_index": np.asarray([row["variation_index"] for row in manifests], np.int8),
        "variation_seed": np.asarray([row["variation_seed"] for row in manifests], np.int64),
        "initial_locomotion_phase_fraction": np.asarray([row["initial_locomotion_phase_fraction"] for row in manifests], float),
        "command_onset_delay_s": np.asarray([row["command_onset_delay_s"] for row in manifests], float),
    })
    if outputs is not None:
        packed["slip_oracle_calibration_candidate"] = outputs["slip"]
        packed["sink_oracle_calibration_candidate"] = outputs["sink"]
    np.savez_compressed(path, **packed)


def main() -> None:
    args = parse_args()
    conditions = acquisition_conditions(args)
    planned = protocol(args, conditions)
    if not args.execute:
        print(json.dumps(planned, indent=2))
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
    planned["controller"].update({"policy_path": str(policy_path), "policy_sha256": policy_hash})
    traces: list[dict[str, np.ndarray]] = []
    manifests: list[dict[str, object]] = []
    episodes: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, condition in enumerate(conditions):
        trace, metadata, current = collect_run(condition, policy_path, args.duration_s)
        traces.append(trace)
        manifests.append(metadata)
        episodes.extend(current)
        elapsed = time.perf_counter() - started
        eta = elapsed / (index + 1) * (len(conditions) - index - 1)
        print(
            f"[{index + 1}/{len(conditions)}] {condition.run_id} "
            f"phase={condition.variation.initial_locomotion_phase_fraction:.3f} "
            f"onset={condition.variation.command_onset_delay_s * 1000:.0f}ms "
            f"fall={metadata['fall_occurred']} elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )
    duplicate_rows = duplicate_trace_audit(manifests, traces)
    (output / "protocol.json").write_text(json.dumps(planned, indent=2) + "\n", encoding="utf-8")
    if args.smoke:
        save_packed(output / "traces.npz", traces, manifests)
        (output / "manifest.json").write_text(json.dumps(manifests, indent=2) + "\n", encoding="utf-8")
        write_csv(output / "duplicate_trace_audit.csv", duplicate_rows)
        smoke = {
            "smoke_runs": len(manifests),
            "duplicate_pair_count": sum(bool(row["duplicate"]) for row in duplicate_rows),
            "all_finite": all(bool(row["finite_state"]) for row in manifests),
        }
        (output / "summary.json").write_text(json.dumps(smoke, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(smoke, indent=2))
        return

    split_rows = assign_calibration_splits(manifests)
    split_by_run = {str(row["run_id"]): str(row["split"]) for row in split_rows}
    for metadata in manifests:
        metadata["calibration_split"] = split_by_run[str(metadata["run_id"])]
    for episode in episodes:
        episode["calibration_split"] = split_by_run[str(episode["run_id"])]
    split_audit = split_integrity(split_rows, episodes)
    physical = annotate_physical_sources(manifests, traces, split_rows)
    for metadata in manifests:
        metadata.update(physical[str(metadata["run_id"])] )
    slip_candidates = evaluate_candidates(
        "slip", SLIP_THRESHOLD_GRID_M, manifests, traces, physical
    )
    sink_candidates = evaluate_candidates(
        "sink", SINK_THRESHOLD_GRID_M, manifests, traces, physical
    )
    selected = {
        "slip": select_candidate(slip_candidates),
        "sink": select_candidate(sink_candidates),
    }
    validation, selected_outputs = validation_rows(
        manifests, traces, physical, selected
    )
    calibrated_episodes = calibrated_episode_rows(
        episodes, split_by_run, manifests, selected_outputs
    )
    censor_rows = fall_censor_rows(manifests, traces, selected_outputs)
    summary = readiness_summary(
        manifests, physical, duplicate_rows, split_audit,
        selected, validation, selected_outputs,
    )
    summary["wall_time_s"] = time.perf_counter() - started
    save_packed(output / "traces.npz", traces, manifests, selected_outputs)
    (output / "manifest.json").write_text(json.dumps(manifests, indent=2) + "\n", encoding="utf-8")
    (output / "split_manifest.json").write_text(json.dumps({
        "policy": "whole run/contact episode; v00/v01 train, v02 validation",
        "test_or_final_used": False,
        "integrity": split_audit,
        "runs": split_rows,
    }, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "oracle_threshold_candidates.csv", slip_candidates + sink_candidates)
    write_csv(output / "validation_metrics.csv", validation)
    write_csv(output / "episode_metrics.csv", calibrated_episodes)
    write_csv(output / "duplicate_trace_audit.csv", duplicate_rows)
    write_csv(output / "fall_censor_audit.csv", censor_rows)
    plots(output, manifests, traces, physical, selected, selected_outputs)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "audit.md").write_text(audit_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
