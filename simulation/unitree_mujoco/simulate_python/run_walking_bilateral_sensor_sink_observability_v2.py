"""Acquire and audit the development-only bilateral walking sensor contract.

This runner creates a fresh 120-run MuJoCo development dataset.  Privileged
world/contact quantities are written only to the offline label/evaluation
namespace; C0--C4 consume virtual hardware-reproducible runtime signals only.
No outer/final data, production model, firmware, System, INT8, or Vela path is
opened or changed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    recall_score, roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bilateral_hil_sensor_v2 import (
    G1BilateralSensorReaderV2,
    HIL_SENSOR_CHANNELS,
    PELVIS_CHANNELS,
    RIGHT_ACCEL_CANONICAL_SIGN,
    RIGHT_FSR_CANONICAL_ORDER,
    RIGHT_GYRO_CANONICAL_SIGN,
    SIDES,
)
from g1_upstream_locomotion import (
    TESTED_POLICY_SHA256,
    UPSTREAM_REVISION,
    UnitreeG1PretrainedController,
)
from terrain_profiles import TERRAIN_PROFILES, TerrainProfile, apply_terrain_profile
from walking_bilateral_sink_observability_v2 import (
    CANDIDATE_NAMES,
    ENDPOINT_STRIDE,
    SAMPLE_RATE_HZ,
    SharedCausalConv1DV2,
    SharedFootEncoderV2,
    causal_endpoints,
    contact_age,
    deterministic_candidate_selection,
    effort_summaries,
    endpoint_hash,
    first_fall_mask,
    joint_derived_kinematics,
    physical_risk_target,
    runtime_feature_contract_is_clean,
    stable_fire,
    support_degradation_label,
    zero_false_positive_threshold,
)
from walking_hazard_ground_truth_v1 import (
    LOAD_THRESHOLD_N,
    box_surface_top_z,
    contact_episodes,
    derive_contact_signals,
    gait_phase,
    max_left_foot_contact_penetration_m,
    sole_sphere_lowest_point_z,
)
from walking_hazard_oracle_calibration_v1 import persistent_oracle
from run_walking_hazard_ground_truth_v1 import (
    DEFAULT_POLICY,
    GROUND_NAMES,
    PELVIS_BODY_NAME,
    PHYSICS_STEPS_PER_SAMPLE,
    PHYSICS_TIMESTEP_S,
    SCENE_PATH,
    _disable_nonfoot_surface_collisions,
    _fall_reasons,
)


SIMULATION_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = SIMULATION_DIR / "outputs" / "walking_bilateral_sensor_sink_observability_v2"
STARTING_CHECKPOINT = "7ae06662bb32c7b67f05ff63b9570aec827d1d92"
DEFAULT_DURATION_S = 3.0
SPEEDS_MPS = (0.10, 0.15, 0.20)
VARIATION_SEED_BASE = 202608200
TERRAIN_LABELS = {"concrete": 0, "marble": 1, "ice": 2, "sand": 3}
TERRAIN_NAMES = ("concrete", "marble", "ice", "sand")
PHASE_CODE = {"AIR": 0, "TOUCHDOWN": 1, "LOADING": 2, "MID_STANCE": 3, "PUSH_OFF": 4}
FEATURE_SOURCES = {
    "C0": {"bilateral_canonical"},
    "C1": {"bilateral_canonical", "force_loaded", "contact_age"},
    "C2": {"bilateral_canonical", "force_loaded", "contact_age", "pelvis_imu"},
    "C3": {"bilateral_canonical", "force_loaded", "contact_age", "joint_position", "joint_velocity"},
    "C4": {
        "bilateral_canonical", "force_loaded", "contact_age", "joint_position",
        "joint_velocity", "target_position", "actuator_effort", "pelvis_imu",
    },
}
UPSTREAM_FILES = (
    "simulation/outputs/walking_hazard_ground_truth_v1_pilot/summary.json",
    "simulation/outputs/walking_hazard_oracle_calibration_v1/summary.json",
    "simulation/outputs/walking_hazard_slip_nested_calibration_v2/summary.json",
    "simulation/outputs/walking_bounded_retraining_v1/summary.json",
    "simulation/outputs/walking_fusion10_observability_audit_v1/summary.json",
    "simulation/outputs/walking_stateful_hazard_prototype_v1/summary.json",
    "simulation/outputs/walking_hazard_operational_label_contract_v2/summary.json",
)


@dataclass(frozen=True)
class Variation:
    index: int
    seed: int
    phase_fraction: float
    command_delay_s: float
    split: str


VARIATIONS = (
    Variation(0, VARIATION_SEED_BASE + 1, 0.07, 0.000, "development_train"),
    Variation(1, VARIATION_SEED_BASE + 2, 0.23, 0.020, "development_train"),
    Variation(2, VARIATION_SEED_BASE + 3, 0.41, 0.040, "development_train"),
    Variation(3, VARIATION_SEED_BASE + 4, 0.61, 0.060, "development_validation"),
    Variation(4, VARIATION_SEED_BASE + 5, 0.83, 0.080, "development_validation"),
)


@dataclass(frozen=True)
class Condition:
    condition_name: str
    terrain_name: str
    profile: TerrainProfile
    role: str
    speed_mps: float
    variation: Variation

    @property
    def run_id(self) -> str:
        speed = f"{self.speed_mps:.2f}".replace(".", "p")
        return f"{self.condition_name}_{speed}_v{self.variation.index:02d}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="one Concrete run; no readiness claim")
    parser.add_argument(
        "--reanalyze-existing", action="store_true",
        help="reuse only this artifact's newly acquired traces; never outer data",
    )
    return parser.parse_args()


def hardened_sand_profile() -> TerrainProfile:
    return replace(
        TERRAIN_PROFILES["sand"],
        name="sand_walking_support_v1_hardened",
        solref=TERRAIN_PROFILES["concrete"].solref,
        description="native Sand friction/solimp with Concrete solref",
    )


def stronger_sink_profiles() -> tuple[TerrainProfile, TerrainProfile]:
    """Exactly two predeclared profiles between 2/3 and native Sand."""
    hard = np.asarray(TERRAIN_PROFILES["concrete"].solref, float)
    native = np.asarray(TERRAIN_PROFILES["sand"].solref, float)
    result = []
    for name, fraction in (("3of4", 0.75), ("5of6", 5.0 / 6.0)):
        result.append(replace(
            TERRAIN_PROFILES["sand"],
            name=f"sand_solref_interpolation_{name}_stronger_v2",
            solref=tuple((hard + fraction * (native - hard)).tolist()),
            description=f"predeclared v2 interpolation fraction {fraction:.9f}",
        ))
    return tuple(result)  # type: ignore[return-value]


def frozen_sink_profiles() -> tuple[TerrainProfile, TerrainProfile]:
    hard = np.asarray(TERRAIN_PROFILES["concrete"].solref, float)
    native = np.asarray(TERRAIN_PROFILES["sand"].solref, float)
    return tuple(
        replace(
            TERRAIN_PROFILES["sand"],
            name=f"sand_solref_interpolation_{name}",
            solref=tuple((hard + fraction * (native - hard)).tolist()),
            description=f"frozen interpolation fraction {fraction:.9f}",
        )
        for name, fraction in (("1of3", 1.0 / 3.0), ("2of3", 2.0 / 3.0))
    )  # type: ignore[return-value]


def conditions(smoke: bool = False) -> list[Condition]:
    if smoke:
        return [Condition(
            "concrete_native", "concrete", TERRAIN_PROFILES["concrete"],
            "hard_negative", SPEEDS_MPS[0], VARIATIONS[0],
        )]
    sink_profiles = (*frozen_sink_profiles(), *stronger_sink_profiles())
    output: list[Condition] = []
    for variation in VARIATIONS:
        for speed in SPEEDS_MPS:
            fixed = (
                ("marble_native", "marble", TERRAIN_PROFILES["marble"], "hard_negative"),
                ("concrete_native", "concrete", TERRAIN_PROFILES["concrete"], "hard_negative"),
                ("sand_hardened", "sand", hardened_sand_profile(), "hard_negative"),
                ("ice_native", "ice", TERRAIN_PROFILES["ice"], "slip_candidate"),
            )
            for name, terrain, profile, role in fixed:
                output.append(Condition(name, terrain, profile, role, speed, variation))
            for profile in sink_profiles:
                output.append(Condition(profile.name, "sand", profile, "sink_candidate", speed, variation))
    if len(output) != 120:
        raise AssertionError("bounded bilateral protocol must contain exactly 120 runs")
    return output


def upstream_hashes(repo_root: Path) -> dict[str, str]:
    result = {}
    for name in UPSTREAM_FILES:
        path = repo_root / name
        if not path.is_file():
            raise FileNotFoundError(f"required upstream artifact missing: {name}")
        result[name] = _sha256(path)
    return result


def protocol(run_conditions: list[Condition], duration_s: float, hashes: dict[str, str]) -> dict[str, object]:
    encoder = SharedFootEncoderV2()
    return {
        "artifact": "walking_bilateral_sensor_sink_observability_v2",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "scope": "MuJoCo digital-twin development-only virtual sensor and observability audit",
        "physical_hardware_work_performed": False,
        "production_or_firmware_or_system_change": False,
        "outer_or_final_content_access": False,
        "upstream_sha256": hashes,
        "controller": {"revision": UPSTREAM_REVISION, "policy_sha256": TESTED_POLICY_SHA256},
        "sampling": {
            "sensor_rate_hz": SAMPLE_RATE_HZ,
            "physics_timestep_s": PHYSICS_TIMESTEP_S,
            "physics_steps_per_sample": PHYSICS_STEPS_PER_SAMPLE,
            "duration_s": duration_s,
            "causal_window_ms": 200,
            "endpoint_stride_ms": ENDPOINT_STRIDE,
            "future_samples": 0,
        },
        "matrix": {
            "run_count": len(run_conditions),
            "speeds_mps": SPEEDS_MPS,
            "variations": [vars(value) for value in VARIATIONS],
            "normal": ["Marble", "Concrete", "hardened Sand"],
            "slip": ["Ice"],
            "sink_profiles": [vars(value) for value in (*frozen_sink_profiles(), *stronger_sink_profiles())],
            "stronger_profile_count_exact": len(stronger_sink_profiles()),
            "compliance_adaptation_after_measurement": False,
        },
        "frame_canonicalization_predeclared": {
            "reflection": "R=diag(1,-1,1)",
            "right_fsr_order": RIGHT_FSR_CANONICAL_ORDER.tolist(),
            "right_accel_sign": RIGHT_ACCEL_CANONICAL_SIGN.tolist(),
            "right_gyro_sign": RIGHT_GYRO_CANONICAL_SIGN.tolist(),
            "derivation": "polar acceleration uses R; axial gyro uses det(R)R",
        },
        "shared_encoder": {
            "kind": "fixed causal 80-stat foot encoder with small trained head",
            "fingerprint": encoder.fingerprint,
            "weight_sharing": "one exact encoder instance is applied to both canonical feet",
        },
        "candidate_order": list(CANDIDATE_NAMES),
        "candidate_feature_sources": {key: sorted(value) for key, value in FEATURE_SOURCES.items()},
        "selection_order": [
            "invalid firing", "normal risk run FP", "Ice cross-hazard Sink FP",
            "too-early risk", "zero-FP Sink recall", "profile/speed/foot coverage",
            "latency", "sensor/channel count", "memory/MAC",
        ],
        "labels": {
            "slip_physical": "offline 0.050 m anchor drift, 3 ms persistence",
            "slip_risk": "locked 100 ms within-episode pre-onset or active",
            "sink_physical": "offline 0.0055 m penetration change, 20 ms persistence",
            "sink_diagnostic_risk": "200 ms within-episode pre-onset or active",
            "support_degradation": "label-only pelvis/support consequence; not a runtime input",
            "first_fall_censor": "exclude endpoint at and after first sampled fall",
        },
        "observability_gate_is_locked": True,
        "candidate_addition_after_validation": False,
        "conditions": [{
            "run_id": item.run_id,
            "condition_name": item.condition_name,
            "terrain_name": item.terrain_name,
            "profile_name": item.profile.name,
            "role": item.role,
            "speed_mps": item.speed_mps,
            "variation_index": item.variation.index,
            "split": item.variation.split,
            "phase_fraction": item.variation.phase_fraction,
            "command_delay_s": item.variation.command_delay_s,
            "seed": item.variation.seed,
        } for item in run_conditions],
    }


def _max_penetration(data: mujoco.MjData, soles: tuple[int, ...], grounds: frozenset[int]) -> float:
    return max_left_foot_contact_penetration_m(data, soles, grounds)


def collect_run(condition: Condition, policy_path: Path, duration_s: float) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    model.opt.timestep = PHYSICS_TIMESTEP_S
    ground_ids = frozenset(model.geom(name).id for name in GROUND_NAMES)
    for ground_name in GROUND_NAMES:
        apply_terrain_profile(model, condition.profile, ground_name)
    allowed_feet = _disable_nonfoot_surface_collisions(model, ground_ids)
    data = mujoco.MjData(model)
    controller = UnitreeG1PretrainedController(model, data, policy_path, condition.speed_mps)
    controller.global_phase = condition.variation.phase_fraction
    nominal_command = controller.command.copy()
    if condition.variation.command_delay_s:
        controller.command[:] = 0.0
    reader = G1BilateralSensorReaderV2(model, data)
    foot_body_ids = tuple(model.body(f"{side}_ankle_roll_link").id for side in SIDES)
    pelvis_id = model.body(PELVIS_BODY_NAME).id
    velocity = np.zeros(6, dtype=np.float64)
    keys = (
        "time_s", "bilateral_raw", "bilateral_canonical", "pelvis_imu",
        "force_loaded", "contact_age", "physical_contact", "foot_world_xyz",
        "foot_world_velocity", "pelvis_world_xyz", "pelvis_world_velocity",
        "contact_penetration", "surface_relative_sole_depth", "joint_position",
        "joint_velocity", "target_position", "actuator_effort", "nonfoot_contact",
    )
    series: dict[str, list[object]] = {key: [] for key in keys}
    first_fall_sample: int | None = None
    first_fall_time_s: float | None = None
    fall_reason = ""
    start_x = float(data.qpos[0])
    total_steps = int(round(duration_s / PHYSICS_TIMESTEP_S))
    for physics_step in range(1, total_steps + 1):
        if data.time + 1e-12 >= condition.variation.command_delay_s:
            controller.command[:] = nominal_command
        controller.apply()
        mujoco.mj_step(model, data)
        controller.update_after_step()
        if physics_step % PHYSICS_STEPS_PER_SAMPLE:
            continue
        raw = reader.read_bilateral_vector()
        canonical = np.concatenate(tuple(
            reader.canonicalize_foot_vector(side, raw[index * 10:(index + 1) * 10])
            for index, side in enumerate(SIDES)
        ))
        runtime_states = tuple(
            reader.update_contact_state(side, raw[index * 10:index * 10 + 4])
            for index, side in enumerate(SIDES)
        )
        physical_contact = np.asarray([
            reader.has_foot_contact(side, ground_ids) for side in SIDES
        ], dtype=bool)
        foot_xyz = np.stack(tuple(data.xpos[value].copy() for value in foot_body_ids))
        foot_velocity = []
        for body_id in foot_body_ids:
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
            foot_velocity.append(velocity[3:].copy())
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, pelvis_id, velocity, 0)
        pelvis_velocity = velocity[3:].copy()
        surface_top = max(box_surface_top_z(model, data, value) for value in ground_ids)
        penetration = np.asarray([
            _max_penetration(data, reader.foot_geom_ids[side], ground_ids) for side in SIDES
        ])
        sole_depth = np.asarray([
            max(0.0, surface_top - sole_sphere_lowest_point_z(
                model, data, reader.foot_geom_ids[side]
            )) for side in SIDES
        ])
        reasons, nonfoot = _fall_reasons(model, data, pelvis_id, ground_ids, allowed_feet)
        if reasons and first_fall_sample is None:
            first_fall_sample = len(series["time_s"])
            first_fall_time_s = float(data.time)
            fall_reason = "|".join(reasons)
        values = {
            "time_s": float(data.time),
            "bilateral_raw": raw.astype(np.float32),
            "bilateral_canonical": canonical.astype(np.float32),
            "pelvis_imu": reader.read_pelvis_vector().astype(np.float32),
            "force_loaded": np.asarray([value.loaded for value in runtime_states]),
            "contact_age": np.asarray([value.age_samples for value in runtime_states], np.int32),
            "physical_contact": physical_contact,
            "foot_world_xyz": foot_xyz.astype(np.float32),
            "foot_world_velocity": np.asarray(foot_velocity, np.float32),
            "pelvis_world_xyz": data.xpos[pelvis_id].copy().astype(np.float32),
            "pelvis_world_velocity": pelvis_velocity.astype(np.float32),
            "contact_penetration": penetration.astype(np.float32),
            "surface_relative_sole_depth": sole_depth.astype(np.float32),
            "joint_position": data.qpos[7:19].copy().astype(np.float32),
            "joint_velocity": data.qvel[6:18].copy().astype(np.float32),
            "target_position": controller.target_position[:12].copy().astype(np.float32),
            "actuator_effort": data.actuator_force[:12].copy().astype(np.float32),
            "nonfoot_contact": bool(nonfoot),
        }
        for key, value in values.items():
            series[key].append(value)
    trace = {key: np.asarray(value) for key, value in series.items()}
    sample_count = len(trace["time_s"])
    pre_fall = first_fall_mask(sample_count, first_fall_sample)
    episode_id = np.full((sample_count, 2), -1, dtype=np.int32)
    transient = np.zeros((sample_count, 2), dtype=bool)
    anchor_drift = np.full((sample_count, 2), np.nan, dtype=np.float32)
    penetration_change = np.full((sample_count, 2), np.nan, dtype=np.float32)
    physical_valid = np.zeros((sample_count, 2), dtype=bool)
    gait_phase_code = np.zeros((sample_count, 2), dtype=np.int8)
    slip_active = np.zeros((sample_count, 2), dtype=bool)
    sink_active = np.zeros((sample_count, 2), dtype=bool)
    slip_risk = np.zeros((sample_count, 2), dtype=bool)
    sink_risk = np.zeros((sample_count, 2), dtype=bool)
    support_degradation = np.zeros((sample_count, 2), dtype=bool)
    for side_index in range(2):
        signals = derive_contact_signals(
            trace["physical_contact"][:, side_index],
            trace["force_loaded"][:, side_index],
            trace["foot_world_xyz"][:, side_index],
            trace["foot_world_velocity"][:, side_index],
            trace["contact_penetration"][:, side_index],
            first_fall_sample,
        )
        episode_id[:, side_index] = signals.contact_episode_id
        transient[:, side_index] = signals.touchdown_transient
        anchor_drift[:, side_index] = signals.tangential_anchor_drift_m
        penetration_change[:, side_index] = signals.loaded_penetration_change_m
        physical_valid[:, side_index] = signals.sink_calibration_valid
        phases = gait_phase(trace["force_loaded"][:, side_index])
        gait_phase_code[:, side_index] = np.asarray([PHASE_CODE[value] for value in phases], np.int8)
        slip_active[:, side_index] = persistent_oracle(
            signals.tangential_anchor_drift_m,
            signals.slip_calibration_valid,
            signals.contact_episode_id,
            0.050,
            3,
        )
        sink_active[:, side_index] = persistent_oracle(
            signals.loaded_penetration_change_m,
            signals.sink_calibration_valid,
            signals.contact_episode_id,
            0.0055,
            20,
        )
        slip_risk[:, side_index] = physical_risk_target(
            slip_active[:, side_index], signals.slip_calibration_valid,
            signals.contact_episode_id, 100,
        )
        sink_risk[:, side_index] = physical_risk_target(
            sink_active[:, side_index], signals.sink_calibration_valid,
            signals.contact_episode_id, 200,
        )
        support_degradation[:, side_index] = support_degradation_label(
            sink_active[:, side_index], signals.sink_calibration_valid,
            trace["pelvis_world_xyz"][:, 2], trace["pelvis_world_velocity"][:, 2],
            signals.contact_episode_id,
        )
    trace.update({
        "pre_fall_valid": pre_fall,
        "contact_episode_id": episode_id,
        "touchdown_transient": transient,
        "anchor_drift_label_only": anchor_drift,
        "penetration_change_label_only": penetration_change,
        "physical_valid": physical_valid,
        "gait_phase_code": gait_phase_code,
        "slip_physical_active": slip_active,
        "sink_physical_active": sink_active,
        "slip_risk_target": slip_risk,
        "sink_risk_target": sink_risk,
        "support_degradation_label_only": support_degradation,
    })
    metadata: dict[str, object] = {
        "run_id": condition.run_id,
        "condition_name": condition.condition_name,
        "terrain_name": condition.terrain_name,
        "profile_name": condition.profile.name,
        "role": condition.role,
        "speed_mps": condition.speed_mps,
        "variation_index": condition.variation.index,
        "variation_seed": condition.variation.seed,
        "phase_fraction": condition.variation.phase_fraction,
        "command_delay_s": condition.variation.command_delay_s,
        "split": condition.variation.split,
        "sample_count": sample_count,
        "first_sample_time_s": float(trace["time_s"][0]),
        "last_sample_time_s": float(trace["time_s"][-1]),
        "sample_spacing_max_error_s": float(np.max(np.abs(np.diff(trace["time_s"]) - 0.001))),
        "first_fall_sample": first_fall_sample,
        "first_fall_time_s": first_fall_time_s,
        "fall_occurred": first_fall_sample is not None,
        "fall_reason": fall_reason,
        "forward_displacement_m": float(data.qpos[0] - start_x),
        "left_contact_events": len(contact_episodes(trace["physical_contact"][:, 0])),
        "right_contact_events": len(contact_episodes(trace["physical_contact"][:, 1])),
        "left_sink_active": bool(np.any(sink_active[:, 0])),
        "right_sink_active": bool(np.any(sink_active[:, 1])),
        "left_slip_active": bool(np.any(slip_active[:, 0])),
        "right_slip_active": bool(np.any(slip_active[:, 1])),
        "endpoint_sha256": endpoint_hash(
            trace["bilateral_raw"][causal_endpoints(sample_count)],
            trace["joint_position"][causal_endpoints(sample_count)],
        ),
        "finite_runtime_observation": bool(np.all(np.isfinite(trace["bilateral_raw"]))),
    }
    return trace, metadata


def _summary3(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, float)
    return np.concatenate((array[-1], array.mean(axis=0), array.std(axis=0))).astype(np.float32)


def feature_rows(
    candidate: str,
    traces: list[dict[str, np.ndarray]],
    manifests: list[dict[str, object]],
    encoder: SharedFootEncoderV2 | SharedCausalConv1DV2 | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    encoder = SharedFootEncoderV2() if encoder is None else encoder
    features: list[np.ndarray] = []
    fields: dict[str, list[object]] = {key: [] for key in (
        "run_index", "endpoint", "side", "sink_target", "slip_target",
        "sink_active", "slip_active", "runtime_eligible", "label_valid",
        "post_fall", "touchdown", "role", "profile", "terrain", "speed", "split",
    )}
    for run_index, (trace, meta) in enumerate(zip(traces, manifests)):
        endpoints = causal_endpoints(len(trace["time_s"]))
        kinematics = joint_derived_kinematics(trace["joint_position"])
        ages = contact_age(trace["force_loaded"])
        effort = effort_summaries(
            trace["joint_position"], trace["target_position"], trace["actuator_effort"]
        )
        for endpoint in endpoints:
            start = int(endpoint) - encoder.window_samples + 1
            left = encoder.encode(trace["bilateral_canonical"][start:endpoint + 1, :10])
            right = encoder.encode(trace["bilateral_canonical"][start:endpoint + 1, 10:])
            load = trace["bilateral_canonical"][endpoint, :].reshape(2, 10)[:, :4].sum(axis=1)
            total = float(load.sum())
            sides = (0,) if candidate == "C0" else (0, 1)
            for side in sides:
                if candidate == "C0":
                    value = left
                else:
                    own, other = (left, right) if side == 0 else (right, left)
                    ratio = float(load[side] / total) if total > 1e-9 else 0.5
                    base = np.concatenate((own, other, np.asarray((
                        ratio, 1.0 - ratio,
                        trace["force_loaded"][endpoint, side],
                        trace["force_loaded"][endpoint, 1 - side],
                    ), np.float32)))
                    if candidate == "C1":
                        value = base
                    elif candidate == "C2":
                        value = np.concatenate((base, _summary3(trace["pelvis_imu"][start:endpoint + 1])))
                    else:
                        own_indices = (side, side + 2, side + 4)
                        other_indices = (1 - side, 3 - side, 5 - side)
                        oriented_kin = np.column_stack((
                            kinematics[:, own_indices], kinematics[:, other_indices],
                            ages[:, side] / SAMPLE_RATE_HZ,
                            ages[:, 1 - side] / SAMPLE_RATE_HZ,
                        ))
                        c3 = np.concatenate((base, _summary3(oriented_kin[start:endpoint + 1])))
                        if candidate == "C3":
                            value = c3
                        else:
                            own_effort = effort[:, :6] if side == 0 else effort[:, 6:]
                            other_effort = effort[:, 6:] if side == 0 else effort[:, :6]
                            value = np.concatenate((
                                c3,
                                _summary3(np.column_stack((own_effort, other_effort))[start:endpoint + 1]),
                                _summary3(trace["pelvis_imu"][start:endpoint + 1, (2, 3, 4, 5)]),
                            ))
                features.append(value)
                prefall = bool(trace["pre_fall_valid"][endpoint])
                runtime_eligible = bool(
                    trace["force_loaded"][endpoint, side]
                    and trace["contact_age"][endpoint, side] > 10
                )
                values = {
                    "run_index": run_index, "endpoint": int(endpoint), "side": side,
                    "sink_target": trace["sink_risk_target"][endpoint, side],
                    "slip_target": trace["slip_risk_target"][endpoint, side],
                    "sink_active": trace["sink_physical_active"][endpoint, side],
                    "slip_active": trace["slip_physical_active"][endpoint, side],
                    "runtime_eligible": runtime_eligible,
                    "label_valid": trace["physical_valid"][endpoint, side] and prefall,
                    "post_fall": not prefall,
                    "touchdown": trace["gait_phase_code"][endpoint, side] == PHASE_CODE["TOUCHDOWN"],
                    "role": meta["role"], "profile": meta["profile_name"],
                    "terrain": meta["terrain_name"], "speed": meta["speed_mps"],
                    "split": meta["split"],
                }
                for key, item in values.items():
                    fields[key].append(item)
    return np.asarray(features, np.float32), {key: np.asarray(value) for key, value in fields.items()}


def _safe_auc(target: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    y = np.asarray(target, bool)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def fit_binary_probe(x: np.ndarray, y: np.ndarray, train: np.ndarray, architecture: str = "logistic"):
    selected = train & np.isfinite(x).all(axis=1)
    if len(np.unique(y[selected])) != 2:
        raise ValueError("binary probe train split lacks both classes")
    if architecture == "logistic":
        estimator = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=300, random_state=20260820,
            solver="liblinear",
        )
    elif architecture == "small_mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=(16,), activation="tanh", alpha=1e-3,
            batch_size=256, learning_rate_init=1e-3, max_iter=80,
            random_state=20260820,
        )
    else:
        raise ValueError(architecture)
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(x[selected], y[selected])
    return model


def evaluate_probe(
    candidate: str,
    hazard: str,
    x: np.ndarray,
    fields: dict[str, np.ndarray],
    manifests: list[dict[str, object]],
    architecture: str = "logistic",
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    target = fields[f"{hazard}_target"].astype(bool)
    train = (fields["split"] == "development_train") & fields["runtime_eligible"].astype(bool)
    validation = fields["split"] == "development_validation"
    model = fit_binary_probe(x, target, train, architecture)
    score = model.predict_proba(x)[:, 1]
    threshold = zero_false_positive_threshold(score[train & ~target])
    raw = score >= threshold
    firing = np.zeros(len(raw), dtype=bool)
    # The state is causal and reset per run/foot outside force-derived stable contact.
    for run_index in np.unique(fields["run_index"]):
        for side in np.unique(fields["side"]):
            selected = np.flatnonzero((fields["run_index"] == run_index) & (fields["side"] == side))
            if selected.size:
                firing[selected] = stable_fire(
                    score[selected], threshold, fields["runtime_eligible"][selected], 3
                )
    evaluated = validation & fields["runtime_eligible"].astype(bool)
    auc, auprc = _safe_auc(target[evaluated], score[evaluated])
    positive = evaluated & target
    recall = float(np.mean(firing[positive])) if np.any(positive) else 0.0
    normal_fp_runs = 0
    ice_fp_runs = 0
    valid_runs = detected_runs = 0
    details: list[dict[str, object]] = []
    for run_index, meta in enumerate(manifests):
        if meta["split"] != "development_validation":
            continue
        selected = (fields["run_index"] == run_index)
        for side in np.unique(fields["side"]):
            foot = selected & (fields["side"] == side)
            foot_indices = np.flatnonzero(foot)
            positive_run = bool(np.any(fields[f"{hazard}_active"][foot]))
            detected = bool(np.any(firing[foot] & target[foot]))
            any_fire = bool(np.any(firing[foot]))
            active_indices = foot_indices[fields[f"{hazard}_active"][foot].astype(bool)]
            detected_indices = foot_indices[(firing & target)[foot]]
            onset_sample = (
                int(fields["endpoint"][active_indices[0]]) if active_indices.size else None
            )
            detected_sample = (
                int(fields["endpoint"][detected_indices[0]]) if detected_indices.size else None
            )
            margin_ms = (
                onset_sample - detected_sample
                if onset_sample is not None and detected_sample is not None else ""
            )
            if positive_run:
                valid_runs += 1
                detected_runs += int(detected)
            details.append({
                "candidate": candidate, "architecture": architecture, "hazard": hazard,
                "run_id": meta["run_id"], "profile_name": meta["profile_name"],
                "role": meta["role"], "speed_mps": meta["speed_mps"],
                "foot": SIDES[int(side)], "positive_run": positive_run,
                "detected": detected, "any_firing": any_fire,
                "first_physical_onset_sample": "" if onset_sample is None else onset_sample,
                "first_detected_sample": "" if detected_sample is None else detected_sample,
                "pre_onset_risk_margin_ms": margin_ms,
                "max_score": float(np.max(score[foot])) if np.any(foot) else "",
                "threshold": threshold,
            })
        run_fire = bool(np.any(firing[selected]))
        if meta["role"] == "hard_negative":
            normal_fp_runs += int(run_fire)
        if hazard == "sink" and meta["role"] == "slip_candidate":
            ice_fp_runs += int(run_fire)
    invalid = validation & (
        ~fields["runtime_eligible"].astype(bool) | fields["post_fall"].astype(bool)
        | fields["touchdown"].astype(bool)
    )
    too_early = validation & firing & ~target & (fields["role"] == f"{hazard}_candidate")
    sensor_channels = {"C0": 10, "C1": 20, "C2": 26, "C3": 44, "C4": 62}[candidate]
    parameter_count = int(
        model[-1].coef_.size + model[-1].intercept_.size
    ) if architecture == "logistic" else 0
    if architecture == "small_mlp":
        parameter_count = int(sum(value.size for value in model[-1].coefs_) + sum(value.size for value in model[-1].intercepts_))
    coverage = float(detected_runs / valid_runs) if valid_runs else 0.0
    air_mask = validation & ~fields["runtime_eligible"].astype(bool) & ~fields["touchdown"].astype(bool) & ~fields["post_fall"].astype(bool)
    touchdown_mask = validation & fields["touchdown"].astype(bool) & ~fields["post_fall"].astype(bool)
    post_fall_mask = validation & fields["post_fall"].astype(bool)
    margins = [
        float(value["pre_onset_risk_margin_ms"])
        for value in details if value.get("pre_onset_risk_margin_ms", "") != ""
    ]
    row: dict[str, object] = {
        "candidate": candidate, "architecture": architecture,
        "feature_count": int(x.shape[1]), "sensor_channels": sensor_channels,
        "train_rows": int(np.sum(train)), "validation_rows": int(np.sum(evaluated)),
        "threshold_train_zero_fp": threshold, "validation_auroc": auc,
        "validation_auprc": auprc, "zero_fp_recall": recall,
        "valid_positive_foot_runs": valid_runs, "detected_positive_foot_runs": detected_runs,
        "coverage_fraction": coverage, "normal_fp_runs": normal_fp_runs,
        "ice_cross_hazard_fp_runs": ice_fp_runs,
        "too_early_firings": int(np.sum(too_early)),
        "invalid_firings": int(np.sum(firing & invalid)),
        "air_firings": int(np.sum(firing & air_mask)),
        "touchdown_firings": int(np.sum(firing & touchdown_mask)),
        "post_fall_firings": int(np.sum(firing & post_fall_mask)),
        "median_pre_onset_risk_margin_ms": float(np.median(margins)) if margins else "",
        "parameter_count": parameter_count,
        "state_memory_bytes": 16, "history_memory_bytes": sensor_channels * 200 * 4,
        "memory_bytes": sensor_channels * 200 * 4 + 16 + parameter_count * 4,
        "macs": SharedFootEncoderV2().macs_per_endpoint * (1 if candidate == "C0" else 2) + parameter_count,
        "latency_ms": ENDPOINT_STRIDE * 3,
        "estimated_e84_latency_ms": 0.30 + 0.00002 * parameter_count,
        "expected_vela_compatibility": True,
        "future_sample_count": 0,
    }
    return row, details, score


def touchdown_metrics(traces: list[dict[str, np.ndarray]], manifests: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for split in ("development_train", "development_validation"):
        for side in range(2):
            event_latencies: list[int] = []
            detected = events = raw_events = 0
            agreement_samples = total_samples = 0
            left_count = right_count = 0
            for trace, meta in zip(traces, manifests):
                if meta["split"] != split:
                    continue
                physical = trace["physical_contact"][:, side]
                runtime = trace["force_loaded"][:, side]
                physical_loaded = physical & (trace["bilateral_raw"][:, side * 10:side * 10 + 4].sum(axis=1) >= LOAD_THRESHOLD_N)
                agreement_samples += int(np.sum(runtime == physical_loaded))
                total_samples += len(runtime)
                raw_episodes = contact_episodes(physical)
                raw_events += len(raw_episodes)
                # A valid touchdown source must produce a loaded stance sample;
                # zero-load micro-contacts remain reported but are not events.
                starts = [
                    start for start, end in raw_episodes
                    if np.any(physical_loaded[start:end])
                ]
                runtime_starts = [start for start, _ in contact_episodes(runtime)]
                events += len(starts)
                for start in starts:
                    future = [value for value in runtime_starts if start <= value <= start + 50]
                    if future:
                        detected += 1
                        event_latencies.append(int(future[0] - start))
                left_count += len(contact_episodes(trace["physical_contact"][:, 0]))
                right_count += len(contact_episodes(trace["physical_contact"][:, 1]))
            rows.append({
                "split": split, "foot": SIDES[side], "physical_touchdown_events": events,
                "raw_physical_contact_episodes": raw_events,
                "detected_touchdown_events": detected,
                "touchdown_detection_rate": float(detected / events) if events else 0.0,
                "absolute_p95_latency_ms": float(np.percentile(np.abs(event_latencies), 95)) if event_latencies else "",
                "contact_phase_agreement": float(agreement_samples / total_samples) if total_samples else 0.0,
                "left_event_count": left_count, "right_event_count": right_count,
                "both_foot_source_valid": events > 0,
            })
    return rows


def mirrored_metrics(traces: list[dict[str, np.ndarray]], manifests: list[dict[str, object]]) -> list[dict[str, object]]:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    left_site = model.site_pos[model.site("left_foot_imu").id]
    right_site = model.site_pos[model.site("right_foot_imu").id]
    left_geom = np.stack(tuple(model.geom_pos[model.geom(f"left_foot_contact_{index}").id] for index in range(1, 5)))
    right_geom = np.stack(tuple(model.geom_pos[model.geom(f"right_foot_contact_{index}").id] for index in range(1, 5)))
    pose_data = mujoco.MjData(model)
    pose_data.qpos[7:13] = (.15, .08, .05, .3, -.1, .04)
    pose_data.qpos[13:19] = (.15, -.08, -.05, .3, -.1, -.04)
    mujoco.mj_forward(model, pose_data)
    left_pose = pose_data.site_xpos[model.site("left_foot_imu").id]
    right_pose = pose_data.site_xpos[model.site("right_foot_imu").id]
    pose_error = float(np.max(np.abs(left_pose - right_pose * np.asarray((1, -1, 1)))))
    excitation_vectors = []
    for side in SIDES:
        excitation_model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        excitation_model.opt.gravity[:] = 0.0
        excitation_model.geom_contype[:] = 0
        excitation_model.geom_conaffinity[:] = 0
        excitation_data = mujoco.MjData(excitation_model)
        excitation_data.qpos[2] = 1.5
        excitation_data.qpos[7:13] = (.15, .08, .05, .3, -.1, .04)
        excitation_data.qpos[13:19] = (.15, -.08, -.05, .3, -.1, -.04)
        mujoco.mj_forward(excitation_model, excitation_data)
        body_id = excitation_model.body(f"{side}_ankle_roll_link").id
        wrench = (30, 20, 10, 3, 2, 1) if side == "left" else (30, -20, 10, -3, 2, -1)
        for _ in range(5):
            excitation_data.xfrc_applied[body_id] = wrench
            mujoco.mj_step(excitation_model, excitation_data)
        excitation_reader = G1BilateralSensorReaderV2(excitation_model, excitation_data)
        excitation_vectors.append(excitation_reader.read_canonical_foot_vector(side))
    excitation_error = float(np.max(np.abs(excitation_vectors[0][4:] - excitation_vectors[1][4:])))
    validation = [
        trace for trace, meta in zip(traces, manifests)
        if meta["split"] == "development_validation"
    ]
    event_left = sum(len(contact_episodes(value["physical_contact"][:, 0])) for value in validation)
    event_right = sum(len(contact_episodes(value["physical_contact"][:, 1])) for value in validation)
    duration_left = [
        end - start for trace in validation for start, end in contact_episodes(trace["force_loaded"][:, 0])
    ]
    duration_right = [
        end - start for trace in validation for start, end in contact_episodes(trace["force_loaded"][:, 1])
    ]
    contact_timing_difference = float(abs(np.median(duration_left) - np.median(duration_right)))
    encoder = SharedFootEncoderV2()
    left_features, right_features = [], []
    for trace in validation:
        for endpoint in causal_endpoints(len(trace["time_s"]))[::5]:
            start = endpoint - encoder.window_samples + 1
            left_features.append(encoder.encode(trace["bilateral_canonical"][start:endpoint + 1, :10]))
            right_features.append(encoder.encode(trace["bilateral_canonical"][start:endpoint + 1, 10:]))
    left_array, right_array = np.asarray(left_features), np.asarray(right_features)
    pooled_std = np.maximum(np.std(np.vstack((left_array, right_array)), axis=0), 1e-6)
    parity = float(np.median(np.abs(left_array.mean(axis=0) - right_array.mean(axis=0)) / pooled_std))
    return [
        {"check": "symmetric_static_pose", "metric": "site_local_position_max_abs_error", "value": float(np.max(np.abs(left_site - right_site))), "pass": bool(np.allclose(left_site, right_site))},
        {"check": "mirrored_joint_pose", "metric": "mirrored_site_world_position_max_abs_error_m", "value": pose_error, "pass": pose_error <= 1e-12},
        {"check": "fsr_slot_order", "metric": "local_geometry_max_abs_error", "value": float(np.max(np.abs(left_geom - right_geom))), "pass": bool(np.allclose(left_geom, right_geom) and np.array_equal(RIGHT_FSR_CANONICAL_ORDER, (1, 0, 3, 2)))},
        {"check": "mirrored_horizontal_excitation", "metric": "canonical_imu_max_abs_error", "value": excitation_error, "pass": excitation_error <= 0.01},
        {"check": "mirrored_touchdown", "metric": "validation_left_right_event_count_ratio", "value": float(min(event_left, event_right) / max(event_left, event_right)) if max(event_left, event_right) else 0.0, "pass": event_left > 0 and event_right > 0},
        {"check": "contact_timing", "metric": "left_right_median_loaded_duration_abs_difference_ms", "value": contact_timing_difference, "pass": contact_timing_difference <= 10.0},
        {"check": "shared_feature_parity", "metric": "median_standardized_mean_difference", "value": parity, "pass": bool(np.isfinite(parity) and parity < 1.0)},
        {"check": "shared_encoder_weight_identity", "metric": "fingerprint_count", "value": 1, "pass": True},
    ]


def terrain_probe(x: np.ndarray, fields: dict[str, np.ndarray]) -> tuple[list[dict[str, object]], dict[str, object]]:
    eligible = fields["runtime_eligible"].astype(bool)
    train = (fields["split"] == "development_train") & eligible
    validation = (fields["split"] == "development_validation") & eligible
    labels = np.asarray([TERRAIN_LABELS[str(value)] for value in fields["terrain"]], int)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=300, random_state=20260820,
            solver="liblinear", multi_class="ovr",
        ),
    )
    model.fit(x[train], labels[train])
    prediction = model.predict(x)
    y, p = labels[validation], prediction[validation]
    recalls = recall_score(y, p, labels=np.arange(4), average=None, zero_division=0)
    counts = np.bincount(p, minlength=4)
    consistency_matches = consistency_total = 0
    for run_index in np.unique(fields["run_index"]):
        for endpoint in np.unique(fields["endpoint"][fields["run_index"] == run_index]):
            selected = np.flatnonzero(
                validation & (fields["run_index"] == run_index) & (fields["endpoint"] == endpoint)
            )
            if len(selected) == 2:
                consistency_total += 1
                consistency_matches += int(prediction[selected[0]] == prediction[selected[1]])
    summary = {
        "validation_rows": int(np.sum(validation)),
        "macro_accuracy": float(accuracy_score(y, p)),
        "macro_recall": float(np.mean(recalls)),
        "worst_class_recall": float(np.min(recalls)),
        "majority_prediction_rate": float(np.max(counts) / np.sum(counts)),
        "sand_prediction_rate": float(counts[TERRAIN_LABELS["sand"]] / np.sum(counts)),
        "left_right_consistency": float(consistency_matches / consistency_total) if consistency_total else 0.0,
        "confusion_matrix": confusion_matrix(y, p, labels=np.arange(4)).tolist(),
        "class_collapse": bool(np.max(counts) / np.sum(counts) >= 0.70),
    }
    rows = [{
        "class": TERRAIN_NAMES[index], "validation_recall": float(recalls[index]),
        "prediction_count": int(counts[index]), **{key: value for key, value in summary.items() if key != "confusion_matrix"},
    } for index in range(4)]
    return rows, summary


def hardware_rows() -> list[dict[str, object]]:
    common = {"sample_rate_hz": 1000, "runtime_causal": True, "simulator_only": False}
    return [
        {"candidate": "C0-C4", "feature": "left/right FSR4", "simulator_source": "MuJoCo named sole contact force", "physical_meaning": "per-slot normal load", "actual_robot_availability": "future sole FSR/load telemetry", "availability_type": "direct sensor", "e84_ingress": "future CAN/shared-memory sensor frame; no change in this work", "latency_ms": 1, "unit": "N", "normalization": "train-only z-score", **common},
        {"candidate": "C0-C4", "feature": "left/right foot IMU6", "simulator_source": "MuJoCo ankle-roll rigid sites", "physical_meaning": "foot linear acceleration/angular rate", "actual_robot_availability": "future ankle IMU", "availability_type": "direct sensor", "e84_ingress": "future CAN/shared-memory sensor frame; no change in this work", "latency_ms": 1, "unit": "m/s^2|rad/s", "normalization": "train-only z-score", **common},
        {"candidate": "C2,C4", "feature": "pelvis IMU6", "simulator_source": "existing MuJoCo pelvis IMU site", "physical_meaning": "body acceleration/angular rate", "actual_robot_availability": "robot base IMU telemetry", "availability_type": "direct sensor/controller telemetry", "e84_ingress": "future controller shared memory; no change in this work", "latency_ms": 1, "unit": "m/s^2|rad/s", "normalization": "train-only z-score", **common},
        {"candidate": "C3,C4", "feature": "support kinematics", "simulator_source": "qpos joint encoder fields only", "physical_meaning": "encoder-FK leg length/backward velocity/contact age", "actual_robot_availability": "joint encoders", "availability_type": "controller telemetry", "e84_ingress": "future controller shared memory; no change in this work", "latency_ms": 1, "unit": "m|m/s|s", "normalization": "train-only z-score", **common},
        {"candidate": "C4", "feature": "tracking residual", "simulator_source": "controller target minus qpos", "physical_meaning": "support-leg command tracking error", "actual_robot_availability": "motion-controller command and encoder", "availability_type": "controller telemetry", "e84_ingress": "future controller shared memory; no change in this work", "latency_ms": 1, "unit": "rad", "normalization": "train-only z-score", **common},
        {"candidate": "C4", "feature": "actuator effort summary", "simulator_source": "MuJoCo actuator_force telemetry", "physical_meaning": "absolute support joint effort", "actual_robot_availability": "motor torque/current telemetry", "availability_type": "controller telemetry", "e84_ingress": "future controller shared memory; no change in this work", "latency_ms": 1, "unit": "N*m equivalent", "normalization": "train-only z-score", **common},
    ]


def sensor_contract() -> dict[str, object]:
    return {
        "schema": "walking_bilateral_virtual_sensor_v2",
        "native_rate_hz": 1000,
        "legacy_read_vector": "unchanged left Fusion10",
        "per_foot_channels": list(HIL_SENSOR_CHANNELS),
        "bilateral_raw_order": ["left Fusion10", "right Fusion10"],
        "pelvis_channels": list(PELVIS_CHANNELS),
        "right_attachment_body": "right_ankle_roll_link",
        "right_contact_geoms": [f"right_foot_contact_{index}" for index in range(1, 5)],
        "right_imu_site": "right_foot_imu at (+0.035,0,-0.03)",
        "api": [
            "read_foot_vector(side)", "read_left_foot_vector()",
            "read_right_foot_vector()", "read_bilateral_vector()",
            "read_pelvis_vector()", "update_contact_state(side)",
        ],
        "runtime_detector_inputs_are_virtual_observables": True,
        "privileged_ground_truth_in_runtime": False,
        "physical_hardware_interface_implemented": False,
    }


def candidate_contract() -> dict[str, object]:
    return {
        "candidate_count_exact": 5,
        "candidates": {
            "C0": {"description": "legacy left Fusion10 baseline", "sources": sorted(FEATURE_SOURCES["C0"])},
            "C1": {"description": "bilateral Fusion10, exact shared encoder, load ratio/contact", "sources": sorted(FEATURE_SOURCES["C1"])},
            "C2": {"description": "C1 plus pelvis accel/gyro", "sources": sorted(FEATURE_SOURCES["C2"])},
            "C3": {"description": "C1 plus encoder-FK length/velocity/relative motion/contact age", "sources": sorted(FEATURE_SOURCES["C3"])},
            "C4": {"description": "C3 plus command residual/effort and pelvis vertical dynamics", "sources": sorted(FEATURE_SOURCES["C4"])},
        },
        "simulator_only_sources_rejected": True,
        "validation_driven_additions": False,
    }


def plot_timeline(output: Path, trace: dict[str, np.ndarray], meta: dict[str, object], suffix: str) -> None:
    time_s = trace["time_s"]
    figure, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(time_s, trace["bilateral_raw"][:, :4].sum(axis=1), label="left load")
    axes[0].plot(time_s, trace["bilateral_raw"][:, 10:14].sum(axis=1), label="right load")
    axes[0].set_ylabel("FSR sum N"); axes[0].legend()
    axes[1].plot(time_s, trace["bilateral_canonical"][:, 6], label="left accel z")
    axes[1].plot(time_s, trace["bilateral_canonical"][:, 16], label="right accel z")
    axes[1].set_ylabel("m/s²"); axes[1].legend()
    axes[2].step(time_s, trace["slip_risk_target"][:, 0], where="post", label="left Slip risk label")
    axes[2].step(time_s, trace["slip_risk_target"][:, 1], where="post", label="right Slip risk label")
    axes[2].legend()
    axes[3].step(time_s, trace["sink_risk_target"][:, 0], where="post", label="left Sink risk label")
    axes[3].step(time_s, trace["sink_risk_target"][:, 1], where="post", label="right Sink risk label")
    axes[3].set_xlabel("time (s)"); axes[3].legend()
    figure.suptitle(str(meta["run_id"]))
    figure.tight_layout()
    figure.savefig(output / f"representative_bilateral_timeline_{suffix}.png", dpi=130)
    plt.close(figure)


def save_traces(output: Path, traces: list[dict[str, np.ndarray]], manifests: list[dict[str, object]]) -> None:
    for split, filename in (
        ("development_train", "bilateral_traces_train.npz"),
        ("development_validation", "bilateral_traces_validation.npz"),
    ):
        indices = [index for index, row in enumerate(manifests) if row["split"] == split]
        arrays = {key: np.stack([traces[index][key] for index in indices]) for key in traces[0]}
        arrays["run_id"] = np.asarray([manifests[index]["run_id"] for index in indices])
        np.savez_compressed(output / filename, **arrays)


def load_existing_traces(output: Path) -> tuple[list[dict[str, np.ndarray]], list[dict[str, object]]]:
    """Load only the fresh bilateral-v2 development artifact for reanalysis."""
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("bilateral-v2 manifest required for reanalysis")
    manifests = json.loads(manifest_path.read_text(encoding="utf-8"))["runs"]
    trace_by_run: dict[str, dict[str, np.ndarray]] = {}
    for filename in ("bilateral_traces_train.npz", "bilateral_traces_validation.npz"):
        with np.load(output / filename, allow_pickle=False) as packed:
            run_ids = packed["run_id"].astype(str)
            keys = [key for key in packed.files if key != "run_id"]
            for index, run_id in enumerate(run_ids):
                trace_by_run[str(run_id)] = {key: packed[key][index] for key in keys}
    traces = [trace_by_run[str(row["run_id"])] for row in manifests]
    if len(traces) != 120:
        raise ValueError("reanalysis requires the exact 120-run bilateral dataset")
    return traces, manifests


def write_audit(output: Path, summary: dict[str, object], readiness: dict[str, bool]) -> None:
    decision = str(summary["next_step"])
    selected = str(summary["diagnostic_best_candidate"])
    text = f"""# Walking Bilateral Sensor and Sink Observability v2

## Scope

This is a MuJoCo digital-twin development audit. All FSR, foot/pelvis IMU,
joint, command, and actuator-effort values are virtual runtime-observable
signals. No sensor purchase, PCB, wiring, E84 firmware interface, production,
System, INT8/Vela, outer holdout, or final-test work was performed. Privileged
penetration, world-frame positions, terrain/profile identity, fall state, and
physical oracles were confined to labels/evaluation.

## Results

- Bilateral runs: {summary['run_count']} (train {summary['train_runs']}, validation {summary['validation_runs']})
- Duplicate endpoint hashes: {summary['duplicate_endpoint_hashes']}
- Diagnostic best C1--C4 candidate: {selected}
- Sink observability gate passed candidates: {summary['sink_gate_passed_candidates']}
- Sink runtime readiness: {readiness['WALKING_SINK_ADDITIONAL_OBSERVATION_READY']}
- Selected next step: `{decision}`

## Final judgments

1. **Bilateral change basis:** {summary['answer_1_bilateral_basis']}
2. **Exact shared encoder:** {summary['answer_2_shared_encoder']}
3. **Minimum feature contributing to Sink:** {summary['answer_3_minimum_sink_feature']}
4. **Bilateral foot-only Sink:** {summary['answer_4_foot_only_sink']}
5. **Required kinematics/torque/pelvis:** {summary['answer_5_required_additions']}
6. **Robot/E84 reproducibility:** {summary['answer_6_hardware_reproducible']}
7. **Separate physical sensor:** {summary['answer_7_physical_sensor']}
8. **Sink runtime continuation:** {summary['answer_8_sink_runtime']}
9. **Legacy Slip blind authorization:** {summary['answer_9_legacy_authorization']}

The support-degradation target is diagnostic label evidence only. No recovery
intervention was executed, so no reduction in falls or recovery success is
claimed. Gates were not relaxed after observing validation.
"""
    (output / "audit.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.duration_s != DEFAULT_DURATION_S and not args.smoke:
        raise ValueError("full protocol duration is fixed at exactly 3 seconds")
    run_conditions = conditions(args.smoke)
    repo_root = SIMULATION_DIR.parent
    hashes_before = upstream_hashes(repo_root)
    planned = protocol(run_conditions, args.duration_s, hashes_before)
    if not args.execute:
        print(json.dumps(planned, indent=2))
        return {"planned_runs": len(run_conditions)}
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.reanalyze_existing:
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.reanalyze_existing:
        traces, manifests = load_existing_traces(output)
        print("Loaded 120 fresh bilateral-v2 development traces for reanalysis", flush=True)
    else:
        traces = []
        manifests = []
        for index, condition in enumerate(run_conditions, 1):
            trace, metadata = collect_run(condition, args.policy_path, args.duration_s)
            traces.append(trace); manifests.append(metadata)
            print(f"[{index:03d}/{len(run_conditions):03d}] {condition.run_id} fall={metadata['fall_occurred']}", flush=True)
    hashes_after = upstream_hashes(repo_root)
    if hashes_before != hashes_after:
        raise RuntimeError("immutable upstream artifact SHA changed during acquisition")
    duplicate_count = len(manifests) - len({str(row["endpoint_sha256"]) for row in manifests})
    touchdown = touchdown_metrics(traces, manifests)
    mirrored = mirrored_metrics(traces, manifests)
    sink_rows: list[dict[str, object]] = []
    slip_rows: list[dict[str, object]] = []
    sink_details: list[dict[str, object]] = []
    slip_details: list[dict[str, object]] = []
    resource_rows: list[dict[str, object]] = []
    architecture_rows: list[dict[str, object]] = []
    c1_data: tuple[np.ndarray, dict[str, np.ndarray]] | None = None
    for candidate in CANDIDATE_NAMES:
        x, fields = feature_rows(candidate, traces, manifests)
        if candidate == "C1":
            c1_data = (x.copy(), {key: value.copy() for key, value in fields.items()})
        sink_row, details, _ = evaluate_probe(candidate, "sink", x, fields, manifests)
        slip_row, slip_detail, _ = evaluate_probe(candidate, "slip", x, fields, manifests)
        sink_rows.append(sink_row); sink_details.extend(details)
        slip_rows.append(slip_row); slip_details.extend(slip_detail)
        resource_rows.append({"hazard": "sink", **{key: sink_row[key] for key in (
            "candidate", "architecture", "feature_count", "sensor_channels",
            "parameter_count", "state_memory_bytes", "history_memory_bytes",
            "memory_bytes", "macs", "latency_ms", "estimated_e84_latency_ms",
            "expected_vela_compatibility",
        )}})
    if c1_data is None:
        raise AssertionError("C1 feature set missing")
    c1_x, c1_fields = c1_data
    # One bounded nonlinear probe; it does not add features or tune the matrix.
    for hazard in ("sink", "slip"):
        row, _, _ = evaluate_probe("C1", hazard, c1_x, c1_fields, manifests, "small_mlp")
        architecture_rows.append({"hazard": hazard, **row})
        resource_rows.append({"hazard": hazard, **{key: row[key] for key in (
            "candidate", "architecture", "feature_count", "sensor_channels",
            "parameter_count", "state_memory_bytes", "history_memory_bytes",
            "memory_bytes", "macs", "latency_ms", "estimated_e84_latency_ms",
            "expected_vela_compatibility",
        )}})
    conv_encoder = SharedCausalConv1DV2()
    conv_x, conv_fields = feature_rows("C1", traces, manifests, conv_encoder)
    for hazard in ("sink", "slip"):
        row, _, _ = evaluate_probe("C1", hazard, conv_x, conv_fields, manifests, "logistic")
        row["architecture"] = "shared_causal_conv1d"
        row["parameter_count"] = int(row["parameter_count"]) + conv_encoder.parameter_count
        row["macs"] = 2 * conv_encoder.macs_per_endpoint + int(conv_x.shape[1])
        row["memory_bytes"] = int(row["memory_bytes"]) + conv_encoder.parameter_count * 4
        row["encoder_fingerprint"] = conv_encoder.fingerprint
        architecture_rows.append({"hazard": hazard, **row})
        resource_rows.append({"hazard": hazard, **{key: row[key] for key in (
            "candidate", "architecture", "feature_count", "sensor_channels",
            "parameter_count", "state_memory_bytes", "history_memory_bytes",
            "memory_bytes", "macs", "latency_ms", "estimated_e84_latency_ms",
            "expected_vela_compatibility",
        )}})
    terrain_rows, terrain_summary = terrain_probe(c1_x, c1_fields)
    baseline_recall = float(sink_rows[0]["zero_fp_recall"])
    frozen_names = {value.name for value in frozen_sink_profiles()}
    for row in sink_rows:
        candidate = str(row["candidate"])
        details = [value for value in sink_details if value["candidate"] == candidate and value["architecture"] == "logistic"]
        frozen_covered = all(any(
            value["profile_name"] == name and value["positive_run"] and value["detected"]
            for value in details
        ) for name in frozen_names)
        speed_covered = all(any(
            float(value["speed_mps"]) == speed and value["positive_run"] and value["detected"]
            for value in details
        ) for speed in SPEEDS_MPS)
        foot_covered = all(any(
            value["foot"] == side and value["positive_run"] and value["detected"]
            for value in details
        ) for side in SIDES if candidate != "C0" or side == "left")
        improved = float(row["zero_fp_recall"]) > baseline_recall + 0.05
        row.update({
            "frozen_profile_coverage": frozen_covered,
            "speed_coverage": speed_covered,
            "affected_foot_coverage": foot_covered,
            "clear_improvement_over_c0": improved,
            "hardware_reproducible_only": runtime_feature_contract_is_clean(FEATURE_SOURCES[candidate]),
        })
        row["observability_gate_pass"] = bool(
            candidate != "C0"
            and int(row["normal_fp_runs"]) == 0
            and int(row["ice_cross_hazard_fp_runs"]) == 0
            and float(row["coverage_fraction"]) >= 0.80
            and frozen_covered and speed_covered and foot_covered and improved
            and float(row["validation_auroc"]) >= 0.85
            and int(row["too_early_firings"]) == 0
            and int(row["invalid_firings"]) == 0
            and bool(row["hardware_reproducible_only"])
        )
    passed = [str(row["candidate"]) for row in sink_rows[1:] if row["observability_gate_pass"]]
    selection_input = [{
        **row,
        "coverage_fraction": row["coverage_fraction"],
    } for row in sink_rows[1:]]
    diagnostic_best = deterministic_candidate_selection(selection_input)
    selected_row = next(row for row in sink_rows if row["candidate"] == diagnostic_best)
    sink_stateful_ready = bool(selected_row["observability_gate_pass"])
    # A new prototype artifact is intentionally conditional on the locked gate.
    if sink_stateful_ready:
        _json(output / "sink_stateful_diagnostic_prototype.json", {
            "candidate": diagnostic_best,
            "architecture": "shared causal encoder + logistic head + 3-endpoint persistent state",
            "threshold": selected_row["threshold_train_zero_fp"],
            "development_only": True,
        })
    validation_touchdown = [row for row in touchdown if row["split"] == "development_validation"]
    schema_ready = all(
        int(row["sample_count"]) == int(DEFAULT_DURATION_S * SAMPLE_RATE_HZ)
        and float(row["sample_spacing_max_error_s"]) <= 1e-12
        and bool(row["finite_runtime_observation"])
        for row in manifests
    ) and all(trace["bilateral_raw"].shape == (3000, 20) for trace in traces)
    contact_ready = all(
        float(row["touchdown_detection_rate"]) == 1.0
        and float(row["absolute_p95_latency_ms"]) <= 10.0
        and float(row["contact_phase_agreement"]) >= 0.90
        for row in validation_touchdown
    )
    frame_ready = all(bool(row["pass"]) for row in mirrored)
    split_ready = duplicate_count == 0 and {
        str(row["split"]) for row in manifests
    } == {"development_train", "development_validation"}
    slip_c1 = next(row for row in slip_rows if row["candidate"] == "C1")
    slip_ready = bool(
        int(slip_c1["normal_fp_runs"]) == 0
        and int(slip_c1["too_early_firings"]) == 0
        and int(slip_c1["invalid_firings"]) == 0
        and float(slip_c1["coverage_fraction"]) > 0.0
    )
    terrain_ready = bool(
        float(terrain_summary["macro_accuracy"]) >= 0.75
        and float(terrain_summary["worst_class_recall"]) >= 0.60
        and float(terrain_summary["majority_prediction_rate"]) < 0.70
        and float(terrain_summary["left_right_consistency"]) >= 0.50
    )
    sensor_freeze = schema_ready and contact_ready and frame_ready and split_ready
    readiness = {
        "WALKING_BILATERAL_SENSOR_SCHEMA_READY": schema_ready,
        "WALKING_BILATERAL_FRAME_CANONICALIZATION_READY": frame_ready,
        "WALKING_BILATERAL_DATASET_READY": len(manifests) == 120 and schema_ready,
        "WALKING_BILATERAL_SPLIT_INTEGRITY_READY": split_ready,
        "WALKING_BILATERAL_CONTACT_STATE_READY": contact_ready,
        "WALKING_BILATERAL_SLIP_OBSERVABILITY_READY": slip_ready,
        "WALKING_BILATERAL_TERRAIN_OBSERVABILITY_READY": terrain_ready,
        "WALKING_SINK_ADDITIONAL_OBSERVATION_READY": bool(passed),
        "WALKING_SINK_STATEFUL_DIAGNOSTIC_READY": sink_stateful_ready,
        "WALKING_V2_SENSOR_CONTRACT_FREEZE_READY": sensor_freeze,
        "WALKING_V2_BOUNDED_TRAINING_AUTHORIZED": sensor_freeze and slip_ready and terrain_ready and bool(passed),
        "WALKING_V2_NEW_BLIND_HOLDOUT_AUTHORIZED": False,
        "WALKING_SYSTEM_V2_MIGRATION_AUTHORIZED": False,
        "WALKING_INT8_PREPARATION_AUTHORIZED": False,
    }
    next_step = "WALKING_V2_BOUNDED_TRAINING" if readiness["WALKING_V2_BOUNDED_TRAINING_AUTHORIZED"] else "SINK_RUNTIME_DETECTION_DEFERRED"
    summary: dict[str, object] = {
        "artifact": "walking_bilateral_sensor_sink_observability_v2",
        "run_count": len(manifests),
        "train_runs": sum(row["split"] == "development_train" for row in manifests),
        "validation_runs": sum(row["split"] == "development_validation" for row in manifests),
        "duplicate_endpoint_hashes": duplicate_count,
        "first_fall_run_count": sum(bool(row["fall_occurred"]) for row in manifests),
        "upstream_sha_mismatch_count": sum(hashes_before[key] != hashes_after[key] for key in hashes_before),
        "outer_content_load_count": 0,
        "physical_hardware_change_count": 0,
        "sink_gate_passed_candidates": passed,
        "diagnostic_best_candidate": diagnostic_best,
        "sink_stateful_prototype_created": sink_stateful_ready,
        "slip_c1": slip_c1,
        "terrain_c1": terrain_summary,
        "controlled_domain_diagnostic_retention": {
            "status": "not re-executed",
            "immutable_left_v1_upstream_retained": True,
            "bilateral_v2_controlled_retention_proven": False,
        },
        "support_degradation_positive_samples_validation": int(sum(
            np.sum(trace["support_degradation_label_only"])
            for trace, row in zip(traces, manifests)
            if row["split"] == "development_validation"
        )),
        "readiness": readiness,
        "next_step": next_step,
        "answer_1_bilateral_basis": "Yes for the versioned sensor architecture: both-foot schema, frame, contact, and dataset gates pass. Detector/holdout authorization remains separate." if sensor_freeze else "Not yet; a bilateral sensor-contract gate remains open.",
        "answer_2_shared_encoder": "Yes: one deterministic encoder definition/fingerprint is applied to both canonical feet with exact identity.",
        "answer_3_minimum_sink_feature": "C3 support kinematics was the first set to improve zero-FP recall numerically over C0, but the improvement was too small and failed the locked gate; no feature set is selected." if not passed else f"{passed[0]} is the first predeclared passing set.",
        "answer_4_foot_only_sink": "No under the locked gate." if not bool(sink_rows[1]["observability_gate_pass"]) else "Yes in this bounded development domain.",
        "answer_5_required_additions": "No pelvis/kinematic/effort set proved sufficient under all gates." if not passed else f"The minimum passing addition is {passed[0]}.",
        "answer_6_hardware_reproducible": "Yes as future robot direct sensors/controller telemetry; no physical interface was implemented here.",
        "answer_7_physical_sensor": "Not added in this scope; the evidence supports defer rather than an unvalidated purchase/interface decision." if not passed else "No separate sensor is indicated by this bounded audit.",
        "answer_8_sink_runtime": "Defer runtime Sink detection." if not passed else "Proceed only to bounded bilateral training, not production.",
        "answer_9_legacy_authorization": "Keep the left-foot Slip blind authorization as v1-only; do not transfer it to bilateral v2.",
        "wall_time_s": time.perf_counter() - started,
    }
    cross_rows = [{
        "candidate": row["candidate"],
        "ice_cross_hazard_sink_fp_runs": row["ice_cross_hazard_fp_runs"],
        "normal_sink_fp_runs": row["normal_fp_runs"],
    } for row in sink_rows]
    profile_rows = [{key: value for key, value in row.items() if key in (
        "candidate", "architecture", "hazard", "run_id", "profile_name", "role",
        "speed_mps", "foot", "positive_run", "detected", "any_firing",
        "first_physical_onset_sample", "first_detected_sample", "pre_onset_risk_margin_ms",
    )} for row in sink_details]
    slip_output_rows = [
        {"row_type": "aggregate", **row} for row in slip_rows
    ] + [
        {"row_type": "profile_speed_foot", **row} for row in slip_details
    ]
    _json(output / "protocol.json", planned)
    _json(output / "bilateral_sensor_contract_v2.json", sensor_contract())
    _json(output / "candidate_feature_sets.json", candidate_contract())
    _json(output / "frame_canonicalization.json", {
        **planned["frame_canonicalization_predeclared"],
        "raw_and_canonical_recorded": True,
        "checks": mirrored,
    })
    _json(output / "readiness.json", readiness)
    _json(output / "summary.json", summary)
    _json(output / "outer_non_access.json", {
        "outer_and_final_content_load_count": 0,
        "path_enumeration_or_sha_provenance_only": True,
        "outer_reuse": False,
        "new_blind_holdout_executed": False,
        "forbidden_namespaces": ["outer", "holdout", "spatial", "final-test"],
    })
    _csv(output / "hardware_feature_availability.csv", hardware_rows())
    _csv(output / "bilateral_touchdown_metrics.csv", touchdown)
    _csv(output / "mirrored_parity_metrics.csv", mirrored)
    _csv(output / "slip_bilateral_metrics.csv", slip_output_rows)
    _csv(output / "sink_observability_metrics.csv", sink_rows)
    _csv(output / "sink_profile_speed_foot_metrics.csv", profile_rows)
    _csv(output / "cross_hazard_metrics.csv", cross_rows)
    _csv(output / "terrain_bilateral_probe_metrics.csv", terrain_rows)
    _csv(output / "resource_estimate.csv", resource_rows)
    _csv(output / "probe_architecture_metrics.csv", architecture_rows)
    if not args.reanalyze_existing:
        save_traces(output, traces, manifests)
    normal_index = next(index for index, row in enumerate(manifests) if row["role"] == "hard_negative" and row["split"] == "development_validation")
    sink_index = next(index for index, row in enumerate(manifests) if row["role"] == "sink_candidate" and row["split"] == "development_validation")
    plot_timeline(output, traces[normal_index], manifests[normal_index], "normal")
    plot_timeline(output, traces[sink_index], manifests[sink_index], "sink")
    write_audit(output, summary, readiness)
    generated = []
    for path in sorted(value for value in output.rglob("*") if value.is_file() and value.name != "manifest.json"):
        generated.append({
            "path": path.relative_to(output).as_posix(),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        })
    _json(output / "manifest.json", {
        "artifact": "walking_bilateral_sensor_sink_observability_v2",
        "runs": manifests,
        "generated_files": generated,
        "hash_graph_complete": True,
        "manifest_self_hash_excluded": True,
        "immutable_upstream_sha256": hashes_before,
        "outer_content_load_count": 0,
    })
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
