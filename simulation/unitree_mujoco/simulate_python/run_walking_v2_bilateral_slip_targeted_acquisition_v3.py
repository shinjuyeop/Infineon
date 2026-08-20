"""Acquire and audit a targeted bilateral Slip development corpus.

This task is acquisition-only.  It does not train, select, export, or lock a
Slip detector and it never opens a blind/outer/final artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from bilateral_hil_sensor_v2 import G1BilateralSensorReaderV2, SIDES
from g1_upstream_locomotion import (
    DEFAULT_ANGLES,
    TESTED_POLICY_SHA256,
    UPSTREAM_REVISION,
    UnitreeG1PretrainedController,
    gravity_orientation,
)
from terrain_profiles import TERRAIN_PROFILES, apply_terrain_profile
from walking_hazard_ground_truth_v1 import (
    contact_episodes,
    derive_contact_signals,
    gait_phase,
    max_left_foot_contact_penetration_m,
)
from walking_hazard_oracle_calibration_v1 import persistent_oracle
from run_walking_hazard_ground_truth_v1 import (
    DEFAULT_POLICY,
    GROUND_NAMES,
    PELVIS_BODY_NAME,
    SCENE_PATH,
    _disable_nonfoot_surface_collisions,
    _fall_reasons,
)
from walking_v2_bilateral_slip_targeted_acquisition_v3 import (
    CONTROL_TYPES,
    PHASE_BINS,
    PHYSICS_STEPS_PER_SAMPLE,
    PHYSICS_TIMESTEP_S,
    SAMPLE_RATE_HZ,
    SEVERITIES,
    SIDES as CONTRACT_SIDES,
    SLIP_PERSISTENCE_MS,
    SLIP_THRESHOLD_M,
    SPEEDS_MPS,
    VARIATIONS,
    RunCondition,
    acquisition_matrix,
    assigned_future_fold,
    deterministic_initial_perturbation,
    friction_vector,
    full_trace_sha256,
    material_profiles,
    onset_phase,
    phase_spec,
    validate_future_fold_rows,
)


REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO / "simulation/outputs/walking_v2_bilateral_slip_targeted_acquisition_v3"
STARTING_CHECKPOINT = "64271f14fbcfbd4981d86df8929d2904ec4a744d"
SOURCE = "simulation/outputs/walking_bilateral_sensor_sink_observability_v2"
JOINT = "simulation/outputs/walking_v2_joint_terrain_slip_redesign_v1"
SLIP_V2 = "simulation/outputs/walking_v2_slip_redesign_iteration_v2"
OPERATIONAL = "simulation/outputs/walking_hazard_operational_label_contract_v2"
ORACLE = "simulation/outputs/walking_hazard_oracle_calibration_v1"
RECOVERY = "simulation/outputs/walking_v2_bilateral_slip_targeted_acquisition_v3_attempt_001_failed_postprocessing"
TERRAIN_FILES = {
    "model": f"{JOINT}/terrain_candidate_model.npz",
    "normalization": f"{JOINT}/terrain_candidate_normalization.json",
    "config": f"{JOINT}/terrain_candidate_config.json",
    "selection_lock": f"{JOINT}/terrain_selection_lock.json",
}
TERRAIN_REFERENCES = {
    "protocol": f"{JOINT}/protocol.json",
    "data_manifest": f"{JOINT}/data_manifest.json",
    "split_manifest": f"{JOINT}/split_manifest.json",
    "validation_metrics": f"{JOINT}/terrain_validation_metrics.csv",
    "resource_report": f"{JOINT}/resource_report.json",
}
INPUTS = (
    {"path": f"{SOURCE}/manifest.json", "access": "json", "purpose": "existing 120-run development metadata"},
    {"path": f"{SOURCE}/bilateral_traces_train.npz", "access": "npz", "purpose": "former train partition, reclassified development-only"},
    {"path": f"{SOURCE}/bilateral_traces_validation.npz", "access": "npz", "purpose": "former validation partition, reclassified development-only"},
    {"path": f"{OPERATIONAL}/summary.json", "access": "json", "purpose": "immutable operational-label contract"},
    {"path": f"{ORACLE}/summary.json", "access": "json", "purpose": "immutable physical-oracle calibration"},
    {"path": f"{SLIP_V2}/protocol.json", "access": "json", "purpose": "current development Slip semantics"},
    {"path": f"{SLIP_V2}/summary.json", "access": "json", "purpose": "current diagnostic fallback state"},
    {"path": f"{SLIP_V2}/input_allowlist.json", "access": "hash", "purpose": "prior artifact-access contract"},
    {"path": f"{SLIP_V2}/terrain_immutable_verification.json", "access": "hash", "purpose": "prior Terrain immutability evidence"},
    {"path": f"{JOINT}/summary.json", "access": "json", "purpose": "locked Terrain selection summary"},
    {"path": f"{JOINT}/provenance.json", "access": "json", "purpose": "locked Terrain provenance"},
    {"path": f"{JOINT}/readiness.json", "access": "json", "purpose": "locked Terrain readiness"},
    *({"path": value, "access": "hash", "purpose": f"locked Terrain {key} reference"} for key, value in TERRAIN_REFERENCES.items()),
    {"path": TERRAIN_FILES["model"], "access": "hash", "purpose": "immutable Terrain model"},
    {"path": TERRAIN_FILES["normalization"], "access": "hash", "purpose": "immutable Terrain normalization"},
    {"path": TERRAIN_FILES["config"], "access": "hash", "purpose": "immutable Terrain config"},
    {"path": TERRAIN_FILES["selection_lock"], "access": "json_and_hash", "purpose": "immutable Terrain selection lock"},
    {"path": "simulation/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx", "access": "hash_and_runtime", "purpose": "fixed walking policy"},
    {"path": "simulation/unitree_mujoco/unitree_robots/g1/scene_walking_terrain_transition.xml", "access": "hash_and_runtime", "purpose": "fixed MuJoCo scene"},
    {"path": "simulation/unitree_mujoco/unitree_robots/g1/g1_29dof.xml", "access": "hash_and_runtime", "purpose": "fixed MuJoCo robot include"},
    {"path": "simulation/unitree_mujoco/simulate_python/terrain_profiles.py", "access": "hash", "purpose": "existing material parameters"},
    {"path": "simulation/unitree_mujoco/simulate_python/walking_hazard_ground_truth_v1.py", "access": "hash", "purpose": "immutable physical signal derivation"},
    {"path": "simulation/unitree_mujoco/simulate_python/walking_hazard_oracle_calibration_v1.py", "access": "hash", "purpose": "immutable Slip oracle implementation"},
    {"path": "simulation/unitree_mujoco/simulate_python/bilateral_hil_sensor_v2.py", "access": "hash", "purpose": "bilateral Fusion20 schema"},
    {"path": "simulation/unitree_mujoco/simulate_python/g1_upstream_locomotion.py", "access": "hash", "purpose": "fixed controller implementation"},
    {"path": "simulation/unitree_mujoco/simulate_python/run_walking_bilateral_sensor_sink_observability_v2.py", "access": "hash", "purpose": "existing development acquisition implementation"},
    {"path": "simulation/unitree_mujoco/simulate_python/walking_bilateral_sink_observability_v2.py", "access": "hash", "purpose": "existing bilateral trace contract"},
    {"path": "simulation/unitree_mujoco/simulate_python/test/test_walking_bilateral_sensor_sink_observability_v2.py", "access": "hash", "purpose": "existing schema regression contract"},
    {"path": "simulation/unitree_mujoco/simulate_python/run_walking_v2_slip_redesign_iteration_v2.py", "access": "hash", "purpose": "prior Slip development runner"},
    {"path": "simulation/unitree_mujoco/simulate_python/walking_v2_slip_redesign_iteration_v2.py", "access": "hash", "purpose": "prior Slip state contract"},
    {"path": "simulation/unitree_mujoco/simulate_python/test_walking_v2_slip_redesign_iteration_v2.py", "access": "hash", "purpose": "prior Slip regression contract"},
    {"path": f"{RECOVERY}/protocol.json", "access": "hash", "purpose": "preserved first-attempt acquisition protocol"},
    {"path": f"{RECOVERY}/provenance_precheck.json", "access": "hash", "purpose": "preserved first-attempt precheck"},
    {"path": f"{RECOVERY}/run_manifest.csv", "access": "csv", "purpose": "first-attempt deterministic replay hashes"},
    {"path": f"{RECOVERY}/physical_episode_ledger.csv", "access": "hash", "purpose": "preserved first-attempt physical ledger"},
    {"path": f"{RECOVERY}/positive_source_audit.csv", "access": "hash", "purpose": "preserved first-attempt positive audit"},
    {"path": f"{RECOVERY}/control_source_audit.csv", "access": "hash", "purpose": "preserved first-attempt control audit"},
    {"path": f"{RECOVERY}/pair_parity_audit.csv", "access": "hash", "purpose": "preserved first-attempt parity audit"},
    {"path": f"{RECOVERY}/duplicate_audit.json", "access": "hash", "purpose": "preserved first-attempt duplicate audit"},
    {"path": f"{RECOVERY}/failure.json", "access": "json", "purpose": "explicit first-attempt failure record"},
)
FORBIDDEN_PATH_TOKENS = (
    "/outer/", "_outer_", "holdout", "spatial_final", "spatial-final",
    "final_test", "final-test",
)
TRACE_HASH_KEYS = (
    "time_s", "bilateral_canonical", "force_loaded", "contact_age",
    "physical_contact", "foot_world_xyz_label_only", "anchor_drift_label_only",
    "slip_physical_active", "pre_fall_valid", "contact_episode_id",
    "touchdown_transient", "patch_active",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class ArtifactAccessGuard:
    """Exact-path, fail-closed ledger for every pre-existing artifact read."""

    def __init__(
        self, repo: Path, allowed: list[str], ledger: Path, *, resume: bool = False
    ) -> None:
        self.repo = repo.resolve()
        self.allowed = set(allowed)
        self.ledger_path = ledger
        if resume:
            saved = json.loads(ledger.read_text(encoding="utf-8"))
            self.events = list(saved["events"])
            self.blocked = int(saved["blocked_access_count"])
        else:
            self.events: list[dict[str, object]] = []
            self.blocked = 0
            self._flush()

    def _flush(self) -> None:
        write_json(self.ledger_path, {
            "exact_paths_only": True,
            "forbidden_tokens": list(FORBIDDEN_PATH_TOKENS),
            "events": self.events,
            "blocked_access_count": self.blocked,
            "all_accesses_completed": all(row["status"] == "completed" for row in self.events),
        })

    def _resolve(self, relative: str, purpose: str, access: str) -> Path:
        normalized = relative.replace("\\", "/")
        blocked_token = next((token for token in FORBIDDEN_PATH_TOKENS if token in f"/{normalized.lower()}"), None)
        if normalized not in self.allowed or blocked_token is not None:
            self.blocked += 1
            self.events.append({
                "path": normalized, "purpose": purpose, "access": access,
                "status": "blocked", "forbidden_token": blocked_token or "not_allowlisted",
            })
            self._flush()
            raise PermissionError(f"artifact read blocked: {normalized}")
        path = (self.repo / normalized).resolve()
        if self.repo not in path.parents or not path.is_file():
            self.blocked += 1
            self.events.append({
                "path": normalized, "purpose": purpose, "access": access,
                "status": "blocked", "forbidden_token": "missing_or_outside_repo",
            })
            self._flush()
            raise FileNotFoundError(normalized)
        return path

    def _record(self, relative: str, purpose: str, access: str, digest: str) -> None:
        self.events.append({
            "path": relative, "purpose": purpose, "access": access,
            "status": "completed", "sha256": digest,
        })
        self._flush()

    def hash_input(self, relative: str, purpose: str) -> str:
        path = self._resolve(relative, purpose, "sha256")
        digest = sha256_file(path)
        self._record(relative, purpose, "sha256", digest)
        return digest

    def read_json(self, relative: str, purpose: str) -> object:
        path = self._resolve(relative, purpose, "json")
        digest = sha256_file(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        self._record(relative, purpose, "json", digest)
        return value

    def load_npz(self, relative: str, purpose: str) -> dict[str, np.ndarray]:
        path = self._resolve(relative, purpose, "npz")
        digest = sha256_file(path)
        with np.load(path, allow_pickle=False) as archive:
            value = {key: archive[key] for key in archive.files}
        self._record(relative, purpose, "npz", digest)
        return value

    def read_csv(self, relative: str, purpose: str) -> list[dict[str, str]]:
        path = self._resolve(relative, purpose, "csv")
        digest = sha256_file(path)
        with path.open(encoding="utf-8", newline="") as stream:
            value = list(csv.DictReader(stream))
        self._record(relative, purpose, "csv", digest)
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-postprocess", action="store_true")
    parser.add_argument("--repair-audit-postprocess", action="store_true")
    return parser.parse_args()


def _protocol(conditions: list[RunCondition], duration_s: float) -> dict[str, object]:
    profiles = material_profiles()
    return {
        "artifact": "walking_v2_bilateral_slip_targeted_acquisition_v3",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "scope": "MuJoCo digital-twin targeted bilateral Slip acquisition and audit only",
        "development_only": True,
        "training_selection_or_lock_performed": False,
        "blind_or_outer_or_final_access": False,
        "controller": {"revision": UPSTREAM_REVISION, "policy_sha256": TESTED_POLICY_SHA256},
        "planned_runs": 216,
        "planned_positive_runs": 108,
        "planned_control_runs": 108,
        "matched_pairs": 108,
        "sampling": {
            "rate_hz": SAMPLE_RATE_HZ,
            "physics_timestep_s": PHYSICS_TIMESTEP_S,
            "physics_steps_per_sample": PHYSICS_STEPS_PER_SAMPLE,
            "duration_s": duration_s,
            "exact_timestamp_formula": "sample k is (k+1)/1000 seconds",
        },
        "matrix": {
            "speeds_mps": list(SPEEDS_MPS), "target_feet": list(CONTRACT_SIDES),
            "target_onset_phases": [vars(value) for value in PHASE_BINS],
            "positive_severities": list(SEVERITIES),
            "variations": [vars(value) for value in VARIATIONS],
            "positive_cell_variations": 3,
            "control_assignment": "(severity_index + foot_index + phase_index + variation_index) modulo 2",
            "control_types": list(CONTROL_TYPES),
        },
        "materials": {key: vars(value) for key, value in profiles.items()},
        "moderate_profile_contract": {
            "open_ended_search": False, "pilot_used": False,
            "derivation": profiles["moderate_ice_preregistered"].derivation,
        },
        "localized_patch": {
            "geometry": {
                side: vars(next(row.patch for row in conditions if row.target_foot == side))
                for side in CONTRACT_SIDES
            },
            "height_preserving": True, "height_delta_m": 0.0,
            "no_penetration_or_compliance_change": True,
            "target_geom_only": True,
            "phase_activation": "physical target-foot contact age; identical within each pair",
            "friction_only_intervention": True,
        },
        "oracle": {
            "observable": "MuJoCo tangential contact-anchor drift (label/evaluation only)",
            "threshold_m": SLIP_THRESHOLD_M, "persistence_ms": SLIP_PERSISTENCE_MS,
            "semantics_changed": False,
            "runtime_detector_input": False,
            "post_fall_censored": True, "air_negative": True,
        },
        "pair_contract": {
            "same": [
                "initial policy state", "gait phase", "command timing", "speed",
                "target foot", "patch position and geometry", "policy observation timing",
                "random seed", "duration", "non-material simulation configuration",
            ],
            "only_difference": "surface/contact friction material profile",
        },
        "future_folds": {
            "count": 3, "grouped_by": ["pair ID", "simulation variation", "run", "contact episode"],
            "assignment": "source-qualified variation_index modulo 3",
            "training_in_this_task": False,
        },
        "forbidden_actions": [
            "model training/selection/lock", "blind holdout access/generation",
            "Terrain write/retrain", "production/System/INT8/Vela/E84/HIL write",
            "Sink runtime output", "failed attempt deletion/replacement/relabeling",
        ],
        "conditions": [{
            **{key: value for key, value in vars(row).items() if key != "patch"},
            "patch": vars(row.patch), "pair_fingerprint": row.pair_fingerprint,
        } for row in conditions],
    }


def _initial_policy_observation(controller: UnitreeG1PretrainedController) -> np.ndarray:
    model, data = controller.model, controller.data
    gyro_id = model.sensor("imu_gyro").id
    address = int(model.sensor_adr[gyro_id])
    dimension = int(model.sensor_dim[gyro_id])
    gyro = data.sensordata[address:address + dimension]
    return np.concatenate((
        gyro, gravity_orientation(data.qpos[3:7]), controller.command,
        (np.sin(2.0 * np.pi * controller.global_phase), np.cos(2.0 * np.pi * controller.global_phase)),
        data.qpos[7:] - DEFAULT_ANGLES, data.qvel[6:], controller.action,
    )).astype(np.float32)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _target_ground_contacts(
    data: mujoco.MjData, target_geoms: frozenset[int], grounds: frozenset[int]
) -> list[int]:
    result = []
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        pair = {int(contact.geom1), int(contact.geom2)}
        if pair & target_geoms and pair & grounds:
            result.append(contact_id)
    return result


def _apply_patch_friction(
    data: mujoco.MjData,
    contact_ids: list[int],
    condition: RunCondition,
    contact_age_ms: float,
    friction: np.ndarray,
) -> int:
    if contact_age_ms + 1e-12 < phase_spec(condition.target_phase).friction_activation_ms:
        return 0
    applied = 0
    for contact_id in contact_ids:
        contact = data.contact[contact_id]
        if condition.patch.contains(np.asarray(contact.pos)):
            contact.friction[:] = friction
            applied += 1
    return applied


def collect_run(
    condition: RunCondition, policy_path: Path
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    model.opt.timestep = PHYSICS_TIMESTEP_S
    ground_ids = frozenset(model.geom(name).id for name in GROUND_NAMES)
    for name in GROUND_NAMES:
        apply_terrain_profile(model, TERRAIN_PROFILES["concrete"], name)
    allowed_feet = _disable_nonfoot_surface_collisions(model, ground_ids)
    data = mujoco.MjData(model)
    controller = UnitreeG1PretrainedController(model, data, policy_path, condition.speed_mps)
    pose_dx, pose_dy, velocity_dx = deterministic_initial_perturbation(condition.seed)
    data.qpos[0] += pose_dx
    data.qpos[1] += pose_dy + condition.lateral_offset_m
    data.qvel[0] += velocity_dx
    controller.global_phase = condition.phase_fraction
    nominal_command = controller.command.copy()
    if condition.command_delay_s:
        controller.command[:] = 0.0
    mujoco.mj_forward(model, data)
    initial_observation = _initial_policy_observation(controller)
    reader = G1BilateralSensorReaderV2(model, data)
    target_geoms = frozenset(reader.foot_geom_ids[condition.target_foot])
    foot_body_ids = tuple(model.body(f"{side}_ankle_roll_link").id for side in SIDES)
    pelvis_id = model.body(PELVIS_BODY_NAME).id
    velocity = np.zeros(6, dtype=np.float64)
    profile = material_profiles()[condition.material_profile]
    contact_friction = friction_vector(profile)
    keys = (
        "time_s", "bilateral_canonical", "force_loaded", "contact_age",
        "physical_contact", "foot_world_xyz_label_only", "foot_world_velocity_label_only",
        "contact_penetration_label_only", "patch_active", "patch_contact_count",
    )
    series: dict[str, list[object]] = {key: [] for key in keys}
    first_fall_sample: int | None = None
    first_fall_time_s: float | None = None
    fall_reason = ""
    target_contact_age_steps = 0
    total_patch_contacts = 0
    total_steps = int(round(condition.duration_s / PHYSICS_TIMESTEP_S))
    for physics_step in range(1, total_steps + 1):
        if data.time + 1e-12 >= condition.command_delay_s:
            controller.command[:] = nominal_command
        controller.apply()
        mujoco.mj_step1(model, data)
        target_contacts = _target_ground_contacts(data, target_geoms, ground_ids)
        target_contact_age_steps = target_contact_age_steps + 1 if target_contacts else 0
        applied = _apply_patch_friction(
            data, target_contacts, condition,
            target_contact_age_steps * PHYSICS_TIMESTEP_S * 1000.0,
            contact_friction,
        )
        total_patch_contacts += applied
        mujoco.mj_step2(model, data)
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
        penetration = np.asarray([
            max_left_foot_contact_penetration_m(data, reader.foot_geom_ids[side], ground_ids)
            for side in SIDES
        ])
        reasons, nonfoot = _fall_reasons(model, data, pelvis_id, ground_ids, allowed_feet)
        if reasons and first_fall_sample is None:
            first_fall_sample = len(series["time_s"])
            first_fall_time_s = float(data.time)
            fall_reason = "|".join(reasons)
        values = {
            "time_s": float(data.time),
            "bilateral_canonical": canonical.astype(np.float32),
            "force_loaded": np.asarray([value.loaded for value in runtime_states]),
            "contact_age": np.asarray([value.age_samples for value in runtime_states], np.int32),
            "physical_contact": physical_contact,
            "foot_world_xyz_label_only": foot_xyz.astype(np.float32),
            "foot_world_velocity_label_only": np.asarray(foot_velocity, np.float32),
            "contact_penetration_label_only": penetration.astype(np.float32),
            "patch_active": bool(applied),
            "patch_contact_count": int(applied),
        }
        for key, value in values.items():
            series[key].append(value)
    trace = {key: np.asarray(value) for key, value in series.items()}
    sample_count = len(trace["time_s"])
    pre_fall = np.ones(sample_count, dtype=bool)
    if first_fall_sample is not None:
        pre_fall[first_fall_sample:] = False
    episode_id = np.full((sample_count, 2), -1, dtype=np.int32)
    transient = np.zeros((sample_count, 2), dtype=bool)
    anchor_drift = np.full((sample_count, 2), np.nan, dtype=np.float32)
    tangential_velocity = np.full((sample_count, 2), np.nan, dtype=np.float32)
    slip_valid = np.zeros((sample_count, 2), dtype=bool)
    gait_phase_code = np.zeros((sample_count, 2), dtype=np.int8)
    slip_active = np.zeros((sample_count, 2), dtype=bool)
    phase_codes = {"AIR": 0, "TOUCHDOWN": 1, "LOADING": 2, "MID_STANCE": 3, "PUSH_OFF": 4}
    for side_index in range(2):
        signals = derive_contact_signals(
            trace["physical_contact"][:, side_index],
            trace["force_loaded"][:, side_index],
            trace["foot_world_xyz_label_only"][:, side_index],
            trace["foot_world_velocity_label_only"][:, side_index],
            trace["contact_penetration_label_only"][:, side_index],
            first_fall_sample,
        )
        episode_id[:, side_index] = signals.contact_episode_id
        transient[:, side_index] = signals.touchdown_transient
        anchor_drift[:, side_index] = signals.tangential_anchor_drift_m
        tangential_velocity[:, side_index] = signals.tangential_velocity_mps
        slip_valid[:, side_index] = signals.slip_calibration_valid
        gait_phase_code[:, side_index] = np.asarray([
            phase_codes[value] for value in gait_phase(trace["force_loaded"][:, side_index])
        ], dtype=np.int8)
        slip_active[:, side_index] = persistent_oracle(
            signals.tangential_anchor_drift_m,
            signals.slip_calibration_valid,
            signals.contact_episode_id,
            SLIP_THRESHOLD_M,
            SLIP_PERSISTENCE_MS,
        )
    trace.update({
        "pre_fall_valid": pre_fall,
        "contact_episode_id": episode_id,
        "touchdown_transient": transient,
        "anchor_drift_label_only": anchor_drift,
        "tangential_velocity_label_only": tangential_velocity,
        "slip_calibration_valid_label_only": slip_valid,
        "gait_phase_code": gait_phase_code,
        "slip_physical_active": slip_active,
    })
    loaded_touchdowns = []
    for side in range(2):
        transitions = np.flatnonzero(trace["force_loaded"][:, side] & ~np.r_[False, trace["force_loaded"][:-1, side]])
        loaded_touchdowns.append(int(transitions[0]) if transitions.size else None)
    metadata: dict[str, object] = {
        **{key: value for key, value in vars(condition).items() if key != "patch"},
        "patch": vars(condition.patch),
        "pair_fingerprint": condition.pair_fingerprint,
        "initial_policy_observation_sha256": _array_sha256(initial_observation),
        "sample_count": sample_count,
        "first_sample_time_s": float(trace["time_s"][0]),
        "last_sample_time_s": float(trace["time_s"][-1]),
        "sample_spacing_max_error_s": float(np.max(np.abs(np.diff(trace["time_s"]) - 0.001))),
        "first_fall_sample": first_fall_sample,
        "first_fall_time_s": first_fall_time_s,
        "fall_occurred": first_fall_sample is not None,
        "fall_reason": fall_reason,
        "post_fall_excluded_sample_count": int(np.sum(~pre_fall)),
        "left_first_loaded_touchdown_sample": loaded_touchdowns[0],
        "right_first_loaded_touchdown_sample": loaded_touchdowns[1],
        "patch_contact_application_count": total_patch_contacts,
        "finite_fusion20": bool(np.all(np.isfinite(trace["bilateral_canonical"]))),
        "full_trace_sha256": full_trace_sha256(trace, TRACE_HASH_KEYS),
        "discarded": False,
        "replaced": False,
        "silently_relabelled": False,
    }
    return trace, metadata


def _active_intervals(mask: np.ndarray) -> str:
    return json.dumps([[start, end] for start, end in contact_episodes(mask)], separators=(",", ":"))


def build_ledgers(
    traces: list[dict[str, np.ndarray]], manifests: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    episode_rows: list[dict[str, object]] = []
    positive_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    fall_rows: list[dict[str, object]] = []
    for trace, meta in zip(traces, manifests):
        first_onsets: list[int | None] = []
        for side in range(2):
            candidates = np.flatnonzero(trace["slip_physical_active"][:, side] & trace["pre_fall_valid"])
            first_onsets.append(int(candidates[0]) if candidates.size else None)
        if first_onsets[0] is not None and first_onsets[1] is not None:
            if abs(first_onsets[0] - first_onsets[1]) <= 50:
                actual_first = "bilateral_ambiguous"
            else:
                actual_first = SIDES[int(first_onsets[1] < first_onsets[0])]
        elif first_onsets[0] is not None:
            actual_first = "left"
        elif first_onsets[1] is not None:
            actual_first = "right"
        else:
            actual_first = "none"
        target_side = SIDES.index(str(meta["target_foot"]))
        contra_side = 1 - target_side
        target_onset = first_onsets[target_side]
        contralateral = first_onsets[contra_side] is not None
        target_phase_actual = "none"
        target_age_ms: int | None = None
        target_episode: int | None = None
        target_touchdown: int | None = None
        target_loaded = trace["force_loaded"][:, target_side] & trace["pre_fall_valid"]
        target_transitions = np.flatnonzero(target_loaded & ~np.r_[False, target_loaded[:-1]])
        if target_transitions.size:
            target_touchdown = int(target_transitions[0])
            target_episode = int(trace["contact_episode_id"][target_touchdown, target_side])
        if target_onset is not None:
            target_episode = int(trace["contact_episode_id"][target_onset, target_side])
            episode = np.flatnonzero(trace["contact_episode_id"][:, target_side] == target_episode)
            loaded = episode[trace["force_loaded"][episode, target_side]]
            if loaded.size:
                target_touchdown = int(loaded[0])
                target_age_ms = target_onset - target_touchdown + 1
                target_phase_actual = onset_phase(target_age_ms)
        clean_unilateral = actual_first == meta["target_foot"] and not contralateral
        phase_match = target_phase_actual == meta["target_phase"]
        valid_target_positive = bool(
            meta["role"] == "positive" and target_onset is not None
            and clean_unilateral and phase_match
            and trace["pre_fall_valid"][target_onset]
            and trace["slip_calibration_valid_label_only"][target_onset, target_side]
            and not trace["touchdown_transient"][target_onset, target_side]
        )
        common = {
            "run_id": meta["run_id"], "pair_id": meta["pair_id"], "role": meta["role"],
            "speed_mps": meta["speed_mps"], "target_foot": meta["target_foot"],
            "target_phase": meta["target_phase"], "actual_onset_phase": target_phase_actual,
            "severity": meta["severity"], "material_profile": meta["material_profile"],
            "control_type": meta["control_type"], "variation_index": meta["variation_index"],
            "actual_first_affected_foot": actual_first,
            "target_touchdown_sample": "" if target_touchdown is None else target_touchdown,
            "target_touchdown_time_s": "" if target_touchdown is None else float(trace["time_s"][target_touchdown]),
            "target_physical_onset_sample": "" if target_onset is None else target_onset,
            "target_physical_onset_time_s": "" if target_onset is None else float(trace["time_s"][target_onset]),
            "touchdown_to_onset_ms": "" if target_age_ms is None else target_age_ms,
            "target_contact_episode_id": "" if target_episode is None else target_episode,
            "contralateral_physical_slip_contamination": contralateral,
            "valid_target_positive": valid_target_positive,
            "wrong_foot_onset": actual_first not in {"none", meta["target_foot"]},
            "bilateral_ambiguous": actual_first == "bilateral_ambiguous",
            "no_onset": target_onset is None,
            "fall_occurred": meta["fall_occurred"],
            "first_fall_sample": "" if meta["first_fall_sample"] is None else meta["first_fall_sample"],
            "fall_reason": meta["fall_reason"],
            "post_fall_excluded_sample_count": meta["post_fall_excluded_sample_count"],
            "discarded": False, "replaced": False, "silently_relabelled": False,
        }
        if meta["role"] == "positive":
            positive_rows.append(common)
        else:
            valid_onset_count = int(np.sum([
                value is not None for value in first_onsets
            ]))
            control_rows.append({
                **common,
                "physical_slip_foot_count": valid_onset_count,
                "physical_slip_onset_free": valid_onset_count == 0,
                "stable_loaded_contact_coverage": bool(np.any(trace["slip_calibration_valid_label_only"])),
                "valid_control_source": valid_onset_count == 0 and bool(np.any(trace["slip_calibration_valid_label_only"])),
            })
        fall_rows.append({
            "run_id": meta["run_id"], "pair_id": meta["pair_id"], "role": meta["role"],
            "fall_occurred": meta["fall_occurred"], "first_fall_sample": common["first_fall_sample"],
            "first_fall_time_s": "" if meta["first_fall_time_s"] is None else meta["first_fall_time_s"],
            "fall_reason": meta["fall_reason"],
            "pre_fall_valid_sample_count": int(np.sum(trace["pre_fall_valid"])),
            "post_fall_stored_sample_count": int(np.sum(~trace["pre_fall_valid"])),
            "post_fall_excluded_from_labels": True,
        })
        for side, side_name in enumerate(SIDES):
            for episode_id in sorted(set(trace["contact_episode_id"][:, side]) - {-1}):
                samples = np.flatnonzero(trace["contact_episode_id"][:, side] == episode_id)
                valid = samples[trace["pre_fall_valid"][samples]]
                loaded = samples[trace["force_loaded"][samples, side]]
                active = samples[trace["slip_physical_active"][samples, side] & trace["pre_fall_valid"][samples]]
                touchdown = int(loaded[0]) if loaded.size else None
                onset = int(active[0]) if active.size else None
                episode_rows.append({
                    "run_id": meta["run_id"], "pair_id": meta["pair_id"], "role": meta["role"],
                    "target_foot": meta["target_foot"], "foot": side_name,
                    "actual_first_affected_foot": actual_first,
                    "contact_episode_id": int(episode_id), "episode_start_sample": int(samples[0]),
                    "episode_end_sample_exclusive": int(samples[-1] + 1),
                    "touchdown_sample": "" if touchdown is None else touchdown,
                    "physical_onset_sample": "" if onset is None else onset,
                    "time_from_touchdown_to_onset_ms": "" if onset is None or touchdown is None else onset - touchdown + 1,
                    "active_physical_slip_intervals": _active_intervals(trace["slip_physical_active"][:, side] & (trace["contact_episode_id"][:, side] == episode_id)),
                    "maximum_contact_anchor_drift_m": float(np.nanmax(trace["anchor_drift_label_only"][valid, side])) if valid.size else "",
                    "stable_loaded_contact_valid": bool(np.any(trace["slip_calibration_valid_label_only"][samples, side])),
                    "fall_time_s": "" if meta["first_fall_time_s"] is None else meta["first_fall_time_s"],
                    "fall_reason": meta["fall_reason"],
                    "first_fall_censor_boundary": "" if meta["first_fall_sample"] is None else meta["first_fall_sample"],
                    "post_fall_excluded_sample_count": int(np.sum(~trace["pre_fall_valid"][samples])),
                    "air_mask_sample_count": int(np.sum(~trace["physical_contact"][samples, side])),
                    "touchdown_mask_sample_count": int(np.sum(trace["touchdown_transient"][samples, side])),
                    "invalid_mask_sample_count": int(np.sum(~trace["slip_calibration_valid_label_only"][samples, side])),
                    "contralateral_physical_slip_contamination": contralateral,
                })
    return episode_rows, positive_rows, control_rows, fall_rows


def pair_audits(
    traces: list[dict[str, np.ndarray]], manifests: list[dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_pair: dict[str, dict[str, int]] = {}
    for index, row in enumerate(manifests):
        by_pair.setdefault(str(row["pair_id"]), {})[str(row["role"])] = index
    pair_manifest: list[dict[str, object]] = []
    parity: list[dict[str, object]] = []
    for pair_id in sorted(by_pair):
        positive_index = by_pair[pair_id]["positive"]
        control_index = by_pair[pair_id]["control"]
        positive, control = manifests[positive_index], manifests[control_index]
        side = SIDES.index(str(positive["target_foot"]))
        touchdown_key = f"{SIDES[side]}_first_loaded_touchdown_sample"
        touchdown_equal = positive[touchdown_key] == control[touchdown_key]
        configuration_equal = positive["pair_fingerprint"] == control["pair_fingerprint"]
        initial_equal = positive["initial_policy_observation_sha256"] == control["initial_policy_observation_sha256"]
        time_equal = np.array_equal(traces[positive_index]["time_s"], traces[control_index]["time_s"])
        row = {
            "pair_id": pair_id, "positive_run_id": positive["run_id"], "control_run_id": control["run_id"],
            "pair_fingerprint": positive["pair_fingerprint"], "speed_mps": positive["speed_mps"],
            "target_foot": positive["target_foot"], "target_phase": positive["target_phase"],
            "severity": positive["severity"], "control_type": control["control_type"],
            "variation_index": positive["variation_index"], "seed": positive["seed"],
        }
        pair_manifest.append(row)
        parity.append({
            **row,
            "initial_policy_observation_equal": initial_equal,
            "command_timing_equal": positive["command_delay_s"] == control["command_delay_s"],
            "gait_phase_equal": positive["phase_fraction"] == control["phase_fraction"],
            "target_touchdown_equal": touchdown_equal,
            "speed_equal": positive["speed_mps"] == control["speed_mps"],
            "patch_geometry_equal": positive["patch"] == control["patch"],
            "seed_equal": positive["seed"] == control["seed"],
            "duration_equal": positive["duration_s"] == control["duration_s"],
            "timestamps_equal": time_equal,
            "non_material_configuration_equal": configuration_equal,
            "only_material_profile_differs": positive["material_profile"] != control["material_profile"],
            "parity_pass": bool(initial_equal and touchdown_equal and time_equal and configuration_equal),
        })
    return pair_manifest, parity


def _distribution_rows(
    positives: list[dict[str, object]], field: str, allowed: list[object]
) -> list[dict[str, object]]:
    return [{
        field: value,
        "planned_positive_runs": sum(row[field] == value for row in positives),
        "valid_target_positive_runs": sum(row[field] == value and bool(row["valid_target_positive"]) for row in positives),
    } for value in allowed]


def causal_pair_statistics(
    traces: list[dict[str, np.ndarray]], manifests: list[dict[str, object]],
    positives: list[dict[str, object]],
) -> list[dict[str, object]]:
    def finite_mean(value: np.ndarray) -> float | str:
        values = np.asarray(value, dtype=float)
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else ""

    index_by_run = {str(row["run_id"]): index for index, row in enumerate(manifests)}
    control_by_pair = {
        str(row["pair_id"]): str(row["run_id"]) for row in manifests if row["role"] == "control"
    }
    rows = []
    for audit in positives:
        pos_index = index_by_run[str(audit["run_id"])]
        ctl_index = index_by_run[control_by_pair[str(audit["pair_id"])]]
        positive, control = traces[pos_index], traces[ctl_index]
        side = SIDES.index(str(audit["target_foot"])); other = 1 - side
        onset_value = audit["target_physical_onset_sample"]
        if onset_value == "":
            touchdown = audit["target_touchdown_sample"]
            if touchdown == "":
                continue
            endpoint = min(len(positive["time_s"]) - 1, int(touchdown) + phase_spec(str(audit["target_phase"])).maximum_onset_ms)
        else:
            endpoint = int(onset_value)
        start = max(0, endpoint - 200)
        selected = slice(start, endpoint + 1)
        pos_b = positive["bilateral_canonical"][selected].reshape(-1, 2, 10)
        ctl_b = control["bilateral_canonical"][selected].reshape(-1, 2, 10)
        categories = {
            "target_foot_fsr": (pos_b[:, side, :4], ctl_b[:, side, :4]),
            "target_foot_imu": (pos_b[:, side, 4:], ctl_b[:, side, 4:]),
            "contralateral_fsr": (pos_b[:, other, :4], ctl_b[:, other, :4]),
            "contralateral_imu": (pos_b[:, other, 4:], ctl_b[:, other, 4:]),
            "bilateral_asymmetry": (pos_b[:, side] - pos_b[:, other], ctl_b[:, side] - ctl_b[:, other]),
            "contact_anchor_drift_label_only": (
                positive["anchor_drift_label_only"][selected, side],
                control["anchor_drift_label_only"][selected, side],
            ),
        }
        for category, (first, second) in categories.items():
            finite = np.isfinite(first) & np.isfinite(second)
            delta = np.abs(np.asarray(first, float) - np.asarray(second, float))[finite]
            rows.append({
                "pair_id": audit["pair_id"], "positive_run_id": audit["run_id"],
                "control_run_id": control_by_pair[str(audit["pair_id"])],
                "speed_mps": audit["speed_mps"], "target_foot": audit["target_foot"],
                "target_phase": audit["target_phase"], "severity": audit["severity"],
                "category": category, "history_start_sample": start,
                "history_end_sample_inclusive": endpoint, "future_sample_count": 0,
                "finite_comparison_values": int(delta.size),
                "mean_absolute_pair_difference": float(np.mean(delta)) if delta.size else "",
                "maximum_absolute_pair_difference": float(np.max(delta)) if delta.size else "",
                "positive_mean": finite_mean(first), "control_mean": finite_mean(second),
            })
    return rows


def future_fold_manifest(
    old_manifest: dict[str, object], old_arrays: list[dict[str, np.ndarray]],
    new_manifests: list[dict[str, object]], new_traces: list[dict[str, np.ndarray]],
) -> dict[str, object]:
    old_by_run: dict[str, tuple[dict[str, np.ndarray], int]] = {}
    for arrays in old_arrays:
        for index, run_id in enumerate(arrays["run_id"].astype(str)):
            old_by_run[str(run_id)] = (arrays, index)
    rows: list[dict[str, object]] = []
    for meta in old_manifest["runs"]:  # type: ignore[index]
        run_id = str(meta["run_id"])
        arrays, index = old_by_run[run_id]
        variation = int(meta["variation_index"])
        fold = assigned_future_fold("existing_development", variation)
        episodes = arrays["contact_episode_id"][index]
        for side, side_name in enumerate(SIDES):
            values = sorted(set(episodes[:, side].astype(int)) - {-1})
            for episode_value in values or [-1]:
                episode = int(episode_value)
                rows.append({
                    "source": "existing_development", "fold": fold, "pair_id": "",
                    "variation_group": f"existing_development:v{variation}", "run_id": run_id,
                    "episode_group": f"existing_development:{run_id}:{side_name}:{episode}",
                    "foot": side_name, "contact_episode_id": episode,
                    "speed_mps": meta["speed_mps"], "target_phase": "legacy_development",
                    "severity": "legacy_development", "role": meta["role"],
                })
    for meta, trace in zip(new_manifests, new_traces):
        variation = int(meta["variation_index"])
        fold = assigned_future_fold("targeted_acquisition_v3", variation)
        for side, side_name in enumerate(SIDES):
            values = sorted(set(trace["contact_episode_id"][:, side].astype(int)) - {-1})
            for episode_value in values or [-1]:
                episode = int(episode_value)
                rows.append({
                    "source": "targeted_acquisition_v3", "fold": fold,
                    "pair_id": meta["pair_id"],
                    "variation_group": f"targeted_acquisition_v3:v{variation}",
                    "run_id": meta["run_id"],
                    "episode_group": f"targeted_acquisition_v3:{meta['run_id']}:{side_name}:{episode}",
                    "foot": side_name, "contact_episode_id": episode,
                    "speed_mps": meta["speed_mps"], "target_phase": meta["target_phase"],
                    "severity": meta["severity"], "role": meta["role"],
                    "control_type": meta["control_type"],
                })
    audit = validate_future_fold_rows(rows)
    coverage = []
    for fold in range(3):
        selected = [row for row in rows if row["source"] == "targeted_acquisition_v3" and row["fold"] == fold]
        coverage.append({
            "fold": fold,
            "speeds": sorted({row["speed_mps"] for row in selected}),
            "feet": sorted({row["foot"] for row in selected}),
            "severities": sorted({row["severity"] for row in selected}),
            "roles": sorted({row["role"] for row in selected}),
            "control_types": sorted({row.get("control_type", "") for row in selected if row["role"] == "control"}),
            "onset_phase_targets": sorted({row["target_phase"] for row in selected}),
        })
    required = {
        "speeds": list(SPEEDS_MPS), "feet": list(SIDES), "severities": list(SEVERITIES),
        "roles": ["control", "positive"], "control_types": list(CONTROL_TYPES),
        "onset_phase_targets": sorted(value.name for value in PHASE_BINS),
    }
    coverage_pass = all(all(row[key] == sorted(value) for key, value in required.items()) for row in coverage)
    audit.update({
        "old_development_run_count": len(old_manifest["runs"]),  # type: ignore[index]
        "new_development_run_count": len(new_manifests),
        "all_data_development_only": True,
        "matched_pairs_same_fold": True,
        "fold_coverage": coverage,
        "fold_coverage_pass": coverage_pass,
        "valid": bool(audit["valid"] and coverage_pass),
    })
    return {"version": "walking_v2_future_nested_fold_contract_v3", "audit": audit, "rows": rows}


def _label_support(
    positives: list[dict[str, object]]
) -> dict[str, object]:
    valid = [row for row in positives if row["valid_target_positive"]]
    details = []
    for foot in SIDES:
        for speed in SPEEDS_MPS:
            selected = [row for row in valid if row["target_foot"] == foot and row["speed_mps"] == speed]
            ages = [int(row["touchdown_to_onset_ms"]) for row in selected]
            details.append({
                "foot": foot, "speed_mps": speed, "valid_event_count": len(selected),
                "actionable_0_100ms": any(age >= 1 for age in ages),
                "early_precursor_farther_than_100ms": any(age >= 102 for age in ages),
                "active_evidence": bool(selected),
            })
    return {"rows": details, "pass": all(
        row["actionable_0_100ms"] and row["early_precursor_farther_than_100ms"] and row["active_evidence"]
        for row in details
    )}


def create_plots(
    output: Path, traces: list[dict[str, np.ndarray]], manifests: list[dict[str, object]],
    positives: list[dict[str, object]], controls: list[dict[str, object]],
) -> list[str]:
    valid = next((row for row in positives if row["valid_target_positive"]), positives[0])
    by_run = {str(row["run_id"]): index for index, row in enumerate(manifests)}
    control_run = next(row["run_id"] for row in controls if row["pair_id"] == valid["pair_id"])
    pos = traces[by_run[str(valid["run_id"])]]; ctl = traces[by_run[str(control_run)]]
    side = SIDES.index(str(valid["target_foot"])); other = 1 - side
    has_physical_onset = valid["target_physical_onset_sample"] != ""
    if has_physical_onset:
        endpoint = int(valid["target_physical_onset_sample"])
    else:
        endpoint = min(
            len(pos["time_s"]) - 1,
            int(valid["target_touchdown_sample"])
            + phase_spec(str(valid["target_phase"])).maximum_onset_ms,
        )
    start = max(0, endpoint - 250); end = min(len(pos["time_s"]), endpoint + 251)
    x = (np.arange(start, end) - endpoint).astype(float)
    p = pos["bilateral_canonical"][start:end].reshape(-1, 2, 10)
    c = ctl["bilateral_canonical"][start:end].reshape(-1, 2, 10)
    paths = []

    def save(name: str) -> None:
        path = output / name
        if path.exists():
            plt.close(); paths.append(name); return
        plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(); paths.append(name)

    xlabel = "ms from physical onset" if has_physical_onset else "ms from preregistered reference endpoint (no onset)"
    plt.figure(figsize=(8, 4)); plt.plot(x, p[:, side, :4].sum(1), label="positive"); plt.plot(x, c[:, side, :4].sum(1), label="control"); plt.axvline(0, color="k", linestyle="--"); plt.xlabel(xlabel); plt.ylabel("target FSR sum (N)"); plt.legend(); save("matched_target_foot_fsr.png")
    plt.figure(figsize=(8, 4)); plt.plot(x, np.linalg.norm(p[:, side, 4:7], axis=1), label="positive accel"); plt.plot(x, np.linalg.norm(c[:, side, 4:7], axis=1), label="control accel"); plt.plot(x, np.linalg.norm(p[:, side, 7:], axis=1), label="positive gyro"); plt.plot(x, np.linalg.norm(c[:, side, 7:], axis=1), label="control gyro"); plt.axvline(0, color="k", linestyle="--"); plt.xlabel(xlabel); plt.ylabel("target IMU magnitude"); plt.legend(); save("matched_target_foot_imu.png")
    plt.figure(figsize=(8, 4)); plt.plot(x, p[:, side, :4].sum(1) - p[:, other, :4].sum(1), label="positive"); plt.plot(x, c[:, side, :4].sum(1) - c[:, other, :4].sum(1), label="control"); plt.axvline(0, color="k", linestyle="--"); plt.xlabel(xlabel); plt.ylabel("bilateral FSR asymmetry (N)"); plt.legend(); save("bilateral_asymmetry_around_onset.png")
    valid_rows = [row for row in positives if row["valid_target_positive"]]
    plt.figure(figsize=(8, 4));
    positions = np.arange(len(PHASE_BINS)); width = .35
    left = [sum(row["actual_onset_phase"] == phase.name and row["target_foot"] == "left" for row in valid_rows) for phase in PHASE_BINS]
    right = [sum(row["actual_onset_phase"] == phase.name and row["target_foot"] == "right" for row in valid_rows) for phase in PHASE_BINS]
    plt.bar(positions - width / 2, left, width, label="left"); plt.bar(positions + width / 2, right, width, label="right"); plt.xticks(positions, [value.name for value in PHASE_BINS], rotation=15); plt.ylabel("valid onset runs"); plt.legend(); save("onset_distribution_by_phase_and_foot.png")
    categories = list(SEVERITIES) + list(CONTROL_TYPES)
    values = []
    for category in categories:
        maxima = []
        for trace, meta in zip(traces, manifests):
            if meta["material_profile"] == category:
                maxima.append(float(np.nanmax(trace["anchor_drift_label_only"])))
        values.append(maxima)
    plt.figure(figsize=(9, 4)); plt.boxplot(values, labels=categories); plt.xticks(rotation=15); plt.ylabel("max anchor drift (m)"); save("anchor_drift_by_profile.png")
    return paths


def _gate_summary(
    manifests: list[dict[str, object]], traces: list[dict[str, np.ndarray]],
    positives: list[dict[str, object]], controls: list[dict[str, object]],
    parity: list[dict[str, object]], terrain_same: bool, guard: ArtifactAccessGuard,
    folds: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    expected_samples = int(round(manifests[0]["duration_s"] * SAMPLE_RATE_HZ))
    schema = bool(
        len(manifests) == 216
        and len({row["run_id"] for row in manifests}) == 216
        and len({row["pair_id"] for row in manifests}) == 108
        and all(int(row["sample_count"]) == expected_samples for row in manifests)
        and all(float(row["sample_spacing_max_error_s"]) <= 1e-12 for row in manifests)
        and all(trace["bilateral_canonical"].shape == (expected_samples, 20) for trace in traces)
        and all(bool(row["finite_fusion20"]) for row in manifests)
    )
    duplicate_count = len(manifests) - len({str(row["full_trace_sha256"]) for row in manifests})
    parity_pass = all(bool(row["parity_pass"]) for row in parity)
    valid = [row for row in positives if row["valid_target_positive"]]
    cells = {(row["speed_mps"], row["target_foot"], row["target_phase"], row["severity"]) for row in valid}
    all_cells = len(cells) == 36
    left = sum(row["target_foot"] == "left" for row in valid)
    right = sum(row["target_foot"] == "right" for row in valid)
    ratio = left / right if right else (1.0 if left == 0 else 999.0)
    each_speed = {f"{speed:.2f}": sum(row["speed_mps"] == speed for row in valid) for speed in SPEEDS_MPS}
    positive_source = bool(
        all_cells and len(valid) >= 0.8 * 108 and 0.80 <= ratio <= 1.25
        and all(value >= 24 for value in each_speed.values())
        and all(row["target_foot"] in SIDES for row in valid)
    )
    control_source = bool(
        all(bool(row["physical_slip_onset_free"]) and bool(row["stable_loaded_contact_coverage"]) for row in controls)
        and {row["control_type"] for row in controls} == set(CONTROL_TYPES)
        and {row["speed_mps"] for row in controls} == set(SPEEDS_MPS)
        and {row["target_foot"] for row in controls} == set(SIDES)
    )
    phase_counts = {phase.name: sum(row["actual_onset_phase"] == phase.name for row in valid) for phase in PHASE_BINS}
    timing_distribution = bool(
        all(value > 0 for value in phase_counts.values())
        and max(phase_counts.values(), default=0) <= 0.5 * max(1, len(valid))
    )
    label_support = _label_support(positives)
    onset_diversity = bool(timing_distribution and label_support["pass"])
    affected_balance = bool(left > 0 and right > 0 and 0.80 <= ratio <= 1.25)
    masks_valid = all(
        not np.any(trace["slip_physical_active"] & ~trace["pre_fall_valid"][:, None])
        and not np.any(trace["slip_physical_active"] & ~trace["slip_calibration_valid_label_only"])
        for trace in traces
    )
    integrity = bool(
        schema and duplicate_count == 0 and parity_pass and masks_valid
        and terrain_same and guard.blocked == 0
    )
    data_ready = bool(integrity and positive_source and control_source and onset_diversity and affected_balance)
    fold_ready = bool(folds["audit"]["valid"])
    retraining = bool(data_ready and fold_ready)
    gates = {
        "schema_integrity": integrity, "exact_1khz_fusion20": schema,
        "duplicate_full_trace_hash_count": duplicate_count,
        "pair_configuration_parity": parity_pass, "masks_valid": masks_valid,
        "terrain_byte_identical": terrain_same, "forbidden_artifact_access_count": guard.blocked,
        "positive_source": positive_source, "valid_target_positive_runs": len(valid),
        "valid_positive_fraction": len(valid) / 108.0, "all_36_positive_cells_covered": all_cells,
        "left_valid_positive_runs": left, "right_valid_positive_runs": right,
        "left_right_valid_ratio": ratio, "valid_positive_by_speed": each_speed,
        "control_source": control_source,
        "control_physical_onset_run_count": sum(not bool(row["physical_slip_onset_free"]) for row in controls),
        "timing_distribution": timing_distribution, "valid_positive_by_onset_phase": phase_counts,
        "label_support": label_support, "onset_diversity": onset_diversity,
        "affected_foot_balance": affected_balance, "future_fold": fold_ready,
        "acquisition_data": data_ready, "targeted_retraining_authorized": retraining,
    }
    readiness = {
        "WALKING_V2_SLIP_ACQUISITION_PROTOCOL_READY": True,
        "WALKING_V2_SLIP_ACQUISITION_PROVENANCE_READY": terrain_same and guard.blocked == 0,
        "WALKING_V2_SLIP_BILATERAL_SCHEMA_READY": schema,
        "WALKING_V2_SLIP_COUNTERFACTUAL_PARITY_READY": parity_pass,
        "WALKING_V2_SLIP_POSITIVE_SOURCE_READY": positive_source,
        "WALKING_V2_SLIP_CONTROL_SOURCE_READY": control_source,
        "WALKING_V2_SLIP_ONSET_DIVERSITY_READY": onset_diversity,
        "WALKING_V2_SLIP_AFFECTED_FOOT_BALANCE_READY": affected_balance,
        "WALKING_V2_SLIP_ACQUISITION_DATA_READY": data_ready,
        "WALKING_V2_SLIP_FUTURE_FOLD_READY": fold_ready,
        "WALKING_V2_SLIP_TARGETED_RETRAINING_AUTHORIZED": retraining,
        "WALKING_V2_FRESH_BLIND_HOLDOUT_AUTHORIZED": False,
        "WALKING_V2_SYSTEM_MIGRATION_AUTHORIZED": False,
        "WALKING_V2_INT8_PREPARATION_AUTHORIZED": False,
        "WALKING_V2_TERRAIN_LOCK_PRESERVED": terrain_same,
        "SINK_RUNTIME_DETECTION_DEFERRED": True,
    }
    return gates, readiness


def _audit_text(summary: dict[str, object]) -> str:
    gates = summary["gates"]
    next_step = summary["next_step"]
    return f"""# Targeted Bilateral Slip Development Acquisition v3

All generated runs are development-only. No model was trained, selected, exported, or locked.

1. Planned/unique executed corpus runs: {summary['planned_runs']}/{summary['executed_runs']}; simulator invocations {summary['simulator_invocation_count']} including {summary['deterministic_recovery_replay_count']} exact recovery replays.
2. Valid target-foot physical positive runs: {gates['valid_target_positive_runs']}/108.
3. Every speed x foot x phase x severity cell valid: {gates['all_36_positive_cells_covered']}.
4. Left/right valid positives balanced: {gates['affected_foot_balance']} ({gates['left_valid_positive_runs']}/{gates['right_valid_positive_runs']}, ratio {gates['left_right_valid_ratio']}).
5. Every control free of physical Slip onset: {gates['control_source']} (onset runs {gates['control_physical_onset_run_count']}).
6. Matched-pair configuration parity: {gates['pair_configuration_parity']}.
7. Strong and moderate physical source ready: {summary['strong_and_moderate_physically_valid']}.
8. Actionable, early-precursor and active-evidence support for both feet/all speeds: {gates['label_support']['pass']}.
9. Discarded/replaced/silently relabelled runs: 0/0/0.
10. Forbidden outer/holdout/final artifact accesses: {gates['forbidden_artifact_access_count']}.
11. Terrain byte-identical: {gates['terrain_byte_identical']}.
12. Targeted Slip retraining authorized: {gates['targeted_retraining_authorized']}.
13. Fresh blind holdout, System migration and INT8 remain unauthorized: True.
14. Sink remains `SINK_RUNTIME_DETECTION_DEFERRED`; no Sink runtime artifact exists.

Next step: `{next_step}`
"""


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if str(value) not in {"True", "False"}:
        raise ValueError(f"not a serialized bool: {value!r}")
    return str(value) == "True"


def repair_audit_postprocess(args: argparse.Namespace) -> None:
    """Rebuild derived ledgers from the immutable trace archive, without simulation."""
    output = args.output_dir.resolve()
    with np.load(output / "traces.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    run_ids = arrays.pop("run_id").astype(str)
    traces = [
        {key: value[index] for key, value in arrays.items()}
        for index in range(len(run_ids))
    ]
    conditions = {row.run_id: row for row in acquisition_matrix(args.duration_s)}
    dynamic = {row["run_id"]: row for row in _csv_rows(output / "run_manifest.csv")}
    manifests: list[dict[str, object]] = []
    for run_id in run_ids:
        condition = conditions[run_id]
        row = dynamic[run_id]
        manifests.append({
            **{key: value for key, value in vars(condition).items() if key != "patch"},
            "patch": vars(condition.patch),
            "first_fall_sample": None if row["first_fall_sample"] == "" else int(row["first_fall_sample"]),
            "first_fall_time_s": None if row["first_fall_time_s"] == "" else float(row["first_fall_time_s"]),
            "fall_occurred": _as_bool(row["fall_occurred"]),
            "fall_reason": row["fall_reason"],
            "post_fall_excluded_sample_count": int(row["post_fall_excluded_sample_count"]),
        })
    episode_rows, positive_rows, control_rows, fall_rows = build_ledgers(traces, manifests)
    causal_rows = causal_pair_statistics(traces, manifests, positive_rows)
    write_csv(output / "physical_episode_ledger.csv", episode_rows)
    write_csv(output / "positive_source_audit.csv", positive_rows)
    write_csv(output / "control_source_audit.csv", control_rows)
    write_csv(output / "fall_censor_audit.csv", fall_rows)
    write_csv(output / "causal_feature_pair_statistics.csv", causal_rows)
    write_csv(output / "onset_phase_distribution.csv", _distribution_rows(
        positive_rows, "actual_onset_phase",
        [value.name for value in PHASE_BINS] + ["out_of_contract", "none"],
    ))
    write_csv(output / "affected_foot_distribution.csv", _distribution_rows(
        positive_rows, "target_foot", list(SIDES),
    ))
    write_csv(output / "speed_distribution.csv", _distribution_rows(
        positive_rows, "speed_mps", list(SPEEDS_MPS),
    ))
    write_json(output / "audit_repair.json", {
        "simulation_run_count": 0,
        "source_trace_archive_sha256": sha256_file(output / "traces.npz"),
        "reason": "populate touchdown-relative causal histories for preserved no-onset attempts",
        "positive_rows": len(positive_rows), "control_rows": len(control_rows),
        "causal_pair_statistic_rows": len(causal_rows),
        "original_derived_outputs_preserved_in": "postprocessing_pre_audit_repair",
    })


def resume_postprocess(args: argparse.Namespace) -> dict[str, object]:
    """Finish the interrupted postprocess without executing another simulation."""
    output = args.output_dir.resolve()
    required = (
        "traces.npz", "run_manifest.csv", "positive_source_audit.csv",
        "control_source_audit.csv", "pair_parity_audit.csv",
        "future_nested_fold_manifest.json", "artifact_access_log.json",
        "terrain_immutable_verification.json", "oracle_immutable_verification.json",
        "postprocessing_failure_002.json",
    )
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"postprocess recovery inputs missing: {missing}")
    forbidden_overwrites = (
        "summary.json", "readiness.json", "acquisition_manifest.json",
        "provenance.json", "audit.md",
    )
    existing = [name for name in forbidden_overwrites if (output / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite completed postprocess outputs: {existing}")
    started = time.perf_counter()
    with np.load(output / "traces.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    run_ids = arrays.pop("run_id").astype(str)
    traces = [
        {key: value[index] for key, value in arrays.items()}
        for index in range(len(run_ids))
    ]
    condition_by_run = {row.run_id: row for row in acquisition_matrix(args.duration_s)}
    dynamic_by_run = {row["run_id"]: row for row in _csv_rows(output / "run_manifest.csv")}
    if set(run_ids) != set(condition_by_run) or set(run_ids) != set(dynamic_by_run):
        raise RuntimeError("recovery run IDs do not match the frozen acquisition matrix")
    manifests: list[dict[str, object]] = []
    for run_id in run_ids:
        condition = condition_by_run[run_id]
        dynamic = dynamic_by_run[run_id]
        manifests.append({
            **{key: value for key, value in vars(condition).items() if key != "patch"},
            "patch": vars(condition.patch),
            "pair_fingerprint": condition.pair_fingerprint,
            "initial_policy_observation_sha256": dynamic["initial_policy_observation_sha256"],
            "sample_count": int(dynamic["sample_count"]),
            "sample_spacing_max_error_s": float(dynamic["sample_spacing_max_error_s"]),
            "first_fall_sample": None if dynamic["first_fall_sample"] == "" else int(dynamic["first_fall_sample"]),
            "first_fall_time_s": None if dynamic["first_fall_time_s"] == "" else float(dynamic["first_fall_time_s"]),
            "fall_occurred": _as_bool(dynamic["fall_occurred"]),
            "fall_reason": dynamic["fall_reason"],
            "post_fall_excluded_sample_count": int(dynamic["post_fall_excluded_sample_count"]),
            "left_first_loaded_touchdown_sample": int(dynamic["left_first_loaded_touchdown_sample"]),
            "right_first_loaded_touchdown_sample": int(dynamic["right_first_loaded_touchdown_sample"]),
            "patch_contact_application_count": int(dynamic["patch_contact_application_count"]),
            "finite_fusion20": _as_bool(dynamic["finite_fusion20"]),
            "full_trace_sha256": dynamic["full_trace_sha256"],
            "discarded": _as_bool(dynamic["discarded"]),
            "replaced": _as_bool(dynamic["replaced"]),
            "silently_relabelled": _as_bool(dynamic["silently_relabelled"]),
        })
    positive_rows: list[dict[str, object]] = []
    for row in _csv_rows(output / "positive_source_audit.csv"):
        positive_rows.append({
            **row, "speed_mps": float(row["speed_mps"]),
            "variation_index": int(row["variation_index"]),
            **{key: _as_bool(row[key]) for key in (
                "contralateral_physical_slip_contamination", "valid_target_positive",
                "wrong_foot_onset", "bilateral_ambiguous", "no_onset", "fall_occurred",
                "discarded", "replaced", "silently_relabelled",
            )},
        })
    control_rows: list[dict[str, object]] = []
    for row in _csv_rows(output / "control_source_audit.csv"):
        control_rows.append({
            **row, "speed_mps": float(row["speed_mps"]),
            "variation_index": int(row["variation_index"]),
            "physical_slip_foot_count": int(row["physical_slip_foot_count"]),
            **{key: _as_bool(row[key]) for key in (
                "contralateral_physical_slip_contamination", "valid_target_positive",
                "wrong_foot_onset", "bilateral_ambiguous", "no_onset", "fall_occurred",
                "discarded", "replaced", "silently_relabelled", "physical_slip_onset_free",
                "stable_loaded_contact_coverage", "valid_control_source",
            )},
        })
    parity: list[dict[str, object]] = []
    for row in _csv_rows(output / "pair_parity_audit.csv"):
        parity.append({**row, **{key: _as_bool(row[key]) for key in (
            "initial_policy_observation_equal", "command_timing_equal", "gait_phase_equal",
            "target_touchdown_equal", "speed_equal", "patch_geometry_equal", "seed_equal",
            "duration_equal", "timestamps_equal", "non_material_configuration_equal",
            "only_material_profile_differs", "parity_pass",
        )}})
    fold_manifest = json.loads((output / "future_nested_fold_manifest.json").read_text(encoding="utf-8"))
    plots = create_plots(output, traces, manifests, positive_rows, control_rows)
    guard = ArtifactAccessGuard(
        REPO, [row["path"] for row in INPUTS], output / "artifact_access_log.json",
        resume=True,
    )
    terrain_verification = json.loads((output / "terrain_immutable_verification.json").read_text(encoding="utf-8"))
    terrain_initial = terrain_verification["initial_sha256"]
    terrain_final = {
        key: guard.hash_input(path, f"resumed final immutable Terrain {key}")
        for key, path in TERRAIN_FILES.items()
    }
    terrain_same = terrain_final == terrain_initial
    terrain_verification.update({
        "final_sha256": terrain_final, "byte_identical_before_after": terrain_same,
        "verified_after_acquisition": True, "postprocess_resumed_without_simulation": True,
    })
    write_json(output / "terrain_immutable_verification.json", terrain_verification)
    oracle_verification = json.loads((output / "oracle_immutable_verification.json").read_text(encoding="utf-8"))
    oracle_initial = oracle_verification["initial_sha256"]
    oracle_final = {
        "operational_label_summary": guard.hash_input(INPUTS[3]["path"], "resumed final operational-label hash"),
        "oracle_calibration_summary": guard.hash_input(INPUTS[4]["path"], "resumed final physical-oracle calibration hash"),
        "ground_truth_source": guard.hash_input("simulation/unitree_mujoco/simulate_python/walking_hazard_ground_truth_v1.py", "resumed final physical signal source hash"),
        "oracle_source": guard.hash_input("simulation/unitree_mujoco/simulate_python/walking_hazard_oracle_calibration_v1.py", "resumed final Slip oracle source hash"),
    }
    oracle_same = oracle_final == oracle_initial
    oracle_verification.update({
        "final_sha256": oracle_final, "byte_identical_before_after": oracle_same,
        "verified_after_acquisition": True, "postprocess_resumed_without_simulation": True,
    })
    write_json(output / "oracle_immutable_verification.json", oracle_verification)
    if not terrain_same or not oracle_same:
        raise RuntimeError("immutable Terrain/oracle input changed during postprocess recovery")
    gates, readiness = _gate_summary(
        manifests, traces, positive_rows, control_rows, parity,
        terrain_same, guard, fold_manifest,
    )
    write_json(output / "readiness.json", readiness)
    write_json(output / "acquisition_manifest.json", {
        "artifact": "walking_v2_bilateral_slip_targeted_acquisition_v3",
        "development_only": True, "planned_runs": 216, "executed_runs": 216,
        "positive_runs": 108, "control_runs": 108, "pair_count": 108,
        "all_attempts_preserved": True, "discarded_run_count": 0,
        "replaced_run_count": 0, "silently_relabelled_run_count": 0,
        "pilot_run_count": 0, "model_training_count": 0, "selection_lock_created": False,
        "simulator_invocation_count": 432, "deterministic_recovery_replay_count": 216,
        "first_attempt_archive": RECOVERY,
        "first_attempt_raw_traces_reconstructed_exactly": True,
        "recovery_full_trace_hash_mismatch_count": 0,
        "postprocess_resume_simulation_run_count": 0,
        "trace_archive": "traces.npz", "trace_archive_sha256": sha256_file(output / "traces.npz"),
    })
    strong_valid = any(row["valid_target_positive"] and row["severity"] == SEVERITIES[0] for row in positive_rows)
    moderate_valid = any(row["valid_target_positive"] and row["severity"] == SEVERITIES[1] for row in positive_rows)
    if readiness["WALKING_V2_SLIP_TARGETED_RETRAINING_AUTHORIZED"]:
        next_step = "SLIP_TARGETED_RETRAINING_V3"
    elif not strong_valid or not moderate_valid or sum(
        int(row["patch_contact_application_count"]) == 0
        for row in manifests if row["role"] == "positive"
    ):
        next_step = "SLIP_SCENARIO_GENERATOR_REDESIGN"
    else:
        next_step = "ADDITIONAL_TARGETED_SLIP_ACQUISITION"
    summary: dict[str, object] = {
        "artifact": "walking_v2_bilateral_slip_targeted_acquisition_v3",
        "starting_checkpoint": STARTING_CHECKPOINT, "development_only": True,
        "planned_runs": 216, "executed_runs": 216,
        "simulator_invocation_count": 432, "deterministic_recovery_replay_count": 216,
        "first_attempt_archive": RECOVERY,
        "first_attempt_raw_traces_reconstructed_exactly": True,
        "recovery_full_trace_hash_mismatch_count": 0,
        "postprocess_resume_simulation_run_count": 0,
        "planned_positive_runs": 108, "executed_positive_runs": len(positive_rows),
        "planned_control_runs": 108, "executed_control_runs": len(control_rows),
        "all_attempts_preserved": True, "model_training_count": 0,
        "slip_model_or_selection_lock_created": False,
        "blind_artifact_accessed": False, "blind_artifact_generated": False,
        "strong_profile_physically_valid": strong_valid,
        "moderate_profile_physically_valid": moderate_valid,
        "strong_and_moderate_physically_valid": strong_valid and moderate_valid,
        "gates": gates, "readiness": readiness, "plots": plots,
        "postprocessing_failures_preserved": [
            f"{RECOVERY}/failure.json", "postprocessing_failure_002.json",
        ],
        "next_step": next_step,
        "postprocess_recovery_wall_time_seconds": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    core_files = sorted(
        path for path in output.rglob("*") if path.is_file()
        and path.relative_to(output).as_posix() not in {"provenance.json", "audit.md", "summary.json"}
    )
    recovery_hashes = {
        row["path"]: row["sha256"] for row in guard.events
        if row["status"] == "completed" and str(row["path"]).startswith(f"{RECOVERY}/")
    }
    write_json(output / "provenance.json", {
        "artifact": "walking_v2_bilateral_slip_targeted_acquisition_v3",
        "protocol_sha256": sha256_file(output / "protocol.json"),
        "starting_checkpoint": STARTING_CHECKPOINT,
        "generated_files": [{
            "path": path.relative_to(output).as_posix(),
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
        } for path in core_files],
        "hash_graph_complete": True,
        "excluded_nodes": ["provenance.json (self)", "summary.json (points to provenance)", "audit.md (points to summary)"],
        "forbidden_artifact_access_count": guard.blocked,
        "terrain_byte_identical": terrain_same, "oracle_byte_identical": oracle_same,
        "first_attempt_archive": RECOVERY,
        "first_attempt_recovery_sha256": recovery_hashes,
        "first_attempt_archive_files": [{
            "path": path.relative_to(REPO).as_posix(),
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
        } for path in sorted((REPO / RECOVERY).iterdir()) if path.is_file()],
        "postprocess_resumed_without_simulation": True,
    })
    summary["provenance_sha256"] = sha256_file(output / "provenance.json")
    write_json(output / "summary.json", summary)
    audit = _audit_text(summary)
    audit += f"\nSummary SHA256: `{sha256_file(output / 'summary.json')}`\n"
    (output / "audit.md").write_text(audit, encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.execute:
        raise RuntimeError("acquisition requires explicit --execute")
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=REPO, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if head != STARTING_CHECKPOINT:
        raise RuntimeError(f"unexpected starting checkpoint: {head}")
    if args.duration_s != 3.0:
        raise ValueError("the full preregistered acquisition duration is fixed at 3.0 seconds")
    if args.repair_audit_postprocess:
        repair_audit_postprocess(args)
        return resume_postprocess(args)
    if args.resume_postprocess:
        return resume_postprocess(args)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite an existing generated dataset")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    conditions = acquisition_matrix(args.duration_s)
    protocol = _protocol(conditions, args.duration_s)
    allowlist = {"version": "walking_v2_slip_acquisition_v3_allowlist", "exact_paths_only": True, "inputs": list(INPUTS)}
    forbidden = {
        "version": "walking_v2_slip_acquisition_v3_forbidden_policy",
        "fail_closed": True, "tokens": list(FORBIDDEN_PATH_TOKENS),
        "forbidden_namespaces": ["outer", "holdout", "spatial-final", "final-test"],
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "input_allowlist.json", allowlist)
    write_json(output / "forbidden_path_policy.json", forbidden)
    write_json(output / "material_profiles.json", {
        "profiles": {key: vars(value) for key, value in material_profiles().items()},
        "positive_profile_count_exact": 2, "control_profile_count_exact": 2,
        "moderate_source_pilot_used": False, "open_ended_search_used": False,
    })
    write_json(output / "patch_geometry.json", {
        "height_preserving": True, "height_delta_m": 0.0,
        "friction_only": True, "penetration_or_compliance_change": False,
        "patches": {side: vars(next(row.patch for row in conditions if row.target_foot == side)) for side in SIDES},
        "target_foot_geom_mapping": "only named target-foot sole contacts receive the pair material",
        "phase_activation_ms": {value.name: value.friction_activation_ms for value in PHASE_BINS},
    })
    protocol_sha = sha256_file(output / "protocol.json")
    guard = ArtifactAccessGuard(REPO, [row["path"] for row in INPUTS], output / "artifact_access_log.json")
    old_manifest = guard.read_json(INPUTS[0]["path"], INPUTS[0]["purpose"])
    old_train = guard.load_npz(INPUTS[1]["path"], INPUTS[1]["purpose"])
    old_validation = guard.load_npz(INPUTS[2]["path"], INPUTS[2]["purpose"])
    operational = guard.read_json(INPUTS[3]["path"], INPUTS[3]["purpose"])
    oracle_summary = guard.read_json(INPUTS[4]["path"], INPUTS[4]["purpose"])
    slip_protocol = guard.read_json(INPUTS[5]["path"], INPUTS[5]["purpose"])
    slip_summary = guard.read_json(INPUTS[6]["path"], INPUTS[6]["purpose"])
    prior_hashes = {row["path"]: guard.hash_input(row["path"], row["purpose"]) for row in INPUTS[7:9]}
    joint_summary = guard.read_json(INPUTS[9]["path"], INPUTS[9]["purpose"])
    joint_provenance = guard.read_json(INPUTS[10]["path"], INPUTS[10]["purpose"])
    joint_readiness = guard.read_json(INPUTS[11]["path"], INPUTS[11]["purpose"])
    reference_hashes = {row["path"]: guard.hash_input(row["path"], row["purpose"]) for row in INPUTS[12:17]}
    terrain_initial = {
        key: guard.hash_input(path, f"initial immutable Terrain {key}")
        for key, path in TERRAIN_FILES.items() if key != "selection_lock"
    }
    terrain_lock = guard.read_json(TERRAIN_FILES["selection_lock"], "verify immutable Terrain selection lock")
    terrain_initial["selection_lock"] = guard.hash_input(TERRAIN_FILES["selection_lock"], "initial immutable Terrain selection-lock hash")
    terrain_valid = bool(
        terrain_lock["model_sha256"] == terrain_initial["model"]
        and terrain_lock["normalization_sha256"] == terrain_initial["normalization"]
        and terrain_lock["config_sha256"] == terrain_initial["config"]
        and terrain_lock["protocol_sha256"] == reference_hashes[TERRAIN_REFERENCES["protocol"]]
        and terrain_lock["data_manifest_sha256"] == reference_hashes[TERRAIN_REFERENCES["data_manifest"]]
        and terrain_lock["split_manifest_sha256"] == reference_hashes[TERRAIN_REFERENCES["split_manifest"]]
        and terrain_lock["validation_metrics_sha256"] == reference_hashes[TERRAIN_REFERENCES["validation_metrics"]]
        and terrain_lock["resource_report_sha256"] == reference_hashes[TERRAIN_REFERENCES["resource_report"]]
        and joint_summary.get("terrain", {}).get("selected_or_fallback") == "T2_seed_202608211"
    )
    if not terrain_valid:
        raise RuntimeError("Terrain selection lock verification failed before acquisition")
    other_hashes = {row["path"]: guard.hash_input(row["path"], row["purpose"]) for row in INPUTS[21:]}
    policy_hash = guard.hash_input(INPUTS[21]["path"], INPUTS[21]["purpose"])
    scene_hash = guard.hash_input(INPUTS[22]["path"], INPUTS[22]["purpose"])
    robot_hash = guard.hash_input(INPUTS[23]["path"], INPUTS[23]["purpose"])
    if policy_hash != TESTED_POLICY_SHA256:
        raise RuntimeError("walking policy hash mismatch")
    recovery_rows = guard.read_csv(
        f"{RECOVERY}/run_manifest.csv", "load first-attempt trace hashes for deterministic recovery"
    )
    recovery_failure = guard.read_json(
        f"{RECOVERY}/failure.json", "verify explicit first-attempt failure record"
    )
    recovery_hash_by_run = {
        row["run_id"]: row["full_trace_sha256"] for row in recovery_rows
    }
    if (
        len(recovery_rows) != 216 or len(recovery_hash_by_run) != 216
        or recovery_failure.get("generated_simulation_run_count") != 216
    ):
        raise RuntimeError("first-attempt recovery evidence is incomplete")
    oracle_initial = {
        "operational_label_summary": guard.hash_input(INPUTS[3]["path"], "initial operational-label hash"),
        "oracle_calibration_summary": guard.hash_input(INPUTS[4]["path"], "initial physical-oracle calibration hash"),
        "ground_truth_source": other_hashes["simulation/unitree_mujoco/simulate_python/walking_hazard_ground_truth_v1.py"],
        "oracle_source": other_hashes["simulation/unitree_mujoco/simulate_python/walking_hazard_oracle_calibration_v1.py"],
    }
    oracle_valid = bool(
        SLIP_THRESHOLD_M == 0.050 and SLIP_PERSISTENCE_MS == 3
        and "ACTIONABLE_RISK" in slip_protocol["states"]
        and slip_summary["development_only"] is True
        and isinstance(operational, dict) and isinstance(oracle_summary, dict)
    )
    if not oracle_valid:
        raise RuntimeError("operational label/oracle contract verification failed")
    terrain_verification = {
        "selected_candidate": "T2_seed_202608211", "valid": True,
        "selection_lock_references_verified": True, "initial_sha256": terrain_initial,
        "final_sha256": None, "byte_identical_before_after": None,
        "terrain_files_written": 0,
    }
    oracle_verification = {
        "valid": True, "threshold_m": SLIP_THRESHOLD_M,
        "persistence_ms": SLIP_PERSISTENCE_MS, "semantics_changed": False,
        "runtime_input": False, "initial_sha256": oracle_initial,
        "final_sha256": None, "byte_identical_before_after": None,
    }
    write_json(output / "terrain_immutable_verification.json", terrain_verification)
    write_json(output / "oracle_immutable_verification.json", oracle_verification)
    write_json(output / "provenance_precheck.json", {
        "starting_checkpoint": STARTING_CHECKPOINT, "protocol_sha256_before_first_run": protocol_sha,
        "allowlist_sha256_before_first_run": sha256_file(output / "input_allowlist.json"),
        "forbidden_policy_sha256_before_first_run": sha256_file(output / "forbidden_path_policy.json"),
        "terrain_verified_before_first_run": True, "oracle_verified_before_first_run": True,
        "policy_sha256": policy_hash, "scene_sha256": scene_hash, "robot_xml_sha256": robot_hash,
        "prior_artifact_hashes": prior_hashes,
        "joint_summary_loaded": isinstance(joint_summary, dict),
        "joint_provenance_loaded": isinstance(joint_provenance, dict),
        "joint_readiness_loaded": isinstance(joint_readiness, dict),
        "deterministic_recovery": {
            "first_attempt_archive": RECOVERY,
            "first_attempt_run_count": len(recovery_rows),
            "replay_required_because": recovery_failure["failure_stage"],
            "replay_hash_match_required": True,
        },
        "first_simulation_started_after_this_file_was_written": True,
    })
    traces: list[dict[str, np.ndarray]] = []
    manifests: list[dict[str, object]] = []
    for index, condition in enumerate(conditions, start=1):
        trace, metadata = collect_run(condition, args.policy_path)
        traces.append(trace); manifests.append(metadata)
        if index % 12 == 0 or index == len(conditions):
            print(f"completed {index}/{len(conditions)} acquisition runs", flush=True)
    recovery_mismatches = sorted(
        str(row["run_id"]) for row in manifests
        if recovery_hash_by_run.get(str(row["run_id"])) != row["full_trace_sha256"]
    )
    if recovery_mismatches:
        raise RuntimeError(
            f"deterministic recovery replay hash mismatch: {recovery_mismatches[:5]}"
        )
    episode_rows, positive_rows, control_rows, fall_rows = build_ledgers(traces, manifests)
    pair_manifest, parity = pair_audits(traces, manifests)
    fold_manifest = future_fold_manifest(old_manifest, [old_train, old_validation], manifests, traces)
    causal_rows = causal_pair_statistics(traces, manifests, positive_rows)
    duplicate_count = len(manifests) - len({str(row["full_trace_sha256"]) for row in manifests})
    write_json(output / "duplicate_audit.json", {
        "run_count": len(manifests), "unique_full_trace_hash_count": len({row["full_trace_sha256"] for row in manifests}),
        "duplicate_full_trace_hash_count": duplicate_count,
        "trace_hash_keys": list(TRACE_HASH_KEYS), "pass": duplicate_count == 0,
    })
    write_csv(output / "counterfactual_pair_manifest.csv", pair_manifest)
    write_csv(output / "run_manifest.csv", manifests)
    write_csv(output / "physical_episode_ledger.csv", episode_rows)
    write_csv(output / "positive_source_audit.csv", positive_rows)
    write_csv(output / "control_source_audit.csv", control_rows)
    write_csv(output / "fall_censor_audit.csv", fall_rows)
    write_csv(output / "pair_parity_audit.csv", parity)
    write_csv(output / "causal_feature_pair_statistics.csv", causal_rows)
    write_csv(output / "onset_phase_distribution.csv", _distribution_rows(positive_rows, "actual_onset_phase", [value.name for value in PHASE_BINS] + ["out_of_contract", "none"]))
    write_csv(output / "affected_foot_distribution.csv", _distribution_rows(positive_rows, "target_foot", list(SIDES)))
    write_csv(output / "speed_distribution.csv", _distribution_rows(positive_rows, "speed_mps", list(SPEEDS_MPS)))
    write_json(output / "future_nested_fold_manifest.json", fold_manifest)
    stacked = {key: np.stack([trace[key] for trace in traces]) for key in traces[0]}
    stacked["run_id"] = np.asarray([row["run_id"] for row in manifests])
    np.savez_compressed(output / "traces.npz", **stacked)
    plots = create_plots(output, traces, manifests, positive_rows, control_rows)
    terrain_final = {
        key: guard.hash_input(path, f"final immutable Terrain {key}")
        for key, path in TERRAIN_FILES.items()
    }
    terrain_same = terrain_final == terrain_initial
    terrain_verification.update({
        "final_sha256": terrain_final, "byte_identical_before_after": terrain_same,
        "verified_after_acquisition": True,
    })
    write_json(output / "terrain_immutable_verification.json", terrain_verification)
    oracle_final = {
        "operational_label_summary": guard.hash_input(INPUTS[3]["path"], "final operational-label hash"),
        "oracle_calibration_summary": guard.hash_input(INPUTS[4]["path"], "final physical-oracle calibration hash"),
        "ground_truth_source": guard.hash_input("simulation/unitree_mujoco/simulate_python/walking_hazard_ground_truth_v1.py", "final physical signal source hash"),
        "oracle_source": guard.hash_input("simulation/unitree_mujoco/simulate_python/walking_hazard_oracle_calibration_v1.py", "final Slip oracle source hash"),
    }
    oracle_verification.update({
        "final_sha256": oracle_final, "byte_identical_before_after": oracle_final == oracle_initial,
        "verified_after_acquisition": True,
    })
    write_json(output / "oracle_immutable_verification.json", oracle_verification)
    if not terrain_same or oracle_final != oracle_initial:
        raise RuntimeError("immutable Terrain/oracle input changed during acquisition")
    gates, readiness = _gate_summary(
        manifests, traces, positive_rows, control_rows, parity,
        terrain_same, guard, fold_manifest,
    )
    write_json(output / "readiness.json", readiness)
    write_json(output / "acquisition_manifest.json", {
        "artifact": "walking_v2_bilateral_slip_targeted_acquisition_v3",
        "development_only": True, "planned_runs": 216, "executed_runs": len(manifests),
        "positive_runs": len(positive_rows), "control_runs": len(control_rows),
        "pair_count": len(pair_manifest), "all_attempts_preserved": True,
        "discarded_run_count": 0, "replaced_run_count": 0, "silently_relabelled_run_count": 0,
        "pilot_run_count": 0, "model_training_count": 0, "selection_lock_created": False,
        "simulator_invocation_count": 432,
        "deterministic_recovery_replay_count": 216,
        "first_attempt_archive": RECOVERY,
        "first_attempt_raw_traces_reconstructed_exactly": True,
        "recovery_full_trace_hash_mismatch_count": len(recovery_mismatches),
        "trace_archive": "traces.npz", "trace_archive_sha256": sha256_file(output / "traces.npz"),
    })
    strong_valid = any(row["valid_target_positive"] and row["severity"] == SEVERITIES[0] for row in positive_rows)
    moderate_valid = any(row["valid_target_positive"] and row["severity"] == SEVERITIES[1] for row in positive_rows)
    if readiness["WALKING_V2_SLIP_TARGETED_RETRAINING_AUTHORIZED"]:
        next_step = "SLIP_TARGETED_RETRAINING_V3"
    elif not strong_valid or not moderate_valid or sum(row["patch_contact_application_count"] == 0 for row in manifests if row["role"] == "positive"):
        next_step = "SLIP_SCENARIO_GENERATOR_REDESIGN"
    else:
        next_step = "ADDITIONAL_TARGETED_SLIP_ACQUISITION"
    summary = {
        "artifact": "walking_v2_bilateral_slip_targeted_acquisition_v3",
        "starting_checkpoint": STARTING_CHECKPOINT, "development_only": True,
        "planned_runs": 216, "executed_runs": len(manifests),
        "simulator_invocation_count": 432,
        "deterministic_recovery_replay_count": 216,
        "first_attempt_archive": RECOVERY,
        "first_attempt_raw_traces_reconstructed_exactly": True,
        "recovery_full_trace_hash_mismatch_count": len(recovery_mismatches),
        "planned_positive_runs": 108, "executed_positive_runs": len(positive_rows),
        "planned_control_runs": 108, "executed_control_runs": len(control_rows),
        "all_attempts_preserved": True, "model_training_count": 0,
        "slip_model_or_selection_lock_created": False,
        "blind_artifact_accessed": False, "blind_artifact_generated": False,
        "strong_profile_physically_valid": strong_valid,
        "moderate_profile_physically_valid": moderate_valid,
        "strong_and_moderate_physically_valid": strong_valid and moderate_valid,
        "gates": gates, "readiness": readiness, "plots": plots,
        "next_step": next_step, "wall_time_seconds": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    core_files = sorted(
        path for path in output.rglob("*") if path.is_file()
        and path.relative_to(output).as_posix() not in {"provenance.json", "audit.md", "summary.json"}
    )
    provenance = {
        "artifact": "walking_v2_bilateral_slip_targeted_acquisition_v3",
        "protocol_sha256": protocol_sha, "starting_checkpoint": STARTING_CHECKPOINT,
        "generated_files": [{
            "path": path.relative_to(output).as_posix(),
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
        } for path in core_files],
        "hash_graph_complete": True,
        "excluded_nodes": ["provenance.json (self)", "summary.json (points to provenance)", "audit.md (points to summary)"],
        "forbidden_artifact_access_count": guard.blocked,
        "terrain_byte_identical": terrain_same, "oracle_byte_identical": oracle_final == oracle_initial,
        "first_attempt_archive": RECOVERY,
        "first_attempt_recovery_sha256": {
            path: digest for path, digest in other_hashes.items() if path.startswith(f"{RECOVERY}/")
        },
        "first_attempt_archive_files": [{
            "path": path.relative_to(REPO).as_posix(),
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
        } for path in sorted((REPO / RECOVERY).iterdir()) if path.is_file()],
    }
    write_json(output / "provenance.json", provenance)
    summary["provenance_sha256"] = sha256_file(output / "provenance.json")
    write_json(output / "summary.json", summary)
    audit = _audit_text(summary)
    audit += f"\nSummary SHA256: `{sha256_file(output / 'summary.json')}`\n"
    (output / "audit.md").write_text(audit, encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
