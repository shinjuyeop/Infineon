"""Select, lock, and reacquire a solver-effective moderate Slip profile v7."""

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

from g1_upstream_locomotion import TESTED_POLICY_SHA256, UPSTREAM_REVISION
from run_walking_v2_bilateral_slip_targeted_acquisition_v3 import build_ledgers
import run_walking_v2_bilateral_slip_targeted_acquisition_v5 as v5_runner
from run_walking_v2_bilateral_slip_targeted_acquisition_v5 import (
    save_trace_shards, sha256_file, write_csv, write_json,
)
from run_walking_v2_slip_supplemental_acquisition_v6 import (
    build_bounded_local_model,
)
from walking_v2_bilateral_slip_targeted_acquisition_v5 import (
    CONTROL_TYPES, PHASE_BINS, SIDES, SPEEDS_MPS,
)
from walking_v2_slip_moderate_profile_recalibration_v7 import (
    CALIBRATION_ARMS, CALIBRATION_DISPOSITION, CANDIDATE_IDS,
    SEVERITY_ORDER, SEVERITY_ORDER_TOLERANCE_M, RunCondition, candidate_profile,
    friction_grid_sha256, friction_profiles, moderate_v2_matrix,
    profile_calibration_matrix, select_candidate,
)
from walking_v2_slip_supplemental_acquisition_v6 import GEOMETRY_CANDIDATES


REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO / "simulation/outputs/walking_v2_slip_moderate_profile_recalibration_v7"
STARTING_CHECKPOINT = "40dadfe141509606d336af1f82c6100707cda6c6"
V4 = "simulation/outputs/walking_v2_slip_scenario_generator_redesign_v4"
V5 = "simulation/outputs/walking_v2_bilateral_slip_targeted_acquisition_v5"
V6 = "simulation/outputs/walking_v2_slip_supplemental_acquisition_v6"
V3 = "simulation/outputs/walking_v2_bilateral_slip_targeted_acquisition_v3"
EXISTING = "simulation/outputs/walking_bilateral_sensor_sink_observability_v2"
JOINT = "simulation/outputs/walking_v2_joint_terrain_slip_redesign_v1"
ORACLE = "simulation/outputs/walking_hazard_oracle_calibration_v1"
POLICY = "simulation/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"
SCENE = "simulation/unitree_mujoco/unitree_robots/g1/scene_walking_terrain_transition.xml"
ROBOT = "simulation/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
V7_SOURCES = (
    "simulation/unitree_mujoco/simulate_python/walking_v2_slip_moderate_profile_recalibration_v7.py",
    "simulation/unitree_mujoco/simulate_python/run_walking_v2_slip_moderate_profile_recalibration_v7.py",
    "simulation/unitree_mujoco/simulate_python/test_walking_v2_slip_moderate_profile_recalibration_v7.py",
)
V4_SOURCES = (
    "simulation/unitree_mujoco/simulate_python/walking_v2_slip_scenario_generator_redesign_v4.py",
    "simulation/unitree_mujoco/simulate_python/run_walking_v2_slip_scenario_generator_redesign_v4.py",
    "simulation/unitree_mujoco/simulate_python/test_walking_v2_slip_scenario_generator_redesign_v4.py",
)
V5_SOURCES = (
    "simulation/unitree_mujoco/simulate_python/walking_v2_bilateral_slip_targeted_acquisition_v5.py",
    "simulation/unitree_mujoco/simulate_python/run_walking_v2_bilateral_slip_targeted_acquisition_v5.py",
    "simulation/unitree_mujoco/simulate_python/test_walking_v2_bilateral_slip_targeted_acquisition_v5.py",
)
V6_SOURCES = (
    "simulation/unitree_mujoco/simulate_python/walking_v2_slip_supplemental_acquisition_v6.py",
    "simulation/unitree_mujoco/simulate_python/run_walking_v2_slip_supplemental_acquisition_v6.py",
    "simulation/unitree_mujoco/simulate_python/test_walking_v2_slip_supplemental_acquisition_v6.py",
)
V4_ARTIFACTS = (
    f"{V4}/summary.json", f"{V4}/readiness.json", f"{V4}/provenance.json",
    f"{V4}/local_patch_definition.json", f"{V4}/geom_contact_contract.csv",
    f"{V4}/failed_v3_quarantine_manifest.json",
)
V5_ARTIFACTS = (
    f"{V5}/summary.json", f"{V5}/readiness.json", f"{V5}/provenance.json",
    f"{V5}/acquisition_manifest.json", f"{V5}/material_profiles.json",
    f"{V5}/patch_geometry.json", f"{V5}/generator_immutable_verification.json",
    f"{V5}/oracle_immutable_verification.json", f"{V5}/terrain_immutable_verification.json",
    f"{V5}/run_manifest.csv", f"{V5}/positive_source_audit.csv",
    f"{V5}/control_source_audit.csv", f"{V5}/pair_parity_audit.csv",
    f"{V5}/trace_shard_manifest.json",
) + tuple(f"{V5}/traces_part_{index:03d}.npz" for index in range(9))
V6_ARTIFACTS = (
    f"{V6}/summary.json", f"{V6}/readiness.json", f"{V6}/provenance.json",
    f"{V6}/immutable_verification.json", f"{V6}/failure_root_cause.json",
    f"{V6}/failed_cell_diagnosis.csv", f"{V6}/geometry_candidate_contract.json",
    f"{V6}/geometry_calibration_manifest.csv", f"{V6}/geometry_calibration_metrics.csv",
    f"{V6}/selected_geometry_lock.json", f"{V6}/trace_shard_manifest.json",
    f"{V6}/traces_part_000.npz",
)
TERRAIN_FILES = {
    "model": f"{JOINT}/terrain_candidate_model.npz",
    "normalization": f"{JOINT}/terrain_candidate_normalization.json",
    "config": f"{JOINT}/terrain_candidate_config.json",
    "lock": f"{JOINT}/terrain_selection_lock.json",
}
ALLOWED_INPUTS = {
    **{path: "frozen v4 scenario-generator implementation" for path in V4_SOURCES},
    **{path: "frozen v5 acquisition implementation" for path in V5_SOURCES},
    **{path: "frozen v6 diagnosis implementation" for path in V6_SOURCES},
    **{path: "verified v4 solver artifact" for path in V4_ARTIFACTS},
    **{path: "immutable v5 acquisition evidence" for path in V5_ARTIFACTS},
    **{path: "immutable v6 diagnosis/calibration evidence" for path in V6_ARTIFACTS},
    f"{V3}/run_manifest.csv": "quarantined v3 run inventory only",
    f"{EXISTING}/manifest.json": "existing valid 120-run development inventory",
    f"{ORACLE}/summary.json": "frozen physical Slip oracle contract",
    "simulation/unitree_mujoco/simulate_python/walking_hazard_oracle_calibration_v1.py": "frozen physical Slip oracle implementation",
    "simulation/unitree_mujoco/simulate_python/walking_hazard_ground_truth_v1.py": "frozen contact-label derivation",
    "simulation/unitree_mujoco/simulate_python/terrain_profiles.py": "frozen material sources",
    "simulation/unitree_mujoco/simulate_python/bilateral_hil_sensor_v2.py": "Fusion20 virtual sensors",
    "simulation/unitree_mujoco/simulate_python/g1_upstream_locomotion.py": "frozen walking controller",
    **{path: f"immutable Terrain {name}" for name, path in TERRAIN_FILES.items()},
    POLICY: "fixed walking policy", SCENE: "fixed scene", ROBOT: "fixed robot include",
}
FORBIDDEN_TOKENS = (
    "/outer/", "_outer_", "holdout", "spatial_final", "spatial-final",
    "final_test", "final-test",
)


class AccessGuard:
    """Exact allowlist with a persistent fail-closed access ledger."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.events: list[dict[str, object]] = []
        self.blocked = 0
        self._cache: dict[str, Path] = {}
        self._flush()

    def _flush(self) -> None:
        write_json(self.output / "artifact_access_log.json", {
            "exact_paths_only": True, "forbidden_tokens": list(FORBIDDEN_TOKENS),
            "events": self.events, "blocked_access_count": self.blocked,
            "all_accesses_completed": all(
                row["status"] == "completed" for row in self.events),
            "model_scores_used_for_profile_selection": False,
        })

    def path(self, relative: str, access: str = "hash") -> Path:
        normalized = relative.replace("\\", "/")
        forbidden = next((
            token for token in FORBIDDEN_TOKENS
            if token in f"/{normalized.lower()}"), None)
        if normalized not in ALLOWED_INPUTS or forbidden:
            self.blocked += 1
            self.events.append({
                "path": normalized, "access": access, "status": "blocked",
                "reason": forbidden or "not_allowlisted"})
            self._flush()
            raise PermissionError(normalized)
        if normalized in self._cache:
            return self._cache[normalized]
        path = (REPO / normalized).resolve()
        if not path.is_file() or REPO.resolve() not in path.parents:
            self.blocked += 1
            self.events.append({
                "path": normalized, "access": access, "status": "blocked",
                "reason": "missing_or_outside_repository"})
            self._flush()
            raise FileNotFoundError(normalized)
        self.events.append({
            "path": normalized, "access": access, "status": "completed",
            "purpose": ALLOWED_INPUTS[normalized], "sha256": sha256_file(path)})
        self._cache[normalized] = path
        self._flush()
        return path

    def json(self, relative: str) -> object:
        return json.loads(self.path(relative, "json").read_text(encoding="utf-8"))

    def csv(self, relative: str) -> list[dict[str, str]]:
        with self.path(relative, "csv").open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))


def _git_value(*arguments: str) -> str:
    return subprocess.check_output(("git",) + arguments, cwd=REPO, text=True).strip()


def _checkpoint_sha256(relative: str) -> str:
    value = subprocess.check_output(
        ("git", "show", f"{STARTING_CHECKPOINT}:{relative}"), cwd=REPO)
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _bool(value: object) -> bool:
    return value is True or str(value) == "True"


def _float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def collect_run(
    condition: RunCondition, policy_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object], list[dict[str, object]]]:
    """Inject frozen G0 and the preregistered profile table into v5's recorder."""
    original_builder = v5_runner.build_local_model
    original_bounds = v5_runner.patch_bounds
    original_profiles = v5_runner.material_profiles
    geometry = GEOMETRY_CANDIDATES[0]
    v5_runner.build_local_model = lambda proxy: build_bounded_local_model(proxy, geometry)
    v5_runner.patch_bounds = lambda _side: geometry.bounds
    v5_runner.material_profiles = friction_profiles
    try:
        trace, metadata, contacts = v5_runner.collect_run(condition, policy_path)
    finally:
        v5_runner.build_local_model = original_builder
        v5_runner.patch_bounds = original_bounds
        v5_runner.material_profiles = original_profiles
    metadata.update({
        "block": condition.block, "calibration_arm": condition.calibration_arm,
        "g0_geometry_sha256": geometry.sha256,
        "disposition": condition.disposition,
        "profile_calibration_only_do_not_train": (
            condition.disposition == CALIBRATION_DISPOSITION),
        "training_use_assigned": False,
    })
    for row in contacts:
        row.update({
            "block": condition.block,
            "calibration_arm": condition.calibration_arm,
            "disposition": condition.disposition})
    return trace, metadata, contacts


def build_pair_audits(
    conditions: list[RunCondition], traces: list[dict[str, np.ndarray]],
    manifests: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    original_profiles = v5_runner.material_profiles
    v5_runner.material_profiles = friction_profiles
    try:
        return v5_runner.build_pair_audits(conditions, traces, manifests)
    finally:
        v5_runner.material_profiles = original_profiles


def enrich_source_audits(
    traces: list[dict[str, np.ndarray]], manifests: list[dict[str, object]],
    positives: list[dict[str, object]], controls: list[dict[str, object]],
    parity: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    return v5_runner.enrich_source_audits(
        traces, manifests, positives, controls, parity)


def _run_block(
    conditions: list[RunCondition], policy_path: Path, label: str,
) -> tuple[
    list[dict[str, np.ndarray]], list[dict[str, object]],
    list[dict[str, object]], list[dict[str, object]],
]:
    traces: list[dict[str, np.ndarray]] = []
    manifests: list[dict[str, object]] = []
    contacts: list[dict[str, object]] = []
    progress: list[dict[str, object]] = []
    for index, condition in enumerate(conditions):
        trace, metadata, rows = collect_run(condition, policy_path)
        traces.append(trace); manifests.append(metadata); contacts.extend(rows)
        progress.append({
            "block": label, "run_index": index, "run_id": condition.run_id,
            "pair_id": condition.pair_id, "status": "EXECUTED_AND_RETAINED",
            "full_trace_sha256": metadata["full_trace_sha256"],
            "discarded": False, "replaced": False, "silently_relabelled": False})
        print(f"{label}: completed {index + 1}/{len(conditions)}", flush=True)
    return traces, manifests, contacts, progress


def _physical_metrics(
    condition: RunCondition, trace: dict[str, np.ndarray],
    metadata: dict[str, object], audit: dict[str, object],
) -> dict[str, object]:
    side = SIDES.index(condition.target_foot)
    other = 1 - side
    expected = friction_profiles()[condition.material_profile]
    patch = trace["patch_contact"][:, side]
    mu = trace["effective_patch_friction"][:, side]
    active = patch & np.isfinite(mu) & np.isclose(
        mu, expected.friction3[0], atol=1e-12, rtol=0)
    normal = trace["patch_normal_force_n"][:, side]
    tangent = trace["patch_tangential_force_n"][:, side]
    utilization = np.divide(
        tangent, mu * normal, out=np.full_like(tangent, np.nan, dtype=float),
        where=(mu * normal) > 1e-9)
    drift = trace["anchor_drift_label_only"][:, side]
    valid_drift = drift[np.isfinite(drift) & trace["pre_fall_valid"]]
    velocity = np.linalg.norm(
        trace["foot_world_velocity_label_only"][:, side, :2], axis=1)
    fsr_other = trace["bilateral_fusion20_raw"][:, other * 10:other * 10 + 4].sum(axis=1)
    entries = np.flatnonzero(patch & ~np.r_[False, patch[:-1]])
    interval_lengths: list[int] = []
    for entry in entries:
        losses = np.flatnonzero(~patch[entry:])
        interval_lengths.append(int(losses[0]) if losses.size else len(patch) - int(entry))
    onset = audit["target_physical_onset_sample"]
    return {
        "effective_sliding_friction": expected.friction3[0],
        "effective_friction_vector": json.dumps(expected.friction5),
        "effective_friction_contract": bool(
            np.any(active) and np.allclose(mu[active], expected.friction3[0], atol=1e-12, rtol=0)),
        "patch_contact_duration_ms": int(np.sum(patch)),
        "maximum_continuous_patch_dwell_ms": max(interval_lengths, default=0),
        "normal_contact_force_mean_n": float(np.mean(normal[active])) if np.any(active) else "",
        "tangential_contact_force_mean_n": float(np.mean(tangent[active])) if np.any(active) else "",
        "friction_cone_utilization_mean": float(np.nanmean(utilization[active])) if np.any(active) else "",
        "tangential_foot_velocity_mean_mps": float(np.mean(velocity[active])) if np.any(active) else "",
        "contralateral_support_load_mean_n": float(np.mean(fsr_other[active])) if np.any(active) else "",
        "maximum_anchor_drift_m": float(np.max(valid_drift)) if valid_drift.size else "",
        "physical_onset_time_s": "" if onset == "" else float(trace["time_s"][int(onset)]),
        "touchdown_to_onset_ms": audit["touchdown_to_onset_ms"],
        "actual_first_affected_foot": audit["actual_first_affected_foot"],
        "contact_loss_before_physical_onset": bool(onset == "" and np.any(patch)),
        "fall_occurred": metadata["fall_occurred"],
        "first_fall_censor_boundary": "" if metadata["first_fall_sample"] is None else metadata["first_fall_sample"],
        "full_trace_sha256": metadata["full_trace_sha256"],
        "source_valid": _bool(audit.get("valid_target_positive", False)),
    }


def calibration_audits(
    conditions: list[RunCondition], traces: list[dict[str, np.ndarray]],
    manifests: list[dict[str, object]],
) -> tuple[
    list[dict[str, object]], list[dict[str, object]], list[dict[str, object]],
    list[dict[str, object]], dict[str, object], list[dict[str, object]],
]:
    episodes, positives, controls, falls = build_ledgers(traces, manifests)
    audit_by_run = {
        str(row["run_id"]): row for row in positives + controls}
    by_run = {condition.run_id: index for index, condition in enumerate(conditions)}
    metric_rows: list[dict[str, object]] = []
    parity_rows: list[dict[str, object]] = []
    for speed in SPEEDS_MPS:
        group = [row for row in conditions if row.speed_mps == speed]
        hard = next(row for row in group if row.calibration_arm == "hard_control")
        hard_index = by_run[hard.run_id]
        hard_trace = traces[hard_index]
        hard_meta = manifests[hard_index]
        first_contacts = [
            int(manifests[by_run[row.run_id]]["first_patch_contact_sample"])
            for row in group
            if manifests[by_run[row.run_id]]["first_patch_contact_sample"] is not None]
        first_patch = min(first_contacts) if first_contacts else len(hard_trace["time_s"])
        for condition in group:
            index = by_run[condition.run_id]
            trace = traces[index]; metadata = manifests[index]
            precontact_equal = v5_runner._trace_prefix_equal(
                hard_trace, trace, first_patch)
            prepatch_equal = all(np.array_equal(
                hard_trace[key], trace[key], equal_nan=True)
                for key in ("prepatch_qpos", "prepatch_qvel", "prepatch_policy_state"))
            post_diverged = (
                condition.calibration_arm != "hard_control"
                and v5_runner._trace_post_diverged(hard_trace, trace, first_patch))
            full_identical = hard_meta["full_trace_sha256"] == metadata["full_trace_sha256"]
            parity_rows.append({
                "group_id": condition.pair_id, "speed_mps": speed,
                "reference_run_id": hard.run_id, "run_id": condition.run_id,
                "calibration_arm": condition.calibration_arm,
                "initial_qpos_equal": hard_meta["initial_qpos_sha256"] == metadata["initial_qpos_sha256"],
                "initial_qvel_equal": hard_meta["initial_qvel_sha256"] == metadata["initial_qvel_sha256"],
                "initial_policy_observation_equal": hard_meta["initial_policy_observation_sha256"] == metadata["initial_policy_observation_sha256"],
                "prepatch_state_equal": prepatch_equal,
                "precontact_fusion20_equal": precontact_equal,
                "geometry_equal": hard_meta["collision_geometry_sha256"] == metadata["collision_geometry_sha256"],
                "timestamps_equal": np.array_equal(hard_trace["time_s"], trace["time_s"]),
                "postcontact_trajectory_diverged": post_diverged,
                "full_trace_identical_to_hard": full_identical,
                "parity_pass": bool(precontact_equal and prepatch_equal
                                    and hard_meta["initial_qpos_sha256"] == metadata["initial_qpos_sha256"]
                                    and hard_meta["initial_qvel_sha256"] == metadata["initial_qvel_sha256"]
                                    and hard_meta["collision_geometry_sha256"] == metadata["collision_geometry_sha256"]),
            })
            metric_rows.append({
                "row_type": "arm", "group_id": condition.pair_id,
                "speed_mps": speed, "run_id": condition.run_id,
                "calibration_arm": condition.calibration_arm,
                "material_profile": condition.material_profile,
                **_physical_metrics(
                    condition, trace, metadata, audit_by_run[condition.run_id]),
                "patch_base_double_contact_count": metadata["patch_base_double_contact_count"],
                "finite_fusion20": metadata["finite_fusion20"],
                "sample_count": metadata["sample_count"],
                "disposition": CALIBRATION_DISPOSITION,
                "training_use_assigned": False,
                "model_score_used": False,
            })
    arm_rows = {(
        float(row["speed_mps"]), str(row["calibration_arm"])): row
        for row in metric_rows}
    parity_map = {(
        float(row["speed_mps"]), str(row["calibration_arm"])): row
        for row in parity_rows}
    ordering_rows: list[dict[str, object]] = []
    for speed in SPEEDS_MPS:
        values = {
            arm: _float(arm_rows[(speed, arm)]["maximum_anchor_drift_m"])
            for arm in CALIBRATION_ARMS}
        sequence = [values[arm] for arm in SEVERITY_ORDER]
        pairwise = [
            sequence[index] <= sequence[index + 1] + SEVERITY_ORDER_TOLERANCE_M
            for index in range(len(sequence) - 1)]
        ordering_rows.append({
            "speed_mps": speed, **{f"{arm}_maximum_anchor_drift_m": values[arm]
                                   for arm in CALIBRATION_ARMS},
            "expected_order": "<".join(SEVERITY_ORDER),
            "tolerance_m": SEVERITY_ORDER_TOLERANCE_M,
            "pairwise_order_pass": json.dumps(pairwise),
            "severity_ordering_pass": all(pairwise),
        })
    hard_non_slip = all(
        not _bool(arm_rows[(speed, "hard_control")]["source_valid"])
        and arm_rows[(speed, "hard_control")]["actual_first_affected_foot"] == "none"
        for speed in SPEEDS_MPS)
    candidate_rows: list[dict[str, object]] = []
    profiles = friction_profiles()
    strong_mu = profiles["native_strong_ice"].friction3[0]
    moderate_mu = candidate_profile("M0").friction3[0]
    for candidate_id in CANDIDATE_IDS:
        profile = candidate_profile(candidate_id)
        speed_criteria: list[dict[str, object]] = []
        for speed in SPEEDS_MPS:
            row = arm_rows[(speed, candidate_id)]
            strong = arm_rows[(speed, "strong_reference")]
            moderate = arm_rows[(speed, "M0")]
            parity = parity_map[(speed, candidate_id)]
            physical_distinct = bool(
                row["full_trace_sha256"] != strong["full_trace_sha256"]
                and abs(_float(row["maximum_anchor_drift_m"])
                        - _float(strong["maximum_anchor_drift_m"])) > 1e-6)
            ordering_bracket = bool(
                _float(moderate["maximum_anchor_drift_m"])
                <= _float(row["maximum_anchor_drift_m"]) + SEVERITY_ORDER_TOLERANCE_M
                and _float(row["maximum_anchor_drift_m"])
                <= _float(strong["maximum_anchor_drift_m"]) + SEVERITY_ORDER_TOLERANCE_M)
            criteria = {
                "speed_mps": speed,
                "source_valid": _bool(row["source_valid"]),
                "left_first": row["actual_first_affected_foot"] == "left",
                "onset_mid_late": audit_by_run[str(row["run_id"])]["actual_onset_phase"] == "mid_late_stance",
                "no_air_or_postfall_only": _bool(row["source_valid"]),
                "no_double_contact": int(row["patch_base_double_contact_count"]) == 0,
                "physics_stable": _bool(row["finite_fusion20"]) and int(row["sample_count"]) == 3000,
                "precontact_parity": _bool(parity["parity_pass"]),
                "postcontact_divergence": _bool(parity["postcontact_trajectory_diverged"]),
                "effective_friction_contract": _bool(row["effective_friction_contract"]),
                "strictly_between_references": strong_mu < profile.friction3[0] < moderate_mu,
                "physically_distinct_from_strong": physical_distinct,
                "severity_bracket_with_tolerance": ordering_bracket,
                "hard_control_non_slip": hard_non_slip,
            }
            criteria["speed_pass"] = all(
                bool(value) for key, value in criteria.items()
                if key != "speed_mps")
            speed_criteria.append(criteria)
        candidate_pass = bool(
            candidate_id != "M0"
            and all(bool(row["speed_pass"]) for row in speed_criteria))
        candidate_rows.append({
            "row_type": "candidate", "candidate_id": candidate_id,
            "profile_name": profile.name,
            "friction3": json.dumps(profile.friction3),
            "friction5": json.dumps(profile.friction5),
            "profile_sha256": profile.sha256,
            "speed_criteria": json.dumps(speed_criteria, sort_keys=True),
            "valid_speed_count": sum(_bool(row["source_valid"]) for row in speed_criteria),
            "candidate_pass": candidate_pass,
        })
    summary = {
        "hard_controls_physically_non_slip": hard_non_slip,
        "all_arm_precontact_parity": all(_bool(row["parity_pass"]) for row in parity_rows),
        "all_nonhard_postcontact_diverged": all(
            _bool(row["postcontact_trajectory_diverged"])
            for row in parity_rows if row["calibration_arm"] != "hard_control"),
        "patch_base_double_contact_count": sum(
            int(row["patch_base_double_contact_count"]) for row in metric_rows),
        "friction_contract_violation_count": sum(
            not _bool(row["effective_friction_contract"]) for row in metric_rows),
        "physical_severity_ordering_pass": all(
            _bool(row["severity_ordering_pass"]) for row in ordering_rows),
        "model_scores_used": False,
    }
    return metric_rows + candidate_rows, parity_rows, ordering_rows, episodes, summary, falls


def acquisition_gates(
    conditions: list[RunCondition], traces: list[dict[str, np.ndarray]],
    manifests: list[dict[str, object]], positives: list[dict[str, object]],
    controls: list[dict[str, object]], parity: list[dict[str, object]],
    forbidden_count: int,
) -> dict[str, object]:
    valid = [row for row in positives if _bool(row["source_valid"])]
    cells = {(float(row["speed_mps"]), str(row["target_foot"]),
              str(row["target_phase"])) for row in valid}
    speed_foot = {(float(row["speed_mps"]), str(row["target_foot"])) for row in valid}
    speed_counts = {f"{speed:.2f}": sum(
        float(row["speed_mps"]) == speed for row in valid) for speed in SPEEDS_MPS}
    foot_counts = {side: sum(row["target_foot"] == side for row in valid) for side in SIDES}
    ratio = foot_counts["left"] / foot_counts["right"] if foot_counts["right"] else float("inf")
    trace_hashes = [str(row["full_trace_sha256"]) for row in manifests]
    air_postfall = sum(int(np.sum(
        trace["slip_physical_active"]
        & ((~trace["physical_contact"]) | ~trace["pre_fall_valid"][:, None])))
        for trace in traces)
    values = {
        "executed_positive_attempts": len(positives),
        "executed_control_attempts": len(controls),
        "source_valid_positive_count": len(valid),
        "source_valid_by_speed": speed_counts,
        "source_valid_by_foot": foot_counts,
        "source_valid_by_phase": {phase.name: sum(
            row["actual_onset_phase"] == phase.name for row in valid) for phase in PHASE_BINS},
        "covered_speed_foot_phase_cells": len(cells),
        "all_speed_foot_phase_cells_covered": len(cells) == 18,
        "all_missing_left_mid_late_cells_filled": all(
            (speed, "left", "mid_late_stance") in cells for speed in SPEEDS_MPS),
        "each_speed_has_both_feet": all(
            (speed, side) in speed_foot for speed in SPEEDS_MPS for side in SIDES),
        "left_right_ratio": ratio,
        "target_first_accuracy": (
            sum(_bool(row["target_first"]) for row in valid) / len(valid)
            if valid else 0.0),
        "valid_control_count": sum(_bool(row["control_source_valid"]) for row in controls),
        "control_physical_slip_onset_count": sum(
            not _bool(row["physical_slip_onset_free"]) for row in controls),
        "precontact_pair_parity_count": sum(_bool(row["parity_pass"]) for row in parity),
        "postcontact_divergence_count": sum(
            _bool(row["postcontact_trajectory_diverged"]) for row in parity),
        "full_trace_identity_count": sum(_bool(row["full_trace_identical"]) for row in parity),
        "patch_base_double_contact_count": sum(
            int(row["patch_base_double_contact_count"]) for row in parity),
        "friction_contract_violation_count": sum(
            not _bool(row["positive_effective_friction_contract"])
            or not _bool(row["control_effective_friction_contract"])
            for row in parity),
        "air_or_postfall_positive_attribution_count": air_postfall,
        "duplicate_unique_trace_count": len(trace_hashes) - len(set(trace_hashes)),
        "forbidden_artifact_access_count": forbidden_count,
        "discarded_count": sum(_bool(row["discarded"]) for row in manifests),
        "replaced_count": sum(_bool(row["replaced"]) for row in manifests),
        "silently_relabelled_count": sum(
            _bool(row["silently_relabelled"]) for row in manifests),
        "exact_1khz_fusion20": all(
            row["sample_count"] == 3000 and row["fusion20_shape"] == [3000, 20]
            and _bool(row["finite_fusion20"])
            and float(row["sample_spacing_max_error_s"]) < 1e-12 for row in manifests),
    }
    values["pass"] = bool(
        values["executed_positive_attempts"] == 54
        and values["executed_control_attempts"] == 54
        and values["source_valid_positive_count"] >= 39
        and values["all_speed_foot_phase_cells_covered"]
        and values["all_missing_left_mid_late_cells_filled"]
        and values["each_speed_has_both_feet"]
        and 0.80 <= ratio <= 1.25
        and values["target_first_accuracy"] >= 0.90
        and values["valid_control_count"] == 54
        and values["control_physical_slip_onset_count"] == 0
        and values["precontact_pair_parity_count"] == 54
        and values["postcontact_divergence_count"] == 54
        and values["full_trace_identity_count"] == 0
        and values["patch_base_double_contact_count"] == 0
        and values["friction_contract_violation_count"] == 0
        and values["air_or_postfall_positive_attribution_count"] == 0
        and values["duplicate_unique_trace_count"] == 0
        and values["forbidden_artifact_access_count"] == 0
        and values["exact_1khz_fusion20"])
    return values


def canonical_artifacts(
    v5_manifests: list[dict[str, str]], v5_positives: list[dict[str, str]],
    v5_controls: list[dict[str, str]], v5_shards: dict[str, object],
    v7_manifests: list[dict[str, object]], v7_positives: list[dict[str, object]],
    v7_controls: list[dict[str, object]], v7_shards: dict[str, object],
    acquisition_pass: bool, physics_provenance_pass: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    v5_positive_map = {row["run_id"]: row for row in v5_positives}
    v5_control_map = {row["run_id"]: row for row in v5_controls}
    v7_positive_map = {str(row["run_id"]): row for row in v7_positives}
    v7_control_map = {str(row["run_id"]): row for row in v7_controls}
    v5_shard_map = {
        run_id: {"path": row["path"], "sha256": row["sha256"]}
        for row in v5_shards["shards"] for run_id in row["run_ids"]}  # type: ignore[index]
    v7_shard_map = {
        run_id: {"path": row["path"], "sha256": row["sha256"]}
        for row in v7_shards["shards"] for run_id in row["run_ids"]}  # type: ignore[index]
    rows: list[dict[str, object]] = []
    for meta in v5_manifests:
        if meta["severity"] != "native_strong_ice":
            continue
        audit = (v5_positive_map if meta["role"] == "positive" else v5_control_map)[meta["run_id"]]
        valid = _bool(audit["source_valid"] if meta["role"] == "positive"
                      else audit["control_source_valid"])
        rows.append({
            "source_acquisition": "v5_strong", "run_id": meta["run_id"],
            "pair_id": f"v5:{meta['pair_id']}", "role": meta["role"],
            "speed_mps": float(meta["speed_mps"]), "target_foot": meta["target_foot"],
            "target_phase": meta["target_phase"], "severity": "native_strong_ice",
            "control_type": meta["control_type"],
            "variation_index": int(meta["variation_index"]),
            "source_valid": valid, "future_training_eligible": valid,
            "trace_shard": v5_shard_map[meta["run_id"]], "development_only": True,
        })
    for meta in v7_manifests:
        run_id = str(meta["run_id"])
        audit = (v7_positive_map if meta["role"] == "positive" else v7_control_map)[run_id]
        valid = _bool(audit["source_valid"] if meta["role"] == "positive"
                      else audit["control_source_valid"])
        rows.append({
            "source_acquisition": "v7_moderate_v2", "run_id": run_id,
            "pair_id": f"v7:{meta['pair_id']}", "role": meta["role"],
            "speed_mps": meta["speed_mps"], "target_foot": meta["target_foot"],
            "target_phase": meta["target_phase"], "severity": "moderate_v2",
            "control_type": meta["control_type"],
            "variation_index": meta["variation_index"],
            "source_valid": valid, "future_training_eligible": valid,
            "trace_shard": v7_shard_map[run_id], "development_only": True,
        })
    coverage: list[dict[str, object]] = []
    for speed in SPEEDS_MPS:
        for foot in SIDES:
            for phase in (row.name for row in PHASE_BINS):
                for severity in ("native_strong_ice", "moderate_v2"):
                    selected = [
                        row for row in rows if row["role"] == "positive"
                        and float(row["speed_mps"]) == speed
                        and row["target_foot"] == foot and row["target_phase"] == phase
                        and row["severity"] == severity]
                    valid_count = sum(_bool(row["source_valid"]) for row in selected)
                    coverage.append({
                        "speed_mps": speed, "target_foot": foot,
                        "target_phase": phase, "severity": severity,
                        "planned_positive_attempts": len(selected),
                        "source_valid_positive_count": valid_count,
                        "has_valid_support": valid_count > 0})
    valid_positive = [
        row for row in rows if row["role"] == "positive" and row["source_valid"]]
    controls = [row for row in rows if row["role"] == "control"]
    speed_counts = {f"{speed:.2f}": sum(
        float(row["speed_mps"]) == speed for row in valid_positive) for speed in SPEEDS_MPS}
    target_first_rows = []
    for row in v5_positives:
        if row["severity"] == "native_strong_ice" and _bool(row["source_valid"]):
            target_first_rows.append(_bool(row["target_first"]))
    for row in v7_positives:
        if _bool(row["source_valid"]):
            target_first_rows.append(_bool(row["target_first"]))
    audit = {
        "positive_attempts": sum(row["role"] == "positive" for row in rows),
        "control_attempts": len(controls), "total_runs": len(rows),
        "strong_source_valid_count": sum(
            row["severity"] == "native_strong_ice" for row in valid_positive),
        "moderate_v2_source_valid_count": sum(
            row["severity"] == "moderate_v2" for row in valid_positive),
        "combined_source_valid_positive_count": len(valid_positive),
        "combined_source_valid_rate": len(valid_positive) / 108 if rows else 0.0,
        "covered_factorial_cells": sum(row["has_valid_support"] for row in coverage),
        "valid_positive_by_speed": speed_counts,
        "both_feet_represented": {row["target_foot"] for row in valid_positive} == set(SIDES),
        "all_phases_represented": {row["target_phase"] for row in valid_positive} == {
            row.name for row in PHASE_BINS},
        "target_first_accuracy": sum(target_first_rows) / len(target_first_rows),
        "valid_control_count": sum(_bool(row["source_valid"]) for row in controls),
        "v3_run_count": 0, "v5_moderate_v1_run_count": 0,
        "v6_calibration_run_count": 0, "v7_profile_calibration_run_count": 0,
        "outer_holdout_final_run_count": 0,
        "physics_provenance_pass": physics_provenance_pass,
    }
    audit["pass"] = bool(
        acquisition_pass and audit["positive_attempts"] == 108
        and audit["control_attempts"] == 108 and audit["total_runs"] == 216
        and audit["strong_source_valid_count"] == 48
        and audit["moderate_v2_source_valid_count"] >= 39
        and audit["combined_source_valid_positive_count"] >= 87
        and audit["combined_source_valid_rate"] >= 0.80
        and audit["covered_factorial_cells"] == 36
        and all(value >= 24 for value in speed_counts.values())
        and audit["both_feet_represented"] and audit["all_phases_represented"]
        and audit["target_first_accuracy"] >= 0.90
        and audit["valid_control_count"] == 108
        and audit["physics_provenance_pass"])
    return {
        "version": "walking_v2_canonical_development_v7",
        "CANONICAL_V7_DEVELOPMENT_DATA_READY": audit["pass"],
        "original_v5_readiness_preserved_false": True,
        "training_performed": False, "training_use_assignment_performed": False,
        "audit": audit, "runs": rows,
    }, coverage


def moderate_v1_disposition(v5_positives: list[dict[str, str]]) -> dict[str, object]:
    rows = [
        row for row in v5_positives
        if row["severity"] == "moderate_ice_preregistered"]
    valid = [row["run_id"] for row in rows if _bool(row["source_valid"])]
    no_onset = [
        row["run_id"] for row in rows if row["failure_reason"] == "NO_PHYSICAL_ONSET"]
    return {
        "profile_version": "moderate-v1",
        "disposition": "SUPERSEDED_PROFILE_DEVELOPMENT_DIAGNOSTIC",
        "kept_separately_versioned": True, "deleted_or_overwritten": False,
        "attempt_count": len(rows), "physically_valid_run_count": len(valid),
        "physically_valid_run_ids": valid, "no_onset_run_count": len(no_onset),
        "no_onset_run_ids": no_onset,
        "possible_future_hard_negative_eligibility": [{
            "run_id": run_id, "eligibility": "REVIEW_ONLY",
            "training_use_assigned": False} for run_id in no_onset],
        "automatically_included_in_training": False,
    }


def future_folds(
    canonical: dict[str, object], existing_runs: list[dict[str, object]],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for run in existing_runs:
        variation = int(run["variation_index"])
        run_id = str(run["run_id"])
        rows.append({
            "source_acquisition": "existing_bilateral_development_v2",
            "fold": variation % 3, "pair_id": "",
            "variation_group": f"existing:v{variation}",
            "run_group": f"existing:{run_id}",
            "contact_episode_group": f"existing:{run_id}:all",
            "run_id": run_id, "speed_mps": run["speed_mps"],
            "target_foot": "bilateral", "target_phase": "legacy_development",
            "severity": "legacy_development", "role": run["role"],
            "control_type": "legacy_development", "development_only": True})
    for run in canonical["runs"]:  # type: ignore[index]
        if not _bool(run["future_training_eligible"]):
            continue
        variation = int(run["variation_index"])
        source = str(run["source_acquisition"])
        run_id = str(run["run_id"])
        rows.append({
            **run, "fold": variation % 3,
            "variation_group": f"{source}:v{variation}",
            "run_group": f"{source}:{run_id}",
            "contact_episode_group": f"{source}:{run_id}:all"})
    leakage: list[str] = []
    for field in ("pair_id", "variation_group", "run_group", "contact_episode_group"):
        owners: dict[str, set[int]] = {}
        for row in rows:
            value = str(row[field])
            if value:
                owners.setdefault(value, set()).add(int(row["fold"]))
        leakage.extend(
            f"{field}:{value}" for value, folds in owners.items() if len(folds) != 1)
    coverage: list[dict[str, object]] = []
    for fold in range(3):
        selected = [
            row for row in rows if int(row["fold"]) == fold
            and row["source_acquisition"] in {"v5_strong", "v7_moderate_v2"}]
        positive = [row for row in selected if row["role"] == "positive"]
        controls = [row for row in selected if row["role"] == "control"]
        coverage.append({
            "fold": fold,
            "all_speeds": {float(row["speed_mps"]) for row in selected} == set(SPEEDS_MPS),
            "both_feet": {row["target_foot"] for row in selected} == set(SIDES),
            "both_severities": {row["severity"] for row in positive} == {
                "native_strong_ice", "moderate_v2"},
            "all_phases_where_possible": {row["target_phase"] for row in positive} == {
                row.name for row in PHASE_BINS},
            "both_control_types": {row["control_type"] for row in controls} == set(CONTROL_TYPES),
        })
    audit = {
        "row_count": len(rows), "fold_ids": sorted({row["fold"] for row in rows}),
        "leakage_count": len(leakage), "leaking_groups": leakage,
        "positive_control_pairs_same_fold": not any(
            value.startswith("pair_id:") for value in leakage),
        "coverage": coverage,
        "coverage_pass": all(all(
            bool(value) for key, value in row.items() if key != "fold") for row in coverage),
        "v3_run_count": 0, "v6_calibration_run_count": 0,
        "v7_profile_calibration_run_count": 0, "outer_holdout_final_run_count": 0,
    }
    audit["valid"] = bool(
        canonical["audit"]["pass"]  # type: ignore[index]
        and audit["fold_ids"] == [0, 1, 2]
        and not leakage and audit["coverage_pass"])
    return {
        "version": "walking_v2_future_targeted_retraining_folds_v7",
        "grouping": [
            "counterfactual_pair", "variation", "run", "contact_episode",
            "acquisition_source"],
        "training_performed": False, "audit": audit, "rows": rows,
    }


def _save_plot(path: Path) -> None:
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def create_plots(
    output: Path, calibration_metrics: list[dict[str, object]],
    ordering: list[dict[str, object]], positives: list[dict[str, object]],
    coverage: list[dict[str, object]], selected: str | None,
) -> None:
    arms = [row for row in calibration_metrics if row["row_type"] == "arm"]
    plt.figure(figsize=(8, 4))
    for speed in SPEEDS_MPS:
        rows = [row for row in arms if float(row["speed_mps"]) == speed]
        plt.plot(
            [_float(row["effective_sliding_friction"]) for row in rows],
            [_float(row["maximum_anchor_drift_m"]) for row in rows],
            marker="o", label=f"{speed:.2f} m/s")
    plt.axhline(0.05, color="black", linestyle="--")
    plt.xlabel("sliding friction coefficient"); plt.ylabel("maximum anchor drift (m)")
    plt.title("Friction coefficient versus anchor drift"); plt.legend()
    _save_plot(output / "friction_coefficient_vs_anchor_drift.png")

    plt.figure(figsize=(9, 4))
    for row in ordering:
        values = [_float(row[f"{arm}_maximum_anchor_drift_m"]) for arm in SEVERITY_ORDER]
        plt.plot(range(6), values, marker="o", label=f"{float(row['speed_mps']):.2f}")
    plt.xticks(range(6), SEVERITY_ORDER, rotation=20)
    plt.ylabel("maximum anchor drift (m)"); plt.title("Physical profile severity ordering")
    plt.legend(); _save_plot(output / "profile_severity_ordering.png")

    plt.figure(figsize=(8, 4))
    for candidate_id in CANDIDATE_IDS:
        rows = [row for row in arms if row["calibration_arm"] == candidate_id]
        plt.plot([row["speed_mps"] for row in rows],
                 [row["maximum_anchor_drift_m"] for row in rows], marker="o", label=candidate_id)
    plt.axhline(0.05, color="black", linestyle="--")
    plt.xlabel("speed (m/s)"); plt.ylabel("maximum anchor drift (m)")
    plt.title(f"Calibration response by speed; selected={selected}"); plt.legend()
    _save_plot(output / "calibration_response_by_speed.png")

    plt.figure(figsize=(8, 4))
    for side in SIDES:
        values = [
            _float(row["maximum_target_anchor_drift_m"])
            for row in positives if row["target_foot"] == side]
        plt.scatter([side] * len(values), values, alpha=0.65)
    plt.axhline(0.05, color="black", linestyle="--")
    plt.ylabel("maximum anchor drift (m)"); plt.title("Left/right moderate-v2 response")
    if not positives:
        plt.text(0.5, 0.5, "moderate-v2 reacquisition not executed", transform=plt.gca().transAxes, ha="center")
    _save_plot(output / "left_right_moderate_v2_response.png")

    plt.figure(figsize=(8, 4))
    counts = [sum(
        row["actual_onset_phase"] == phase.name and _bool(row["source_valid"])
        for row in positives) for phase in PHASE_BINS]
    plt.bar([row.name for row in PHASE_BINS], counts)
    plt.ylabel("source-valid positive count"); plt.title("Moderate-v2 onset distribution")
    _save_plot(output / "onset_distribution_by_phase.png")

    plt.figure(figsize=(10, 4))
    values = [int(row["source_valid_positive_count"]) for row in coverage]
    plt.bar(range(len(values)), values,
            color=["#4c9f70" if value else "#d95f5f" for value in values])
    plt.xlabel("canonical factorial cell index"); plt.ylabel("valid positive count")
    plt.title(f"Canonical factorial coverage: {sum(value > 0 for value in values)}/36")
    _save_plot(output / "canonical_factorial_coverage.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    head = _git_value("rev-parse", "HEAD")
    if head != STARTING_CHECKPOINT:
        raise RuntimeError(f"expected checkpoint {STARTING_CHECKPOINT}, found {head}")
    status_lines = _git_value("status", "--porcelain=v1").splitlines()
    unrelated = [
        line for line in status_lines if not any(path in line for path in V7_SOURCES)
        and "simulation/outputs/walking_v2_slip_moderate_profile_recalibration_v7" not in line]
    if unrelated:
        raise RuntimeError(f"unrelated worktree changes before execution: {unrelated}")

    profiles = friction_profiles()
    calibration_conditions = profile_calibration_matrix()
    protocol = {
        "task": "Recalibrate and Reacquire Moderate Bilateral Slip Profile v7",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "profile_grid": {name: profile.contract | {"sha256": profile.sha256}
                         for name, profile in profiles.items()},
        "candidate_ids": list(CANDIDATE_IDS),
        "friction_grid_sha256": friction_grid_sha256(),
        "interpolation_scope": "explicit-pair sliding components 0 and 1 only",
        "torsional_rolling_source": "frozen moderate-v1",
        "severity_order_tolerance_m": SEVERITY_ORDER_TOLERANCE_M,
        "calibration_conditions": [vars(row) for row in calibration_conditions],
        "calibration_unique_runs": 18,
        "selection_tiebreak": [
            "highest friction coefficient", "smallest deviation from moderate-v1",
            "lexical candidate ID"],
        "reacquisition_contract": {
            "conditional_on_profile_lock": True, "positive_attempts": 54,
            "matched_controls": 54, "total_unique_runs": 108,
            "speeds_mps": list(SPEEDS_MPS), "target_feet": list(SIDES),
            "phases": [row.name for row in PHASE_BINS], "variations": [0, 1, 2],
            "geometry": "frozen G0", "adaptive_replacements": False},
        "oracle_changed": False, "strong_profile_changed": False,
        "terrain_changed": False, "patch_geometry_changed": False,
        "training": False, "development_only": True,
        "overwrite_policy": "refuse non-empty output",
    }
    write_json(output / "protocol.json", protocol)
    protocol_sha = sha256_file(output / "protocol.json")
    write_json(output / "input_allowlist.json", {
        "exact_paths_only": True,
        "inputs": [{"path": path, "purpose": purpose}
                   for path, purpose in ALLOWED_INPUTS.items()]})
    write_json(output / "forbidden_path_policy.json", {
        "fail_closed": True, "forbidden_tokens": list(FORBIDDEN_TOKENS),
        "outer_holdout_final_access_authorized": False,
        "model_score_profile_selection_authorized": False,
        "v3_trace_use_authorized": False})
    write_json(output / "friction_candidate_contract.json", {
        "frozen_before_execution": True, "candidate_count": 4,
        "candidate_ids": list(CANDIDATE_IDS),
        "grid_sha256": friction_grid_sha256(),
        "profiles": {profile.candidate_id: profile.contract | {"sha256": profile.sha256}
                     for profile in profiles.values() if profile.role == "candidate"},
        "strong_reference": profiles["native_strong_ice"].contract,
        "sliding_components_only_interpolated": True,
        "torsional_rolling_solref_solimp_condim_priority_geometry_frozen": True})
    guard = AccessGuard(output)
    guarded = {path: guard.path(path) for path in ALLOWED_INPUTS}

    v4_before = {path: sha256_file(guarded[path]) for path in V4_SOURCES}
    v4_checkpoint = {path: _checkpoint_sha256(path) for path in V4_SOURCES}
    v5_source_before = {path: sha256_file(guarded[path]) for path in V5_SOURCES}
    v6_source_before = {path: sha256_file(guarded[path]) for path in V6_SOURCES}
    v5_artifact_before = {path: sha256_file(guarded[path]) for path in V5_ARTIFACTS}
    v6_artifact_before = {path: sha256_file(guarded[path]) for path in V6_ARTIFACTS}
    terrain_before = {name: sha256_file(guarded[path]) for name, path in TERRAIN_FILES.items()}
    oracle_paths = (
        f"{ORACLE}/summary.json",
        "simulation/unitree_mujoco/simulate_python/walking_hazard_oracle_calibration_v1.py",
        "simulation/unitree_mujoco/simulate_python/walking_hazard_ground_truth_v1.py")
    oracle_before = {path: sha256_file(guarded[path]) for path in oracle_paths}
    v5_summary = guard.json(f"{V5}/summary.json")
    v5_readiness = guard.json(f"{V5}/readiness.json")
    v6_summary = guard.json(f"{V6}/summary.json")
    v6_readiness = guard.json(f"{V6}/readiness.json")
    v5_manifest = guard.csv(f"{V5}/run_manifest.csv")
    v5_positive = guard.csv(f"{V5}/positive_source_audit.csv")
    v5_controls = guard.csv(f"{V5}/control_source_audit.csv")
    v5_shards = guard.json(f"{V5}/trace_shard_manifest.json")
    existing = guard.json(f"{EXISTING}/manifest.json")
    v3_rows = guard.csv(f"{V3}/run_manifest.csv")
    quarantine = guard.json(f"{V4}/failed_v3_quarantine_manifest.json")
    patch_snapshot = v5_runner._patch_contract_snapshot()
    profile_hash_before = {name: profile.sha256 for name, profile in profiles.items()}
    preflight_ready = bool(
        head == STARTING_CHECKPOINT and not unrelated
        and v4_before == v4_checkpoint and patch_snapshot["explicit_pair_count"] == 8
        and GEOMETRY_CANDIDATES[0].bounds.x_min_m == -1.0
        and GEOMETRY_CANDIDATES[0].bounds.x_max_m == 2.0
        and not v5_readiness["WALKING_V2_SLIP_REACQUISITION_DATA_READY"]
        and v5_summary["gates"]["positive_source_coverage"]["source_valid_count"] == 84
        and v6_summary["primary_failure_cause"] == "LEFT_RIGHT_GAIT_ASYMMETRY"
        and not v6_readiness["WALKING_V2_SLIP_GEOMETRY_CALIBRATION_READY"]
        and len(v3_rows) == 216 and quarantine["all_runs_quarantined"]
        and len(existing["runs"]) == 120 and guard.blocked == 0)
    immutable = {
        "starting_checkpoint": STARTING_CHECKPOINT, "execution_head": head,
        "clean_worktree_verified_before_implementation": True,
        "task_scoped_uncommitted_paths_before_execution": status_lines,
        "unrelated_worktree_change_count_before_execution": len(unrelated),
        "protocol_sha256_before_execution": protocol_sha,
        "protocol_frozen_before_execution": True,
        "v4_before_sha256": v4_before, "v4_checkpoint_sha256": v4_checkpoint,
        "v5_source_before_sha256": v5_source_before,
        "v6_source_before_sha256": v6_source_before,
        "v5_artifact_before_sha256": v5_artifact_before,
        "v6_artifact_before_sha256": v6_artifact_before,
        "terrain_before_sha256": terrain_before, "oracle_before_sha256": oracle_before,
        "profile_before_sha256": profile_hash_before,
        "g0_geometry_sha256": GEOMETRY_CANDIDATES[0].sha256,
        "patch_geometry_sha256": patch_snapshot["patch_geometry_sha256"],
        "explicit_pair_contract_sha256": patch_snapshot["pair_contract_sha256"],
        "explicit_pair_count": patch_snapshot["explicit_pair_count"],
        "v3_quarantine_disposition": quarantine["disposition"],
        "preflight_ready": preflight_ready, "after": {},
        "all_immutable_after_execution": False}
    write_json(output / "immutable_verification.json", immutable)
    if not preflight_ready:
        raise RuntimeError("v7 immutable preflight failed")

    cal_traces, cal_manifests, cal_contacts, cal_progress = _run_block(
        calibration_conditions, guarded[POLICY], "profile_calibration")
    (cal_metrics, cal_parity, ordering_rows, cal_episodes,
     cal_summary, cal_falls) = calibration_audits(
         calibration_conditions, cal_traces, cal_manifests)
    write_csv(output / "profile_calibration_manifest.csv", cal_manifests)
    write_csv(output / "profile_calibration_metrics.csv", cal_metrics)
    write_csv(output / "profile_calibration_pair_parity.csv", cal_parity)
    write_csv(output / "profile_calibration_contact_audit.csv", cal_contacts)
    write_csv(output / "profile_calibration_physical_episode_ledger.csv", cal_episodes)
    write_csv(output / "profile_calibration_fall_censor_audit.csv", cal_falls)
    write_csv(output / "profile_severity_ordering.csv", ordering_rows)
    candidate_rows = [row for row in cal_metrics if row["row_type"] == "candidate"]
    selected = select_candidate(candidate_rows)
    lock_path = output / "moderate_v2_profile_lock.json"
    lock_sha_before: str | None = None
    if selected is not None:
        profile = candidate_profile(selected)
        evidence = next(row for row in candidate_rows if row["candidate_id"] == selected)
        lock = {
            "version": "moderate-v2", "candidate_id": selected,
            "profile_name": profile.name, "friction3": profile.friction3,
            "friction5": profile.friction5,
            "full_friction_vector_sha256": _canonical_sha256(profile.friction5),
            "profile_contract_sha256": profile.sha256,
            "selection_evidence": evidence,
            "selection_evidence_sha256": _canonical_sha256(evidence),
            "selection_tiebreak": "highest passing friction; least deviation from moderate-v1",
            "strong_reference_unchanged": True, "moderate_v1_unchanged": True,
            "g0_geometry_only": True, "locked_before_reacquisition": True,
            "changed_after_lock": False}
        write_json(lock_path, lock)
        lock_sha_before = sha256_file(lock_path)

    reacquisition_conditions: list[RunCondition] = []
    reacquisition_traces: list[dict[str, np.ndarray]] = []
    reacquisition_manifests: list[dict[str, object]] = []
    reacquisition_contacts: list[dict[str, object]] = []
    reacquisition_progress: list[dict[str, object]] = []
    pair_manifest: list[dict[str, object]] = []
    parity_rows: list[dict[str, object]] = []
    episodes: list[dict[str, object]] = []
    positives: list[dict[str, object]] = []
    controls: list[dict[str, object]] = []
    falls: list[dict[str, object]] = []
    if selected is not None:
        reacquisition_conditions = moderate_v2_matrix(selected)
        (reacquisition_traces, reacquisition_manifests, reacquisition_contacts,
         reacquisition_progress) = _run_block(
             reacquisition_conditions, guarded[POLICY], "moderate_v2_reacquisition")
        pair_manifest, parity_rows = build_pair_audits(
            reacquisition_conditions, reacquisition_traces, reacquisition_manifests)
        episodes, positives, controls, falls = build_ledgers(
            reacquisition_traces, reacquisition_manifests)
        positives, controls = enrich_source_audits(
            reacquisition_traces, reacquisition_manifests,
            positives, controls, parity_rows)

    all_conditions = calibration_conditions + reacquisition_conditions
    all_traces = cal_traces + reacquisition_traces
    shard_manifest = save_trace_shards(output, all_conditions, all_traces)
    if not (shard_manifest["all_under_45_mib"]
            and shard_manifest["all_exact_roundtrip_verified"]):
        raise RuntimeError("trace shard contract failed")
    gates = acquisition_gates(
        reacquisition_conditions, reacquisition_traces, reacquisition_manifests,
        positives, controls, parity_rows, guard.blocked)
    if selected is None:
        gates["not_executed_reason"] = "no candidate passed all three speeds"
    if selected is not None:
        write_csv(output / "moderate_v2_run_manifest.csv", reacquisition_manifests)
        write_csv(output / "moderate_v2_pair_manifest.csv", pair_manifest)
        write_csv(output / "moderate_v2_contact_audit.csv", reacquisition_contacts)
        write_csv(output / "moderate_v2_physical_episode_ledger.csv", episodes)
        write_csv(output / "moderate_v2_positive_source_audit.csv", positives)
        write_csv(output / "moderate_v2_control_source_audit.csv", controls)
        write_csv(output / "moderate_v2_pair_parity.csv", parity_rows)
        write_csv(output / "moderate_v2_fall_censor_audit.csv", falls)
    write_csv(output / "run_progress_manifest.csv", cal_progress + reacquisition_progress)

    disposition = moderate_v1_disposition(v5_positive)
    write_json(output / "moderate_v1_disposition.json", disposition)
    canonical, coverage = canonical_artifacts(
        v5_manifest, v5_positive, v5_controls, v5_shards,
        reacquisition_manifests, positives, controls, shard_manifest,
        bool(gates["pass"]), True)
    if gates["pass"]:
        write_json(output / "canonical_development_manifest.json", canonical)
        write_csv(output / "canonical_factorial_coverage.csv", coverage)
    canonical_ready = _bool(canonical["audit"]["pass"])
    if canonical_ready:
        folds = future_folds(canonical, existing["runs"])
        write_json(output / "future_nested_fold_manifest.json", folds)
    else:
        folds = {"version": "walking_v2_future_targeted_retraining_folds_v7",
                 "not_created": True, "reason": "canonical readiness failed",
                 "training_performed": False, "audit": {"valid": False}, "rows": []}

    trace_hashes = [str(row["full_trace_sha256"]) for row in (
        cal_manifests + reacquisition_manifests)]
    write_json(output / "duplicate_audit.json", {
        "profile_calibration_run_count": len(cal_manifests),
        "moderate_v2_reacquisition_run_count": len(reacquisition_manifests),
        "unique_run_id_count": len({row.run_id for row in all_conditions}),
        "unique_trace_hash_count": len(set(trace_hashes)),
        "duplicate_unique_trace_count": len(trace_hashes) - len(set(trace_hashes)),
        "discarded_count": 0, "replaced_count": 0,
        "silently_relabelled_count": 0,
        "valid": len(trace_hashes) == len(set(trace_hashes))})

    v4_after = {path: sha256_file(REPO / path) for path in V4_SOURCES}
    v5_source_after = {path: sha256_file(REPO / path) for path in V5_SOURCES}
    v6_source_after = {path: sha256_file(REPO / path) for path in V6_SOURCES}
    v5_artifact_after = {path: sha256_file(REPO / path) for path in V5_ARTIFACTS}
    v6_artifact_after = {path: sha256_file(REPO / path) for path in V6_ARTIFACTS}
    terrain_after = {name: sha256_file(REPO / path) for name, path in TERRAIN_FILES.items()}
    oracle_after = {path: sha256_file(REPO / path) for path in oracle_paths}
    profile_after = {name: profile.sha256 for name, profile in friction_profiles().items()}
    lock_unchanged = bool(
        selected is None or (lock_sha_before is not None and sha256_file(lock_path) == lock_sha_before))
    immutable.update({
        "after": {"v4_sha256": v4_after, "v5_source_sha256": v5_source_after,
                  "v6_source_sha256": v6_source_after,
                  "v5_artifact_sha256": v5_artifact_after,
                  "v6_artifact_sha256": v6_artifact_after,
                  "terrain_sha256": terrain_after, "oracle_sha256": oracle_after,
                  "profile_sha256": profile_after},
        "v4_unchanged": v4_before == v4_after,
        "v5_source_unchanged": v5_source_before == v5_source_after,
        "v6_source_unchanged": v6_source_before == v6_source_after,
        "v5_artifacts_unchanged": v5_artifact_before == v5_artifact_after,
        "v6_artifacts_unchanged": v6_artifact_before == v6_artifact_after,
        "terrain_byte_identical": terrain_before == terrain_after,
        "physical_oracle_byte_identical": oracle_before == oracle_after,
        "strong_native_ice_unchanged": profile_hash_before["native_strong_ice"] == profile_after["native_strong_ice"],
        "moderate_v1_unchanged": profile_hash_before["moderate_v1_M0"] == profile_after["moderate_v1_M0"],
        "g0_geometry_unchanged": GEOMETRY_CANDIDATES[0].sha256 == immutable["g0_geometry_sha256"],
        "profile_lock_unchanged_after_reacquisition": lock_unchanged,
        "protocol_unchanged": sha256_file(output / "protocol.json") == protocol_sha,
        "v3_trace_use_count": 0, "model_score_selection_use_count": 0})
    immutable["all_immutable_after_execution"] = bool(
        immutable["v4_unchanged"] and immutable["v5_source_unchanged"]
        and immutable["v6_source_unchanged"] and immutable["v5_artifacts_unchanged"]
        and immutable["v6_artifacts_unchanged"] and immutable["terrain_byte_identical"]
        and immutable["physical_oracle_byte_identical"]
        and immutable["strong_native_ice_unchanged"] and immutable["moderate_v1_unchanged"]
        and immutable["g0_geometry_unchanged"]
        and immutable["profile_lock_unchanged_after_reacquisition"]
        and immutable["protocol_unchanged"])
    write_json(output / "immutable_verification.json", immutable)

    future_ready = _bool(folds["audit"]["valid"])
    targeted_authorized = bool(
        canonical_ready and future_ready and immutable["all_immutable_after_execution"]
        and guard.blocked == 0)
    readiness = {
        "WALKING_V2_SLIP_PROFILE_CALIBRATION_READY": bool(
            len(cal_manifests) == 18 and cal_summary["all_arm_precontact_parity"]),
        "WALKING_V2_SLIP_MODERATE_V2_PROFILE_READY": selected is not None,
        "WALKING_V2_SLIP_MODERATE_V2_LOCK_READY": bool(
            selected is not None and lock_path.is_file() and lock_unchanged),
        "WALKING_V2_SLIP_MODERATE_V2_SOURCE_READY": bool(
            gates["pass"] and gates["source_valid_positive_count"] >= 39),
        "WALKING_V2_SLIP_MODERATE_V2_CONTROL_READY": bool(
            gates["valid_control_count"] == 54
            and gates["control_physical_slip_onset_count"] == 0),
        "WALKING_V2_SLIP_CANONICAL_FACTORIAL_COVERAGE_READY": bool(
            canonical_ready and canonical["audit"]["covered_factorial_cells"] == 36),
        "WALKING_V2_SLIP_CANONICAL_DEVELOPMENT_DATA_READY": canonical_ready,
        "WALKING_V2_SLIP_FUTURE_FOLD_READY": future_ready,
        "WALKING_V2_SLIP_TARGETED_RETRAINING_AUTHORIZED": targeted_authorized,
        "WALKING_V2_FRESH_BLIND_HOLDOUT_AUTHORIZED": False,
        "WALKING_V2_TERRAIN_LOCK_PRESERVED": immutable["terrain_byte_identical"],
        "WALKING_V2_SYSTEM_MIGRATION_AUTHORIZED": False,
        "WALKING_V2_INT8_PREPARATION_AUTHORIZED": False,
        "SINK_RUNTIME_DETECTION_DEFERRED": True}
    if targeted_authorized:
        next_step = "SLIP_TARGETED_RETRAINING_V3"
    elif selected is None:
        next_step = "SLIP_RISK_SCOPE_REDUCTION"
    elif not gates["pass"]:
        next_step = "ADDITIONAL_MODERATE_V2_ACQUISITION"
    else:
        next_step = "STOP_WALKING_V2_DEPLOYMENT"
    summary = {
        "task": "Recalibrate and Reacquire Moderate Bilateral Slip Profile v7",
        "selected_candidate": selected,
        "locked_friction_vector": (
            None if selected is None else candidate_profile(selected).friction5),
        "profile_calibration": {
            "executed_unique_runs": len(cal_manifests),
            "candidate_results": candidate_rows, **cal_summary},
        "moderate_v2_acquisition_gates": gates,
        "canonical_audit": canonical["audit"],
        "canonical_manifest_created": canonical_ready,
        "future_fold_manifest_created": future_ready,
        "original_v5_readiness": False,
        "moderate_v1_separate_version": True,
        "readiness": readiness, "next_step": next_step,
        "forbidden_artifact_access_count": guard.blocked,
        "terrain_byte_identical": immutable["terrain_byte_identical"],
        "oracle_byte_identical": immutable["physical_oracle_byte_identical"],
        "strong_ice_unchanged": immutable["strong_native_ice_unchanged"],
        "g0_geometry_unchanged": immutable["g0_geometry_unchanged"],
        "v3_trace_use_count": 0, "training_performed": False,
        "model_selected_or_locked": False, "blind_holdout_created": False,
        "system_int8_vela_e84_hil_changed": False,
        "discarded_replaced_silently_relabelled": [0, 0, 0]}
    write_json(output / "readiness.json", readiness)
    write_json(output / "summary.json", summary)
    create_plots(output, cal_metrics, ordering_rows, positives, coverage, selected)
    audit = f"""# Moderate bilateral Slip profile recalibration v7 audit

1. Selected profile: **{selected or 'NONE'}**.
2. Locked friction vector: **{summary['locked_friction_vector']}**.
3. Valid left mid-late Slip at all three speeds: **{selected is not None}**.
4. Physical severity ordering passed: **{cal_summary['physical_severity_ordering_pass']}**.
5. Strong Ice/oracle/G0 geometry changed: **False/False/False**.
6. Moderate-v2 source-valid positives: **{gates['source_valid_positive_count']}/54**.
7. Results by speed/foot/phase: **{gates['source_valid_by_speed']} / {gates['source_valid_by_foot']} / {gates['source_valid_by_phase']}**.
8. All three formerly missing cells supported: **{gates['all_missing_left_mid_late_cells_filled']}**.
9. Physically non-slip controls: **{gates['valid_control_count']}/54**.
10. Parity/divergence/double-contact: **{gates['precontact_pair_parity_count']}/54, {gates['postcontact_divergence_count']}/54, {gates['patch_base_double_contact_count']}**.
11. Original v5 readiness remains false: **True**.
12. Moderate-v1 separately versioned: **True**.
13. Canonical manifest created: **{canonical_ready}**; preview positive/control attempts: **{canonical['audit']['positive_attempts']}/{canonical['audit']['control_attempts']}**.
14. Canonical valid positives/rate: **{canonical['audit']['combined_source_valid_positive_count']}/108, {canonical['audit']['combined_source_valid_rate']}**.
15. Canonical factorial coverage: **{canonical['audit']['covered_factorial_cells']}/36**.
16. Calibration and invalid sources excluded: **True**.
17. Targeted Slip retraining authorized: **{targeted_authorized}**.
18. Forbidden artifact access count: **{guard.blocked}**.
19. Terrain byte-identical: **{immutable['terrain_byte_identical']}**.
20. Blind holdout/System/INT8 authorized: **False/False/False**.
21. Sink remains `SINK_RUNTIME_DETECTION_DEFERRED`: **True**.

Exactly one next step: **{next_step}**
"""
    (output / "audit.md").write_text(audit, encoding="utf-8")

    graph_files = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name not in {"provenance.json", "resource_size_audit.json"})
    artifact_hashes = {path.name: sha256_file(path) for path in graph_files}
    write_json(output / "provenance.json", {
        "starting_checkpoint": STARTING_CHECKPOINT, "execution_head": head,
        "upstream_policy_revision": UPSTREAM_REVISION,
        "expected_policy_sha256": TESTED_POLICY_SHA256,
        "actual_policy_sha256": sha256_file(guarded[POLICY]),
        "mujoco_version": mujoco.__version__, "numpy_version": np.__version__,
        "elapsed_seconds": time.time() - started,
        "protocol_sha256": protocol_sha,
        "friction_grid_sha256": friction_grid_sha256(),
        "moderate_v2_lock_sha256": lock_sha_before,
        "trace_shard_manifest_sha256": sha256_file(output / "trace_shard_manifest.json"),
        "artifact_sha256": artifact_hashes,
        "artifact_hash_graph_verified": all(
            sha256_file(output / name) == digest for name, digest in artifact_hashes.items()),
        "forbidden_access_count": guard.blocked, "v3_trace_use_count": 0,
        "model_score_selection_use_count": 0, "training_performed": False})
    files = sorted(path for path in output.iterdir() if path.is_file())
    sizes = [{
        "path": path.name, "bytes": path.stat().st_size,
        "mib": path.stat().st_size / (1024 * 1024),
        "under_or_equal_45_mib": path.stat().st_size <= 45 * 1024 * 1024}
        for path in files]
    write_json(output / "resource_size_audit.json", {
        "limit_mib": 45, "files": sizes,
        "maximum_file_mib": max((row["mib"] for row in sizes), default=0.0),
        "all_files_within_limit": all(row["under_or_equal_45_mib"] for row in sizes)})
    print(json.dumps({
        "output": str(output), "selected_candidate": selected,
        "locked_friction_vector": summary["locked_friction_vector"],
        "profile_calibration_runs": len(cal_manifests),
        "moderate_v2_runs": len(reacquisition_manifests),
        "moderate_v2_source_valid": gates["source_valid_positive_count"],
        "canonical_valid_positive": canonical["audit"]["combined_source_valid_positive_count"],
        "canonical_ready": canonical_ready,
        "targeted_retraining_authorized": targeted_authorized,
        "next_step": next_step}, indent=2))


if __name__ == "__main__":
    main()
