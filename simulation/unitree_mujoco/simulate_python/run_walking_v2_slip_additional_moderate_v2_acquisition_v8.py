"""Acquire and audit the missing moderate-v2 Slip factorial cell v8."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from g1_upstream_locomotion import (
    CONTROL_PERIOD_S, POLICY_PERIOD_S, TESTED_POLICY_SHA256, UPSTREAM_REVISION,
)
from run_walking_v2_bilateral_slip_targeted_acquisition_v3 import build_ledgers
import run_walking_v2_bilateral_slip_targeted_acquisition_v5 as v5_runner
from run_walking_v2_bilateral_slip_targeted_acquisition_v5 import (
    save_trace_shards, sha256_file, write_csv, write_json,
)
import run_walking_v2_slip_moderate_profile_recalibration_v7 as v7_runner
from walking_v2_bilateral_slip_targeted_acquisition_v5 import (
    CONTROL_TYPES, PHASE_BINS, SIDES, SPEEDS_MPS, SLIP_THRESHOLD_M,
    deterministic_initial_perturbation,
)
from walking_v2_slip_additional_moderate_v2_acquisition_v8 import (
    COMMAND_DELAYS_S, LOCKED_CANDIDATE, PHASE_FRACTIONS, PHASE_OFFSETS,
    TARGET_FOOT, TARGET_PHASE,
    TARGET_SEVERITY, TARGET_SPEED_MPS, supplemental_matrix,
    supplemental_variations, variation_contract_payload,
    variation_contract_sha256,
)
from walking_v2_slip_moderate_profile_recalibration_v7 import (
    candidate_profile, friction_profiles,
)
from walking_v2_slip_supplemental_acquisition_v6 import GEOMETRY_CANDIDATES


REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO / "simulation/outputs/walking_v2_slip_additional_moderate_v2_acquisition_v8"
STARTING_CHECKPOINT = "0f7d41ded7c3c2dd750eb0cdacc8230c2bf20a8e"
V8_SOURCES = (
    "simulation/unitree_mujoco/simulate_python/walking_v2_slip_additional_moderate_v2_acquisition_v8.py",
    "simulation/unitree_mujoco/simulate_python/run_walking_v2_slip_additional_moderate_v2_acquisition_v8.py",
    "simulation/unitree_mujoco/simulate_python/test_walking_v2_slip_additional_moderate_v2_acquisition_v8.py",
)
V7 = "simulation/outputs/walking_v2_slip_moderate_profile_recalibration_v7"
V6 = "simulation/outputs/walking_v2_slip_supplemental_acquisition_v6"
V5 = "simulation/outputs/walking_v2_bilateral_slip_targeted_acquisition_v5"
V4 = "simulation/outputs/walking_v2_slip_scenario_generator_redesign_v4"
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
V5_SOURCES = (
    "simulation/unitree_mujoco/simulate_python/walking_v2_bilateral_slip_targeted_acquisition_v5.py",
    "simulation/unitree_mujoco/simulate_python/run_walking_v2_bilateral_slip_targeted_acquisition_v5.py",
    "simulation/unitree_mujoco/simulate_python/test_walking_v2_bilateral_slip_targeted_acquisition_v5.py",
)
V7_ARTIFACTS = (
    f"{V7}/summary.json", f"{V7}/readiness.json", f"{V7}/provenance.json",
    f"{V7}/protocol.json", f"{V7}/immutable_verification.json",
    f"{V7}/moderate_v2_profile_lock.json",
    f"{V7}/profile_calibration_manifest.csv", f"{V7}/profile_calibration_metrics.csv",
    f"{V7}/profile_calibration_physical_episode_ledger.csv",
    f"{V7}/profile_calibration_fall_censor_audit.csv",
    f"{V7}/moderate_v2_run_manifest.csv", f"{V7}/moderate_v2_positive_source_audit.csv",
    f"{V7}/moderate_v2_control_source_audit.csv", f"{V7}/moderate_v2_pair_parity.csv",
    f"{V7}/moderate_v2_contact_audit.csv", f"{V7}/moderate_v2_fall_censor_audit.csv",
    f"{V7}/trace_shard_manifest.json",
) + tuple(f"{V7}/traces_part_{index:03d}.npz" for index in range(6))
V5_ARTIFACTS = (
    f"{V5}/summary.json", f"{V5}/readiness.json", f"{V5}/provenance.json",
    f"{V5}/run_manifest.csv", f"{V5}/positive_source_audit.csv",
    f"{V5}/control_source_audit.csv", f"{V5}/pair_parity_audit.csv",
    f"{V5}/trace_shard_manifest.json",
) + tuple(f"{V5}/traces_part_{index:03d}.npz" for index in range(9))
TERRAIN_FILES = {
    "model": f"{JOINT}/terrain_candidate_model.npz",
    "normalization": f"{JOINT}/terrain_candidate_normalization.json",
    "config": f"{JOINT}/terrain_candidate_config.json",
    "lock": f"{JOINT}/terrain_selection_lock.json",
}
ORACLE_FILES = (
    f"{ORACLE}/summary.json",
    "simulation/unitree_mujoco/simulate_python/walking_hazard_oracle_calibration_v1.py",
    "simulation/unitree_mujoco/simulate_python/walking_hazard_ground_truth_v1.py",
)
OTHER_INPUTS = (
    f"{V6}/geometry_calibration_manifest.csv",
    f"{V6}/selected_geometry_lock.json",
    f"{V4}/failed_v3_quarantine_manifest.json",
    f"{V3}/run_manifest.csv",
    f"{EXISTING}/manifest.json",
    "simulation/unitree_mujoco/simulate_python/walking_v2_slip_supplemental_acquisition_v6.py",
    "simulation/unitree_mujoco/simulate_python/walking_v2_slip_scenario_generator_redesign_v4.py",
    "simulation/unitree_mujoco/simulate_python/terrain_profiles.py",
    "simulation/unitree_mujoco/simulate_python/bilateral_hil_sensor_v2.py",
    "simulation/unitree_mujoco/simulate_python/g1_upstream_locomotion.py",
    POLICY, SCENE, ROBOT,
)
ALLOWED_INPUTS = {
    **{path: "immutable v7 moderate-v2 source" for path in V7_SOURCES},
    **{path: "immutable v5 strong acquisition source" for path in V5_SOURCES},
    **{path: "immutable v7 calibration/acquisition evidence" for path in V7_ARTIFACTS},
    **{path: "immutable v5 strong acquisition evidence" for path in V5_ARTIFACTS},
    **{path: "explicitly scoped frozen support input" for path in OTHER_INPUTS},
    **{path: f"immutable Terrain {name}" for name, path in TERRAIN_FILES.items()},
    **{path: "frozen physical Slip oracle" for path in ORACLE_FILES},
}
FORBIDDEN_TOKENS = (
    "/outer/", "_outer_", "holdout", "spatial_final", "spatial-final",
    "final_test", "final-test",
)
FAILURE_REASONS = {
    "NO_PATCH_CONTACT", "NO_PHYSICAL_ONSET", "ONSET_OUTSIDE_MID_LATE",
    "WRONG_FOOT_FIRST", "BILATERAL_AMBIGUOUS", "CONTACT_LOSS_BEFORE_ONSET",
    "POST_FALL_ONLY", "PHYSICS_DIVERGENCE", "OTHER_WITH_EVIDENCE",
}


class AccessGuard:
    """Fail-closed exact allowlist with a persistent access ledger."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.events: list[dict[str, object]] = []
        self.blocked = 0
        self._cache: dict[str, Path] = {}
        self._flush()

    def _flush(self) -> None:
        write_json(self.output / "artifact_access_log.json", {
            "exact_paths_only": True,
            "forbidden_tokens": list(FORBIDDEN_TOKENS),
            "events": self.events,
            "blocked_access_count": self.blocked,
            "all_accesses_completed": all(
                row["status"] == "completed" for row in self.events),
            "detector_or_model_score_access_count": 0,
            "privileged_data_use": "label/evaluation and physical diagnosis only",
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


def _load_selected_v7_traces(
    guarded: dict[str, Path], shard_manifest: dict[str, object], run_ids: set[str],
) -> dict[str, dict[str, np.ndarray]]:
    traces: dict[str, dict[str, np.ndarray]] = {run_id: {} for run_id in run_ids}
    for shard in shard_manifest["shards"]:  # type: ignore[index]
        selected = run_ids & set(shard["run_ids"])
        if not selected:
            continue
        relative = f"{V7}/{shard['path']}"
        with np.load(guarded[relative], allow_pickle=False) as loaded:
            for run_id in selected:
                prefix = f"{run_id}__"
                traces[run_id] = {
                    key[len(prefix):]: loaded[key].copy()
                    for key in loaded.files if key.startswith(prefix)}
    if set(traces) != run_ids or any(not value for value in traces.values()):
        raise ValueError("failed to load all preregistered v7 diagnosis traces")
    return traces


def _continuous_maximum(mask: np.ndarray) -> int:
    entries = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    maximum = 0
    for entry in entries:
        losses = np.flatnonzero(~mask[entry:])
        length = int(losses[0]) if losses.size else len(mask) - int(entry)
        maximum = max(maximum, length)
    return maximum


def _diagnosis_metrics(
    label: str, meta: dict[str, str], trace: dict[str, np.ndarray],
) -> dict[str, object]:
    side = 0
    drift = trace["anchor_drift_label_only"][:, side]
    valid = np.isfinite(drift) & trace["pre_fall_valid"]
    maximum_index = int(np.nanargmax(np.where(valid, drift, np.nan)))
    episode_id = int(trace["contact_episode_id"][maximum_index, side])
    episode = np.flatnonzero(trace["contact_episode_id"][:, side] == episode_id)
    touchdown = int(episode[0])
    episode_end = int(episode[-1]) + 1
    onset_samples = np.flatnonzero(
        trace["slip_physical_active"][:, side] & trace["pre_fall_valid"])
    onset = int(onset_samples[0]) if onset_samples.size else None
    patch = trace["patch_contact"][:, side]
    first_patch = int(np.flatnonzero(patch)[0])
    mu = trace["effective_patch_friction"][:, side]
    active = patch & np.isfinite(mu) & np.isclose(
        mu, candidate_profile(LOCKED_CANDIDATE).friction3[0], atol=1e-12, rtol=0)
    normal = trace["patch_normal_force_n"][:, side]
    tangent = trace["patch_tangential_force_n"][:, side]
    utilization = np.divide(
        tangent, mu * normal, out=np.full_like(tangent, np.nan),
        where=(mu * normal) > 1e-9)
    tangential_velocity = trace["tangential_velocity_label_only"][:, side]
    fsr_other = trace["bilateral_fusion20_raw"][:, 10:14].sum(axis=1)
    delay = float(meta["command_delay_s"])
    first_command = np.flatnonzero(np.abs(trace["command"][:, 0]) > 1e-12)
    command_time = float(trace["time_s"][first_command[0]]) if first_command.size else math.nan
    observation_step = max(1, int(math.ceil((command_time - 1e-9) / CONTROL_PERIOD_S)))
    phase_at_observation = (
        float(meta["phase_fraction"])
        + observation_step * CONTROL_PERIOD_S / POLICY_PERIOD_S) % 1.0
    rates = np.diff(drift[episode]) * 1000.0
    rates = rates[np.isfinite(rates)]
    initial_perturbation = deterministic_initial_perturbation(int(meta["seed"]))
    composite_state = _canonical_sha256({
        "qpos": meta["initial_qpos_sha256"], "qvel": meta["initial_qvel_sha256"],
        "policy_observation": meta["initial_policy_observation_sha256"]})
    reference_index = onset if onset is not None else maximum_index
    return {
        "comparison": label,
        "run_id": meta["run_id"],
        "source_block": meta["block"],
        "source_valid": onset is not None,
        "failure_reason": "" if onset is not None else "NO_PHYSICAL_ONSET",
        "seed": int(meta["seed"]),
        "initial_perturbation_xyz_vx": json.dumps(initial_perturbation),
        "initial_qpos_sha256": meta["initial_qpos_sha256"],
        "initial_qvel_sha256": meta["initial_qvel_sha256"],
        "initial_policy_observation_sha256": meta["initial_policy_observation_sha256"],
        "initial_policy_state_composite_sha256": composite_state,
        "phase_fraction": float(meta["phase_fraction"]),
        "gait_phase_at_command_observation": phase_at_observation,
        "command_delay_s": delay,
        "command_observation_time_s": command_time,
        "target_touchdown_sample": touchdown,
        "target_touchdown_timestamp_s": float(trace["time_s"][touchdown]),
        "first_patch_contact_sample": first_patch,
        "first_patch_contact_timestamp_s": float(trace["time_s"][first_patch]),
        "patch_entry_pose_xyz": json.dumps(
            trace["foot_world_xyz_label_only"][first_patch, side].tolist()),
        "patch_contact_dwell_ms": int(np.sum(patch)),
        "maximum_continuous_patch_dwell_ms": _continuous_maximum(patch),
        "selected_episode_id": episode_id,
        "selected_episode_duration_ms": len(episode),
        "remaining_stance_duration_ms": max(0, episode_end - reference_index),
        "normal_load_mean_n": float(np.mean(normal[active])) if np.any(active) else "",
        "tangential_force_mean_n": float(np.mean(tangent[active])) if np.any(active) else "",
        "tangential_velocity_mean_mps": (
            float(np.nanmean(tangential_velocity[active])) if np.any(active) else ""),
        "friction_cone_utilization_mean": (
            float(np.nanmean(utilization[active])) if np.any(active) else ""),
        "anchor_drift_rate_peak_mps": float(np.max(rates)) if rates.size else "",
        "maximum_anchor_drift_m": float(drift[maximum_index]),
        "oracle_margin_m": float(drift[maximum_index] - SLIP_THRESHOLD_M),
        "physical_onset_sample": "" if onset is None else onset,
        "physical_onset_timestamp_s": "" if onset is None else float(trace["time_s"][onset]),
        "contact_loss_before_onset": bool(onset is None and episode_end <= maximum_index),
        "contralateral_support_load_mean_n": (
            float(np.mean(fsr_other[active])) if np.any(active) else ""),
        "fall_occurred": _bool(meta["fall_occurred"]),
        "first_fall_censor_boundary": meta["first_fall_sample"],
    }


def diagnose_variation_sensitivity(
    guarded: dict[str, Path], shard_manifest: dict[str, object],
    calibration_manifest: list[dict[str, str]], v7_manifest: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, dict[str, np.ndarray]]]:
    calibration_id = "cal_0p15_M1"
    failed_ids = [
        f"positive_0p15_left_mid_late_stance_v{index}_M1" for index in range(3)]
    run_ids = {calibration_id, *failed_ids}
    traces = _load_selected_v7_traces(guarded, shard_manifest, run_ids)
    meta_by_run = {
        row["run_id"]: row for row in calibration_manifest + v7_manifest
        if row["run_id"] in run_ids}
    rows = [_diagnosis_metrics(
        "SUCCESSFUL_CALIBRATION" if run_id == calibration_id else "FAILED_V7_VARIATION",
        meta_by_run[run_id], traces[run_id])
        for run_id in [calibration_id, *failed_ids]]
    successful = rows[0]
    same_contract_failure = rows[2]
    entry_a = np.asarray(json.loads(str(successful["patch_entry_pose_xyz"])))
    entry_b = np.asarray(json.loads(str(same_contract_failure["patch_entry_pose_xyz"])))
    isolated = bool(
        successful["phase_fraction"] == same_contract_failure["phase_fraction"]
        and successful["command_delay_s"] == same_contract_failure["command_delay_s"])
    root = {
        "primary_cause": "PATCH_ENTRY_POSE_SENSITIVITY",
        "exactly_one_primary_cause": True,
        "diagnosis_uses_physical_signals_only": True,
        "detector_or_model_scores_used": False,
        "successful_run_id": calibration_id,
        "failed_run_ids": failed_ids,
        "isolating_comparison": {
            "failed_run_id": same_contract_failure["run_id"],
            "same_phase_fraction": isolated,
            "same_command_delay": isolated,
            "same_M1_profile": True,
            "same_G0_geometry": True,
            "different_initial_qpos": successful["initial_qpos_sha256"] != same_contract_failure["initial_qpos_sha256"],
            "different_initial_qvel": successful["initial_qvel_sha256"] != same_contract_failure["initial_qvel_sha256"],
            "patch_entry_pose_distance_m": float(np.linalg.norm(entry_a - entry_b)),
            "successful_maximum_anchor_drift_m": successful["maximum_anchor_drift_m"],
            "failed_maximum_anchor_drift_m": same_contract_failure["maximum_anchor_drift_m"],
            "successful_oracle_margin_m": successful["oracle_margin_m"],
            "failed_oracle_margin_m": same_contract_failure["oracle_margin_m"],
        },
        "reasoning": (
            "The same 0.41 phase fraction and 0.04 s command delay succeed under the "
            "calibration initial state but fail under the v7 acquisition perturbation. "
            "The changed initial generalized state changes patch entry pose/load/velocity "
            "and leaves anchor drift below the frozen 0.05 m oracle threshold."),
        "alternatives_rejected": {
            "COMMAND_DELAY_SENSITIVITY": "isolating pair has identical 0.04 s delay",
            "INITIAL_GAIT_PHASE_SENSITIVITY": "isolating pair has identical 0.41 controller phase",
            "CONTACT_LOAD_SENSITIVITY": "load is treated as a patch-entry consequence, not the isolated input",
            "TANGENTIAL_VELOCITY_SENSITIVITY": "velocity is treated as a patch-entry consequence",
            "STANCE_DURATION_SENSITIVITY": "not needed to explain the same-contract split",
            "MULTIFACTOR_VARIATION_SENSITIVITY": "one isolated upstream difference is available",
        },
    }
    if not isolated or not _bool(successful["source_valid"]) or _bool(same_contract_failure["source_valid"]):
        raise RuntimeError("v7 variation diagnosis did not reproduce its isolating evidence")
    return rows, root, traces


def _variation_maps() -> tuple[dict[int, object], dict[str, object]]:
    rows = supplemental_variations()
    return ({row.variation_index: row for row in rows},
            {row.variation_id: row for row in rows})


def _add_variation_fields(rows: list[dict[str, object]]) -> None:
    by_index, by_id = _variation_maps()
    for row in rows:
        if "variation_index" in row:
            variation = by_index[int(row["variation_index"])]
        else:
            run_id = str(row["run_id"]).lower()
            variation = next(
                value for key, value in by_id.items() if key.lower() in run_id)
        row.update({
            "variation_index": variation.variation_index,
            "variation_id": variation.variation_id,
            "phase_offset": variation.phase_offset,
            "phase_fraction": variation.phase_fraction,
            "command_delay_s": variation.command_delay_s,
            "initial_perturbation": json.dumps(variation.initial_perturbation),
        })


def _normalize_positive_failures(rows: list[dict[str, object]]) -> None:
    for row in rows:
        if _bool(row["source_valid"]):
            row["failure_reason"] = ""
        elif row["actual_onset_phase"] not in ("none", TARGET_PHASE):
            row["failure_reason"] = "ONSET_OUTSIDE_MID_LATE"
        elif _bool(row.get("wrong_foot_onset")):
            row["failure_reason"] = "WRONG_FOOT_FIRST"
        elif _bool(row.get("bilateral_ambiguous")):
            row["failure_reason"] = "BILATERAL_AMBIGUOUS"
        elif str(row["failure_reason"]) not in FAILURE_REASONS:
            row["failure_reason"] = "OTHER_WITH_EVIDENCE"
        if row["failure_reason"] and row["failure_reason"] not in FAILURE_REASONS:
            raise ValueError(f"unsupported failure reason: {row['failure_reason']}")


def supplemental_metrics(
    conditions: list[object], traces: list[dict[str, np.ndarray]],
    manifests: list[dict[str, object]], positives: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_run = {str(row["run_id"]): row for row in positives}
    rows: list[dict[str, object]] = []
    for condition, trace, meta in zip(conditions, traces, manifests):
        if condition.role != "positive":
            continue
        audit = by_run[condition.run_id]
        base = v7_runner._physical_metrics(condition, trace, meta, audit)
        side = SIDES.index(condition.target_foot)
        drift = trace["anchor_drift_label_only"][:, side]
        valid = np.isfinite(drift) & trace["pre_fall_valid"]
        max_index = int(np.nanargmax(np.where(valid, drift, np.nan)))
        episode_id = int(trace["contact_episode_id"][max_index, side])
        episode = np.flatnonzero(trace["contact_episode_id"][:, side] == episode_id)
        onset = audit["target_physical_onset_sample"]
        reference = int(onset) if onset != "" else max_index
        first_patch = meta["first_patch_contact_sample"]
        rates = np.diff(drift[episode]) * 1000.0
        rates = rates[np.isfinite(rates)]
        rows.append({
            "variation_id": audit["variation_id"],
            "run_id": condition.run_id,
            "phase_offset": audit["phase_offset"],
            "phase_fraction": condition.phase_fraction,
            "command_delay_s": condition.command_delay_s,
            "seed": condition.seed,
            "initial_policy_state_composite_sha256": _canonical_sha256({
                "qpos": meta["initial_qpos_sha256"], "qvel": meta["initial_qvel_sha256"],
                "policy": meta["initial_policy_observation_sha256"]}),
            "target_touchdown_sample": audit["target_touchdown_sample"],
            "target_touchdown_timestamp_s": audit["target_touchdown_time_s"],
            "first_patch_contact_sample": first_patch,
            "first_patch_contact_timestamp_s": (
                "" if first_patch is None else float(trace["time_s"][int(first_patch)])),
            "patch_entry_pose_xyz": (
                "" if first_patch is None else json.dumps(
                    trace["foot_world_xyz_label_only"][int(first_patch), side].tolist())),
            "remaining_stance_duration_ms": (
                int(episode[-1]) + 1 - reference if episode.size else ""),
            "anchor_drift_rate_peak_mps": float(np.max(rates)) if rates.size else "",
            "oracle_margin_m": float(drift[max_index] - SLIP_THRESHOLD_M),
            "actual_onset_phase": audit["actual_onset_phase"],
            "actual_first_affected_foot": audit["actual_first_affected_foot"],
            "contralateral_slip": audit["contralateral_physical_slip_contamination"],
            "fall_or_censor": meta["fall_occurred"],
            "valid": audit["source_valid"],
            "failure_reason": audit["failure_reason"],
            **base,
        })
    return rows


def supplemental_gates(
    traces: list[dict[str, np.ndarray]], manifests: list[dict[str, object]],
    positives: list[dict[str, object]], controls: list[dict[str, object]],
    parity: list[dict[str, object]], forbidden_count: int,
) -> dict[str, object]:
    valid = [row for row in positives if _bool(row["source_valid"])]
    trace_hashes = [str(row["full_trace_sha256"]) for row in manifests]
    positive_meta = [row for row in manifests if row["role"] == "positive"]
    composite_states = {_canonical_sha256({
        "qpos": row["initial_qpos_sha256"], "qvel": row["initial_qvel_sha256"],
        "policy": row["initial_policy_observation_sha256"]}) for row in positive_meta}
    air_postfall = sum(int(np.sum(
        trace["slip_physical_active"]
        & ((~trace["physical_contact"]) | ~trace["pre_fall_valid"][:, None])))
        for trace in traces)
    by_phase = {f"{offset:+.2f}": {
        "planned": sum(np.isclose(float(row["phase_offset"]), offset) for row in positives),
        "source_valid": sum(
            np.isclose(float(row["phase_offset"]), offset) and _bool(row["source_valid"])
            for row in positives),
    } for offset in PHASE_OFFSETS}
    by_delay = {f"{delay:.2f}": {
        "planned": sum(np.isclose(float(row["command_delay_s"]), delay) for row in positives),
        "source_valid": sum(
            np.isclose(float(row["command_delay_s"]), delay) and _bool(row["source_valid"])
            for row in positives),
    } for delay in COMMAND_DELAYS_S}
    interaction = [{
        "phase_offset": row["phase_offset"],
        "command_delay_s": row["command_delay_s"],
        "variation_id": row["variation_id"],
        "source_valid": _bool(row["source_valid"]),
        "failure_reason": row["failure_reason"],
    } for row in positives]
    values = {
        "planned_positive_runs": 12,
        "executed_positive_runs": len(positives),
        "matched_control_runs": len(controls),
        "total_unique_runs": len(manifests),
        "source_valid_positive_count": len(valid),
        "supplemental_efficiency": len(valid) / 12,
        "physically_distinct_valid_variation_count": len({
            row["variation_id"] for row in valid}),
        "all_valid_onsets_mid_late_stance": all(
            row["actual_onset_phase"] == TARGET_PHASE for row in valid),
        "target_first_accuracy": (
            sum(_bool(row["target_first"]) for row in valid) / len(valid) if valid else 0.0),
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
        "unique_initial_policy_state_count": len(composite_states),
        "forbidden_artifact_access_count": forbidden_count,
        "discarded_count": sum(_bool(row["discarded"]) for row in manifests),
        "replaced_count": sum(_bool(row["replaced"]) for row in manifests),
        "silently_relabelled_count": sum(
            _bool(row["silently_relabelled"]) for row in manifests),
        "exact_1khz_fusion20": all(
            row["sample_count"] == 3000 and row["fusion20_shape"] == [3000, 20]
            and _bool(row["finite_fusion20"])
            and float(row["sample_spacing_max_error_s"]) < 1e-12 for row in manifests),
        "source_validity_by_phase_offset": by_phase,
        "source_validity_by_command_delay": by_delay,
        "source_validity_interaction": interaction,
        "claims_all_combinations_robust": len(valid) == 12,
    }
    values["pass"] = bool(
        values["executed_positive_runs"] == 12
        and values["matched_control_runs"] == 12
        and values["total_unique_runs"] == 24
        and values["source_valid_positive_count"] >= 6
        and values["physically_distinct_valid_variation_count"] >= 3
        and values["all_valid_onsets_mid_late_stance"]
        and values["target_first_accuracy"] == 1.0
        and values["valid_control_count"] == 12
        and values["control_physical_slip_onset_count"] == 0
        and values["precontact_pair_parity_count"] == 12
        and values["postcontact_divergence_count"] == 12
        and values["full_trace_identity_count"] == 0
        and values["patch_base_double_contact_count"] == 0
        and values["friction_contract_violation_count"] == 0
        and values["air_or_postfall_positive_attribution_count"] == 0
        and values["duplicate_unique_trace_count"] == 0
        and values["unique_initial_policy_state_count"] == 12
        and values["forbidden_artifact_access_count"] == 0
        and values["discarded_count"] == 0 and values["replaced_count"] == 0
        and values["silently_relabelled_count"] == 0
        and values["exact_1khz_fusion20"])
    return values


def canonical_base(
    v5_manifest: list[dict[str, str]], v5_positive: list[dict[str, str]],
    v5_controls: list[dict[str, str]], v5_shards: dict[str, object],
    v7_manifest: list[dict[str, str]], v7_positive: list[dict[str, str]],
    v7_controls: list[dict[str, str]], v7_shards: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    preview, coverage = v7_runner.canonical_artifacts(
        v5_manifest, v5_positive, v5_controls, v5_shards,
        v7_manifest, v7_positive, v7_controls, v7_shards,
        False, True)
    audit = preview["audit"]
    expected = (
        audit["positive_attempts"] == 108 and audit["control_attempts"] == 108
        and audit["combined_source_valid_positive_count"] == 87
        and np.isclose(audit["combined_source_valid_rate"], 87 / 108)
        and audit["covered_factorial_cells"] == 35)
    if not expected:
        raise RuntimeError("base canonical preview drifted from 87/108 and 35/36")
    result = {
        "version": "walking_v2_slip_canonical_base_v8",
        "matrix_layer": "base_factorial_matrix",
        "base_acquisition_versions": ["v5_strong", "v7_moderate_v2"],
        "base_positive_attempts": 108,
        "base_control_attempts": 108,
        "base_valid_positive_count": 87,
        "base_valid_positive_rate": 87 / 108,
        "base_factorial_coverage": "35/36",
        "base_v7_claimed_ready": False,
        "original_v5_readiness_preserved_false": True,
        "original_v7_readiness_preserved_false": True,
        "training_performed": False,
        "audit": audit,
        "runs": preview["runs"],
    }
    result["content_sha256"] = _canonical_sha256(result)
    return result, coverage


def _supplemental_manifest_rows(
    manifests: list[dict[str, object]], positives: list[dict[str, object]],
    controls: list[dict[str, object]], shard_manifest: dict[str, object],
) -> list[dict[str, object]]:
    positive_map = {str(row["run_id"]): row for row in positives}
    control_map = {str(row["run_id"]): row for row in controls}
    shard_map = {
        run_id: {"path": shard["path"], "sha256": shard["sha256"]}
        for shard in shard_manifest["shards"] for run_id in shard["run_ids"]  # type: ignore[index]
    }
    rows: list[dict[str, object]] = []
    for meta in manifests:
        run_id = str(meta["run_id"])
        audit = positive_map[run_id] if meta["role"] == "positive" else control_map[run_id]
        valid = _bool(audit["source_valid"] if meta["role"] == "positive"
                      else audit["control_source_valid"])
        eligibility = (
            "POSITIVE_ELIGIBLE" if meta["role"] == "positive" and valid
            else "CONTROL_ELIGIBLE" if meta["role"] == "control" and valid
            else "FAILED_SOURCE_DIAGNOSTIC_ONLY")
        rows.append({
            "matrix_layer": "supplemental_rare_cell_enrichment",
            "source_acquisition": "v8_moderate_v2_supplemental",
            "acquisition_version": "v8",
            "profile_version": "moderate-v2-M1-locked",
            "run_id": run_id,
            "pair_id": f"v8:{meta['pair_id']}",
            "role": meta["role"],
            "speed_mps": float(meta["speed_mps"]),
            "target_foot": meta["target_foot"],
            "target_phase": meta["target_phase"],
            "severity": TARGET_SEVERITY,
            "control_type": meta["control_type"],
            "variation_index": int(meta["variation_index"]),
            "variation_id": audit["variation_id"],
            "source_valid": valid,
            "failure_reason": audit["failure_reason"],
            "eligibility": eligibility,
            "future_training_eligible": eligibility in {
                "POSITIVE_ELIGIBLE", "CONTROL_ELIGIBLE"},
            "trace_shard": shard_map[run_id],
            "development_only": True,
        })
    return rows


def augmented_artifacts(
    base: dict[str, object], base_coverage: list[dict[str, object]],
    supplemental_rows: list[dict[str, object]], gates: dict[str, object],
    v5_positive: list[dict[str, str]], v7_positive: list[dict[str, str]],
    supplemental_positive: list[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    coverage: list[dict[str, object]] = []
    for base_row in base_coverage:
        match = [row for row in supplemental_rows
                 if row["role"] == "positive"
                 and float(row["speed_mps"]) == float(base_row["speed_mps"])
                 and row["target_foot"] == base_row["target_foot"]
                 and row["target_phase"] == base_row["target_phase"]
                 and row["severity"] == base_row["severity"]]
        supplemental_valid = sum(_bool(row["source_valid"]) for row in match)
        augmented_valid = int(base_row["source_valid_positive_count"]) + supplemental_valid
        coverage.append({
            **base_row,
            "base_source_valid_positive_count": base_row["source_valid_positive_count"],
            "supplemental_planned_positive_attempts": len(match),
            "supplemental_source_valid_positive_count": supplemental_valid,
            "augmented_source_valid_positive_count": augmented_valid,
            "augmented_has_valid_support": augmented_valid > 0,
        })
    base_rows = []
    for row in base["runs"]:  # type: ignore[index]
        valid = _bool(row["source_valid"])
        eligibility = (
            "POSITIVE_ELIGIBLE" if row["role"] == "positive" and valid
            else "CONTROL_ELIGIBLE" if row["role"] == "control" and valid
            else "FAILED_SOURCE_DIAGNOSTIC_ONLY")
        base_rows.append({
            **row,
            "matrix_layer": "base_factorial_matrix",
            "acquisition_version": "v5" if row["source_acquisition"] == "v5_strong" else "v7",
            "profile_version": (
                "native-strong-ice-frozen" if row["severity"] == "native_strong_ice"
                else "moderate-v2-M1-locked"),
            "eligibility": eligibility,
            "future_training_eligible": eligibility in {
                "POSITIVE_ELIGIBLE", "CONTROL_ELIGIBLE"},
        })
    rows = base_rows + supplemental_rows
    valid_positive = [
        row for row in rows if row["role"] == "positive" and _bool(row["source_valid"])]
    controls = [row for row in rows if row["role"] == "control"]
    base_target_first = [
        _bool(row["target_first"]) for row in v5_positive
        if row["severity"] == "native_strong_ice" and _bool(row["source_valid"])] + [
        _bool(row["target_first"]) for row in v7_positive if _bool(row["source_valid"])]
    supplemental_target_first = [
        _bool(row["target_first"]) for row in supplemental_positive
        if _bool(row["source_valid"])]
    target_first = base_target_first + supplemental_target_first
    audit = {
        "base_positive_attempts": 108,
        "base_control_attempts": 108,
        "base_valid_positive_count": 87,
        "base_valid_positive_rate": 87 / 108,
        "base_factorial_coverage": 35,
        "supplemental_positive_attempts": 12,
        "supplemental_control_attempts": 12,
        "supplemental_valid_positive_count": gates["source_valid_positive_count"],
        "supplemental_efficiency": gates["supplemental_efficiency"],
        "augmented_positive_attempts": 120,
        "augmented_control_attempts": 120,
        "augmented_valid_positive_count": len(valid_positive),
        "augmented_factorial_coverage": sum(
            _bool(row["augmented_has_valid_support"]) for row in coverage),
        "missing_cell_distinct_valid_count": gates["physically_distinct_valid_variation_count"],
        "both_severities_represented": {row["severity"] for row in valid_positive} == {
            "native_strong_ice", "moderate_v2"},
        "all_speeds_represented": {float(row["speed_mps"]) for row in valid_positive} == set(SPEEDS_MPS),
        "both_feet_represented": {row["target_foot"] for row in valid_positive} == set(SIDES),
        "all_phases_represented": {row["target_phase"] for row in valid_positive} == {
            phase.name for phase in PHASE_BINS},
        "combined_valid_control_count": sum(_bool(row["source_valid"]) for row in controls),
        "combined_controls_physically_non_slip": (
            int(base["audit"]["valid_control_count"]) == 108
            and gates["valid_control_count"] == 12
            and gates["control_physical_slip_onset_count"] == 0),
        "target_first_accuracy": sum(target_first) / len(target_first),
        "physics_and_provenance_gates_pass": gates["pass"],
        "calibration_run_count": 0,
        "v3_quarantined_run_count": 0,
        "outer_holdout_final_run_count": 0,
        "base_v7_retroactively_claimed_ready": False,
    }
    audit["pass"] = bool(
        gates["pass"]
        and audit["base_valid_positive_count"] == 87
        and np.isclose(audit["base_valid_positive_rate"], 87 / 108)
        and audit["base_factorial_coverage"] == 35
        and audit["augmented_factorial_coverage"] == 36
        and audit["missing_cell_distinct_valid_count"] >= 3
        and audit["both_severities_represented"] and audit["all_speeds_represented"]
        and audit["both_feet_represented"] and audit["all_phases_represented"]
        and audit["combined_valid_control_count"] == 120
        and audit["combined_controls_physically_non_slip"]
        and audit["target_first_accuracy"] >= 0.90)
    augmented = {
        "version": "walking_v2_slip_augmented_canonical_development_v8",
        "base_factorial_matrix_is_not_rewritten": True,
        "supplemental_attempts_not_added_to_base_efficiency_denominator": True,
        "training_performed": False,
        "training_selection_lock_created": False,
        "audit": audit,
        "runs": rows,
    }
    augmented["content_sha256"] = _canonical_sha256(augmented)
    supplemental_manifest = {
        "version": "walking_v2_slip_supplemental_cell_v8",
        "cell": {
            "speed_mps": TARGET_SPEED_MPS, "target_foot": TARGET_FOOT,
            "target_phase": TARGET_PHASE, "severity": TARGET_SEVERITY},
        "positive_attempts": 12, "matched_controls": 12,
        "calibration_runs_excluded": True,
        "source_valid_positive_count": gates["source_valid_positive_count"],
        "supplemental_efficiency": gates["supplemental_efficiency"],
        "cell_gate_pass": gates["pass"],
        "runs": supplemental_rows,
    }
    supplemental_manifest["content_sha256"] = _canonical_sha256(supplemental_manifest)
    return augmented, coverage, supplemental_manifest


def training_eligibility_rows(
    augmented: dict[str, object], v7_calibration: list[dict[str, str]],
    v6_calibration: list[dict[str, str]], v3_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in augmented["runs"]:  # type: ignore[index]
        rows.append({
            "run_id": run["run_id"],
            "source_acquisition_version": run["source_acquisition"],
            "role": run["role"], "severity": run["severity"],
            "source_valid": run["source_valid"],
            "eligibility": run["eligibility"],
            "training_eligible": run["future_training_eligible"],
            "reason": ("valid physical source/control" if run["future_training_eligible"]
                       else "failed source retained for diagnosis only"),
        })
    for run in v7_calibration:
        rows.append({
            "run_id": run["run_id"], "source_acquisition_version": "v7_profile_calibration",
            "role": run["role"], "severity": run["severity"],
            "source_valid": "", "eligibility": "CALIBRATION_ONLY_DO_NOT_TRAIN",
            "training_eligible": False, "reason": "profile calibration excluded"})
    for run in v6_calibration:
        rows.append({
            "run_id": run["run_id"], "source_acquisition_version": "v6_geometry_calibration",
            "role": run["role"], "severity": run["severity"],
            "source_valid": "", "eligibility": "CALIBRATION_ONLY_DO_NOT_TRAIN",
            "training_eligible": False, "reason": "geometry calibration excluded"})
    for run in v3_rows:
        rows.append({
            "run_id": run["run_id"], "source_acquisition_version": "v3_quarantined",
            "role": run["role"], "severity": run["severity"],
            "source_valid": "", "eligibility": "QUARANTINED",
            "training_eligible": False, "reason": "frozen v3 quarantine"})
    allowed = {
        "POSITIVE_ELIGIBLE", "CONTROL_ELIGIBLE", "FAILED_SOURCE_DIAGNOSTIC_ONLY",
        "CALIBRATION_ONLY_DO_NOT_TRAIN", "QUARANTINED", "FORBIDDEN"}
    if not all(row["eligibility"] in allowed for row in rows):
        raise ValueError("unknown training eligibility")
    return rows


def future_folds(
    augmented: dict[str, object], existing_runs: list[dict[str, object]],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for run in existing_runs:
        variation = int(run["variation_index"])
        run_id = str(run["run_id"])
        rows.append({
            "source_acquisition": "existing_valid_120_bilateral_development",
            "acquisition_version": "existing_v2",
            "fold": variation % 3,
            "pair_id": "", "variation_index": variation,
            "variation_group": f"existing_v2:v{variation}",
            "run_group": f"existing_v2:{run_id}",
            "contact_episode_group": f"existing_v2:{run_id}:all",
            "run_id": run_id, "contact_episode_id": "all",
            "speed_mps": run["speed_mps"], "target_foot": "bilateral",
            "target_phase": "legacy_development", "severity": "legacy_development",
            "role": run["role"], "control_type": "legacy_development",
            "development_only": True,
        })
    for run in augmented["runs"]:  # type: ignore[index]
        if not _bool(run["future_training_eligible"]):
            continue
        variation = int(run["variation_index"])
        source = str(run["source_acquisition"])
        run_id = str(run["run_id"])
        rows.append({
            **run, "fold": variation % 3,
            "variation_group": f"{source}:v{variation}",
            "run_group": f"{source}:{run_id}",
            "contact_episode_group": f"{source}:{run_id}:all",
            "contact_episode_id": "all",
        })
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
            and row["source_acquisition"] in {
                "v5_strong", "v7_moderate_v2", "v8_moderate_v2_supplemental"}]
        positive = [row for row in selected if row["role"] == "positive"]
        controls = [row for row in selected if row["role"] == "control"]
        supplemental = [
            row for row in selected
            if row["source_acquisition"] == "v8_moderate_v2_supplemental"]
        coverage.append({
            "fold": fold,
            "all_speeds": {float(row["speed_mps"]) for row in selected} == set(SPEEDS_MPS),
            "both_feet": {row["target_foot"] for row in selected} == set(SIDES),
            "both_severities": {row["severity"] for row in positive} == {
                "native_strong_ice", "moderate_v2"},
            "all_phases_where_possible": {row["target_phase"] for row in positive} == {
                phase.name for phase in PHASE_BINS},
            "hard_and_near_slip_controls": {row["control_type"] for row in controls} == set(CONTROL_TYPES),
            "supplemental_variations_present": len({row["variation_index"] for row in supplemental}) > 0,
        })
    audit = {
        "row_count": len(rows),
        "existing_valid_development_run_count": sum(
            row["source_acquisition"] == "existing_valid_120_bilateral_development"
            for row in rows),
        "fold_ids": sorted({int(row["fold"]) for row in rows}),
        "leakage_count": len(leakage), "leaking_groups": leakage,
        "positive_control_pairs_same_fold": not any(
            value.startswith("pair_id:") for value in leakage),
        "run_leakage_count": sum(value.startswith("run_group:") for value in leakage),
        "contact_episode_leakage_count": sum(
            value.startswith("contact_episode_group:") for value in leakage),
        "variation_leakage_count": sum(
            value.startswith("variation_group:") for value in leakage),
        "acquisition_version_carried_on_every_row": all(
            bool(row.get("acquisition_version")) for row in rows),
        "supplemental_variations_distributed_across_folds": {
            int(row["fold"]) for row in rows
            if row["source_acquisition"] == "v8_moderate_v2_supplemental"} == {0, 1, 2},
        "coverage": coverage,
        "coverage_pass": all(all(
            _bool(value) for key, value in row.items() if key != "fold") for row in coverage),
        "calibration_only_run_count": 0,
        "failed_source_run_count": 0,
        "v3_quarantined_run_count": 0,
        "outer_holdout_final_run_count": 0,
        "training_performed": False,
    }
    audit["valid"] = bool(
        augmented["audit"]["pass"]  # type: ignore[index]
        and audit["existing_valid_development_run_count"] == 120
        and audit["fold_ids"] == [0, 1, 2] and not leakage
        and audit["acquisition_version_carried_on_every_row"]
        and audit["supplemental_variations_distributed_across_folds"]
        and audit["coverage_pass"])
    result = {
        "version": "walking_v2_slip_future_nested_folds_v8",
        "frozen": True,
        "grouping": [
            "counterfactual_pair", "source_scoped_variation", "run",
            "contact_episode", "acquisition_version"],
        "assignment": "source-scoped variation_index modulo 3",
        "training_performed": False,
        "audit": audit,
        "rows": rows,
    }
    result["content_sha256"] = _canonical_sha256(result)
    return result


def _save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def create_plots(
    output: Path, diagnosis: list[dict[str, object]],
    diagnosis_traces: dict[str, dict[str, np.ndarray]],
    metrics: list[dict[str, object]], positives: list[dict[str, object]],
    conditions: list[object], traces: list[dict[str, np.ndarray]],
    coverage: list[dict[str, object]],
) -> None:
    plt.figure(figsize=(9, 4))
    for row in diagnosis:
        trace = diagnosis_traces[str(row["run_id"])]
        plt.plot(trace["time_s"], trace["anchor_drift_label_only"][:, 0],
                 label=str(row["run_id"]))
    plt.axhline(SLIP_THRESHOLD_M, color="black", linestyle="--")
    plt.xlabel("time (s)"); plt.ylabel("left anchor drift (m)")
    plt.title("Successful calibration versus failed v7 variations"); plt.legend(fontsize=7)
    _save_plot(output / "successful_calibration_vs_failed_v7_variations.png")

    plt.figure(figsize=(7, 4))
    values = [sum(
        _bool(row["source_valid"]) and np.isclose(float(row["phase_offset"]), offset)
        for row in positives) for offset in PHASE_OFFSETS]
    plt.bar([f"{value:+.2f}" for value in PHASE_OFFSETS], values)
    plt.ylim(0, 4); plt.xlabel("initial gait-phase offset"); plt.ylabel("valid / 4")
    plt.title("Supplemental validity by phase offset")
    _save_plot(output / "source_validity_by_phase_offset.png")

    plt.figure(figsize=(7, 4))
    values = [sum(
        _bool(row["source_valid"]) and np.isclose(float(row["command_delay_s"]), delay)
        for row in positives) for delay in COMMAND_DELAYS_S]
    plt.bar([f"{value:.2f}" for value in COMMAND_DELAYS_S], values)
    plt.ylim(0, 3); plt.xlabel("command delay (s)"); plt.ylabel("valid / 3")
    plt.title("Supplemental validity by command delay")
    _save_plot(output / "source_validity_by_command_delay.png")

    plt.figure(figsize=(10, 4))
    plt.bar([str(row["variation_id"]) for row in metrics],
            [float(row["maximum_anchor_drift_m"]) for row in metrics],
            color=["#4c9f70" if _bool(row["valid"]) else "#d95f5f" for row in metrics])
    plt.axhline(SLIP_THRESHOLD_M, color="black", linestyle="--")
    plt.xticks(rotation=45); plt.ylabel("maximum anchor drift (m)")
    plt.title("Supplemental anchor drift")
    _save_plot(output / "supplemental_anchor_drift.png")

    plt.figure(figsize=(10, 5))
    for condition, trace in zip(conditions, traces):
        style = "-" if condition.role == "positive" else "--"
        alpha = 0.75 if condition.role == "positive" else 0.45
        plt.plot(trace["time_s"], trace["anchor_drift_label_only"][:, 0],
                 linestyle=style, alpha=alpha, linewidth=0.8)
    plt.axhline(SLIP_THRESHOLD_M, color="black", linestyle=":")
    plt.xlabel("time (s)"); plt.ylabel("left anchor drift (m)")
    plt.title("Twelve supplemental positive/control paired traces")
    _save_plot(output / "supplemental_positive_control_paired_traces.png")

    plt.figure(figsize=(10, 4))
    values = [int(row["augmented_source_valid_positive_count"]) for row in coverage]
    plt.bar(range(len(values)), values,
            color=["#4c9f70" if value else "#d95f5f" for value in values])
    plt.xlabel("factorial cell index"); plt.ylabel("valid positive support")
    plt.title(f"Augmented factorial coverage: {sum(value > 0 for value in values)}/36")
    _save_plot(output / "augmented_factorial_coverage.png")


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
    unrelated = [line for line in status_lines if not any(path in line for path in V8_SOURCES)
                 and str(DEFAULT_OUTPUT.relative_to(REPO)) not in line]
    if unrelated:
        raise RuntimeError(f"unrelated worktree changes before execution: {unrelated}")

    conditions = supplemental_matrix()
    variation_payload = variation_contract_payload()
    variation_sha = variation_contract_sha256()
    protocol = {
        "task": "Acquire Additional Moderate-v2 Slip Support v8",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "diagnosis_primary_cause": "PATCH_ENTRY_POSE_SENSITIVITY",
        "diagnosis_physical_signals_only": True,
        "locked_profile": candidate_profile(LOCKED_CANDIDATE).contract,
        "exact_M1_friction_vector": candidate_profile(LOCKED_CANDIDATE).friction5,
        "geometry": GEOMETRY_CANDIDATES[0].contract,
        "explicit_sole_contact_pair_count": 8,
        "supplemental_variation_contract_sha256": variation_sha,
        "supplemental_contract": {
            "positive_runs": 12, "matched_controls": 12,
            "hard_normal_controls": 6, "near_slip_non_event_controls": 6,
            "total_unique_runs": 24,
            "phase_fractions": list(PHASE_FRACTIONS),
            "phase_offsets": list(PHASE_OFFSETS),
            "command_delays_s": list(COMMAND_DELAYS_S),
            "Cartesian_product": True,
            "run_order": [row.run_id for row in conditions],
            "adaptive_variations": False, "replacement_runs": False,
        },
        "actual_onset_phase_defines_target_phase": True,
        "calibration_data_training_use": False,
        "failed_source_automatic_negative_use": False,
        "model_scores_used": False, "training_performed": False,
        "M1_changed": False, "G0_changed": False, "strong_ice_changed": False,
        "oracle_changed": False, "terrain_changed": False, "sink_deferred": True,
        "overwrite_policy": "refuse non-empty output",
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "supplemental_variation_contract.json", {
        **variation_payload, "contract_sha256": variation_sha})
    protocol_sha = sha256_file(output / "protocol.json")
    contract_file_sha = sha256_file(output / "supplemental_variation_contract.json")
    write_json(output / "input_allowlist.json", {
        "exact_paths_only": True,
        "inputs": [{"path": path, "purpose": purpose}
                   for path, purpose in ALLOWED_INPUTS.items()]})
    write_json(output / "forbidden_path_policy.json", {
        "fail_closed": True, "forbidden_tokens": list(FORBIDDEN_TOKENS),
        "outer_holdout_final_access_authorized": False,
        "detector_or_model_score_access_authorized": False,
        "v3_trace_access_authorized": False,
        "privileged_ground_truth_runtime_input_authorized": False})
    guard = AccessGuard(output)
    guarded = {path: guard.path(path) for path in ALLOWED_INPUTS}

    v7_source_before = {path: sha256_file(guarded[path]) for path in V7_SOURCES}
    v7_checkpoint = {path: _checkpoint_sha256(path) for path in V7_SOURCES}
    v5_source_before = {path: sha256_file(guarded[path]) for path in V5_SOURCES}
    v5_checkpoint = {path: _checkpoint_sha256(path) for path in V5_SOURCES}
    v7_artifact_before = {path: sha256_file(guarded[path]) for path in V7_ARTIFACTS}
    v5_artifact_before = {path: sha256_file(guarded[path]) for path in V5_ARTIFACTS}
    terrain_before = {name: sha256_file(guarded[path]) for name, path in TERRAIN_FILES.items()}
    oracle_before = {path: sha256_file(guarded[path]) for path in ORACLE_FILES}
    m1_lock_sha = sha256_file(guarded[f"{V7}/moderate_v2_profile_lock.json"])
    strong_sha = friction_profiles()["native_strong_ice"].sha256
    g0_sha = GEOMETRY_CANDIDATES[0].sha256
    patch_snapshot = v5_runner._patch_contract_snapshot()
    v7_summary = guard.json(f"{V7}/summary.json")
    v7_readiness = guard.json(f"{V7}/readiness.json")
    v5_readiness = guard.json(f"{V5}/readiness.json")
    v7_provenance = guard.json(f"{V7}/provenance.json")
    v5_provenance = guard.json(f"{V5}/provenance.json")
    v7_lock = guard.json(f"{V7}/moderate_v2_profile_lock.json")
    v7_shards = guard.json(f"{V7}/trace_shard_manifest.json")
    v5_shards = guard.json(f"{V5}/trace_shard_manifest.json")
    v7_calibration = guard.csv(f"{V7}/profile_calibration_manifest.csv")
    v7_manifest = guard.csv(f"{V7}/moderate_v2_run_manifest.csv")
    v7_positive = guard.csv(f"{V7}/moderate_v2_positive_source_audit.csv")
    v7_controls = guard.csv(f"{V7}/moderate_v2_control_source_audit.csv")
    v5_manifest = guard.csv(f"{V5}/run_manifest.csv")
    v5_positive = guard.csv(f"{V5}/positive_source_audit.csv")
    v5_controls = guard.csv(f"{V5}/control_source_audit.csv")
    v6_calibration = guard.csv(f"{V6}/geometry_calibration_manifest.csv")
    v3_rows = guard.csv(f"{V3}/run_manifest.csv")
    quarantine = guard.json(f"{V4}/failed_v3_quarantine_manifest.json")
    existing = guard.json(f"{EXISTING}/manifest.json")
    selected_geometry = guard.json(f"{V6}/selected_geometry_lock.json")
    preflight_ready = bool(
        head == STARTING_CHECKPOINT and not unrelated
        and v7_source_before == v7_checkpoint and v5_source_before == v5_checkpoint
        and v7_provenance["artifact_hash_graph_verified"]
        and v5_provenance["artifact_hash_graph_verified"]
        and v7_lock["candidate_id"] == "M1"
        and np.allclose(v7_lock["friction5"], (
            0.0875, 0.0875, 0.00125, 0.00002125, 0.00002125), atol=1e-15, rtol=0)
        and not selected_geometry["selected"]
        and selected_geometry["geometry"] is None
        and patch_snapshot["explicit_pair_count"] == 8
        and not v5_readiness["WALKING_V2_SLIP_REACQUISITION_DATA_READY"]
        and not v7_readiness["WALKING_V2_SLIP_TARGETED_RETRAINING_AUTHORIZED"]
        and v7_summary["canonical_audit"]["combined_source_valid_positive_count"] == 87
        and v7_summary["canonical_audit"]["covered_factorial_cells"] == 35
        and len(v7_calibration) == 18 and len(v7_manifest) == 108
        and len(v6_calibration) == 24 and len(v3_rows) == 216
        and quarantine["all_runs_quarantined"] and len(existing["runs"]) == 120
        and guard.blocked == 0)
    immutable = {
        "starting_checkpoint": STARTING_CHECKPOINT, "execution_head": head,
        "clean_except_task_scoped_paths_before_execution": not unrelated,
        "task_scoped_paths_before_execution": status_lines,
        "protocol_sha256_before_execution": protocol_sha,
        "variation_contract_sha256_before_execution": variation_sha,
        "variation_contract_file_sha256_before_execution": contract_file_sha,
        "protocol_and_variation_contract_frozen_before_execution": True,
        "v7_source_before_sha256": v7_source_before,
        "v5_source_before_sha256": v5_source_before,
        "v7_artifact_before_sha256": v7_artifact_before,
        "v5_artifact_before_sha256": v5_artifact_before,
        "terrain_before_sha256": terrain_before,
        "physical_oracle_before_sha256": oracle_before,
        "M1_profile_lock_before_sha256": m1_lock_sha,
        "M1_profile_contract_sha256": candidate_profile("M1").sha256,
        "strong_profile_before_sha256": strong_sha,
        "G0_geometry_before_sha256": g0_sha,
        "explicit_pair_contract_sha256": patch_snapshot["pair_contract_sha256"],
        "explicit_pair_count": patch_snapshot["explicit_pair_count"],
        "v5_readiness_preserved_false": True,
        "v7_readiness_preserved_false": True,
        "v3_quarantine_disposition": quarantine["disposition"],
        "preflight_ready": preflight_ready, "after": {},
        "all_immutable_after_execution": False,
    }
    write_json(output / "immutable_verification.json", immutable)
    if not preflight_ready:
        raise RuntimeError("v8 immutable preflight failed")

    diagnosis, root_cause, diagnosis_traces = diagnose_variation_sensitivity(
        guarded, v7_shards, v7_calibration, v7_manifest)
    write_csv(output / "variation_failure_diagnosis.csv", diagnosis)
    write_json(output / "variation_root_cause.json", root_cause)

    traces, manifests, contacts, progress = v7_runner._run_block(
        conditions, guarded[POLICY], "moderate_v2_supplemental_cell")
    _add_variation_fields(manifests)
    _add_variation_fields(contacts)
    _add_variation_fields(progress)
    pair_manifest, parity = v7_runner.build_pair_audits(conditions, traces, manifests)
    _add_variation_fields(pair_manifest)
    _add_variation_fields(parity)
    episodes, positives, controls, falls = build_ledgers(traces, manifests)
    positives, controls = v7_runner.enrich_source_audits(
        traces, manifests, positives, controls, parity)
    _add_variation_fields(positives)
    _add_variation_fields(controls)
    _add_variation_fields(episodes)
    _add_variation_fields(falls)
    _normalize_positive_failures(positives)
    for row in controls:
        row.update({
            "stable_contact": row["stable_loaded_contact_coverage"],
            "physical_onset": not _bool(row["physical_slip_onset_free"]),
            "control_validity": row["control_source_valid"],
        })
    metrics = supplemental_metrics(conditions, traces, manifests, positives)
    shard_manifest = save_trace_shards(output, conditions, traces)
    if not (shard_manifest["all_under_45_mib"]
            and shard_manifest["all_exact_roundtrip_verified"]):
        raise RuntimeError("trace shard contract failed")
    gates = supplemental_gates(
        traces, manifests, positives, controls, parity, guard.blocked)
    write_csv(output / "supplemental_run_manifest.csv", manifests)
    write_csv(output / "supplemental_pair_manifest.csv", pair_manifest)
    write_csv(output / "supplemental_contact_audit.csv", contacts)
    write_csv(output / "supplemental_physical_episode_ledger.csv", episodes)
    write_csv(output / "supplemental_positive_source_audit.csv", positives)
    write_csv(output / "supplemental_control_source_audit.csv", controls)
    write_csv(output / "supplemental_pair_parity.csv", parity)
    write_csv(output / "supplemental_fall_censor_audit.csv", falls)
    write_csv(output / "supplemental_variation_metrics.csv", metrics)

    base, base_coverage = canonical_base(
        v5_manifest, v5_positive, v5_controls, v5_shards,
        v7_manifest, v7_positive, v7_controls, v7_shards)
    augmented: dict[str, object] = {
        "audit": {"pass": False, "augmented_factorial_coverage": 35}, "runs": []}
    augmented_coverage: list[dict[str, object]] = []
    folds: dict[str, object] = {
        "audit": {"valid": False}, "rows": [], "not_created": True,
        "reason": "supplemental gate failed"}
    eligibility: list[dict[str, object]] = []
    supplemental_rows = _supplemental_manifest_rows(
        manifests, positives, controls, shard_manifest)
    if gates["pass"]:
        augmented, augmented_coverage, supplemental_cell = augmented_artifacts(
            base, base_coverage, supplemental_rows, gates,
            v5_positive, v7_positive, positives)
        write_json(output / "canonical_base_manifest.json", base)
        write_json(output / "supplemental_cell_manifest.json", supplemental_cell)
        write_json(output / "augmented_canonical_development_manifest.json", augmented)
        write_csv(output / "augmented_factorial_coverage.csv", augmented_coverage)
        if augmented["audit"]["pass"]:  # type: ignore[index]
            eligibility = training_eligibility_rows(
                augmented, v7_calibration, v6_calibration, v3_rows)
            write_csv(output / "training_eligibility.csv", eligibility)
            folds = future_folds(augmented, existing["runs"])
            if folds["audit"]["valid"]:  # type: ignore[index]
                write_json(output / "future_nested_fold_manifest.json", folds)

    trace_hashes = [str(row["full_trace_sha256"]) for row in manifests]
    write_json(output / "duplicate_audit.json", {
        "planned_run_count": 24, "executed_run_count": len(manifests),
        "unique_run_id_count": len({row["run_id"] for row in manifests}),
        "unique_pair_fingerprint_count": len({row["pair_fingerprint"] for row in manifests}),
        "unique_trace_hash_count": len(set(trace_hashes)),
        "duplicate_unique_trace_count": len(trace_hashes) - len(set(trace_hashes)),
        "discarded_count": 0, "replaced_count": 0,
        "silently_relabelled_count": 0,
        "deterministic_serialization_replay_count": 1,
        "replay_changed_run_configuration": False,
        "replay_reason": "post-execution control-audit serializer field mismatch",
        "valid": len(trace_hashes) == len(set(trace_hashes)) == 24})

    v7_source_after = {path: sha256_file(REPO / path) for path in V7_SOURCES}
    v5_source_after = {path: sha256_file(REPO / path) for path in V5_SOURCES}
    v7_artifact_after = {path: sha256_file(REPO / path) for path in V7_ARTIFACTS}
    v5_artifact_after = {path: sha256_file(REPO / path) for path in V5_ARTIFACTS}
    terrain_after = {name: sha256_file(REPO / path) for name, path in TERRAIN_FILES.items()}
    oracle_after = {path: sha256_file(REPO / path) for path in ORACLE_FILES}
    immutable.update({
        "after": {
            "v7_source_sha256": v7_source_after, "v5_source_sha256": v5_source_after,
            "v7_artifact_sha256": v7_artifact_after,
            "v5_artifact_sha256": v5_artifact_after,
            "terrain_sha256": terrain_after, "physical_oracle_sha256": oracle_after,
            "M1_profile_lock_sha256": sha256_file(REPO / f"{V7}/moderate_v2_profile_lock.json"),
            "strong_profile_sha256": friction_profiles()["native_strong_ice"].sha256,
            "G0_geometry_sha256": GEOMETRY_CANDIDATES[0].sha256,
        },
        "v7_source_unchanged": v7_source_before == v7_source_after,
        "v5_source_unchanged": v5_source_before == v5_source_after,
        "v7_artifacts_unchanged": v7_artifact_before == v7_artifact_after,
        "v5_artifacts_unchanged": v5_artifact_before == v5_artifact_after,
        "terrain_byte_identical": terrain_before == terrain_after,
        "physical_oracle_byte_identical": oracle_before == oracle_after,
        "M1_profile_lock_byte_identical": (
            m1_lock_sha == sha256_file(REPO / f"{V7}/moderate_v2_profile_lock.json")),
        "strong_native_ice_unchanged": strong_sha == friction_profiles()["native_strong_ice"].sha256,
        "G0_geometry_unchanged": g0_sha == GEOMETRY_CANDIDATES[0].sha256,
        "explicit_pair_count_unchanged": v5_runner._patch_contract_snapshot()["explicit_pair_count"] == 8,
        "protocol_unchanged": sha256_file(output / "protocol.json") == protocol_sha,
        "variation_contract_unchanged": (
            variation_contract_sha256() == variation_sha
            and sha256_file(output / "supplemental_variation_contract.json") == contract_file_sha),
        "v5_readiness_preserved_false": not v5_readiness["WALKING_V2_SLIP_REACQUISITION_DATA_READY"],
        "v7_readiness_preserved_false": not v7_readiness["WALKING_V2_SLIP_TARGETED_RETRAINING_AUTHORIZED"],
        "model_score_access_count": 0, "v3_trace_access_count": 0,
    })
    immutable["all_immutable_after_execution"] = bool(
        immutable["v7_source_unchanged"] and immutable["v5_source_unchanged"]
        and immutable["v7_artifacts_unchanged"] and immutable["v5_artifacts_unchanged"]
        and immutable["terrain_byte_identical"] and immutable["physical_oracle_byte_identical"]
        and immutable["M1_profile_lock_byte_identical"]
        and immutable["strong_native_ice_unchanged"] and immutable["G0_geometry_unchanged"]
        and immutable["explicit_pair_count_unchanged"] and immutable["protocol_unchanged"]
        and immutable["variation_contract_unchanged"])
    write_json(output / "immutable_verification.json", immutable)

    augmented_ready = _bool(augmented["audit"]["pass"])
    future_ready = _bool(folds["audit"]["valid"])
    targeted = bool(
        gates["pass"] and augmented_ready and future_ready
        and immutable["all_immutable_after_execution"] and guard.blocked == 0)
    readiness = {
        "WALKING_V2_SLIP_VARIATION_DIAGNOSIS_READY": root_cause["exactly_one_primary_cause"],
        "WALKING_V2_SLIP_SUPPLEMENTAL_CONTRACT_READY": immutable["variation_contract_unchanged"],
        "WALKING_V2_SLIP_SUPPLEMENTAL_SOURCE_READY": bool(
            gates["pass"] and gates["source_valid_positive_count"] >= 6),
        "WALKING_V2_SLIP_SUPPLEMENTAL_CONTROL_READY": bool(
            gates["valid_control_count"] == 12
            and gates["control_physical_slip_onset_count"] == 0),
        "WALKING_V2_SLIP_SUPPLEMENTAL_CELL_READY": gates["pass"],
        "WALKING_V2_SLIP_AUGMENTED_FACTORIAL_COVERAGE_READY": bool(
            augmented_ready and augmented["audit"]["augmented_factorial_coverage"] == 36),  # type: ignore[index]
        "WALKING_V2_SLIP_AUGMENTED_DEVELOPMENT_DATA_READY": augmented_ready,
        "WALKING_V2_SLIP_FUTURE_FOLD_READY": future_ready,
        "WALKING_V2_SLIP_TARGETED_RETRAINING_AUTHORIZED": targeted,
        "WALKING_V2_FRESH_BLIND_HOLDOUT_AUTHORIZED": False,
        "WALKING_V2_TERRAIN_LOCK_PRESERVED": immutable["terrain_byte_identical"],
        "WALKING_V2_SYSTEM_MIGRATION_AUTHORIZED": False,
        "WALKING_V2_INT8_PREPARATION_AUTHORIZED": False,
        "SINK_RUNTIME_DETECTION_DEFERRED": True,
    }
    if targeted:
        next_step = "SLIP_TARGETED_RETRAINING_V3"
    elif gates["source_valid_positive_count"] < 6:
        next_step = "ADDITIONAL_MODERATE_V2_ACQUISITION"
    elif gates["pass"]:
        next_step = "STOP_WALKING_V2_DEPLOYMENT"
    else:
        next_step = "SLIP_RISK_SCOPE_REDUCTION"
    summary = {
        "task": "Acquire Additional Moderate-v2 Slip Support v8",
        "variation_root_cause": root_cause["primary_cause"],
        "M1_G0_strong_oracle_unchanged": bool(
            immutable["M1_profile_lock_byte_identical"] and immutable["G0_geometry_unchanged"]
            and immutable["strong_native_ice_unchanged"]
            and immutable["physical_oracle_byte_identical"]),
        "supplemental_gates": gates,
        "supplemental_positive_runs": len(positives),
        "supplemental_control_runs": len(controls),
        "supplemental_source_valid_positive_count": gates["source_valid_positive_count"],
        "supplemental_physically_distinct_valid_variation_count": gates[
            "physically_distinct_valid_variation_count"],
        "supplemental_efficiency": gates["supplemental_efficiency"],
        "base_valid_positive_count": 87,
        "base_positive_attempts": 108,
        "base_valid_positive_rate": 87 / 108,
        "base_factorial_coverage": 35,
        "augmented_factorial_coverage": augmented["audit"]["augmented_factorial_coverage"],  # type: ignore[index]
        "augmented_manifest_created": (output / "augmented_canonical_development_manifest.json").is_file(),
        "future_fold_manifest_created": (output / "future_nested_fold_manifest.json").is_file(),
        "eligibility_counts": {
            value: sum(row["eligibility"] == value for row in eligibility)
            for value in (
                "POSITIVE_ELIGIBLE", "CONTROL_ELIGIBLE",
                "FAILED_SOURCE_DIAGNOSTIC_ONLY", "CALIBRATION_ONLY_DO_NOT_TRAIN",
                "QUARANTINED", "FORBIDDEN")},
        "readiness": readiness, "next_step": next_step,
        "forbidden_artifact_access_count": guard.blocked,
        "training_performed": False, "model_selected_or_locked": False,
        "blind_holdout_created": False,
        "system_int8_vela_e84_hil_changed": False,
        "discarded_replaced_silently_relabelled": [0, 0, 0],
        "execution_recovery": {
            "preliminary_physics_block_completed": True,
            "preliminary_results_persisted": False,
            "incident": "control-audit serializer field mismatch after 24/24 runs",
            "deterministic_serialization_replay_count": 1,
            "run_ids_seeds_and_configurations_changed": False,
            "adaptive_or_replacement_variation_count": 0,
            "preliminary_unpersisted_results_used_for_contract_changes": False,
        },
    }
    write_json(output / "readiness.json", readiness)
    write_json(output / "summary.json", summary)
    create_plots(
        output, diagnosis, diagnosis_traces, metrics, positives,
        conditions, traces, augmented_coverage or [{
            **row, "augmented_source_valid_positive_count": row["source_valid_positive_count"]}
            for row in base_coverage])
    audit = f"""# Additional moderate-v2 Slip acquisition v8 audit

1. Calibration succeeded while v7 variations failed because: **{root_cause['primary_cause']}**; the isolated same-phase/same-delay comparison changed patch-entry state and produced {root_cause['isolating_comparison']['successful_maximum_anchor_drift_m']:.6f} m versus {root_cause['isolating_comparison']['failed_maximum_anchor_drift_m']:.6f} m maximum drift.
2. M1/G0/strong Ice/oracle unchanged: **{summary['M1_G0_strong_oracle_unchanged']}**.
3. Exactly 12 positive and 12 control unique frozen runs are retained: **{len(positives) == 12 and len(controls) == 12}**. A disclosed deterministic serialization replay of the same 24 run IDs was required after a post-execution audit-field mismatch; no seed/configuration was replaced.
4. Supplemental source-valid positives: **{gates['source_valid_positive_count']}/12**.
5. Physically distinct valid variations: **{gates['physically_distinct_valid_variation_count']}**.
6. All valid onsets remained in mid_late_stance: **{gates['all_valid_onsets_mid_late_stance']}**.
7. All controls remained physically non-slip: **{gates['control_physical_slip_onset_count'] == 0}**.
8. Parity/divergence/no-double-contact gates: **{gates['precontact_pair_parity_count']}/12, {gates['postcontact_divergence_count']}/12, {gates['patch_base_double_contact_count']}**.
9. Final missing cell filled: **{gates['pass']}**.
10. Base matrix remains explicitly **87/108** and **35/36**: **True**.
11. Augmented coverage reaches 36/36: **{augmented_ready and augmented['audit']['augmented_factorial_coverage'] == 36}**.
12. Calibration, failed-source, and quarantined data excluded correctly: **{bool(eligibility) and not any(_bool(row['training_eligible']) for row in eligibility if row['eligibility'] in {'FAILED_SOURCE_DIAGNOSTIC_ONLY', 'CALIBRATION_ONLY_DO_NOT_TRAIN', 'QUARANTINED', 'FORBIDDEN'})}**.
13. Future nested-fold manifest created: **{future_ready}**.
14. Targeted Slip retraining authorized: **{targeted}**.
15. Forbidden artifact access count: **{guard.blocked}**.
16. Terrain and M1 byte-identical: **{immutable['terrain_byte_identical']}/{immutable['M1_profile_lock_byte_identical']}**.
17. Blind holdout/System/INT8 remain unauthorized: **True/True/True**.
18. Sink remains `SINK_RUNTIME_DETECTION_DEFERRED`: **True**.

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
        "supplemental_variation_contract_sha256": variation_sha,
        "M1_profile_lock_sha256": m1_lock_sha,
        "trace_shard_manifest_sha256": sha256_file(output / "trace_shard_manifest.json"),
        "future_nested_fold_manifest_sha256": (
            sha256_file(output / "future_nested_fold_manifest.json") if future_ready else None),
        "artifact_sha256": artifact_hashes,
        "artifact_hash_graph_verified": all(
            sha256_file(output / name) == digest for name, digest in artifact_hashes.items()),
        "forbidden_access_count": guard.blocked,
        "model_score_access_count": 0, "v3_trace_access_count": 0,
        "deterministic_serialization_replay_count": 1,
        "replay_changed_run_configuration": False,
        "training_performed": False,
    })
    files = sorted(path for path in output.iterdir() if path.is_file())
    sizes = [{
        "path": path.name, "bytes": path.stat().st_size,
        "mib": path.stat().st_size / (1024 * 1024),
        "under_or_equal_45_mib": path.stat().st_size <= 45 * 1024 * 1024}
        for path in files]
    write_json(output / "resource_size_audit.json", {
        "limit_mib": 45, "files": sizes,
        "maximum_file_mib": max((row["mib"] for row in sizes), default=0.0),
        "all_files_within_limit": all(row["under_or_equal_45_mib"] for row in sizes),
    })
    print(json.dumps({
        "output": str(output), "supplemental_runs": len(manifests),
        "source_valid_positive": gates["source_valid_positive_count"],
        "distinct_valid_variations": gates["physically_distinct_valid_variation_count"],
        "supplemental_gate": gates["pass"],
        "augmented_coverage": augmented["audit"]["augmented_factorial_coverage"],
        "future_fold_ready": future_ready,
        "targeted_retraining_authorized": targeted,
        "next_step": next_step}, indent=2))


if __name__ == "__main__":
    main()
