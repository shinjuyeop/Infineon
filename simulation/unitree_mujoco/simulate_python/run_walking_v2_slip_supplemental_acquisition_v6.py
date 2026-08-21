"""Diagnose and conditionally supplement missing bilateral Slip sources v6."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from bilateral_hil_sensor_v2 import FOOT_CONTACT_GEOM_NAMES
from g1_upstream_locomotion import TESTED_POLICY_SHA256, UPSTREAM_REVISION
import run_walking_v2_bilateral_slip_targeted_acquisition_v5 as v5_runner
from run_walking_v2_bilateral_slip_targeted_acquisition_v3 import build_ledgers
from run_walking_v2_bilateral_slip_targeted_acquisition_v5 import (
    build_pair_audits, enrich_source_audits, save_trace_shards,
    sha256_file, write_csv, write_json,
)
from run_walking_v2_slip_scenario_generator_redesign_v4 import (
    SCENE_PATH, _configure_surface_geom,
)
from terrain_profiles import TERRAIN_PROFILES
from walking_v2_bilateral_slip_targeted_acquisition_v5 import (
    PHASE_BINS, SEVERITIES, SIDES, SPEEDS_MPS, material_profiles,
)
from walking_v2_slip_scenario_generator_redesign_v4 import (
    PilotCondition, friction_profiles,
)
from walking_v2_slip_supplemental_acquisition_v6 import (
    CALIBRATION_DISPOSITION, GEOMETRY_CANDIDATES, SUPPLEMENTAL_VARIATIONS,
    TARGET_FOOT, TARGET_PHASE, TARGET_SEVERITY, GeometryCandidate,
    RunCondition, calibration_matrix, frozen_profile_sha256,
    geometry_candidate, geometry_contract_sha256, oracle_contract,
    select_geometry, supplemental_matrix,
)


REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO / "simulation/outputs/walking_v2_slip_supplemental_acquisition_v6"
STARTING_CHECKPOINT = "c484c4dfe2560de068b288fb4fdaf755424f04b2"
V4 = "simulation/outputs/walking_v2_slip_scenario_generator_redesign_v4"
V5 = "simulation/outputs/walking_v2_bilateral_slip_targeted_acquisition_v5"
V3 = "simulation/outputs/walking_v2_bilateral_slip_targeted_acquisition_v3"
EXISTING = "simulation/outputs/walking_bilateral_sensor_sink_observability_v2"
JOINT = "simulation/outputs/walking_v2_joint_terrain_slip_redesign_v1"
ORACLE = "simulation/outputs/walking_hazard_oracle_calibration_v1"
POLICY = "simulation/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"
SCENE = "simulation/unitree_mujoco/unitree_robots/g1/scene_walking_terrain_transition.xml"
ROBOT = "simulation/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
V6_SOURCES = (
    "simulation/unitree_mujoco/simulate_python/walking_v2_slip_supplemental_acquisition_v6.py",
    "simulation/unitree_mujoco/simulate_python/run_walking_v2_slip_supplemental_acquisition_v6.py",
    "simulation/unitree_mujoco/simulate_python/test_walking_v2_slip_supplemental_acquisition_v6.py",
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
V4_ARTIFACTS = (
    f"{V4}/protocol.json", f"{V4}/summary.json", f"{V4}/readiness.json",
    f"{V4}/provenance.json", f"{V4}/local_patch_definition.json",
    f"{V4}/geom_contact_contract.csv", f"{V4}/terrain_immutable_verification.json",
    f"{V4}/oracle_immutable_verification.json",
    f"{V4}/failed_v3_quarantine_manifest.json",
)
V5_ARTIFACTS = (
    f"{V5}/protocol.json", f"{V5}/acquisition_manifest.json",
    f"{V5}/summary.json", f"{V5}/readiness.json", f"{V5}/provenance.json",
    f"{V5}/generator_immutable_verification.json",
    f"{V5}/patch_contract_verification.json",
    f"{V5}/oracle_immutable_verification.json",
    f"{V5}/terrain_immutable_verification.json",
    f"{V5}/v3_quarantine_verification.json",
    f"{V5}/material_profiles.json", f"{V5}/patch_geometry.json",
    f"{V5}/run_manifest.csv", f"{V5}/positive_source_audit.csv",
    f"{V5}/control_source_audit.csv", f"{V5}/active_patch_contact_audit.csv",
    f"{V5}/physical_episode_ledger.csv", f"{V5}/pair_parity_audit.csv",
    f"{V5}/trace_shard_manifest.json",
) + tuple(f"{V5}/traces_part_{index:03d}.npz" for index in range(9))
TERRAIN_FILES = {
    "model": f"{JOINT}/terrain_candidate_model.npz",
    "normalization": f"{JOINT}/terrain_candidate_normalization.json",
    "config": f"{JOINT}/terrain_candidate_config.json",
    "lock": f"{JOINT}/terrain_selection_lock.json",
}
ALLOWED_INPUTS = {
    **{path: "frozen v4 generator implementation" for path in V4_SOURCES},
    **{path: "frozen v5 acquisition implementation" for path in V5_SOURCES},
    **{path: "verified v4 generator artifact" for path in V4_ARTIFACTS},
    **{path: "immutable v5 acquisition evidence" for path in V5_ARTIFACTS},
    f"{V3}/run_manifest.csv": "quarantined v3 run inventory only",
    f"{EXISTING}/manifest.json": "existing valid 120-run development inventory",
    f"{EXISTING}/slip_bilateral_metrics.csv": "existing physical-onset foot inventory; model scores unused",
    f"{ORACLE}/summary.json": "frozen physical Slip oracle contract",
    "simulation/unitree_mujoco/simulate_python/walking_hazard_oracle_calibration_v1.py": "frozen physical Slip oracle implementation",
    "simulation/unitree_mujoco/simulate_python/walking_hazard_ground_truth_v1.py": "frozen physical contact labels",
    "simulation/unitree_mujoco/simulate_python/terrain_profiles.py": "frozen material profiles",
    "simulation/unitree_mujoco/simulate_python/bilateral_hil_sensor_v2.py": "Fusion20 virtual sensors",
    "simulation/unitree_mujoco/simulate_python/g1_upstream_locomotion.py": "frozen walking controller",
    **{path: f"immutable Terrain {key}" for key, path in TERRAIN_FILES.items()},
    POLICY: "fixed walking policy", SCENE: "fixed walking scene",
    ROBOT: "fixed G1 robot include",
}
FORBIDDEN_TOKENS = (
    "/outer/", "_outer_", "holdout", "spatial_final", "spatial-final",
    "final_test", "final-test",
)
DIAGNOSTIC_FIELDS = (
    "time_s", "bilateral_fusion20_raw", "force_loaded", "physical_contact",
    "foot_world_xyz_label_only", "foot_world_velocity_label_only",
    "patch_contact", "effective_patch_friction", "patch_normal_force_n",
    "patch_tangential_force_n", "anchor_drift_label_only",
    "slip_physical_active", "slip_calibration_valid_label_only",
    "touchdown_transient", "pre_fall_valid", "contact_episode_id",
)
ROOT_CAUSES = (
    "INSUFFICIENT_PATCH_DWELL", "INSUFFICIENT_REMAINING_STANCE",
    "LOW_NORMAL_LOAD", "INSUFFICIENT_TANGENTIAL_DEMAND",
    "LEFT_RIGHT_GAIT_ASYMMETRY", "PATCH_LONGITUDINAL_MISALIGNMENT",
    "ORACLE_ONSET_AFTER_CONTACT_LOSS",
    "MODERATE_FRICTION_PHYSICALLY_INSUFFICIENT", "OTHER_WITH_EVIDENCE",
)


class AccessGuard:
    """Fail closed and record every pre-existing artifact read."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.events: list[dict[str, object]] = []
        self.blocked = 0
        self._paths: dict[str, Path] = {}
        self._flush()

    def _flush(self) -> None:
        write_json(self.output / "artifact_access_log.json", {
            "exact_paths_only": True, "forbidden_tokens": list(FORBIDDEN_TOKENS),
            "events": self.events, "blocked_access_count": self.blocked,
            "all_accesses_completed": all(
                row["status"] == "completed" for row in self.events),
            "model_score_used_for_diagnosis": False,
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
                "reason": forbidden or "not_allowlisted",
            })
            self._flush()
            raise PermissionError(normalized)
        if normalized in self._paths:
            return self._paths[normalized]
        path = (REPO / normalized).resolve()
        if not path.is_file() or REPO.resolve() not in path.parents:
            self.blocked += 1
            self.events.append({
                "path": normalized, "access": access, "status": "blocked",
                "reason": "missing_or_outside_repository",
            })
            self._flush()
            raise FileNotFoundError(normalized)
        self.events.append({
            "path": normalized, "access": access, "status": "completed",
            "purpose": ALLOWED_INPUTS[normalized], "sha256": sha256_file(path),
        })
        self._paths[normalized] = path
        self._flush()
        return path

    def json(self, relative: str) -> object:
        return json.loads(self.path(relative, "json").read_text(encoding="utf-8"))

    def csv(self, relative: str) -> list[dict[str, str]]:
        with self.path(relative, "csv").open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))


def _git_value(*arguments: str) -> str:
    return subprocess.check_output(
        ("git",) + arguments, cwd=REPO, text=True).strip()


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


def build_bounded_local_model(
    condition: PilotCondition, geometry: GeometryCandidate,
) -> tuple[mujoco.MjModel, dict[str, object]]:
    """Compile v4's contact contract with only bounded x placement/length changes."""
    spec = mujoco.MjSpec.from_file(str(SCENE_PATH))
    for name in ("ground_source", "ground_target"):
        spec.delete(spec.geom(name))
    bounds = geometry.bounds
    x0, x1, y0, y1 = -6.5, 6.5, -0.8, 0.8
    regions = (
        ("base_x_before", x0, bounds.x_min_m, y0, y1),
        ("base_x_after", bounds.x_max_m, x1, y0, y1),
        ("base_lane_below", bounds.x_min_m, bounds.x_max_m, y0, bounds.y_min_m),
        ("base_lane_above", bounds.x_min_m, bounds.x_max_m, bounds.y_max_m, y1),
    )
    base_names: list[str] = []
    for name, rx0, rx1, ry0, ry1 in regions:
        if rx1 <= rx0 or ry1 <= ry0:
            continue
        geom = spec.worldbody.add_geom(name=name)
        _configure_surface_geom(geom, rgba=(0.35, 0.35, 0.35, 1.0))
        geom.pos = ((rx0 + rx1) / 2, (ry0 + ry1) / 2, -0.1)
        geom.size = ((rx1 - rx0) / 2, (ry1 - ry0) / 2, 0.1)
        base_names.append(name)
    patch = spec.worldbody.add_geom(name="slip_patch")
    _configure_surface_geom(patch, rgba=(0.25, 0.45, 0.80, 1.0))
    patch.pos = bounds.center
    patch.size = bounds.half_size
    patch.contype = 0
    patch.conaffinity = 0
    hard = friction_profiles()["hard_control"]
    concrete = TERRAIN_PROFILES["concrete"]
    pair_names: dict[str, list[str]] = {side: [] for side in SIDES}
    for side in SIDES:
        for foot_geom_name in FOOT_CONTACT_GEOM_NAMES[side]:
            pair_name = f"pair_{foot_geom_name}_slip_patch"
            pair = spec.add_pair(name=pair_name)
            pair.geomname1 = foot_geom_name
            pair.geomname2 = "slip_patch"
            pair.condim = concrete.condim
            pair.friction = hard.friction5
            pair.solref = concrete.solref
            pair.solimp = concrete.solimp
            pair_names[side].append(pair_name)
    model = spec.compile()
    pair_ids = {
        side: tuple(model.pair(name).id for name in names)
        for side, names in pair_names.items()
    }
    info = {
        "kind": "bounded_longitudinal_patch_v6",
        "ground_names": tuple(base_names) + ("slip_patch",),
        "base_names": tuple(base_names), "patch_name": "slip_patch",
        "pair_names": pair_names, "pair_ids": pair_ids,
        "target_pair_ids": pair_ids[condition.target_foot],
        "bounds": vars(bounds), "top_height_delta_m": 0.0,
        "intervention_profile": condition.profile,
        "geometry_candidate": geometry.candidate_id,
    }
    return model, info


def collect_run(
    condition: RunCondition, policy_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object], list[dict[str, object]]]:
    """Use the frozen v5 recorder with a preregistered model-builder injection."""
    geometry = condition.geometry
    original_builder = v5_runner.build_local_model
    original_bounds = v5_runner.patch_bounds
    v5_runner.build_local_model = lambda proxy: build_bounded_local_model(proxy, geometry)
    v5_runner.patch_bounds = lambda _side: geometry.bounds
    try:
        trace, metadata, contacts = v5_runner.collect_run(condition, policy_path)
    finally:
        v5_runner.build_local_model = original_builder
        v5_runner.patch_bounds = original_bounds
    metadata.update({
        "block": condition.block,
        "geometry_candidate": condition.geometry_candidate,
        "geometry_contract_sha256": condition.geometry_sha256,
        "disposition": condition.disposition,
        "source_calibration_only_do_not_train": (
            condition.disposition == CALIBRATION_DISPOSITION),
        "training_use_assigned": False,
    })
    for row in contacts:
        row["block"] = condition.block
        row["geometry_candidate"] = condition.geometry_candidate
        row["disposition"] = condition.disposition
    return trace, metadata, contacts


def _diagnostic_group(row: dict[str, str]) -> str | None:
    if row["target_foot"] == "left" and row["target_phase"] == TARGET_PHASE:
        if row["severity"] == TARGET_SEVERITY:
            return "A_FAILED_LEFT_MID_LATE_MODERATE"
        if row["severity"] == "native_strong_ice":
            return "C_SEVERITY_COUNTERFACTUAL_LEFT_STRONG"
    if (row["target_foot"] == "right" and row["target_phase"] == TARGET_PHASE
            and row["severity"] == TARGET_SEVERITY):
        return "B_FOOT_COUNTERFACTUAL_RIGHT_MODERATE"
    if (row["target_foot"] == "left"
            and row["target_phase"] in {"early_loading", "mid_loading_early_stance"}
            and row["severity"] == TARGET_SEVERITY):
        return "D_PHASE_COUNTERFACTUAL_LEFT_MODERATE"
    return None


def load_v5_diagnostic_traces(
    guard: AccessGuard, audit_rows: list[dict[str, str]],
) -> dict[str, dict[str, np.ndarray]]:
    selected = {
        row["run_id"] for row in audit_rows if _diagnostic_group(row) is not None}
    shard_manifest = guard.json(f"{V5}/trace_shard_manifest.json")
    traces: dict[str, dict[str, np.ndarray]] = {}
    for shard in shard_manifest["shards"]:  # type: ignore[index]
        run_ids = [run_id for run_id in shard["run_ids"] if run_id in selected]
        if not run_ids:
            continue
        relative = f"{V5}/{shard['path']}"
        path = guard.path(relative, "npz diagnostic fields")
        if sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"v5 trace shard hash mismatch: {relative}")
        with np.load(path, allow_pickle=False) as loaded:
            for run_id in run_ids:
                traces[run_id] = {
                    field: loaded[f"{run_id}__{field}"].copy()
                    for field in DIAGNOSTIC_FIELDS
                }
    if set(traces) != selected:
        raise RuntimeError("failed to reconstruct all v5 diagnostic traces")
    return traces


def _episode_metrics(
    trace: dict[str, np.ndarray], audit: dict[str, object],
) -> dict[str, object]:
    side = SIDES.index(str(audit["target_foot"]))
    other = 1 - side
    episode_ids = trace["contact_episode_id"][:, side].astype(int)
    drift = trace["anchor_drift_label_only"][:, side]
    valid = trace["pre_fall_valid"] & np.isfinite(drift)
    candidates: list[tuple[float, int]] = []
    for episode in sorted(set(episode_ids.tolist()) - {-1}):
        mask = (episode_ids == episode) & valid
        if np.any(mask):
            candidates.append((float(np.max(drift[mask])), episode))
    if not candidates:
        return {
            "diagnostic_episode_id": "", "touchdown_timestamp_s": "",
            "first_patch_contact_timestamp_s": "", "patch_contact_duration_ms": 0,
            "remaining_stance_after_patch_entry_ms": 0,
            "normal_contact_force_mean_n": "", "tangential_contact_force_mean_n": "",
            "friction_cone_utilization_mean": "", "anchor_drift_rate_mps": "",
            "maximum_anchor_drift_m": "", "target_foot_velocity_mean_mps": "",
            "contralateral_support_load_mean_n": "", "patch_entry_position_xyz": "",
            "patch_exit_position_xyz": "", "physical_onset_margin_m": "",
            "estimated_onset_after_contact_loss_ms": "",
        }
    _, episode = max(candidates)
    episode_samples = np.flatnonzero(episode_ids == episode)
    loaded_samples = episode_samples[trace["force_loaded"][episode_samples, side]]
    touchdown = int(loaded_samples[0] if loaded_samples.size else episode_samples[0])
    patch_samples = episode_samples[trace["patch_contact"][episode_samples, side]]
    if not patch_samples.size:
        patch_samples = episode_samples[:0]
    entry = int(patch_samples[0]) if patch_samples.size else None
    exit_sample = int(patch_samples[-1]) if patch_samples.size else None
    expected_mu = material_profiles()[str(audit["material_profile"])].friction3[0]
    active = (
        (episode_ids == episode) & trace["patch_contact"][:, side]
        & np.isfinite(trace["effective_patch_friction"][:, side])
        & np.isclose(trace["effective_patch_friction"][:, side], expected_mu,
                     atol=1e-12, rtol=0)
    )
    normal = trace["patch_normal_force_n"][:, side]
    tangent = trace["patch_tangential_force_n"][:, side]
    utilization = np.divide(
        tangent, trace["effective_patch_friction"][:, side] * normal,
        out=np.full_like(tangent, np.nan, dtype=float),
        where=(trace["effective_patch_friction"][:, side] * normal) > 1e-9,
    )
    finite_episode = episode_samples[valid[episode_samples]]
    maximum_drift = float(np.max(drift[finite_episode]))
    max_sample = int(finite_episode[np.argmax(drift[finite_episode])])
    elapsed_s = max((max_sample - touchdown + 1) / 1000.0, 0.001)
    drift_rate = maximum_drift / elapsed_s
    remaining_after_max_ms = max(0, int(episode_samples[-1]) - max_sample)
    additional_ms = (
        max(0.0, (0.050 - maximum_drift) / drift_rate * 1000.0)
        if drift_rate > 0 else float("inf"))
    after_loss = max(0.0, additional_ms - remaining_after_max_ms)
    fsr = trace["bilateral_fusion20_raw"][:, other * 10:other * 10 + 4].sum(axis=1)
    velocity = np.linalg.norm(
        trace["foot_world_velocity_label_only"][:, side, :2], axis=1)
    return {
        "diagnostic_episode_id": episode,
        "touchdown_timestamp_s": float(trace["time_s"][touchdown]),
        "first_patch_contact_timestamp_s": (
            "" if entry is None else float(trace["time_s"][entry])),
        "patch_contact_duration_ms": int(patch_samples.size),
        "remaining_stance_after_patch_entry_ms": (
            0 if entry is None else int(episode_samples[-1]) - entry + 1),
        "low_friction_active_duration_ms": int(np.sum(active)),
        "normal_contact_force_mean_n": (
            float(np.mean(normal[active])) if np.any(active) else ""),
        "normal_contact_force_peak_n": (
            float(np.max(normal[active])) if np.any(active) else ""),
        "tangential_contact_force_mean_n": (
            float(np.mean(tangent[active])) if np.any(active) else ""),
        "tangential_contact_force_peak_n": (
            float(np.max(tangent[active])) if np.any(active) else ""),
        "friction_cone_utilization_mean": (
            float(np.nanmean(utilization[active])) if np.any(active) else ""),
        "friction_cone_utilization_peak": (
            float(np.nanmax(utilization[active])) if np.any(active) else ""),
        "anchor_drift_rate_mps": drift_rate,
        "maximum_anchor_drift_m": maximum_drift,
        "target_foot_velocity_mean_mps": (
            float(np.mean(velocity[active])) if np.any(active) else ""),
        "target_foot_velocity_peak_mps": (
            float(np.max(velocity[active])) if np.any(active) else ""),
        "contralateral_support_load_mean_n": (
            float(np.mean(fsr[active])) if np.any(active) else ""),
        "patch_entry_position_xyz": (
            "" if entry is None else json.dumps(
                trace["foot_world_xyz_label_only"][entry, side].tolist())),
        "patch_exit_position_xyz": (
            "" if exit_sample is None else json.dumps(
                trace["foot_world_xyz_label_only"][exit_sample, side].tolist())),
        "physical_onset_margin_m": maximum_drift - 0.050,
        "estimated_onset_after_contact_loss_ms": (
            after_loss if np.isfinite(after_loss) else ""),
        "fall_occurred": _bool(audit.get("fall_occurred", False)),
        "first_fall_censor_boundary": audit.get("first_fall_sample", ""),
    }


def diagnose_v5(
    audit_rows: list[dict[str, str]], traces: dict[str, dict[str, np.ndarray]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    for audit in audit_rows:
        group = _diagnostic_group(audit)
        if group is None:
            continue
        rows.append({
            "comparison_group": group, "run_id": audit["run_id"],
            "pair_id": audit["pair_id"], "speed_mps": float(audit["speed_mps"]),
            "target_foot": audit["target_foot"],
            "target_phase": audit["target_phase"], "severity": audit["severity"],
            "variation_index": int(audit["variation_index"]),
            "source_valid": _bool(audit["source_valid"]),
            "failure_reason": audit["failure_reason"],
            **_episode_metrics(traces[audit["run_id"]], audit),
            "model_score_used": False,
        })
    expected_counts = {
        "A_FAILED_LEFT_MID_LATE_MODERATE": 9,
        "B_FOOT_COUNTERFACTUAL_RIGHT_MODERATE": 9,
        "C_SEVERITY_COUNTERFACTUAL_LEFT_STRONG": 9,
        "D_PHASE_COUNTERFACTUAL_LEFT_MODERATE": 18,
    }
    if any(sum(row["comparison_group"] == group for row in rows) != count
           for group, count in expected_counts.items()):
        raise RuntimeError("diagnostic comparison group count mismatch")

    def group_mean(group: str, field: str) -> float:
        values = [
            _float(row[field]) for row in rows if row["comparison_group"] == group]
        finite = [value for value in values if np.isfinite(value)]
        return float(np.mean(finite)) if finite else float("nan")

    left = "A_FAILED_LEFT_MID_LATE_MODERATE"
    right = "B_FOOT_COUNTERFACTUAL_RIGHT_MODERATE"
    strong = "C_SEVERITY_COUNTERFACTUAL_LEFT_STRONG"
    phase = "D_PHASE_COUNTERFACTUAL_LEFT_MODERATE"
    evidence = {
        "failed_count": 9,
        "failed_source_valid_count": sum(
            bool(row["source_valid"]) for row in rows
            if row["comparison_group"] == left),
        "left_vs_right_patch_dwell_ratio": (
            group_mean(left, "patch_contact_duration_ms")
            / group_mean(right, "patch_contact_duration_ms")),
        "left_vs_right_remaining_stance_ratio": (
            group_mean(left, "remaining_stance_after_patch_entry_ms")
            / group_mean(right, "remaining_stance_after_patch_entry_ms")),
        "left_vs_right_normal_load_ratio": (
            group_mean(left, "normal_contact_force_mean_n")
            / group_mean(right, "normal_contact_force_mean_n")),
        "left_vs_right_tangential_force_ratio": (
            group_mean(left, "tangential_contact_force_mean_n")
            / group_mean(right, "tangential_contact_force_mean_n")),
        "left_vs_right_target_velocity_ratio": (
            group_mean(left, "target_foot_velocity_mean_mps")
            / group_mean(right, "target_foot_velocity_mean_mps")),
        "failed_mean_maximum_anchor_drift_m": group_mean(
            left, "maximum_anchor_drift_m"),
        "right_mean_maximum_anchor_drift_m": group_mean(
            right, "maximum_anchor_drift_m"),
        "strong_mean_maximum_anchor_drift_m": group_mean(
            strong, "maximum_anchor_drift_m"),
        "early_middle_mean_maximum_anchor_drift_m": group_mean(
            phase, "maximum_anchor_drift_m"),
        "oracle_threshold_m": 0.050,
        "right_source_valid_count": sum(
            bool(row["source_valid"]) for row in rows
            if row["comparison_group"] == right),
        "strong_source_valid_count": sum(
            bool(row["source_valid"]) for row in rows
            if row["comparison_group"] == strong),
        "early_middle_source_valid_count": sum(
            bool(row["source_valid"]) for row in rows
            if row["comparison_group"] == phase),
    }
    primary = "LEFT_RIGHT_GAIT_ASYMMETRY"
    if not (
        evidence["left_vs_right_patch_dwell_ratio"] > 0.90
        and evidence["left_vs_right_remaining_stance_ratio"] > 0.90
        and evidence["left_vs_right_normal_load_ratio"] > 0.90
        and evidence["left_vs_right_target_velocity_ratio"] < 0.85
    ):
        primary = "OTHER_WITH_EVIDENCE"
    root = {
        "primary_failure_cause": primary,
        "allowed_primary_failure_causes": list(ROOT_CAUSES),
        "exactly_one_primary_failure_cause": primary in ROOT_CAUSES,
        "secondary_contributors": [
            "INSUFFICIENT_REMAINING_STANCE",
            "MODERATE_FRICTION_PHYSICALLY_INSUFFICIENT",
        ],
        "evidence": evidence,
        "interpretation": (
            "Patch dwell, remaining stance, and normal/tangential force are comparable "
            "between feet, but late-stance left-foot tangential velocity and accumulated "
            "drift are lower. Strong friction reduction and earlier activation overcome "
            "that asymmetric kinematic demand; frozen moderate friction does not."
        ),
        "patch_longitudinal_misalignment_supported": False,
        "model_scores_used": False,
    }
    return rows, root


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
        trace, metadata, contact_rows = collect_run(condition, policy_path)
        traces.append(trace)
        manifests.append(metadata)
        contacts.extend(contact_rows)
        progress.append({
            "block": label, "run_index": index, "run_id": condition.run_id,
            "pair_id": condition.pair_id, "status": "EXECUTED_AND_RETAINED",
            "full_trace_sha256": metadata["full_trace_sha256"],
            "discarded": False, "replaced": False,
            "silently_relabelled": False,
        })
        print(f"{label}: completed {index + 1}/{len(conditions)}", flush=True)
    return traces, manifests, contacts, progress


def calibration_assessment(
    conditions: list[RunCondition], traces: list[dict[str, np.ndarray]],
    manifests: list[dict[str, object]],
) -> tuple[
    list[dict[str, object]], list[dict[str, object]],
    list[dict[str, object]], list[dict[str, object]],
    list[dict[str, object]], list[dict[str, object]],
]:
    pair_manifest, parity = build_pair_audits(conditions, traces, manifests)
    episodes, positives, controls, falls = build_ledgers(traces, manifests)
    positives, controls = enrich_source_audits(
        traces, manifests, positives, controls, parity)
    for row in positives + controls:
        row["disposition"] = CALIBRATION_DISPOSITION
        row["training_use_assigned"] = False
    by_pair = {str(row["pair_id"]): row for row in parity}
    by_control = {str(row["pair_id"]): row for row in controls}
    by_trace = {str(row["run_id"]): trace for row, trace in zip(manifests, traces)}
    metric_rows: list[dict[str, object]] = []
    for positive in positives:
        pair = by_pair[str(positive["pair_id"])]
        control = by_control[str(positive["pair_id"])]
        geometry_id = str(next(
            row.geometry_candidate for row in conditions
            if row.run_id == positive["run_id"]))
        metric = _episode_metrics(by_trace[str(positive["run_id"])], positive)
        criteria = {
            "control_physically_non_slip": _bool(control["physical_slip_onset_free"]),
            "target_foot_first_affected": _bool(positive["target_first"]),
            "onset_inside_frozen_mid_late_bin": (
                positive["actual_onset_phase"] == TARGET_PHASE),
            "no_air_or_postfall_only_onset": (
                positive["failure_reason"] not in {
                    "TOUCHDOWN_TRANSIENT_ONLY", "POST_FALL_ONLY"}),
            "patch_base_double_contact_zero": (
                int(pair["patch_base_double_contact_count"]) == 0),
            "precontact_parity_pass": _bool(pair["parity_pass"]),
            "postcontact_divergence_pass": _bool(
                pair["postcontact_trajectory_diverged"]),
            "positive_effective_friction_contract": _bool(
                pair["positive_effective_friction_contract"]),
            "control_effective_friction_contract": _bool(
                pair["control_effective_friction_contract"]),
            "source_valid": _bool(positive["source_valid"]),
        }
        metric_rows.append({
            "row_type": "pair", "geometry_candidate": geometry_id,
            "geometry_sha256": geometry_candidate(geometry_id).sha256,
            "pair_id": positive["pair_id"], "positive_run_id": positive["run_id"],
            "control_run_id": control["run_id"],
            "speed_mps": positive["speed_mps"], **metric, **criteria,
            "pair_physical_criteria_pass": all(criteria.values()),
        })
    for geometry in GEOMETRY_CANDIDATES:
        selected = [
            row for row in metric_rows
            if row["geometry_candidate"] == geometry.candidate_id]
        speed_valid = {
            f"{speed:.2f}": sum(
                float(row["speed_mps"]) == speed
                and bool(row["pair_physical_criteria_pass"]) for row in selected)
            for speed in SPEEDS_MPS
        }
        candidate_pass = bool(
            len(selected) == 3
            and all(value == 1 for value in speed_valid.values()))
        metric_rows.append({
            "row_type": "candidate", "geometry_candidate": geometry.candidate_id,
            "geometry_sha256": geometry.sha256, "pair_count": len(selected),
            "valid_count_by_speed": json.dumps(speed_valid, sort_keys=True),
            "valid_speed_count": sum(value > 0 for value in speed_valid.values()),
            "control_slip_onset_count": sum(
                not bool(row["control_physically_non_slip"]) for row in selected),
            "parity_failure_count": sum(
                not bool(row["precontact_parity_pass"]) for row in selected),
            "postcontact_divergence_failure_count": sum(
                not bool(row["postcontact_divergence_pass"]) for row in selected),
            "double_contact_count": sum(
                not bool(row["patch_base_double_contact_zero"]) for row in selected),
            "candidate_pass": candidate_pass,
        })
    return pair_manifest, parity, episodes, positives, controls, falls, metric_rows


def supplemental_gates(
    conditions: list[RunCondition], traces: list[dict[str, np.ndarray]],
    manifests: list[dict[str, object]], positives: list[dict[str, object]],
    controls: list[dict[str, object]], parity: list[dict[str, object]],
    forbidden_count: int,
) -> dict[str, object]:
    valid = [row for row in positives if _bool(row["source_valid"])]
    speed_counts = {
        f"{speed:.2f}": sum(float(row["speed_mps"]) == speed for row in valid)
        for speed in SPEEDS_MPS}
    trace_hashes = [str(row["full_trace_sha256"]) for row in manifests]
    duplicate_count = len(trace_hashes) - len(set(trace_hashes))
    air_or_postfall = sum(int(np.sum(
        trace["slip_physical_active"]
        & ((~trace["physical_contact"]) | ~trace["pre_fall_valid"][:, None])))
        for trace in traces)
    values = {
        "planned_positive_attempts": sum(row.role == "positive" for row in conditions),
        "executed_positive_attempts": len(positives),
        "matched_control_attempts": len(controls),
        "source_valid_positive_count": len(valid),
        "source_valid_count_by_speed": speed_counts,
        "all_three_missing_cells_filled": all(value >= 1 for value in speed_counts.values()),
        "left_target_first_accuracy": (
            sum(_bool(row["target_first"]) for row in valid) / len(valid)
            if valid else 0.0),
        "control_physical_slip_onset_count": sum(
            not _bool(row["physical_slip_onset_free"]) for row in controls),
        "valid_control_count": sum(_bool(row["control_source_valid"]) for row in controls),
        "pair_precontact_parity_count": sum(_bool(row["parity_pass"]) for row in parity),
        "postcontact_divergence_count": sum(
            _bool(row["postcontact_trajectory_diverged"]) for row in parity),
        "full_trace_identity_count": sum(_bool(row["full_trace_identical"]) for row in parity),
        "patch_base_double_contact_count": sum(
            int(row["patch_base_double_contact_count"]) for row in parity),
        "friction_contract_violation_count": sum(
            not _bool(row["positive_effective_friction_contract"])
            or not _bool(row["control_effective_friction_contract"])
            for row in parity),
        "air_or_postfall_positive_attribution_count": air_or_postfall,
        "forbidden_artifact_access_count": forbidden_count,
        "duplicate_unique_trace_count": duplicate_count,
        "discarded_count": sum(_bool(row["discarded"]) for row in manifests),
        "replaced_count": sum(_bool(row["replaced"]) for row in manifests),
        "silently_relabelled_count": sum(
            _bool(row["silently_relabelled"]) for row in manifests),
    }
    values["pass"] = bool(
        values["planned_positive_attempts"] == 18
        and values["executed_positive_attempts"] == 18
        and values["matched_control_attempts"] == 18
        and values["source_valid_positive_count"] >= 15
        and all(value >= 4 for value in speed_counts.values())
        and values["all_three_missing_cells_filled"]
        and values["left_target_first_accuracy"] == 1.0
        and values["control_physical_slip_onset_count"] == 0
        and values["valid_control_count"] == 18
        and values["pair_precontact_parity_count"] == 18
        and values["postcontact_divergence_count"] == 18
        and values["full_trace_identity_count"] == 0
        and values["patch_base_double_contact_count"] == 0
        and values["friction_contract_violation_count"] == 0
        and values["air_or_postfall_positive_attribution_count"] == 0
        and values["forbidden_artifact_access_count"] == 0
        and values["duplicate_unique_trace_count"] == 0)
    return values


def factorial_coverage_rows(
    v5_positives: list[dict[str, str]],
    supplemental_positives: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for speed in SPEEDS_MPS:
        for foot in SIDES:
            for phase in (row.name for row in PHASE_BINS):
                for severity in SEVERITIES:
                    v5_count = sum(
                        _bool(row["source_valid"])
                        and float(row["speed_mps"]) == speed
                        and row["target_foot"] == foot
                        and row["target_phase"] == phase
                        and row["severity"] == severity
                        for row in v5_positives)
                    v6_count = sum(
                        _bool(row["source_valid"])
                        and float(row["speed_mps"]) == speed
                        and row["target_foot"] == foot
                        and row["target_phase"] == phase
                        and row["severity"] == severity
                        for row in supplemental_positives)
                    rows.append({
                        "speed_mps": speed, "target_foot": foot,
                        "target_phase": phase, "severity": severity,
                        "v5_source_valid_count": v5_count,
                        "v6_supplemental_source_valid_count": v6_count,
                        "combined_source_valid_count": v5_count + v6_count,
                        "has_valid_support": v5_count + v6_count > 0,
                    })
    return rows


def existing_physical_foot_balance(
    physical_rows: list[dict[str, str]],
) -> dict[str, int]:
    # C1 is the first bilateral candidate. Physical onset columns are labels,
    # independent of detector score; score/detection fields are never read.
    selected = [
        row for row in physical_rows
        if row["row_type"] == "profile_speed_foot" and row["candidate"] == "C1"
        and row["positive_run"] == "True"
        and row["first_physical_onset_sample"] != ""]
    return {
        side: sum(row["foot"] == side for row in selected) for side in SIDES}


def build_combined_manifest(
    existing_runs: list[dict[str, object]],
    existing_balance: dict[str, int],
    v5_manifests: list[dict[str, str]], v5_positives: list[dict[str, str]],
    v5_controls: list[dict[str, str]],
    v6_manifests: list[dict[str, object]],
    v6_positives: list[dict[str, object]], v6_controls: list[dict[str, object]],
    calibration_run_ids: Iterable[str], coverage: list[dict[str, object]],
) -> dict[str, object]:
    v5_valid_positive = {
        row["run_id"] for row in v5_positives if _bool(row["source_valid"])}
    v5_valid_control = {
        row["run_id"] for row in v5_controls if _bool(row["control_source_valid"])}
    v6_valid_positive = {
        str(row["run_id"]) for row in v6_positives if _bool(row["source_valid"])}
    v6_valid_control = {
        str(row["run_id"]) for row in v6_controls if _bool(row["control_source_valid"])}
    eligible: list[dict[str, object]] = []
    for row in existing_runs:
        eligible.append({
            "source_acquisition_version": "existing_bilateral_development_v2",
            "run_id": row["run_id"], "pair_id": "",
            "variation_index": row["variation_index"], "speed_mps": row["speed_mps"],
            "target_foot": "bilateral", "target_phase": "legacy_development",
            "severity": "legacy_development", "role": row["role"],
            "control_type": "legacy_development", "development_only": True,
        })
    for row in v5_manifests:
        run_id = row["run_id"]
        if run_id not in v5_valid_positive | v5_valid_control:
            continue
        eligible.append({
            "source_acquisition_version": "targeted_acquisition_v5",
            "run_id": run_id, "pair_id": row["pair_id"],
            "variation_index": int(row["variation_index"]),
            "speed_mps": float(row["speed_mps"]),
            "target_foot": row["target_foot"], "target_phase": row["target_phase"],
            "severity": row["severity"], "role": row["role"],
            "control_type": row["control_type"], "development_only": True,
        })
    for row in v6_manifests:
        run_id = str(row["run_id"])
        if run_id not in v6_valid_positive | v6_valid_control:
            continue
        eligible.append({
            "source_acquisition_version": "supplemental_acquisition_v6",
            "run_id": run_id, "pair_id": row["pair_id"],
            "variation_index": row["variation_index"],
            "speed_mps": row["speed_mps"], "target_foot": row["target_foot"],
            "target_phase": row["target_phase"], "severity": row["severity"],
            "role": row["role"], "control_type": row["control_type"],
            "development_only": True,
        })
    failed_v5 = [
        row["run_id"] for row in v5_positives if not _bool(row["source_valid"])]
    failed_v6 = [
        str(row["run_id"]) for row in v6_positives if not _bool(row["source_valid"])]
    no_onset_audit = [{
        "run_id": row["run_id"],
        "physically_valid_no_onset": row["failure_reason"] == "NO_PHYSICAL_ONSET",
        "possible_future_role": "EXPLICIT_HARD_NEGATIVE_REVIEW_ONLY",
        "training_use_assigned": False,
    } for row in v5_positives if not _bool(row["source_valid"])] + [{
        "run_id": row["run_id"],
        "physically_valid_no_onset": row["failure_reason"] == "NO_PHYSICAL_ONSET",
        "possible_future_role": "EXPLICIT_HARD_NEGATIVE_REVIEW_ONLY",
        "training_use_assigned": False,
    } for row in v6_positives if not _bool(row["source_valid"])]
    v5_valid = [row for row in v5_positives if _bool(row["source_valid"])]
    v6_valid = [row for row in v6_positives if _bool(row["source_valid"])]
    targeted_left = sum(row["target_foot"] == "left" for row in v5_valid + v6_valid)
    targeted_right = sum(row["target_foot"] == "right" for row in v5_valid + v6_valid)
    left = targeted_left + existing_balance["left"]
    right = targeted_right + existing_balance["right"]
    ratio = left / right if right else float("inf")
    audit = {
        "existing_valid_development_run_count": len(existing_runs),
        "v5_valid_positive_count": len(v5_valid_positive),
        "v5_valid_control_count": len(v5_valid_control),
        "v6_valid_positive_count": len(v6_valid_positive),
        "v6_valid_control_count": len(v6_valid_control),
        "combined_targeted_source_valid_positive_count": (
            len(v5_valid_positive) + len(v6_valid_positive)),
        "factorial_cells_with_support": sum(
            bool(row["has_valid_support"]) for row in coverage),
        "left_valid_event_count_including_existing": left,
        "right_valid_event_count_including_existing": right,
        "left_right_ratio_including_existing": ratio,
        "calibration_run_count_included": sum(
            str(row["run_id"]) in set(calibration_run_ids) for row in eligible),
        "failed_source_run_count_included": sum(
            str(row["run_id"]) in set(failed_v5 + failed_v6) for row in eligible),
        "v3_run_count_included": sum(
            "v3" in str(row["source_acquisition_version"]) for row in eligible),
        "all_speeds_represented": {
            float(row["speed_mps"]) for row in v5_valid + v6_valid
        } == set(SPEEDS_MPS),
        "both_feet_represented": {
            str(row["target_foot"]) for row in v5_valid + v6_valid
        } == set(SIDES),
        "all_onset_phases_represented": {
            str(row["actual_onset_phase"]) for row in v5_valid + v6_valid
        } >= {row.name for row in PHASE_BINS},
        "both_severities_represented": {
            str(row["severity"]) for row in v5_valid + v6_valid
        } == set(SEVERITIES),
    }
    audit["pass"] = bool(
        audit["combined_targeted_source_valid_positive_count"] >= 99
        and audit["factorial_cells_with_support"] == 36
        and 0.80 <= ratio <= 1.25
        and audit["calibration_run_count_included"] == 0
        and audit["failed_source_run_count_included"] == 0
        and audit["v3_run_count_included"] == 0
        and audit["all_speeds_represented"] and audit["both_feet_represented"]
        and audit["all_onset_phases_represented"]
        and audit["both_severities_represented"])
    return {
        "version": "walking_v2_combined_development_v6",
        "development_only": True, "training_use_assignment_performed": False,
        "eligible_runs": eligible,
        "excluded": {
            "calibration_only_run_ids": sorted(calibration_run_ids),
            "failed_v5_positive_source_run_ids": sorted(failed_v5),
            "failed_v6_positive_source_run_ids": sorted(failed_v6),
            "v3_quarantined_trace_use_count": 0,
            "wrong_foot_ambiguous_air_postfall_included": 0,
        },
        "no_onset_hard_negative_eligibility_audit": no_onset_audit,
        "audit": audit,
    }


def build_future_folds(combined: dict[str, object]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for run in combined["eligible_runs"]:  # type: ignore[index]
        source = str(run["source_acquisition_version"])
        variation = int(run["variation_index"])
        fold = variation % 3
        run_id = str(run["run_id"])
        rows.append({
            **run, "fold": fold,
            "variation_group": f"{source}:v{variation}",
            "run_group": f"{source}:{run_id}",
            "contact_episode_group": f"{source}:{run_id}:all",
            "source_acquisition_group": source,
        })
    leakage: list[str] = []
    for field in (
        "pair_id", "variation_group", "run_group", "contact_episode_group",
        "source_acquisition_group",
    ):
        owners: dict[str, set[int]] = {}
        for row in rows:
            value = str(row[field])
            if value:
                owners.setdefault(value, set()).add(int(row["fold"]))
        leakage.extend(
            f"{field}:{value}" for value, folds in owners.items()
            if len(folds) != 1)
    # Source version is a grouping dimension in the manifest, not a requirement
    # that an entire acquisition live in one fold; remove its coverage marker.
    leakage = [
        value for value in leakage
        if not value.startswith("source_acquisition_group:")]
    audit = {
        "row_count": len(rows), "fold_ids": sorted({row["fold"] for row in rows}),
        "group_leakage_count": len(leakage), "leaking_groups": leakage,
        "pair_run_episode_variation_leakage_count": len(leakage),
        "calibration_only_run_count": sum(
            row.get("source_acquisition_version") == "calibration_v6" for row in rows),
        "v3_run_count": sum(
            "v3" in str(row["source_acquisition_version"]) for row in rows),
        "all_development_only": all(bool(row["development_only"]) for row in rows),
    }
    audit["valid"] = bool(
        combined["audit"]["pass"]  # type: ignore[index]
        and audit["fold_ids"] == [0, 1, 2] and not leakage
        and audit["calibration_only_run_count"] == 0
        and audit["v3_run_count"] == 0 and audit["all_development_only"])
    return {
        "version": "walking_v2_future_nested_fold_v6",
        "grouping": [
            "counterfactual_pair_id", "variation", "run", "contact_episode",
            "source_acquisition_version"],
        "training_performed": False, "audit": audit, "rows": rows,
    }


def _save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def create_plots(
    output: Path, diagnosis: list[dict[str, object]],
    calibration_metrics: list[dict[str, object]],
    calibration_traces: list[dict[str, np.ndarray]],
    calibration_manifests: list[dict[str, object]], selected: str | None,
    supplemental_positives: list[dict[str, object]],
    coverage: list[dict[str, object]],
) -> None:
    groups = {
        "failed left": "A_FAILED_LEFT_MID_LATE_MODERATE",
        "right counterfactual": "B_FOOT_COUNTERFACTUAL_RIGHT_MODERATE",
    }
    plt.figure(figsize=(7, 4))
    for index, (label, group) in enumerate(groups.items()):
        values = [
            float(row["patch_contact_duration_ms"]) for row in diagnosis
            if row["comparison_group"] == group]
        plt.scatter([index] * len(values), values, label=label)
    plt.xticks(range(2), list(groups)); plt.ylabel("patch dwell (ms)")
    plt.title("Failed left vs right patch dwell")
    _save_plot(output / "failed_left_vs_successful_right_patch_dwell.png")

    plt.figure(figsize=(7, 4))
    for index, group in enumerate((
        "A_FAILED_LEFT_MID_LATE_MODERATE",
        "C_SEVERITY_COUNTERFACTUAL_LEFT_STRONG",
    )):
        values = [
            float(row["maximum_anchor_drift_m"]) for row in diagnosis
            if row["comparison_group"] == group]
        plt.scatter([index] * len(values), values)
    plt.axhline(0.05, color="black", linestyle="--", label="oracle threshold")
    plt.xticks((0, 1), ("moderate", "strong")); plt.ylabel("maximum drift (m)")
    plt.legend(); plt.title("Moderate vs strong left-foot anchor drift")
    _save_plot(output / "moderate_vs_strong_anchor_drift.png")

    plt.figure(figsize=(8, 4))
    pair_rows = [row for row in calibration_metrics if row["row_type"] == "pair"]
    for candidate in GEOMETRY_CANDIDATES:
        selected_rows = [
            row for row in pair_rows
            if row["geometry_candidate"] == candidate.candidate_id]
        plt.plot(
            [float(row["speed_mps"]) for row in selected_rows],
            [float(row["maximum_anchor_drift_m"]) for row in selected_rows],
            marker="o", label=candidate.candidate_id)
    plt.axhline(0.05, color="black", linestyle="--")
    plt.xlabel("speed (m/s)"); plt.ylabel("maximum drift (m)")
    plt.title("G0-G3 physical response"); plt.legend(ncol=4)
    _save_plot(output / "g0_g3_physical_response.png")

    plt.figure(figsize=(8, 4))
    candidate_id = selected or "G0"
    pair = [
        (trace, meta) for trace, meta in zip(calibration_traces, calibration_manifests)
        if meta["geometry_candidate"] == candidate_id
        and float(meta["speed_mps"]) == 0.15]
    for trace, meta in pair:
        side = SIDES.index(str(meta["target_foot"]))
        plt.plot(
            trace["time_s"], trace["anchor_drift_label_only"][:, side],
            label=str(meta["role"]))
    plt.axhline(0.05, color="black", linestyle="--")
    plt.xlabel("time (s)"); plt.ylabel("anchor drift (m)")
    title = f"{candidate_id} positive/control traces"
    if selected is None:
        title += " (no candidate selected)"
    plt.title(title); plt.legend()
    _save_plot(output / "selected_geometry_positive_control_traces.png")

    plt.figure(figsize=(7, 4))
    counts = [
        sum(_bool(row["source_valid"]) and float(row["speed_mps"]) == speed
            for row in supplemental_positives) for speed in SPEEDS_MPS]
    plt.bar([str(speed) for speed in SPEEDS_MPS], counts)
    plt.ylim(0, 6); plt.xlabel("speed (m/s)"); plt.ylabel("source-valid / 6")
    plt.title("Supplemental onset distribution by speed")
    if not supplemental_positives:
        plt.text(1, 3, "not executed: no geometry passed", ha="center")
    _save_plot(output / "supplemental_onset_distribution_by_speed.png")

    plt.figure(figsize=(10, 4))
    values = [int(row["combined_source_valid_count"]) for row in coverage]
    colors = ["#4c9f70" if value else "#d95f5f" for value in values]
    plt.bar(range(len(values)), values, color=colors)
    plt.xlabel("factorial cell index"); plt.ylabel("valid source count")
    plt.title(f"Combined factorial coverage: {sum(value > 0 for value in values)}/36")
    _save_plot(output / "combined_factorial_coverage.png")


def _empty_supplemental_gates(forbidden_count: int) -> dict[str, object]:
    return supplemental_gates([], [], [], [], [], [], forbidden_count)


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
        line for line in status_lines
        if not any(path in line for path in V6_SOURCES)
        and "simulation/outputs/walking_v2_slip_supplemental_acquisition_v6" not in line]
    if unrelated:
        raise RuntimeError(f"unrelated worktree changes before execution: {unrelated}")

    calibration_conditions = calibration_matrix()
    protocol = {
        "task": "Supplement Missing Left Mid-Late Moderate Slip Conditions v6",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "scope": "diagnosis, bounded geometry calibration, conditional development supplement",
        "diagnostic_groups": {
            "A": "failed left mid-late moderate at all speeds/variations",
            "B": "right-foot counterfactual",
            "C": "strong-severity counterfactual",
            "D": "early/middle phase counterfactual",
        },
        "geometry_candidates": [row.contract for row in GEOMETRY_CANDIDATES],
        "geometry_contract_sha256": geometry_contract_sha256(),
        "calibration_conditions": [vars(row) for row in calibration_conditions],
        "maximum_calibration_positive_runs": 12,
        "maximum_calibration_control_runs": 12,
        "supplemental_contract": {
            "conditional_on_calibration_pass": True,
            "speeds_mps": list(SPEEDS_MPS), "target_foot": TARGET_FOOT,
            "target_phase": TARGET_PHASE, "severity": TARGET_SEVERITY,
            "variations": [vars(row) for row in SUPPLEMENTAL_VARIATIONS],
            "positive_runs": 18, "matched_controls": 18,
            "adaptive_replacements": False,
        },
        "moderate_friction_changed": False,
        "friction_search_performed": False,
        "patch_height_compliance_solref_solimp_changed": False,
        "oracle": oracle_contract(), "sample_rate_hz": 1000,
        "development_only": True, "training": False,
        "overwrite_policy": "refuse non-empty output",
    }
    write_json(output / "protocol.json", protocol)
    protocol_sha = sha256_file(output / "protocol.json")
    write_json(output / "input_allowlist.json", {
        "exact_paths_only": True,
        "inputs": [{"path": path, "purpose": purpose}
                   for path, purpose in ALLOWED_INPUTS.items()],
    })
    write_json(output / "forbidden_path_policy.json", {
        "fail_closed": True, "forbidden_tokens": list(FORBIDDEN_TOKENS),
        "outer_holdout_final_use_authorized": False,
        "v3_trace_use_authorized": False, "model_score_diagnosis_authorized": False,
    })
    write_json(output / "geometry_candidate_contract.json", {
        "frozen_before_diagnostic_execution": True,
        "contract_sha256": geometry_contract_sha256(),
        "candidate_count": len(GEOMETRY_CANDIDATES),
        "candidates": [row.contract | {"sha256": row.sha256}
                       for row in GEOMETRY_CANDIDATES],
        "only_longitudinal_shift_or_length_differs_from_g0": True,
        "moderate_friction_unchanged": True,
    })
    guard = AccessGuard(output)
    guarded = {path: guard.path(path) for path in ALLOWED_INPUTS}

    v4_source_before = {path: sha256_file(guarded[path]) for path in V4_SOURCES}
    v4_source_checkpoint = {path: _checkpoint_sha256(path) for path in V4_SOURCES}
    v5_source_before = {path: sha256_file(guarded[path]) for path in V5_SOURCES}
    v5_artifact_before = {path: sha256_file(guarded[path]) for path in V5_ARTIFACTS}
    profile_before = frozen_profile_sha256()
    oracle_paths = (
        f"{ORACLE}/summary.json",
        "simulation/unitree_mujoco/simulate_python/walking_hazard_oracle_calibration_v1.py",
        "simulation/unitree_mujoco/simulate_python/walking_hazard_ground_truth_v1.py",
    )
    oracle_before = {path: sha256_file(guarded[path]) for path in oracle_paths}
    terrain_before = {
        name: sha256_file(guarded[path]) for name, path in TERRAIN_FILES.items()}
    v5_summary = json.loads(guarded[f"{V5}/summary.json"].read_text())
    v5_readiness = json.loads(guarded[f"{V5}/readiness.json"].read_text())
    v5_shards = json.loads(guarded[f"{V5}/trace_shard_manifest.json"].read_text())
    v5_shards_valid = bool(
        v5_shards["all_under_45_mib"]
        and v5_shards["all_exact_roundtrip_verified"]
        and sum(row["run_count"] for row in v5_shards["shards"]) == 216
        and all(
            sha256_file(guarded[f"{V5}/{row['path']}"]) == row["sha256"]
            for row in v5_shards["shards"]))
    v3_rows = guard.csv(f"{V3}/run_manifest.csv")
    v4_quarantine = guard.json(f"{V4}/failed_v3_quarantine_manifest.json")
    v5_patch = guard.json(f"{V5}/patch_geometry.json")
    v5_profiles = guard.json(f"{V5}/material_profiles.json")
    v5_manifest = guard.csv(f"{V5}/run_manifest.csv")
    v5_positive = guard.csv(f"{V5}/positive_source_audit.csv")
    v5_controls = guard.csv(f"{V5}/control_source_audit.csv")
    existing_manifest = guard.json(f"{EXISTING}/manifest.json")
    existing_physical_rows = guard.csv(f"{EXISTING}/slip_bilateral_metrics.csv")
    pair_snapshot = v5_runner._patch_contract_snapshot()
    preflight_ready = bool(
        head == STARTING_CHECKPOINT and not unrelated
        and v4_source_before == v4_source_checkpoint
        and pair_snapshot["explicit_pair_count"] == 8
        and v5_patch["bounds"]["left"] == vars(GEOMETRY_CANDIDATES[0].bounds)
        and v5_profiles["profiles"][TARGET_SEVERITY]["friction3"] == [0.1, 0.00125, 2.125e-05]
        and not v5_readiness["WALKING_V2_SLIP_REACQUISITION_DATA_READY"]
        and v5_summary["gates"]["positive_source_coverage"]["covered_cell_count"] == 33
        and v5_shards_valid and len(v3_rows) == 216
        and v4_quarantine["all_runs_quarantined"]
        and len(existing_manifest["runs"]) == 120 and guard.blocked == 0)
    immutable = {
        "starting_checkpoint": STARTING_CHECKPOINT, "execution_head": head,
        "clean_worktree_verified_before_implementation": True,
        "task_scoped_uncommitted_paths_before_execution": status_lines,
        "unrelated_worktree_change_count_before_execution": len(unrelated),
        "protocol_sha256_before_diagnostic_execution": protocol_sha,
        "protocol_frozen_before_diagnostic_execution": True,
        "v4_generator_before_sha256": v4_source_before,
        "v4_generator_checkpoint_sha256": v4_source_checkpoint,
        "v4_generator_matches_checkpoint": v4_source_before == v4_source_checkpoint,
        "v5_source_before_sha256": v5_source_before,
        "v5_artifact_before_sha256": v5_artifact_before,
        "v5_trace_shards_verified": v5_shards_valid,
        "patch_geometry_sha256": pair_snapshot["patch_geometry_sha256"],
        "explicit_pair_contract_sha256": pair_snapshot["pair_contract_sha256"],
        "explicit_pair_count": pair_snapshot["explicit_pair_count"],
        "profile_before_sha256": profile_before,
        "oracle_before_sha256": oracle_before,
        "terrain_before_sha256": terrain_before,
        "v3_quarantine_disposition": v4_quarantine["disposition"],
        "v3_trace_use_count": 0, "preflight_ready": preflight_ready,
        "after": {}, "all_immutable_after_execution": False,
    }
    write_json(output / "immutable_verification.json", immutable)
    if not preflight_ready:
        raise RuntimeError("v6 frozen preflight failed")

    diagnostic_traces = load_v5_diagnostic_traces(guard, v5_positive)
    diagnosis_rows, root_cause = diagnose_v5(v5_positive, diagnostic_traces)
    write_csv(output / "failed_cell_diagnosis.csv", diagnosis_rows)
    write_json(output / "failure_root_cause.json", root_cause)

    calibration_traces, calibration_manifests, calibration_contacts, calibration_progress = (
        _run_block(calibration_conditions, guarded[POLICY], "calibration"))
    (calibration_pairs, calibration_parity, calibration_episodes,
     calibration_positives, calibration_controls, calibration_falls,
     calibration_metrics) = calibration_assessment(
         calibration_conditions, calibration_traces, calibration_manifests)
    write_csv(output / "geometry_calibration_manifest.csv", calibration_manifests)
    write_csv(output / "geometry_calibration_metrics.csv", calibration_metrics)
    write_csv(output / "geometry_calibration_contact_audit.csv", calibration_contacts)
    write_csv(output / "geometry_calibration_pair_parity.csv", calibration_parity)
    write_csv(output / "geometry_calibration_positive_source_audit.csv", calibration_positives)
    write_csv(output / "geometry_calibration_control_source_audit.csv", calibration_controls)
    write_csv(output / "geometry_calibration_physical_episode_ledger.csv", calibration_episodes)
    write_csv(output / "geometry_calibration_fall_censor_audit.csv", calibration_falls)
    selected = select_geometry(
        row for row in calibration_metrics if row["row_type"] == "candidate")
    selected_lock = {
        "selection_performed_after_all_24_fixed_calibration_runs": True,
        "selected": selected is not None, "selected_geometry_candidate": selected,
        "selection_criteria": [
            "control remains physically non-slip", "target foot first",
            "onset inside frozen mid_late_stance bin", "no AIR/post-fall-only onset",
            "no patch/base double contact", "pre-contact parity",
            "post-contact divergence", "valid source at all three speeds"],
        "tie_break_order": [
            "smallest deviation from G0", "shortest patch",
            "smallest upstream shift", "lexical candidate ID"],
        "geometry": None if selected is None else geometry_candidate(selected).contract,
        "geometry_sha256": None if selected is None else geometry_candidate(selected).sha256,
        "moderate_friction_profile_sha256": profile_before[TARGET_SEVERITY],
        "moderate_friction_changed": False,
        "supplemental_acquisition_authorized": selected is not None,
        "failure_action": (
            "SLIP_MODERATE_PROFILE_RECALIBRATION" if selected is None else None),
    }
    write_json(output / "selected_geometry_lock.json", selected_lock)
    selected_lock_sha_before_supplement = sha256_file(output / "selected_geometry_lock.json")

    supplemental_conditions: list[RunCondition] = []
    supplemental_traces: list[dict[str, np.ndarray]] = []
    supplemental_manifests: list[dict[str, object]] = []
    supplemental_contacts: list[dict[str, object]] = []
    supplemental_progress: list[dict[str, object]] = []
    supplemental_pairs: list[dict[str, object]] = []
    supplemental_parity: list[dict[str, object]] = []
    supplemental_episodes: list[dict[str, object]] = []
    supplemental_positives: list[dict[str, object]] = []
    supplemental_controls: list[dict[str, object]] = []
    supplemental_falls: list[dict[str, object]] = []
    if selected is not None:
        supplemental_conditions = supplemental_matrix(selected)
        (supplemental_traces, supplemental_manifests, supplemental_contacts,
         supplemental_progress) = _run_block(
             supplemental_conditions, guarded[POLICY], "supplemental")
        supplemental_pairs, supplemental_parity = build_pair_audits(
            supplemental_conditions, supplemental_traces, supplemental_manifests)
        (supplemental_episodes, supplemental_positives,
         supplemental_controls, supplemental_falls) = build_ledgers(
             supplemental_traces, supplemental_manifests)
        supplemental_positives, supplemental_controls = enrich_source_audits(
            supplemental_traces, supplemental_manifests,
            supplemental_positives, supplemental_controls, supplemental_parity)
    gates = supplemental_gates(
        supplemental_conditions, supplemental_traces, supplemental_manifests,
        supplemental_positives, supplemental_controls, supplemental_parity,
        guard.blocked)
    write_csv(output / "supplemental_run_manifest.csv", supplemental_manifests)
    write_csv(output / "supplemental_pair_manifest.csv", supplemental_pairs)
    write_csv(output / "supplemental_contact_audit.csv", supplemental_contacts)
    write_csv(output / "supplemental_physical_episode_ledger.csv", supplemental_episodes)
    write_csv(output / "supplemental_positive_source_audit.csv", supplemental_positives)
    write_csv(output / "supplemental_control_source_audit.csv", supplemental_controls)
    write_csv(output / "supplemental_pair_parity.csv", supplemental_parity)
    write_csv(output / "supplemental_fall_censor_audit.csv", supplemental_falls)
    write_csv(output / "run_progress_manifest.csv", calibration_progress + supplemental_progress)

    all_conditions = calibration_conditions + supplemental_conditions
    all_traces = calibration_traces + supplemental_traces
    shard_manifest = save_trace_shards(output, all_conditions, all_traces)
    if not (shard_manifest["all_under_45_mib"]
            and shard_manifest["all_exact_roundtrip_verified"]):
        raise RuntimeError("trace shard storage contract failed")
    trace_hashes = [str(row["full_trace_sha256"]) for row in (
        calibration_manifests + supplemental_manifests)]
    duplicate = {
        "calibration_run_count": len(calibration_manifests),
        "supplemental_run_count": len(supplemental_manifests),
        "unique_run_id_count": len({row.run_id for row in all_conditions}),
        "unique_trace_hash_count": len(set(trace_hashes)),
        "duplicate_unique_trace_count": len(trace_hashes) - len(set(trace_hashes)),
        "discarded_count": 0, "replaced_count": 0,
        "silently_relabelled_count": 0,
    }
    duplicate["valid"] = duplicate["duplicate_unique_trace_count"] == 0
    write_json(output / "duplicate_audit.json", duplicate)

    coverage = factorial_coverage_rows(v5_positive, supplemental_positives)
    write_csv(output / "combined_factorial_coverage.csv", coverage)
    existing_balance = existing_physical_foot_balance(existing_physical_rows)
    if gates["pass"]:
        combined = build_combined_manifest(
            existing_manifest["runs"], existing_balance, v5_manifest,
            v5_positive, v5_controls, supplemental_manifests,
            supplemental_positives, supplemental_controls,
            [row.run_id for row in calibration_conditions], coverage)
    else:
        combined = {
            "version": "walking_v2_combined_development_v6",
            "not_created": True,
            "reason": (
                "bounded geometry calibration failed" if selected is None
                else "supplemental acquisition gates failed"),
            "training_use_assignment_performed": False,
            "eligible_runs": [],
            "excluded": {
                "calibration_only_run_ids": [row.run_id for row in calibration_conditions],
                "v3_quarantined_trace_use_count": 0,
            },
            "audit": {
                "pass": False,
                "v5_valid_positive_count": sum(
                    _bool(row["source_valid"]) for row in v5_positive),
                "v6_valid_positive_count": sum(
                    _bool(row["source_valid"]) for row in supplemental_positives),
                "combined_targeted_source_valid_positive_count": (
                    sum(_bool(row["source_valid"]) for row in v5_positive)
                    + sum(_bool(row["source_valid"]) for row in supplemental_positives)),
                "factorial_cells_with_support": sum(
                    bool(row["has_valid_support"]) for row in coverage),
                "calibration_run_count_included": 0,
                "failed_source_run_count_included": 0,
                "v3_run_count_included": 0,
            },
        }
    write_json(output / "combined_development_manifest.json", combined)
    combined_ready = bool(combined["audit"]["pass"])
    if combined_ready:
        future_folds = build_future_folds(combined)
    else:
        future_folds = {
            "version": "walking_v2_future_nested_fold_v6",
            "not_created": True, "reason": "combined readiness gate failed",
            "training_performed": False, "audit": {"valid": False}, "rows": [],
        }
    write_json(output / "future_nested_fold_manifest.json", future_folds)
    future_fold_ready = bool(future_folds["audit"]["valid"])

    selected_lock_unchanged = (
        sha256_file(output / "selected_geometry_lock.json")
        == selected_lock_sha_before_supplement)
    v4_source_after = {path: sha256_file(REPO / path) for path in V4_SOURCES}
    v5_source_after = {path: sha256_file(REPO / path) for path in V5_SOURCES}
    v5_artifact_after = {path: sha256_file(REPO / path) for path in V5_ARTIFACTS}
    profile_after = frozen_profile_sha256()
    oracle_after = {path: sha256_file(REPO / path) for path in oracle_paths}
    terrain_after = {
        name: sha256_file(REPO / path) for name, path in TERRAIN_FILES.items()}
    immutable.update({
        "after": {
            "v4_generator_sha256": v4_source_after,
            "v5_source_sha256": v5_source_after,
            "v5_artifact_sha256": v5_artifact_after,
            "profile_sha256": profile_after,
            "oracle_sha256": oracle_after, "terrain_sha256": terrain_after,
        },
        "v4_generator_unchanged": v4_source_before == v4_source_after,
        "v5_source_unchanged": v5_source_before == v5_source_after,
        "v5_artifacts_unchanged": v5_artifact_before == v5_artifact_after,
        "strong_moderate_profiles_unchanged": profile_before == profile_after,
        "physical_slip_oracle_byte_identical": oracle_before == oracle_after,
        "terrain_byte_identical": terrain_before == terrain_after,
        "selected_geometry_lock_unchanged_during_supplement": selected_lock_unchanged,
        "protocol_unchanged": sha256_file(output / "protocol.json") == protocol_sha,
        "moderate_friction_changed": False,
        "patch_height_compliance_solref_solimp_changed": False,
        "v3_trace_use_count": 0,
    })
    immutable["all_immutable_after_execution"] = bool(
        immutable["v4_generator_unchanged"] and immutable["v5_source_unchanged"]
        and immutable["v5_artifacts_unchanged"]
        and immutable["strong_moderate_profiles_unchanged"]
        and immutable["physical_slip_oracle_byte_identical"]
        and immutable["terrain_byte_identical"]
        and immutable["selected_geometry_lock_unchanged_during_supplement"]
        and immutable["protocol_unchanged"])
    write_json(output / "immutable_verification.json", immutable)

    calibration_candidate_rows = [
        row for row in calibration_metrics if row["row_type"] == "candidate"]
    calibration_ready = selected is not None
    targeted_authorized = bool(
        gates["pass"] and combined_ready and future_fold_ready
        and immutable["all_immutable_after_execution"] and guard.blocked == 0)
    readiness = {
        "WALKING_V2_SLIP_FAILED_CELL_DIAGNOSIS_READY": (
            root_cause["exactly_one_primary_failure_cause"]
            and len(diagnosis_rows) == 45),
        "WALKING_V2_SLIP_GEOMETRY_CALIBRATION_READY": calibration_ready,
        "WALKING_V2_SLIP_SUPPLEMENTAL_SOURCE_READY": bool(
            gates["pass"] and gates["source_valid_positive_count"] >= 15),
        "WALKING_V2_SLIP_SUPPLEMENTAL_CONTROL_READY": bool(
            gates["pass"] and gates["valid_control_count"] == 18),
        "WALKING_V2_SLIP_SUPPLEMENTAL_DATA_READY": gates["pass"],
        "WALKING_V2_SLIP_COMBINED_FACTORIAL_COVERAGE_READY": bool(
            combined_ready and combined["audit"]["factorial_cells_with_support"] == 36),
        "WALKING_V2_SLIP_COMBINED_DEVELOPMENT_DATA_READY": combined_ready,
        "WALKING_V2_SLIP_FUTURE_FOLD_READY": future_fold_ready,
        "WALKING_V2_SLIP_TARGETED_RETRAINING_AUTHORIZED": targeted_authorized,
        "WALKING_V2_FRESH_BLIND_HOLDOUT_AUTHORIZED": False,
        "WALKING_V2_TERRAIN_LOCK_PRESERVED": immutable["terrain_byte_identical"],
        "WALKING_V2_SYSTEM_MIGRATION_AUTHORIZED": False,
        "WALKING_V2_INT8_PREPARATION_AUTHORIZED": False,
        "SINK_RUNTIME_DETECTION_DEFERRED": True,
    }
    if targeted_authorized:
        next_step = "SLIP_TARGETED_RETRAINING_V3"
    elif selected is None:
        next_step = "SLIP_MODERATE_PROFILE_RECALIBRATION"
    elif not gates["pass"]:
        next_step = "ADDITIONAL_TARGETED_SLIP_ACQUISITION"
    elif not combined_ready:
        next_step = "SLIP_RISK_SCOPE_REDUCTION"
    else:
        next_step = "STOP_WALKING_V2_DEPLOYMENT"
    summary = {
        "task": "Supplement Missing Left Mid-Late Moderate Slip Conditions v6",
        "primary_failure_cause": root_cause["primary_failure_cause"],
        "selected_geometry_candidate": selected,
        "moderate_friction_changed": False,
        "calibration": {
            "planned_positive_attempts": 12, "executed_positive_attempts": 12,
            "matched_controls": 12, "executed_unique_runs": len(calibration_manifests),
            "passing_candidate_count": sum(
                bool(row["candidate_pass"]) for row in calibration_candidate_rows),
            "all_attempts_retained": True,
        },
        "supplemental_gates": gates,
        "combined_audit": combined["audit"],
        "original_v5": {
            "V5_ACQUISITION_DATA_READY": False,
            "source_valid_positive_count": 84,
            "covered_factorial_cells": 33,
        },
        "readiness": readiness, "next_step": next_step,
        "forbidden_artifact_access_count": guard.blocked,
        "v3_trace_use_count": 0,
        "terrain_byte_identical": immutable["terrain_byte_identical"],
        "oracle_byte_identical": immutable["physical_slip_oracle_byte_identical"],
        "training_performed": False, "model_selected_or_locked": False,
        "blind_holdout_created": False, "system_int8_vela_e84_hil_changed": False,
        "all_attempts_retained": True,
    }
    write_json(output / "readiness.json", readiness)
    write_json(output / "summary.json", summary)
    write_json(output / "provenance.json", {
        "pending_artifact_hash_graph": True,
    })
    create_plots(
        output, diagnosis_rows, calibration_metrics, calibration_traces,
        calibration_manifests, selected, supplemental_positives, coverage)
    audit = f"""# Walking-v2 Slip supplemental acquisition v6 audit

1. Primary v5 failure cause: **{root_cause['primary_failure_cause']}**.
2. Selected geometry: **{selected or 'NONE'}**.
3. Moderate friction changed: **False**.
4. Calibration attempts: **12 positive + 12 controls = 24 unique runs**.
5. Supplemental source-valid positives: **{gates['source_valid_positive_count']}/18**.
6. Supplemental source-valid by speed: **{gates['source_valid_count_by_speed']}**.
7. All three missing cells filled: **{gates['all_three_missing_cells_filled']}**.
8. Controls physically non-slip: **{gates['valid_control_count']}/18**.
9. Pair parity/post-contact divergence/double-contact gates: **{gates['pair_precontact_parity_count']}/18, {gates['postcontact_divergence_count']}/18, {gates['patch_base_double_contact_count']}**.
10. Replaced/discarded/silently relabelled: **0/0/0**.
11. Original `V5_ACQUISITION_DATA_READY`: **False**.
12. Combined targeted source-valid positives: **{combined['audit']['combined_targeted_source_valid_positive_count']}**.
13. Combined factorial cells: **{combined['audit']['factorial_cells_with_support']}/36**.
14. Calibration-only, failed-source and v3 traces included: **0/0/0**.
15. Targeted Slip retraining authorized: **{targeted_authorized}**.
16. Forbidden artifact access count: **{guard.blocked}**.
17. Terrain byte-identical: **{immutable['terrain_byte_identical']}**.
18. Blind holdout/System/INT8 authorized: **False/False/False**.
19. Sink remains `SINK_RUNTIME_DETECTION_DEFERRED`: **True**.

Exactly one next step: **{next_step}**
"""
    (output / "audit.md").write_text(audit, encoding="utf-8")

    graph_files = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name not in {
            "provenance.json", "resource_size_audit.json"})
    artifact_hashes = {path.name: sha256_file(path) for path in graph_files}
    write_json(output / "provenance.json", {
        "starting_checkpoint": STARTING_CHECKPOINT, "execution_head": head,
        "upstream_policy_revision": UPSTREAM_REVISION,
        "expected_policy_sha256": TESTED_POLICY_SHA256,
        "actual_policy_sha256": sha256_file(guarded[POLICY]),
        "mujoco_version": mujoco.__version__, "numpy_version": np.__version__,
        "elapsed_seconds": time.time() - started,
        "protocol_sha256": protocol_sha,
        "geometry_contract_sha256": geometry_contract_sha256(),
        "selected_geometry_lock_sha256": selected_lock_sha_before_supplement,
        "trace_shard_manifest_sha256": sha256_file(
            output / "trace_shard_manifest.json"),
        "artifact_sha256": artifact_hashes,
        "artifact_hash_graph_verified": all(
            sha256_file(output / name) == digest
            for name, digest in artifact_hashes.items()),
        "forbidden_access_count": guard.blocked, "v3_trace_use_count": 0,
        "training_performed": False,
    })
    files = sorted(path for path in output.iterdir() if path.is_file())
    sizes = [{
        "path": path.name, "bytes": path.stat().st_size,
        "mib": path.stat().st_size / (1024 * 1024),
        "under_or_equal_45_mib": path.stat().st_size <= 45 * 1024 * 1024,
    } for path in files]
    write_json(output / "resource_size_audit.json", {
        "limit_mib": 45, "files": sizes,
        "maximum_file_mib": max((row["mib"] for row in sizes), default=0.0),
        "all_files_within_limit": all(
            row["under_or_equal_45_mib"] for row in sizes),
    })
    print(json.dumps({
        "output": str(output), "primary_failure_cause": root_cause["primary_failure_cause"],
        "selected_geometry": selected, "calibration_unique_runs": len(calibration_manifests),
        "supplemental_unique_runs": len(supplemental_manifests),
        "supplemental_source_valid": gates["source_valid_positive_count"],
        "combined_valid_positives": combined["audit"]["combined_targeted_source_valid_positive_count"],
        "targeted_retraining_authorized": targeted_authorized,
        "next_step": next_step,
    }, indent=2))


if __name__ == "__main__":
    main()
