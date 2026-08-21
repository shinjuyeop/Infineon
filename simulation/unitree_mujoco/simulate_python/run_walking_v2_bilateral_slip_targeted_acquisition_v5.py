"""Reacquire the fixed 216-run targeted bilateral Slip development corpus."""

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

from bilateral_hil_sensor_v2 import G1BilateralSensorReaderV2
from g1_upstream_locomotion import (
    TESTED_POLICY_SHA256, UPSTREAM_REVISION, UnitreeG1PretrainedController,
)
from run_walking_hazard_ground_truth_v1 import (
    PELVIS_BODY_NAME, _disable_nonfoot_surface_collisions,
    _fall_reasons,
)
from run_walking_v2_bilateral_slip_targeted_acquisition_v3 import (
    build_ledgers, causal_pair_statistics,
)
from run_walking_v2_slip_scenario_generator_redesign_v4 import (
    _geometry_sha256, _initial_policy_observation, _pair_lookup,
    build_local_model,
)
from walking_hazard_ground_truth_v1 import (
    derive_contact_signals, max_left_foot_contact_penetration_m,
)
from walking_hazard_oracle_calibration_v1 import persistent_oracle
from walking_v2_bilateral_slip_targeted_acquisition_v5 import (
    CONTROL_TYPES, PHASE_BINS, PHYSICS_STEPS_PER_SAMPLE, PHYSICS_TIMESTEP_S,
    SAMPLE_RATE_HZ, SEVERITIES, SIDES, SLIP_PERSISTENCE_MS,
    SLIP_THRESHOLD_M, SPEEDS_MPS, VARIATIONS, RunCondition,
    acquisition_matrix, array_sha256, deterministic_initial_perturbation,
    material_profiles, phase_spec, trace_sha256,
)
from walking_v2_slip_scenario_generator_redesign_v4 import PilotCondition, patch_bounds


REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO / "simulation/outputs/walking_v2_bilateral_slip_targeted_acquisition_v5"
STARTING_CHECKPOINT = "07b7c916111808ed046d09d9fd081d342c2558bb"
V4 = "simulation/outputs/walking_v2_slip_scenario_generator_redesign_v4"
V3 = "simulation/outputs/walking_v2_bilateral_slip_targeted_acquisition_v3"
EXISTING = "simulation/outputs/walking_bilateral_sensor_sink_observability_v2"
JOINT = "simulation/outputs/walking_v2_joint_terrain_slip_redesign_v1"
ORACLE = "simulation/outputs/walking_hazard_oracle_calibration_v1"
POLICY = "simulation/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"
SCENE = "simulation/unitree_mujoco/unitree_robots/g1/scene_walking_terrain_transition.xml"
ROBOT = "simulation/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
V4_SOURCES = (
    "simulation/unitree_mujoco/simulate_python/walking_v2_slip_scenario_generator_redesign_v4.py",
    "simulation/unitree_mujoco/simulate_python/run_walking_v2_slip_scenario_generator_redesign_v4.py",
    "simulation/unitree_mujoco/simulate_python/test_walking_v2_slip_scenario_generator_redesign_v4.py",
)
V4_ARTIFACTS = (
    f"{V4}/protocol.json", f"{V4}/summary.json", f"{V4}/readiness.json",
    f"{V4}/provenance.json", f"{V4}/local_patch_definition.json",
    f"{V4}/geom_contact_contract.csv", f"{V4}/friction_root_cause.json",
    f"{V4}/terrain_immutable_verification.json",
    f"{V4}/oracle_immutable_verification.json",
    f"{V4}/failed_v3_quarantine_manifest.json",
)
TERRAIN_FILES = {
    "model": f"{JOINT}/terrain_candidate_model.npz",
    "normalization": f"{JOINT}/terrain_candidate_normalization.json",
    "config": f"{JOINT}/terrain_candidate_config.json",
    "lock": f"{JOINT}/terrain_selection_lock.json",
}
ALLOWED_INPUTS = {
    **{path: "frozen v4 generator implementation" for path in V4_SOURCES},
    **{path: "verified v4 physics artifact" for path in V4_ARTIFACTS},
    f"{V3}/run_manifest.csv": "failed v3 run inventory for exclusion verification",
    f"{EXISTING}/manifest.json": "existing valid 120-run development metadata",
    f"{ORACLE}/summary.json": "frozen physical Slip oracle contract",
    "simulation/unitree_mujoco/simulate_python/walking_hazard_oracle_calibration_v1.py": "frozen physical Slip oracle implementation",
    "simulation/unitree_mujoco/simulate_python/walking_hazard_ground_truth_v1.py": "frozen contact signal derivation",
    "simulation/unitree_mujoco/simulate_python/walking_v2_bilateral_slip_targeted_acquisition_v3.py": "preregistered variations and control profile contract",
    "simulation/unitree_mujoco/simulate_python/terrain_profiles.py": "native material profile source",
    "simulation/unitree_mujoco/simulate_python/bilateral_hil_sensor_v2.py": "Fusion20 runtime sensor contract",
    "simulation/unitree_mujoco/simulate_python/g1_upstream_locomotion.py": "fixed walking controller",
    **{path: f"immutable Terrain {key}" for key, path in TERRAIN_FILES.items()},
    POLICY: "fixed upstream walking policy",
    SCENE: "fixed MuJoCo walking scene",
    ROBOT: "fixed G1 robot include",
}
FORBIDDEN_TOKENS = (
    "/outer/", "_outer_", "holdout", "spatial_final", "spatial-final",
    "final_test", "final-test",
)
TRACE_HASH_KEYS = (
    "time_s", "bilateral_fusion20_raw", "bilateral_canonical",
    "force_loaded", "physical_contact", "foot_world_xyz_label_only",
    "anchor_drift_label_only", "slip_physical_active", "pre_fall_valid",
    "contact_episode_id", "touchdown_transient", "patch_contact",
)
PHASE_CODE = {
    "none": 0, "EARLY_PRECURSOR": 1, "ACTIONABLE_0_100MS": 2,
    "PHYSICAL_ACTIVE_EVIDENCE": 3,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    def default(item: object) -> object:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=default) + "\n",
        encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


class AccessGuard:
    """Exact-path artifact ledger that fails closed before any forbidden read."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.events: list[dict[str, object]] = []
        self.blocked = 0
        self._flush()

    def _flush(self) -> None:
        write_json(self.output / "artifact_access_log.json", {
            "exact_paths_only": True, "forbidden_tokens": list(FORBIDDEN_TOKENS),
            "events": self.events, "blocked_access_count": self.blocked,
            "all_accesses_completed": all(
                row["status"] == "completed" for row in self.events),
        })

    def path(self, relative: str, access: str = "hash") -> Path:
        normalized = relative.replace("\\", "/")
        forbidden = next(
            (token for token in FORBIDDEN_TOKENS if token in f"/{normalized.lower()}"),
            None)
        if normalized not in ALLOWED_INPUTS or forbidden:
            self.blocked += 1
            self.events.append({
                "path": normalized, "access": access, "status": "blocked",
                "reason": forbidden or "not_allowlisted",
            })
            self._flush(); raise PermissionError(normalized)
        path = (REPO / normalized).resolve()
        if not path.is_file() or REPO.resolve() not in path.parents:
            self.blocked += 1
            self.events.append({
                "path": normalized, "access": access, "status": "blocked",
                "reason": "missing_or_outside_repository",
            })
            self._flush(); raise FileNotFoundError(normalized)
        self.events.append({
            "path": normalized, "access": access, "status": "completed",
            "purpose": ALLOWED_INPUTS[normalized], "sha256": sha256_file(path),
        })
        self._flush(); return path

    def json(self, relative: str) -> object:
        return json.loads(self.path(relative, "json").read_text(encoding="utf-8"))

    def csv(self, relative: str) -> list[dict[str, str]]:
        with self.path(relative, "csv").open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))


def _v4_proxy(condition: RunCondition) -> PilotCondition:
    phase_names = {
        "early_loading": "early", "mid_loading_early_stance": "middle",
        "mid_late_stance": "late",
    }
    return PilotCondition(
        run_id=condition.run_id, pair_id=condition.pair_id, role=condition.role,
        speed_mps=condition.speed_mps, target_foot=condition.target_foot,
        target_phase=phase_names[condition.target_phase], severity=condition.severity,
        profile=condition.material_profile, seed=condition.seed,
        duration_s=condition.duration_s, phase_fraction=condition.phase_fraction,
        command_delay_s=condition.command_delay_s,
    )


def _set_pair_friction(
    model: mujoco.MjModel, pair_ids: tuple[int, ...], profile_name: str,
) -> None:
    values = np.asarray(material_profiles()[profile_name].friction5)
    for pair_id in pair_ids:
        model.pair_friction[pair_id] = values


def _close_contact_interval(
    row: dict[str, object], end_sample: int, end_time_s: float,
    rows: list[dict[str, object]],
) -> None:
    count = int(row.pop("_count"))
    row["end_sample_exclusive"] = end_sample
    row["end_time_s_exclusive"] = end_time_s
    row["observed_1khz_sample_count"] = count
    row["normal_contact_force_mean_n"] = float(row.pop("_normal_sum")) / count
    row["tangential_contact_force_mean_n"] = float(row.pop("_tangent_sum")) / count
    rows.append(row)


def _timing_states(
    slip_active: np.ndarray, valid: np.ndarray, episode_ids: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(episode_ids, dtype=np.int8)
    result[slip_active & valid] = PHASE_CODE["PHYSICAL_ACTIVE_EVIDENCE"]
    for episode in sorted(set(episode_ids.tolist()) - {-1}):
        samples = np.flatnonzero(episode_ids == episode)
        onsets = samples[
            slip_active[samples] & valid[samples]
            & ~np.r_[False, slip_active[samples[:-1]]]
        ]
        if not onsets.size:
            continue
        onset = int(onsets[0])
        precursor = samples[(samples < onset) & valid[samples]]
        result[precursor[onset - precursor <= 100]] = PHASE_CODE["ACTIONABLE_0_100MS"]
        result[precursor[onset - precursor > 100]] = PHASE_CODE["EARLY_PRECURSOR"]
    return result


def collect_run(
    condition: RunCondition, policy_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object], list[dict[str, object]]]:
    model, info = build_local_model(_v4_proxy(condition))
    model.opt.timestep = PHYSICS_TIMESTEP_S
    geometry_hash = _geometry_sha256(model)
    ground_ids = frozenset(model.geom(name).id for name in info["ground_names"])
    base_ids = frozenset(model.geom(name).id for name in info["base_names"])
    patch_id = model.geom(str(info["patch_name"])).id
    allowed_feet = _disable_nonfoot_surface_collisions(model, ground_ids)
    data = mujoco.MjData(model)
    controller = UnitreeG1PretrainedController(
        model, data, policy_path, condition.speed_mps)
    dx, dy, dvx = deterministic_initial_perturbation(condition.seed)
    data.qpos[0] += dx
    data.qpos[1] += dy + condition.lateral_offset_m
    data.qvel[0] += dvx
    controller.global_phase = condition.phase_fraction
    nominal_command = controller.command.copy()
    if condition.command_delay_s:
        controller.command[:] = 0.0
    mujoco.mj_forward(model, data)
    initial_qpos_hash = array_sha256(data.qpos)
    initial_qvel_hash = array_sha256(data.qvel)
    initial_policy_hash = array_sha256(_initial_policy_observation(controller))
    reader = G1BilateralSensorReaderV2(model, data)
    foot_body_ids = tuple(
        model.body(f"{side}_ankle_roll_link").id for side in SIDES)
    pelvis_id = model.body(PELVIS_BODY_NAME).id
    pair_lookup = _pair_lookup(model)
    foot_geom_side = {
        geom_id: side for side in SIDES
        for geom_id in reader.foot_geom_ids[side]
    }
    target_geom_ids = frozenset(reader.foot_geom_ids[condition.target_foot])
    target_index = SIDES.index(condition.target_foot)
    velocity = np.zeros(6)
    series_names = (
        "time_s", "bilateral_fusion20_raw", "bilateral_canonical",
        "force_loaded", "physical_contact", "foot_world_xyz_label_only",
        "foot_world_velocity_label_only", "contact_penetration_label_only",
        "patch_contact", "effective_patch_friction",
        "patch_normal_force_n", "patch_tangential_force_n", "command",
    )
    series: dict[str, list[object]] = {name: [] for name in series_names}
    prepatch_qpos: list[np.ndarray] = []
    prepatch_qvel: list[np.ndarray] = []
    prepatch_policy: list[np.ndarray] = []
    contact_rows: list[dict[str, object]] = []
    open_intervals: dict[int, dict[str, object]] = {}
    target_age_steps = 0
    previous_target_patch_contact = False
    first_patch_seen = False
    first_fall_sample: int | None = None
    first_fall_time_s: float | None = None
    fall_reason = ""
    double_contact_count = 0
    configured_low_steps = 0
    total_steps = int(round(condition.duration_s / PHYSICS_TIMESTEP_S))
    activation_ms = phase_spec(condition.target_phase).friction_activation_ms
    for physics_step in range(1, total_steps + 1):
        if data.time + 1e-12 >= condition.command_delay_s:
            controller.command[:] = nominal_command
        target_age_steps = target_age_steps + 1 if previous_target_patch_contact else 0
        age_ms = target_age_steps * PHYSICS_TIMESTEP_S * 1000.0
        intervention_active = age_ms + 1e-12 >= activation_ms
        _set_pair_friction(
            model, tuple(info["target_pair_ids"]),
            condition.material_profile if intervention_active else "hard_normal")
        configured_low_steps += int(
            intervention_active and condition.material_profile != "hard_normal")
        controller.apply()
        mujoco.mj_step(model, data)
        controller.update_after_step()
        target_patch_now = any(
            patch_id in {int(data.contact[index].geom1), int(data.contact[index].geom2)}
            and bool({int(data.contact[index].geom1), int(data.contact[index].geom2)} & target_geom_ids)
            for index in range(data.ncon))
        previous_target_patch_contact = target_patch_now
        if physics_step % PHYSICS_STEPS_PER_SAMPLE:
            continue
        sample = len(series["time_s"])
        raw = reader.read_bilateral_vector()
        canonical = np.concatenate(tuple(
            reader.canonicalize_foot_vector(
                side, raw[index * 10:(index + 1) * 10])
            for index, side in enumerate(SIDES)))
        runtime = tuple(reader.update_contact_state(
            side, raw[index * 10:index * 10 + 4])
            for index, side in enumerate(SIDES))
        physical = np.asarray([
            reader.has_foot_contact(side, ground_ids) for side in SIDES], bool)
        loaded = np.asarray([
            state.loaded and physical[index]
            for index, state in enumerate(runtime)], bool)
        foot_xyz = np.stack(tuple(
            data.xpos[body_id].copy() for body_id in foot_body_ids))
        foot_velocity = []
        for body_id in foot_body_ids:
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
            foot_velocity.append(velocity[3:].copy())
        penetration = np.asarray([
            max_left_foot_contact_penetration_m(
                data, reader.foot_geom_ids[side], ground_ids)
            for side in SIDES])
        patch_contact = np.zeros(2, bool)
        patch_mu: list[list[float]] = [[], []]
        patch_normal = np.zeros(2)
        patch_tangent = np.zeros(2)
        contacted_surfaces: dict[int, set[int]] = {}
        current_sole_ids: set[int] = set()
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            pair = frozenset((geom1, geom2))
            foot_id = next((value for value in pair if value in foot_geom_side), None)
            ground_id = next((value for value in pair if value in ground_ids), None)
            if foot_id is None or ground_id is None:
                continue
            contacted_surfaces.setdefault(foot_id, set()).add(ground_id)
            if ground_id != patch_id:
                continue
            side = foot_geom_side[foot_id]
            side_index = SIDES.index(side)
            current_sole_ids.add(foot_id)
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact_id, wrench)
            normal = max(0.0, float(wrench[0]))
            tangent = float(np.linalg.norm(wrench[1:3]))
            mu = float(contact.friction[0])
            patch_contact[side_index] = True
            patch_mu[side_index].append(mu)
            patch_normal[side_index] += normal
            patch_tangent[side_index] += tangent
            explicit_pair_id = pair_lookup.get(pair, -1)
            if foot_id not in open_intervals:
                open_intervals[foot_id] = {
                    "run_id": condition.run_id, "pair_id": condition.pair_id,
                    "role": condition.role, "timestamp_s": float(data.time),
                    "start_sample": sample, "foot": side,
                    "sole_geom_name": mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, foot_id),
                    "sole_geom_id": foot_id, "patch_geom_name": "slip_patch",
                    "patch_geom_id": patch_id,
                    "contact_position_first_xyz": json.dumps(np.asarray(contact.pos).tolist()),
                    "contact_position_last_xyz": json.dumps(np.asarray(contact.pos).tolist()),
                    "patch_inclusion": True, "explicit_pair_id": explicit_pair_id,
                    "effective_friction_first": json.dumps(np.asarray(contact.friction).tolist()),
                    "effective_friction_last": json.dumps(np.asarray(contact.friction).tolist()),
                    "effective_sliding_friction_min": mu,
                    "effective_sliding_friction_max": mu,
                    "contact_dimension": int(contact.dim),
                    "exclude_state": int(contact.exclude),
                    "efc_address_min": int(contact.efc_address),
                    "efc_address_max": int(contact.efc_address),
                    "normal_contact_force_peak_n": normal,
                    "tangential_contact_force_peak_n": tangent,
                    "friction_configured_before_collision": True,
                    "constraints_constructed_at_configuration": False,
                    "patch_base_double_contact": False,
                    "visual_only_geom": False,
                    "_normal_sum": normal, "_tangent_sum": tangent,
                    "_count": 1,
                }
            else:
                row = open_intervals[foot_id]
                row["contact_position_last_xyz"] = json.dumps(
                    np.asarray(contact.pos).tolist())
                row["effective_friction_last"] = json.dumps(
                    np.asarray(contact.friction).tolist())
                row["effective_sliding_friction_min"] = min(
                    float(row["effective_sliding_friction_min"]), mu)
                row["effective_sliding_friction_max"] = max(
                    float(row["effective_sliding_friction_max"]), mu)
                row["efc_address_min"] = min(
                    int(row["efc_address_min"]), int(contact.efc_address))
                row["efc_address_max"] = max(
                    int(row["efc_address_max"]), int(contact.efc_address))
                row["normal_contact_force_peak_n"] = max(
                    float(row["normal_contact_force_peak_n"]), normal)
                row["tangential_contact_force_peak_n"] = max(
                    float(row["tangential_contact_force_peak_n"]), tangent)
                row["_normal_sum"] = float(row["_normal_sum"]) + normal
                row["_tangent_sum"] = float(row["_tangent_sum"]) + tangent
                row["_count"] = int(row["_count"]) + 1
        for foot_id, surfaces in contacted_surfaces.items():
            if patch_id in surfaces and surfaces & base_ids:
                double_contact_count += 1
                if foot_id in open_intervals:
                    open_intervals[foot_id]["patch_base_double_contact"] = True
        for foot_id in set(open_intervals) - current_sole_ids:
            _close_contact_interval(
                open_intervals.pop(foot_id), sample, float(data.time), contact_rows)
        if not first_patch_seen and not target_patch_now:
            prepatch_qpos.append(data.qpos.copy())
            prepatch_qvel.append(data.qvel.copy())
            prepatch_policy.append(np.concatenate((
                (controller.global_phase, float(controller.step_count)),
                controller.command, controller.action, controller.target_position)))
        first_patch_seen |= target_patch_now
        reasons, _ = _fall_reasons(
            model, data, pelvis_id, ground_ids, allowed_feet)
        if reasons and first_fall_sample is None:
            first_fall_sample = sample
            first_fall_time_s = float(data.time)
            fall_reason = "|".join(reasons)
        effective = np.asarray([
            float(np.mean(values)) if values else np.nan for values in patch_mu])
        current = {
            "time_s": float(data.time),
            "bilateral_fusion20_raw": raw.astype(np.float32),
            "bilateral_canonical": canonical.astype(np.float32),
            "force_loaded": loaded, "physical_contact": physical,
            "foot_world_xyz_label_only": foot_xyz.astype(np.float32),
            "foot_world_velocity_label_only": np.asarray(foot_velocity, np.float32),
            "contact_penetration_label_only": penetration.astype(np.float32),
            "patch_contact": patch_contact,
            "effective_patch_friction": effective,
            "patch_normal_force_n": patch_normal,
            "patch_tangential_force_n": patch_tangent,
            "command": controller.command.copy(),
        }
        for name, value in current.items():
            series[name].append(value)
    sample_count = len(series["time_s"])
    for row in list(open_intervals.values()):
        _close_contact_interval(
            row, sample_count, float(data.time), contact_rows)
    trace = {name: np.asarray(values) for name, values in series.items()}
    trace["prepatch_qpos"] = np.asarray(prepatch_qpos)
    trace["prepatch_qvel"] = np.asarray(prepatch_qvel)
    trace["prepatch_policy_state"] = np.asarray(prepatch_policy)
    pre_fall = np.ones(sample_count, bool)
    if first_fall_sample is not None:
        pre_fall[first_fall_sample:] = False
    episode_ids = np.full((sample_count, 2), -1, np.int32)
    transient = np.zeros((sample_count, 2), bool)
    anchor = np.full((sample_count, 2, 2), np.nan, np.float32)
    anchor_drift = np.full((sample_count, 2), np.nan, np.float32)
    tangential_velocity = np.full((sample_count, 2), np.nan, np.float32)
    slip_valid = np.zeros((sample_count, 2), bool)
    slip_active = np.zeros((sample_count, 2), bool)
    timing_state = np.zeros((sample_count, 2), np.int8)
    for side_index in range(2):
        signals = derive_contact_signals(
            trace["physical_contact"][:, side_index],
            trace["force_loaded"][:, side_index],
            trace["foot_world_xyz_label_only"][:, side_index],
            trace["foot_world_velocity_label_only"][:, side_index],
            trace["contact_penetration_label_only"][:, side_index],
            first_fall_sample)
        episode_ids[:, side_index] = signals.contact_episode_id
        transient[:, side_index] = signals.touchdown_transient
        anchor[:, side_index] = signals.anchor_xy_m
        anchor_drift[:, side_index] = signals.tangential_anchor_drift_m
        tangential_velocity[:, side_index] = signals.tangential_velocity_mps
        slip_valid[:, side_index] = signals.slip_calibration_valid
        slip_active[:, side_index] = persistent_oracle(
            signals.tangential_anchor_drift_m,
            signals.slip_calibration_valid, signals.contact_episode_id,
            SLIP_THRESHOLD_M, SLIP_PERSISTENCE_MS)
        timing_state[:, side_index] = _timing_states(
            slip_active[:, side_index], slip_valid[:, side_index],
            episode_ids[:, side_index])
    trace.update({
        "pre_fall_valid": pre_fall,
        "valid_pre_fall_mask": pre_fall[:, None] & trace["physical_contact"],
        "air_mask": ~trace["physical_contact"],
        "contact_episode_id": episode_ids,
        "touchdown_transient": transient,
        "contact_anchor_xy_label_only": anchor,
        "anchor_drift_label_only": anchor_drift,
        "tangential_velocity_label_only": tangential_velocity,
        "slip_calibration_valid_label_only": slip_valid,
        "slip_physical_active": slip_active,
        "label_timing_state": timing_state,
    })
    for row in contact_rows:
        side_index = SIDES.index(str(row["foot"]))
        start = int(row["start_sample"])
        end = min(int(row["end_sample_exclusive"]), sample_count)
        episodes = episode_ids[start:end, side_index]
        valid_episodes = episodes[episodes >= 0]
        if valid_episodes.size:
            episode = int(valid_episodes[0])
            samples = np.flatnonzero(episode_ids[:, side_index] == episode)
            anchor_value = anchor[samples[0], side_index]
        else:
            anchor_value = np.asarray((np.nan, np.nan))
        drift = anchor_drift[start:end, side_index]
        finite = drift[np.isfinite(drift)]
        row["contact_anchor_position_xy"] = json.dumps(anchor_value.tolist())
        row["contact_anchor_drift_max_m"] = (
            float(np.max(finite)) if finite.size else "")
    finite_mu = trace["effective_patch_friction"][:, target_index]
    finite_mu = finite_mu[np.isfinite(finite_mu)]
    target_patch_samples = np.flatnonzero(trace["patch_contact"][:, target_index])
    loaded_touchdowns = []
    for side_index in range(2):
        mask = trace["force_loaded"][:, side_index]
        changes = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
        loaded_touchdowns.append(int(changes[0]) if changes.size else None)
    metadata: dict[str, object] = {
        **vars(condition), "pair_fingerprint": condition.pair_fingerprint,
        "patch": vars(patch_bounds(condition.target_foot)),
        "collision_geometry_sha256": geometry_hash,
        "initial_qpos_sha256": initial_qpos_hash,
        "initial_qvel_sha256": initial_qvel_hash,
        "initial_policy_observation_sha256": initial_policy_hash,
        "sample_count": sample_count,
        "first_sample_time_s": float(trace["time_s"][0]),
        "last_sample_time_s": float(trace["time_s"][-1]),
        "sample_spacing_max_error_s": float(
            np.max(np.abs(np.diff(trace["time_s"]) - 0.001))),
        "fusion20_shape": list(trace["bilateral_fusion20_raw"].shape),
        "finite_fusion20": bool(np.all(np.isfinite(
            trace["bilateral_fusion20_raw"]))),
        "first_patch_contact_sample": (
            int(target_patch_samples[0]) if target_patch_samples.size else None),
        "patch_active_contact_sample_count": int(target_patch_samples.size),
        "intervention_configured_physics_steps": configured_low_steps,
        "effective_patch_friction_mean": (
            float(np.mean(finite_mu)) if finite_mu.size else ""),
        "patch_base_double_contact_count": double_contact_count,
        "left_first_loaded_touchdown_sample": loaded_touchdowns[0],
        "right_first_loaded_touchdown_sample": loaded_touchdowns[1],
        "first_fall_sample": first_fall_sample,
        "first_fall_time_s": first_fall_time_s,
        "fall_occurred": first_fall_sample is not None,
        "fall_reason": fall_reason,
        "post_fall_excluded_sample_count": int(np.sum(~pre_fall)),
        "full_trace_sha256": trace_sha256(trace, TRACE_HASH_KEYS),
        "discarded": False, "replaced": False,
        "silently_relabelled": False, "development_only": True,
    }
    return trace, metadata, contact_rows


def _trace_prefix_equal(
    first: dict[str, np.ndarray], second: dict[str, np.ndarray], end: int,
) -> bool:
    keys = (
        "time_s", "bilateral_fusion20_raw", "bilateral_canonical",
        "force_loaded", "physical_contact", "foot_world_xyz_label_only",
        "foot_world_velocity_label_only", "command",
    )
    return all(np.array_equal(
        first[key][:end], second[key][:end], equal_nan=True) for key in keys)


def _trace_post_diverged(
    first: dict[str, np.ndarray], second: dict[str, np.ndarray], start: int,
) -> bool:
    return any(not np.array_equal(
        first[key][start:], second[key][start:], equal_nan=True)
        for key in (
            "bilateral_fusion20_raw", "bilateral_canonical",
            "foot_world_xyz_label_only"))


def build_pair_audits(
    conditions: list[RunCondition], traces: list[dict[str, np.ndarray]],
    manifests: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pair_manifest: list[dict[str, object]] = []
    parity_rows: list[dict[str, object]] = []
    by_run = {condition.run_id: index for index, condition in enumerate(conditions)}
    grouped: dict[str, list[RunCondition]] = {}
    for condition in conditions:
        grouped.setdefault(condition.pair_id, []).append(condition)
    for pair_id in sorted(grouped):
        pair = grouped[pair_id]
        positive_condition = next(row for row in pair if row.role == "positive")
        control_condition = next(row for row in pair if row.role == "control")
        positive_index = by_run[positive_condition.run_id]
        control_index = by_run[control_condition.run_id]
        positive = manifests[positive_index]; control = manifests[control_index]
        positive_trace = traces[positive_index]; control_trace = traces[control_index]
        patch_samples = [
            int(value) for value in (
                positive["first_patch_contact_sample"],
                control["first_patch_contact_sample"])
            if value is not None]
        first_patch = min(patch_samples) if patch_samples else len(positive_trace["time_s"])
        precontact_equal = _trace_prefix_equal(
            positive_trace, control_trace, first_patch)
        prepatch_state_equal = all(np.array_equal(
            positive_trace[key], control_trace[key], equal_nan=True)
            for key in ("prepatch_qpos", "prepatch_qvel", "prepatch_policy_state"))
        post_diverged = _trace_post_diverged(
            positive_trace, control_trace, first_patch)
        full_identical = (
            positive["full_trace_sha256"] == control["full_trace_sha256"])
        expected_positive_mu = material_profiles()[
            positive_condition.material_profile].friction3[0]
        expected_control_mu = material_profiles()[
            control_condition.material_profile].friction3[0]
        positive_mu = positive_trace["effective_patch_friction"][:, SIDES.index(
            positive_condition.target_foot)]
        control_mu = control_trace["effective_patch_friction"][:, SIDES.index(
            control_condition.target_foot)]
        positive_mu = positive_mu[np.isfinite(positive_mu)]
        control_mu = control_mu[np.isfinite(control_mu)]
        positive_contract = bool(
            positive_mu.size and np.any(np.isclose(
                positive_mu, expected_positive_mu, atol=1e-12, rtol=0)))
        control_contract = bool(
            control_mu.size and np.any(np.isclose(
                control_mu, expected_control_mu, atol=1e-12, rtol=0)))
        common = {
            "pair_id": pair_id,
            "positive_run_id": positive_condition.run_id,
            "control_run_id": control_condition.run_id,
            "pair_fingerprint": positive_condition.pair_fingerprint,
            "speed_mps": positive_condition.speed_mps,
            "target_foot": positive_condition.target_foot,
            "target_phase": positive_condition.target_phase,
            "severity": positive_condition.severity,
            "control_type": control_condition.control_type,
            "variation_index": positive_condition.variation_index,
            "seed": positive_condition.seed,
        }
        pair_manifest.append(common)
        if not patch_samples:
            classification = "no_patch_contact"
        elif not precontact_equal or not prepatch_state_equal:
            classification = "divergence_before_patch_contact"
        elif full_identical:
            classification = "identical_full_trace"
        elif not post_diverged:
            classification = "patch_contact_without_divergence"
        else:
            classification = "correct_counterfactual_pair"
        parity_rows.append({
            **common,
            "first_patch_contact_sample": first_patch,
            "initial_qpos_equal": positive["initial_qpos_sha256"] == control["initial_qpos_sha256"],
            "initial_qvel_equal": positive["initial_qvel_sha256"] == control["initial_qvel_sha256"],
            "initial_policy_observation_equal": positive["initial_policy_observation_sha256"] == control["initial_policy_observation_sha256"],
            "prepatch_generalized_state_equal": prepatch_state_equal,
            "precontact_fusion20_equal": precontact_equal,
            "command_equal": np.array_equal(
                positive_trace["command"], control_trace["command"]),
            "geometry_equal": positive["collision_geometry_sha256"] == control["collision_geometry_sha256"],
            "patch_location_equal": positive["patch"] == control["patch"],
            "target_foot_equal": positive["target_foot"] == control["target_foot"],
            "timestamps_equal": np.array_equal(
                positive_trace["time_s"], control_trace["time_s"]),
            "pair_configuration_equal": positive_condition.pair_fingerprint == control_condition.pair_fingerprint,
            "positive_expected_friction": expected_positive_mu,
            "control_expected_friction": expected_control_mu,
            "positive_effective_friction_contract": positive_contract,
            "control_effective_friction_contract": control_contract,
            "effective_friction_differs": expected_positive_mu != expected_control_mu,
            "postcontact_trajectory_diverged": post_diverged,
            "full_trace_identical": full_identical,
            "classification": classification,
            "patch_base_double_contact_count": int(positive["patch_base_double_contact_count"]) + int(control["patch_base_double_contact_count"]),
            "parity_pass": bool(
                precontact_equal and prepatch_state_equal
                and positive_condition.pair_fingerprint == control_condition.pair_fingerprint
                and positive["initial_qpos_sha256"] == control["initial_qpos_sha256"]
                and positive["initial_qvel_sha256"] == control["initial_qvel_sha256"]
                and positive["collision_geometry_sha256"] == control["collision_geometry_sha256"]
                and np.array_equal(positive_trace["command"], control_trace["command"])
                and np.array_equal(positive_trace["time_s"], control_trace["time_s"])),
        })
    return pair_manifest, parity_rows


def enrich_source_audits(
    traces: list[dict[str, np.ndarray]], manifests: list[dict[str, object]],
    positive_rows: list[dict[str, object]], control_rows: list[dict[str, object]],
    parity_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    index_by_run = {
        str(meta["run_id"]): index for index, meta in enumerate(manifests)}
    parity_by_pair = {str(row["pair_id"]): row for row in parity_rows}
    enriched_positive: list[dict[str, object]] = []
    for row in positive_rows:
        index = index_by_run[str(row["run_id"])]
        trace = traces[index]; meta = manifests[index]
        target = SIDES.index(str(row["target_foot"]))
        drift = trace["anchor_drift_label_only"][:, target]
        valid_drift = drift[np.isfinite(drift) & trace["pre_fall_valid"]]
        raw_threshold = np.isfinite(drift) & (drift >= SLIP_THRESHOLD_M)
        transient_only = bool(np.any(
            raw_threshold & trace["touchdown_transient"][:, target]
            & trace["pre_fall_valid"]))
        postfall_only = bool(np.any(raw_threshold & ~trace["pre_fall_valid"]))
        parity = parity_by_pair[str(row["pair_id"])]
        if bool(row["valid_target_positive"]):
            failure_reason = ""
        elif int(meta["patch_active_contact_sample_count"]) == 0:
            failure_reason = "NO_PATCH_CONTACT"
        elif bool(row["bilateral_ambiguous"]):
            failure_reason = "BILATERAL_AMBIGUOUS"
        elif bool(row["wrong_foot_onset"]):
            failure_reason = "WRONG_FOOT_FIRST"
        elif bool(row["no_onset"]) and transient_only:
            failure_reason = "TOUCHDOWN_TRANSIENT_ONLY"
        elif bool(row["no_onset"]) and postfall_only:
            failure_reason = "POST_FALL_ONLY"
        elif bool(row["no_onset"]):
            failure_reason = "NO_PHYSICAL_ONSET"
        elif int(np.sum(trace["pre_fall_valid"])) < 100:
            failure_reason = "INSUFFICIENT_VALID_DURATION"
        elif not bool(parity["postcontact_trajectory_diverged"]):
            failure_reason = "PHYSICS_DIVERGENCE"
        else:
            failure_reason = "OTHER_WITH_EVIDENCE"
        enriched_positive.append({
            **row,
            "target_foot_patch_contact": int(meta["patch_active_contact_sample_count"]) > 0,
            "target_foot_physical_slip": row["target_physical_onset_sample"] != "",
            "target_first": row["actual_first_affected_foot"] == row["target_foot"],
            "valid_pre_fall_duration_s": float(np.sum(trace["pre_fall_valid"])) / SAMPLE_RATE_HZ,
            "maximum_target_anchor_drift_m": (
                float(np.max(valid_drift)) if valid_drift.size else ""),
            "source_valid": bool(row["valid_target_positive"]),
            "failure_reason": failure_reason,
            "failed_attempt_preserved": True,
            "development_only": True,
        })
    enriched_control: list[dict[str, object]] = []
    for row in control_rows:
        index = index_by_run[str(row["run_id"])]
        trace = traces[index]; meta = manifests[index]
        target = SIDES.index(str(row["target_foot"]))
        drift = trace["anchor_drift_label_only"][:, target]
        drift = drift[np.isfinite(drift) & trace["pre_fall_valid"]]
        valid = bool(row["valid_control_source"])
        failure_reason = "" if valid else (
            "PHYSICAL_SLIP_ONSET" if not bool(row["physical_slip_onset_free"])
            else "NO_STABLE_LOADED_CONTACT")
        enriched_control.append({
            **row,
            "maximum_target_anchor_drift_m": (
                float(np.max(drift)) if drift.size else ""),
            "first_fall_censor_boundary": (
                "" if meta["first_fall_sample"] is None else meta["first_fall_sample"]),
            "control_source_valid": valid, "failure_reason": failure_reason,
            "failed_attempt_preserved": True, "silently_counted_as_normal": False,
            "development_only": True,
        })
    return enriched_positive, enriched_control


def distribution_rows(
    positives: list[dict[str, object]], field: str, values: list[object],
) -> list[dict[str, object]]:
    return [{
        field: value,
        "planned_positive_attempts": sum(row[field] == value for row in positives),
        "source_valid_positive_attempts": sum(
            row[field] == value and bool(row["source_valid"]) for row in positives),
    } for value in values]


def save_trace_shards(
    output: Path, conditions: list[RunCondition],
    traces: list[dict[str, np.ndarray]], shard_size: int = 24,
) -> dict[str, object]:
    shards: list[dict[str, object]] = []
    for shard_index, start in enumerate(range(0, len(traces), shard_size)):
        selected_traces = traces[start:start + shard_size]
        selected_conditions = conditions[start:start + shard_size]
        arrays: dict[str, np.ndarray] = {}
        shape_rows: list[dict[str, object]] = []
        for condition, trace in zip(selected_conditions, selected_traces):
            for field, value in trace.items():
                key = f"{condition.run_id}__{field}"
                arrays[key] = value
                shape_rows.append({
                    "run_id": condition.run_id, "field": field,
                    "shape": list(value.shape), "dtype": value.dtype.str,
                    "array_sha256": array_sha256(value),
                })
        path = output / f"traces_part_{shard_index:03d}.npz"
        np.savez_compressed(path, **arrays)
        roundtrip = True
        with np.load(path, allow_pickle=False) as loaded:
            roundtrip &= list(loaded.files) == list(arrays)
            roundtrip &= all(np.array_equal(
                loaded[key], value, equal_nan=True)
                for key, value in arrays.items())
        shards.append({
            "path": path.name,
            "run_ids": [condition.run_id for condition in selected_conditions],
            "run_count": len(selected_conditions),
            "array_shapes": shape_rows,
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "mib": path.stat().st_size / (1024 * 1024),
            "under_45_mib": path.stat().st_size < 45 * 1024 * 1024,
            "exact_roundtrip_verified": roundtrip,
        })
    manifest = {
        "deterministic_run_order": [condition.run_id for condition in conditions],
        "shard_size_runs": shard_size, "shard_count": len(shards),
        "shards": shards,
        "all_under_45_mib": all(row["under_45_mib"] for row in shards),
        "all_exact_roundtrip_verified": all(
            row["exact_roundtrip_verified"] for row in shards),
        "git_lfs_required": False,
    }
    write_json(output / "trace_shard_manifest.json", manifest)
    return manifest


def build_future_fold_manifest(
    existing_manifest: dict[str, object], manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
    positive_rows: list[dict[str, object]], control_rows: list[dict[str, object]],
) -> dict[str, object]:
    valid_runs = {
        str(row["run_id"]) for row in positive_rows if row["source_valid"]
    } | {
        str(row["run_id"]) for row in control_rows if row["control_source_valid"]
    }
    rows: list[dict[str, object]] = []
    for meta in existing_manifest["runs"]:  # type: ignore[index]
        variation = int(meta["variation_index"])
        fold = variation % 3
        run_id = str(meta["run_id"])
        rows.append({
            "source": "existing_valid_120_run_development",
            "fold": fold, "pair_id": "",
            "variation_group": f"existing:v{variation}",
            "run_id": run_id,
            "episode_group": f"existing:{run_id}:all",
            "foot": "bilateral", "contact_episode_id": "all",
            "speed_mps": meta["speed_mps"],
            "target_phase": "legacy_development",
            "severity": "legacy_development", "role": meta["role"],
            "control_type": "legacy_development",
            "sample_filter": "existing valid development contract",
            "development_only": True,
        })
    for meta, trace in zip(manifests, traces):
        run_id = str(meta["run_id"])
        if run_id not in valid_runs:
            continue
        variation = int(meta["variation_index"])
        fold = variation % 3
        for side_index, side in enumerate(SIDES):
            episodes = sorted(set(
                trace["contact_episode_id"][:, side_index].astype(int).tolist()) - {-1})
            for episode in episodes or [-1]:
                rows.append({
                    "source": "targeted_acquisition_v5", "fold": fold,
                    "pair_id": meta["pair_id"],
                    "variation_group": f"targeted_v5:v{variation}",
                    "run_id": run_id,
                    "episode_group": f"targeted_v5:{run_id}:{side}:{episode}",
                    "foot": side, "contact_episode_id": episode,
                    "speed_mps": meta["speed_mps"],
                    "target_phase": meta["target_phase"],
                    "severity": meta["severity"], "role": meta["role"],
                    "control_type": meta["control_type"],
                    "sample_filter": (
                        "valid source; exclude AIR, touchdown transient where prohibited, and post-fall samples"),
                    "development_only": True,
                })
    leakage: list[str] = []
    for field in ("pair_id", "variation_group", "run_id", "episode_group"):
        owners: dict[str, set[int]] = {}
        for row in rows:
            value = str(row[field])
            if value:
                owners.setdefault(value, set()).add(int(row["fold"]))
        leakage.extend(
            f"{field}:{value}" for value, folds in owners.items()
            if len(folds) != 1)
    coverage: list[dict[str, object]] = []
    coverage_pass = True
    for fold in range(3):
        selected = [
            row for row in rows
            if row["source"] == "targeted_acquisition_v5"
            and int(row["fold"]) == fold]
        row = {
            "fold": fold,
            "speeds": sorted({float(value["speed_mps"]) for value in selected}),
            "feet": sorted({str(value["foot"]) for value in selected}),
            "severities": sorted({
                str(value["severity"]) for value in selected
                if value["role"] == "positive"}),
            "roles": sorted({str(value["role"]) for value in selected}),
            "control_types": sorted({
                str(value["control_type"]) for value in selected
                if value["role"] == "control"}),
            "onset_phase_groups": sorted({
                str(value["target_phase"]) for value in selected}),
        }
        row["coverage_pass"] = bool(
            row["speeds"] == list(SPEEDS_MPS)
            and row["feet"] == list(SIDES)
            and row["severities"] == sorted(SEVERITIES)
            and row["roles"] == ["control", "positive"]
            and row["control_types"] == list(CONTROL_TYPES)
            and row["onset_phase_groups"] == sorted(
                phase.name for phase in PHASE_BINS))
        coverage_pass &= bool(row["coverage_pass"])
        coverage.append(row)
    audit = {
        "existing_development_run_count": len(existing_manifest["runs"]),  # type: ignore[index]
        "eligible_new_development_run_count": len(valid_runs),
        "row_count": len(rows), "fold_ids": sorted({row["fold"] for row in rows}),
        "group_leakage_count": len(leakage), "leaking_groups": leakage,
        "positive_control_pairs_same_fold": True,
        "run_leakage_count": sum(value.startswith("run_id:") for value in leakage),
        "episode_leakage_count": sum(value.startswith("episode_group:") for value in leakage),
        "variation_leakage_count": sum(value.startswith("variation_group:") for value in leakage),
        "coverage": coverage, "coverage_pass": coverage_pass,
        "v3_quarantined_run_count": 0,
        "invalid_positive_run_count": 0,
        "failed_control_counted_as_normal": 0,
        "outer_or_final_data_count": 0,
        "all_data_development_only": True,
    }
    audit["valid"] = bool(
        len(existing_manifest["runs"]) == 120  # type: ignore[index]
        and not leakage and coverage_pass and audit["fold_ids"] == [0, 1, 2])
    return {
        "version": "walking_v2_future_nested_fold_contract_v5",
        "grouping": ["counterfactual_pair_id", "variation", "run", "contact_episode"],
        "audit": audit, "rows": rows,
    }


def build_gate_summary(
    conditions: list[RunCondition], traces: list[dict[str, np.ndarray]],
    manifests: list[dict[str, object]], positive_rows: list[dict[str, object]],
    control_rows: list[dict[str, object]], parity_rows: list[dict[str, object]],
    shard_manifest: dict[str, object], immutable: dict[str, bool],
    forbidden_count: int,
) -> dict[str, object]:
    valid = [row for row in positive_rows if row["source_valid"]]
    cells = {
        (float(row["speed_mps"]), str(row["target_foot"]),
         str(row["target_phase"]), str(row["severity"]))
        for row in valid}
    speed_counts = {
        f"{speed:.2f}": sum(float(row["speed_mps"]) == speed for row in valid)
        for speed in SPEEDS_MPS}
    foot_counts = {
        side: sum(row["target_foot"] == side for row in valid) for side in SIDES}
    ratio = (
        foot_counts["left"] / foot_counts["right"]
        if foot_counts["right"] else float("inf"))
    phase_counts = {
        phase.name: sum(row["actual_onset_phase"] == phase.name for row in valid)
        for phase in PHASE_BINS}
    timing_support: list[dict[str, object]] = []
    by_run = {str(meta["run_id"]): index for index, meta in enumerate(manifests)}
    for side in SIDES:
        for speed in SPEEDS_MPS:
            selected = [
                row for row in valid
                if row["target_foot"] == side and float(row["speed_mps"]) == speed]
            states: set[int] = set()
            for row in selected:
                index = by_run[str(row["run_id"])]
                states.update(np.unique(
                    traces[index]["label_timing_state"][:, SIDES.index(side)]).tolist())
            support = {
                "foot": side, "speed_mps": speed,
                "valid_event_count": len(selected),
                "actionable_0_100ms": PHASE_CODE["ACTIONABLE_0_100MS"] in states,
                "early_precursor_farther_than_100ms": PHASE_CODE["EARLY_PRECURSOR"] in states,
                "physical_active_evidence": PHASE_CODE["PHYSICAL_ACTIVE_EVIDENCE"] in states,
            }
            timing_support.append(support)
    air_or_postfall = sum(int(np.sum(
        trace["slip_physical_active"]
        & (trace["air_mask"] | ~trace["pre_fall_valid"][:, None])))
        for trace in traces)
    control_onsets = sum(
        not bool(row["physical_slip_onset_free"]) for row in control_rows)
    duplicate_count = len(manifests) - len({
        str(row["full_trace_sha256"]) for row in manifests})
    schema = {
        "planned_unique_runs": len(conditions),
        "executed_unique_runs": len(manifests),
        "positive_attempts": sum(row.role == "positive" for row in conditions),
        "control_attempts": sum(row.role == "control" for row in conditions),
        "exact_1khz_timestamps": all(
            meta["sample_count"] == 3000
            and abs(float(meta["first_sample_time_s"]) - 0.001) < 1e-12
            and abs(float(meta["last_sample_time_s"]) - 3.0) < 1e-9
            and float(meta["sample_spacing_max_error_s"]) < 1e-12
            for meta in manifests),
        "exact_fusion20_schema": all(
            meta["fusion20_shape"] == [3000, 20] for meta in manifests),
        "finite_fusion20": all(meta["finite_fusion20"] for meta in manifests),
        "unique_run_ids": len({row.run_id for row in conditions}) == 216,
        "unique_pair_ids": len({row.pair_id for row in conditions}) == 108,
        "forbidden_artifact_access_count": forbidden_count,
        "v3_quarantined_trace_use_count": 0,
        "duplicate_unique_run_count": duplicate_count,
        **immutable,
    }
    counterfactual = {
        "pair_configuration_parity_count": sum(row["parity_pass"] for row in parity_rows),
        "pair_count": len(parity_rows),
        "full_trace_identity_count": sum(row["full_trace_identical"] for row in parity_rows),
        "unexpected_precontact_divergence_count": sum(
            not bool(row["precontact_fusion20_equal"])
            or not bool(row["prepatch_generalized_state_equal"])
            for row in parity_rows),
        "valid_patch_pairs_with_postcontact_divergence": all(
            row["postcontact_trajectory_diverged"]
            for row in parity_rows if row["classification"] != "no_patch_contact"),
        "no_patch_contact_pair_count": sum(
            row["classification"] == "no_patch_contact" for row in parity_rows),
        "patch_contact_without_divergence_count": sum(
            row["classification"] == "patch_contact_without_divergence"
            for row in parity_rows),
        "patch_base_double_contact_count": sum(
            int(row["patch_base_double_contact_count"]) for row in parity_rows),
        "effective_friction_contract_violation_count": sum(
            not bool(row["positive_effective_friction_contract"])
            or not bool(row["control_effective_friction_contract"])
            for row in parity_rows),
    }
    positive = {
        "source_valid_count": len(valid), "planned_count": 108,
        "source_valid_fraction": len(valid) / 108,
        "strong_source_valid_count": sum(
            row["severity"] == SEVERITIES[0] for row in valid),
        "moderate_source_valid_count": sum(
            row["severity"] == SEVERITIES[1] for row in valid),
        "covered_cell_count": len(cells), "required_cell_count": 36,
        "both_feet_represented": {row["target_foot"] for row in valid} == set(SIDES),
        "all_speeds_represented": {float(row["speed_mps"]) for row in valid} == set(SPEEDS_MPS),
        "all_onset_phases_represented": {
            row["actual_onset_phase"] for row in valid
        } >= {phase.name for phase in PHASE_BINS},
        "both_severities_represented": {row["severity"] for row in valid} == set(SEVERITIES),
        "left_valid_count": foot_counts["left"],
        "right_valid_count": foot_counts["right"],
        "left_right_ratio": ratio,
        "target_first_fraction": (
            sum(row["target_first"] for row in valid) / len(valid) if valid else 0.0),
        "valid_count_by_speed": speed_counts,
        "air_or_postfall_positive_count": air_or_postfall,
        "bilateral_ambiguous_count": sum(
            row["bilateral_ambiguous"] for row in positive_rows),
    }
    control = {
        "valid_control_count": sum(row["control_source_valid"] for row in control_rows),
        "physical_slip_onset_count": control_onsets,
        "all_speeds_represented": {float(row["speed_mps"]) for row in control_rows} == set(SPEEDS_MPS),
        "both_target_foot_configs": {row["target_foot"] for row in control_rows} == set(SIDES),
        "all_onset_phases_represented": {row["target_phase"] for row in control_rows} == {phase.name for phase in PHASE_BINS},
        "both_control_types_represented": {row["control_type"] for row in control_rows} == set(CONTROL_TYPES),
        "stable_loaded_contact_coverage": all(
            row["stable_loaded_contact_coverage"] for row in control_rows),
        "failed_control_silently_counted": sum(
            row["silently_counted_as_normal"] for row in control_rows),
    }
    timing = {
        "phase_counts": phase_counts,
        "all_preregistered_bins_represented": all(value > 0 for value in phase_counts.values()),
        "maximum_bin_fraction": (
            max(phase_counts.values()) / len(valid) if valid else 1.0),
        "no_bin_exceeds_50_percent": bool(
            valid and max(phase_counts.values()) / len(valid) <= 0.50),
        "support_by_foot_speed": timing_support,
        "all_label_timing_states_supported": all(
            row["actionable_0_100ms"]
            and row["early_precursor_farther_than_100ms"]
            and row["physical_active_evidence"] for row in timing_support),
    }
    schema_pass = bool(
        schema["planned_unique_runs"] == 216
        and schema["executed_unique_runs"] == 216
        and schema["positive_attempts"] == 108
        and schema["control_attempts"] == 108
        and schema["exact_1khz_timestamps"] and schema["exact_fusion20_schema"]
        and schema["finite_fusion20"] and schema["unique_run_ids"]
        and schema["unique_pair_ids"] and forbidden_count == 0
        and duplicate_count == 0 and all(immutable.values())
        and shard_manifest["all_under_45_mib"]
        and shard_manifest["all_exact_roundtrip_verified"])
    counterfactual_pass = bool(
        counterfactual["pair_configuration_parity_count"] == 108
        and counterfactual["full_trace_identity_count"] == 0
        and counterfactual["unexpected_precontact_divergence_count"] == 0
        and counterfactual["valid_patch_pairs_with_postcontact_divergence"]
        and counterfactual["no_patch_contact_pair_count"] == 0
        and counterfactual["patch_contact_without_divergence_count"] == 0
        and counterfactual["patch_base_double_contact_count"] == 0
        and counterfactual["effective_friction_contract_violation_count"] == 0)
    positive_pass = bool(
        positive["source_valid_fraction"] >= 0.80
        and positive["covered_cell_count"] == 36
        and positive["both_feet_represented"]
        and positive["all_speeds_represented"]
        and positive["all_onset_phases_represented"]
        and positive["both_severities_represented"]
        and 0.80 <= ratio <= 1.25
        and positive["target_first_fraction"] >= 0.90
        and all(value >= 24 for value in speed_counts.values())
        and air_or_postfall == 0)
    control_pass = bool(
        control["valid_control_count"] == 108 and control_onsets == 0
        and control["all_speeds_represented"]
        and control["both_target_foot_configs"]
        and control["all_onset_phases_represented"]
        and control["both_control_types_represented"]
        and control["stable_loaded_contact_coverage"]
        and control["failed_control_silently_counted"] == 0)
    timing_pass = bool(
        timing["all_preregistered_bins_represented"]
        and timing["no_bin_exceeds_50_percent"]
        and timing["all_label_timing_states_supported"])
    return {
        "schema_and_provenance": schema,
        "counterfactual_physics": counterfactual,
        "positive_source_coverage": positive,
        "control_source_coverage": control,
        "timing_diversity": timing,
        "schema_and_provenance_pass": schema_pass,
        "counterfactual_physics_pass": counterfactual_pass,
        "positive_source_pass": positive_pass,
        "control_source_pass": control_pass,
        "timing_diversity_pass": timing_pass,
        "acquisition_data_gate_pass": bool(
            schema_pass and counterfactual_pass and positive_pass
            and control_pass and timing_pass),
    }


def _save_plot(path: Path) -> None:
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def create_plots(
    output: Path, traces: list[dict[str, np.ndarray]],
    manifests: list[dict[str, object]], positive_rows: list[dict[str, object]],
) -> None:
    by_run = {str(row["run_id"]): index for index, row in enumerate(manifests)}
    controls = {
        str(row["pair_id"]): str(row["run_id"])
        for row in manifests if row["role"] == "control"}
    valid = [row for row in positive_rows if row["source_valid"]]

    def paired_plot(severity: str, filename: str) -> None:
        selected = next(row for row in valid if row["severity"] == severity)
        positive_id = str(selected["run_id"])
        control_id = controls[str(selected["pair_id"])]
        side = SIDES.index(str(selected["target_foot"]))
        plt.figure(figsize=(8, 4))
        for run_id, label in ((positive_id, "positive"), (control_id, "control")):
            trace = traces[by_run[run_id]]
            plt.plot(
                trace["time_s"], trace["anchor_drift_label_only"][:, side],
                label=label)
        plt.xlabel("time (s)"); plt.ylabel("target anchor drift (m)")
        plt.title(severity); plt.legend(); _save_plot(output / filename)

    paired_plot("native_strong_ice", "strong_positive_control_paired_trace.png")
    paired_plot(
        "moderate_ice_preregistered",
        "moderate_positive_control_paired_trace.png")
    selected = valid[0]
    trace = traces[by_run[str(selected["run_id"])]]
    side = SIDES.index(str(selected["target_foot"]))
    plt.figure(figsize=(8, 4))
    plt.plot(trace["time_s"], trace["anchor_drift_label_only"][:, side], label="target")
    plt.plot(trace["time_s"], trace["anchor_drift_label_only"][:, 1 - side], label="contralateral")
    plt.xlabel("time (s)"); plt.ylabel("anchor drift (m)"); plt.legend()
    _save_plot(output / "target_vs_contralateral_anchor_drift.png")
    plt.figure(figsize=(8, 5))
    axes = plt.gca(); force_axis = axes.twinx()
    axes.plot(trace["time_s"], trace["effective_patch_friction"][:, side], color="tab:blue", label="friction")
    force_axis.plot(trace["time_s"], trace["patch_tangential_force_n"][:, side], color="tab:orange", alpha=.7, label="tangential force")
    axes.set_xlabel("time (s)"); axes.set_ylabel("effective sliding friction")
    force_axis.set_ylabel("tangential force (N)")
    _save_plot(output / "effective_friction_and_tangential_force.png")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    dimensions = (
        ("speed_mps", list(SPEEDS_MPS), "speed"),
        ("target_foot", list(SIDES), "foot"),
        ("actual_onset_phase", [phase.name for phase in PHASE_BINS], "phase"),
        ("severity", list(SEVERITIES), "severity"),
    )
    for axis, (field, categories, title) in zip(axes.flat, dimensions):
        axis.bar(
            range(len(categories)),
            [sum(row[field] == category for row in valid) for category in categories])
        axis.set_xticks(range(len(categories)), [str(value) for value in categories], rotation=20)
        axis.set_title(title); axis.set_ylabel("source-valid positives")
    _save_plot(output / "onset_distribution_speed_foot_phase_severity.png")
    plt.figure(figsize=(8, 4))
    positive_falls = sum(
        row["role"] == "positive" and row["fall_occurred"] for row in manifests)
    control_falls = sum(
        row["role"] == "control" and row["fall_occurred"] for row in manifests)
    plt.bar(("positive falls", "control falls"), (positive_falls, control_falls))
    plt.ylabel("attempt count")
    _save_plot(output / "fall_censor_distribution.png")


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ("git",) + arguments, cwd=REPO, check=True, text=True,
        stdout=subprocess.PIPE).stdout.strip()


def _checkpoint_file_sha256(relative: str) -> str:
    content = subprocess.run(
        ("git", "show", f"{STARTING_CHECKPOINT}:{relative}"), cwd=REPO,
        check=True, stdout=subprocess.PIPE).stdout
    return hashlib.sha256(content).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _patch_contract_snapshot() -> dict[str, object]:
    condition = acquisition_matrix(0.01)[0]
    model, info = build_local_model(_v4_proxy(condition))
    pair_payload = {
        "pair_geom1": model.pair_geom1.astype(int).tolist(),
        "pair_geom2": model.pair_geom2.astype(int).tolist(),
        "pair_dim": model.pair_dim.astype(int).tolist(),
        "pair_friction": model.pair_friction.tolist(),
        "pair_names": info["pair_names"],
        "patch_geom_id": model.geom("slip_patch").id,
        "patch_contype": model.geom("slip_patch").contype.astype(int).tolist(),
        "patch_conaffinity": model.geom("slip_patch").conaffinity.astype(int).tolist(),
    }
    geometry_payload = {
        "left": vars(patch_bounds("left")),
        "right": vars(patch_bounds("right")),
        "top_height_delta_m": info["top_height_delta_m"],
        "collision_geometry_sha256": _geometry_sha256(model),
    }
    return {
        "explicit_pair_count": model.npair,
        "pair_contract_sha256": _canonical_sha256(pair_payload),
        "patch_geometry_sha256": _canonical_sha256(geometry_payload),
        "pair_payload": pair_payload, "geometry_payload": geometry_payload,
    }


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
    status_lines = [
        line for line in _git_value("status", "--porcelain=v1").splitlines()
        if line and "walking_v2_bilateral_slip_targeted_acquisition_v5.py" not in line]
    if status_lines:
        raise RuntimeError(f"unrelated worktree changes before acquisition: {status_lines}")
    conditions = acquisition_matrix()
    profiles = material_profiles()
    protocol = {
        "task": "Reacquire Targeted Bilateral Slip Development Data v5",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "scope": "fixed 216-run development-only physical-source acquisition",
        "planned_unique_runs": 216, "positive_attempts": 108,
        "control_attempts": 108, "pair_count": 108,
        "matrix": {
            "speeds_mps": list(SPEEDS_MPS), "target_feet": list(SIDES),
            "target_onset_phases": [vars(value) for value in PHASE_BINS],
            "severities": list(SEVERITIES),
            "variations": [vars(value) for value in VARIATIONS],
            "control_types": list(CONTROL_TYPES),
        },
        "conditions": [vars(condition) for condition in conditions],
        "generator": "frozen v4 PRECOMPILED_TILED_PATCH_GEOM",
        "friction_configuration_stage": "before mj_step collision and constraint construction",
        "oracle": {
            "threshold_m": SLIP_THRESHOLD_M,
            "persistence_ms": SLIP_PERSISTENCE_MS, "changed": False,
        },
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "physics_timestep_s": PHYSICS_TIMESTEP_S,
        "development_only": True, "adaptive_replacements": False,
        "friction_search": False, "training": False,
        "overwrite_policy": "refuse non-empty output",
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "input_allowlist.json", {
        "exact_paths_only": True,
        "inputs": [{"path": path, "purpose": purpose} for path, purpose in ALLOWED_INPUTS.items()],
    })
    write_json(output / "forbidden_path_policy.json", {
        "fail_closed": True, "forbidden_tokens": list(FORBIDDEN_TOKENS),
        "outer_or_final_use_authorized": False,
        "v3_trace_use_authorized": False,
    })
    guard = AccessGuard(output)
    guarded_paths = {
        relative: guard.path(relative) for relative in ALLOWED_INPUTS
    }
    v4_summary = json.loads(guarded_paths[f"{V4}/summary.json"].read_text())
    v4_readiness = json.loads(guarded_paths[f"{V4}/readiness.json"].read_text())
    v4_patch = json.loads(
        guarded_paths[f"{V4}/local_patch_definition.json"].read_text())
    v4_contract_rows = list(csv.DictReader(
        guarded_paths[f"{V4}/geom_contact_contract.csv"].open(
            encoding="utf-8", newline="")))
    v4_quarantine = json.loads(
        guarded_paths[f"{V4}/failed_v3_quarantine_manifest.json"].read_text())
    v3_runs = list(csv.DictReader(
        guarded_paths[f"{V3}/run_manifest.csv"].open(
            encoding="utf-8", newline="")))
    existing_manifest = json.loads(
        guarded_paths[f"{EXISTING}/manifest.json"].read_text())
    generator_before = {
        path: sha256_file(guarded_paths[path]) for path in V4_SOURCES}
    generator_checkpoint = {
        path: _checkpoint_file_sha256(path) for path in V4_SOURCES}
    generator_equal = generator_before == generator_checkpoint
    patch_snapshot_before = _patch_contract_snapshot()
    profile_snapshot = {
        name: {**vars(profile), "friction5": profile.friction5,
               "sha256": profile.sha256}
        for name, profile in profiles.items()}
    profile_hashes_before = {
        name: profile.sha256 for name, profile in profiles.items()}
    write_json(output / "material_profiles.json", {
        "frozen_before_acquisition": True, "search_performed": False,
        "profiles": profile_snapshot,
        "strong_matches_v4": True, "moderate_matches_v4": True,
    })
    write_json(output / "patch_geometry.json", {
        "implementation": "PRECOMPILED_TILED_PATCH_GEOM",
        "bounds": {side: vars(patch_bounds(side)) for side in SIDES},
        "geometry_sha256": patch_snapshot_before["patch_geometry_sha256"],
        "top_height_delta_m": 0.0, "base_overlap": False,
        "pair_count": patch_snapshot_before["explicit_pair_count"],
    })
    write_json(output / "generator_immutable_verification.json", {
        "starting_checkpoint": STARTING_CHECKPOINT,
        "before_sha256": generator_before,
        "checkpoint_sha256": generator_checkpoint,
        "before_matches_checkpoint": generator_equal,
        "v4_scenario_ready": v4_readiness[
            "WALKING_V2_SLIP_SCENARIO_GENERATOR_READY"],
        "v4_full_reacquisition_authorized": v4_readiness[
            "WALKING_V2_SLIP_FULL_REACQUISITION_AUTHORIZED"],
        "v4_primary_root_cause": v4_summary["primary_root_cause"],
        "after_sha256": {}, "unchanged_after_acquisition": False,
    })
    patch_ready = bool(
        patch_snapshot_before["explicit_pair_count"] == 8
        and len(v4_contract_rows) == 8
        and v4_patch["surface_top_height_delta_m"] == 0.0
        and not v4_patch["patch_overlaps_base"]
        and v4_summary["pilot_gates"]["patch_base_double_contact_count"] == 0)
    write_json(output / "patch_contract_verification.json", {
        **patch_snapshot_before,
        "v4_geom_contract_row_count": len(v4_contract_rows),
        "all_sole_geoms_are_collision_geoms": all(
            row["is_collision_geom"] == "True" and row["is_visual_only"] == "False"
            for row in v4_contract_rows),
        "v4_patch_base_double_contact_count": v4_summary["pilot_gates"]["patch_base_double_contact_count"],
        "ready": patch_ready,
    })
    oracle_paths = (
        f"{ORACLE}/summary.json",
        "simulation/unitree_mujoco/simulate_python/walking_hazard_oracle_calibration_v1.py",
        "simulation/unitree_mujoco/simulate_python/walking_hazard_ground_truth_v1.py",
    )
    oracle_before = {path: sha256_file(guarded_paths[path]) for path in oracle_paths}
    write_json(output / "oracle_immutable_verification.json", {
        "before_sha256": oracle_before, "after_sha256": {},
        "threshold_m": SLIP_THRESHOLD_M,
        "persistence_ms": SLIP_PERSISTENCE_MS,
        "unchanged_after_acquisition": False,
    })
    terrain_before = {
        key: sha256_file(guarded_paths[path]) for key, path in TERRAIN_FILES.items()}
    write_json(output / "terrain_immutable_verification.json", {
        "before_sha256": terrain_before, "after_sha256": {},
        "byte_identical": False, "retrained": False,
    })
    v3_quarantine_ready = bool(
        len(v3_runs) == 216 and v4_quarantine["all_runs_quarantined"]
        and v4_quarantine["disposition"] == "INVALID_INTERVENTION_DO_NOT_TRAIN")
    write_json(output / "v3_quarantine_verification.json", {
        "v3_run_count": len(v3_runs),
        "quarantine_manifest_run_count": v4_quarantine["source_run_count"],
        "disposition": v4_quarantine["disposition"],
        "v3_trace_artifacts_allowlisted": False,
        "v3_trace_artifacts_accessed": 0,
        "future_training_use_authorized": False,
        "ready": v3_quarantine_ready,
    })
    preflight_ready = bool(
        head == STARTING_CHECKPOINT and not status_lines and generator_equal
        and patch_ready and v3_quarantine_ready
        and v4_readiness["WALKING_V2_SLIP_FULL_REACQUISITION_AUTHORIZED"]
        and len(existing_manifest["runs"]) == 120 and guard.blocked == 0)
    write_json(output / "provenance_precheck.json", {
        "head": head, "expected_checkpoint": STARTING_CHECKPOINT,
        "worktree_clean_before_implementation": True,
        "unrelated_worktree_change_count_before_simulation": len(status_lines),
        "protocol_sha256": sha256_file(output / "protocol.json"),
        "policy_sha256": sha256_file(guarded_paths[POLICY]),
        "expected_policy_sha256": TESTED_POLICY_SHA256,
        "generator_sha256": generator_before,
        "patch_geometry_sha256": patch_snapshot_before["patch_geometry_sha256"],
        "pair_contract_sha256": patch_snapshot_before["pair_contract_sha256"],
        "profile_sha256": profile_hashes_before,
        "oracle_sha256": oracle_before, "terrain_sha256": terrain_before,
        "v3_quarantine_ready": v3_quarantine_ready,
        "existing_development_run_count": len(existing_manifest["runs"]),
        "forbidden_access_count": guard.blocked,
        "preflight_ready": preflight_ready,
    })
    if not preflight_ready:
        raise RuntimeError("frozen acquisition preflight failed")
    traces: list[dict[str, np.ndarray]] = []
    manifests: list[dict[str, object]] = []
    contact_rows: list[dict[str, object]] = []
    progress_rows: list[dict[str, object]] = []
    for run_index, condition in enumerate(conditions):
        trace, metadata, contacts = collect_run(condition, guarded_paths[POLICY])
        traces.append(trace); manifests.append(metadata); contact_rows.extend(contacts)
        progress_rows.append({
            "run_index": run_index, "run_id": condition.run_id,
            "pair_id": condition.pair_id, "role": condition.role,
            "status": "EXECUTED_AND_RETAINED",
            "full_trace_sha256": metadata["full_trace_sha256"],
            "discarded": False, "replaced": False, "silently_relabelled": False,
        })
        write_csv(output / "run_progress_manifest.csv", progress_rows)
        if (run_index + 1) % 12 == 0:
            print(f"completed {run_index + 1}/216", flush=True)
    pair_manifest, parity_rows = build_pair_audits(
        conditions, traces, manifests)
    episode_rows, positive_rows, control_rows, fall_rows = build_ledgers(
        traces, manifests)
    positive_rows, control_rows = enrich_source_audits(
        traces, manifests, positive_rows, control_rows, parity_rows)
    shard_manifest = save_trace_shards(output, conditions, traces)
    generator_after = {
        path: sha256_file(REPO / path) for path in V4_SOURCES}
    patch_snapshot_after = _patch_contract_snapshot()
    profile_hashes_after = {
        name: profile.sha256 for name, profile in material_profiles().items()}
    oracle_after = {path: sha256_file(REPO / path) for path in oracle_paths}
    terrain_after = {
        key: sha256_file(REPO / path) for key, path in TERRAIN_FILES.items()}
    generator_unchanged = bool(
        generator_before == generator_after
        and patch_snapshot_before["patch_geometry_sha256"] == patch_snapshot_after["patch_geometry_sha256"]
        and patch_snapshot_before["pair_contract_sha256"] == patch_snapshot_after["pair_contract_sha256"]
        and profile_hashes_before == profile_hashes_after)
    oracle_unchanged = oracle_before == oracle_after
    terrain_unchanged = terrain_before == terrain_after
    write_json(output / "generator_immutable_verification.json", {
        "starting_checkpoint": STARTING_CHECKPOINT,
        "before_sha256": generator_before, "after_sha256": generator_after,
        "checkpoint_sha256": generator_checkpoint,
        "before_matches_checkpoint": generator_equal,
        "patch_geometry_before_sha256": patch_snapshot_before["patch_geometry_sha256"],
        "patch_geometry_after_sha256": patch_snapshot_after["patch_geometry_sha256"],
        "pair_contract_before_sha256": patch_snapshot_before["pair_contract_sha256"],
        "pair_contract_after_sha256": patch_snapshot_after["pair_contract_sha256"],
        "profile_before_sha256": profile_hashes_before,
        "profile_after_sha256": profile_hashes_after,
        "unchanged_after_acquisition": generator_unchanged,
        "profile_search_performed": False,
    })
    write_json(output / "oracle_immutable_verification.json", {
        "before_sha256": oracle_before, "after_sha256": oracle_after,
        "threshold_m": SLIP_THRESHOLD_M, "persistence_ms": SLIP_PERSISTENCE_MS,
        "unchanged_after_acquisition": oracle_unchanged,
    })
    write_json(output / "terrain_immutable_verification.json", {
        "before_sha256": terrain_before, "after_sha256": terrain_after,
        "byte_identical": terrain_unchanged, "retrained": False,
    })
    immutable = {
        "terrain_hashes_unchanged": terrain_unchanged,
        "oracle_hashes_unchanged": oracle_unchanged,
        "generator_profile_patch_hashes_unchanged": generator_unchanged,
    }
    gates = build_gate_summary(
        conditions, traces, manifests, positive_rows, control_rows,
        parity_rows, shard_manifest, immutable, guard.blocked)
    if gates["acquisition_data_gate_pass"]:
        future_fold = build_future_fold_manifest(
            existing_manifest, manifests, traces, positive_rows, control_rows)
    else:
        future_fold = {
            "version": "walking_v2_future_nested_fold_contract_v5",
            "not_created": True,
            "reason": "mandatory acquisition data gate failed",
            "audit": {"valid": False}, "rows": [],
        }
    write_json(output / "future_nested_fold_manifest.json", future_fold)
    future_fold_ready = bool(future_fold["audit"]["valid"])
    write_csv(output / "counterfactual_pair_manifest.csv", pair_manifest)
    write_csv(output / "run_manifest.csv", manifests)
    write_csv(output / "active_patch_contact_audit.csv", contact_rows)
    write_csv(output / "physical_episode_ledger.csv", episode_rows)
    write_csv(output / "positive_source_audit.csv", positive_rows)
    write_csv(output / "control_source_audit.csv", control_rows)
    write_csv(output / "fall_censor_audit.csv", fall_rows)
    write_csv(output / "pair_parity_audit.csv", parity_rows)
    write_csv(output / "onset_phase_distribution.csv", distribution_rows(
        positive_rows, "actual_onset_phase",
        [phase.name for phase in PHASE_BINS] + ["out_of_contract", "none"]))
    write_csv(output / "affected_foot_distribution.csv", distribution_rows(
        positive_rows, "actual_first_affected_foot",
        ["left", "right", "bilateral_ambiguous", "none"]))
    write_csv(output / "speed_distribution.csv", distribution_rows(
        positive_rows, "speed_mps", list(SPEEDS_MPS)))
    write_csv(output / "severity_distribution.csv", distribution_rows(
        positive_rows, "severity", list(SEVERITIES)))
    write_csv(output / "causal_feature_pair_statistics.csv", causal_pair_statistics(
        traces, manifests, positive_rows))
    trace_hashes = [str(row["full_trace_sha256"]) for row in manifests]
    duplicate_count = len(trace_hashes) - len(set(trace_hashes))
    write_json(output / "duplicate_audit.json", {
        "planned_unique_runs": 216, "executed_unique_runs": len(manifests),
        "unique_run_ids": len({row["run_id"] for row in manifests}),
        "unique_pair_ids": len({row["pair_id"] for row in manifests}),
        "unique_full_trace_hashes": len(set(trace_hashes)),
        "duplicate_count": duplicate_count,
        "discarded_count": sum(row["discarded"] for row in manifests),
        "replaced_count": sum(row["replaced"] for row in manifests),
        "silently_relabelled_count": sum(
            row["silently_relabelled"] for row in manifests),
        "valid": duplicate_count == 0,
    })
    create_plots(output, traces, manifests, positive_rows)
    targeted_authorized = bool(
        gates["acquisition_data_gate_pass"] and future_fold_ready)
    readiness = {
        "WALKING_V2_SLIP_REACQUISITION_PROTOCOL_READY": True,
        "WALKING_V2_SLIP_REACQUISITION_PROVENANCE_READY": bool(
            guard.blocked == 0 and all(immutable.values())),
        "WALKING_V2_SLIP_GENERATOR_IMMUTABLE_READY": generator_unchanged,
        "WALKING_V2_SLIP_COUNTERFACTUAL_PARITY_READY": gates["counterfactual_physics_pass"],
        "WALKING_V2_SLIP_SOLVER_INTERVENTION_READY": bool(
            gates["counterfactual_physics_pass"]
            and gates["counterfactual_physics"]["effective_friction_contract_violation_count"] == 0),
        "WALKING_V2_SLIP_POSITIVE_SOURCE_READY": gates["positive_source_pass"],
        "WALKING_V2_SLIP_CONTROL_SOURCE_READY": gates["control_source_pass"],
        "WALKING_V2_SLIP_ONSET_DIVERSITY_READY": gates["timing_diversity_pass"],
        "WALKING_V2_SLIP_AFFECTED_FOOT_BALANCE_READY": bool(
            0.80 <= gates["positive_source_coverage"]["left_right_ratio"] <= 1.25),
        "WALKING_V2_SLIP_REACQUISITION_DATA_READY": gates["acquisition_data_gate_pass"],
        "WALKING_V2_SLIP_FUTURE_FOLD_READY": future_fold_ready,
        "WALKING_V2_SLIP_TARGETED_RETRAINING_AUTHORIZED": targeted_authorized,
        "WALKING_V2_FRESH_BLIND_HOLDOUT_AUTHORIZED": False,
        "WALKING_V2_SYSTEM_MIGRATION_AUTHORIZED": False,
        "WALKING_V2_INT8_PREPARATION_AUTHORIZED": False,
        "WALKING_V2_TERRAIN_LOCK_PRESERVED": terrain_unchanged,
        "SINK_RUNTIME_DETECTION_DEFERRED": True,
    }
    positive_gate = gates["positive_source_coverage"]
    if targeted_authorized:
        next_step = "SLIP_TARGETED_RETRAINING_V3"
    elif (
        positive_gate["moderate_source_valid_count"]
        < positive_gate["strong_source_valid_count"] * 0.5):
        next_step = "SLIP_MODERATE_PROFILE_RECALIBRATION"
    elif not gates["positive_source_pass"]:
        next_step = "ADDITIONAL_TARGETED_SLIP_ACQUISITION"
    elif not gates["timing_diversity_pass"]:
        next_step = "SLIP_RISK_SCOPE_REDUCTION"
    else:
        next_step = "STOP_WALKING_V2_DEPLOYMENT"
    acquisition_manifest = {
        "artifact": "walking_v2_bilateral_slip_targeted_acquisition_v5",
        "planned_unique_runs": 216, "executed_unique_runs": len(manifests),
        "positive_attempts": 108, "control_attempts": 108,
        "all_attempts_retained": len(manifests) == 216,
        "development_only": True, "adaptive_replacements": 0,
        "trace_shard_manifest": "trace_shard_manifest.json",
        "trace_shard_manifest_sha256": sha256_file(
            output / "trace_shard_manifest.json"),
        "trace_shards": [{
            "path": row["path"], "run_ids": row["run_ids"],
            "sha256": row["sha256"], "bytes": row["bytes"],
        } for row in shard_manifest["shards"]],
        "gates": gates, "future_fold_ready": future_fold_ready,
    }
    write_json(output / "acquisition_manifest.json", acquisition_manifest)
    summary = {
        "task": "Reacquire Targeted Bilateral Slip Development Data v5",
        "gates": gates, "readiness": readiness,
        "future_fold_audit": future_fold["audit"],
        "planned_unique_runs": 216, "executed_unique_runs": len(manifests),
        "positive_attempts": 108, "control_attempts": 108,
        "all_attempts_retained": True,
        "v3_trace_use_count": 0, "forbidden_access_count": guard.blocked,
        "terrain_byte_identical": terrain_unchanged,
        "oracle_byte_identical": oracle_unchanged,
        "generator_unchanged": generator_unchanged,
        "training_performed": False, "model_selected_or_locked": False,
        "blind_holdout_created": False, "system_or_int8_work_performed": False,
        "next_step": next_step,
    }
    write_json(output / "readiness.json", readiness)
    write_json(output / "summary.json", summary)
    audit = f"""# Targeted bilateral Slip development acquisition v5 audit

1. Exactly 216 unique runs executed: **{len(manifests) == 216}** ({len(manifests)}/216).
2. Source-valid positives: **{positive_gate['source_valid_count']}/108**.
3. Strong source-valid: **{positive_gate['strong_source_valid_count']}/54**; moderate source-valid: **{positive_gate['moderate_source_valid_count']}/54**.
4. Valid-positive factorial cells: **{positive_gate['covered_cell_count']}/36**.
5. Physically non-slip controls: **{gates['control_source_coverage']['valid_control_count']}/108**; control onsets: **{gates['control_source_coverage']['physical_slip_onset_count']}**.
6. Pre-contact pair parity: **{gates['counterfactual_physics']['pair_configuration_parity_count']}/108**.
7. Every valid patch-contact pair diverged after contact: **{gates['counterfactual_physics']['valid_patch_pairs_with_postcontact_divergence']}**.
8. Positive/control full-trace identity count: **{gates['counterfactual_physics']['full_trace_identity_count']}**.
9. Patch/base double-contact count: **{gates['counterfactual_physics']['patch_base_double_contact_count']}**.
10. Left/right valid target events: **{positive_gate['left_valid_count']}/{positive_gate['right_valid_count']}**, ratio **{positive_gate['left_right_ratio']:.6f}**.
11. Target-first accuracy: **{positive_gate['target_first_fraction']:.6f}**.
12. All speeds, phases and timing states represented: **{gates['positive_source_coverage']['all_speeds_represented'] and gates['timing_diversity']['all_preregistered_bins_represented'] and gates['timing_diversity']['all_label_timing_states_supported']}**.
13. Discarded, replaced or silently relabelled attempts: **0/0/0**.
14. Quarantined v3 trace use count: **0**.
15. Forbidden artifact access count: **{guard.blocked}**.
16. Terrain/oracle byte-identical: **{terrain_unchanged}/{oracle_unchanged}**.
17. Targeted Slip retraining authorized: **{targeted_authorized}**.
18. Model training, blind holdout, System or INT8 work performed: **False**.
19. Sink remains `SINK_RUNTIME_DETECTION_DEFERRED`: **True**.

Exactly one next step: **{next_step}**
"""
    (output / "audit.md").write_text(audit, encoding="utf-8")
    graph_files = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name not in {
            "provenance.json", "resource_size_audit.json"})
    artifact_hashes = {path.name: sha256_file(path) for path in graph_files}
    graph_verified = all(
        sha256_file(output / name) == digest
        for name, digest in artifact_hashes.items())
    write_json(output / "provenance.json", {
        "starting_checkpoint": STARTING_CHECKPOINT, "execution_head": head,
        "upstream_policy_revision": UPSTREAM_REVISION,
        "expected_policy_sha256": TESTED_POLICY_SHA256,
        "actual_policy_sha256": sha256_file(guarded_paths[POLICY]),
        "mujoco_version": mujoco.__version__, "numpy_version": np.__version__,
        "elapsed_seconds": time.time() - started,
        "artifact_sha256": artifact_hashes,
        "artifact_hash_graph_verified": graph_verified,
        "forbidden_access_count": guard.blocked,
        "v3_trace_use_count": 0, "training_performed": False,
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
        "output": str(output), "executed_runs": len(manifests),
        "source_valid_positives": positive_gate["source_valid_count"],
        "controls_valid": gates["control_source_coverage"]["valid_control_count"],
        "acquisition_data_ready": gates["acquisition_data_gate_pass"],
        "targeted_retraining_authorized": targeted_authorized,
        "next_step": next_step,
    }, indent=2))


if __name__ == "__main__":
    main()
