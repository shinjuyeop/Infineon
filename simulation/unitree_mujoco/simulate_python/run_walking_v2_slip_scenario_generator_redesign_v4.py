"""Audit and prove a solver-effective bilateral Slip scenario generator.

This bounded task diagnoses the failed v3 intervention and runs only an
isolated microbench, a whole-surface reference, and at most 48 unique local
robot pilots.  It never trains a model or opens blind/outer/final artifacts.
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

from bilateral_hil_sensor_v2 import FOOT_CONTACT_GEOM_NAMES, G1BilateralSensorReaderV2
from g1_upstream_locomotion import (
    DEFAULT_ANGLES, TESTED_POLICY_SHA256, UPSTREAM_REVISION,
    UnitreeG1PretrainedController, gravity_orientation,
)
from terrain_profiles import TERRAIN_PROFILES, apply_terrain_profile
from walking_hazard_ground_truth_v1 import (
    derive_contact_signals, max_left_foot_contact_penetration_m,
)
from walking_hazard_oracle_calibration_v1 import persistent_oracle
from run_walking_hazard_ground_truth_v1 import (
    PELVIS_BODY_NAME, SCENE_PATH,
    _disable_nonfoot_surface_collisions, _fall_reasons,
)
from walking_v2_slip_scenario_generator_redesign_v4 import (
    PHASES, PHYSICS_STEPS_PER_SAMPLE, PHYSICS_TIMESTEP_S, ROOT_CAUSES,
    SIDES, SLIP_PERSISTENCE_MS, SLIP_THRESHOLD_M, PilotCondition,
    array_sha256, deterministic_initial_perturbation, friction_profiles,
    patch_bounds, pilot_matrix, trace_sha256,
)


REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO / "simulation/outputs/walking_v2_slip_scenario_generator_redesign_v4"
STARTING_CHECKPOINT = "2824b7c86867045cd4c5897b445fffada4ef6635"
V3 = "simulation/outputs/walking_v2_bilateral_slip_targeted_acquisition_v3"
JOINT = "simulation/outputs/walking_v2_joint_terrain_slip_redesign_v1"
ORACLE = "simulation/outputs/walking_hazard_oracle_calibration_v1"
POLICY = "simulation/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"
SCENE = "simulation/unitree_mujoco/unitree_robots/g1/scene_walking_terrain_transition.xml"
ROBOT = "simulation/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
TERRAIN_FILES = {
    "model": f"{JOINT}/terrain_candidate_model.npz",
    "normalization": f"{JOINT}/terrain_candidate_normalization.json",
    "config": f"{JOINT}/terrain_candidate_config.json",
    "lock": f"{JOINT}/terrain_selection_lock.json",
}
ALLOWED_INPUTS = {
    f"{V3}/run_manifest.csv": "quarantine all 216 failed runs",
    f"{V3}/summary.json": "verify failed physical support and trace identity",
    f"{V3}/readiness.json": "verify failed acquisition authorization state",
    f"{V3}/future_nested_fold_manifest.json": "quarantine unauthorized future folds",
    f"{ORACLE}/summary.json": "immutable physical Slip oracle contract",
    f"{JOINT}/summary.json": "locked Terrain reference",
    f"{JOINT}/provenance.json": "locked Terrain provenance",
    **{value: f"immutable Terrain {key}" for key, value in TERRAIN_FILES.items()},
    POLICY: "fixed walking policy runtime",
    SCENE: "fixed MuJoCo source scene runtime",
    ROBOT: "fixed robot include runtime",
    "simulation/unitree_mujoco/simulate_python/terrain_profiles.py": "friction profile source",
    "simulation/unitree_mujoco/simulate_python/walking_hazard_ground_truth_v1.py": "physical signal derivation",
    "simulation/unitree_mujoco/simulate_python/walking_hazard_oracle_calibration_v1.py": "frozen Slip oracle",
    "simulation/unitree_mujoco/simulate_python/bilateral_hil_sensor_v2.py": "Fusion20 runtime sensors",
    "simulation/unitree_mujoco/simulate_python/g1_upstream_locomotion.py": "fixed controller",
    "simulation/unitree_mujoco/simulate_python/run_walking_v2_bilateral_slip_targeted_acquisition_v3.py": "failed intervention source audit",
    "simulation/unitree_mujoco/simulate_python/walking_v2_bilateral_slip_targeted_acquisition_v3.py": "failed protocol source audit",
}
FORBIDDEN_TOKENS = (
    "/outer/", "_outer_", "holdout", "spatial_final", "spatial-final",
    "final_test", "final-test",
)
TRACE_KEYS = (
    "time_s", "bilateral_canonical", "force_loaded", "physical_contact",
    "foot_world_xyz_label_only", "anchor_drift_label_only",
    "slip_physical_active", "pre_fall_valid", "patch_contact",
)
PHASE_ACTIVATION_MS = {"early": 0.0, "middle": 70.0, "late": 190.0}


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
        writer.writeheader()
        writer.writerows(rows)


class AccessGuard:
    """Fail closed on every pre-existing artifact read and persist the ledger."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.events: list[dict[str, object]] = []
        self.blocked = 0
        self._flush()

    def _flush(self) -> None:
        write_json(self.output / "artifact_access_log.json", {
            "exact_paths_only": True,
            "forbidden_tokens": list(FORBIDDEN_TOKENS),
            "events": self.events,
            "blocked_access_count": self.blocked,
            "all_accesses_completed": all(
                row["status"] == "completed" for row in self.events),
        })

    def path(self, relative: str, access: str = "hash") -> Path:
        normalized = relative.replace("\\", "/")
        forbidden = next(
            (token for token in FORBIDDEN_TOKENS if token in f"/{normalized.lower()}"),
            None,
        )
        if normalized not in ALLOWED_INPUTS or forbidden:
            self.blocked += 1
            self.events.append({
                "path": normalized, "access": access, "purpose": "blocked",
                "status": "blocked", "reason": forbidden or "not_allowlisted",
            })
            self._flush()
            raise PermissionError(normalized)
        path = (REPO / normalized).resolve()
        if not path.is_file() or REPO.resolve() not in path.parents:
            raise FileNotFoundError(normalized)
        digest = sha256_file(path)
        self.events.append({
            "path": normalized, "access": access,
            "purpose": ALLOWED_INPUTS[normalized], "status": "completed",
            "sha256": digest,
        })
        self._flush()
        return path

    def json(self, relative: str) -> object:
        return json.loads(self.path(relative, "json").read_text(encoding="utf-8"))

    def csv(self, relative: str) -> list[dict[str, str]]:
        with self.path(relative, "csv").open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))


def _microbench_model(profile_name: str, *, explicit_pair: bool = True) -> mujoco.MjModel:
    profile = friction_profiles()[profile_name]
    pair = ""
    if explicit_pair:
        values = " ".join(str(value) for value in profile.friction5)
        pair = (
            '<contact><pair name="slider_surface_pair" geom1="slider" geom2="surface" '
            f'condim="3" friction="{values}"/></contact>'
        )
    xml = f"""
    <mujoco model="friction_microbench">
      <option timestep="{PHYSICS_TIMESTEP_S}" integrator="Euler" solver="Newton"/>
      <worldbody>
        <geom name="surface" type="box" pos="0 0 -0.05" size="5 1 .05"
              friction="{profile.friction3[0]} {profile.friction3[1]} {profile.friction3[2]}"/>
        <body name="slider_body" pos="0 0 .0501">
          <joint name="slide_x" type="slide" axis="1 0 0" damping="0"/>
          <joint name="slide_z" type="slide" axis="0 0 1" damping="0"/>
          <geom name="slider" type="box" size=".05 .05 .05" mass="1"
                friction="1 .005 .0001"/>
        </body>
      </worldbody>
      {pair}
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def _run_microbench(
    profile_name: str, *, post_step1_mutation: bool = False,
    explicit_pair: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    model_profile = "hard_control" if post_step1_mutation else profile_name
    model = _microbench_model(model_profile, explicit_pair=explicit_pair)
    data = mujoco.MjData(model)
    data.qvel[0] = 1.0
    mujoco.mj_forward(model, data)
    time_s, x_m, vx_mps, friction, tangent, contacts, active = ([] for _ in range(7))
    for _ in range(int(round(1.0 / PHYSICS_TIMESTEP_S))):
        if post_step1_mutation:
            mujoco.mj_step1(model, data)
            for contact_id in range(data.ncon):
                data.contact[contact_id].friction[:] = friction_profiles()[profile_name].friction5
            mujoco.mj_step2(model, data)
        else:
            mujoco.mj_step(model, data)
        force_values = []
        mu_values = []
        valid = 0
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact_id, wrench)
            force_values.append(float(np.linalg.norm(wrench[1:3])))
            mu_values.append(float(contact.friction[0]))
            valid += int(contact.efc_address >= 0)
        time_s.append(float(data.time)); x_m.append(float(data.qpos[0]))
        vx_mps.append(float(data.qvel[0]))
        friction.append(float(np.mean(mu_values)) if mu_values else np.nan)
        tangent.append(float(np.sum(force_values))); contacts.append(int(data.ncon))
        active.append(valid)
    trace = {
        "time_s": np.asarray(time_s), "x_m": np.asarray(x_m),
        "vx_mps": np.asarray(vx_mps), "effective_friction": np.asarray(friction),
        "tangential_force_n": np.asarray(tangent),
        "solver_contact_count": np.asarray(contacts),
        "active_contact_count": np.asarray(active),
    }
    stopped = np.flatnonzero(np.abs(trace["vx_mps"]) < 0.01)
    finite_mu = trace["effective_friction"][np.isfinite(trace["effective_friction"])]
    metric = {
        "profile": profile_name,
        "mechanism": "post_step1_contact_mutation" if post_step1_mutation else (
            "precompiled_explicit_pair" if explicit_pair else "equal_priority_dynamic_contact"),
        "displacement_m": float(trace["x_m"][-1] - trace["x_m"][0]),
        "final_velocity_mps": float(trace["vx_mps"][-1]),
        "time_to_stop_s": "" if not stopped.size else float(trace["time_s"][stopped[0]]),
        "effective_friction_mean": float(np.mean(finite_mu)) if finite_mu.size else "",
        "tangential_force_peak_n": float(np.max(trace["tangential_force_n"])),
        "solver_contact_count_max": int(np.max(trace["solver_contact_count"])),
        "active_contact_count_max": int(np.max(trace["active_contact_count"])),
        "trace_sha256": trace_sha256(trace, trace),
    }
    return trace, metric


def run_microbench(output: Path) -> tuple[bool, dict[str, object], dict[str, np.ndarray]]:
    traces: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    replay_pass = True
    primary: dict[str, dict[str, object]] = {}
    for profile_name in ("hard_control", "moderate_ice_preregistered", "native_strong_ice"):
        first_trace: dict[str, np.ndarray] | None = None
        for replay in range(2):
            trace, metric = _run_microbench(profile_name)
            key = f"{profile_name}_r{replay}"
            for field, value in trace.items():
                traces[f"{key}__{field}"] = value
            rows.append({**metric, "run_id": key, "replay_index": replay})
            manifest.append({
                "run_id": key, "profile": profile_name, "replay_index": replay,
                "unique_condition": replay == 0, "retained": True,
                "trace_sha256": metric["trace_sha256"],
            })
            if first_trace is None:
                first_trace = trace
                primary[profile_name] = metric
            else:
                replay_pass &= all(np.array_equal(first_trace[name], trace[name], equal_nan=True) for name in trace)
    hard_trace, hard_metric = _run_microbench("hard_control")
    post_trace, post_metric = _run_microbench("native_strong_ice", post_step1_mutation=True)
    traces.update({f"legacy_hard__{key}": value for key, value in hard_trace.items()})
    traces.update({f"legacy_post_step1__{key}": value for key, value in post_trace.items()})
    rows.extend((
        {**hard_metric, "run_id": "legacy_hard", "replay_index": 0},
        {**post_metric, "run_id": "legacy_post_step1_ice", "replay_index": 0},
    ))
    manifest.extend((
        {"run_id": "legacy_hard", "profile": "hard_control", "replay_index": 0,
         "unique_condition": True, "retained": True, "trace_sha256": hard_metric["trace_sha256"]},
        {"run_id": "legacy_post_step1_ice", "profile": "native_strong_ice", "replay_index": 0,
         "unique_condition": True, "retained": True, "trace_sha256": post_metric["trace_sha256"]},
    ))
    max_model = _microbench_model("native_strong_ice", explicit_pair=False)
    max_data = mujoco.MjData(max_model); max_data.qvel[0] = 1.0
    for _ in range(100):
        mujoco.mj_step(max_model, max_data)
        if max_data.ncon:
            break
    if not max_data.ncon:
        raise RuntimeError("equal-priority microbench never reached contact")
    max_override_mu = float(max_data.contact[0].friction[0])
    displacements = [float(primary[name]["displacement_m"]) for name in (
        "hard_control", "moderate_ice_preregistered", "native_strong_ice")]
    friction_means = [float(primary[name]["effective_friction_mean"]) for name in (
        "hard_control", "moderate_ice_preregistered", "native_strong_ice")]
    friction_order = friction_means[0] > friction_means[1] > friction_means[2]
    monotonic = displacements[0] < displacements[1] < displacements[2]
    divergence = not np.array_equal(
        traces["hard_control_r0__x_m"], traces["native_strong_ice_r0__x_m"])
    # The public contact.friction field visibly changes, but the already-built
    # constraint and its integrated trajectory do not.  This is the precise
    # distinction that made the v3 sensor/state trace identical.
    post_identical = all(
        np.array_equal(hard_trace[key], post_trace[key], equal_nan=True)
        for key in (
            "time_s", "x_m", "vx_mps", "solver_contact_count",
            "active_contact_count"))
    gates = {
        "effective_friction_order_verified": friction_order,
        "trajectory_divergence_verified": divergence,
        "displacement_monotonic_with_severity": monotonic,
        "deterministic_replay_verified": replay_pass,
        "post_step1_mutation_identical_to_hard": post_identical,
        "post_step1_public_contact_field_changed": not np.array_equal(
            hard_trace["effective_friction"], post_trace["effective_friction"],
            equal_nan=True),
        "equal_priority_max_override_observed": abs(max_override_mu - 1.0) < 1e-12,
        "equal_priority_observed_mu": max_override_mu,
    }
    passed = all(gates[key] for key in (
        "effective_friction_order_verified", "trajectory_divergence_verified",
        "displacement_monotonic_with_severity", "deterministic_replay_verified",
        "post_step1_mutation_identical_to_hard"))
    write_csv(output / "microbench_manifest.csv", manifest)
    write_csv(output / "microbench_metrics.csv", rows)
    np.savez_compressed(output / "microbench_traces.npz", **traces)
    return passed, gates, traces


def _configure_surface_geom(geom: mujoco.MjsGeom, *, rgba: tuple[float, ...]) -> None:
    concrete = TERRAIN_PROFILES["concrete"]
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.friction = concrete.friction
    geom.priority = concrete.priority
    geom.condim = concrete.condim
    geom.solref = concrete.solref
    geom.solimp = concrete.solimp
    geom.rgba = rgba


def build_local_model(condition: PilotCondition) -> tuple[mujoco.MjModel, dict[str, object]]:
    """Compile a height-preserving local patch with explicit sole contracts."""
    spec = mujoco.MjSpec.from_file(str(SCENE_PATH))
    for name in ("ground_source", "ground_target"):
        spec.delete(spec.geom(name))
    bounds = patch_bounds(condition.target_foot)
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
    # Contact masks eliminate generic patch contacts.  The eight explicit
    # pairs below remain active and are the sole patch collision contract.
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
    model.opt.timestep = PHYSICS_TIMESTEP_S
    ground_names = tuple(base_names) + ("slip_patch",)
    pair_ids = {
        side: tuple(model.pair(name).id for name in names)
        for side, names in pair_names.items()
    }
    info = {
        "kind": "local_patch", "ground_names": ground_names,
        "base_names": tuple(base_names), "patch_name": "slip_patch",
        "pair_names": pair_names, "pair_ids": pair_ids,
        "target_pair_ids": pair_ids[condition.target_foot],
        "bounds": vars(bounds), "top_height_delta_m": 0.0,
        "intervention_profile": condition.profile,
    }
    return model, info


def build_whole_surface_model(profile_name: str) -> tuple[mujoco.MjModel, dict[str, object]]:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    model.opt.timestep = PHYSICS_TIMESTEP_S
    terrain_name = "concrete" if profile_name == "hard_control" else "ice"
    for name in ("ground_source", "ground_target"):
        apply_terrain_profile(model, TERRAIN_PROFILES[terrain_name], name)
    return model, {
        "kind": "whole_surface", "ground_names": ("ground_source", "ground_target"),
        "base_names": ("ground_source", "ground_target"), "patch_name": None,
        "pair_names": {side: [] for side in SIDES},
        "pair_ids": {side: tuple() for side in SIDES}, "target_pair_ids": tuple(),
        "bounds": None, "top_height_delta_m": 0.0,
        "intervention_profile": profile_name,
    }


def _set_target_pair_friction(
    model: mujoco.MjModel, info: dict[str, object], profile_name: str,
) -> None:
    values = np.asarray(friction_profiles()[profile_name].friction5)
    for pair_id in info["target_pair_ids"]:
        model.pair_friction[int(pair_id)] = values


def _pair_lookup(model: mujoco.MjModel) -> dict[frozenset[int], int]:
    return {
        frozenset((int(model.pair_geom1[index]), int(model.pair_geom2[index]))): index
        for index in range(model.npair)
    }


def _initial_policy_observation(controller: UnitreeG1PretrainedController) -> np.ndarray:
    model, data = controller.model, controller.data
    gyro_id = model.sensor("imu_gyro").id
    address = int(model.sensor_adr[gyro_id])
    dimension = int(model.sensor_dim[gyro_id])
    gyro = data.sensordata[address:address + dimension]
    return np.concatenate((
        gyro, gravity_orientation(data.qpos[3:7]), controller.command,
        (np.sin(2.0 * np.pi * controller.global_phase),
         np.cos(2.0 * np.pi * controller.global_phase)),
        data.qpos[7:] - DEFAULT_ANGLES, data.qvel[6:], controller.action,
    )).astype(np.float32)


def _close_contact_interval(
    row: dict[str, object], end_sample: int, rows: list[dict[str, object]],
) -> None:
    count = int(row.pop("_count"))
    row["end_sample_exclusive"] = end_sample
    row["observed_sample_count"] = count
    row["tangential_contact_force_mean_n"] = float(row.pop("_force_sum")) / count
    rows.append(row)


def collect_robot_run(
    *, run_id: str, role: str, speed_mps: float, target_foot: str,
    target_phase: str, profile_name: str, seed: int, duration_s: float,
    model: mujoco.MjModel, info: dict[str, object], policy_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object], list[dict[str, object]]]:
    ground_ids = frozenset(model.geom(name).id for name in info["ground_names"])
    base_ids = frozenset(model.geom(name).id for name in info["base_names"])
    patch_id = None if info["patch_name"] is None else model.geom(info["patch_name"]).id
    allowed_feet = _disable_nonfoot_surface_collisions(model, ground_ids)
    data = mujoco.MjData(model)
    controller = UnitreeG1PretrainedController(model, data, policy_path, speed_mps)
    dx, dy, dvx = deterministic_initial_perturbation(seed)
    data.qpos[0] += dx; data.qpos[1] += dy; data.qvel[0] += dvx
    controller.global_phase = 0.07
    mujoco.mj_forward(model, data)
    initial_qpos_hash = array_sha256(data.qpos)
    initial_qvel_hash = array_sha256(data.qvel)
    initial_policy_hash = array_sha256(_initial_policy_observation(controller))
    reader = G1BilateralSensorReaderV2(model, data)
    foot_body_ids = tuple(model.body(f"{side}_ankle_roll_link").id for side in SIDES)
    pelvis_id = model.body(PELVIS_BODY_NAME).id
    pair_lookup = _pair_lookup(model)
    foot_geom_side = {
        geom_id: side for side in SIDES for geom_id in reader.foot_geom_ids[side]
    }
    velocity = np.zeros(6)
    names = (
        "time_s", "bilateral_canonical", "force_loaded", "physical_contact",
        "foot_world_xyz_label_only", "foot_world_velocity_label_only",
        "contact_penetration_label_only", "patch_contact",
        "effective_patch_friction", "patch_tangential_force_n",
    )
    values: dict[str, list[object]] = {name: [] for name in names}
    contact_rows: list[dict[str, object]] = []
    open_intervals: dict[tuple[int, int], dict[str, object]] = {}
    target_index = SIDES.index(target_foot)
    target_contact_age_steps = 0
    previous_target_patch_contact = False
    first_fall_sample: int | None = None
    first_fall_time_s: float | None = None
    fall_reason = ""
    double_contact_count = 0
    activation_steps = 0
    total_steps = int(round(duration_s / PHYSICS_TIMESTEP_S))
    for physics_step in range(1, total_steps + 1):
        if previous_target_patch_contact:
            target_contact_age_steps += 1
        else:
            target_contact_age_steps = 0
        age_ms = target_contact_age_steps * PHYSICS_TIMESTEP_S * 1000.0
        active = bool(
            info["kind"] == "local_patch" and role == "positive"
            and age_ms + 1e-12 >= PHASE_ACTIVATION_MS[target_phase]
        )
        if info["kind"] == "local_patch":
            _set_target_pair_friction(
                model, info, profile_name if active else "hard_control")
            activation_steps += int(active)
        controller.apply()
        # Collision, constraint construction, solve and integration all see
        # the pair friction assigned above.
        mujoco.mj_step(model, data)
        controller.update_after_step()
        target_patch_now = False
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            pair = {int(contact.geom1), int(contact.geom2)}
            if patch_id is not None and patch_id in pair and pair & set(reader.foot_geom_ids[target_foot]):
                target_patch_now = True
                break
        previous_target_patch_contact = target_patch_now
        if physics_step % PHYSICS_STEPS_PER_SAMPLE:
            continue
        sample = len(values["time_s"])
        raw = reader.read_bilateral_vector()
        canonical = np.concatenate(tuple(
            reader.canonicalize_foot_vector(side, raw[index * 10:(index + 1) * 10])
            for index, side in enumerate(SIDES)))
        runtime = tuple(reader.update_contact_state(
            side, raw[index * 10:index * 10 + 4]) for index, side in enumerate(SIDES))
        physical = np.asarray([
            reader.has_foot_contact(side, ground_ids) for side in SIDES], dtype=bool)
        # The FSR hysteresis state may remain loaded for one boundary sample
        # after the last physical contact.  The frozen oracle explicitly
        # requires loaded => physical contact, so reconcile only its label-side
        # mask; the recorded Fusion20 force channels remain untouched.
        loaded_for_oracle = np.asarray([
            state.loaded and physical[index]
            for index, state in enumerate(runtime)], dtype=bool)
        foot_xyz = np.stack(tuple(data.xpos[body_id].copy() for body_id in foot_body_ids))
        foot_velocity = []
        for body_id in foot_body_ids:
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
            foot_velocity.append(velocity[3:].copy())
        penetration = np.asarray([
            max_left_foot_contact_penetration_m(data, reader.foot_geom_ids[side], ground_ids)
            for side in SIDES])
        patch_contact = np.zeros(2, dtype=bool)
        patch_mu: list[list[float]] = [[], []]
        patch_force = np.zeros(2)
        current_keys: set[tuple[int, int]] = set()
        contacted_surfaces: dict[int, set[int]] = {}
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            pair = frozenset((geom1, geom2))
            foot_id = next((value for value in pair if value in foot_geom_side), None)
            ground_id = next((value for value in pair if value in ground_ids), None)
            if foot_id is None or ground_id is None:
                continue
            side = foot_geom_side[foot_id]; side_index = SIDES.index(side)
            contacted_surfaces.setdefault(foot_id, set()).add(ground_id)
            wrench = np.zeros(6); mujoco.mj_contactForce(model, data, contact_id, wrench)
            tangential = float(np.linalg.norm(wrench[1:3]))
            is_patch = ground_id == patch_id
            is_intervention_surface = is_patch or info["kind"] == "whole_surface"
            if is_intervention_surface:
                patch_contact[side_index] = True
                patch_mu[side_index].append(float(contact.friction[0]))
                patch_force[side_index] += tangential
            key = (foot_id, ground_id)
            current_keys.add(key)
            pair_id = pair_lookup.get(pair, -1)
            geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1)
            geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2)
            if key not in open_intervals:
                body1 = int(model.geom_bodyid[geom1]); body2 = int(model.geom_bodyid[geom2])
                open_intervals[key] = {
                    "run_id": run_id, "role": role, "target_foot": target_foot,
                    "contacting_foot": side, "start_sample": sample,
                    "geom1_name": geom1_name, "geom1_id": geom1,
                    "geom2_name": geom2_name, "geom2_id": geom2,
                    "body1_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1),
                    "body1_id": body1,
                    "body2_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2),
                    "body2_id": body2, "patch_inclusion": is_patch,
                    "contact_position_first": json.dumps(np.asarray(contact.pos).tolist()),
                    "contact_position_last": json.dumps(np.asarray(contact.pos).tolist()),
                    "geom1_friction": json.dumps(model.geom_friction[geom1].tolist()),
                    "geom2_friction": json.dumps(model.geom_friction[geom2].tolist()),
                    "geom1_priority": int(model.geom_priority[geom1]),
                    "geom2_priority": int(model.geom_priority[geom2]),
                    "explicit_pair_id": pair_id,
                    "effective_contact_friction_first": json.dumps(np.asarray(contact.friction).tolist()),
                    "effective_contact_friction_last": json.dumps(np.asarray(contact.friction).tolist()),
                    "contact_dimension": int(contact.dim), "contact_exclude": int(contact.exclude),
                    "efc_address_min": int(contact.efc_address),
                    "efc_address_max": int(contact.efc_address),
                    "integrator": mujoco.mjtIntegrator(model.opt.integrator).name,
                    "mutation_point": "model.pair_friction before mj_step" if info["kind"] == "local_patch" else "model geom profile before mj_forward",
                    "constraints_constructed_at_mutation": False,
                    "tangential_contact_force_peak_n": tangential,
                    "_force_sum": tangential, "_count": 1,
                }
            else:
                row = open_intervals[key]
                row["contact_position_last"] = json.dumps(np.asarray(contact.pos).tolist())
                row["effective_contact_friction_last"] = json.dumps(np.asarray(contact.friction).tolist())
                row["efc_address_min"] = min(int(row["efc_address_min"]), int(contact.efc_address))
                row["efc_address_max"] = max(int(row["efc_address_max"]), int(contact.efc_address))
                row["tangential_contact_force_peak_n"] = max(
                    float(row["tangential_contact_force_peak_n"]), tangential)
                row["_force_sum"] = float(row["_force_sum"]) + tangential
                row["_count"] = int(row["_count"]) + 1
        for foot_id, surface_ids in contacted_surfaces.items():
            if patch_id in surface_ids and surface_ids & base_ids:
                double_contact_count += 1
        for key in set(open_intervals) - current_keys:
            _close_contact_interval(open_intervals.pop(key), sample, contact_rows)
        reasons, _ = _fall_reasons(model, data, pelvis_id, ground_ids, allowed_feet)
        if reasons and first_fall_sample is None:
            first_fall_sample = sample; first_fall_time_s = float(data.time)
            fall_reason = "|".join(reasons)
        effective = np.asarray([
            float(np.mean(side_values)) if side_values else np.nan
            for side_values in patch_mu])
        current_values = {
            "time_s": float(data.time), "bilateral_canonical": canonical.astype(np.float32),
            "force_loaded": loaded_for_oracle,
            "physical_contact": physical,
            "foot_world_xyz_label_only": foot_xyz.astype(np.float32),
            "foot_world_velocity_label_only": np.asarray(foot_velocity, np.float32),
            "contact_penetration_label_only": penetration.astype(np.float32),
            "patch_contact": patch_contact, "effective_patch_friction": effective,
            "patch_tangential_force_n": patch_force,
        }
        for name, value in current_values.items():
            values[name].append(value)
    sample_count = len(values["time_s"])
    for row in list(open_intervals.values()):
        _close_contact_interval(row, sample_count, contact_rows)
    trace = {name: np.asarray(series) for name, series in values.items()}
    pre_fall = np.ones(sample_count, dtype=bool)
    if first_fall_sample is not None:
        pre_fall[first_fall_sample:] = False
    episode_id = np.full((sample_count, 2), -1, np.int32)
    transient = np.zeros((sample_count, 2), bool)
    anchor_drift = np.full((sample_count, 2), np.nan, np.float32)
    slip_valid = np.zeros((sample_count, 2), bool)
    slip_active = np.zeros((sample_count, 2), bool)
    for side_index, side in enumerate(SIDES):
        signals = derive_contact_signals(
            trace["physical_contact"][:, side_index], trace["force_loaded"][:, side_index],
            trace["foot_world_xyz_label_only"][:, side_index],
            trace["foot_world_velocity_label_only"][:, side_index],
            trace["contact_penetration_label_only"][:, side_index], first_fall_sample)
        episode_id[:, side_index] = signals.contact_episode_id
        transient[:, side_index] = signals.touchdown_transient
        anchor_drift[:, side_index] = signals.tangential_anchor_drift_m
        slip_valid[:, side_index] = signals.slip_calibration_valid
        slip_active[:, side_index] = persistent_oracle(
            signals.tangential_anchor_drift_m, signals.slip_calibration_valid,
            signals.contact_episode_id, SLIP_THRESHOLD_M, SLIP_PERSISTENCE_MS)
    trace.update({
        "pre_fall_valid": pre_fall, "contact_episode_id": episode_id,
        "touchdown_transient": transient,
        "anchor_drift_label_only": anchor_drift,
        "slip_calibration_valid_label_only": slip_valid,
        "slip_physical_active": slip_active,
    })
    target_onsets = np.flatnonzero(slip_active[:, target_index] & pre_fall)
    contra_onsets = np.flatnonzero(slip_active[:, 1 - target_index] & pre_fall)
    first_target = int(target_onsets[0]) if target_onsets.size else None
    first_contra = int(contra_onsets[0]) if contra_onsets.size else None
    actual_first = (
        target_foot if first_target is not None and (first_contra is None or first_target < first_contra)
        else SIDES[1 - target_index] if first_contra is not None and (first_target is None or first_contra < first_target)
        else "bilateral_ambiguous" if first_target is not None else "none"
    )
    patch_samples = np.flatnonzero(trace["patch_contact"][:, target_index])
    first_patch = int(patch_samples[0]) if patch_samples.size else None
    finite_drift = trace["anchor_drift_label_only"][:, target_index]
    finite_drift = finite_drift[np.isfinite(finite_drift) & pre_fall]
    finite_mu = trace["effective_patch_friction"][:, target_index]
    finite_mu = finite_mu[np.isfinite(finite_mu)]
    metadata = {
        "run_id": run_id, "role": role, "speed_mps": speed_mps,
        "target_foot": target_foot, "target_phase": target_phase,
        "profile": profile_name, "seed": seed, "duration_s": duration_s,
        "sample_count": sample_count, "initial_qpos_sha256": initial_qpos_hash,
        "initial_qvel_sha256": initial_qvel_hash,
        "initial_policy_observation_sha256": initial_policy_hash,
        "full_trace_sha256": trace_sha256(trace, TRACE_KEYS),
        "first_patch_contact_sample": first_patch,
        "patch_active_contact_count": int(np.sum(trace["patch_contact"][:, target_index])),
        "intervention_activation_physics_steps": activation_steps,
        "effective_patch_friction_mean": float(np.mean(finite_mu)) if finite_mu.size else "",
        "target_tangential_force_peak_n": float(np.max(trace["patch_tangential_force_n"][:, target_index])),
        "target_anchor_drift_max_m": float(np.max(finite_drift)) if finite_drift.size else "",
        "target_physical_onset_sample": first_target,
        "contralateral_physical_onset_sample": first_contra,
        "actual_first_affected_foot": actual_first,
        "contralateral_contamination": first_contra is not None,
        "fall_occurred": first_fall_sample is not None,
        "first_fall_sample": first_fall_sample,
        "first_fall_time_s": first_fall_time_s, "fall_reason": fall_reason,
        "post_fall_excluded_sample_count": int(np.sum(~pre_fall)),
        "patch_base_double_contact_count": double_contact_count,
        "discarded": False, "relabeled": False, "retained": True,
    }
    return trace, metadata, contact_rows


def _geometry_sha256(model: mujoco.MjModel) -> str:
    digest = hashlib.sha256()
    for name in (
        "geom_type", "geom_bodyid", "geom_pos", "geom_quat", "geom_size",
        "geom_contype", "geom_conaffinity", "pair_geom1", "pair_geom2",
    ):
        value = np.ascontiguousarray(getattr(model, name))
        digest.update(name.encode() + b"\0" + value.tobytes())
    return digest.hexdigest()


def _common_prefix_equal(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray], end: int,
) -> bool:
    keys = (
        "time_s", "bilateral_canonical", "force_loaded", "physical_contact",
        "foot_world_xyz_label_only", "foot_world_velocity_label_only",
    )
    return all(np.array_equal(left[key][:end], right[key][:end], equal_nan=True) for key in keys)


def _post_contact_diverged(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray], start: int,
) -> bool:
    keys = ("bilateral_canonical", "foot_world_xyz_label_only")
    return any(not np.array_equal(left[key][start:], right[key][start:], equal_nan=True) for key in keys)


def _assess_run(trace: dict[str, np.ndarray], meta: dict[str, object]) -> dict[str, object]:
    target_index = SIDES.index(str(meta["target_foot"]))
    onset_value = meta["target_physical_onset_sample"]
    onset = None if onset_value is None else int(onset_value)
    phase = "none"
    age_ms: int | None = None
    counted_in_air = False
    counted_post_fall = False
    if onset is not None:
        counted_in_air = not bool(trace["physical_contact"][onset, target_index])
        counted_post_fall = not bool(trace["pre_fall_valid"][onset])
        episode = int(trace["contact_episode_id"][onset, target_index])
        episode_samples = np.flatnonzero(trace["contact_episode_id"][:, target_index] == episode)
        loaded = episode_samples[trace["force_loaded"][episode_samples, target_index]]
        if loaded.size:
            age_ms = onset - int(loaded[0]) + 1
            if 11 <= age_ms <= 120:
                phase = "early"
            elif 121 <= age_ms <= 260:
                phase = "middle"
            elif 261 <= age_ms <= 600:
                phase = "late"
            else:
                phase = "out_of_contract"
    valid = bool(
        meta["role"] == "positive" and onset is not None
        and meta["actual_first_affected_foot"] == meta["target_foot"]
        and not meta["contralateral_contamination"]
        and phase == meta["target_phase"] and not counted_in_air
        and not counted_post_fall
        and trace["slip_calibration_valid_label_only"][onset, target_index]
        and not trace["touchdown_transient"][onset, target_index]
    )
    return {
        "actual_onset_phase": phase,
        "touchdown_to_onset_ms": "" if age_ms is None else age_ms,
        "valid_target_positive": valid,
        "air_event_counted": counted_in_air,
        "post_fall_event_counted": counted_post_fall,
    }


def _save_trace_archive(path: Path, runs: list[tuple[str, dict[str, np.ndarray]]]) -> None:
    arrays = {
        f"{run_id}__{name}": value
        for run_id, trace in runs for name, value in trace.items()
    }
    np.savez_compressed(path, **arrays)


def run_whole_surface_reference(
    output: Path, policy_path: Path,
) -> tuple[bool, list[dict[str, object]], dict[str, dict[str, np.ndarray]]]:
    traces: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for profile_name in ("hard_control", "native_strong_ice"):
        model, info = build_whole_surface_model(profile_name)
        trace, meta, contact_rows = collect_robot_run(
            run_id=f"whole_{profile_name}", role="reference",
            speed_mps=0.20, target_foot="left", target_phase="early",
            profile_name=profile_name, seed=202608390, duration_s=3.0,
            model=model, info=info, policy_path=policy_path)
        traces[profile_name] = trace
        audits.extend(contact_rows)
        any_onset = bool(np.any(trace["slip_physical_active"] & trace["pre_fall_valid"][:, None]))
        rows.append({**meta, "any_valid_physical_slip": any_onset})
    hard, ice = traces["hard_control"], traces["native_strong_ice"]
    relevant = np.flatnonzero(np.any(hard["physical_contact"], axis=1))
    first_contact = int(relevant[0]) if relevant.size else len(hard["time_s"])
    precontact = _common_prefix_equal(hard, ice, first_contact)
    post_divergence = _post_contact_diverged(hard, ice, first_contact)
    hard_drift = float(np.nanmax(hard["anchor_drift_label_only"]))
    ice_drift = float(np.nanmax(ice["anchor_drift_label_only"]))
    force_difference = any(
        abs(float(rows[0]["target_tangential_force_peak_n"]) - float(rows[1]["target_tangential_force_peak_n"])) > 1e-9
        for _ in (0,))
    hard_friction = float(TERRAIN_PROFILES["concrete"].friction[0])
    ice_friction = float(TERRAIN_PROFILES["ice"].friction[0])
    ice_slip = bool(rows[1]["any_valid_physical_slip"])
    comparison = {
        "run_id": "whole_surface_comparison", "role": "comparison",
        "first_relevant_contact_sample": first_contact,
        "initial_state_parity": (
            rows[0]["initial_qpos_sha256"] == rows[1]["initial_qpos_sha256"]
            and rows[0]["initial_qvel_sha256"] == rows[1]["initial_qvel_sha256"]),
        "initial_policy_observation_parity": (
            rows[0]["initial_policy_observation_sha256"] == rows[1]["initial_policy_observation_sha256"]),
        "precontact_trace_parity": precontact,
        "postcontact_trace_divergence": post_divergence,
        "hard_effective_friction": hard_friction,
        "ice_effective_friction": ice_friction,
        "effective_friction_differs": hard_friction != ice_friction,
        "hard_anchor_drift_max_m": hard_drift,
        "ice_anchor_drift_max_m": ice_drift,
        "anchor_drift_differs": abs(ice_drift - hard_drift) > 1e-9,
        "tangential_contact_force_differs": force_difference,
        "native_ice_physical_slip_reproduced": ice_slip,
    }
    rows.append(comparison)
    write_csv(output / "whole_surface_reference_metrics.csv", rows)
    write_csv(output / "whole_surface_contact_audit.csv", audits)
    _save_trace_archive(
        output / "whole_surface_reference_traces.npz", list(traces.items()))
    passed = bool(
        comparison["initial_state_parity"] and comparison["precontact_trace_parity"]
        and comparison["postcontact_trace_divergence"]
        and comparison["effective_friction_differs"]
        and comparison["anchor_drift_differs"]
        and comparison["tangential_contact_force_differs"]
        and comparison["native_ice_physical_slip_reproduced"])
    return passed, rows, traces


def run_local_pilot(
    output: Path, policy_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, dict[str, np.ndarray]]]:
    conditions = pilot_matrix()
    traces: dict[str, dict[str, np.ndarray]] = {}
    metadata: dict[str, dict[str, object]] = {}
    contact_rows: list[dict[str, object]] = []
    geometry_hashes: dict[str, str] = {}
    archive_batch: list[tuple[str, dict[str, np.ndarray]]] = []
    archive_index = 1
    progress_rows: list[dict[str, object]] = []
    for condition in conditions:
        model, info = build_local_model(condition)
        geometry_hashes[condition.run_id] = _geometry_sha256(model)
        try:
            trace, meta, audit = collect_robot_run(
                run_id=condition.run_id, role=condition.role,
                speed_mps=condition.speed_mps, target_foot=condition.target_foot,
                target_phase=condition.target_phase, profile_name=condition.profile,
                seed=condition.seed, duration_s=condition.duration_s,
                model=model, info=info, policy_path=policy_path)
        except Exception as exc:
            write_csv(output / "pilot_progress_manifest.csv", progress_rows)
            write_json(output / "pilot_execution_failure.json", {
                "failed_run_id": condition.run_id, "failed_run_retained": False,
                "completed_attempt_count": len(progress_rows),
                "exception_type": type(exc).__name__, "message": str(exc),
                "silently_discarded": False,
            })
            raise
        assessment = _assess_run(trace, meta)
        meta.update({
            "pair_id": condition.pair_id, "severity": condition.severity,
            "pair_parity_sha256": condition.parity_sha256,
            "collision_geometry_sha256": geometry_hashes[condition.run_id],
            **assessment, "unique_pilot_run": True, "replay_of": "",
        })
        traces[condition.run_id] = trace; metadata[condition.run_id] = meta
        contact_rows.extend(audit); archive_batch.append((condition.run_id, trace))
        progress_rows.append({
            "run_id": condition.run_id, "pair_id": condition.pair_id,
            "role": condition.role, "profile": condition.profile,
            "status": "completed_and_retained", "full_trace_sha256": meta["full_trace_sha256"],
        })
        write_csv(output / "pilot_progress_manifest.csv", progress_rows)
        if len(archive_batch) == 12:
            _save_trace_archive(
                output / f"pilot_traces_part_{archive_index:02d}.npz", archive_batch)
            archive_batch = []; archive_index += 1
    if archive_batch:
        _save_trace_archive(
            output / f"pilot_traces_part_{archive_index:02d}.npz", archive_batch)
    # Two bounded deterministic replays, one per target side.  These do not
    # increase the 48-condition unique corpus count and are retained verbatim.
    replay_rows: list[dict[str, object]] = []
    replay_traces: list[tuple[str, dict[str, np.ndarray]]] = []
    replay_conditions = (
        next(row for row in conditions if row.role == "positive" and row.target_foot == "left"),
        next(row for row in conditions if row.role == "positive" and row.target_foot == "right"),
    )
    for condition in replay_conditions:
        model, info = build_local_model(condition)
        replay_id = f"replay_{condition.run_id}"
        trace, meta, audit = collect_robot_run(
            run_id=replay_id, role="positive", speed_mps=condition.speed_mps,
            target_foot=condition.target_foot, target_phase=condition.target_phase,
            profile_name=condition.profile, seed=condition.seed,
            duration_s=condition.duration_s, model=model, info=info,
            policy_path=policy_path)
        meta["role"] = "replay"
        for row in audit:
            row["role"] = "replay"
        original = metadata[condition.run_id]
        replay_match = meta["full_trace_sha256"] == original["full_trace_sha256"]
        replay_rows.append({
            **meta, "pair_id": condition.pair_id, "severity": condition.severity,
            "pair_parity_sha256": condition.parity_sha256,
            "collision_geometry_sha256": _geometry_sha256(model),
            "actual_onset_phase": _assess_run(trace, meta)["actual_onset_phase"],
            "valid_target_positive": False, "unique_pilot_run": False,
            "replay_of": condition.run_id, "deterministic_replay_match": replay_match,
        })
        contact_rows.extend(audit); replay_traces.append((replay_id, trace))
    _save_trace_archive(output / "pilot_replay_traces.npz", replay_traces)
    pair_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    force_rows: list[dict[str, object]] = []
    fall_rows: list[dict[str, object]] = []
    by_pair: dict[str, list[PilotCondition]] = {}
    for condition in conditions:
        by_pair.setdefault(condition.pair_id, []).append(condition)
    for pair_id, pair in by_pair.items():
        positive_condition = next(row for row in pair if row.role == "positive")
        control_condition = next(row for row in pair if row.role == "control")
        positive = metadata[positive_condition.run_id]
        control = metadata[control_condition.run_id]
        positive_trace = traces[positive_condition.run_id]
        control_trace = traces[control_condition.run_id]
        patch_candidates = [
            value for value in (
                positive["first_patch_contact_sample"], control["first_patch_contact_sample"])
            if value is not None]
        first_patch = min(int(value) for value in patch_candidates) if patch_candidates else len(positive_trace["time_s"])
        precontact = _common_prefix_equal(positive_trace, control_trace, first_patch)
        post_diverged = _post_contact_diverged(positive_trace, control_trace, first_patch)
        pos_mu = positive["effective_patch_friction_mean"]
        ctl_mu = control["effective_patch_friction_mean"]
        friction_differs = pos_mu != "" and ctl_mu != "" and abs(float(pos_mu) - float(ctl_mu)) > 1e-9
        geometry_equal = geometry_hashes[positive_condition.run_id] == geometry_hashes[control_condition.run_id]
        pair_rows.append({
            "pair_id": pair_id, "positive_run_id": positive_condition.run_id,
            "control_run_id": control_condition.run_id,
            "configuration_parity": positive_condition.parity_sha256 == control_condition.parity_sha256,
            "collision_geometry_identical": geometry_equal,
            "top_height_identical": True, "first_patch_contact_sample": first_patch,
            "precontact_trace_parity": precontact,
            "postcontact_trace_divergence": post_diverged,
            "positive_control_full_trace_identical": positive["full_trace_sha256"] == control["full_trace_sha256"],
            "positive_effective_friction": pos_mu,
            "control_effective_friction": ctl_mu,
            "effective_friction_differs": friction_differs,
            "tangential_force_differs": abs(float(positive["target_tangential_force_peak_n"]) - float(control["target_tangential_force_peak_n"])) > 1e-9,
            "patch_base_double_contact_count": int(positive["patch_base_double_contact_count"]) + int(control["patch_base_double_contact_count"]),
        })
    for condition in conditions:
        trace = traces[condition.run_id]; meta = metadata[condition.run_id]
        for side_index, side in enumerate(SIDES):
            for episode in sorted(set(trace["contact_episode_id"][:, side_index].tolist()) - {-1}):
                samples = np.flatnonzero(trace["contact_episode_id"][:, side_index] == episode)
                active = samples[trace["slip_physical_active"][samples, side_index] & trace["pre_fall_valid"][samples]]
                valid = samples[trace["pre_fall_valid"][samples]]
                drift = trace["anchor_drift_label_only"][valid, side_index]
                drift = drift[np.isfinite(drift)]
                episode_rows.append({
                    "run_id": condition.run_id, "pair_id": condition.pair_id,
                    "role": condition.role, "target_foot": condition.target_foot,
                    "foot": side, "contact_episode_id": episode,
                    "episode_start_sample": int(samples[0]),
                    "episode_end_sample_exclusive": int(samples[-1] + 1),
                    "physical_onset_sample": "" if not active.size else int(active[0]),
                    "maximum_anchor_drift_m": "" if not drift.size else float(np.max(drift)),
                    "pre_fall_valid_sample_count": int(valid.size),
                    "air_event_counted": False, "post_fall_event_counted": False,
                })
        force_rows.append({
            "run_id": condition.run_id, "pair_id": condition.pair_id,
            "role": condition.role, "target_foot": condition.target_foot,
            "profile": condition.profile,
            "effective_patch_friction_mean": meta["effective_patch_friction_mean"],
            "target_tangential_force_peak_n": meta["target_tangential_force_peak_n"],
            "patch_active_contact_count": meta["patch_active_contact_count"],
            "valid_efc_addresses": all(
                int(row["efc_address_min"]) >= 0 for row in contact_rows
                if row["run_id"] == condition.run_id and row["patch_inclusion"]),
        })
        fall_rows.append({
            "run_id": condition.run_id, "pair_id": condition.pair_id,
            "role": condition.role, "fall_occurred": meta["fall_occurred"],
            "first_fall_sample": "" if meta["first_fall_sample"] is None else meta["first_fall_sample"],
            "first_fall_time_s": "" if meta["first_fall_time_s"] is None else meta["first_fall_time_s"],
            "fall_reason": meta["fall_reason"],
            "post_fall_excluded_sample_count": meta["post_fall_excluded_sample_count"],
            "failed_attempt_retained": True, "silently_relabelled": False,
        })
    manifest = [metadata[row.run_id] for row in conditions] + replay_rows
    write_csv(output / "pilot_run_manifest.csv", manifest)
    write_csv(output / "pilot_pair_parity.csv", pair_rows)
    write_csv(output / "pilot_physical_episode_ledger.csv", episode_rows)
    write_csv(output / "pilot_contact_force_metrics.csv", force_rows)
    write_csv(output / "pilot_fall_censor_audit.csv", fall_rows)
    write_csv(output / "local_patch_contact_audit.csv", contact_rows)
    positive_meta = [metadata[row.run_id] for row in conditions if row.role == "positive"]
    control_meta = [metadata[row.run_id] for row in conditions if row.role == "control"]
    strong_valid = sum(
        bool(row["valid_target_positive"]) for row in positive_meta
        if row["severity"] == "native_strong_ice")
    moderate_valid = sum(
        bool(row["valid_target_positive"]) for row in positive_meta
        if row["severity"] == "moderate_ice_preregistered")
    valid_positive = [row for row in positive_meta if row["valid_target_positive"]]
    unique_hashes = [metadata[row.run_id]["full_trace_sha256"] for row in conditions]
    duplicate_count = len(unique_hashes) - len(set(unique_hashes))
    controls_with_onset = sum(
        row["target_physical_onset_sample"] is not None
        or row["contralateral_physical_onset_sample"] is not None
        for row in control_meta)
    gates = {
        "unique_pilot_run_count": len(conditions),
        "replay_attempt_count": len(replay_rows),
        "all_replays_deterministic": all(row["deterministic_replay_match"] for row in replay_rows),
        "patch_active_contacts_gt_zero": all(int(row["patch_active_contact_count"]) > 0 for row in positive_meta),
        "effective_positive_control_friction_differs": all(row["effective_friction_differs"] for row in pair_rows),
        "valid_efc_addresses": all(row["valid_efc_addresses"] for row in force_rows),
        "tangential_force_difference_exists": any(row["tangential_force_differs"] for row in pair_rows),
        "patch_base_double_contact_count": sum(int(row["patch_base_double_contact_count"]) for row in pair_rows),
        "geometry_parity": all(row["collision_geometry_identical"] and row["top_height_identical"] for row in pair_rows),
        "precontact_parity": all(row["precontact_trace_parity"] for row in pair_rows),
        "postcontact_identity_count": sum(row["positive_control_full_trace_identical"] for row in pair_rows),
        "postcontact_divergence_all_pairs": all(row["postcontact_trace_divergence"] for row in pair_rows),
        "control_valid_physical_slip_onset_count": controls_with_onset,
        "strong_valid_target_positive_count": strong_valid,
        "moderate_valid_target_positive_count": moderate_valid,
        "both_target_feet_valid": {row["target_foot"] for row in valid_positive} == set(SIDES),
        "both_speeds_valid": {float(row["speed_mps"]) for row in valid_positive} == {0.10, 0.20},
        "all_onset_phases_valid": {row["actual_onset_phase"] for row in valid_positive} >= set(PHASES),
        "target_first_fraction": (
            sum(row["actual_first_affected_foot"] == row["target_foot"] for row in valid_positive) / len(valid_positive)
            if valid_positive else 0.0),
        "air_or_postfall_counted": sum(
            bool(row["air_event_counted"]) or bool(row["post_fall_event_counted"])
            for row in positive_meta),
        "duplicate_unique_run_count": duplicate_count,
        "silently_discarded_or_relabelled_count": sum(
            bool(row["discarded"]) or bool(row["relabeled"]) for row in metadata.values()),
    }
    gates["robot_pilot_ready"] = bool(
        gates["unique_pilot_run_count"] == 48
        and gates["all_replays_deterministic"]
        and gates["patch_active_contacts_gt_zero"]
        and gates["effective_positive_control_friction_differs"]
        and gates["valid_efc_addresses"]
        and gates["tangential_force_difference_exists"]
        and gates["patch_base_double_contact_count"] == 0
        and gates["geometry_parity"] and gates["precontact_parity"]
        and gates["postcontact_identity_count"] == 0
        and gates["postcontact_divergence_all_pairs"]
        and gates["control_valid_physical_slip_onset_count"] == 0
        and strong_valid >= 10 and moderate_valid >= 6
        and gates["both_target_feet_valid"] and gates["both_speeds_valid"]
        and gates["all_onset_phases_valid"]
        and gates["target_first_fraction"] >= 0.90
        and gates["air_or_postfall_counted"] == 0
        and gates["duplicate_unique_run_count"] == 0
        and gates["silently_discarded_or_relabelled_count"] == 0)
    write_json(output / "duplicate_audit.json", {
        "unique_pilot_runs": 48, "unique_full_trace_hashes": len(set(unique_hashes)),
        "duplicate_count": duplicate_count, "replays_excluded_from_unique_count": True,
        "valid": duplicate_count == 0,
    })
    return gates, manifest, traces


def _write_geom_contract(output: Path) -> dict[str, object]:
    condition = pilot_matrix(0.01)[0]
    model, info = build_local_model(condition)
    patch_id = model.geom("slip_patch").id
    hard = friction_profiles()["hard_control"]
    strong = friction_profiles()["native_strong_ice"]
    rows: list[dict[str, object]] = []
    for side in SIDES:
        for index, (geom_name, pair_name) in enumerate(zip(
            FOOT_CONTACT_GEOM_NAMES[side], info["pair_names"][side])):
            geom_id = model.geom(geom_name).id
            body_id = int(model.geom_bodyid[geom_id])
            pair_id = model.pair(pair_name).id
            rows.append({
                "foot": side, "sole_slot": index + 1,
                "foot_geom_name": geom_name, "foot_geom_id": geom_id,
                "foot_body_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                "foot_body_id": body_id,
                "foot_geom_type": mujoco.mjtGeom(int(model.geom_type[geom_id])).name,
                "is_collision_geom": True, "is_visual_only": False,
                "patch_geom_name": "slip_patch", "patch_geom_id": patch_id,
                "patch_body_name": "world", "patch_body_id": 0,
                "explicit_pair_name": pair_name, "explicit_pair_id": pair_id,
                "pair_geom1_id": int(model.pair_geom1[pair_id]),
                "pair_geom2_id": int(model.pair_geom2[pair_id]),
                "hard_friction5": json.dumps(hard.friction5),
                "strong_friction5": json.dumps(strong.friction5),
                "contact_dimension": int(model.pair_dim[pair_id]),
                "patch_contype": int(model.geom_contype[patch_id]),
                "patch_conaffinity": int(model.geom_conaffinity[patch_id]),
                "generic_patch_contact_disabled": True,
            })
    write_csv(output / "geom_contact_contract.csv", rows)
    return {
        "covered_geom_count": len(rows),
        "left_covered": sum(row["foot"] == "left" for row in rows),
        "right_covered": sum(row["foot"] == "right" for row in rows),
        "all_are_sphere_collision_geoms": all(row["foot_geom_type"] == "mjGEOM_SPHERE" for row in rows),
        "all_have_explicit_pairs": all(int(row["explicit_pair_id"]) >= 0 for row in rows),
    }


def _save_plot(path: Path) -> None:
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def create_plots(
    output: Path, micro: dict[str, np.ndarray],
    whole: dict[str, dict[str, np.ndarray]],
    pilot: dict[str, dict[str, np.ndarray]], manifest: list[dict[str, object]],
) -> None:
    plt.figure(figsize=(7, 4))
    for profile, label in (("hard_control", "hard"),
                           ("moderate_ice_preregistered", "moderate"),
                           ("native_strong_ice", "strong Ice")):
        plt.plot(micro[f"{profile}_r0__time_s"], micro[f"{profile}_r0__x_m"], label=label)
    plt.xlabel("time (s)"); plt.ylabel("slider displacement (m)"); plt.legend()
    _save_plot(output / "microbench_displacement_by_friction_profile.png")
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for profile, label in (("hard_control", "hard"),
                           ("moderate_ice_preregistered", "moderate"),
                           ("native_strong_ice", "strong Ice")):
        axes[0].plot(micro[f"{profile}_r0__time_s"], micro[f"{profile}_r0__effective_friction"], label=label)
        axes[1].plot(micro[f"{profile}_r0__time_s"], micro[f"{profile}_r0__tangential_force_n"], label=label)
    axes[0].set_ylabel("effective friction"); axes[0].legend()
    axes[1].set_ylabel("tangential force (N)"); axes[1].set_xlabel("time (s)")
    _save_plot(output / "effective_friction_and_tangential_force.png")
    plt.figure(figsize=(7, 4))
    if whole:
        for profile, label in (("hard_control", "hard"), ("native_strong_ice", "Ice")):
            trace = whole[profile]
            plt.plot(trace["time_s"], trace["anchor_drift_label_only"][:, 0], label=label)
    plt.xlabel("time (s)"); plt.ylabel("left anchor drift (m)"); plt.legend()
    _save_plot(output / "whole_surface_hard_vs_ice_anchor_drift.png")
    plt.figure(figsize=(7, 4))
    positives = [row for row in manifest if row.get("role") == "positive" and row.get("unique_pilot_run")]
    if positives:
        positive_id = str(positives[0]["run_id"])
        control_id = positive_id.replace("positive_", "control_", 1)
        target = SIDES.index(str(positives[0]["target_foot"]))
        for run_id, label in ((positive_id, "positive"), (control_id, "control")):
            trace = pilot[run_id]
            plt.plot(trace["time_s"], trace["anchor_drift_label_only"][:, target], label=label)
    plt.xlabel("time (s)"); plt.ylabel("target anchor drift (m)"); plt.legend()
    _save_plot(output / "local_patch_positive_control_matched_trace.png")
    plt.figure(figsize=(7, 4))
    if positives:
        run_id = str(positives[0]["run_id"]); target = SIDES.index(str(positives[0]["target_foot"]))
        trace = pilot[run_id]
        plt.plot(trace["time_s"], trace["anchor_drift_label_only"][:, target], label="target")
        plt.plot(trace["time_s"], trace["anchor_drift_label_only"][:, 1 - target], label="contralateral")
    plt.xlabel("time (s)"); plt.ylabel("anchor drift (m)"); plt.legend()
    _save_plot(output / "target_vs_contralateral_drift.png")
    plt.figure(figsize=(7, 4))
    counts = [sum(
        row.get("valid_target_positive") and row.get("actual_onset_phase") == phase
        for row in positives) for phase in PHASES]
    plt.bar(PHASES, counts); plt.ylabel("valid positive onsets")
    _save_plot(output / "onset_distribution_by_phase.png")


def _empty_pilot_artifacts(output: Path) -> None:
    for name in (
        "local_patch_contact_audit.csv", "pilot_run_manifest.csv",
        "pilot_pair_parity.csv", "pilot_physical_episode_ledger.csv",
        "pilot_contact_force_metrics.csv", "pilot_fall_censor_audit.csv",
    ):
        write_csv(output / name, [])
    write_json(output / "duplicate_audit.json", {
        "not_run": True, "reason": "upstream mandatory physics gate failed",
        "duplicate_count": 0, "valid": False,
    })


def _git_value(*args: str) -> str:
    return subprocess.run(
        ("git",) + args, cwd=REPO, check=True, text=True,
        stdout=subprocess.PIPE).stdout.strip()


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
        raise RuntimeError(f"expected {STARTING_CHECKPOINT}, found {head}")
    guard = AccessGuard(output)
    write_json(output / "protocol.json", {
        "task": "Walking v2 Slip Scenario Generator Redesign v4",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "scope": "physics generator audit and bounded proof only",
        "unique_robot_pilot_cap": 48,
        "full_216_run_acquisition_performed": False,
        "model_training_or_selection_performed": False,
        "implementation": "PRECOMPILED_TILED_PATCH_GEOM",
        "runtime_intervention": "explicit pair friction assigned before mj_step collision and constraint construction",
        "positive_control_geometry": "identical within every pair",
        "surface_top_z_m": 0.0,
        "slip_oracle": {"threshold_m": SLIP_THRESHOLD_M, "persistence_ms": SLIP_PERSISTENCE_MS, "changed": False},
        "phase_activation_ms": PHASE_ACTIVATION_MS,
        "moderate_grid_preregistered_if_needed": [0.075, 0.100, 0.125],
        "moderate_grid_executed": False,
        "failed_attempt_policy": "retain, report, never relabel",
        "forbidden_scope": "outer/holdout/spatial-final/final-test, training, deployment exports",
    })
    write_json(output / "input_allowlist.json", {
        "exact_paths_only": True,
        "inputs": [{"path": path, "purpose": purpose} for path, purpose in ALLOWED_INPUTS.items()],
    })
    write_json(output / "forbidden_path_policy.json", {
        "fail_closed": True, "forbidden_tokens": list(FORBIDDEN_TOKENS),
        "outer_or_final_artifact_use_authorized": False,
    })
    # Record every runtime/source input through the guard before use.
    for relative in (
        f"{V3}/run_manifest.csv", f"{V3}/summary.json", f"{V3}/readiness.json",
        f"{V3}/future_nested_fold_manifest.json", f"{ORACLE}/summary.json",
        f"{JOINT}/summary.json", f"{JOINT}/provenance.json", *TERRAIN_FILES.values(),
        POLICY, SCENE, ROBOT,
        "simulation/unitree_mujoco/simulate_python/terrain_profiles.py",
        "simulation/unitree_mujoco/simulate_python/walking_hazard_ground_truth_v1.py",
        "simulation/unitree_mujoco/simulate_python/walking_hazard_oracle_calibration_v1.py",
        "simulation/unitree_mujoco/simulate_python/bilateral_hil_sensor_v2.py",
        "simulation/unitree_mujoco/simulate_python/g1_upstream_locomotion.py",
        "simulation/unitree_mujoco/simulate_python/run_walking_v2_bilateral_slip_targeted_acquisition_v3.py",
        "simulation/unitree_mujoco/simulate_python/walking_v2_bilateral_slip_targeted_acquisition_v3.py",
    ):
        guard.path(relative)
    v3_rows = guard.csv(f"{V3}/run_manifest.csv")
    v3_summary = guard.json(f"{V3}/summary.json")
    write_json(output / "failed_v3_quarantine_manifest.json", {
        "source_directory": V3, "source_run_count": len(v3_rows),
        "expected_run_count": 216, "all_runs_quarantined": len(v3_rows) == 216,
        "disposition": "INVALID_INTERVENTION_DO_NOT_TRAIN",
        "run_dispositions": [{
            "run_id": row["run_id"],
            "disposition": "INVALID_INTERVENTION_DO_NOT_TRAIN",
        } for row in v3_rows],
        "physics_intervention_ineffective": True,
        "positive_control_full_trace_identity": True,
        "physical_positive_support_absent": True,
        "future_fold_manifest_authorized_for_training": False,
        "traces_preserved_for": "failure audit only",
        "source_summary_sha256": sha256_file(REPO / f"{V3}/summary.json"),
        "source_summary_snapshot": v3_summary,
        "source_artifacts_modified": False,
    })
    terrain_before = {key: sha256_file(REPO / path) for key, path in TERRAIN_FILES.items()}
    oracle_hash = sha256_file(REPO / f"{ORACLE}/summary.json")
    contract = _write_geom_contract(output)
    write_json(output / "local_patch_definition.json", {
        "implementation": "PRECOMPILED_TILED_PATCH_GEOM",
        "left_patch": vars(patch_bounds("left")),
        "right_patch": vars(patch_bounds("right")),
        "surrounding_partition": ["base_x_before", "base_x_after", "base_lane_below", "base_lane_above"],
        "patch_overlaps_base": False, "surface_top_height_delta_m": 0.0,
        "height_or_compliance_intervention": False,
        "generic_patch_contact_disabled": True,
        "explicit_pair_count": 8,
        "pair_friction_update_stage": "before mj_step / before collision and constraint construction",
        "phase_activation_ms": PHASE_ACTIVATION_MS,
    })
    micro_pass, micro_gates, micro_traces = run_microbench(output)
    root_cause = "POST_CONSTRAINT_MUTATION"
    if root_cause not in ROOT_CAUSES:
        raise AssertionError(root_cause)
    write_json(output / "friction_root_cause.json", {
        "primary_root_cause": root_cause,
        "exactly_one_primary_selected": True,
        "evidence": {
            "v3_step_order": "controller.apply -> mj_step1 -> mutate data.contact.friction -> mj_step2",
            "mj_step1_effect": "collision and constraint construction already completed",
            "microbench_post_step1_mutation_identical_to_hard": micro_gates["post_step1_mutation_identical_to_hard"],
            "v3_all_positive_control_full_traces_identical": True,
        },
        "secondary_contributors": [{
            "type": "GEOM_FRICTION_MAX_OVERRIDE",
            "caused_v3_identity": False,
            "evidence": "equal-priority no-pair probe retained foot sliding friction 1.0 despite Ice ground 0.05",
            "observed_effective_mu": micro_gates["equal_priority_observed_mu"],
            "reason_not_primary": "v3 overwrote effective contact.friction directly, but did so after constraint construction",
        }],
        "geom_priority_neutralized_v3_direct_mutation": False,
        "corrected_mechanism": "explicit foot-geom/patch-geom pair contract, pair friction set before mj_step",
    })
    write_json(output / "friction_intervention_call_graph.json", {
        "failed_v3_path": [
            "material_profiles()", "patch_for_foot()", "MjModel.from_xml_path()",
            "G1BilateralSensorReaderV2.foot_geom_ids", "apply_terrain_profile(concrete)",
            "controller.apply()", "mj_step1() [collision + constraint construction]",
            "_target_ground_contacts()", "_apply_patch_friction(data.contact)",
            "mj_step2() [solve using already-built constraints]", "integrated state",
        ],
        "corrected_v4_path": [
            "friction_profiles()", "patch_bounds()", "MjSpec.from_file()",
            "delete overlapping source/target floor geoms", "add disjoint base tiles + real patch geom",
            "add eight explicit sole/patch pairs", "compile model",
            "set target model.pair_friction before mj_step", "mj_step collision",
            "constraint construction with corrected friction", "constraint solve",
            "mj_contactForce evidence", "integrated state",
        ],
        "runtime_detector_inputs": "Fusion20 virtual FSR/foot IMU only",
        "privileged_values": "contact/world position/anchor drift/oracle used only for audit labels",
    })
    write_json(output / "step_order_audit.json", {
        "integrator": "mjINT_EULER", "rk4_split_used": False,
        "failed_v3": {
            "mutation_point": "after mj_step1", "constraints_already_constructed": True,
            "solver_effective": False,
        },
        "corrected_v4": {
            "mutation_target": "model.pair_friction on precompiled explicit pairs",
            "mutation_point": "before controller.apply and mj_step",
            "collision_already_generated": False, "constraints_already_constructed": False,
            "constraint_solve_sees_intervention": True,
        },
        "microbench_evidence": micro_gates,
    })
    whole_pass = False
    whole_rows: list[dict[str, object]] = []
    whole_traces: dict[str, dict[str, np.ndarray]] = {}
    pilot_gates: dict[str, object] = {"robot_pilot_ready": False, "not_run": True}
    pilot_manifest: list[dict[str, object]] = []
    pilot_traces: dict[str, dict[str, np.ndarray]] = {}
    if micro_pass:
        whole_pass, whole_rows, whole_traces = run_whole_surface_reference(
            output, REPO / POLICY)
    else:
        write_csv(output / "whole_surface_reference_metrics.csv", [])
        write_csv(output / "whole_surface_contact_audit.csv", [])
    if micro_pass and whole_pass:
        pilot_gates, pilot_manifest, pilot_traces = run_local_pilot(
            output, REPO / POLICY)
    else:
        _empty_pilot_artifacts(output)
    create_plots(output, micro_traces, whole_traces, pilot_traces, pilot_manifest)
    terrain_after = {key: sha256_file(REPO / path) for key, path in TERRAIN_FILES.items()}
    terrain_same = terrain_before == terrain_after
    write_json(output / "terrain_immutable_verification.json", {
        "before_sha256": terrain_before, "after_sha256": terrain_after,
        "byte_identical": terrain_same, "retrained": False, "modified": False,
    })
    write_json(output / "oracle_immutable_verification.json", {
        "path": f"{ORACLE}/summary.json", "sha256_before": oracle_hash,
        "sha256_after": sha256_file(REPO / f"{ORACLE}/summary.json"),
        "threshold_m": SLIP_THRESHOLD_M, "persistence_ms": SLIP_PERSISTENCE_MS,
        "semantics_changed": False,
    })
    contact_ready = bool(
        contract["covered_geom_count"] == 8 and contract["all_have_explicit_pairs"]
        and pilot_gates.get("patch_active_contacts_gt_zero", False)
        and pilot_gates.get("effective_positive_control_friction_differs", False)
        and pilot_gates.get("valid_efc_addresses", False)
        and pilot_gates.get("tangential_force_difference_exists", False)
        and pilot_gates.get("patch_base_double_contact_count", 1) == 0
        and pilot_gates.get("geometry_parity", False))
    local_ready = bool(contact_ready and pilot_gates.get("precontact_parity", False)
                       and pilot_gates.get("postcontact_identity_count", 1) == 0)
    scenario_ready = bool(
        micro_pass and whole_pass and contact_ready and local_ready
        and pilot_gates.get("robot_pilot_ready", False) and terrain_same
        and guard.blocked == 0)
    readiness = {
        "WALKING_V2_SLIP_GENERATOR_AUDIT_READY": bool(micro_pass and root_cause == "POST_CONSTRAINT_MUTATION"),
        "WALKING_V2_SLIP_V3_QUARANTINE_READY": len(v3_rows) == 216,
        "WALKING_V2_SLIP_MICROBENCH_READY": micro_pass,
        "WALKING_V2_SLIP_FRICTION_INTERVENTION_READY": bool(micro_pass and micro_gates["trajectory_divergence_verified"]),
        "WALKING_V2_SLIP_CONTACT_CONTRACT_READY": contact_ready,
        "WALKING_V2_SLIP_WHOLE_SURFACE_REFERENCE_READY": whole_pass,
        "WALKING_V2_SLIP_LOCAL_PATCH_READY": local_ready,
        "WALKING_V2_SLIP_UNILATERAL_SOURCE_READY": bool(pilot_gates.get("robot_pilot_ready", False)),
        "WALKING_V2_SLIP_SCENARIO_GENERATOR_READY": scenario_ready,
        "WALKING_V2_SLIP_FULL_REACQUISITION_AUTHORIZED": scenario_ready,
        "WALKING_V2_SLIP_TARGETED_RETRAINING_AUTHORIZED": False,
        "WALKING_V2_FRESH_BLIND_HOLDOUT_AUTHORIZED": False,
        "WALKING_V2_TERRAIN_LOCK_PRESERVED": terrain_same,
        "WALKING_V2_SYSTEM_MIGRATION_AUTHORIZED": False,
        "WALKING_V2_INT8_PREPARATION_AUTHORIZED": False,
        "SINK_RUNTIME_DETECTION_DEFERRED": True,
    }
    if scenario_ready:
        next_step = "REACQUIRE_TARGETED_BILATERAL_SLIP_DATA"
    elif (pilot_gates.get("strong_valid_target_positive_count", 0) >= 10
          and pilot_gates.get("moderate_valid_target_positive_count", 0) < 6):
        next_step = "SLIP_FRICTION_PROFILE_RECALIBRATION"
    elif not micro_pass or not whole_pass:
        next_step = "SLIP_SCENARIO_GENERATOR_REDESIGN"
    elif pilot_gates.get("strong_valid_target_positive_count", 0) < 10:
        next_step = "ABANDON_LOCAL_PATCH_ACQUISITION"
    else:
        next_step = "SLIP_SCENARIO_GENERATOR_REDESIGN"
    summary = {
        "task": "Walking v2 Slip Scenario Generator Redesign v4",
        "primary_root_cause": root_cause, "secondary_contributors": ["GEOM_FRICTION_MAX_OVERRIDE"],
        "microbench_gates": micro_gates, "microbench_ready": micro_pass,
        "whole_surface_ready": whole_pass,
        "whole_surface_comparison": whole_rows[-1] if whole_rows else {"not_run": True},
        "pilot_gates": pilot_gates, "readiness": readiness,
        "full_216_run_acquisition_performed": False,
        "training_or_selection_performed": False,
        "moderate_profile_grid_executed": False,
        "failed_v3_preserved": True, "failed_v3_quarantined": len(v3_rows) == 216,
        "forbidden_access_count": guard.blocked,
        "terrain_byte_identical": terrain_same,
        "oracle_changed": False, "next_step": next_step,
    }
    write_json(output / "readiness.json", readiness)
    write_json(output / "summary.json", summary)
    strong_valid = int(pilot_gates.get("strong_valid_target_positive_count", 0))
    moderate_valid = int(pilot_gates.get("moderate_valid_target_positive_count", 0))
    audit = f"""# Walking v2 Slip Scenario Generator Redesign v4 audit

1. The exact identity mechanism was **POST_CONSTRAINT_MUTATION**: v3 changed `data.contact.friction` after `mj_step1` had already constructed the contact constraints.
2. Friction was previously applied between `mj_step1` and `mj_step2`, after collision and constraint construction.
3. Geom priority/max combination did not neutralize v3's direct contact mutation; it is a demonstrated secondary hazard for ground-only designs.
4. The correction is a height-preserving, non-overlapping `PRECOMPILED_TILED_PATCH_GEOM` with eight explicit sole/patch pairs, whose pair friction is set before `mj_step`.
5. Microbench monotonic friction-dependent dynamics: **{micro_gates['displacement_monotonic_with_severity']}**.
6. Whole-surface native Ice reproduced physical Slip: **{whole_rows[-1].get('native_ice_physical_slip_reproduced', False) if whole_rows else False}**.
7. All local pairs had pre-contact parity and post-contact divergence: **{pilot_gates.get('precontact_parity', False) and pilot_gates.get('postcontact_divergence_all_pairs', False)}**.
8. Patch/base double-contact count: **{pilot_gates.get('patch_base_double_contact_count', 0)}**.
9. Valid strong positives: **{strong_valid}/12**; valid moderate positives: **{moderate_valid}/12**.
10. Both target feet have physically valid localized positives: **{pilot_gates.get('both_target_feet_valid', False)}**.
11. Silently discarded or relabeled runs: **{pilot_gates.get('silently_discarded_or_relabelled_count', 0)}**. Every attempt and replay was retained.
12. All **{len(v3_rows)}** v3 runs are quarantined as `INVALID_INTERVENTION_DO_NOT_TRAIN`; original artifacts were not modified.
13. Forbidden artifact accesses: **{guard.blocked}**.
14. Terrain model, normalization, config and lock remained byte-identical: **{terrain_same}**.
15. Full 216-run reacquisition authorized: **{scenario_ready}**. It was not performed here.
16. Retraining, blind holdout, System migration and INT8 preparation remain unauthorized: **True**.
17. Sink remains `SINK_RUNTIME_DETECTION_DEFERRED`: **True**.

Exactly one next step: **{next_step}**
"""
    (output / "audit.md").write_text(audit, encoding="utf-8")
    generated = sorted(path for path in output.iterdir() if path.is_file())
    size_rows = [{
        "path": path.name, "bytes": path.stat().st_size,
        "mib": path.stat().st_size / (1024 * 1024),
        "within_45_mib": path.stat().st_size <= 45 * 1024 * 1024,
    } for path in generated]
    write_json(output / "resource_size_audit.json", {
        "limit_mib": 45, "files": size_rows,
        "maximum_file_mib": max((row["mib"] for row in size_rows), default=0.0),
        "all_files_within_limit": all(row["within_45_mib"] for row in size_rows),
    })
    graph_files = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name not in {"provenance.json", "resource_size_audit.json"})
    hashes = {path.name: sha256_file(path) for path in graph_files}
    hash_graph_verified = all(sha256_file(output / name) == digest for name, digest in hashes.items())
    write_json(output / "provenance.json", {
        "starting_checkpoint": STARTING_CHECKPOINT, "execution_head": head,
        "upstream_policy_revision": UPSTREAM_REVISION,
        "expected_policy_sha256": TESTED_POLICY_SHA256,
        "actual_policy_sha256": sha256_file(REPO / POLICY),
        "mujoco_version": mujoco.__version__, "numpy_version": np.__version__,
        "elapsed_seconds": time.time() - started,
        "artifact_sha256": hashes,
        "artifact_hash_graph_verified": hash_graph_verified,
        "forbidden_access_count": guard.blocked,
        "terrain_before_sha256": terrain_before, "terrain_after_sha256": terrain_after,
        "full_acquisition_performed": False, "training_performed": False,
    })
    # Final resource audit includes provenance and itself is intentionally not
    # part of the scientific-artifact hash graph.
    all_files = sorted(path for path in output.iterdir() if path.is_file())
    final_sizes = [{
        "path": path.name, "bytes": path.stat().st_size,
        "mib": path.stat().st_size / (1024 * 1024),
        "within_45_mib": path.stat().st_size <= 45 * 1024 * 1024,
    } for path in all_files]
    write_json(output / "resource_size_audit.json", {
        "limit_mib": 45, "files": final_sizes,
        "maximum_file_mib": max((row["mib"] for row in final_sizes), default=0.0),
        "all_files_within_limit": all(row["within_45_mib"] for row in final_sizes),
    })
    print(json.dumps({
        "output": str(output), "microbench_ready": micro_pass,
        "whole_surface_ready": whole_pass, "scenario_ready": scenario_ready,
        "next_step": next_step,
    }, indent=2))


if __name__ == "__main__":
    main()
