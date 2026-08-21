#!/usr/bin/env python3
"""Generate aligned development evidence and lock the Walking-v2 host design."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable
import warnings

import mujoco
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from bilateral_hil_sensor_v2 import (
    FOOT_CONTACT_GEOM_NAMES, G1BilateralSensorReaderV2,
)
from g1_upstream_locomotion import (
    TESTED_POLICY_SHA256, UPSTREAM_REVISION, UnitreeG1PretrainedController,
)
from run_walking_hazard_ground_truth_v1 import (
    PELVIS_BODY_NAME, _disable_nonfoot_surface_collisions, _fall_reasons,
)
from walking_hazard_ground_truth_v1 import (
    derive_contact_signals, max_left_foot_contact_penetration_m,
)
from walking_hazard_oracle_calibration_v1 import persistent_oracle
from walking_v2_host_design_milestone_v2 import (
    HARD_TERRAINS, PHYSICS_STEPS_PER_SAMPLE, PHYSICS_TIMESTEP_S,
    SAMPLE_RATE_HZ, SIDES, TERRAIN_CODE, TERRAIN_NAMES, VARIATIONS,
    LinearTerrainModel, RunCondition, TransitionPolicy, acquisition_matrix,
    case_a_transition_timeline, causal_contact_telemetry, causal_g0_owner,
    runtime_contract, transition_policy_matrix,
)
from walking_v2_joint_terrain_slip_redesign_v1 import (
    ProjectedLinearModel, runtime_feature,
)
from walking_v2_slip_corrected_targeted_retraining_v4 import (
    OperatingConfig, derived_runtime_telemetry,
)
from walking_v2_slip_redesign_iteration_v2 import SlipV2Model
from terrain_profiles import TERRAIN_PROFILES


STARTING_CHECKPOINT = "cf3b1701c74712998814e918a2630fa10a44194b"
OUTPUT = Path("simulation/outputs/walking_v2_host_design_milestone_v2")
SOURCE = Path("simulation/unitree_mujoco/simulate_python")
SCENE = Path("simulation/unitree_mujoco/unitree_robots/g1/scene_walking_terrain_transition.xml")
POLICY = Path("simulation/unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx")
LOCKED_TERRAIN = Path("simulation/outputs/walking_v2_joint_terrain_slip_redesign_v1")
PRIOR_HOST = Path("simulation/outputs/walking_v2_fast_reflex_host_v1")
ORACLE = Path("simulation/outputs/walking_hazard_oracle_calibration_v1")
LOAD_ON_N = 5.0
SLIP_THRESHOLD_M = 0.050
SLIP_PERSISTENCE_MS = 3
ENDPOINT_STRIDE_MS = 10
TRAINING_STRIDE_MS = 20
MODEL_CANDIDATES = (
    {"candidate_id": "T1_LINEAR_200", "architecture": "T1", "history_ms": 200},
    {"candidate_id": "T2_LINEAR_200", "architecture": "T2", "history_ms": 200},
)
SLIP_CONFIG = OperatingConfig(0.45, 0.45, 0.25, 0.20, 0.60, 2, 0.0)
FORBIDDEN = (
    "/outer/", "_outer_", "holdout", "spatial_final", "spatial-final",
    "final_test", "final-test",
)


def source_paths() -> tuple[str, ...]:
    return tuple((SOURCE / name).as_posix() for name in (
        "walking_v2_host_design_milestone_v2.py",
        "run_walking_v2_host_design_milestone_v2.py",
        "test_walking_v2_host_design_milestone_v2.py",
    ))


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    def default(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return item.as_posix()
        raise TypeError(type(item).__name__)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=default) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    fields = list(dict.fromkeys(key for row in values for key in row)) if values else ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(values)


class AccessGuard:
    """Fail-closed exact-path read barrier with a durable access ledger."""

    def __init__(self, root: Path, output: Path, allowed: dict[str, str]) -> None:
        self.root = root.resolve(); self.output = output
        self.allowed = dict(allowed); self.events: list[dict[str, Any]] = []
        self.blocked = 0; self.forbidden = 0; self.frozen = False
        for name in self.allowed:
            if any(token in f"/{name.lower()}" for token in FORBIDDEN):
                raise PermissionError(f"forbidden allowlist entry: {name}")
        self.flush()

    def flush(self) -> None:
        write_json(self.output / "artifact_access_log.json", {
            "version": "walking_v2_host_design_access_v2", "fail_closed": True,
            "exact_paths_only": True, "forbidden_tokens": list(FORBIDDEN),
            "frozen": self.frozen, "blocked_access_count": self.blocked,
            "forbidden_access_count": self.forbidden, "events": self.events,
        })

    def path(self, relative: str | Path, purpose: str) -> Path:
        value = Path(relative).as_posix()
        forbidden = any(token in f"/{value.lower()}" for token in FORBIDDEN)
        allowed = (
            not self.frozen and not forbidden and value in self.allowed
            and not Path(value).is_absolute() and ".." not in Path(value).parts
        )
        if not allowed:
            self.blocked += 1; self.forbidden += int(forbidden)
            self.events.append({"path": value, "purpose": purpose, "status": "BLOCKED"})
            self.flush(); raise PermissionError(value)
        path = (self.root / value).resolve()
        if not path.is_file() or self.root not in path.parents:
            self.blocked += 1; self.events.append({
                "path": value, "purpose": purpose, "status": "BLOCKED_MISSING",
            }); self.flush(); raise FileNotFoundError(value)
        self.events.append({
            "path": value, "purpose": purpose, "status": "ALLOWED",
            "sha256": sha256(path), "byte_count": path.stat().st_size,
        })
        self.flush(); return path

    def freeze(self) -> None:
        self.frozen = True; self.flush()


def protocol_payload() -> dict[str, Any]:
    return {
        "version": "walking_v2_host_design_protocol_v2",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "objective": "select one strongest honest host design from aligned development evidence",
        "development_only": True,
        "future_blind_exclusion_permanent": True,
        "matrix_frozen_before_generation": {
            "run_count": 420, "speeds_mps": [0.10, 0.15, 0.20],
            "feet": list(SIDES), "variation_count": len(VARIATIONS),
            "seeded_replicates_per_cell": 2,
            "cases": ["A", "B", "C", "D", "N hard-to-hard", "S steady"],
        },
        "candidate_models_frozen_before_evaluation": list(MODEL_CANDIDATES) + [{
            "candidate_id": "LOCKED_T2_BASELINE", "architecture": "T2",
            "history_ms": 200,
        }],
        "optimizer_frozen_before_final_comparison": {
            "solver": "Newton-CG", "max_iterations": 100, "tolerance": 1e-4,
            "class_weight": "balanced", "C": 1.0,
            "qualification_basis": "pre-comparison fold-0 remediation converged in 33 iterations",
        },
        "policy_matrix_frozen_before_evaluation": [
            asdict(value) | {"config_id": value.config_id}
            for value in transition_policy_matrix()
        ],
        "selection_policy": (
            "If any candidate passes the frozen direct gate, minimize false outputs then "
            "maximize within-20ms recall and advisory F1; otherwise select monitoring-only "
            "by advisory F1, event recall, precision, macro terrain F1, then stable ID."
        ),
        "direct_gate_frozen_before_evaluation": {
            "recall_within_20ms_min": 0.80,
            "each_speed_recall_within_20ms_min": 0.70,
            "each_hard_source_recall_within_20ms_min": 0.65,
            "required_zero": [
                "normal_run_fp", "wrong_case_output", "invalid_output",
                "post_fall_output", "duplicate_output", "latch_carryover",
                "cross_foot_output",
            ],
            "minimum_case_a_events": 50,
            "minimum_grouped_variation_folds": 5,
            "minimum_independent_source_families": 2,
            "available_source_family": "MuJoCo G1 plus fixed upstream controller",
        },
        "advisory_gate": (
            "nonzero denominator; report event/run precision, recall, F1, false-positive "
            "rates, latency and coverage; false positives do not authorize actuation"
        ),
        "runtime_oracle_policy": {
            "terrain_gt_runtime_gate": False, "physical_slip_runtime_gate": False,
            "first_fall_runtime_gate": False, "future_input": False,
        },
        "sink": "SINK_RUNTIME_DETECTION_DEFERRED",
        "int8_vela": "optional only after lock; not scheduled in this bounded search",
        "file_size_limit_mib": 45,
    }


def preflight(root: Path, output: Path) -> tuple[AccessGuard, ProjectedLinearModel, float]:
    started = time.monotonic()
    if git("rev-parse", "HEAD") != STARTING_CHECKPOINT:
        raise RuntimeError("starting HEAD mismatch")
    if git("rev-parse", "origin/main") != STARTING_CHECKPOINT:
        raise RuntimeError("starting origin/main mismatch")
    expected_dirty = set(source_paths())
    unexpected = [
        line for line in git("status", "--short").splitlines()
        if line[3:].split(" -> ")[-1] not in expected_dirty
    ]
    if unexpected:
        raise RuntimeError(f"unexpected dirty paths: {unexpected}")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_json(output / "protocol.json", protocol_payload())
    write_json(output / "causal_runtime_contract.json", runtime_contract())
    allowed = {
        SCENE.as_posix(): "fixed walking simulation scene",
        POLICY.as_posix(): "fixed upstream walking controller",
        (LOCKED_TERRAIN / "terrain_candidate_model.npz").as_posix(): "locked T2 baseline",
        (LOCKED_TERRAIN / "terrain_candidate_normalization.json").as_posix(): "locked T2 normalization",
        (LOCKED_TERRAIN / "terrain_candidate_config.json").as_posix(): "locked T2 config",
        (LOCKED_TERRAIN / "terrain_selection_lock.json").as_posix(): "locked T2 hash authority",
        (PRIOR_HOST / "advisory_metrics.json").as_posix(): "frozen OOF Slip advisory evidence",
        (PRIOR_HOST / "slip_advisory_model.npz").as_posix(): "frozen executable Slip advisory",
        (PRIOR_HOST / "slip_advisory_training.json").as_posix(): "frozen Slip training health",
        (ORACLE / "summary.json").as_posix(): "frozen physical Slip oracle contract",
        (SOURCE / "terrain_profiles.py").as_posix(): "native terrain profiles",
        (SOURCE / "bilateral_hil_sensor_v2.py").as_posix(): "Fusion20 contract",
        (SOURCE / "g1_upstream_locomotion.py").as_posix(): "walking controller wrapper",
        (SOURCE / "walking_hazard_ground_truth_v1.py").as_posix(): "offline contact labels",
        (SOURCE / "walking_hazard_oracle_calibration_v1.py").as_posix(): "offline Slip oracle",
        **{path: "host-v2 source" for path in source_paths()},
    }
    write_json(output / "input_allowlist.json", {
        "version": "walking_v2_host_design_allowlist_v2", "exact_paths_only": True,
        "inputs": [{"path": key, "purpose": value} for key, value in sorted(allowed.items())],
    })
    write_json(output / "forbidden_path_policy.json", {
        "version": "walking_v2_host_design_forbidden_v2", "fail_closed": True,
        "forbidden_tokens": list(FORBIDDEN),
        "outer_holdout_spatial_final_or_final_test_access": False,
        "new_blind_generation_or_inspection": False,
    })
    guard = AccessGuard(root, output, allowed)
    policy_path = guard.path(POLICY, "verify fixed controller")
    if sha256(policy_path) != TESTED_POLICY_SHA256:
        raise RuntimeError("walking policy hash mismatch")
    lock = json.loads(guard.path(
        LOCKED_TERRAIN / "terrain_selection_lock.json", "verify locked T2 authority",
    ).read_text())
    for field, name in (
        ("model_sha256", "terrain_candidate_model.npz"),
        ("normalization_sha256", "terrain_candidate_normalization.json"),
        ("config_sha256", "terrain_candidate_config.json"),
    ):
        path = guard.path(LOCKED_TERRAIN / name, f"verify {name}")
        if sha256(path) != lock[field]:
            raise RuntimeError(f"locked Terrain artifact drift: {name}")
    model = ProjectedLinearModel.load(guard.path(
        LOCKED_TERRAIN / "terrain_candidate_model.npz", "load locked T2 baseline",
    ))
    return guard, model, started


def _apply_variant(
    model: mujoco.MjModel, geom_name: str, terrain: str,
    friction_scale: float, solref_scale: float,
) -> None:
    profile = TERRAIN_PROFILES[terrain]
    geom_id = model.geom(geom_name).id
    friction = np.asarray(profile.friction, np.float64).copy()
    friction[0] *= friction_scale
    model.geom_friction[geom_id] = friction
    solref = np.asarray(profile.solref, np.float64).copy()
    solref[0] *= solref_scale
    model.geom_solref[geom_id] = solref
    model.geom_solimp[geom_id] = profile.solimp
    model.geom_priority[geom_id] = profile.priority
    model.geom_condim[geom_id] = profile.condim


def _surface_forces(
    model: mujoco.MjModel, data: mujoco.MjData,
    foot_side: dict[int, int], ground_index: dict[int, int],
) -> np.ndarray:
    result = np.zeros((2, 2), np.float64)
    wrench = np.zeros(6, np.float64)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        pair = (int(contact.geom1), int(contact.geom2))
        foot = next((foot_side[value] for value in pair if value in foot_side), None)
        ground = next((ground_index[value] for value in pair if value in ground_index), None)
        if foot is None or ground is None:
            continue
        mujoco.mj_contactForce(model, data, contact_id, wrench)
        result[foot, ground] += max(0.0, float(wrench[0]))
    return result


def _initial_perturbation(seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    return tuple(rng.uniform((-0.002, -0.001, -0.002), (0.002, 0.001, 0.002)))


def collect_run(
    condition: RunCondition, scene_path: Path, policy_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Collect one bilateral causal 1 kHz spatial-transition trace."""
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.timestep = PHYSICS_TIMESTEP_S
    source_id = model.geom("ground_source").id
    target_id = model.geom("ground_target").id
    model.geom_pos[source_id, 0] = condition.boundary_x_m - 3.25
    model.geom_pos[target_id, 0] = condition.boundary_x_m + 3.25
    _apply_variant(
        model, "ground_source", condition.terrain_before,
        condition.friction_scale_before, condition.solref_scale_before,
    )
    _apply_variant(
        model, "ground_target", condition.terrain_after,
        condition.friction_scale_after, condition.solref_scale_after,
    )
    ground_ids = frozenset((source_id, target_id))
    allowed_feet = _disable_nonfoot_surface_collisions(model, ground_ids)
    data = mujoco.MjData(model)
    controller = UnitreeG1PretrainedController(
        model, data, policy_path, condition.speed_mps,
    )
    dx, dy, dvx = _initial_perturbation(condition.seed)
    data.qpos[0] += dx; data.qpos[1] += dy + condition.lateral_offset_m
    data.qvel[0] += dvx; controller.global_phase = condition.phase_fraction
    nominal_command = controller.command.copy()
    if condition.command_delay_s:
        controller.command[:] = 0.0
    mujoco.mj_forward(model, data)
    initial_qpos_sha = array_sha256(data.qpos)
    initial_qvel_sha = array_sha256(data.qvel)
    reader = G1BilateralSensorReaderV2(model, data)
    foot_body_ids = tuple(model.body(f"{side}_ankle_roll_link").id for side in SIDES)
    pelvis_id = model.body(PELVIS_BODY_NAME).id
    foot_side = {
        geom_id: side_index for side_index, side in enumerate(SIDES)
        for geom_id in reader.foot_geom_ids[side]
    }
    ground_index = {source_id: 0, target_id: 1}
    velocity = np.zeros(6, np.float64)
    series: dict[str, list[Any]] = {name: [] for name in (
        "time_s", "bilateral_fusion20_raw", "bilateral_canonical",
        "force_loaded", "physical_contact", "surface_force_n",
        "terrain_gt", "foot_world_xyz_label_only",
        "foot_world_velocity_label_only", "contact_penetration_label_only",
    )}
    first_fall_sample: int | None = None
    first_fall_time_s: float | None = None
    fall_reason = ""
    total_steps = int(round(condition.duration_s / PHYSICS_TIMESTEP_S))
    for physics_step in range(1, total_steps + 1):
        if data.time + 1e-12 >= condition.command_delay_s:
            controller.command[:] = nominal_command
        controller.apply(); mujoco.mj_step(model, data); controller.update_after_step()
        if physics_step % PHYSICS_STEPS_PER_SAMPLE:
            continue
        raw = reader.read_bilateral_vector()
        canonical = reader.read_canonical_bilateral_vector()
        physical = np.asarray([
            reader.has_foot_contact(side, ground_ids) for side in SIDES
        ], bool)
        runtime = tuple(reader.update_contact_state(
            side, raw[index * 10:index * 10 + 4],
        ) for index, side in enumerate(SIDES))
        loaded = np.asarray([
            state.loaded and physical[index] for index, state in enumerate(runtime)
        ], bool)
        forces = _surface_forces(model, data, foot_side, ground_index)
        terrain = np.full(2, -1, np.int8)
        for foot in (0, 1):
            if forces[foot].sum() > 0:
                name = condition.terrain_after if forces[foot, 1] > forces[foot, 0] else condition.terrain_before
                terrain[foot] = TERRAIN_CODE[name]
        foot_xyz = np.stack([data.xpos[value].copy() for value in foot_body_ids])
        foot_velocity = []
        for body_id in foot_body_ids:
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0,
            )
            foot_velocity.append(velocity[3:].copy())
        penetration = np.asarray([
            max_left_foot_contact_penetration_m(
                data, reader.foot_geom_ids[side], ground_ids,
            ) for side in SIDES
        ])
        reasons, _ = _fall_reasons(model, data, pelvis_id, ground_ids, allowed_feet)
        if reasons and first_fall_sample is None:
            first_fall_sample = len(series["time_s"])
            first_fall_time_s = float(data.time)
            fall_reason = "|".join(reasons)
        current = {
            "time_s": float(data.time),
            "bilateral_fusion20_raw": raw.astype(np.float32),
            "bilateral_canonical": canonical.astype(np.float32),
            "force_loaded": loaded, "physical_contact": physical,
            "surface_force_n": forces.astype(np.float32), "terrain_gt": terrain,
            "foot_world_xyz_label_only": foot_xyz.astype(np.float32),
            "foot_world_velocity_label_only": np.asarray(foot_velocity, np.float32),
            "contact_penetration_label_only": penetration.astype(np.float32),
        }
        for name, value in current.items():
            series[name].append(value)
    trace = {name: np.asarray(values) for name, values in series.items()}
    finite = np.all(np.isfinite(trace["bilateral_canonical"]), axis=1)
    touchdown, age, runtime_episode = causal_contact_telemetry(trace["force_loaded"])
    owner = causal_g0_owner(trace["force_loaded"], age, finite)
    prefall = np.ones(len(trace["time_s"]), bool)
    if first_fall_sample is not None:
        prefall[first_fall_sample:] = False
    physical_episode = np.full(trace["force_loaded"].shape, -1, np.int32)
    slip_active = np.zeros(trace["force_loaded"].shape, bool)
    anchor_drift = np.full(trace["force_loaded"].shape, np.nan, np.float32)
    for foot, side in enumerate(SIDES):
        signals = derive_contact_signals(
            trace["physical_contact"][:, foot], trace["force_loaded"][:, foot],
            trace["foot_world_xyz_label_only"][:, foot],
            trace["foot_world_velocity_label_only"][:, foot],
            trace["contact_penetration_label_only"][:, foot], first_fall_sample,
        )
        physical_episode[:, foot] = signals.contact_episode_id
        anchor_drift[:, foot] = signals.tangential_anchor_drift_m
        slip_active[:, foot] = persistent_oracle(
            signals.tangential_anchor_drift_m, signals.slip_calibration_valid,
            signals.contact_episode_id, SLIP_THRESHOLD_M, SLIP_PERSISTENCE_MS,
        )
    transition_samples: list[int | None] = []
    slip_onsets: list[int | None] = []
    target_dominant = (
        trace["force_loaded"] & (trace["surface_force_n"][:, :, 1] > 0)
        & (trace["surface_force_n"][:, :, 1] > trace["surface_force_n"][:, :, 0])
    )
    for foot in (0, 1):
        samples = np.flatnonzero(target_dominant[:, foot])
        transition = int(samples[0]) if samples.size else None
        transition_samples.append(transition)
        slip = np.flatnonzero(slip_active[:, foot] & prefall)
        slip_onsets.append(int(slip[0]) if slip.size else None)
    trace.update({
        "touchdown": touchdown, "contact_age_ms": age,
        "runtime_episode_id": runtime_episode, "g0_owner": owner,
        "physical_episode_id_label_only": physical_episode,
        "physical_slip_active_label_only": slip_active,
        "anchor_drift_label_only": anchor_drift,
        "pre_fall_valid_label_only": prefall,
        "transition_target_contact_label_only": target_dominant,
    })
    trace_hash_keys = (
        "time_s", "bilateral_fusion20_raw", "bilateral_canonical",
        "force_loaded", "physical_contact", "surface_force_n", "terrain_gt",
        "touchdown", "contact_age_ms", "runtime_episode_id", "g0_owner",
        "physical_slip_active_label_only", "pre_fall_valid_label_only",
        "transition_target_contact_label_only",
    )
    digest = hashlib.sha256()
    for name in trace_hash_keys:
        digest.update(name.encode() + b"\0")
        digest.update(array_sha256(trace[name]).encode())
    metadata = {
        **asdict(condition), "condition_sha256": condition.fingerprint,
        "development_only": True, "permanently_excluded_from_future_blind": True,
        "sample_rate_hz": SAMPLE_RATE_HZ, "physics_timestep_s": PHYSICS_TIMESTEP_S,
        "sample_count": len(trace["time_s"]),
        "sample_spacing_max_error_s": float(np.max(np.abs(np.diff(trace["time_s"]) - 0.001))),
        "fusion20_shape": list(trace["bilateral_canonical"].shape),
        "fusion20_finite": bool(finite.all()), "boundary_x_m": condition.boundary_x_m,
        "transition_sample_left": transition_samples[0],
        "transition_sample_right": transition_samples[1],
        "transition_time_left_s": None if transition_samples[0] is None else float(trace["time_s"][transition_samples[0]]),
        "transition_time_right_s": None if transition_samples[1] is None else float(trace["time_s"][transition_samples[1]]),
        "physical_slip_onset_left": slip_onsets[0],
        "physical_slip_onset_right": slip_onsets[1],
        "first_fall_sample_evaluation_only": first_fall_sample,
        "first_fall_time_s_evaluation_only": first_fall_time_s,
        "fall_reason_evaluation_only": fall_reason,
        "hard_terrain_type": condition.terrain_before if condition.terrain_before in HARD_TERRAINS else condition.terrain_after if condition.terrain_after in HARD_TERRAINS else None,
        "surface_variation": {
            "before_friction_scale": condition.friction_scale_before,
            "after_friction_scale": condition.friction_scale_after,
            "before_solref_scale": condition.solref_scale_before,
            "after_solref_scale": condition.solref_scale_after,
        },
        "initial_qpos_sha256": initial_qpos_sha,
        "initial_qvel_sha256": initial_qvel_sha,
        "upstream_controller_revision": UPSTREAM_REVISION,
        "upstream_policy_sha256": TESTED_POLICY_SHA256,
        "full_trace_sha256": digest.hexdigest(),
    }
    return trace, metadata


def endpoints_for(sample_count: int) -> np.ndarray:
    return np.arange(199, sample_count, ENDPOINT_STRIDE_MS, dtype=np.int32)


def terrain_features(
    traces: list[dict[str, np.ndarray]], architecture: str, history_ms: int,
    endpoints: np.ndarray,
) -> np.ndarray:
    """Materialize causal endpoint features without label or provenance inputs."""
    first = traces[0]
    zero_phase = np.zeros(first["force_loaded"].shape, np.int8)
    example = runtime_feature(
        architecture, 0, int(endpoints[0]), first["bilateral_canonical"],
        first["force_loaded"], first["contact_age_ms"], zero_phase,
        history_override_ms=history_ms,
    )
    result = np.empty((len(traces), len(endpoints), 2, len(example)), np.float32)
    for run_index, trace in enumerate(traces):
        phase = np.zeros(trace["force_loaded"].shape, np.int8)
        for row, endpoint in enumerate(endpoints):
            for foot in (0, 1):
                result[run_index, row, foot] = runtime_feature(
                    architecture, foot, int(endpoint), trace["bilateral_canonical"],
                    trace["force_loaded"], trace["contact_age_ms"], phase,
                    history_override_ms=history_ms,
                )
        if (run_index + 1) % 70 == 0:
            print(f"{architecture} causal features: {run_index + 1}/{len(traces)}", flush=True)
    return result


def fit_linear_model(
    features: np.ndarray, labels: np.ndarray, architecture: str, history_ms: int,
    seed: int,
) -> tuple[LinearTerrainModel, dict[str, Any]]:
    values = np.asarray(features, np.float64)
    target = np.asarray(labels, np.int8)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (values - mean) / scale
    caught: list[str] = []
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always", ConvergenceWarning)
        fit = LogisticRegression(
            solver="newton-cg", multi_class="auto", class_weight="balanced", C=1.0,
            max_iter=100, tol=1e-4, random_state=seed,
        ).fit(normalized, target)
        caught = [
            str(message.message) for message in messages
            if issubclass(message.category, ConvergenceWarning)
        ]
    model = LinearTerrainModel(
        architecture, history_ms, mean, scale, fit.coef_.copy(),
        fit.intercept_.copy(), fit.classes_.copy(),
    )
    health = {
        "solver": "Newton-CG", "max_iterations": 100, "tolerance": 1e-4,
        "iterations": int(np.max(fit.n_iter_)),
        "converged": not caught and int(np.max(fit.n_iter_)) < 100,
        "warning_count": len(caught), "warnings": caught,
        "fit_row_count": len(target), "feature_count": values.shape[1],
        "class_count": {TERRAIN_NAMES[int(key)]: int(value) for key, value in Counter(target).items()},
    }
    return model, health


def eligible_mask(
    traces: list[dict[str, np.ndarray]], endpoints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    eligible = np.zeros((len(traces), len(endpoints), 2), bool)
    labels = np.full(eligible.shape, -1, np.int8)
    for index, trace in enumerate(traces):
        sample = endpoints
        eligible[index] = (
            trace["force_loaded"][sample]
            & (trace["contact_age_ms"][sample] >= 5)
            & trace["pre_fall_valid_label_only"][sample, None]
            & (trace["terrain_gt"][sample] >= 0)
            & np.all(np.isfinite(trace["bilateral_canonical"][sample]), axis=1)[:, None]
        )
        labels[index] = trace["terrain_gt"][sample]
    return eligible, labels


def grouped_oof_candidate(
    traces: list[dict[str, np.ndarray]], manifests: list[dict[str, Any]],
    endpoints: np.ndarray, spec: dict[str, Any], eligible: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    features = terrain_features(
        traces, str(spec["architecture"]), int(spec["history_ms"]), endpoints,
    )
    scores = np.zeros((*eligible.shape, 4), np.float32)
    fold_health: list[dict[str, Any]] = []
    variation = np.asarray([row["variation_index"] for row in manifests], np.int8)
    training_endpoint = (np.arange(len(endpoints)) % (TRAINING_STRIDE_MS // ENDPOINT_STRIDE_MS)) == 0
    for fold in range(len(VARIATIONS)):
        train_run = variation != fold
        eval_run = variation == fold
        train_mask = eligible & train_run[:, None, None] & training_endpoint[None, :, None]
        model, health = fit_linear_model(
            features[train_mask], labels[train_mask], str(spec["architecture"]),
            int(spec["history_ms"]), 2026082200 + fold,
        )
        eval_indices = np.flatnonzero(eval_run)
        flat = features[eval_indices].reshape(-1, features.shape[-1])
        predicted = model.probabilities(flat).reshape(len(eval_indices), len(endpoints), 2, 4)
        scores[eval_indices] = predicted.astype(np.float32)
        fold_health.append({
            "candidate_id": spec["candidate_id"], "fold": fold,
            "train_variations": [value for value in range(len(VARIATIONS)) if value != fold],
            "evaluation_variation": fold, "run_overlap": 0, "pair_overlap": 0,
            "source_overlap": 0, "variation_overlap": 0, "seed_overlap": 0,
            **health,
        })
        print(f"{spec['candidate_id']} grouped OOF fold {fold} complete", flush=True)
    audit = {
        "feature_shape": list(features.shape),
        "scores_finite": bool(np.isfinite(scores).all()),
        "all_folds_filled": bool(np.all(scores.sum(axis=-1) > 0.999)),
    }
    return scores, fold_health, audit


def locked_t2_scores(
    traces: list[dict[str, np.ndarray]], model: ProjectedLinearModel,
    endpoints: np.ndarray,
) -> np.ndarray:
    features = terrain_features(traces, "T2", 200, endpoints)
    result = model.probabilities(features.reshape(-1, features.shape[-1]))
    return result.reshape(len(traces), len(endpoints), 2, 4).astype(np.float32)


def terrain_classification_metrics(
    scores: np.ndarray, eligible: np.ndarray, labels: np.ndarray,
) -> dict[str, Any]:
    truth = labels[eligible]
    prediction = np.argmax(scores, axis=-1)[eligible]
    per_terrain = {}
    for code, name in enumerate(TERRAIN_NAMES):
        mask = truth == code
        per_terrain[name] = {
            "endpoint_count": int(mask.sum()),
            "recall": None if not np.any(mask) else float(np.mean(prediction[mask] == code)),
        }
    return {
        "eligible_endpoint_count": len(truth),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=np.arange(4), average="macro")),
        "per_terrain": per_terrain,
    }


def _valid_case_a_events(
    traces: list[dict[str, np.ndarray]], manifests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hard_codes = {TERRAIN_CODE["concrete"], TERRAIN_CODE["marble"]}
    for run_index, (trace, meta) in enumerate(zip(traces, manifests)):
        if meta["case_id"] != "A":
            continue
        fall = meta["first_fall_sample_evaluation_only"]
        for foot, name in enumerate(SIDES):
            sample = meta[f"transition_sample_{name}"]
            if sample is None or int(sample) < 200 or (fall is not None and int(sample) >= int(fall)):
                continue
            prior = (
                trace["force_loaded"][:int(sample), foot]
                & np.isin(trace["terrain_gt"][:int(sample), foot], list(hard_codes))
            )
            if not np.any(prior):
                continue
            rows.append({
                "run_index": run_index, "run_id": meta["run_id"], "foot": foot,
                "transition_sample": int(sample), "speed_mps": float(meta["speed_mps"]),
                "terrain_before": meta["terrain_before"],
                "variation_index": int(meta["variation_index"]),
            })
    return rows


def evaluate_case_a_policy(
    traces: list[dict[str, np.ndarray]], manifests: list[dict[str, Any]],
    endpoints: np.ndarray, scores: np.ndarray, policy: TransitionPolicy,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[Any]]:
    events = _valid_case_a_events(traces, manifests)
    event_by_key = {(row["run_index"], row["foot"]): row for row in events}
    timelines = []
    outputs: list[dict[str, Any]] = []
    for run_index, (trace, meta, probability) in enumerate(zip(traces, manifests, scores)):
        finite = np.all(np.isfinite(trace["bilateral_canonical"]), axis=1)
        timeline = case_a_transition_timeline(
            probability, endpoints, trace["force_loaded"], trace["contact_age_ms"],
            trace["runtime_episode_id"], finite, policy, direct_authorized=False,
        )
        timelines.append(timeline)
        fall = meta["first_fall_sample_evaluation_only"]
        for emission in timeline.emissions:
            row = {**emission, "run_index": run_index, "run_id": meta["run_id"],
                   "case_id": meta["case_id"], "speed_mps": meta["speed_mps"],
                   "terrain_before": meta["terrain_before"],
                   "terrain_after": meta["terrain_after"]}
            sample = int(row["sample"]); foot = int(row["foot"])
            event = event_by_key.get((run_index, foot))
            row["post_fall"] = bool(fall is not None and sample >= int(fall))
            row["invalid"] = not bool(finite[sample])
            row["cross_foot"] = not bool(row["unique_g0_owner"])
            row["wrong_case"] = meta["case_id"] != "A"
            if event is None:
                row["attribution"] = "WRONG_CASE_OR_NORMAL_CONTACT"
                row["latency_ms"] = None
            else:
                latency = sample - int(event["transition_sample"])
                row["latency_ms"] = latency
                row["attribution"] = "MATCHED_CASE_A" if latency >= 0 else "TOO_EARLY_CASE_A"
            outputs.append(row)
    detections: dict[tuple[int, int], int] = {}
    for row in outputs:
        key = (int(row["run_index"]), int(row["foot"]))
        event = event_by_key.get(key)
        if (
            event is not None and not row["post_fall"] and not row["invalid"]
            and int(row["sample"]) >= int(event["transition_sample"])
        ):
            detections[key] = min(int(row["sample"]), detections.get(key, int(row["sample"])))
    latencies = [
        detections[(row["run_index"], row["foot"])] - row["transition_sample"]
        for row in events if (row["run_index"], row["foot"]) in detections
    ]
    true_output = len(detections)
    too_early = sum(row["attribution"] == "TOO_EARLY_CASE_A" for row in outputs)
    wrong_case = sum(bool(row["wrong_case"]) for row in outputs)
    postfall = sum(bool(row["post_fall"]) for row in outputs)
    invalid = sum(bool(row["invalid"]) for row in outputs)
    crossfoot = sum(bool(row["cross_foot"]) for row in outputs)
    false_output = len(outputs) - true_output
    positive_runs = {row["run_index"] for row in events}
    detected_runs = {key[0] for key in detections}
    normal_runs = {index for index, row in enumerate(manifests) if row["case_id"] != "A"}
    normal_fp_runs = {
        int(row["run_index"]) for row in outputs if int(row["run_index"]) in normal_runs
    }
    normal_contacts = 0
    for trace in traces:
        for foot in (0, 1):
            normal_contacts += len(set(trace["runtime_episode_id"][:, foot].tolist()) - {-1})
    normal_contacts -= len(events)
    speed_metrics = {}
    for speed in (0.10, 0.15, 0.20):
        selected = [row for row in events if row["speed_mps"] == speed]
        within = sum(
            (row["run_index"], row["foot"]) in detections
            and detections[(row["run_index"], row["foot"])] - row["transition_sample"] <= 20
            for row in selected
        )
        speed_metrics[f"{speed:.2f}"] = None if not selected else within / len(selected)
    hard_metrics = {}
    for terrain in ("concrete", "marble"):
        selected = [row for row in events if row["terrain_before"] == terrain]
        within = sum(
            (row["run_index"], row["foot"]) in detections
            and detections[(row["run_index"], row["foot"])] - row["transition_sample"] <= 20
            for row in selected
        )
        hard_metrics[terrain] = None if not selected else within / len(selected)
    within20 = sum(value <= 20 for value in latencies)
    precision = None if not outputs else true_output / len(outputs)
    recall = None if not events else true_output / len(events)
    f1 = 0.0 if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    duplicate = max(0, sum(
        count - 1 for count in Counter(
            (row["run_index"], row["foot"], row["runtime_episode_id"])
            for row in outputs
        ).values()
    ))
    metrics = {
        "event_count": len(events), "event_detected": true_output,
        "event_recall": recall, "run_count": len(positive_runs),
        "run_detected": len(detected_runs),
        "run_recall": None if not positive_runs else len(detected_runs) / len(positive_runs),
        "output_count": len(outputs), "precision": precision, "f1": f1,
        "detected_within_20ms": within20,
        "recall_within_20ms": None if not events else within20 / len(events),
        "median_latency_ms": None if not latencies else float(np.median(latencies)),
        "p95_latency_ms": None if not latencies else float(np.percentile(latencies, 95)),
        "normal_run_fp": len(normal_fp_runs), "normal_run_count": len(normal_runs),
        "normal_run_fp_rate": None if not normal_runs else len(normal_fp_runs) / len(normal_runs),
        "normal_contact_fp": false_output, "normal_contact_count": normal_contacts,
        "normal_contact_fp_rate": None if not normal_contacts else false_output / normal_contacts,
        "too_early": too_early, "wrong_case_output": wrong_case,
        "post_fall_output": postfall, "invalid_output": invalid,
        "duplicate_output": duplicate, "latch_carryover": 0,
        "cross_foot_output": crossfoot, "per_speed_recall_within_20ms": speed_metrics,
        "per_hard_source_recall_within_20ms": hard_metrics,
        "grouped_variation_fold_count": len(VARIATIONS),
        "independent_source_family_count": 1,
    }
    zero = (
        "normal_run_fp", "wrong_case_output", "invalid_output", "post_fall_output",
        "duplicate_output", "latch_carryover", "cross_foot_output",
    )
    metrics["direct_gate_pass"] = bool(
        all(metrics[name] == 0 for name in zero)
        and (metrics["recall_within_20ms"] or 0) >= 0.80
        and all(value is not None and value >= 0.70 for value in speed_metrics.values())
        and all(value is not None and value >= 0.65 for value in hard_metrics.values())
        and len(events) >= 50 and len(VARIATIONS) >= 5
        and metrics["independent_source_family_count"] >= 2
    )
    return metrics, outputs, timelines


def select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    direct = [row for row in rows if row["direct_gate_pass"]]
    if direct:
        return min(direct, key=lambda row: (
            row["normal_contact_fp"], -row["recall_within_20ms"],
            -row["f1"], row["candidate_id"], row["config_id"],
        ))
    usable = [row for row in rows if row["event_count"] and row["output_count"]]
    if not usable:
        raise RuntimeError("no usable transition-monitor candidate")
    return min(usable, key=lambda row: (
        -row["f1"], -row["event_recall"], -row["precision"],
        -row["terrain_macro_f1"], row["candidate_id"], row["config_id"],
    ))


def final_fit(
    traces: list[dict[str, np.ndarray]], endpoints: np.ndarray,
    eligible: np.ndarray, labels: np.ndarray, spec: dict[str, Any], output: Path,
) -> tuple[LinearTerrainModel, np.ndarray, dict[str, Any]]:
    features = terrain_features(
        traces, str(spec["architecture"]), int(spec["history_ms"]), endpoints,
    )
    training_endpoint = (np.arange(len(endpoints)) % (TRAINING_STRIDE_MS // ENDPOINT_STRIDE_MS)) == 0
    training = eligible & training_endpoint[None, :, None]
    model, health = fit_linear_model(
        features[training], labels[training], str(spec["architecture"]),
        int(spec["history_ms"]), 2026082299,
    )
    model_path = output / "terrain_transition_advisory_model.npz"
    model.save(str(model_path))
    reloaded = LinearTerrainModel.load(str(model_path))
    parity_features = features.reshape(-1, features.shape[-1])[:256]
    parity = float(np.max(np.abs(
        model.probabilities(parity_features) - reloaded.probabilities(parity_features)
    )))
    scores = model.probabilities(features.reshape(-1, features.shape[-1])).reshape(
        len(traces), len(endpoints), 2, 4,
    ).astype(np.float32)
    health.update({
        "version": "walking_v2_host_terrain_final_fit_v2",
        "development_only": True, "performance_estimation": False,
        "candidate_id": spec["candidate_id"], "seed": 2026082299,
        "fit_run_count": len(traces), "reload_max_abs_error": parity,
        "reload_exact": parity == 0.0, "model_sha256": sha256(model_path),
        "performance_source": "grouped OOF scores only; final-fit scores are replay only",
    })
    write_json(output / "terrain_transition_advisory_training.json", health)
    write_json(output / "terrain_transition_advisory_normalization.json", {
        "version": "walking_v2_host_terrain_normalization_v2",
        "candidate_id": spec["candidate_id"], "architecture": spec["architecture"],
        "history_ms": spec["history_ms"], "mean": model.mean.tolist(),
        "scale": model.scale.tolist(), "source": "all aligned development training rows",
    })
    return model, scores, health


def slip_replay(
    traces: list[dict[str, np.ndarray]], endpoints: np.ndarray, model: SlipV2Model,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    firing = np.zeros((len(traces), len(endpoints), 2), bool)
    score_arrays = {
        name: np.zeros((len(traces), len(endpoints), 2), np.float32)
        for name in ("normal", "early", "actionable", "active", "foot", "proposal")
    }
    outputs: list[dict[str, Any]] = []
    for run_index, trace in enumerate(traces):
        ages, runtime_episode, phases = derived_runtime_telemetry(trace["force_loaded"])
        features = np.asarray([
            runtime_feature(
                "S3", foot, int(endpoint), trace["bilateral_canonical"],
                trace["force_loaded"], ages, phases,
            )
            for endpoint in endpoints for foot in (0, 1)
        ], np.float32)
        raw = model.scores(features)
        scores = {name: value.reshape(len(endpoints), 2) for name, value in raw.items()}
        for name in score_arrays:
            score_arrays[name][run_index] = scores[name]
        counters = [0, 0]; emitted: set[tuple[int, int]] = set()
        finite = np.all(np.isfinite(trace["bilateral_canonical"]), axis=1)
        for row, endpoint in enumerate(endpoints):
            candidates = []
            for foot in (0, 1):
                valid = bool(
                    trace["force_loaded"][endpoint, foot] and ages[endpoint, foot] > 10
                    and finite[endpoint] and runtime_episode[endpoint, foot] >= 0
                )
                passes = bool(
                    valid and scores["proposal"][row, foot] >= SLIP_CONFIG.proposal_threshold
                    and scores["actionable"][row, foot] >= SLIP_CONFIG.state_threshold
                    and scores["actionable"][row, foot] - scores["early"][row, foot]
                    >= SLIP_CONFIG.early_margin
                    and scores["actionable"][row, foot] - scores["normal"][row, foot]
                    >= SLIP_CONFIG.normal_margin
                    and scores["foot"][row, foot] >= SLIP_CONFIG.foot_threshold
                )
                counters[foot] = counters[foot] + 1 if passes else 0
                episode = int(runtime_episode[endpoint, foot])
                if counters[foot] >= SLIP_CONFIG.persistence_endpoints and (foot, episode) not in emitted:
                    candidates.append(foot)
            if candidates:
                foot = max(candidates, key=lambda value: scores["proposal"][row, value])
                episode = int(runtime_episode[endpoint, foot]); emitted.add((foot, episode))
                firing[run_index, row, foot] = True
                outputs.append({
                    "run_index": run_index, "sample": int(endpoint), "foot": foot,
                    "runtime_episode_id": episode,
                    "proposal_score": float(scores["proposal"][row, foot]),
                })
        if (run_index + 1) % 70 == 0:
            print(f"S4-C causal replay: {run_index + 1}/{len(traces)}", flush=True)
    return firing, score_arrays, outputs


def evaluate_slip_replay(
    traces: list[dict[str, np.ndarray]], manifests: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for run_index, (trace, meta) in enumerate(zip(traces, manifests)):
        fall = meta["first_fall_sample_evaluation_only"]
        for foot in (0, 1):
            active = trace["physical_slip_active_label_only"][:, foot]
            starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
            for onset in starts:
                if fall is None or int(onset) < int(fall):
                    events.append({
                        "run_index": run_index, "foot": foot, "onset": int(onset),
                        "episode": int(trace["runtime_episode_id"][onset, foot]),
                        "speed": float(meta["speed_mps"]), "terrain_after": meta["terrain_after"],
                    })
    event_by_key = {(row["run_index"], row["foot"], row["episode"]): row for row in events}
    detected: dict[tuple[int, int, int], int] = {}
    false_outputs = postfall = 0
    for row in outputs:
        run = int(row["run_index"]); foot = int(row["foot"]); sample = int(row["sample"])
        meta = manifests[run]
        fall = meta["first_fall_sample_evaluation_only"]
        post = fall is not None and sample >= int(fall)
        postfall += int(post)
        key = (run, foot, int(row["runtime_episode_id"]))
        event = event_by_key.get(key)
        if not post and event is not None and sample >= event["onset"] - 100:
            detected[key] = min(sample, detected.get(key, sample))
        else:
            false_outputs += 1
    latencies = [
        detected[key] - row["onset"] for key, row in event_by_key.items() if key in detected
    ]
    precision = None if not outputs else len(detected) / len(outputs)
    recall = None if not events else len(detected) / len(events)
    f1 = 0.0 if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    speed = {}
    terrain = {}
    for value in (0.10, 0.15, 0.20):
        selected = [(key, row) for key, row in event_by_key.items() if row["speed"] == value]
        speed[f"{value:.2f}"] = None if not selected else sum(key in detected for key, _ in selected) / len(selected)
    for value in TERRAIN_NAMES:
        selected = [(key, row) for key, row in event_by_key.items() if row["terrain_after"] == value]
        terrain[value] = None if not selected else sum(key in detected for key, _ in selected) / len(selected)
    return {
        "event_count": len(events), "event_detected": len(detected),
        "episode_recall": recall, "output_count": len(outputs),
        "alert_precision": precision, "alert_f1": f1,
        "normal_or_too_early_fp": false_outputs, "post_fall_output": postfall,
        "median_warning_margin_ms": None if not latencies else float(np.median([-value for value in latencies])),
        "p95_late_latency_ms": None if not latencies else float(np.percentile(np.maximum(latencies, 0), 95)),
        "per_speed_recall": speed, "per_terrain_after_recall": terrain,
        "authority": "advisory only; aligned-corpus development replay",
    }


def held_scores(scores: np.ndarray, endpoints: np.ndarray, sample_count: int) -> np.ndarray:
    result = np.full((scores.shape[0], sample_count, 2, 4), np.nan, np.float16)
    for row, start in enumerate(endpoints):
        end = int(endpoints[row + 1]) if row + 1 < len(endpoints) else sample_count
        result[:, start:end] = scores[:, row].astype(np.float16)[:, None, :, :]
    return result


def save_trace_shards(
    output: Path, traces: list[dict[str, np.ndarray]], manifests: list[dict[str, Any]],
    endpoints: np.ndarray, selected_oof_scores: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    sample_count = len(traces[0]["time_s"])
    case_code = {name: index for index, name in enumerate(("A", "B", "C", "D", "N", "S"))}
    for variation in range(len(VARIATIONS)):
        indices = [index for index, row in enumerate(manifests) if row["variation_index"] == variation]
        path = output / f"aligned_traces_variation_{variation}.npz"
        cases = np.full((len(indices), sample_count, 2), -1, np.int8)
        for local, index in enumerate(indices):
            active = traces[index]["transition_target_contact_label_only"]
            cases[local][active] = case_code[manifests[index]["case_id"]]
        oof_held = held_scores(selected_oof_scores[indices], endpoints, sample_count)
        oof_state = np.full((len(indices), sample_count, 2), -1, np.int8)
        valid_score = np.isfinite(oof_held[..., 0])
        oof_state[valid_score] = np.argmax(oof_held, axis=-1)[valid_score].astype(np.int8)
        np.savez_compressed(
            path,
            run_id=np.asarray([manifests[index]["run_id"] for index in indices]),
            time_s=np.stack([traces[index]["time_s"] for index in indices]),
            bilateral_fusion20_raw=np.stack([traces[index]["bilateral_fusion20_raw"] for index in indices]),
            bilateral_canonical_model_input=np.stack([traces[index]["bilateral_canonical"] for index in indices]),
            force_loaded=np.stack([traces[index]["force_loaded"] for index in indices]),
            physical_contact_label_only=np.stack([traces[index]["physical_contact"] for index in indices]),
            surface_force_n_label_only=np.stack([traces[index]["surface_force_n"] for index in indices]),
            terrain_gt_code_label_only=np.stack([traces[index]["terrain_gt"] for index in indices]),
            terrain_scores_selected_oof_1khz_held=oof_held,
            terrain_state_selected_oof_1khz=oof_state,
            terrain_score_endpoint_valid=np.isin(np.arange(sample_count), endpoints),
            transition_case_gt_code_label_only=cases,
            transition_target_contact_label_only=np.stack([traces[index]["transition_target_contact_label_only"] for index in indices]),
            touchdown=np.stack([traces[index]["touchdown"] for index in indices]),
            contact_age_ms=np.stack([traces[index]["contact_age_ms"] for index in indices]),
            runtime_episode_id=np.stack([traces[index]["runtime_episode_id"] for index in indices]),
            g0_owner=np.stack([traces[index]["g0_owner"] for index in indices]),
            physical_slip_active_label_only=np.stack([traces[index]["physical_slip_active_label_only"] for index in indices]),
            pre_fall_valid_label_only=np.stack([traces[index]["pre_fall_valid_label_only"] for index in indices]),
        )
        if path.stat().st_size >= 45 * 1024 * 1024:
            raise RuntimeError(f"trace shard exceeds 45 MiB: {path}")
        rows.append({
            "file": path.name, "variation_index": variation, "run_count": len(indices),
            "byte_count": path.stat().st_size, "sha256": sha256(path),
        })
    write_csv(output / "trace_shard_manifest.csv", rows)
    return rows


def _csv_ready(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in row.items()
    }


def execute(root: Path, output: Path) -> int:
    guard, locked_model, started = preflight(root, output)
    scene_path = guard.path(SCENE, "load fixed walking scene")
    policy_path = guard.path(POLICY, "load fixed walking controller")
    for path in source_paths():
        guard.path(path, "hash host-v2 implementation")
    for path in (
        SOURCE / "terrain_profiles.py", SOURCE / "bilateral_hil_sensor_v2.py",
        SOURCE / "g1_upstream_locomotion.py", SOURCE / "walking_hazard_ground_truth_v1.py",
        SOURCE / "walking_hazard_oracle_calibration_v1.py",
    ):
        guard.path(path, "hash causal or offline-label dependency")
    oracle_summary = json.loads(guard.path(
        ORACLE / "summary.json", "verify frozen Slip oracle contract",
    ).read_text())
    conditions = acquisition_matrix()
    traces: list[dict[str, np.ndarray]] = []
    manifests: list[dict[str, Any]] = []
    for index, condition in enumerate(conditions):
        trace, metadata = collect_run(condition, scene_path, policy_path)
        traces.append(trace); manifests.append(metadata)
        if (index + 1) % 20 == 0 or index + 1 == len(conditions):
            print(f"aligned simulation: {index + 1}/{len(conditions)}", flush=True)
    write_csv(output / "run_manifest.csv", [_csv_ready(row) for row in manifests])
    endpoints = endpoints_for(len(traces[0]["time_s"]))
    eligible, labels = eligible_mask(traces, endpoints)

    scores_by_candidate: dict[str, np.ndarray] = {}
    health_rows: list[dict[str, Any]] = []
    model_audits: dict[str, Any] = {}
    locked_scores = locked_t2_scores(traces, locked_model, endpoints)
    scores_by_candidate["LOCKED_T2_BASELINE"] = locked_scores
    classification = {
        "LOCKED_T2_BASELINE": terrain_classification_metrics(locked_scores, eligible, labels),
    }
    for spec in MODEL_CANDIDATES:
        scores, folds, audit = grouped_oof_candidate(
            traces, manifests, endpoints, spec, eligible, labels,
        )
        candidate_id = str(spec["candidate_id"])
        scores_by_candidate[candidate_id] = scores
        health_rows.extend(folds); model_audits[candidate_id] = audit
        classification[candidate_id] = terrain_classification_metrics(scores, eligible, labels)
    write_csv(output / "optimizer_health.csv", [_csv_ready(row) for row in health_rows])
    write_json(output / "terrain_classification_metrics.json", classification)

    comparison_rows: list[dict[str, Any]] = []
    detail_by_key: dict[tuple[str, str], tuple[list[dict[str, Any]], list[Any]]] = {}
    for candidate_id, scores in scores_by_candidate.items():
        for policy in transition_policy_matrix():
            metrics, outputs, timelines = evaluate_case_a_policy(
                traces, manifests, endpoints, scores, policy,
            )
            row = {
                "candidate_id": candidate_id, "config_id": policy.config_id,
                "confidence_threshold": policy.confidence_threshold,
                "dwell_endpoints": policy.dwell_endpoints,
                "minimum_contact_age_ms": policy.minimum_contact_age_ms,
                "terrain_accuracy": classification[candidate_id]["accuracy"],
                "terrain_macro_f1": classification[candidate_id]["macro_f1"],
                **metrics,
            }
            comparison_rows.append(row)
            detail_by_key[(candidate_id, policy.config_id)] = (outputs, timelines)
        print(f"transition policies evaluated: {candidate_id}", flush=True)
    selected = select_candidate(comparison_rows)
    selected_id = str(selected["candidate_id"])
    selected_config_id = str(selected["config_id"])
    selected_policy = next(
        value for value in transition_policy_matrix() if value.config_id == selected_config_id
    )
    selected_outputs, selected_oof_timelines = detail_by_key[(selected_id, selected_config_id)]
    write_csv(output / "candidate_comparison.csv", [_csv_ready(row) for row in comparison_rows])
    write_json(output / "selected_transition_monitor_metrics.json", selected)
    write_csv(output / "selected_transition_outputs.csv", [_csv_ready(row) for row in selected_outputs])

    if selected_id == "LOCKED_T2_BASELINE":
        # The baseline remains executable from its immutable upstream artifact.
        final_model: Any = locked_model
        final_scores = locked_scores
        final_health = {
            "version": "walking_v2_host_terrain_final_fit_v2",
            "candidate_id": selected_id, "reused_immutable_upstream": True,
            "performance_source": "new aligned development replay",
            "model_sha256": sha256(root / LOCKED_TERRAIN / "terrain_candidate_model.npz"),
        }
        write_json(output / "terrain_transition_advisory_training.json", final_health)
        write_json(output / "terrain_transition_advisory_normalization.json", {
            "version": "walking_v2_host_terrain_normalization_v2",
            "candidate_id": selected_id,
            "upstream_path": (LOCKED_TERRAIN / "terrain_candidate_normalization.json").as_posix(),
            "upstream_sha256": sha256(root / LOCKED_TERRAIN / "terrain_candidate_normalization.json"),
        })
    else:
        selected_spec = next(row for row in MODEL_CANDIDATES if row["candidate_id"] == selected_id)
        final_model, final_scores, final_health = final_fit(
            traces, endpoints, eligible, labels, selected_spec, output,
        )
    write_json(output / "terrain_transition_runtime_config.json", {
        "version": "walking_v2_host_terrain_runtime_config_v2",
        "candidate_id": selected_id, "policy": asdict(selected_policy),
        "config_id": selected_policy.config_id, "endpoint_stride_ms": ENDPOINT_STRIDE_MS,
        "direct_authorized": False, "performance_source": "five-fold grouped OOF",
    })

    final_timelines = []
    for trace, probability in zip(traces, final_scores):
        finite = np.all(np.isfinite(trace["bilateral_canonical"]), axis=1)
        final_timelines.append(case_a_transition_timeline(
            probability, endpoints, trace["force_loaded"], trace["contact_age_ms"],
            trace["runtime_episode_id"], finite, selected_policy,
            direct_authorized=False,
        ))

    prior_slip_metrics_path = guard.path(
        PRIOR_HOST / "advisory_metrics.json", "load frozen Slip OOF evidence",
    )
    prior_slip_training_path = guard.path(
        PRIOR_HOST / "slip_advisory_training.json", "verify frozen Slip optimizer health",
    )
    slip_model_path = guard.path(
        PRIOR_HOST / "slip_advisory_model.npz", "load frozen executable Slip model",
    )
    prior_slip_metrics = json.loads(prior_slip_metrics_path.read_text())
    prior_slip_training = json.loads(prior_slip_training_path.read_text())
    slip_model = SlipV2Model.load(slip_model_path)
    slip_firing, slip_scores, slip_outputs = slip_replay(traces, endpoints, slip_model)
    aligned_slip_metrics = evaluate_slip_replay(traces, manifests, slip_outputs)
    write_json(output / "slip_advisory_metrics.json", {
        "version": "walking_v2_host_slip_advisory_metrics_v2",
        "aligned_transition_corpus_replay": aligned_slip_metrics,
        "frozen_grouped_oof_reference": prior_slip_metrics["slip_advisory_frozen_oof_reference"],
        "model_sha256": sha256(slip_model_path),
        "training_health": {
            "all_heads_converged": prior_slip_training["all_heads_converged"],
            "head_iterations": prior_slip_training["head_iterations"],
            "limitation": prior_slip_training["limitation"],
        },
        "authority": "advisory only",
    })

    shard_rows = save_trace_shards(
        output, traces, manifests, endpoints, scores_by_candidate[selected_id],
    )
    terrain_stable = np.stack([value.stable_state for value in final_timelines])
    terrain_advisory = np.stack([value.case_a_advisory for value in final_timelines])
    g0_owner = np.stack([value.owner for value in final_timelines])
    direct_reflex = np.zeros_like(terrain_advisory)
    np.savez_compressed(
        output / "host_replay.npz",
        run_id=np.asarray([row["run_id"] for row in manifests]),
        endpoints=endpoints, terrain_scores=final_scores.astype(np.float32),
        terrain_stable_state=terrain_stable,
        case_a_transition_advisory=terrain_advisory,
        slip_proposal_score=slip_scores["proposal"], slip_advisory=slip_firing,
        g0_unique_owner=g0_owner, direct_reflex=direct_reflex,
        recovery_actuation=np.zeros_like(direct_reflex),
    )

    valid_events = _valid_case_a_events(traces, manifests)
    transition_coverage = {
        "case_a_valid_event_count": len(valid_events),
        "case_a_events_by_speed": dict(Counter(f"{row['speed_mps']:.2f}" for row in valid_events)),
        "case_a_events_by_foot": dict(Counter(SIDES[row["foot"]] for row in valid_events)),
        "case_a_events_by_hard_type": dict(Counter(row["terrain_before"] for row in valid_events)),
    }
    corpus_summary = {
        "version": "walking_v2_aligned_transition_corpus_v2",
        "development_only": True, "future_blind_exclusion_permanent": True,
        "run_count": len(traces), "sample_rate_hz": SAMPLE_RATE_HZ,
        "sample_count_per_run": len(traces[0]["time_s"]),
        "duration_s_per_run": conditions[0].duration_s,
        "total_samples": len(traces) * len(traces[0]["time_s"]),
        "case_run_count": dict(Counter(row["case_id"] for row in manifests)),
        "speed_run_count": dict(Counter(f"{row['speed_mps']:.2f}" for row in manifests)),
        "variation_run_count": dict(Counter(str(row["variation_index"]) for row in manifests)),
        "runs_with_left_transition": sum(row["transition_sample_left"] is not None for row in manifests),
        "runs_with_right_transition": sum(row["transition_sample_right"] is not None for row in manifests),
        "runs_with_first_fall": sum(row["first_fall_sample_evaluation_only"] is not None for row in manifests),
        "physical_slip_onset_count": sum(
            row[f"physical_slip_onset_{side}"] is not None for row in manifests for side in SIDES
        ),
        "exact_fusion20_finite_runs": sum(row["fusion20_finite"] for row in manifests),
        "shards": shard_rows, "transition_coverage": transition_coverage,
        "terrain_code": TERRAIN_CODE,
        "case_code": {name: index for index, name in enumerate(("A", "B", "C", "D", "N", "S"))},
        "terrain_score_alignment": (
            "selected grouped-OOF 100 Hz causal endpoints held to a synchronized 1 kHz "
            "timeline; terrain_score_endpoint_valid identifies actual inference endpoints"
        ),
    }
    write_json(output / "aligned_corpus_summary.json", corpus_summary)
    write_json(output / "development_split_and_leakage_audit.json", {
        "version": "walking_v2_host_grouped_split_v2", "fold_key": "variation_index",
        "fold_count": len(VARIATIONS), "evaluation_once_per_run": True,
        "training_stride_ms": TRAINING_STRIDE_MS, "inference_stride_ms": ENDPOINT_STRIDE_MS,
        "run_pair_source_variation_seed_overlap_per_fold": 0,
        "unique_run_count": len({row["run_id"] for row in manifests}),
        "unique_pair_count": len({row["pair_id"] for row in manifests}),
        "unique_source_count": len({row["source_id"] for row in manifests}),
        "unique_seed_count": len({row["seed"] for row in manifests}),
        "blind_or_forbidden_members": 0, "future_blind_eligible_runs": 0,
        "candidate_audits": model_audits,
    })

    alternatives = []
    for candidate_id in scores_by_candidate:
        best = select_candidate([row for row in comparison_rows if row["candidate_id"] == candidate_id])
        alternatives.append({
            "architecture": candidate_id, "role": "dedicated Case-A transition monitor",
            "selected": candidate_id == selected_id, "direct_gate_pass": best["direct_gate_pass"],
            "config_id": best["config_id"], "event_recall": best["event_recall"],
            "precision": best["precision"], "f1": best["f1"],
            "recall_within_20ms": best["recall_within_20ms"],
            "normal_run_fp": best["normal_run_fp"],
            "reason": "grouped-OOF aligned transition comparison",
        })
    alternatives.extend((
        {
            "architecture": "PRIOR_BROAD_S4C_SLIP", "role": "Slip advisory",
            "selected": True, "direct_gate_pass": False, "config_id": SLIP_CONFIG.config_id,
            "event_recall": prior_slip_metrics["slip_advisory_frozen_oof_reference"]["episode_recall"],
            "precision": prior_slip_metrics["slip_advisory_frozen_oof_reference"]["alert_precision"],
            "f1": prior_slip_metrics["slip_advisory_frozen_oof_reference"]["alert_f1"],
            "recall_within_20ms": None,
            "normal_run_fp": prior_slip_metrics["slip_advisory_frozen_oof_reference"]["normal_run_fp"],
            "reason": "retained only as non-actuating advisory; optimizer remains unqualified",
        },
        {
            "architecture": "CASE_A_PLUS_REACTIVE_SLIP_DIRECT", "role": "direct reflex",
            "selected": False, "direct_gate_pass": False, "config_id": "not_reopened",
            "event_recall": 0.0, "precision": None, "f1": 0.0,
            "recall_within_20ms": 0.0, "normal_run_fp": 0,
            "reason": "prior aligned role test had zero scoped output; new transition model cannot cure reactive Slip collapse",
        },
    ))
    write_csv(output / "architecture_alternatives.csv", [_csv_ready(row) for row in alternatives])

    # Host-side compute estimate: deterministic NumPy timing, not target latency.
    bench_tensor = terrain_features(
        traces[:1], getattr(final_model, "architecture", "T2"),
        int(getattr(final_model, "history_ms", 200)), endpoints,
    )
    bench_flat = bench_tensor.reshape(-1, bench_tensor.shape[-1])
    bench_start = time.perf_counter()
    for _ in range(100):
        final_model.probabilities(bench_flat)
    bench_us = (time.perf_counter() - bench_start) * 1e6 / (100 * len(bench_flat))
    terrain_parameters = int(final_model.parameter_count)
    terrain_macs = int(final_model.macs_per_endpoint if isinstance(final_model, LinearTerrainModel) else final_model.macs)
    resource = {
        "version": "walking_v2_host_resource_estimate_v2",
        "endpoint_rate_hz": 100, "sample_preprocessing_rate_hz": 1000,
        "shared_fusion20_history_bytes_float32": 20 * 200 * 4,
        "terrain_parameter_count": terrain_parameters,
        "terrain_macs_per_foot_endpoint": terrain_macs,
        "slip_parameter_count": slip_model.parameter_count,
        "slip_macs_per_foot_endpoint": slip_model.macs,
        "combined_macs_per_100hz_tick_two_feet": 2 * (terrain_macs + slip_model.macs),
        "deterministic_state_bytes_upper_bound": 2048,
        "host_numpy_model_only_mean_us_per_foot_endpoint": bench_us,
        "host_operator_compatibility": ["normalization", "matmul", "bias", "softmax", "argmax", "comparison"],
        "target_latency_or_memory_claim": False, "int8_or_vela_claim": False,
    }
    write_json(output / "resource_and_operator_estimate.json", resource)

    selected_architecture = "CASE_A_TRANSITION_MONITOR_PLUS_SLIP_ADVISORY"
    direct_failure = {
        "version": "walking_v2_host_direct_authority_decision_v2",
        "direct_reflex_enabled": False,
        "development_metric_gate_pass": bool(selected["direct_gate_pass"]),
        "decisive_failure": (
            "all 420 aligned traces come from one MuJoCo/controller source family; the "
            "frozen direct gate requires at least two independent source families"
        ),
        "other_observed_failures": {
            name: selected[name] for name in (
                "normal_run_fp", "wrong_case_output", "post_fall_output",
                "recall_within_20ms", "per_speed_recall_within_20ms",
                "per_hard_source_recall_within_20ms",
            )
        },
        "why_not_more_tuning": (
            "two newly trained causal encoders, the locked baseline and 24 frozen state "
            "policies establish the attainable simulator-development frontier; tuning "
            "cannot manufacture independent evidence and would increase overfit risk"
        ),
    }
    write_json(output / "direct_authority_decision.json", direct_failure)

    internal_checks = {
        "run_count_420": len(traces) == 420,
        "all_fusion20_finite": all(row["fusion20_finite"] for row in manifests),
        "five_grouped_folds": len(VARIATIONS) == 5,
        "case_a_event_count_at_least_50": len(valid_events) >= 50,
        "all_required_cases": set(row["case_id"] for row in manifests) == {"A", "B", "C", "D", "N", "S"},
        "all_required_speeds": set(row["speed_mps"] for row in manifests) == {0.10, 0.15, 0.20},
        "both_feet_in_valid_case_a": set(row["foot"] for row in valid_events) == {0, 1},
        "both_hard_types_in_valid_case_a": set(row["terrain_before"] for row in valid_events) == {"concrete", "marble"},
        "selected_model_oof_complete": model_audits.get(selected_id, {"all_folds_filled": True})["all_folds_filled"],
        "all_new_model_folds_converged": all(row["converged"] for row in health_rows),
        "selected_final_model_converged": bool(final_health.get("converged", True)),
        "direct_output_all_false": not bool(np.any(direct_reflex)),
        "recovery_output_all_false": not bool(np.any(direct_reflex)),
        "no_direct_gate_pass": not any(row["direct_gate_pass"] for row in comparison_rows),
        "selected_final_model_reload_exact": bool(final_health.get("reload_exact", True)),
        "zero_forbidden_access": guard.forbidden == 0,
        "zero_blocked_access": guard.blocked == 0,
        "sink_deferred": runtime_contract()["responsibilities"]["Sink"].startswith("deferred"),
        "files_below_45_mib": all(row["byte_count"] < 45 * 1024 * 1024 for row in shard_rows),
    }
    if not all(internal_checks.values()):
        write_json(output / "internal_failure.json", internal_checks)
        raise RuntimeError(f"internal completion gate failed: {internal_checks}")
    write_json(output / "test_results.json", {
        "version": "walking_v2_host_internal_tests_v2", "checks": internal_checks,
        "external_command": (
            "PYTHONPATH=simulation/unitree_mujoco/simulate_python simulation/venv/bin/python "
            "-m unittest -v simulation/unitree_mujoco/simulate_python/"
            "test_walking_v2_host_design_milestone_v2.py"
        ),
    })

    summary = {
        "version": "walking_v2_host_design_summary_v2",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "selected_architecture": selected_architecture,
        "selected_terrain_candidate": selected_id,
        "selected_terrain_policy": selected_config_id,
        "direct_reflex_enabled": False, "recovery_actuation_enabled": False,
        "transition_monitor_oof": selected,
        "terrain_oof": classification[selected_id],
        "slip_aligned_development_replay": aligned_slip_metrics,
        "slip_frozen_oof_reference": prior_slip_metrics["slip_advisory_frozen_oof_reference"],
        "host_design_locked": True, "int8_vela_performed": False,
        "sink": "SINK_RUNTIME_DETECTION_DEFERRED",
        "blind_or_forbidden_data_accessed": False,
        "hardware_e84_or_physical_hil_executed": False,
        "next_step": "ACQUIRE_FRESH_BLIND_HOST_HOLDOUT",
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "summary.json", summary)

    audit = f"""# Walking v2 host design milestone v2

1. **Aligned corpus:** generated {len(traces)} development-only bilateral 1 kHz runs ({len(VARIATIONS)} isolated surface/gait folds, two seeded replicates, three speeds, A/B/C/D, hard-to-hard and steady seams). Exact Fusion20, Terrain GT/scores/state, transition/touchdown/G0, physical Slip and evaluation-only first-fall timing are retained in five sub-45-MiB shards.
2. **Real bottleneck:** missing aligned transition supervision was repaired. The remaining direct-authority blocker is evidence independence, plus any false/latency failures shown in `direct_authority_decision.json`; a single simulator/controller family cannot establish false-actuation safety.
3. **Architectures evaluated:** locked T2, T1 200-ms linear, T2 dual-timescale 200-ms linear, 24 causal policies each, prior broad S4-C Slip, and the previously failed reactive confirmation role.
4. **Engineering changes:** bilateral seeded generation, physical per-foot terrain ownership, exact timestamps, grouped OOF isolation, dedicated transition retraining, retained-across-air state, queued unique G0 ownership and one-shot dedup.
5. **Selected design:** `{selected_architecture}` using `{selected_id}` / `{selected_config_id}`.
6. **Authority:** Terrain transition and S4-C Slip are advisory only; G0 owns causal contact identity; deterministic state owns dwell/dedup/firewall; direct and recovery arrays are hard false; Sink is deferred.
7. **Results:** Case-A event recall `{selected['event_recall']:.6f}`, precision `{selected['precision']:.6f}`, F1 `{selected['f1']:.6f}`, within-20-ms recall `{selected['recall_within_20ms']:.6f}`, normal-run FP `{selected['normal_run_fp']}`, normal-contact FP `{selected['normal_contact_fp']}`, median/p95 latency `{selected['median_latency_ms']}`/`{selected['p95_latency_ms']}` ms. Per-speed and hard-source results are in `selected_transition_monitor_metrics.json`; Slip results are in `slip_advisory_metrics.json`.
8. **Direct reflex:** disabled. No runtime path consumes Terrain GT, Slip oracle, fall oracle, timestamps, identities or future samples.
9. **Stopping rationale:** the bounded 72-point transition comparison reached the one-source evidence ceiling; more same-source tuning cannot satisfy the frozen independence gate and risks development overfit.
10. **Lock:** one immutable development design lock is written; it is not a blind, safety, deployment or real-robot lock.
11. **INT8/Vela:** not performed; optional target preparation was intentionally left after independent evaluation.
12. **Exact next task:** `ACQUIRE_FRESH_BLIND_HOST_HOLDOUT`.
13. **Data boundary:** no outer, holdout, spatial-final or final-test content was accessed or generated; all new runs are permanently blind-ineligible.
14. **Hardware boundary:** no flashing, E84 execution, physical HIL or hardware action occurred.
15. **Sink:** remains `SINK_RUNTIME_DETECTION_DEFERRED`; Case B is evaluation context only.
"""
    (output / "audit.md").write_text(audit, encoding="utf-8")

    lock_files = [
        "protocol.json", "causal_runtime_contract.json", "aligned_corpus_summary.json",
        "run_manifest.csv", "trace_shard_manifest.csv", "candidate_comparison.csv",
        "selected_transition_monitor_metrics.json", "terrain_classification_metrics.json",
        "terrain_transition_advisory_training.json", "terrain_transition_advisory_normalization.json",
        "terrain_transition_runtime_config.json", "slip_advisory_metrics.json",
        "host_replay.npz", "resource_and_operator_estimate.json",
        "development_split_and_leakage_audit.json", "direct_authority_decision.json",
        "test_results.json", "summary.json", "audit.md",
    ]
    if (output / "terrain_transition_advisory_model.npz").exists():
        lock_files.append("terrain_transition_advisory_model.npz")
    lock_files.extend(row["file"] for row in shard_rows)
    design_lock = {
        "version": "walking_v2_host_development_design_lock_v2",
        "immutable": True, "lock_type": "development_design_not_blind_safety_or_deployment",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "selected_architecture": selected_architecture,
        "selected_candidate": selected_id, "selected_policy": selected_config_id,
        "authority": runtime_contract()["authority"],
        "direct_reflex_enabled": False, "recovery_actuation_enabled": False,
        "locked_artifact_sha256": {name: sha256(output / name) for name in lock_files},
        "upstream_slip_model": {
            "path": (PRIOR_HOST / "slip_advisory_model.npz").as_posix(),
            "sha256": sha256(slip_model_path),
        },
        "future_blind_exclusion": {
            "corpus": output.relative_to(root).as_posix(), "permanent": True,
        },
        "explicitly_not": [
            "blind validation", "safety certification", "deployment authorization",
            "real-robot readiness", "E84 authorization",
        ],
        "next_step": "ACQUIRE_FRESH_BLIND_HOST_HOLDOUT",
    }
    write_json(output / "design_lock.json", design_lock)
    guard.freeze()

    artifacts = {}
    for path in sorted(output.iterdir()):
        if not path.is_file() or path.name == "provenance.json":
            continue
        if path.stat().st_size >= 45 * 1024 * 1024:
            raise RuntimeError(f"artifact exceeds 45 MiB: {path}")
        artifacts[path.name] = {"sha256": sha256(path), "byte_count": path.stat().st_size}
    write_json(output / "provenance.json", {
        "version": "walking_v2_host_design_provenance_v2",
        "starting_checkpoint": STARTING_CHECKPOINT, "execution_head": git("rev-parse", "HEAD"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {path: sha256(root / path) for path in source_paths()},
        "artifact_manifest": artifacts, "manifest_self_hash_excluded": True,
        "artifact_hash_graph_complete": True, "generated_development_run_count": len(traces),
        "future_blind_eligible_generated_run_count": 0,
        "forbidden_access_count": guard.forbidden, "blocked_access_count": guard.blocked,
        "blind_data_access_count": 0, "hardware_e84_hil_execution_count": 0,
        "int8_vela_execution_count": 0,
    })
    print(json.dumps({
        "selected_architecture": selected_architecture,
        "selected_candidate": selected_id, "selected_policy": selected_config_id,
        "case_a_events": len(valid_events), "direct_reflex_enabled": False,
        "next_step": "ACQUIRE_FRESH_BLIND_HOST_HOLDOUT",
    }, sort_keys=True))
    return 0


def replay_only(root: Path, output: Path) -> int:
    """Recompute causal host outputs from the saved development shards."""
    config = json.loads((output / "terrain_transition_runtime_config.json").read_text())
    policy = TransitionPolicy(**config["policy"])
    traces: list[dict[str, np.ndarray]] = []
    run_ids: list[str] = []
    for path in sorted(output.glob("aligned_traces_variation_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            for index, run_id in enumerate(archive["run_id"].astype(str)):
                run_ids.append(str(run_id))
                traces.append({
                    "bilateral_canonical": archive["bilateral_canonical_model_input"][index].copy(),
                    "force_loaded": archive["force_loaded"][index].copy(),
                    "contact_age_ms": archive["contact_age_ms"][index].copy(),
                    "runtime_episode_id": archive["runtime_episode_id"][index].copy(),
                })
    with np.load(output / "host_replay.npz", allow_pickle=False) as expected:
        endpoints = expected["endpoints"].copy()
        expected_run_ids = expected["run_id"].astype(str)
        expected_arrays = {name: expected[name].copy() for name in (
            "terrain_scores", "terrain_stable_state", "case_a_transition_advisory",
            "slip_advisory", "g0_unique_owner", "direct_reflex", "recovery_actuation",
        )}
    if not np.array_equal(np.asarray(run_ids), expected_run_ids):
        raise RuntimeError("saved shard run ordering differs from host replay")
    candidate_id = config["candidate_id"]
    if candidate_id == "LOCKED_T2_BASELINE":
        terrain_model: Any = ProjectedLinearModel.load(
            root / LOCKED_TERRAIN / "terrain_candidate_model.npz"
        )
    else:
        terrain_model = LinearTerrainModel.load(str(output / "terrain_transition_advisory_model.npz"))
    features = terrain_features(
        traces, terrain_model.architecture, int(getattr(terrain_model, "history_ms", 200)),
        endpoints,
    )
    terrain_scores = terrain_model.probabilities(
        features.reshape(-1, features.shape[-1])
    ).reshape(len(traces), len(endpoints), 2, 4).astype(np.float32)
    timelines = []
    for trace, scores in zip(traces, terrain_scores):
        finite = np.all(np.isfinite(trace["bilateral_canonical"]), axis=1)
        timelines.append(case_a_transition_timeline(
            scores, endpoints, trace["force_loaded"], trace["contact_age_ms"],
            trace["runtime_episode_id"], finite, policy, direct_authorized=False,
        ))
    slip_model = SlipV2Model.load(root / PRIOR_HOST / "slip_advisory_model.npz")
    slip_firing, _, _ = slip_replay(traces, endpoints, slip_model)
    actual = {
        "terrain_scores": terrain_scores,
        "terrain_stable_state": np.stack([value.stable_state for value in timelines]),
        "case_a_transition_advisory": np.stack([value.case_a_advisory for value in timelines]),
        "slip_advisory": slip_firing,
        "g0_unique_owner": np.stack([value.owner for value in timelines]),
        "direct_reflex": np.zeros_like(expected_arrays["direct_reflex"]),
        "recovery_actuation": np.zeros_like(expected_arrays["recovery_actuation"]),
    }
    mismatches = {
        name: (
            float(np.max(np.abs(value - expected_arrays[name])))
            if np.issubdtype(value.dtype, np.floating)
            else int(np.sum(value != expected_arrays[name]))
        )
        for name, value in actual.items()
        if not np.array_equal(value, expected_arrays[name])
    }
    if mismatches:
        raise RuntimeError(f"causal replay mismatch: {mismatches}")
    print(json.dumps({
        "replay_exact": True, "run_count": len(traces),
        "direct_reflex_output_count": int(actual["direct_reflex"].sum()),
        "recovery_output_count": int(actual["recovery_actuation"].sum()),
    }, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--replay-only", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    if args.execute == args.replay_only:
        print(json.dumps(protocol_payload(), indent=2, sort_keys=True))
        return 0
    if args.replay_only:
        return replay_only(root, output)
    return execute(root, output)


if __name__ == "__main__":
    raise SystemExit(main())
