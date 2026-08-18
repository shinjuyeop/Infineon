"""Materialize and audit continuous-state Terrain Transition v1 pilot traces.

This is deliberately a data/physics foundation, not a classifier or reflex
evaluation.  It reuses the frozen Fast Reflex v2 physical oracle thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import time

import mujoco
import numpy as np

from controlled_excitation import (ExcitationCondition, HorizontalPulse,
    HorizontalPulseExciter, VerticalElasticBandSupport, apply_excitation_condition)
from expanded_terrain_dataset_v1 import make_expanded_run_specification
from hil_sensor import G1HilSensorReader, HIL_SENSOR_CHANNELS
from run_terrain_fast_reflex_v2 import (_foot_oracle, _apply_vertical_pulse,
    SCENES)
from terrain_fast_reflex_v2 import (PHYSICS_STEPS_PER_SAMPLE,
    PHYSICS_TIMESTEP_S, SENSOR_RATE_HZ, V2_ORACLE_CHANNELS, V2_ORACLE_INDEX,
    V2Calibration)
from terrain_profiles import TERRAIN_PROFILES, apply_terrain_profile


SIM_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = SIM_DIR / "outputs" / "terrain_transition_v1_pilot"
SETTLE_TIME_S, TRANSITION_TIME_S, DURATION_S = .500, .650, .800
MASTER_SEED = 20260818
FROZEN_CALIBRATION = V2Calibration(
    minimum_load_N=49.767116968043574, risk_speed_mps=0.03262403628073019,
    risk_speed_trend_mps_per_ms=0.0011596816745374115,
    confirmed_speed_mps=0.04488778284622058, sink_depth_m=0.00025438814774947615,
    downward_speed_mps=0.005117943081648034, tilt_change_rad=0.003772000213602436,
    angular_rate_rad_s=0.07543294195294141, persistence_samples=3,
    provenance="frozen terrain_fast_reflex_v2_pilot train-normal calibration",
)
CASES = {
    # 180 N is an already bounded v2 calibration excitation.  It supplies
    # physical Slip coverage without changing the frozen oracle threshold.
    "A": {"name": "hard_to_ice", "before": "marble", "after": "ice", "horizontal_force_N": 180., "vertical_force_N": 0.},
    "B": {"name": "hard_to_sand", "before": "marble", "after": "sand", "horizontal_force_N": 0., "vertical_force_N": 100.},
    "C": {"name": "ice_to_hard", "before": "ice", "after": "marble", "horizontal_force_N": 0., "vertical_force_N": 0.},
    "D": {"name": "sand_to_hard", "before": "sand", "after": "marble", "horizontal_force_N": 0., "vertical_force_N": 0.},
}


@dataclass
class Run:
    metadata: dict[str, object]
    time_s: np.ndarray
    fusion10: np.ndarray
    oracle: np.ndarray
    terrain_gt: np.ndarray
    state: dict[str, float]


def _sustained(mask: np.ndarray, samples: int) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    for end in np.flatnonzero(np.convolve(mask.astype(np.int8), np.ones(samples, np.int8), "valid") == samples) + samples - 1:
        result[end - samples + 1:end + 1] = True
    return result


def label(oracle: np.ndarray, transition_sample: int) -> dict[str, np.ndarray]:
    """The v2 label semantics, applied to a full continuous trace."""
    c = FROZEN_CALIBRATION; idx = V2_ORACLE_INDEX
    contact = oracle[:, idx["left_contact"]] > .5
    loaded = contact & (oracle[:, idx["contact_normal_force_N"]] >= c.minimum_load_N)
    speed = oracle[:, idx["foot_horizontal_speed_mps"]]
    trend = np.maximum(0., np.diff(speed, prepend=speed[0]))
    confirmed = _sustained(loaded & (speed > c.confirmed_speed_mps), c.persistence_samples)
    risk = _sustained(loaded & (speed > c.risk_speed_mps) & (trend > c.risk_speed_trend_mps_per_ms), c.persistence_samples) | confirmed
    sink = _sustained(loaded & (oracle[:, idx["foot_sink_depth_m"]] > c.sink_depth_m)
                      & (np.maximum(0., -oracle[:, idx["foot_velocity_z_mps"]]) > c.downward_speed_mps), c.persistence_samples)
    angular = np.linalg.norm(oracle[:, [idx["foot_angular_velocity_x_rad_s"], idx["foot_angular_velocity_y_rad_s"]]], axis=1)
    tilt = _sustained(loaded & (oracle[:, idx["foot_tilt_change_rad"]] > c.tilt_change_rad)
                      & (angular > c.angular_rate_rad_s), c.persistence_samples)
    return {"incipient_risk": risk & ~confirmed, "slip_risk": risk,
            "confirmed_slip": confirmed, "sustained_sink": sink, "sustained_tilt": tilt}


def run_one(case_id: str, run_index: int) -> Run:
    case = CASES[case_id]
    model = mujoco.MjModel.from_xml_path(str(SCENES["front_rear"]))
    model.opt.timestep = PHYSICS_TIMESTEP_S
    for ground in ("ground_a", "ground_b"):
        apply_terrain_profile(model, TERRAIN_PROFILES[case["before"]], ground)
    data = mujoco.MjData(model)
    spec = make_expanded_run_specification("sand", "multisine", run_index, run_index)
    condition = ExcitationCondition(f"transition_{case_id}_{run_index:02d}", spec.initial_velocity_x,
        spec.initial_velocity_y, spec.base_height_offset, spec.base_roll_deg, spec.base_pitch_deg)
    qpos, dof = apply_excitation_condition(model, data, condition)
    support = VerticalElasticBandSupport(model, data, qpos, dof, .50)
    pulse = HorizontalPulse(TRANSITION_TIME_S, .100, float(case["horizontal_force_N"]), 1., 0.)
    exciter = HorizontalPulseExciter(model, data, pulse)
    reader = G1HilSensorReader(model, data); foot_ids = frozenset(reader.left_foot_geom_ids)
    ground_ids = [model.geom("ground_a").id, model.geom("ground_b").id]
    body = model.body("left_ankle_roll_link").id; velocity = np.zeros(6); wrench = np.zeros(6)
    times: list[float] = []; sensors: list[np.ndarray] = []; raw: list[tuple[float, ...]] = []
    switched = False; transition_sample = -1; state: dict[str, float] = {}; steps = 0; start = time.perf_counter()
    while data.time + 1e-12 < DURATION_S:
        if not switched and data.time >= TRANSITION_TIME_S - 1e-12:
            # mj_step advances qpos before returning; refresh derived body
            # quantities so the before/after comparison is at the same state.
            mujoco.mj_forward(model, data)
            before_qpos, before_qvel = data.qpos.copy(), data.qvel.copy()
            before_pose = data.xpos[body].copy(); mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0); before_speed = float(np.linalg.norm(velocity))
            before_contacts = data.ncon
            for ground in ("ground_a", "ground_b"):
                apply_terrain_profile(model, TERRAIN_PROFILES[case["after"]], ground)
            mujoco.mj_forward(model, data)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            state = {"qpos_max_abs_delta": float(np.max(np.abs(data.qpos - before_qpos))),
                "qvel_max_abs_delta": float(np.max(np.abs(data.qvel - before_qvel))),
                "foot_pose_max_abs_delta_m": float(np.max(np.abs(data.xpos[body] - before_pose))),
                "foot_velocity_abs_delta_mps": abs(float(np.linalg.norm(velocity)) - before_speed),
                "contact_count_before": float(before_contacts), "contact_count_after_forward": float(data.ncon)}
            switched = True
        support.apply(); exciter.apply(float(data.time));
        # Reuses the v2 bounded vertical load mechanism; it changes no labels.
        class Config: vertical_force_N = float(case["vertical_force_N"]); force_duration_s = .100
        _apply_vertical_pulse(data, exciter.body_id, float(data.time), Config())
        mujoco.mj_step(model, data); steps += 1
        if steps % PHYSICS_STEPS_PER_SAMPLE == 0:
            times.append(float(data.time)); sensors.append(reader.read_vector())
            raw.append(_foot_oracle(model, data, ground_ids, foot_ids, body, velocity, wrench))
            if switched and transition_sample < 0: transition_sample = len(times) - 1
    raw_array = np.asarray(raw, float); transition_sample = int(round(TRANSITION_TIME_S * SENSOR_RATE_HZ))
    # v2 diagnostic postprocess with the authoritative full-run T0 baseline.
    speed = np.linalg.norm(raw_array[:, 4:6], axis=1); depth = np.maximum(0., raw_array[transition_sample, 12] - raw_array[:, 12])
    tilt = np.linalg.norm(raw_array[:, 10:12] - raw_array[transition_sample, 10:12], axis=1)
    oracle = np.column_stack((raw_array[:, :13], speed, depth, tilt, raw_array[:, 13:]))
    metadata = {"run_id": condition.run_id, "case_id": case_id, "case_name": case["name"],
        "terrain_before": case["before"], "terrain_after": case["after"], "surface_family": "multisine",
        "surface_realization": run_index, "seed": MASTER_SEED + run_index, "run_index": run_index,
        "physics_rate_hz": 2000, "sensor_rate_hz": 1000, "physics_timestep_s": PHYSICS_TIMESTEP_S,
        "transition_t0_ms": TRANSITION_TIME_S * 1000., "transition_sample": transition_sample,
        "transition_type": "temporal_contact_profile_switch", "horizontal_force_N": case["horizontal_force_N"],
        "vertical_force_N": case["vertical_force_N"], "wall_time_s": time.perf_counter() - start}
    terrain = np.where(np.arange(len(times)) < transition_sample, case["before"], case["after"])
    return Run(metadata, np.asarray(times), np.asarray(sensors), oracle, terrain, state)


def onset_ms(mask: np.ndarray, transition_sample: int) -> float | None:
    hits = np.flatnonzero(mask & (np.arange(len(mask)) >= transition_sample))
    return None if not len(hits) else float(hits[0] - transition_sample)


def audit(runs: list[Run], labels: list[dict[str, np.ndarray]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    for run, labels_i in zip(runs, labels):
        m = run.metadata; t0 = int(m["transition_sample"]); dt = np.diff(run.time_s)
        terrain_ok = bool(np.all(run.terrain_gt[:t0] == m["terrain_before"]) and np.all(run.terrain_gt[t0:] == m["terrain_after"]))
        valid = bool(len(run.time_s) == 800 and run.fusion10.shape == (800, 10) and np.isfinite(run.fusion10).all() and np.isfinite(run.oracle).all() and np.allclose(dt, .001, atol=1e-9) and terrain_ok and run.state["qpos_max_abs_delta"] == 0 and run.state["qvel_max_abs_delta"] == 0)
        row = {**m, **run.state, "valid": valid, "terrain_gt_correct": terrain_ok,
            "sensor_spacing_valid": bool(np.allclose(dt, .001, atol=1e-9)), "fusion10_finite": bool(np.isfinite(run.fusion10).all()),
            "confirmed_slip_onset_ms": onset_ms(labels_i["confirmed_slip"], t0),
            "sustained_sink_onset_ms": onset_ms(labels_i["sustained_sink"], t0),
            "required_contact_samples": int((run.oracle[:, V2_ORACLE_INDEX["left_contact"]] > .5).sum())}
        rows.append(row)
    cases = {}
    for case_id in CASES:
        group = [row for row in rows if row["case_id"] == case_id]
        def stat(key: str) -> dict[str, float | None]:
            values = [float(row[key]) for row in group if row[key] is not None]
            return {"median_ms": None if not values else float(np.median(values)), "p95_ms": None if not values else float(np.percentile(values, 95))}
        cases[case_id] = {"valid_runs": sum(row["valid"] for row in group), "runs": len(group),
            "confirmed_slip_runs": sum(row["confirmed_slip_onset_ms"] is not None for row in group),
            "sustained_sink_runs": sum(row["sustained_sink_onset_ms"] is not None for row in group),
            "slip_onset": stat("confirmed_slip_onset_ms"), "sink_onset": stat("sustained_sink_onset_ms")}
    ready = len(rows) == 12 and all(row["valid"] for row in rows)
    return rows, {"dataset_name": "terrain_transition_v1", "pilot_runs": len(rows), "valid_runs": sum(row["valid"] for row in rows),
        "cases": cases, "continuity": {"all_qpos_qvel_zero": all(row["qpos_max_abs_delta"] == 0 and row["qvel_max_abs_delta"] == 0 for row in rows)},
        "fusion10": {"all_finite": all(row["fusion10_finite"] for row in rows)}, "terrain_gt": {"all_correct": all(row["terrain_gt_correct"] for row in rows)},
        "TERRAIN_TRANSITION_V1_PILOT_READY": ready}


def plot(output: Path, runs: list[Run], labels: list[dict[str, np.ndarray]]) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = output / "plots"; directory.mkdir(exist_ok=True)
    for case_id in CASES:
        i = next(i for i, r in enumerate(runs) if r.metadata["case_id"] == case_id); run, lab = runs[i], labels[i]; x = run.time_s * 1000.; t0 = float(run.metadata["transition_t0_ms"])
        fig, ax = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
        ax[0].step(x, (run.terrain_gt == run.metadata["terrain_after"]).astype(int), where="post"); ax[0].set_ylabel("terrain GT\n(after=1)")
        ax[1].plot(x, run.fusion10[:, :4]); ax[1].set_ylabel("FSR [N]")
        ax[2].plot(x, run.fusion10[:, 4:7]); ax[2].set_ylabel("Accel")
        ax[3].plot(x, run.fusion10[:, 7:10]); ax[3].set_ylabel("Gyro")
        ax[4].step(x, lab["confirmed_slip"], where="post", label="confirmed slip"); ax[4].step(x, lab["sustained_sink"], where="post", label="sustained sink"); ax[4].legend(); ax[4].set_ylabel("oracle")
        for a in ax: a.axvline(t0, color="k", ls="--"); a.grid(alpha=.25)
        for key, color in (("confirmed_slip", "tab:red"), ("sustained_sink", "tab:orange")):
            onset = onset_ms(lab[key], int(run.metadata["transition_sample"]))
            if onset is not None:
                for a in ax: a.axvline(t0 + onset, color=color, ls=":")
        ax[-1].set_xlabel("simulation time [ms]"); fig.suptitle(f"Case {case_id}: {run.metadata['terrain_before']} → {run.metadata['terrain_after']}"); fig.tight_layout(); fig.savefig(directory / f"case_{case_id.lower()}_representative.png", dpi=150); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--runs-per-case", type=int, default=3); parser.add_argument("--execute", action="store_true"); args = parser.parse_args()
    protocol = {"dataset_name": "terrain_transition_v1", "schema_version": 1, "master_seed": MASTER_SEED, "cases": CASES,
        "transition": {"type": "temporal_contact_profile_switch", "t0_ms": TRANSITION_TIME_S * 1000., "sample": int(TRANSITION_TIME_S * SENSOR_RATE_HZ), "continuous_state": "qpos/qvel never reset"},
        "physics_rate_hz": 2000, "sensor_rate_hz": 1000, "fusion10_channels": HIL_SENSOR_CHANNELS,
        "oracle_semantics": "frozen Fast Reflex v2 confirmed_slip and sustained_sink; terrain name never labels hazard", "frozen_calibration": FROZEN_CALIBRATION.as_dict()}
    if not args.execute: print(json.dumps(protocol, indent=2)); return
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    runs = [run_one(case_id, index) for case_id in CASES for index in range(args.runs_per_case)]
    labels = [label(run.oracle, int(run.metadata["transition_sample"])) for run in runs]
    rows, summary = audit(runs, labels); protocol["measured"] = {"runs": len(runs), "valid_runs": summary["valid_runs"]}
    np.savez_compressed(output / "transition_traces.npz", time_s=np.asarray([r.time_s for r in runs]), fusion10=np.asarray([r.fusion10 for r in runs], np.float32), terrain_gt=np.asarray([r.terrain_gt for r in runs]), oracle=np.asarray([r.oracle for r in runs], np.float32), oracle_channels=np.asarray(V2_ORACLE_CHANNELS), run_id=np.asarray([r.metadata["run_id"] for r in runs]), **{k: np.asarray([x[k] for x in labels], bool) for k in labels[0]})
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as f: writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8"); (output / "audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot(output, runs, labels); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
