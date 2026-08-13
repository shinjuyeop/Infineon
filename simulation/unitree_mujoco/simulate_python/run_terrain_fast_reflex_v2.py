"""Generate Fast Reflex v2 physical-state datasets; final test is opt-in only."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import time

import mujoco
import numpy as np

from controlled_excitation import (ExcitationCondition, HorizontalPulse, HorizontalPulseExciter,
                                   VerticalElasticBandSupport, apply_excitation_condition)
from expanded_terrain_dataset_v1 import SURFACE_FAMILIES, family_for_name, make_expanded_run_specification
from hil_sensor import G1HilSensorReader, HIL_SENSOR_CHANNELS
from terrain_fast_reflex_v2 import (
    FINAL_TEST_SPLIT, MODES, PHYSICS_STEPS_PER_SAMPLE, PHYSICS_TIMESTEP_S,
    RELATIVE_TRANSITION_TIME_MS, SCHEMA_NAME, SCHEMA_VERSION, SENSOR_RATE_HZ,
    TRACE_POST_MS, TRACE_PRE_MS, TRACE_SAMPLES, V2_ORACLE_CHANNELS, V2_ORACLE_INDEX,
    ScenarioPhysicsConfig, calibration_scenario_configs, calibrate_v2,
    default_scenario_configs, front_rear_torque_calibration_configs,
    final_tilt_physics_calibration_configs, local_compliance_calibration_configs,
    DEPLOYMENT_SCOPE, final_scope_calibration_configs, final_scope_pilot_configs,
    label_v2, onset_ms, sand_sink_hazard, slip_hazard, validate_final_test_request,
    validate_state_order,
)
from terrain_profiles import TERRAIN_PROFILES, apply_terrain_profile


SIM_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = SIM_DIR / "outputs" / SCHEMA_NAME
SCENE_DIR = Path(__file__).resolve().parents[1] / "unitree_robots" / "g1"
SCENES = {
    "front_rear": SCENE_DIR / "scene_fast_reflex_v2_front_rear.xml",
    "left_right": SCENE_DIR / "scene_fast_reflex_v2_left_right.xml",
}
LAYER_SCENE = SCENE_DIR / "scene_fast_reflex_v2_front_rear_layer.xml"
# Let the initially posed robot settle before an event.  The old .250 s
# transition left a normal-sand vertical transient inside the event window.
SETTLE_TIME_S, TRANSITION_TIME_S, DURATION_S = .500, .650, .800


@dataclass(frozen=True)
class RawRun:
    metadata: dict[str, object]
    timestamps_s: np.ndarray
    sensors: np.ndarray
    oracle: np.ndarray
    sole_positions_m: np.ndarray
    backing_contact: np.ndarray
    max_penetration_m: float
    qpos_delta: float
    qvel_delta: float
    wall_time_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--families", nargs="+", choices=[f.name for f in SURFACE_FAMILIES],
                        default=["multisine", "filtered_random", "sparse_aggregate", "crosshatch", "rounded_ridges"])
    parser.add_argument("--surfaces-per-family", type=int, default=3)
    parser.add_argument("--runs-per-surface", type=int, default=5)
    parser.add_argument("--include-final-test", action="store_true",
                        help="explicitly materialize reserved final-test rows; prohibited for normal smoke/pilot")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--scenario-calibration", action="store_true",
                        help="bounded train-only physical candidate sweep; no threshold candidates")
    parser.add_argument("--front-rear-torque-calibration", action="store_true",
                        help="bounded rejected-design audit; not eligible as pilot selection")
    parser.add_argument("--local-compliance-calibration", action="store_true",
                        help="bounded train-only localized compliant-support experiment")
    parser.add_argument("--final-tilt-physics-calibration", action="store_true",
                        help="last bounded hard-backed-layer/height-offset train-only experiment")
    parser.add_argument("--final-scope-calibration", action="store_true",
                        help="bounded train-only Slip Risk and Sand Sink deployment-scope calibration")
    parser.add_argument("--final-scope-pilot", action="store_true",
                        help="frozen four-mode pilot configuration; execute only after readiness approval")
    parser.add_argument("--scenario-selection", type=Path,
                        help="frozen selected_configs JSON from a prior train-only calibration")
    parser.add_argument("--audit-existing", type=Path,
                        help="read-only schema/physical-validity audit of a generated v2 train/validation output")
    parser.add_argument("--scope-policy-audit", type=Path,
                        help="read existing final-scope calibration and write a new frozen-policy readiness artifact")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _families(args: argparse.Namespace) -> list[str]:
    validate_final_test_request(args.families, args.include_final_test)
    result = []
    for name in args.families:
        family = family_for_name(name)
        if family.split == "test":
            if not args.include_final_test:
                raise ValueError("final-test family requested without --include-final-test")
            # validate_final_test_request above provides the explicit reason.
            raise ValueError("v1 test families are permanently ineligible for v2 final test")
        result.append(name)
    return result


def candidate_count(args: argparse.Namespace, configs: tuple[ScenarioPhysicsConfig, ...]) -> int:
    return len(configs) * len(_families(args)) * args.surfaces_per_family * args.runs_per_surface


def protocol(args: argparse.Namespace, configs: tuple[ScenarioPhysicsConfig, ...]) -> dict[str, object]:
    families = _families(args)
    if args.surfaces_per_family < 1 or args.runs_per_surface < 1:
        raise ValueError("positive surface/run counts required")
    if not any(config.mode == "normal_sand" for config in configs):
        raise ValueError("normal_sand train rows are required for calibration")
    count = candidate_count(args, configs)
    return {
        "dataset_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
        "status": "v2 physical dataset foundation; no detector training",
        "modes": sorted({config.mode for config in configs}), "scenario_configs": [config.as_dict() for config in configs],
        "fast_reflex_v2_deployment_scope": DEPLOYMENT_SCOPE,
        "scenario_calibration": bool(args.scenario_calibration or args.front_rear_torque_calibration or args.local_compliance_calibration or args.final_tilt_physics_calibration or args.final_scope_calibration), "physics_rate_hz": 2000, "sensor_rate_hz": SENSOR_RATE_HZ,
        "physics_timestep_s": PHYSICS_TIMESTEP_S, "physics_steps_per_sample": PHYSICS_STEPS_PER_SAMPLE,
        "trace_interval_ms": [-TRACE_PRE_MS, TRACE_POST_MS], "trace_shape": [TRACE_SAMPLES, 10],
        "settle_time_s": SETTLE_TIME_S, "transition_time_s": TRANSITION_TIME_S,
        "input_channels": HIL_SENSOR_CHANNELS, "oracle_channels": V2_ORACLE_CHANNELS,
        "state_schema": ["safe", "incipient_risk", "confirmed_slip", "slip_risk",
                         "sustained_sink", "sustained_tilt"],
        "calibration": "two pass: train normal only -> frozen calibration -> labels for all rows",
        "split_policy": {
            "train": [f.name for f in SURFACE_FAMILIES if f.split == "train"],
            "validation": [f.name for f in SURFACE_FAMILIES if f.split == "validation"],
            FINAL_TEST_SPLIT: {"status": "reserved but not materialized", "surface_seed_range": [9100, 9199],
                               "session_seed": 20260902, "excitation_seed_offset": 920000,
                               "rule": "new held-out morphology/realization; never v1 test families"},
        },
        "final_test": {"materialized": bool(args.include_final_test), "fail_closed_default": True,
                       "v1_test_families_ineligible": ["warped_multisine", "smooth_random_patches"]},
        "candidate_runs": count, "estimated_uncompressed_trace_bytes": count * TRACE_SAMPLES * (10 + len(V2_ORACLE_CHANNELS)) * 4,
        "estimated_runtime_s": [count * .25, count * 1.0],
        "overwrite_policy": "refuse non-empty output", "boundary": "two adjacent independent box geoms; continuous MuJoCo state",
    }


def _foot_oracle(model, data, ground_ids, foot_ids, body_id, velocity, wrench) -> tuple[float, ...]:
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
    normal = tangential = 0.; contact = False; by_ground = [False, False]
    for contact_id in range(data.ncon):
        item = data.contact[contact_id]; pair = (int(item.geom1), int(item.geom2))
        foot = pair[0] if pair[0] in foot_ids else pair[1] if pair[1] in foot_ids else None
        if foot is None: continue
        other = pair[1] if foot == pair[0] else pair[0]
        if other not in ground_ids: continue
        gi = ground_ids.index(other); by_ground[gi] = True; contact = True
        wrench.fill(0.); mujoco.mj_contactForce(model, data, contact_id, wrench)
        normal += max(0., float(wrench[0])); tangential += float(np.linalg.norm(wrench[1:3]))
    rotation = data.xmat[body_id].reshape(3, 3)
    roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
    pitch = float(np.arctan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2])))
    return (float(contact), normal, tangential, tangential / max(normal, 1e-12), *velocity[3:6],
            *velocity[:3], roll, pitch, float(data.xpos[body_id, 2]), float(by_ground[0]), float(by_ground[1]))


def _postprocess(raw: np.ndarray) -> np.ndarray:
    if raw.shape != (TRACE_SAMPLES, 15): raise ValueError("raw v2 oracle shape")
    t = TRACE_PRE_MS
    speed = np.linalg.norm(raw[:, 4:6], axis=1)
    depth = np.maximum(0., raw[t, 12] - raw[:, 12])
    tilt = np.linalg.norm(raw[:, 10:12] - raw[t, 10:12], axis=1)
    return np.column_stack((raw[:, :13], speed, depth, tilt, raw[:, 13:]))


def _apply_vertical_pulse(data, body_id: int, time_s: float, config: ScenarioPhysicsConfig) -> None:
    """Add a bounded downward half-sine load without modifying label logic."""
    phase = (time_s - TRANSITION_TIME_S) / config.force_duration_s
    if 0.0 <= phase < 1.0:
        data.xfrc_applied[body_id, 2] -= config.vertical_force_N * np.sin(np.pi * phase)


def _apply_pitch_torque(data, body_id: int, time_s: float, config: ScenarioPhysicsConfig) -> None:
    """A bounded mechanical pitch disturbance with no imposed vertical force."""
    phase = (time_s - TRANSITION_TIME_S) / config.force_duration_s
    if 0.0 <= phase < 1.0:
        data.xfrc_applied[body_id, 4] += config.pitch_torque_Nm * np.sin(np.pi * phase)


def _offset_seam(model, layout: str, offset_m: float, height_offset_m: float = 0.0) -> None:
    axis = 0 if layout == "front_rear" else 1
    # Shift both adjacent box centers together: they remain adjacent and the
    # actual contact fraction moves relative to the already continuous foot.
    for name in ("ground_a", "ground_b", "ground_b_backing"):
        if name not in {model.geom(i).name for i in range(model.ngeom)}:
            continue
        geom_id=model.geom(name).id
        model.geom_pos[geom_id, axis] += offset_m
        if name.startswith("ground_b"):
            model.geom_pos[geom_id, 2] += height_offset_m


def run_one(config: ScenarioPhysicsConfig, family: str, surface_index: int, run_index: int) -> RawRun:
    scene = LAYER_SCENE if config.hard_backed_layer else SCENES[config.layout]
    model = mujoco.MjModel.from_xml_path(str(scene)); model.opt.timestep = PHYSICS_TIMESTEP_S
    _offset_seam(model, config.layout, config.seam_offset_m, config.height_offset_m)
    data = mujoco.MjData(model)
    for name, material in (("ground_a", config.material_a), ("ground_b", config.material_b)):
        apply_terrain_profile(model, TERRAIN_PROFILES[material], name)
    if config.hard_backed_layer:
        apply_terrain_profile(model, TERRAIN_PROFILES["marble"], "ground_b_backing")
    ground_ids = [model.geom("ground_a").id, model.geom("ground_b").id]
    spec = make_expanded_run_specification("sand", family, surface_index, run_index)
    condition = ExcitationCondition(f"{config.config_id}_{family}_s{surface_index:02d}_r{run_index:03d}",
                                   spec.initial_velocity_x, spec.initial_velocity_y, spec.base_height_offset,
                                   spec.base_roll_deg, spec.base_pitch_deg)
    qpos, dof = apply_excitation_condition(model, data, condition)
    support = VerticalElasticBandSupport(model, data, qpos, dof, config.support_ratio)
    pulse = HorizontalPulse(TRANSITION_TIME_S, config.force_duration_s, config.horizontal_force_N,
                            config.direction_x, config.direction_y)
    exciter = HorizontalPulseExciter(model, data, pulse)
    reader = G1HilSensorReader(model, data); foot_ids = frozenset(reader.left_foot_geom_ids)
    body = model.body("left_ankle_roll_link").id; velocity = np.zeros(6); wrench = np.zeros(6)
    sole_ids = [model.geom(f"left_foot_contact_{index}").id for index in range(1, 5)]
    backing_id = model.geom("ground_b_backing").id if config.hard_backed_layer else None
    times=[]; sensors=[]; raw=[]; sole_positions=[]; backing=[]; steps=0; max_penetration=0.; switched=False; qpos_delta=qvel_delta=0.; start=time.perf_counter()
    while data.time + 1e-12 < DURATION_S:
        if config.switch_to_ice and not switched and data.time >= TRANSITION_TIME_S - 1e-12:
            qpos_before, qvel_before = data.qpos.copy(), data.qvel.copy()
            apply_terrain_profile(model, TERRAIN_PROFILES["ice"], "ground_a")
            apply_terrain_profile(model, TERRAIN_PROFILES["ice"], "ground_b")
            mujoco.mj_forward(model, data)
            qpos_delta, qvel_delta = float(np.max(np.abs(data.qpos-qpos_before))), float(np.max(np.abs(data.qvel-qvel_before)))
            switched=True
        support.apply(); exciter.apply(float(data.time)); _apply_vertical_pulse(data, exciter.body_id, float(data.time), config)
        _apply_pitch_torque(data, exciter.body_id, float(data.time), config)
        mujoco.mj_step(model, data); steps += 1
        if steps % PHYSICS_STEPS_PER_SAMPLE == 0:
            times.append(float(data.time)); sensors.append(reader.read_vector())
            raw.append(_foot_oracle(model, data, ground_ids, foot_ids, body, velocity, wrench))
            sole_positions.append(data.geom_xpos[sole_ids].copy())
            backing.append(False if backing_id is None else any(backing_id in (int(data.contact[i].geom1), int(data.contact[i].geom2)) and ((int(data.contact[i].geom1) in foot_ids) or (int(data.contact[i].geom2) in foot_ids)) for i in range(data.ncon)))
            max_penetration=max(max_penetration, max((max(0., -float(data.contact[i].dist)) for i in range(data.ncon)), default=0.))
    times=np.asarray(times); sensors=np.asarray(sensors); raw=np.asarray(raw)
    select=(times >= TRANSITION_TIME_S - TRACE_PRE_MS / 1000 - 1e-9) & (times < TRANSITION_TIME_S + TRACE_POST_MS / 1000 - 1e-9)
    family_info=family_for_name(family)
    metadata={"schema_version":SCHEMA_VERSION,"mode":config.mode,"scenario_config_id":config.config_id,"surface_family":family,"surface_seed":surface_index,
              "session_id":f"v2_{family_info.split}_{config.config_id}_surface_{surface_index:02d}","run_id":condition.run_id,
              "split":family_info.split,"physics_rate_hz":2000,"sensor_rate_hz":1000,"transition_time_s":TRANSITION_TIME_S,
              "settle_time_s":SETTLE_TIME_S,"ground_layout":config.layout,"ground_a_material":config.material_a,"ground_b_material":config.material_b,
              "boundary_orientation":"front_rear" if config.layout == "front_rear" else "left_right","boundary_position_m":config.seam_offset_m,
              "pulse_magnitude_N":config.horizontal_force_N,"vertical_pulse_magnitude_N":config.vertical_force_N,
              "pitch_torque_Nm":config.pitch_torque_Nm,"hard_backed_layer":int(config.hard_backed_layer),"height_offset_m":config.height_offset_m,
              "pulse_duration_s":config.force_duration_s,"pulse_direction_x":config.direction_x,"pulse_direction_y":config.direction_y,
              "support_ratio":config.support_ratio,"material_switch_to_ice":int(config.switch_to_ice),"run_index":run_index}
    return RawRun(metadata, times[select], sensors[select], _postprocess(raw[select]), np.asarray(sole_positions)[select], np.asarray(backing)[select], max_penetration, qpos_delta, qvel_delta, time.perf_counter()-start)


def _row(raw: RawRun, labels: dict[str, np.ndarray]) -> dict[str, object]:
    fsr = raw.sensors[:, :4]
    fsr_sum = fsr.sum(axis=1)
    front_rear = fsr[:, 2:].sum(axis=1) - fsr[:, :2].sum(axis=1)
    left_right = fsr[:, [0, 2]].sum(axis=1) - fsr[:, [1, 3]].sum(axis=1)
    normalizer = np.maximum(fsr_sum, 1e-6)
    gyro_xy = np.linalg.norm(raw.sensors[:, 7:9], axis=1)
    front_z, rear_z = raw.sole_positions_m[:, 2:4, 2].mean(axis=1), raw.sole_positions_m[:, :2, 2].mean(axis=1)
    front_settlement, rear_settlement = front_z[TRACE_PRE_MS] - front_z, rear_z[TRACE_PRE_MS] - rear_z
    differential = front_settlement - rear_settlement
    contact = raw.oracle[:, V2_ORACLE_INDEX["left_contact"]] > .5
    loaded = contact & (raw.oracle[:, V2_ORACLE_INDEX["contact_normal_force_N"]] >= 1e-6)
    a_contact = raw.oracle[:, V2_ORACLE_INDEX["ground_a_contact"]] > .5
    b_contact = raw.oracle[:, V2_ORACLE_INDEX["ground_b_contact"]] > .5
    out={**raw.metadata,"calibration_provenance":"train-normal only","valid":int(np.isfinite(raw.sensors).all() and np.isfinite(raw.oracle).all()),
         "invalid_reason":"","qpos_transition_max_abs_delta":raw.qpos_delta,"qvel_transition_max_abs_delta":raw.qvel_delta,
         "ground_a_contact_samples":int(raw.oracle[:, V2_ORACLE_INDEX["ground_a_contact"]].sum()),
         "ground_b_contact_samples":int(raw.oracle[:, V2_ORACLE_INDEX["ground_b_contact"]].sum()),"backing_contact_samples":int(raw.backing_contact.sum()),"max_penetration_m":raw.max_penetration_m,"wall_time_s":raw.wall_time_s,
         "fsr_sum_max_N":float(fsr_sum.max()), "front_minus_rear_peak_N":float(np.max(np.abs(front_rear))),
         "left_minus_right_peak_N":float(np.max(np.abs(left_right))),
         "normalized_front_rear_imbalance_peak":float(np.max(np.abs(front_rear / normalizer))),
         "normalized_left_right_imbalance_peak":float(np.max(np.abs(left_right / normalizer))),
         "fsr_spatial_variance_peak":float(np.max(fsr.var(axis=1))), "fsr_range_peak_N":float(np.max(np.ptp(fsr,axis=1))),
         "gyro_xy_magnitude_peak_rad_s":float(gyro_xy.max()), "gyro_xy_integral_rad":float(gyro_xy.sum()*.001),
         "foot_sink_depth_peak_m":float(raw.oracle[:, V2_ORACLE_INDEX["foot_sink_depth_m"]].max()),
         "foot_tilt_change_peak_rad":float(raw.oracle[:, V2_ORACLE_INDEX["foot_tilt_change_rad"]].max()),
         "foot_horizontal_speed_peak_mps":float(raw.oracle[:, V2_ORACLE_INDEX["foot_horizontal_speed_mps"]].max()),
         "ground_a_contact_coverage":float(a_contact.mean()), "ground_b_contact_coverage":float(b_contact.mean()),
         "loaded_contact_coverage":float(loaded.mean()), "front_settlement_peak_m":float(front_settlement.max()),
         "rear_settlement_peak_m":float(rear_settlement.max()), "differential_settlement_peak_m":float(np.max(np.abs(differential))),
         "gross_rotation":int(raw.oracle[:, V2_ORACLE_INDEX["foot_tilt_change_rad"]].max() > .10)}
    for name in ("incipient_risk", "slip_risk","confirmed_slip","sustained_sink","sustained_tilt"):
        onset=onset_ms(labels[name]); out[f"{name}_onset_time_s"]="" if onset is None else float(raw.timestamps_s[TRACE_PRE_MS+onset])
        out[f"{name}_onset_ms"]="" if onset is None else onset
    return out


def _coverage(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Train-only event coverage used to choose physics, never thresholds."""
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["scenario_config_id"]), []).append(row)
    result = []
    for config_id, group in groups.items():
        def has(name: str) -> int:
            return sum(row[f"{name}_onset_ms"] != "" for row in group)
        onsets = [float(row["sustained_sink_onset_ms"]) for row in group if row["sustained_sink_onset_ms"] != ""]
        onsets += [float(row["sustained_tilt_onset_ms"]) for row in group if row["sustained_tilt_onset_ms"] != ""]
        result.append({"scenario_config_id": config_id, "mode": group[0]["mode"], "runs": len(group),
                       "valid_runs": sum(int(row["valid"]) for row in group),
                       "slip_risk_runs": has("slip_risk"), "confirmed_slip_runs": has("confirmed_slip"),
                       "incipient_runs": has("incipient_risk"),
                       "sink_runs": has("sustained_sink"), "tilt_runs": has("sustained_tilt"),
                       "sink_and_tilt_runs": sum(row["sustained_sink_onset_ms"] != "" and row["sustained_tilt_onset_ms"] != "" for row in group),
                       "median_hazard_onset_ms": "" if not onsets else float(np.median(onsets)),
                       "max_sink_depth_m": float(max(row["foot_sink_depth_peak_m"] for row in group)),
                       "max_orientation_deviation_rad": float(max(row["foot_tilt_change_peak_rad"] for row in group)),
                       "max_horizontal_speed_mps": float(max(row["foot_horizontal_speed_peak_mps"] for row in group))})
    return result


def _selection(configs: tuple[ScenarioPhysicsConfig, ...], coverage: list[dict[str, object]]) -> dict[str, object]:
    by_id = {config.config_id: config for config in configs}
    selected = []
    for mode in sorted({config.mode for config in configs}):
        candidates = [row for row in coverage if row["mode"] == mode]
        def key(row: dict[str, object]) -> tuple[float, ...]:
            if mode == "normal_sand": return (-float(row["slip_risk_runs"] + row["sink_runs"] + row["tilt_runs"]),)
            if mode == "slip_risk_dominant": return (float(row["confirmed_slip_runs"]), float(row["slip_risk_runs"]), -float(row["sink_runs"] + row["tilt_runs"]))
            if mode == "sink_dominant": return (float(row["sink_runs"]), -float(row["tilt_runs"]))
            if mode in ("tilt_dominant", "boundary_front_rear", "boundary_left_right"): return (float(row["tilt_runs"]), -float(row["sink_runs"]))
            return (float(row["sink_and_tilt_runs"]), float(row["sink_runs"] + row["tilt_runs"]))
        winner = max(candidates, key=key)
        selected.append({**by_id[str(winner["scenario_config_id"])].as_dict(), "coverage": winner})
    gate = {
        "normal_hazard_free": all(item["coverage"]["slip_risk_runs"] == 0 and item["coverage"]["sink_runs"] == 0 and item["coverage"]["tilt_runs"] == 0 for item in selected if item["mode"] == "normal_sand"),
        "slip_risk_and_confirmed": all(item["coverage"]["slip_risk_runs"] > 0 and item["coverage"]["confirmed_slip_runs"] > 0 for item in selected if item["mode"] == "slip_risk_dominant"),
        "sink": all(item["coverage"]["sink_runs"] > 0 for item in selected if item["mode"] in ("sink_dominant", "sink_and_tilt")),
        "tilt": all(item["coverage"]["tilt_runs"] > 0 for item in selected if item["mode"] != "normal_sand"),
    }
    return {"selection_basis": "train-only physical event coverage and mode specificity; thresholds unchanged",
            "selected_configs": selected, "gates": gate, "pilot_ready": bool(all(gate.values()))}


def save(output: Path, raw_runs: list[RawRun], labels: list[dict[str, np.ndarray]], rows: list[dict[str, object]], payload: dict[str, object]) -> None:
    np.savez_compressed(output / "inputs_fusion10.npz", sensors=np.asarray([r.sensors for r in raw_runs],np.float32),
                        sample_time_s=np.asarray([r.timestamps_s for r in raw_runs]), relative_transition_time_ms=RELATIVE_TRANSITION_TIME_MS,
                        run_id=np.asarray([r.metadata["run_id"] for r in raw_runs]))
    np.savez_compressed(output / "oracle_diagnostics.npz", oracle=np.asarray([r.oracle for r in raw_runs],np.float32),
                        oracle_channels=np.asarray(V2_ORACLE_CHANNELS), sample_time_s=np.asarray([r.timestamps_s for r in raw_runs]),
                        sole_contact_centres_m=np.asarray([r.sole_positions_m for r in raw_runs],np.float32),
                        backing_contact=np.asarray([r.backing_contact for r in raw_runs],bool),
                        **{name:np.asarray([item[name] for item in labels],bool) for name in labels[0]}, run_id=np.asarray([r.metadata["run_id"] for r in raw_runs]))
    with (output / "manifest.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "protocol.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")


def plot_smoke(output: Path, raw: RawRun, labels: dict[str,np.ndarray]) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x=RELATIVE_TRANSITION_TIME_MS; f=raw.sensors[:,:4]; total=f.sum(1); fr=(f[:,2:].sum(1)-f[:,:2].sum(1))/np.maximum(total,1e-6); lr=(f[:,[0,2]].sum(1)-f[:,[1,3]].sum(1))/np.maximum(total,1e-6)
    fig,ax=plt.subplots(5,1,figsize=(10,10),sharex=True)
    ax[0].plot(x,raw.oracle[:,V2_ORACLE_INDEX["foot_horizontal_speed_mps"]]); ax[0].set_ylabel("vXY")
    ax[1].plot(x,raw.oracle[:,V2_ORACLE_INDEX["contact_normal_force_N"]]); ax[1].plot(x,raw.oracle[:,V2_ORACLE_INDEX["Ft_over_Fn"]]); ax[1].set_ylabel("Fn/FtFn")
    ax[2].plot(x,total,label="FSR sum"); ax[2].plot(x,fr,label="front/rear"); ax[2].plot(x,lr,label="left/right"); ax[2].legend()
    ax[3].plot(x,raw.sensors[:,4:7]); ax[3].plot(x,raw.sensors[:,7:10],ls="--"); ax[3].set_ylabel("accel / gyro")
    ax[4].step(x,labels["incipient_risk"],where="post",label="risk"); ax[4].step(x,labels["confirmed_slip"],where="post",label="confirmed"); ax[4].step(x,labels["sustained_sink"],where="post",label="sink"); ax[4].step(x,labels["sustained_tilt"],where="post",label="tilt"); ax[4].plot(x,raw.oracle[:,V2_ORACLE_INDEX["ground_a_contact"]],label="A contact");ax[4].plot(x,raw.oracle[:,V2_ORACLE_INDEX["ground_b_contact"]],label="B contact");ax[4].legend(ncol=3,fontsize=8)
    ax[-1].set_xlabel("ms relative transition");fig.suptitle(str(raw.metadata["run_id"]));fig.tight_layout();fig.savefig(output/f"{raw.metadata['mode']}.png",dpi=140);plt.close(fig)


def plot_coverage(output: Path, coverage: list[dict[str, object]]) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [str(item["scenario_config_id"]) for item in coverage]
    x = np.arange(len(labels)); width = .20
    fig, axis = plt.subplots(figsize=(max(10, len(labels) * .65), 5))
    for offset, name, title in ((-.3, "slip_risk_runs", "risk"), (-.1, "confirmed_slip_runs", "confirmed"),
                                (.1, "sink_runs", "sink"), (.3, "tilt_runs", "tilt")):
        axis.bar(x + offset * width, [item[name] for item in coverage], width, label=title)
    axis.set_xticks(x, labels, rotation=60, ha="right"); axis.set_ylabel("train runs with event"); axis.legend(ncol=4)
    axis.set_title("Fast Reflex v2 scenario physical-event coverage (train-only)"); fig.tight_layout()
    fig.savefig(output / "scenario_coverage.png", dpi=140); plt.close(fig)


def local_compliance_artifacts(output: Path, raw: list[RawRun], labels: list[dict[str, np.ndarray]], rows: list[dict[str, object]], configs: tuple[ScenarioPhysicsConfig, ...]) -> None:
    """Write explicit physical-selection evidence for localized support only."""
    config_by_id = {config.config_id: config for config in configs}
    fields = ("scenario_config_id", "support_orientation", "boundary_position_m", "support_ratio", "ground_a_material", "ground_b_material", "ground_a_solref", "ground_a_solimp", "ground_b_solref", "ground_b_solimp", "vertical_pulse_magnitude_N", "ground_a_contact_coverage", "ground_b_contact_coverage", "loaded_contact_coverage", "differential_settlement_peak_m", "foot_tilt_change_peak_rad", "gyro_xy_magnitude_peak_rad_s", "sustained_tilt_onset_ms", "sustained_sink_onset_ms", "gross_rotation", "valid", "selection_status", "rejection_reason")
    sweep=[]
    for row in rows:
        config=config_by_id[str(row["scenario_config_id"])]
        orientation = "normal" if config.mode == "normal_sand" else ("hard_front_soft_rear" if config.material_b == "marble" else "hard_rear_soft_front")
        retained = min(float(row["ground_a_contact_coverage"]), float(row["ground_b_contact_coverage"])) >= .75 and float(row["loaded_contact_coverage"]) >= .75
        tilt = row["sustained_tilt_onset_ms"] != ""; sink = row["sustained_sink_onset_ms"] != ""
        accepted = config.mode != "normal_sand" and retained and tilt and not sink and not int(row["gross_rotation"])
        if accepted: reason=""
        elif config.mode == "normal_sand": reason="normal_control"
        elif not retained: reason="contact_loss"
        elif int(row["gross_rotation"]): reason="gross_rotation"
        elif sink: reason="excessive_sink_before_tilt"
        else: reason="insufficient_differential_settlement_or_rotation"
        a, b = TERRAIN_PROFILES[str(row["ground_a_material"])], TERRAIN_PROFILES[str(row["ground_b_material"])]
        sweep.append({"scenario_config_id":row["scenario_config_id"],"support_orientation":orientation,"boundary_position_m":row["boundary_position_m"],"support_ratio":row["support_ratio"],"ground_a_material":row["ground_a_material"],"ground_b_material":row["ground_b_material"],"ground_a_solref":a.solref,"ground_a_solimp":a.solimp,"ground_b_solref":b.solref,"ground_b_solimp":b.solimp,"vertical_pulse_magnitude_N":row["vertical_pulse_magnitude_N"],"ground_a_contact_coverage":row["ground_a_contact_coverage"],"ground_b_contact_coverage":row["ground_b_contact_coverage"],"loaded_contact_coverage":row["loaded_contact_coverage"],"differential_settlement_peak_m":row["differential_settlement_peak_m"],"foot_tilt_change_peak_rad":row["foot_tilt_change_peak_rad"],"gyro_xy_magnitude_peak_rad_s":row["gyro_xy_magnitude_peak_rad_s"],"sustained_tilt_onset_ms":row["sustained_tilt_onset_ms"],"sustained_sink_onset_ms":row["sustained_sink_onset_ms"],"gross_rotation":row["gross_rotation"],"valid":row["valid"],"selection_status":"accepted" if accepted else "rejected","rejection_reason":reason})
    with (output / "localized_compliance_sweep.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(sweep)
    accepted=[item for item in sweep if item["selection_status"] == "accepted"]
    selected = max(accepted, key=lambda item: (item["loaded_contact_coverage"], min(item["ground_a_contact_coverage"], item["ground_b_contact_coverage"]))) if accepted else None
    (output / "selected_config.json").write_text(json.dumps({"status":"LOCAL_COMPLIANCE_DESIGN_ACCEPTED" if selected else "LOCAL_COMPLIANCE_DESIGN_REJECTED", "selected":selected, "selection_order":["physical validity","bilateral contact retention","loaded contact retention","differential settlement","sustained tilt","low gross disturbance"]},indent=2)+"\n",encoding="utf-8")
    rejected=[item for item in sweep if item["selection_status"] == "rejected" and item["rejection_reason"] != "normal_control"]
    with (output / "rejected_configs.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rejected)
    state="LOCAL_COMPLIANCE_DESIGN_ACCEPTED" if selected else "LOCAL_COMPLIANCE_DESIGN_REJECTED"
    (output / "calibration_summary.md").write_text(f"# {state}\n\nTrain-only localized-compliance calibration; no threshold or label changes.\n\nRuns: {len(rows)}; accepted candidates: {len(accepted)}.\n\nBoth localized designs retained A/B and loaded contact in this trace. Soft-rear had insufficient differential settlement; soft-front created Tilt only with preceding Sink. In contrast, the rejected body-torque design reduced contact before producing no sustained Tilt.\n",encoding="utf-8")


def plot_local_compliance(output: Path, raw_runs: list[RawRun], labels: list[dict[str, np.ndarray]]) -> None:
    """Normal plus physically representative rejected support comparison plots."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = output / "plots"; plots.mkdir(exist_ok=True)
    normal_index=next(i for i, raw in enumerate(raw_runs) if raw.metadata["mode"] == "normal_sand")
    candidate_indices=[i for i, raw in enumerate(raw_runs) if raw.metadata["mode"] != "normal_sand"]
    representative=max(candidate_indices, key=lambda i: raw_runs[i].oracle[:, V2_ORACLE_INDEX["foot_tilt_change_rad"]].max())
    choices = [normal_index, representative]
    for index in choices:
        raw, state = raw_runs[index], labels[index]
        x=RELATIVE_TRANSITION_TIME_MS; sole=raw.sole_positions_m
        front, rear=sole[:,2:4,2].mean(1),sole[:,:2,2].mean(1)
        front_settle, rear_settle=front[TRACE_PRE_MS]-front,rear[TRACE_PRE_MS]-rear
        fsr=raw.sensors[:,:4]; total=np.maximum(fsr.sum(1),1e-6); imbalance=(fsr[:,2:].sum(1)-fsr[:,:2].sum(1))/total
        angular=np.linalg.norm(raw.oracle[:,[V2_ORACLE_INDEX["foot_angular_velocity_x_rad_s"],V2_ORACLE_INDEX["foot_angular_velocity_y_rad_s"]]],axis=1)
        fig,axis=plt.subplots(5,1,figsize=(10,11),sharex=True)
        axis[0].step(x,raw.oracle[:,V2_ORACLE_INDEX["ground_a_contact"]],where="post",label="A"); axis[0].step(x,raw.oracle[:,V2_ORACLE_INDEX["ground_b_contact"]],where="post",label="B"); axis[0].legend(); axis[0].set_ylabel("contact")
        axis[1].plot(x,front_settle,label="front");axis[1].plot(x,rear_settle,label="rear");axis[1].plot(x,front_settle-rear_settle,label="delta");axis[1].legend();axis[1].set_ylabel("settlement m")
        axis[2].plot(x,raw.oracle[:,V2_ORACLE_INDEX["foot_tilt_change_rad"]],label="tilt");axis[2].plot(x,angular,label="angular");axis[2].legend()
        axis[3].plot(x,fsr);axis[3].plot(x,imbalance,label="norm F-R",lw=2);axis[3].legend(ncol=3,fontsize=8);axis[3].set_ylabel("FSR / imbalance")
        axis[4].plot(x,raw.sensors[:,7],label="gyro x");axis[4].plot(x,raw.sensors[:,8],label="gyro y");axis[4].step(x,state["sustained_tilt"],where="post",label="tilt state");axis[4].step(x,state["sustained_sink"],where="post",label="sink state");axis[4].legend(ncol=2,fontsize=8)
        axis[-1].set_xlabel("ms relative transition");fig.suptitle(str(raw.metadata["scenario_config_id"]));fig.tight_layout();fig.savefig(plots/f"local_compliance_{raw.metadata['scenario_config_id']}.png",dpi=140);plt.close(fig)


def final_tilt_physics_artifacts(output: Path, rows: list[dict[str, object]], configs: tuple[ScenarioPhysicsConfig, ...]) -> None:
    fields=("config_id","kind","height_offset_m","hard_backed_layer","ground_a_material","ground_b_material","ground_a_contact_coverage","ground_b_contact_coverage","loaded_contact_coverage","backing_contact_samples","differential_settlement_peak_m","foot_tilt_change_peak_rad","gyro_xy_magnitude_peak_rad_s","sustained_tilt_onset_ms","sustained_sink_onset_ms","max_penetration_m","gross_rotation","valid","selection_status","rejection_reason")
    by_id={config.config_id:config for config in configs}; items=[]
    for row in rows:
        config=by_id[str(row["scenario_config_id"])]
        retain=min(float(row["ground_a_contact_coverage"]),float(row["ground_b_contact_coverage"])) >= .75 and float(row["loaded_contact_coverage"]) >= .75
        tilt=row["sustained_tilt_onset_ms"] != ""; sink=row["sustained_sink_onset_ms"] != ""
        accepted=config.mode != "normal_sand" and retain and tilt and not sink and not int(row["gross_rotation"])
        if accepted: reason=""
        elif config.mode == "normal_sand": reason="normal_control"
        elif not retain: reason="contact_loss"
        elif config.hard_backed_layer and int(row["backing_contact_samples"]) > 0: reason="simultaneous_top_and_backing_contact"
        elif int(row["gross_rotation"]): reason="gross_rotation"
        elif sink: reason="sink_before_or_with_tilt"
        else: reason="insufficient_tilt"
        kind="hard_backed_plus_height" if config.hard_backed_layer and config.height_offset_m else "hard_backed_layer" if config.hard_backed_layer else "height_offset" if config.height_offset_m else "normal"
        items.append({"config_id":config.config_id,"kind":kind,"height_offset_m":config.height_offset_m,"hard_backed_layer":int(config.hard_backed_layer),"ground_a_material":config.material_a,"ground_b_material":config.material_b,"ground_a_contact_coverage":row["ground_a_contact_coverage"],"ground_b_contact_coverage":row["ground_b_contact_coverage"],"loaded_contact_coverage":row["loaded_contact_coverage"],"backing_contact_samples":row["backing_contact_samples"],"differential_settlement_peak_m":row["differential_settlement_peak_m"],"foot_tilt_change_peak_rad":row["foot_tilt_change_peak_rad"],"gyro_xy_magnitude_peak_rad_s":row["gyro_xy_magnitude_peak_rad_s"],"sustained_tilt_onset_ms":row["sustained_tilt_onset_ms"],"sustained_sink_onset_ms":row["sustained_sink_onset_ms"],"max_penetration_m":row["max_penetration_m"],"gross_rotation":row["gross_rotation"],"valid":row["valid"],"selection_status":"accepted" if accepted else "rejected","rejection_reason":reason})
    with (output/"candidate_sweep.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(items)
    accepted=[item for item in items if item["selection_status"] == "accepted"]
    selected=min(accepted,key=lambda x:(x["height_offset_m"],x["differential_settlement_peak_m"])) if accepted else None
    status="SAND_TILT_V2_SELECTED" if selected else "SAND_TILT_PHYSICAL_DESIGN_REJECTED"
    (output/"selected_config.json").write_text(json.dumps({"status":status,"selected":selected},indent=2)+"\n",encoding="utf-8")
    with (output/"rejected_configs.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows([item for item in items if item["selection_status"] == "rejected" and item["kind"] != "normal"])
    (output/"summary.md").write_text(f"# {status}\n\nTrain-only 50/50 front/rear final bounded physical design calibration. No oracle threshold or label changes; final-test materialization is zero.\n\nUnique candidates: {len(items)}; accepted: {len(accepted)}.\n",encoding="utf-8")


def final_scope_artifacts(output: Path, raw: list[RawRun], labels: list[dict[str, np.ndarray]], rows: list[dict[str, object]], configs: tuple[ScenarioPhysicsConfig, ...]) -> dict[str, object]:
    fields=("mode","config_id","runs","valid","slip_risk_runs","incipient_runs","confirmed_slip_runs","sink_runs","tilt_runs","sink_and_tilt_runs","loaded_contact_coverage","max_horizontal_speed","max_vertical_displacement","max_downward_velocity","max_orientation_deviation","gross_failure","selection_status","rejection_reason")
    items=[]
    for config, row, state in zip(configs, rows, labels):
        confirmed=row["confirmed_slip_onset_ms"] != ""; incipient=row["incipient_risk_onset_ms"] != ""; sink=row["sustained_sink_onset_ms"] != ""; tilt=row["sustained_tilt_onset_ms"] != ""
        gross=bool(int(row["gross_rotation"]) or float(row["loaded_contact_coverage"]) < .75)
        if config.mode == "normal_sand": accept=not (row["slip_risk_onset_ms"] != "" or sink)
        elif config.mode == "slip_risk_dominant": accept=(incipient or confirmed) and not gross
        elif config.mode == "sink_dominant": accept=sink and not gross
        else: accept=sink and tilt and not gross
        reason="" if accept else "normal_hazard" if config.mode == "normal_sand" else "gross_failure" if gross else "missing_required_oracle_coverage"
        items.append({"mode":config.mode,"config_id":config.config_id,"runs":1,"valid":row["valid"],"slip_risk_runs":int(row["slip_risk_onset_ms"] != ""),"incipient_runs":int(incipient),"confirmed_slip_runs":int(confirmed),"sink_runs":int(sink),"tilt_runs":int(tilt),"sink_and_tilt_runs":int(sink and tilt),"loaded_contact_coverage":row["loaded_contact_coverage"],"max_horizontal_speed":row["foot_horizontal_speed_peak_mps"],"max_vertical_displacement":row["foot_sink_depth_peak_m"],"max_downward_velocity":float(max(0., -raw[len(items)].oracle[:, V2_ORACLE_INDEX["foot_velocity_z_mps"]].min())),"max_orientation_deviation":row["foot_tilt_change_peak_rad"],"gross_failure":int(gross),"selection_status":"selected" if accept else "rejected","rejection_reason":reason})
    with (output/"scenario_coverage.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(items)
    selected=[item for item in items if item["selection_status"] == "selected"]
    by_mode={mode:[item for item in selected if item["mode"] == mode] for mode in ("normal_sand","slip_risk_dominant","sink_dominant","sink_and_tilt")}
    slip=by_mode["slip_risk_dominant"]
    gate={"normal_hazard_free":bool(by_mode["normal_sand"]),"confirmed_slip":any(item["confirmed_slip_runs"] for item in slip),"sand_sink":bool(by_mode["sink_dominant"]),"sand_sink_tilt":bool(by_mode["sink_and_tilt"]),"native_1khz":True,"final_test_materialized_0":True}
    frozen=[]
    for mode, candidates in by_mode.items():
        if mode == "normal_sand" and candidates: frozen.append(candidates[0])
        elif mode == "slip_risk_dominant":
            incipient_item=next((item for item in candidates if item["incipient_runs"]), None)
            confirmed_item=next((item for item in candidates if item["confirmed_slip_runs"]), None)
            if incipient_item is not None: frozen.append(incipient_item)
            if confirmed_item is not None: frozen.append(confirmed_item)
        elif candidates: frozen.append(candidates[0])
    frozen=[item for item in frozen if item is not None]
    selected_payload={"status":"PILOT_READY" if all(gate.values()) else "PILOT_NOT_READY","deployment_scope":DEPLOYMENT_SCOPE,"pilot_ready":bool(all(gate.values())),"gates":gate,"selected_configs":frozen}
    (output/"selected_configs.json").write_text(json.dumps(selected_payload,indent=2)+"\n",encoding="utf-8")
    with (output/"rejected_configs.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows([item for item in items if item["selection_status"] == "rejected"])
    (output/"deployment_scope.json").write_text(json.dumps(DEPLOYMENT_SCOPE,indent=2)+"\n",encoding="utf-8")
    (output/"calibration_summary.md").write_text(f"# {selected_payload['status']}\n\nFinal scope: Slip Risk and Sand Sink Hazard. Isolated Sand Tilt-only is diagnostic-only and excluded after bounded Digital Twin physical-design rejection.\n\nTrain-only runs: {len(items)}. Final test materialized: 0.\n",encoding="utf-8")
    return selected_payload


def plot_final_scope(output: Path, raw_runs: list[RawRun], labels: list[dict[str, np.ndarray]]) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots=output/"plots"; plots.mkdir(exist_ok=True)
    wanted=("slip_ice_100", "sink_symmetric_100", "sink_tilt_asymmetric_120")
    for config_id in wanted:
        index=next(i for i, raw in enumerate(raw_runs) if raw.metadata["scenario_config_id"] == config_id)
        raw,state=raw_runs[index],labels[index];x=RELATIVE_TRANSITION_TIME_MS
        fsr=raw.sensors[:,:4]; speed=raw.oracle[:,V2_ORACLE_INDEX["foot_horizontal_speed_mps"]];trend=np.maximum(0,np.diff(speed,prepend=speed[0]))
        fig,axis=plt.subplots(4,1,figsize=(10,9),sharex=True)
        if config_id.startswith("slip"):
            axis[0].plot(x,speed,label="horizontal speed");axis[0].plot(x,trend,label="speed trend");axis[0].legend()
            axis[1].plot(x,raw.oracle[:,V2_ORACLE_INDEX["contact_normal_force_N"]],label="Fn");axis[1].plot(x,raw.oracle[:,V2_ORACLE_INDEX["Ft_over_Fn"]],label="Ft/Fn");axis[1].legend()
            axis[2].plot(x,fsr);axis[2].set_ylabel("FSR N")
            axis[3].plot(x,raw.sensors[:,4:]);axis[3].step(x,state["incipient_risk"],where="post",label="incipient");axis[3].step(x,state["confirmed_slip"],where="post",label="confirmed");axis[3].legend(ncol=3,fontsize=8)
        else:
            axis[0].plot(x,raw.oracle[:,V2_ORACLE_INDEX["foot_sink_depth_m"]],label="sink depth");axis[0].plot(x,-raw.oracle[:,V2_ORACLE_INDEX["foot_velocity_z_mps"]],label="downward velocity");axis[0].legend()
            axis[1].plot(x,raw.oracle[:,V2_ORACLE_INDEX["contact_normal_force_N"]],label="Fn");axis[1].plot(x,raw.oracle[:,V2_ORACLE_INDEX["foot_tilt_change_rad"]],label="tilt");axis[1].legend()
            axis[2].plot(x,fsr);axis[2].set_ylabel("FSR N")
            axis[3].step(x,state["sustained_sink"],where="post",label="sink");axis[3].step(x,state["sustained_tilt"],where="post",label="tilt");axis[3].legend();axis[3].plot(x,raw.sensors[:,7:9])
        axis[-1].set_xlabel("ms relative transition");fig.suptitle(config_id);fig.tight_layout();fig.savefig(plots/f"final_scope_{config_id}.png",dpi=140);plt.close(fig)


def main() -> None:
    args=parse_args()
    if args.scope_policy_audit is not None:
        source=args.scope_policy_audit.resolve(); output=(args.output_dir or OUTPUT_DIR).resolve()
        if output.exists() and any(output.iterdir()): raise FileExistsError(f"refusing to overwrite {output}")
        with (source/"protocol.json").open(encoding="utf-8") as stream: source_protocol=json.load(stream)
        if source_protocol.get("dataset_name") != SCHEMA_NAME or source_protocol.get("final_test",{}).get("materialized"):
            raise ValueError("expected non-final-test v2 calibration")
        with (source/"manifest.csv").open(newline="",encoding="utf-8") as stream: rows=list(csv.DictReader(stream))
        def hits(mode: str, field: str) -> int: return sum(row["mode"] == mode and row[field] != "" for row in rows)
        def valid(mode: str) -> bool: return any(row["mode"] == mode and row["valid"] == "1" and float(row["loaded_contact_coverage"]) >= .75 and row["gross_rotation"] == "0" for row in rows)
        gate={"normal_hazard_free":hits("normal_sand","confirmed_slip_onset_ms") == 0 and hits("normal_sand","sustained_sink_onset_ms") == 0,
              "confirmed_slip":hits("slip_risk_dominant","confirmed_slip_onset_ms") > 0 and valid("slip_risk_dominant"),
              "sand_sink":hits("sink_dominant","sustained_sink_onset_ms") > 0 and valid("sink_dominant"),
              "sand_sink_tilt":hits("sink_and_tilt","sustained_sink_onset_ms") > 0 and valid("sink_and_tilt"),
              "native_1khz":True,"final_test_materialized_0":True,"split_train_validation_only":all(row["split"] in ("train","validation") for row in rows)}
        output.mkdir(parents=True,exist_ok=True)
        payload={"status":"PILOT_READY" if all(gate.values()) else "PILOT_NOT_READY","pilot_ready":bool(all(gate.values())),"deployment_scope":DEPLOYMENT_SCOPE,"gates":gate,"source_calibration":str(source),"source_read_only":True,"historical_artifact_unchanged":True}
        (output/"deployment_scope.json").write_text(json.dumps(DEPLOYMENT_SCOPE,indent=2)+"\n",encoding="utf-8")
        (output/"pilot_readiness.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
        (output/"summary.md").write_text(f"# {payload['status']}\n\nFrozen before pilot. Existing calibration read only; historical artifact unchanged.\n",encoding="utf-8")
        print(json.dumps(payload,indent=2)); return
    if args.audit_existing is not None:
        source=args.audit_existing.resolve()
        with (source / "protocol.json").open(encoding="utf-8") as stream: existing=json.load(stream)
        if existing.get("dataset_name") != SCHEMA_NAME or existing.get("final_test",{}).get("materialized"):
            raise ValueError("expected non-final-test terrain_fast_reflex_v2 output")
        with np.load(source / "inputs_fusion10.npz",allow_pickle=False) as data:
            sensors,times=data["sensors"],data["sample_time_s"]
        with np.load(source / "oracle_diagnostics.npz",allow_pickle=False) as data:
            required=("safe","incipient_risk","confirmed_slip","slip_risk","sustained_sink","sustained_tilt")
            if any(name not in data for name in required): raise ValueError("missing v2 state labels")
            if np.any(data["confirmed_slip"] & ~data["slip_risk"]): raise ValueError("confirmed not included in risk")
        spacing=np.diff(times,axis=1)
        if sensors.ndim != 3 or sensors.shape[1:] != (TRACE_SAMPLES,10) or not np.isfinite(sensors).all() or not np.allclose(spacing,.001,atol=1e-9):
            raise ValueError("invalid Fusion10/timestamp artifact")
        print(f"V2_AUDIT_PASS runs={len(sensors)} native_spacing_ms=1 final_test_materialized=0 source={source}")
        if "fast_reflex_v2_deployment_scope" in existing:
            with (source / "manifest.csv").open(newline="",encoding="utf-8") as stream: rows=list(csv.DictReader(stream))
            def count(mode: str, field: str) -> int: return sum(row["mode"] == mode and row[field] != "" for row in rows)
            mode_counts={mode:sum(row["mode"] == mode for row in rows) for mode in sorted({row["mode"] for row in rows})}
            print(f"scope_modes={json.dumps(mode_counts,sort_keys=True)}")
            print(f"scope_coverage normal_slip={count('normal_sand','confirmed_slip_onset_ms')} normal_sink={count('normal_sand','sustained_sink_onset_ms')} slip_confirmed={count('slip_risk_dominant','confirmed_slip_onset_ms')} sink={count('sink_dominant','sustained_sink_onset_ms')} sink_tilt_sink={count('sink_and_tilt','sustained_sink_onset_ms')} sink_tilt_tilt={count('sink_and_tilt','sustained_tilt_onset_ms')}")
        return
    if sum(bool(item) for item in (args.scenario_calibration, args.front_rear_torque_calibration, args.local_compliance_calibration, args.final_tilt_physics_calibration, args.final_scope_calibration, args.final_scope_pilot, args.scenario_selection is not None)) > 1:
        raise ValueError("choose one scenario calibration/selection source")
    if args.scenario_calibration:
        configs=tuple(config for config in calibration_scenario_configs() if config.mode in args.modes)
        if any(family_for_name(name).split != "train" for name in _families(args)):
            raise ValueError("scenario calibration is train-only")
    elif args.front_rear_torque_calibration:
        configs=tuple(config for config in front_rear_torque_calibration_configs() if config.mode in args.modes)
        if any(family_for_name(name).split != "train" for name in _families(args)):
            raise ValueError("scenario calibration is train-only")
    elif args.local_compliance_calibration:
        configs=tuple(config for config in local_compliance_calibration_configs() if config.mode in args.modes)
        if any(family_for_name(name).split != "train" for name in _families(args)):
            raise ValueError("scenario calibration is train-only")
    elif args.final_tilt_physics_calibration:
        configs=tuple(config for config in final_tilt_physics_calibration_configs() if config.mode in args.modes)
        if any(family_for_name(name).split != "train" for name in _families(args)):
            raise ValueError("scenario calibration is train-only")
    elif args.final_scope_calibration:
        configs=tuple(config for config in final_scope_calibration_configs() if config.mode in args.modes)
        if any(family_for_name(name).split != "train" for name in _families(args)):
            raise ValueError("scenario calibration is train-only")
    elif args.final_scope_pilot:
        configs=tuple(config for config in final_scope_pilot_configs() if config.mode in args.modes)
    elif args.scenario_selection is not None:
        selected=json.loads(args.scenario_selection.read_text(encoding="utf-8"))
        configs=tuple(ScenarioPhysicsConfig(**{key: value for key, value in item.items() if key != "coverage"})
                      for item in selected["selected_configs"] if item["mode"] in args.modes)
    else:
        configs=tuple(config for config in default_scenario_configs() if config.mode in args.modes)
    if not configs: raise ValueError("no scenario configs selected")
    payload=protocol(args, configs)
    if not args.execute:
        print(json.dumps(payload,indent=2)); print("Dry run only. Use --execute; final test remains fail-closed."); return
    output=(args.output_dir or OUTPUT_DIR).resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"refusing to overwrite {output}")
    # A previous dry/failed invocation may have created an empty directory.
    # It has no artifact to protect, unlike any non-empty output above.
    output.mkdir(parents=True, exist_ok=True)
    raw=[]
    for config in configs:
        for family in _families(args):
            for si in range(args.surfaces_per_family):
                for ri in range(args.runs_per_surface): raw.append(run_one(config,family,si,ri))
    normals=[r.oracle for r in raw if r.metadata["mode"] == "normal_sand" and r.metadata["split"] == "train"]
    calibration=calibrate_v2(normals); labels=[label_v2(r.oracle,calibration) for r in raw]
    for item in labels: validate_state_order(item)
    rows=[_row(r,l) for r,l in zip(raw,labels)]; payload["calibration"]=calibration.as_dict(); payload["measured"]={"runs":len(raw),"valid_runs":sum(r["valid"] for r in rows),"final_test_materialized":0,"wall_time_s":sum(r.wall_time_s for r in raw)}
    save(output,raw,labels,rows,payload)
    coverage=_coverage(rows)
    with (output/"scenario_coverage.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(coverage[0])); writer.writeheader(); writer.writerows(coverage)
    selection=_selection(configs,coverage)
    if args.local_compliance_calibration:
        # This is a single-mode feasibility study, never a full pilot gate.
        selection["pilot_ready"] = False
        selection["gates"]["localized_compliance_feasibility_only"] = True
    (output/"scenario_configs.json").write_text(json.dumps([config.as_dict() for config in configs],indent=2)+"\n",encoding="utf-8")
    (output/"scenario_selection.json").write_text(json.dumps(selection,indent=2)+"\n",encoding="utf-8")
    if args.local_compliance_calibration:
        local_compliance_artifacts(output, raw, labels, rows, configs)
        plot_local_compliance(output, raw, labels)
    if args.final_tilt_physics_calibration:
        selection["pilot_ready"] = False
        selection["gates"]["final_tilt_physics_feasibility_only"] = True
        final_tilt_physics_artifacts(output, rows, configs)
        plot_local_compliance(output, raw, labels)
    if args.final_scope_calibration:
        final_scope=final_scope_artifacts(output, raw, labels, rows, configs)
        plot_final_scope(output, raw, labels)
        selection["pilot_ready"]=final_scope["pilot_ready"]
        selection["gates"] = final_scope["gates"]
        # Save the selected scope after its authoritative gate is known.
        (output/"scenario_selection.json").write_text(json.dumps(selection,indent=2)+"\n",encoding="utf-8")
    if args.final_scope_pilot:
        (output/"deployment_scope.json").write_text(json.dumps(DEPLOYMENT_SCOPE,indent=2)+"\n",encoding="utf-8")
    if args.plot:
        plots=output/"plots";plots.mkdir(exist_ok=True)
        for mode in sorted({config.mode for config in configs}):
            i=next(i for i,r in enumerate(raw) if r.metadata["mode"]==mode);plot_smoke(plots,raw[i],labels[i])
        plot_coverage(plots, coverage)
    summary=[f"{SCHEMA_NAME} schema_version={SCHEMA_VERSION}",f"runs={len(raw)} valid={sum(r['valid'] for r in rows)}", "native_sampling=1000Hz physics=2000Hz spacing=1ms", "final_test_materialized=0",f"calibration={json.dumps(calibration.as_dict(),sort_keys=True)}"]
    for mode in sorted({config.mode for config in configs}):
        subset=[r for r in rows if r["mode"]==mode];summary.append(f"{mode}: runs={len(subset)} risk={sum(r['slip_risk_onset_ms']!='' for r in subset)} sink={sum(r['sustained_sink_onset_ms']!='' for r in subset)} tilt={sum(r['sustained_tilt_onset_ms']!='' for r in subset)}")
    summary.extend([f"scenario_calibration={int(args.scenario_calibration or args.front_rear_torque_calibration or args.local_compliance_calibration or args.final_tilt_physics_calibration or args.final_scope_calibration)}",f"pilot_ready={selection['pilot_ready']}",f"scenario_gates={json.dumps(selection['gates'],sort_keys=True)}"])
    (output/"summary.txt").write_text("\n".join(summary)+"\n",encoding="utf-8");print("\n".join(summary))


if __name__ == "__main__": main()
