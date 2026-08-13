"""Generate Fast Reflex v2 physical-state datasets; final test is opt-in only."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
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
    calibrate_v2, label_v2, onset_ms, validate_final_test_request, validate_state_order,
)
from terrain_profiles import TERRAIN_PROFILES, apply_terrain_profile


SIM_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = SIM_DIR / "outputs" / SCHEMA_NAME
SCENE_DIR = Path(__file__).resolve().parents[1] / "unitree_robots" / "g1"
SCENES = {
    "front_rear": SCENE_DIR / "scene_fast_reflex_v2_front_rear.xml",
    "left_right": SCENE_DIR / "scene_fast_reflex_v2_left_right.xml",
}
TRANSITION_TIME_S, DURATION_S = .250, .400
PULSE_DURATION_S, PULSE_MAGNITUDE_N = .100, 80.0


@dataclass(frozen=True)
class RawRun:
    metadata: dict[str, object]
    timestamps_s: np.ndarray
    sensors: np.ndarray
    oracle: np.ndarray
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
    parser.add_argument("--audit-existing", type=Path,
                        help="read-only schema/physical-validity audit of a generated v2 train/validation output")
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


def candidate_count(args: argparse.Namespace) -> int:
    return len(args.modes) * len(_families(args)) * args.surfaces_per_family * args.runs_per_surface


def protocol(args: argparse.Namespace) -> dict[str, object]:
    families = _families(args)
    if args.surfaces_per_family < 1 or args.runs_per_surface < 1:
        raise ValueError("positive surface/run counts required")
    if "normal_sand" not in args.modes:
        raise ValueError("normal_sand train rows are required for calibration")
    count = candidate_count(args)
    return {
        "dataset_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
        "status": "v2 physical dataset foundation; no detector training",
        "modes": list(args.modes), "physics_rate_hz": 2000, "sensor_rate_hz": SENSOR_RATE_HZ,
        "physics_timestep_s": PHYSICS_TIMESTEP_S, "physics_steps_per_sample": PHYSICS_STEPS_PER_SAMPLE,
        "trace_interval_ms": [-TRACE_PRE_MS, TRACE_POST_MS], "trace_shape": [TRACE_SAMPLES, 10],
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


def mode_config(mode: str) -> tuple[str, str, str, str, float, float, float]:
    """layout, ground A/B material, support ratio, force magnitude, force direction."""
    if mode == "normal_sand": return "front_rear", "sand", "sand", .70, 0., 1., 0.
    if mode == "sink_dominant": return "front_rear", "sand", "sand", .48, 0., 1., 0.
    if mode == "tilt_dominant": return "front_rear", "marble", "sand", .70, 40., 1., 0.
    if mode == "boundary_front_rear": return "front_rear", "marble", "sand", .70, 80., 1., 0.
    if mode == "boundary_left_right": return "left_right", "marble", "sand", .70, 80., 0., 1.
    if mode == "sink_and_tilt": return "front_rear", "marble", "sand", .48, 80., 1., 0.
    raise ValueError(mode)


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


def run_one(mode: str, family: str, surface_index: int, run_index: int) -> RawRun:
    layout, material_a, material_b, support_ratio, force, dx, dy = mode_config(mode)
    model = mujoco.MjModel.from_xml_path(str(SCENES[layout])); model.opt.timestep = PHYSICS_TIMESTEP_S
    data = mujoco.MjData(model)
    for name, material in (("ground_a", material_a), ("ground_b", material_b)):
        apply_terrain_profile(model, TERRAIN_PROFILES[material], name)
    ground_ids = [model.geom("ground_a").id, model.geom("ground_b").id]
    spec = make_expanded_run_specification("sand", family, surface_index, run_index)
    condition = ExcitationCondition(f"{mode}_{family}_s{surface_index:02d}_r{run_index:03d}",
                                   spec.initial_velocity_x, spec.initial_velocity_y, spec.base_height_offset,
                                   spec.base_roll_deg, spec.base_pitch_deg)
    qpos, dof = apply_excitation_condition(model, data, condition)
    support = VerticalElasticBandSupport(model, data, qpos, dof, support_ratio)
    pulse = HorizontalPulse(TRANSITION_TIME_S, PULSE_DURATION_S, force, dx, dy)
    exciter = HorizontalPulseExciter(model, data, pulse)
    reader = G1HilSensorReader(model, data); foot_ids = frozenset(reader.left_foot_geom_ids)
    body = model.body("left_ankle_roll_link").id; velocity = np.zeros(6); wrench = np.zeros(6)
    times=[]; sensors=[]; raw=[]; steps=0; start=time.perf_counter()
    while data.time + 1e-12 < DURATION_S:
        support.apply(); exciter.apply(float(data.time)); mujoco.mj_step(model, data); steps += 1
        if steps % PHYSICS_STEPS_PER_SAMPLE == 0:
            times.append(float(data.time)); sensors.append(reader.read_vector())
            raw.append(_foot_oracle(model, data, ground_ids, foot_ids, body, velocity, wrench))
    times=np.asarray(times); sensors=np.asarray(sensors); raw=np.asarray(raw)
    select=(times >= TRANSITION_TIME_S - TRACE_PRE_MS / 1000 - 1e-9) & (times < TRANSITION_TIME_S + TRACE_POST_MS / 1000 - 1e-9)
    family_info=family_for_name(family)
    metadata={"schema_version":SCHEMA_VERSION,"mode":mode,"surface_family":family,"surface_seed":surface_index,
              "session_id":f"v2_{family_info.split}_{mode}_surface_{surface_index:02d}","run_id":condition.run_id,
              "split":family_info.split,"physics_rate_hz":2000,"sensor_rate_hz":1000,"transition_time_s":TRANSITION_TIME_S,
              "ground_layout":layout,"ground_a_material":material_a,"ground_b_material":material_b,
              "boundary_orientation":"front_rear" if layout == "front_rear" else "left_right","boundary_position_m":0.,
              "pulse_magnitude_N":force,"support_ratio":support_ratio,"run_index":run_index}
    return RawRun(metadata, times[select], sensors[select], _postprocess(raw[select]), 0., 0., time.perf_counter()-start)


def _row(raw: RawRun, labels: dict[str, np.ndarray]) -> dict[str, object]:
    fsr = raw.sensors[:, :4]
    fsr_sum = fsr.sum(axis=1)
    front_rear = fsr[:, 2:].sum(axis=1) - fsr[:, :2].sum(axis=1)
    left_right = fsr[:, [0, 2]].sum(axis=1) - fsr[:, [1, 3]].sum(axis=1)
    normalizer = np.maximum(fsr_sum, 1e-6)
    gyro_xy = np.linalg.norm(raw.sensors[:, 7:9], axis=1)
    out={**raw.metadata,"calibration_provenance":"train-normal only","valid":int(np.isfinite(raw.sensors).all() and np.isfinite(raw.oracle).all()),
         "invalid_reason":"","qpos_transition_max_abs_delta":raw.qpos_delta,"qvel_transition_max_abs_delta":raw.qvel_delta,
         "ground_a_contact_samples":int(raw.oracle[:, V2_ORACLE_INDEX["ground_a_contact"]].sum()),
         "ground_b_contact_samples":int(raw.oracle[:, V2_ORACLE_INDEX["ground_b_contact"]].sum()),"wall_time_s":raw.wall_time_s,
         "fsr_sum_max_N":float(fsr_sum.max()), "front_minus_rear_peak_N":float(np.max(np.abs(front_rear))),
         "left_minus_right_peak_N":float(np.max(np.abs(left_right))),
         "normalized_front_rear_imbalance_peak":float(np.max(np.abs(front_rear / normalizer))),
         "normalized_left_right_imbalance_peak":float(np.max(np.abs(left_right / normalizer))),
         "fsr_spatial_variance_peak":float(np.max(fsr.var(axis=1))), "fsr_range_peak_N":float(np.max(np.ptp(fsr,axis=1))),
         "gyro_xy_magnitude_peak_rad_s":float(gyro_xy.max()), "gyro_xy_integral_rad":float(gyro_xy.sum()*.001)}
    for name in ("slip_risk","confirmed_slip","sustained_sink","sustained_tilt"):
        onset=onset_ms(labels[name]); out[f"{name}_onset_time_s"]="" if onset is None else float(raw.timestamps_s[TRACE_PRE_MS+onset])
        out[f"{name}_onset_ms"]="" if onset is None else onset
    return out


def save(output: Path, raw_runs: list[RawRun], labels: list[dict[str, np.ndarray]], rows: list[dict[str, object]], payload: dict[str, object]) -> None:
    np.savez_compressed(output / "inputs_fusion10.npz", sensors=np.asarray([r.sensors for r in raw_runs],np.float32),
                        sample_time_s=np.asarray([r.timestamps_s for r in raw_runs]), relative_transition_time_ms=RELATIVE_TRANSITION_TIME_MS,
                        run_id=np.asarray([r.metadata["run_id"] for r in raw_runs]))
    np.savez_compressed(output / "oracle_diagnostics.npz", oracle=np.asarray([r.oracle for r in raw_runs],np.float32),
                        oracle_channels=np.asarray(V2_ORACLE_CHANNELS), sample_time_s=np.asarray([r.timestamps_s for r in raw_runs]),
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


def main() -> None:
    args=parse_args()
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
        return
    payload=protocol(args)
    if not args.execute:
        print(json.dumps(payload,indent=2)); print("Dry run only. Use --execute; final test remains fail-closed."); return
    output=(args.output_dir or OUTPUT_DIR).resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    raw=[]
    for mode in args.modes:
        for family in _families(args):
            for si in range(args.surfaces_per_family):
                for ri in range(args.runs_per_surface): raw.append(run_one(mode,family,si,ri))
    normals=[r.oracle for r in raw if r.metadata["mode"] == "normal_sand" and r.metadata["split"] == "train"]
    calibration=calibrate_v2(normals); labels=[label_v2(r.oracle,calibration) for r in raw]
    for item in labels: validate_state_order(item)
    rows=[_row(r,l) for r,l in zip(raw,labels)]; payload["calibration"]=calibration.as_dict(); payload["measured"]={"runs":len(raw),"valid_runs":sum(r["valid"] for r in rows),"final_test_materialized":0,"wall_time_s":sum(r.wall_time_s for r in raw)}
    save(output,raw,labels,rows,payload)
    if args.plot:
        plots=output/"plots";plots.mkdir()
        for mode in args.modes:
            i=next(i for i,r in enumerate(raw) if r.metadata["mode"]==mode);plot_smoke(plots,raw[i],labels[i])
    summary=[f"{SCHEMA_NAME} schema_version={SCHEMA_VERSION}",f"runs={len(raw)} valid={sum(r['valid'] for r in rows)}", "native_sampling=1000Hz physics=2000Hz spacing=1ms", "final_test_materialized=0",f"calibration={json.dumps(calibration.as_dict(),sort_keys=True)}"]
    for mode in args.modes:
        subset=[r for r in rows if r["mode"]==mode];summary.append(f"{mode}: runs={len(subset)} risk={sum(r['slip_risk_onset_ms']!='' for r in subset)} sink={sum(r['sustained_sink_onset_ms']!='' for r in subset)} tilt={sum(r['sustained_tilt_onset_ms']!='' for r in subset)}")
    (output/"summary.txt").write_text("\n".join(summary)+"\n",encoding="utf-8");print("\n".join(summary))


if __name__ == "__main__": main()
