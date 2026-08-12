"""Generate continuous-state temporal terrain transitions for Fast Reflex v1."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from controlled_excitation import (
    ExcitationCondition,
    HorizontalPulse,
    HorizontalPulseExciter,
    VerticalElasticBandSupport,
    apply_excitation_condition,
)
from expanded_terrain_dataset_v1 import (
    SURFACE_FAMILIES,
    family_for_name,
    make_expanded_run_specification,
)
from hil_sensor import G1HilSensorReader, HIL_SENSOR_CHANNELS
from run_surface_sampling_rate_study import SIMULATION_DIR, write_dict_rows
from run_walking_touchdown_dataset_v1 import (
    SCENE_PATH,
    configure_contact_model,
    configure_walking_surface,
)
from terrain_fast_reflex_v1 import (
    ORACLE_CHANNELS,
    ORACLE_INDEX,
    PHYSICS_STEPS_PER_SAMPLE,
    PHYSICS_TIMESTEP_S,
    PRIMARY_WINDOWS_MS,
    RELATIVE_TRANSITION_TIME_MS,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    SENSOR_RATE_HZ,
    TRACE_POST_MS,
    TRACE_PRE_MS,
    TRACE_SAMPLES,
    FastReflexTrace,
    HazardThresholds,
    calibrate_hazard_thresholds,
    extract_prefix,
    label_trace,
    onset_time_s,
    validate_split_integrity,
    validate_trace,
)
from terrain_profiles import TERRAIN_PROFILES, apply_terrain_profile


OUTPUT_DIR = SIMULATION_DIR / "outputs" / SCHEMA_NAME
TRANSITION_TIME_S = 0.250
DURATION_S = 0.400
SUPPORT_RATIO = 0.70
PULSE_DURATION_S = 0.100
PULSE_MAGNITUDE_N = 80.0
SCENARIO_NAMES = (
    "marble_to_ice",
    "marble_to_sand",
    "marble_to_marble",
    "concrete_to_concrete",
)
NORMAL_SCENARIOS = frozenset(("marble_to_marble", "concrete_to_concrete"))
SCENARIOS = {
    "marble_to_ice": ("marble", "ice", "slip"),
    "marble_to_sand": ("marble", "sand", "sink_or_tilt"),
    "marble_to_marble": ("marble", "marble", "normal"),
    "concrete_to_concrete": ("concrete", "concrete", "normal"),
}


@dataclass(frozen=True)
class RawTransitionRun:
    trace: FastReflexTrace
    qpos_transition_delta: float
    qvel_transition_delta: float
    wall_time_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--scenarios", nargs="+", choices=SCENARIO_NAMES,
        default=list(SCENARIO_NAMES),
    )
    parser.add_argument(
        "--families", nargs="+",
        choices=tuple(family.name for family in SURFACE_FAMILIES),
        default=[family.name for family in SURFACE_FAMILIES],
    )
    parser.add_argument("--surfaces-per-family", type=int, default=3)
    parser.add_argument("--runs-per-surface", type=int, default=5)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def candidate_count(args: argparse.Namespace) -> int:
    return (
        len(args.scenarios)
        * len(args.families)
        * args.surfaces_per_family
        * args.runs_per_surface
    )


def protocol(args: argparse.Namespace) -> dict[str, object]:
    if args.surfaces_per_family <= 0 or args.runs_per_surface <= 0:
        raise ValueError("surface and run counts must be positive")
    if not any(scenario in NORMAL_SCENARIOS for scenario in args.scenarios):
        raise ValueError("at least one normal scenario is required for calibration")
    for family in args.families:
        family_for_name(family)
    runs = candidate_count(args)
    uncompressed = runs * TRACE_SAMPLES * (10 + len(ORACLE_CHANNELS)) * 4
    return {
        "dataset_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "transition acquisition and physical-label foundation; no model training",
        "scenarios": args.scenarios,
        "transition_type": "temporal_parameter_switch",
        "continuous_state": "qpos/qvel are not reset; only floor contact parameters change at t0",
        "surface_geometry": "pre-terrain morphology remains fixed through the temporal switch",
        "sand_limit": "compliance-based engineering approximation; no plastic deformation/history",
        "physics_rate_hz": 2000,
        "sensor_rate_hz": SENSOR_RATE_HZ,
        "physics_timestep_s": PHYSICS_TIMESTEP_S,
        "physics_steps_per_sample": PHYSICS_STEPS_PER_SAMPLE,
        "transition_time_s": TRANSITION_TIME_S,
        "trace_interval_ms": [-TRACE_PRE_MS, TRACE_POST_MS],
        "trace_shape": [TRACE_SAMPLES, len(HIL_SENSOR_CHANNELS)],
        "input_channels": HIL_SENSOR_CHANNELS,
        "oracle_channels": ORACLE_CHANNELS,
        "observation_windows_ms": [1, 2, *PRIMARY_WINDOWS_MS],
        "alignment": ["transition", "hazard_onset"],
        "support_ratio": SUPPORT_RATIO,
        "pulse": {
            "start_time_s": TRANSITION_TIME_S,
            "duration_s": PULSE_DURATION_S,
            "magnitude_N": PULSE_MAGNITUDE_N,
            "direction": "alternating +/-X by deterministic run parity",
        },
        "hazard_ground_truth": {
            "calibration": "train-family normal traces only; test families excluded",
            "slip": "left contact AND Fn>=minimum load AND horizontal foot speed threshold for 3 ms",
            "sink": "loaded left contact AND depth and downward-speed thresholds for 3 ms",
            "tilt": "loaded left contact AND roll/pitch change threshold for 3 ms",
            "terrain_name_is_not_a_hazard_label": True,
        },
        "split_policy": {
            split: [family.name for family in SURFACE_FAMILIES if family.split == split]
            for split in ("train", "validation", "test")
        },
        "surface_families": args.families,
        "surfaces_per_family": args.surfaces_per_family,
        "runs_per_surface": args.runs_per_surface,
        "candidate_runs": runs,
        "estimated_uncompressed_trace_bytes": uncompressed,
        "estimated_runtime_s": [runs * 0.25, runs * 1.0],
        "overwrite_policy": "refuse any non-empty output directory",
        "kpi": {
            "fast_layer": "transition to hazard detection latency <50 ms",
            "research_target": "useful separation with <=20 ms observation",
            "future_reflex_e2e": "hazard detection to command change <30 ms (not measured here)",
        },
    }


def _foot_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    floor_id: int,
    left_geom_ids: frozenset[int],
    foot_body_id: int,
    velocity: np.ndarray,
    wrench: np.ndarray,
) -> tuple[float, ...]:
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, foot_body_id, velocity, 0
    )
    normal, tangential, contact = 0.0, 0.0, False
    for contact_id in range(data.ncon):
        item = data.contact[contact_id]
        pair = {int(item.geom1), int(item.geom2)}
        if floor_id not in pair or not pair.intersection(left_geom_ids):
            continue
        wrench.fill(0.0)
        mujoco.mj_contactForce(model, data, contact_id, wrench)
        normal += max(0.0, float(wrench[0]))
        tangential += float(np.linalg.norm(wrench[1:3]))
        contact = True
    rotation = data.xmat[foot_body_id].reshape(3, 3)
    roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
    pitch = float(
        np.arctan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2]))
    )
    linear = velocity[3:6]
    angular = velocity[:3]
    return (
        float(contact), normal, tangential, tangential / max(normal, 1e-12),
        *linear, *angular, roll, pitch, float(data.xpos[foot_body_id, 2]),
    )


def _postprocess_oracle(raw: np.ndarray) -> np.ndarray:
    if raw.shape != (TRACE_SAMPLES, 13):
        raise ValueError(f"unexpected raw oracle shape {raw.shape}")
    transition_index = TRACE_PRE_MS
    horizontal_speed = np.linalg.norm(raw[:, 4:6], axis=1)
    sink_depth = np.maximum(0.0, raw[transition_index, 12] - raw[:, 12])
    tilt_change = np.linalg.norm(
        raw[:, 10:12] - raw[transition_index, 10:12], axis=1
    )
    return np.column_stack((raw, horizontal_speed, sink_depth, tilt_change))


def run_transition(
    scenario: str,
    family: str,
    surface_index: int,
    run_index: int,
) -> RawTransitionRun:
    pre_terrain, post_terrain, expected_hazard = SCENARIOS[scenario]
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    model.opt.timestep = PHYSICS_TIMESTEP_S
    data = mujoco.MjData(model)
    floor_id, surface = configure_walking_surface(
        model, pre_terrain, family, surface_index
    )
    configure_contact_model(model, floor_id, "foot-spheres-only")
    pre_spec = make_expanded_run_specification(
        pre_terrain, family, surface_index, run_index
    )
    post_spec = make_expanded_run_specification(
        post_terrain, family, surface_index, run_index
    )
    pre_profile = replace(TERRAIN_PROFILES[pre_terrain], friction=pre_spec.friction)
    post_profile = replace(TERRAIN_PROFILES[post_terrain], friction=post_spec.friction)
    apply_terrain_profile(model, pre_profile, "surface_floor")

    condition = ExcitationCondition(
        run_id=f"{scenario}_{family}_s{surface_index:02d}_r{run_index:03d}",
        initial_velocity_x=pre_spec.initial_velocity_x,
        initial_velocity_y=pre_spec.initial_velocity_y,
        base_height_offset=pre_spec.base_height_offset,
        base_roll_deg=pre_spec.base_roll_deg,
        base_pitch_deg=pre_spec.base_pitch_deg,
    )
    qpos_address, dof_address = apply_excitation_condition(model, data, condition)
    support = VerticalElasticBandSupport(
        model, data, qpos_address, dof_address, SUPPORT_RATIO
    )
    direction = 1.0 if (surface_index + run_index) % 2 == 0 else -1.0
    pulse = HorizontalPulse(
        TRANSITION_TIME_S, PULSE_DURATION_S, PULSE_MAGNITUDE_N, direction, 0.0
    )
    exciter = HorizontalPulseExciter(model, data, pulse)
    reader = G1HilSensorReader(model, data)
    foot_body_id = model.body("left_ankle_roll_link").id
    left_geom_ids = frozenset(reader.left_foot_geom_ids)
    velocity = np.zeros(6, dtype=np.float64)
    wrench = np.zeros(6, dtype=np.float64)

    timestamps: list[float] = []
    sensors: list[np.ndarray] = []
    raw_oracle: list[tuple[float, ...]] = []
    switched = False
    qpos_delta = float("nan")
    qvel_delta = float("nan")
    physics_step = 0
    wall_start = time.perf_counter()
    while data.time + 1e-12 < DURATION_S:
        next_time = float(data.time + model.opt.timestep)
        if not switched and next_time + 1e-12 >= TRANSITION_TIME_S:
            before_qpos, before_qvel = data.qpos.copy(), data.qvel.copy()
            apply_terrain_profile(model, post_profile, "surface_floor")
            qpos_delta = float(np.max(np.abs(data.qpos - before_qpos)))
            qvel_delta = float(np.max(np.abs(data.qvel - before_qvel)))
            switched = True
        support.apply()
        exciter.apply(float(data.time))
        mujoco.mj_step(model, data)
        physics_step += 1
        if physics_step % PHYSICS_STEPS_PER_SAMPLE == 0:
            timestamps.append(float(data.time))
            sensors.append(reader.read_vector())
            raw_oracle.append(
                _foot_state(
                    model, data, floor_id, left_geom_ids, foot_body_id,
                    velocity, wrench,
                )
            )
    wall_time = time.perf_counter() - wall_start
    if not switched:
        raise RuntimeError("terrain parameters were never switched")
    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    sensor_array = np.asarray(sensors, dtype=np.float64)
    raw_oracle_array = np.asarray(raw_oracle, dtype=np.float64)
    trace_indices = np.flatnonzero(
        (timestamp_array >= TRANSITION_TIME_S - TRACE_PRE_MS / 1000.0 - 1e-9)
        & (timestamp_array < TRANSITION_TIME_S + TRACE_POST_MS / 1000.0 - 1e-9)
    )
    timestamps_trace = timestamp_array[trace_indices]
    sensors_trace = sensor_array[trace_indices]
    oracle_trace = _postprocess_oracle(raw_oracle_array[trace_indices])
    reasons = []
    if qpos_delta != 0.0 or qvel_delta != 0.0:
        reasons.append("state_discontinuity_at_transition")
    if not np.all(np.isfinite(sensor_array)) or not np.all(np.isfinite(raw_oracle_array)):
        reasons.append("nan_or_inf")
    metadata = {
        "scenario": scenario,
        "transition_type": "temporal_parameter_switch",
        "pre_terrain": pre_terrain,
        "post_terrain": post_terrain,
        "expected_hazard": expected_hazard,
        "surface_family": family,
        "split": surface.split,
        "surface_index": surface_index,
        "surface_seed": surface.surface_seed,
        "run_index": run_index,
        "session_id": f"{scenario}_{surface.split}_{family}_surface_{surface_index:02d}",
        "run_id": condition.run_id,
        "transition_time_s": TRANSITION_TIME_S,
        "pulse_direction_x": direction,
        "pre_friction": "|".join(f"{value:.9g}" for value in pre_profile.friction),
        "post_friction": "|".join(f"{value:.9g}" for value in post_profile.friction),
        "pre_solref": "|".join(f"{value:.9g}" for value in pre_profile.solref),
        "post_solref": "|".join(f"{value:.9g}" for value in post_profile.solref),
        "pre_solimp": "|".join(f"{value:.9g}" for value in pre_profile.solimp),
        "post_solimp": "|".join(f"{value:.9g}" for value in post_profile.solimp),
        "qpos_transition_max_abs_delta": qpos_delta,
        "qvel_transition_max_abs_delta": qvel_delta,
    }
    trace = FastReflexTrace(
        metadata=metadata,
        timestamps_s=timestamps_trace,
        sensors=sensors_trace,
        oracle=oracle_trace,
        slip=np.zeros(TRACE_SAMPLES, dtype=bool),
        sink=np.zeros(TRACE_SAMPLES, dtype=bool),
        tilt=np.zeros(TRACE_SAMPLES, dtype=bool),
        valid=not reasons,
        invalid_reason="|".join(reasons),
    )
    validate_trace(trace)
    return RawTransitionRun(trace, qpos_delta, qvel_delta, wall_time)


def _onset_fields(trace: FastReflexTrace) -> dict[str, object]:
    slip = onset_time_s(trace, trace.slip)
    sink = onset_time_s(trace, trace.sink)
    tilt = onset_time_s(trace, trace.tilt)
    sink_or_tilt = onset_time_s(trace, trace.sink_or_tilt)
    any_hazard_label = trace.slip | trace.sink | trace.tilt
    any_hazard = onset_time_s(trace, any_hazard_label)
    transition = float(trace.metadata["transition_time_s"])
    expected = str(trace.metadata["expected_hazard"])
    target = slip if expected == "slip" else sink_or_tilt if expected == "sink_or_tilt" else None
    actual_types = [
        name for name, label in (
            ("slip", trace.slip), ("sink", trace.sink), ("tilt", trace.tilt)
        )
        if np.any(label)
    ]
    return {
        "slip_onset_time_s": "" if slip is None else slip,
        "sink_onset_time_s": "" if sink is None else sink,
        "tilt_onset_time_s": "" if tilt is None else tilt,
        "hazard_onset_time_s": "" if any_hazard is None else any_hazard,
        "transition_to_hazard_ms": ""
        if any_hazard is None else (any_hazard - transition) * 1000.0,
        "target_hazard_onset_time_s": "" if target is None else target,
        "transition_to_target_hazard_ms": ""
        if target is None else (target - transition) * 1000.0,
        "hazard_type": "normal" if not actual_types else "|".join(actual_types),
        "physical_hazard": int(any_hazard is not None),
    }


def _feature_vector(values: np.ndarray) -> np.ndarray:
    return np.concatenate((values.mean(axis=0), values.std(axis=0), values[-1] - values[0]))


def _separation(first: list[np.ndarray], second: list[np.ndarray]) -> float:
    a = np.asarray([_feature_vector(values) for values in first])
    b = np.asarray([_feature_vector(values) for values in second])
    pooled_values = np.concatenate((a, b))
    scale = pooled_values.std(axis=0)
    scale[scale < 1e-9] = 1.0
    a, b = a / scale, b / scale
    centroid_distance = np.linalg.norm(a.mean(axis=0) - b.mean(axis=0))
    spread = np.sqrt(
        0.5 * np.mean(np.sum((a - a.mean(axis=0)) ** 2, axis=1))
        + 0.5 * np.mean(np.sum((b - b.mean(axis=0)) ** 2, axis=1))
    )
    return float(centroid_distance / max(spread, 1e-12))


def separation_summary(traces: list[FastReflexTrace]) -> list[dict[str, object]]:
    rows = []
    for split in ("train", "validation", "test"):
        normals = [
            trace for trace in traces
            if trace.metadata["scenario"] in NORMAL_SCENARIOS
            and trace.metadata["split"] == split
            and not np.any(trace.slip | trace.sink | trace.tilt)
        ]
        for scenario in ("marble_to_ice", "marble_to_sand"):
            hazards = [
                trace for trace in traces
                if trace.metadata["scenario"] == scenario
                and trace.metadata["split"] == split
            ]
            if not hazards or not normals:
                continue
            for window_ms in PRIMARY_WINDOWS_MS:
                normal_prefix = [extract_prefix(trace, window_ms)[1] for trace in normals]
                hazard_prefix = [extract_prefix(trace, window_ms)[1] for trace in hazards]
                onset_latencies = []
                for trace in hazards:
                    expected = str(trace.metadata["expected_hazard"])
                    label = trace.slip if expected == "slip" else trace.sink_or_tilt
                    onset = onset_time_s(trace, label)
                    if onset is not None:
                        onset_latencies.append(
                            (onset - float(trace.metadata["transition_time_s"])) * 1000.0
                        )
                rows.append(
                    {
                        "split": split,
                        "scenario": scenario,
                        "window_ms": window_ms,
                        "imu_separation": _separation(
                            [values[:, 4:10] for values in normal_prefix],
                            [values[:, 4:10] for values in hazard_prefix],
                        ),
                        "fsr_separation": _separation(
                            [values[:, :4] for values in normal_prefix],
                            [values[:, :4] for values in hazard_prefix],
                        ),
                        "hazard_onset_coverage": (
                            sum(latency < window_ms for latency in onset_latencies) / len(hazards)
                        ),
                        "hazard_onsets": len(onset_latencies),
                        "hazard_runs": len(hazards),
                        "normal_negative_runs": len(normals),
                    }
                )
    return rows


def _plot_trace(output: Path, trace: FastReflexTrace) -> None:
    import matplotlib.pyplot as plt

    relative = RELATIVE_TRANSITION_TIME_MS
    scenario = str(trace.metadata["scenario"])
    if scenario == "marble_to_ice":
        figure, axes = plt.subplots(5, 1, figsize=(9, 10), sharex=True)
        axes[0].plot(relative, trace.sensors[:, :4].sum(axis=1))
        axes[0].set_ylabel("FSR sum [N]")
        axes[1].plot(relative, trace.sensors[:, 4])
        axes[1].set_ylabel("Accel X")
        axes[2].plot(relative, trace.oracle[:, ORACLE_INDEX["foot_horizontal_speed_mps"]])
        axes[2].set_ylabel("Foot vXY")
        axes[3].plot(relative, trace.oracle[:, ORACLE_INDEX["Ft_over_Fn"]])
        axes[3].set_ylabel("Ft/Fn")
        axes[4].step(relative, trace.slip.astype(int), where="post")
        axes[4].set_ylabel("Slip GT")
        hazard = onset_time_s(trace, trace.slip)
    elif scenario == "marble_to_sand":
        figure, axes = plt.subplots(5, 1, figsize=(9, 10), sharex=True)
        axes[0].plot(relative, trace.sensors[:, :4])
        axes[0].set_ylabel("FSR1..4")
        imbalance = trace.sensors[:, :4].max(axis=1) - trace.sensors[:, :4].min(axis=1)
        axes[1].plot(relative, imbalance)
        axes[1].set_ylabel("FSR imbalance")
        axes[2].plot(relative, trace.sensors[:, 7:9])
        axes[2].set_ylabel("Gyro X/Y")
        axes[3].plot(relative, trace.oracle[:, ORACLE_INDEX["foot_sink_depth_m"]] * 1000.0)
        axes[3].set_ylabel("Sink [mm]")
        axes[4].step(relative, trace.sink.astype(int), where="post", label="sink")
        axes[4].step(relative, trace.tilt.astype(int), where="post", label="tilt")
        axes[4].set_ylabel("Hazard GT")
        axes[4].legend()
        hazard = onset_time_s(trace, trace.sink_or_tilt)
    else:
        return
    for axis in axes:
        axis.axvline(0.0, color="black", linestyle="--", label="transition")
        if hazard is not None:
            latency = (hazard - float(trace.metadata["transition_time_s"])) * 1000.0
            axis.axvline(latency, color="red", linestyle=":", label="hazard onset")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("relative transition time [ms]")
    figure.suptitle(str(trace.metadata["run_id"]))
    figure.tight_layout()
    figure.savefig(output / f"{scenario}_diagnostic.png", dpi=150)
    plt.close(figure)


def _save_artifacts(
    output: Path,
    traces: list[FastReflexTrace],
    rows: list[dict[str, object]],
) -> None:
    sensors = np.asarray([trace.sensors for trace in traces], dtype=np.float32)
    timestamps = np.asarray([trace.timestamps_s for trace in traces], dtype=np.float64)
    np.savez_compressed(
        output / "inputs_fusion10.npz",
        sensors=sensors,
        sample_time_s=timestamps,
        relative_transition_time_ms=RELATIVE_TRANSITION_TIME_MS,
        run_id=np.asarray([trace.metadata["run_id"] for trace in traces]),
    )
    np.savez_compressed(
        output / "oracle_diagnostics.npz",
        oracle=np.asarray([trace.oracle for trace in traces], dtype=np.float32),
        oracle_channels=np.asarray(ORACLE_CHANNELS),
        sample_time_s=timestamps,
        relative_transition_time_ms=RELATIVE_TRANSITION_TIME_MS,
        relative_hazard_time_ms=np.asarray(
            [
                np.full(TRACE_SAMPLES, np.nan, dtype=np.float64)
                if (
                    onset := onset_time_s(
                        trace, trace.slip | trace.sink | trace.tilt,
                    )
                ) is None
                else (trace.timestamps_s - onset) * 1000.0
                for trace in traces
            ]
        ),
        slip=np.asarray([trace.slip for trace in traces], dtype=bool),
        sink=np.asarray([trace.sink for trace in traces], dtype=bool),
        tilt=np.asarray([trace.tilt for trace in traces], dtype=bool),
        run_id=np.asarray([trace.metadata["run_id"] for trace in traces]),
    )
    replay = output / "hil_replay"
    replay.mkdir()
    for trace in traces:
        run_id = str(trace.metadata["run_id"])
        np.savez_compressed(
            replay / f"{run_id}.npz",
            sequence=np.arange(TRACE_SAMPLES, dtype=np.uint32),
            timestamp_s=trace.timestamps_s,
            fusion10=trace.sensors.astype(np.float32),
        )
    write_dict_rows(output / "manifest.csv", rows)

    for alignment in ("transition", "hazard"):
        for window_ms in (1, 2, *PRIMARY_WINDOWS_MS):
            selected, selected_ids = [], []
            for trace in traces:
                label = trace.slip | trace.sink | trace.tilt
                if alignment == "hazard" and not np.any(label):
                    continue
                try:
                    _, prefix = extract_prefix(
                        trace, window_ms, alignment,
                        hazard_label=label if alignment == "hazard" else None,
                    )
                except ValueError:
                    # A late onset may not leave the requested post-onset span
                    # inside the canonical transition trace.  Never pad or rerun.
                    continue
                selected.append(prefix)
                selected_ids.append(trace.metadata["run_id"])
            if selected:
                np.savez_compressed(
                    output / f"{alignment}_aligned_{window_ms:02d}ms.npz",
                    sensors=np.asarray(selected, dtype=np.float32),
                    run_id=np.asarray(selected_ids),
                )


def main() -> None:
    args = parse_args()
    payload = protocol(args)
    if not args.execute:
        print(json.dumps(payload, indent=2))
        print("Dry run only. Pass --execute to generate transition traces.")
        return
    output = (args.output_dir or OUTPUT_DIR).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)

    raw_runs = []
    for scenario in args.scenarios:
        for family in args.families:
            for surface_index in range(args.surfaces_per_family):
                for run_index in range(args.runs_per_surface):
                    raw = run_transition(scenario, family, surface_index, run_index)
                    raw_runs.append(raw)
                    print(
                        f"run={raw.trace.metadata['run_id']} state_delta="
                        f"{raw.qpos_transition_delta:.1e}/{raw.qvel_transition_delta:.1e} "
                        f"finite={int(raw.trace.valid)}"
                    )

    calibration = [
        raw.trace for raw in raw_runs
        if raw.trace.metadata["scenario"] in NORMAL_SCENARIOS
        and raw.trace.metadata["split"] == "train"
    ]
    thresholds = calibrate_hazard_thresholds(calibration)
    traces = [label_trace(raw.trace, thresholds) for raw in raw_runs]
    rows = []
    for raw, trace in zip(raw_runs, traces):
        onset = _onset_fields(trace)
        row = {
            **trace.metadata,
            **onset,
            "sample_count": TRACE_SAMPLES,
            "sample_rate_hz": SENSOR_RATE_HZ,
            "valid": int(trace.valid),
            "invalid_reason": trace.invalid_reason,
            "wall_time_s": raw.wall_time_s,
        }
        rows.append(row)
        print(
            f"  scenario={row['scenario']} hazard={row['hazard_type']} "
            f"onset={row['hazard_onset_time_s'] or 'none'} "
            f"latency_ms={row['transition_to_hazard_ms'] or 'none'}"
        )
    validate_split_integrity(rows)
    _save_artifacts(output, traces, rows)
    separation_rows = separation_summary(traces)
    if separation_rows:
        write_dict_rows(output / "window_separation.csv", separation_rows)
    payload["thresholds"] = asdict(thresholds)
    payload["derived_artifact_revision"] = 2
    payload["hazard_metadata_semantics"] = (
        "hazard_type/onset follow physical oracle masks for every scenario; "
        "expected_hazard remains scenario intent only"
    )
    payload["measured"] = {
        "runs": len(traces),
        "valid_runs": sum(trace.valid for trace in traces),
        "hazard_onsets": sum(row["hazard_onset_time_s"] != "" for row in rows),
        "wall_time_s": sum(raw.wall_time_s for raw in raw_runs),
    }
    (output / "protocol.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    if args.plot:
        for scenario in ("marble_to_ice", "marble_to_sand"):
            trace = next(
                (item for item in traces if item.metadata["scenario"] == scenario),
                None,
            )
            if trace is not None:
                _plot_trace(output, trace)
    summary = [
        f"{SCHEMA_NAME} schema_version={SCHEMA_VERSION}",
        f"runs={len(traces)} valid={sum(trace.valid for trace in traces)}",
        "native_sampling=1000Hz physics=2000Hz spacing=1ms",
        f"thresholds={json.dumps(asdict(thresholds), sort_keys=True)}",
    ]
    for scenario in args.scenarios:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        latencies = [
            float(
                row["transition_to_target_hazard_ms"]
                if row["expected_hazard"] != "normal"
                else row["transition_to_hazard_ms"]
            )
            for row in scenario_rows
            if (
                row["transition_to_target_hazard_ms"]
                if row["expected_hazard"] != "normal"
                else row["transition_to_hazard_ms"]
            ) != ""
        ]
        summary.append(
            f"{scenario}: runs={len(scenario_rows)} onsets={len(latencies)} "
            f"latency_ms={','.join(f'{value:.3f}' for value in latencies) or 'none'}"
        )
    (output / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
