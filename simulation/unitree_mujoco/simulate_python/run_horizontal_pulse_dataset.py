"""Generate time-aligned G1 terrain windows with a controlled force pulse."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import threading
import time

import mujoco
import numpy as np

import config
from controlled_excitation import (
    ExcitationCondition,
    HorizontalPulse,
    HorizontalPulseExciter,
    VerticalElasticBandSupport,
    apply_excitation_condition,
    find_allowed_foot_geom_ids,
    generate_pulse_conditions,
    has_nonfoot_floor_contact,
)
from hil_sensor import G1HilSensorReader, HIL_SENSOR_CHANNELS
from run_controlled_terrain_dataset import calculate_run_metrics, validate_run, write_manifest
from slip_diagnostics import DIAGNOSTIC_CHANNELS, G1SlipDiagnosticReader
from terrain_profiles import TERRAIN_PROFILES, apply_terrain_profile


SIMULATE_PYTHON_DIR = Path(__file__).resolve().parent
SIMULATION_DIR = SIMULATE_PYTHON_DIR.parents[1]
DEFAULT_OUTPUT_DIR = SIMULATION_DIR / "outputs" / "horizontal_pulse_terrain_data"
DEFAULT_GUI_OUTPUT_DIR = SIMULATION_DIR / "outputs" / "horizontal_pulse_gui_preview"
DEFAULT_SEED = 20260805
RUN_COLUMNS = (
    "run_id",
    "session_id",
    "terrain",
    "dataset_seed",
    "pulse_direction",
    "base_height_offset",
    "base_roll_deg",
    "base_pitch_deg",
    "support_ratio",
    "pulse_start_time",
    "pulse_duration",
    "pulse_magnitude",
    "pulse_direction_x",
    "pulse_direction_y",
    "timestamp",
    *HIL_SENSOR_CHANNELS,
    *DIAGNOSTIC_CHANNELS,
    "body_floor_collision",
    "body_collision_latched",
    "valid_foot_contact",
    "candidate_valid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain", choices=(*TERRAIN_PROFILES, "all"), required=True)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--support-ratio", type=float, default=0.70)
    parser.add_argument("--pulse-start", type=float, default=0.25)
    parser.add_argument("--pulse-duration", type=float, default=0.20)
    parser.add_argument("--pulse-magnitude", type=float, default=80.0)
    parser.add_argument("--pulse-direction-x", type=float, default=1.0)
    parser.add_argument("--pulse-direction-y", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--gui",
        action="store_true",
        help="show one terrain/run in the passive MuJoCo viewer",
    )
    parser.add_argument(
        "--realtime-factor",
        type=float,
        default=1.0,
        help="GUI simulation speed relative to wall time (default: 1.0)",
    )
    parser.add_argument(
        "--gui-hold-seconds",
        type=float,
        default=3.0,
        help="keep the final GUI frame visible after simulation (default: 3)",
    )
    parser.add_argument(
        "--gui-output-dir",
        type=Path,
        default=DEFAULT_GUI_OUTPUT_DIR,
        help="separate preview output used only with --gui",
    )
    return parser.parse_args()


def diagnostic_metrics(
    timestamps: np.ndarray,
    sensors: np.ndarray,
    diagnostics: np.ndarray,
    pulse: HorizontalPulse,
    sample_rate: float,
) -> dict[str, float | str]:
    pulse_mask = (timestamps >= pulse.start_time) & (
        timestamps < pulse.start_time + pulse.duration
    )
    post_mask = timestamps >= pulse.start_time + pulse.duration
    foot_speed = np.linalg.norm(diagnostics[:, 2:4], axis=1)
    pelvis_speed = np.linalg.norm(diagnostics[:, 0:2], axis=1)
    normal = diagnostics[:, 6]
    tangential = diagnostics[:, 5]
    slip = diagnostics[:, 4]
    contact_mask = normal > 1.0
    active_contact = pulse_mask & contact_mask
    force_sum = sensors[:, :4].sum(axis=1)
    accel_magnitude = np.linalg.norm(sensors[:, 4:7], axis=1)
    gyro_magnitude = np.linalg.norm(sensors[:, 7:10], axis=1)
    pulse_forces = sensors[pulse_mask, :4]
    pulse_impulse_by_point = pulse_forces.sum(axis=0) / sample_rate
    pulse_impulse = float(pulse_impulse_by_point.sum())
    pulse_distribution = pulse_impulse_by_point / max(pulse_impulse, 1e-12)

    settle = ""
    settle_threshold = 0.02
    required = max(2, int(round(0.10 * sample_rate)))
    post_indices = np.flatnonzero(post_mask)
    for index in post_indices:
        if index + required > len(foot_speed):
            break
        if np.all(foot_speed[index:] <= settle_threshold):
            settle = float(timestamps[index] - (pulse.start_time + pulse.duration))
            break

    weighted_ratio = float(tangential[active_contact].sum() / max(normal[active_contact].sum(), 1e-12))
    return {
        "pulse_max_pelvis_speed": float(pelvis_speed[pulse_mask].max()),
        "pulse_max_foot_speed": float(foot_speed[pulse_mask].max()),
        "pulse_slip_displacement": float(slip[pulse_mask][-1] - slip[pulse_mask][0]),
        "total_slip_displacement": float(slip[-1]),
        "pulse_peak_tangential_force": float(tangential[pulse_mask].max()),
        "pulse_mean_force_ratio": weighted_ratio,
        "pulse_contact_fraction": float(np.mean(contact_mask[pulse_mask])),
        "pulse_peak_force_sum": float(force_sum[pulse_mask].max()),
        "pulse_force_impulse": pulse_impulse,
        "pulse_force_distribution_1": float(pulse_distribution[0]),
        "pulse_force_distribution_2": float(pulse_distribution[1]),
        "pulse_force_distribution_3": float(pulse_distribution[2]),
        "pulse_force_distribution_4": float(pulse_distribution[3]),
        "pulse_accel_rms": float(np.sqrt(np.mean(accel_magnitude[pulse_mask] ** 2))),
        "pulse_gyro_rms": float(np.sqrt(np.mean(gyro_magnitude[pulse_mask] ** 2))),
        "pulse_peak_accel_magnitude": float(accel_magnitude[pulse_mask].max()),
        "pulse_peak_gyro_magnitude": float(gyro_magnitude[pulse_mask].max()),
        "post_settling_time": settle,
    }


def write_run_csv(
    output_path: Path,
    terrain: str,
    seed: int,
    condition: ExcitationCondition,
    support_ratio: float,
    pulse: HorizontalPulse,
    timestamps: np.ndarray,
    sensors: np.ndarray,
    diagnostics: np.ndarray,
    body_collision: np.ndarray,
    collision_latched: np.ndarray,
    valid_contact: np.ndarray,
    session_id: str = "",
    pulse_direction_label: str = "",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(RUN_COLUMNS)
        for index, timestamp in enumerate(timestamps):
            writer.writerow(
                (
                    condition.run_id, session_id, terrain, seed,
                    pulse_direction_label,
                    f"{condition.base_height_offset:.9f}",
                    f"{condition.base_roll_deg:.9f}", f"{condition.base_pitch_deg:.9f}",
                    f"{support_ratio:.9f}", f"{pulse.start_time:.9f}",
                    f"{pulse.duration:.9f}", f"{pulse.magnitude:.9f}",
                    f"{pulse.direction_x:.9f}", f"{pulse.direction_y:.9f}",
                    f"{timestamp:.9f}",
                    *(f"{value:.9f}" for value in sensors[index]),
                    *(f"{value:.9f}" for value in diagnostics[index]),
                    int(body_collision[index]), int(collision_latched[index]),
                    int(valid_contact[index]),
                    int(valid_contact[index] and not collision_latched[index]),
                )
            )


def pulse_direction_name(pulse: HorizontalPulse) -> str:
    direction = np.asarray((pulse.direction_x, pulse.direction_y), dtype=np.float64)
    direction /= np.linalg.norm(direction)
    if np.allclose(direction, (1.0, 0.0)):
        return "positive_x"
    if np.allclose(direction, (-1.0, 0.0)):
        return "negative_x"
    return f"unit_{direction[0]:+.6f}_{direction[1]:+.6f}"


def run_window(
    terrain: str,
    condition: ExcitationCondition,
    duration: float,
    sample_rate: float,
    seed: int,
    support_ratio: float,
    pulse: HorizontalPulse,
    output_dir: Path | None,
    gui: bool = False,
    realtime_factor: float = 1.0,
    gui_hold_seconds: float = 3.0,
    session_id: str = "",
    pulse_direction_label: str = "",
    scene_path: Path | None = None,
    support_body_name: str = "torso_link",
    support_site_name: str | None = None,
    pulse_body_name: str = "torso_link",
    pulse_site_name: str | None = None,
) -> dict[str, float | int | str]:
    if not pulse_direction_label:
        pulse_direction_label = pulse_direction_name(pulse)
    scene_path = (
        (SIMULATE_PYTHON_DIR / config.ROBOT_SCENE).resolve()
        if scene_path is None
        else scene_path.resolve()
    )
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.timestep = config.SIMULATE_DT
    floor_id = apply_terrain_profile(model, TERRAIN_PROFILES[terrain])
    data = mujoco.MjData(model)
    qpos_address, dof_address = apply_excitation_condition(model, data, condition)
    support = VerticalElasticBandSupport(
        model,
        data,
        qpos_address,
        dof_address,
        support_ratio,
        application_body_name=support_body_name,
        application_site_name=support_site_name,
    )
    exciter = HorizontalPulseExciter(
        model,
        data,
        pulse,
        body_name=pulse_body_name,
        application_site_name=pulse_site_name,
    )
    sensor_reader = G1HilSensorReader(model, data)
    diagnostic_reader = G1SlipDiagnosticReader(model, data, floor_id)
    allowed_foot_geoms = find_allowed_foot_geom_ids(model)

    expected_samples = int(round(duration * sample_rate))
    sample_period = 1.0 / sample_rate
    next_sample_time = sample_period
    collision_since_sample = False
    collision_latched_value = False
    timestamps: list[float] = []
    sensors: list[np.ndarray] = []
    diagnostics: list[np.ndarray] = []
    body_collision: list[bool] = []
    collision_latched: list[bool] = []
    valid_contact: list[bool] = []

    viewer = None
    viewer_threads: list[threading.Thread] = []
    if gui:
        from mujoco import viewer as mujoco_viewer

        threads_before_launch = set(threading.enumerate())
        viewer = mujoco_viewer.launch_passive(model, data)
        viewer_threads = [
            thread
            for thread in threading.enumerate()
            if thread not in threads_before_launch
        ]

    try:
        wall_start = time.perf_counter()
        simulation_start = float(data.time)
        while data.time + 1e-12 < duration:
            if viewer is not None and not viewer.is_running():
                raise RuntimeError("MuJoCo viewer was closed before the run completed")
            support.apply()
            _, pulse_active = exciter.apply(float(data.time))
            mujoco.mj_step(model, data)
            diagnostic_reader.advance_slip(
                model.opt.timestep,
                pulse.start_time <= data.time,
            )
            collision_since_sample |= has_nonfoot_floor_contact(
                data, floor_id, allowed_foot_geoms
            )
            collision_latched_value |= collision_since_sample
            if data.time + 1e-12 >= next_sample_time:
                sensor_vector = sensor_reader.read_vector()
                diagnostic = diagnostic_reader.read(pulse_active)
                timestamps.append(float(data.time))
                sensors.append(sensor_vector)
                diagnostics.append(diagnostic.values)
                body_collision.append(collision_since_sample)
                collision_latched.append(collision_latched_value)
                valid_contact.append(bool(sensor_vector[:4].sum() > 1.0))
                collision_since_sample = False
                next_sample_time += sample_period
            if viewer is not None:
                viewer.sync()
                target_wall_elapsed = (
                    float(data.time) - simulation_start
                ) / realtime_factor
                remaining = target_wall_elapsed - (time.perf_counter() - wall_start)
                if remaining > 0.0:
                    time.sleep(remaining)

        if viewer is not None and gui_hold_seconds > 0.0:
            hold_deadline = time.perf_counter() + gui_hold_seconds
            while viewer.is_running() and time.perf_counter() < hold_deadline:
                viewer.sync()
                time.sleep(0.02)
    finally:
        if viewer is not None:
            viewer.close()
            # MuJoCo Handle.close() only requests exit.  Wait until its daemon
            # UI/reload threads finish GLFW/OpenGL teardown before Python exits.
            for viewer_thread in viewer_threads:
                viewer_thread.join()

    timestamp_array = np.asarray(timestamps)
    sensor_array = np.asarray(sensors)
    diagnostic_array = np.asarray(diagnostics)
    body_array = np.asarray(body_collision, dtype=bool)
    latched_array = np.asarray(collision_latched, dtype=bool)
    contact_array = np.asarray(valid_contact, dtype=bool)
    validate_run(timestamp_array, sensor_array, expected_samples)
    if diagnostic_array.shape != (expected_samples, len(DIAGNOSTIC_CHANNELS)):
        raise ValueError(f"unexpected diagnostic shape {diagnostic_array.shape}")
    if not np.all(np.isfinite(diagnostic_array)):
        raise ValueError("NaN or Inf in pulse diagnostics")

    relative_path = Path(terrain) / f"{condition.run_id}.csv"
    if output_dir is not None:
        write_run_csv(
            output_dir / relative_path, terrain, seed, condition, support_ratio, pulse,
            timestamp_array, sensor_array, diagnostic_array, body_array, latched_array,
            contact_array,
            session_id=session_id,
            pulse_direction_label=pulse_direction_label,
        )
    metrics = calculate_run_metrics(
        timestamp_array, sensor_array, latched_array, contact_array, sample_rate
    )
    metrics.update(
        diagnostic_metrics(
            timestamp_array, sensor_array, diagnostic_array, pulse, sample_rate
        )
    )
    metrics.update(
        {
            "run_id": condition.run_id, "session_id": session_id,
            "terrain": terrain, "dataset_seed": seed,
            "pulse_direction": pulse_direction_label,
            "base_height_offset": condition.base_height_offset,
            "base_roll_deg": condition.base_roll_deg,
            "base_pitch_deg": condition.base_pitch_deg,
            "support_ratio": support_ratio,
            "support_target_force": support.target_support_force,
            "pulse_start_time": pulse.start_time, "pulse_duration": pulse.duration,
            "pulse_magnitude": pulse.magnitude,
            "pulse_direction_x": pulse.direction_x,
            "pulse_direction_y": pulse.direction_y,
            "scene_path": str(scene_path),
            "support_body": support_body_name,
            "support_site": "" if support_site_name is None else support_site_name,
            "pulse_body": pulse_body_name,
            "pulse_site": "" if pulse_site_name is None else pulse_site_name,
            "csv_path": str(relative_path),
        }
    )
    return metrics


def main() -> None:
    args = parse_args()
    if config.ROBOT != "g1":
        raise ValueError('config.ROBOT must remain "g1"')
    if args.duration * args.sample_rate != round(args.duration * args.sample_rate):
        raise ValueError("duration * sample-rate must be an integer")
    if args.realtime_factor <= 0.0:
        raise ValueError("realtime-factor must be positive")
    if args.gui_hold_seconds < 0.0:
        raise ValueError("gui-hold-seconds must be non-negative")
    if args.gui and (args.terrain == "all" or args.runs != 1):
        raise ValueError("--gui requires one named terrain and --runs 1")
    pulse = HorizontalPulse(
        args.pulse_start, args.pulse_duration, args.pulse_magnitude,
        args.pulse_direction_x, args.pulse_direction_y,
    )
    if pulse.start_time + pulse.duration >= args.duration:
        raise ValueError("pulse must end before the run window")
    output_dir = (
        args.gui_output_dir.resolve() if args.gui else args.output_dir.resolve()
    )
    conditions = generate_pulse_conditions(args.runs, args.seed)
    terrains = tuple(TERRAIN_PROFILES) if args.terrain == "all" else (args.terrain,)
    rows = []
    for terrain in terrains:
        print(f"collecting terrain={terrain} runs={args.runs} pulse={pulse.magnitude}N")
        for condition in conditions:
            metrics = run_window(
                terrain, condition, args.duration, args.sample_rate, args.seed,
                args.support_ratio, pulse, output_dir,
                gui=args.gui,
                realtime_factor=args.realtime_factor,
                gui_hold_seconds=args.gui_hold_seconds,
            )
            rows.append(metrics)
            print(
                f"{terrain}/{condition.run_id} valid={metrics['valid_run']} "
                f"contact={metrics['pulse_contact_fraction']:.2f} "
                f"slip={metrics['pulse_slip_displacement']:.5f}m "
                f"body_collision={metrics['body_collision']}"
            )
    for terrain in terrains:
        terrain_rows = [row for row in rows if row["terrain"] == terrain]
        write_manifest(output_dir / terrain / "manifest.csv", terrain_rows)
    write_manifest(output_dir / "run_manifest.csv", rows)
    print(f"manifest={output_dir / 'run_manifest.csv'}")
    for terrain in terrains:
        terrain_rows = [row for row in rows if row["terrain"] == terrain]
        print(f"terrain={terrain} valid={sum(int(row['valid_run']) for row in terrain_rows)}/{len(terrain_rows)}")


if __name__ == "__main__":
    main()
