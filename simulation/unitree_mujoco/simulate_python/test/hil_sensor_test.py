"""Continuously print the G1 10-channel HIL vector without DDS or a GUI."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import mujoco


SIMULATE_PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMULATE_PYTHON_DIR))

import config  # noqa: E402
from hil_sensor import (  # noqa: E402
    G1HilSensorReader,
    HIL_SENSOR_CHANNELS,
    LEFT_FOOT_CONTACT_GEOM_NAMES,
    format_hil_vector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print G1 foot-contact and IMU values for HIL verification."
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0 / config.VIEWER_DT,
        help="console update rate in Hz (default: %(default).1f)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="stop after this many seconds; 0 runs until Ctrl-C",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rate <= 0:
        raise ValueError("--rate must be greater than zero")
    if args.duration < 0:
        raise ValueError("--duration must not be negative")
    if config.ROBOT != "g1":
        raise ValueError(f'config.ROBOT must be "g1", got {config.ROBOT!r}')

    scene_path = (SIMULATE_PYTHON_DIR / config.ROBOT_SCENE).resolve()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    model.opt.timestep = config.SIMULATE_DT
    reader = G1HilSensorReader(model, data)

    print(f"scene={scene_path}")
    print(f"simulation_rate={1.0 / model.opt.timestep:.1f} Hz")
    print(f"console_rate={args.rate:.1f} Hz")
    print(f"channels={HIL_SENSOR_CHANNELS}")
    print(
        "left_foot_geoms="
        + str(tuple(zip(LEFT_FOOT_CONTACT_GEOM_NAMES, reader.left_foot_geom_ids)))
    )

    output_period = 1.0 / args.rate
    next_output_time = output_period
    wall_start = time.perf_counter()

    try:
        while args.duration == 0.0 or data.time < args.duration:
            step_start = time.perf_counter()
            mujoco.mj_step(model, data)

            if data.time + 1e-12 >= next_output_time:
                print(
                    f"t={data.time:8.3f} HIL={format_hil_vector(reader.read_vector())}",
                    flush=True,
                )
                next_output_time += output_period

            remaining = model.opt.timestep - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass

    wall_elapsed = time.perf_counter() - wall_start
    print(f"stopped_at_sim_time={data.time:.3f}s wall_time={wall_elapsed:.3f}s")


if __name__ == "__main__":
    main()
