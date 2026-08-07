"""Run and analyze the matched friction x roughness factorization study."""

from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from pathlib import Path

import numpy as np

from controlled_excitation import HorizontalPulse, generate_pulse_conditions
from hil_sensor import HIL_SENSOR_CHANNELS
from run_controlled_terrain_dataset import calculate_run_metrics, write_manifest
from run_horizontal_pulse_dataset import DEFAULT_SEED, SIMULATION_DIR, run_window
from run_surface_sampling_rate_study import (
    FSR_FEATURES,
    IMU_FEATURES,
    SURFACE_SCENE_PATH,
    periodogram,
    run_arrays,
    separation,
    spectral_metrics,
    write_dict_rows,
)
from surface_profiles import (
    CONCRETE_PEAK_TO_VALLEY_M,
    GRID_SPACING_M,
    MARBLE_PEAK_TO_VALLEY_M,
    WAVELENGTHS_M,
    configure_factorized_surface,
)
from terrain_profiles import TERRAIN_PROFILES, TerrainProfile


SCENE_PATH = SURFACE_SCENE_PATH
OUTPUT_DIR = SIMULATION_DIR / "outputs" / "surface_factorization_study"
RUN_COUNT = 20
SAMPLE_RATE = 200.0
PULSE = HorizontalPulse(0.25, 0.20, 80.0, 1.0, 0.0)
CELLS = {
    "A_high_smooth": ("concrete", "marble"),
    "B_high_rough": ("concrete", "concrete"),
    "C_low_smooth": ("marble", "marble"),
    "D_low_rough": ("marble", "concrete"),
}
COMPARISONS = {
    "roughness_at_high_friction": ("A_high_smooth", "B_high_rough"),
    "roughness_at_low_friction": ("C_low_smooth", "D_low_rough"),
    "friction_at_smooth_surface": ("A_high_smooth", "C_low_smooth"),
    "friction_at_rough_surface": ("B_high_rough", "D_low_rough"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def factor_configurator(
    model, profile: TerrainProfile, roughness_source: str
) -> int:
    floor_id, _ = configure_factorized_surface(model, profile, roughness_source)
    return floor_id


def derived_metrics(
    timestamps: np.ndarray, sensors: np.ndarray
) -> dict[str, float]:
    forces = sensors[:, :4]
    total = forces.sum(axis=1)
    contact = total > 1.0
    heel = forces[:, :2].sum(axis=1)
    toe = forces[:, 2:].sum(axis=1)
    impulse = forces.sum(axis=0) / SAMPLE_RATE
    impulse_total = max(float(impulse.sum()), 1e-12)
    x_positions = np.asarray((-0.05, -0.05, 0.12, 0.12))
    cop = np.sum(forces[contact] * x_positions, axis=1) / total[contact]
    pulse_mask = (timestamps >= PULSE.start_time) & (
        timestamps < PULSE.start_time + PULSE.duration
    )
    spectral, _ = spectral_metrics(timestamps, sensors, SAMPLE_RATE)
    result = {
        "total_normal_force_mean": float(total.mean()),
        "total_normal_force_rms": float(np.sqrt(np.mean(total**2))),
        "heel_load_fraction": float(impulse[:2].sum() / impulse_total),
        "toe_load_fraction": float(impulse[2:].sum() / impulse_total),
        "cop_proxy_x_mean": float(cop.mean()),
        "cop_proxy_x_rms": float(np.sqrt(np.mean(cop**2))),
        "pulse_contact_fraction": float(np.mean(total[pulse_mask] > 1.0)),
    }
    result.update(spectral)
    mask = (timestamps >= 0.20) & (timestamps < 0.60)
    for axis_index, axis in enumerate("xyz"):
        frequencies, psd = periodogram(sensors[mask, 4 + axis_index], SAMPLE_RATE)
        df = frequencies[1] - frequencies[0]
        for low, high in ((0, 20), (20, 50), (50, 90)):
            band = (frequencies > low) & (frequencies <= high)
            result[f"accel_{axis}_band_{low}_{high}_power"] = float(
                psd[band].sum() * df
            )
    return result


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    conditions = generate_pulse_conditions(RUN_COUNT, DEFAULT_SEED)
    common_solref = TERRAIN_PROFILES["concrete"].solref
    common_solimp = TERRAIN_PROFILES["concrete"].solimp
    assert common_solref == TERRAIN_PROFILES["marble"].solref
    assert common_solimp == TERRAIN_PROFILES["marble"].solimp
    protocol = {
        "cells": CELLS,
        "runs_per_cell": RUN_COUNT,
        "duration_s": 1.0,
        "physics_timestep_s": 0.005,
        "sensor_rate_hz": SAMPLE_RATE,
        "seed": DEFAULT_SEED,
        "pulse": {"peak_N": 80.0, "start_s": 0.25, "duration_s": 0.20},
        "support_ratio": 0.70,
        "high_friction": TERRAIN_PROFILES["concrete"].friction,
        "low_friction": TERRAIN_PROFILES["marble"].friction,
        "rough_ptv_m": CONCRETE_PEAK_TO_VALLEY_M,
        "smooth_ptv_m": MARBLE_PEAK_TO_VALLEY_M,
        "grid_spacing_m": GRID_SPACING_M,
        "wavelengths_m": WAVELENGTHS_M,
        "controlled_solref": common_solref,
        "controlled_solimp": common_solimp,
        "factor_definition": "Only friction tuple and hfield amplitude source vary; solref/solimp are identical and fixed.",
        "noise": False,
        "domain_randomization": False,
        "classifier": False,
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )

    for cell, (friction_source, roughness_source) in CELLS.items():
        profile = TERRAIN_PROFILES[friction_source]
        configurator = partial(
            factor_configurator, roughness_source=roughness_source
        )
        print(f"collecting {cell} runs={RUN_COUNT}")
        for condition in conditions:
            metrics = run_window(
                friction_source,
                condition,
                1.0,
                SAMPLE_RATE,
                DEFAULT_SEED,
                0.70,
                PULSE,
                output / cell,
                scene_path=SCENE_PATH,
                matched_pelvis_output_dir=output / "pelvis_diagnostic" / cell,
                model_configurator=configurator,
            )
            print(
                f"{cell}/{condition.run_id} valid={metrics['valid_run']} "
                f"collision={metrics['body_collision']}"
            )

    rows_by_cell: dict[str, list[dict[str, float | int | str]]] = {}
    for cell, (friction_source, _) in CELLS.items():
        rows_by_cell[cell] = []
        for condition in conditions:
            relative = Path(friction_source) / f"{condition.run_id}.csv"
            timestamps, sensors, collision, contact = run_arrays(output / cell / relative)
            metrics = calculate_run_metrics(
                timestamps, sensors, collision, contact, SAMPLE_RATE
            )
            metrics.update(derived_metrics(timestamps, sensors))
            for index, channel in enumerate(HIL_SENSOR_CHANNELS):
                metrics[f"channel_rms_{channel}"] = float(
                    np.sqrt(np.mean(sensors[:, index] ** 2))
                )
            metrics.update(
                {
                    "run_id": condition.run_id,
                    "cell": cell,
                    "friction_source": CELLS[cell][0],
                    "roughness_source": CELLS[cell][1],
                    "csv_path": str(relative),
                }
            )
            rows_by_cell[cell].append(metrics)
        write_manifest(output / cell / "run_manifest.csv", rows_by_cell[cell])

    valid_ids = {condition.run_id for condition in conditions}
    for rows in rows_by_cell.values():
        valid_ids &= {str(row["run_id"]) for row in rows if int(row["valid_run"])}
    selected = {
        cell: [row for row in rows if str(row["run_id"]) in valid_ids]
        for cell, rows in rows_by_cell.items()
    }
    groups = {
        "10_channel": (*FSR_FEATURES, *IMU_FEATURES),
        "fsr_only": FSR_FEATURES,
        "imu_only": IMU_FEATURES,
    }
    pair_rows: list[dict[str, object]] = []
    for comparison, (first, second) in COMPARISONS.items():
        for group, features in groups.items():
            distance, spread, ratio = separation(
                selected[first], selected[second], features
            )
            pair_rows.append(
                {
                    "comparison": comparison,
                    "cell_1": first,
                    "cell_2": second,
                    "feature_group": group,
                    "centroid_distance": f"{distance:.9f}",
                    "pooled_spread": f"{spread:.9f}",
                    "separation_ratio": f"{ratio:.9f}",
                }
            )
    write_dict_rows(output / "pairwise_separation.csv", pair_rows)

    derived = (
        "total_normal_force_rms",
        "heel_load_fraction",
        "toe_load_fraction",
        "cop_proxy_x_mean",
        "accel_rms",
        "gyro_rms",
        "accel_x_vibration_rms",
        "accel_z_vibration_rms",
        *(f"accel_z_band_{low}_{high}_power" for low, high in ((0, 20), (20, 50), (50, 90))),
    )
    raw = tuple(f"channel_rms_{channel}" for channel in HIL_SENSOR_CHANNELS)
    feature_rows: list[dict[str, object]] = []
    for comparison, (first, second) in COMPARISONS.items():
        for feature in (*raw, *derived):
            distance, spread, ratio = separation(
                selected[first], selected[second], (feature,)
            )
            feature_rows.append(
                {
                    "comparison": comparison,
                    "feature": feature,
                    "centroid_distance": f"{distance:.9f}",
                    "pooled_spread": f"{spread:.9f}",
                    "separation_ratio": f"{ratio:.9f}",
                }
            )
    write_dict_rows(output / "feature_separation.csv", feature_rows)

    all_features = {**groups, "derived": derived}
    effect_rows: list[dict[str, object]] = []
    for group, features in all_features.items():
        matrices = {
            cell: np.asarray([[float(row[f]) for f in features] for row in selected[cell]])
            for cell in CELLS
        }
        scale = np.concatenate(list(matrices.values())).std(axis=0)
        scale[scale < 1e-12] = 1.0
        means = {cell: values.mean(axis=0) / scale for cell, values in matrices.items()}
        rough = 0.5 * ((means["B_high_rough"] - means["A_high_smooth"]) + (means["D_low_rough"] - means["C_low_smooth"]))
        friction = 0.5 * ((means["C_low_smooth"] - means["A_high_smooth"]) + (means["D_low_rough"] - means["B_high_rough"]))
        interaction = (means["D_low_rough"] - means["C_low_smooth"]) - (means["B_high_rough"] - means["A_high_smooth"])
        effect_rows.append(
            {
                "feature_group": group,
                "roughness_main_effect_norm": f"{np.linalg.norm(rough):.9f}",
                "friction_main_effect_norm": f"{np.linalg.norm(friction):.9f}",
                "interaction_effect_norm": f"{np.linalg.norm(interaction):.9f}",
            }
        )
    write_dict_rows(output / "factor_effects.csv", effect_rows)

    cell_rows: list[dict[str, object]] = []
    for cell, rows in rows_by_cell.items():
        cell_rows.append(
            {
                "cell": cell,
                "valid_runs": sum(int(row["valid_run"]) for row in rows),
                "body_collision": sum(int(row["body_collision"]) for row in rows),
                "extreme_force": sum(int(row["extreme_force_outlier"]) for row in rows),
                "extreme_accel": sum(int(row["extreme_accel_outlier"]) for row in rows),
                "contact_fraction_mean": f"{np.mean([float(row['pulse_contact_fraction']) for row in rows]):.9f}",
            }
        )
    write_dict_rows(output / "cell_summary.csv", cell_rows)
    ratios = {(r["comparison"], r["feature_group"]): float(r["separation_ratio"]) for r in pair_rows}
    effects = {r["feature_group"]: r for r in effect_rows}
    lines = [
        "Friction x roughness factorization (200 Hz physics/logging)", "",
        f"Matched valid run IDs: {len(valid_ids)}/{RUN_COUNT}.",
        "solref and solimp were identical in the existing material profiles and held fixed; friction tuple and existing hfield amplitude were the only factors.", "",
    ]
    for row in cell_rows:
        lines.append(f"- {row['cell']}: valid={row['valid_runs']}/{RUN_COUNT}, collision={row['body_collision']}, extreme_force={row['extreme_force']}, extreme_accel={row['extreme_accel']}, pulse_contact={float(row['contact_fraction_mean']):.3f}")
    lines.append("")
    for comparison in COMPARISONS:
        lines.append(f"- {comparison}: 10={ratios[(comparison,'10_channel')]:.6f}, FSR={ratios[(comparison,'fsr_only')]:.6f}, IMU={ratios[(comparison,'imu_only')]:.6f}")
    lines.append("")
    for group, row in effects.items():
        lines.append(f"- effects/{group}: roughness={float(row['roughness_main_effect_norm']):.6f}, friction={float(row['friction_main_effect_norm']):.6f}, interaction={float(row['interaction_effect_norm']):.6f}")
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
