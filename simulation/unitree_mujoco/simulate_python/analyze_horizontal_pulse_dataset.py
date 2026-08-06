"""Analyze pulse-driven slip diagnostics and the unchanged 10-channel input."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analyze_controlled_terrain_data import (
    COLORS,
    FEATURES,
    TERRAINS,
    feature_matrix,
    pairwise_separation,
    save_feature_boxplots,
    save_force_distribution,
    save_separation_heatmap,
    save_window_pca,
)
from hil_sensor import HIL_SENSOR_CHANNELS
from run_horizontal_pulse_dataset import SIMULATION_DIR
from slip_diagnostics import DIAGNOSTIC_CHANNELS


DEFAULT_DATA_DIR = SIMULATION_DIR / "outputs" / "horizontal_pulse_terrain_data"
DEFAULT_ANALYSIS_DIR = SIMULATION_DIR / "outputs" / "horizontal_pulse_terrain_analysis"
DIAGNOSTIC_FEATURES = (
    "pulse_max_pelvis_speed",
    "pulse_max_foot_speed",
    "pulse_slip_displacement",
    "total_slip_displacement",
    "pulse_peak_tangential_force",
    "pulse_mean_force_ratio",
    "pulse_contact_fraction",
)
PULSE_RESPONSE_FEATURES = (
    "pulse_peak_force_sum",
    "accel_rms",
    "gyro_rms",
    "pulse_peak_accel_magnitude",
    "pulse_peak_gyro_magnitude",
    "post_settling_time",
)
PULSE_TEN_CHANNEL_FEATURES = (
    "pulse_force_impulse",
    "pulse_force_distribution_1",
    "pulse_force_distribution_2",
    "pulse_force_distribution_3",
    "pulse_force_distribution_4",
    "pulse_peak_force_sum",
    "pulse_accel_rms",
    "pulse_gyro_rms",
    "pulse_peak_accel_magnitude",
    "pulse_peak_gyro_magnitude",
)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def load_run(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = load_rows(path)
    timestamps = np.asarray([float(row["timestamp"]) for row in rows])
    sensors = np.asarray([[float(row[key]) for key in HIL_SENSOR_CHANNELS] for row in rows])
    diagnostics = np.asarray([[float(row[key]) for key in DIAGNOSTIC_CHANNELS] for row in rows])
    if sensors.shape != (50, 10) or diagnostics.shape != (50, len(DIAGNOSTIC_CHANNELS)):
        raise ValueError(f"unexpected run shape: {path}")
    if np.any(np.diff(timestamps) <= 0) or not np.all(np.isfinite(sensors)) or not np.all(np.isfinite(diagnostics)):
        raise ValueError(f"invalid run values: {path}")
    return timestamps, sensors, diagnostics


def generic_separation(
    rows_by_terrain: dict[str, list[dict[str, str]]],
    features: tuple[str, ...],
    output_path: Path,
) -> list[dict[str, float | str]]:
    all_values = np.concatenate(
        [np.asarray([[float(row[key]) for key in features] for row in rows_by_terrain[t]]) for t in TERRAINS]
    )
    scale = all_values.std(axis=0)
    scale[scale < 1e-12] = 1.0
    results = []
    for first, second in combinations(TERRAINS, 2):
        a = np.asarray([[float(row[key]) for key in features] for row in rows_by_terrain[first]]) / scale
        b = np.asarray([[float(row[key]) for key in features] for row in rows_by_terrain[second]]) / scale
        ca, cb = a.mean(axis=0), b.mean(axis=0)
        sa = np.sqrt(np.mean(np.sum((a - ca) ** 2, axis=1)))
        sb = np.sqrt(np.mean(np.sum((b - cb) ** 2, axis=1)))
        pooled = float(np.sqrt((sa * sa + sb * sb) / 2.0))
        results.append(
            {
                "terrain_1": first,
                "terrain_2": second,
                "centroid_distance": float(np.linalg.norm(ca - cb)),
                "pooled_run_spread": pooled,
                "separation_ratio": float(np.linalg.norm(ca - cb)) / max(pooled, 1e-12),
            }
        )
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return results


def save_boxplots(
    rows_by_terrain: dict[str, list[dict[str, str]]],
    features: tuple[str, ...],
    title: str,
    output_path: Path,
) -> None:
    columns = 3
    rows_count = int(np.ceil(len(features) / columns))
    figure, axes = plt.subplots(rows_count, columns, figsize=(13, 3.8 * rows_count))
    axes_flat = np.atleast_1d(axes).ravel()
    for axis, feature in zip(axes_flat, features):
        values = [
            [float(row[feature]) for row in rows_by_terrain[terrain] if row[feature] != ""]
            for terrain in TERRAINS
        ]
        box = axis.boxplot(values, tick_labels=TERRAINS, patch_artist=True)
        for patch, terrain in zip(box["boxes"], TERRAINS):
            patch.set_facecolor(COLORS[terrain])
            patch.set_alpha(0.65)
        axis.set_title(feature)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    for axis in axes_flat[len(features) :]:
        axis.set_visible(False)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_aligned_example(
    runs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    pulse_start: float,
    pulse_end: float,
    pulse_magnitude: float,
    output_path: Path,
) -> None:
    common_run_ids = set.intersection(
        *(
            {run_id for terrain_name, run_id in runs if terrain_name == terrain}
            for terrain in TERRAINS
        )
    )
    if not common_run_ids:
        raise ValueError("no valid run_id is shared by every terrain")
    selected_run_id = sorted(common_run_ids)[0]
    figure, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for terrain in TERRAINS:
        time, sensors, diagnostics = runs[(terrain, selected_run_id)]
        axes[0].plot(time, diagnostics[:, 2], color=COLORS[terrain], label=terrain)
        axes[1].plot(time, diagnostics[:, 4] * 1000, color=COLORS[terrain])
        axes[2].plot(time, diagnostics[:, 5], color=COLORS[terrain])
        axes[3].plot(time, sensors[:, :4].sum(axis=1), color=COLORS[terrain])
    for axis in axes:
        axis.axvspan(pulse_start, pulse_end, color="black", alpha=0.08)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("foot vx [m/s]")
    axes[1].set_ylabel("slip path [mm]")
    axes[2].set_ylabel("tangent force [N]")
    axes[3].set_ylabel("normal sum [N]")
    axes[3].set_xlabel("time [s]")
    axes[0].legend(ncol=4)
    figure.suptitle(
        f"Time-aligned {pulse_magnitude:g} N half-sine response "
        f"({selected_run_id}); shaded=pulse"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def mean_std(rows: list[dict[str, str]], feature: str) -> tuple[float, float]:
    values = np.asarray([float(row[feature]) for row in rows if row[feature] != ""])
    return float(values.mean()), float(values.std())


def save_pulse_pca(
    valid: dict[str, list[dict[str, str]]],
    runs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    pulse_start: float,
    pulse_end: float,
    output_path: Path,
) -> tuple[float, float]:
    labels, flattened = [], []
    for terrain in TERRAINS:
        for row in valid[terrain]:
            timestamps, sensors, _ = runs[(terrain, row["run_id"])]
            mask = (timestamps >= pulse_start) & (timestamps < pulse_end)
            labels.append(terrain)
            flattened.append(sensors[mask].reshape(-1))
    matrix = np.asarray(flattened)
    scale = matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (matrix - matrix.mean(axis=0)) / scale
    _, singular_values, components = np.linalg.svd(standardized, full_matrices=False)
    projection = standardized @ components[:2].T
    explained = singular_values**2 / np.sum(singular_values**2)
    figure, axis = plt.subplots(figsize=(7, 6))
    for terrain in TERRAINS:
        mask = np.asarray([label == terrain for label in labels])
        axis.scatter(projection[mask, 0], projection[mask, 1], label=terrain, color=COLORS[terrain], alpha=0.75, s=28)
    axis.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    axis.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    axis.set_title("PCA of pulse-only 10x10 sensor windows")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return float(explained[0]), float(explained[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--baseline-ratio", type=float, default=0.246934)
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    analysis_dir = args.analysis_dir.resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)
    all_rows = load_rows(data_dir / "run_manifest.csv")
    pulse_start = float(all_rows[0]["pulse_start_time"])
    pulse_duration = float(all_rows[0]["pulse_duration"])
    pulse_magnitude = float(all_rows[0]["pulse_magnitude"])
    all_by_terrain = {terrain: [row for row in all_rows if row["terrain"] == terrain] for terrain in TERRAINS}
    valid = {terrain: [row for row in all_by_terrain[terrain] if int(row["valid_run"])] for terrain in TERRAINS}
    runs = {}
    windows = {}
    for terrain in TERRAINS:
        for row in valid[terrain]:
            run = load_run(data_dir / row["csv_path"])
            runs[(terrain, row["run_id"])] = run
            windows[(terrain, row["run_id"])] = run[1]

    ten_channel = pairwise_separation(valid, windows, analysis_dir / "ten_channel_separation.csv")
    diagnostic = generic_separation(valid, DIAGNOSTIC_FEATURES, analysis_dir / "diagnostic_separation.csv")
    pulse_ten_channel = generic_separation(
        valid,
        PULSE_TEN_CHANNEL_FEATURES,
        analysis_dir / "pulse_ten_channel_separation.csv",
    )
    save_separation_heatmap(ten_channel, analysis_dir / "ten_channel_separation_heatmap.png")
    pca = save_window_pca(valid, windows, analysis_dir / "ten_channel_pca.png")
    pulse_pca = save_pulse_pca(
        valid,
        runs,
        pulse_start,
        pulse_start + pulse_duration,
        analysis_dir / "pulse_ten_channel_pca.png",
    )
    save_feature_boxplots(valid, analysis_dir / "ten_channel_feature_boxplots.png")
    save_force_distribution(valid, analysis_dir / "force_distribution.png")
    save_boxplots(valid, DIAGNOSTIC_FEATURES, "Slip diagnostic feature distributions", analysis_dir / "diagnostic_feature_boxplots.png")
    save_boxplots(valid, PULSE_RESPONSE_FEATURES, "Pulse response feature distributions", analysis_dir / "pulse_response_boxplots.png")
    save_aligned_example(
        runs,
        pulse_start,
        pulse_start + pulse_duration,
        pulse_magnitude,
        analysis_dir / "aligned_run_001.png",
    )

    cm = next(row for row in ten_channel if row["terrain_1"] == "concrete" and row["terrain_2"] == "marble")
    lines = [
        "Horizontal pulse terrain analysis (no classifier)", "",
        f"Protocol: half-sine +X force, start={pulse_start:.2f}s, "
        f"duration={pulse_duration:.2f}s, peak={pulse_magnitude:g}N; "
        "window=1.0s at 50Hz; support=70%.",
        f"Baseline concrete-marble ratio={args.baseline_ratio:.6f}",
        f"Pulse concrete-marble ratio={float(cm['separation_ratio']):.6f}",
        f"Change={(float(cm['separation_ratio']) / args.baseline_ratio - 1.0) * 100:.2f}%", "",
        "Terrain diagnostics (valid runs):",
    ]
    for terrain in TERRAINS:
        slip = mean_std(valid[terrain], "pulse_slip_displacement")
        speed = mean_std(valid[terrain], "pulse_max_foot_speed")
        tangential = mean_std(valid[terrain], "pulse_peak_tangential_force")
        ratio = mean_std(valid[terrain], "pulse_mean_force_ratio")
        contact = mean_std(valid[terrain], "pulse_contact_fraction")
        settled = [row for row in valid[terrain] if row["post_settling_time"] != ""]
        lines.extend((
            f"- {terrain}: valid={len(valid[terrain])}/{len(all_by_terrain[terrain])}, body_collision={sum(int(r['body_collision']) for r in all_by_terrain[terrain])}",
            f"  pulse_slip={slip[0]*1000:.3f}±{slip[1]*1000:.3f}mm, max_foot_speed={speed[0]:.4f}±{speed[1]:.4f}m/s",
            f"  peak_tangent={tangential[0]:.3f}±{tangential[1]:.3f}N, force_ratio={ratio[0]:.3f}±{ratio[1]:.3f}, pulse_contact={contact[0]*100:.1f}±{contact[1]*100:.1f}%",
            f"  sustained_settling={len(settled)}/{len(valid[terrain])} by 1.0s",
        ))
    lines.extend(("", "10-channel pairwise separation:"))
    for row in ten_channel:
        lines.append(f"- {row['terrain_1']} vs {row['terrain_2']}: {float(row['separation_ratio']):.6f}")
    lines.extend(("", "Diagnostic pairwise separation:"))
    for row in diagnostic:
        lines.append(f"- {row['terrain_1']} vs {row['terrain_2']}: {float(row['separation_ratio']):.6f}")
    lines.extend(("", "Pulse-only 10-channel feature separation (not baseline-equivalent):"))
    for row in pulse_ten_channel:
        lines.append(f"- {row['terrain_1']} vs {row['terrain_2']}: {float(row['separation_ratio']):.6f}")
    lines.extend((
        "", f"10-channel PCA explained: PC1={pca[0]:.6f}, PC2={pca[1]:.6f}",
        f"Pulse-only PCA explained: PC1={pulse_pca[0]:.6f}, PC2={pulse_pca[1]:.6f}",
        f"Extreme flags: force={sum(int(r['extreme_force_outlier']) for r in all_rows)}, accel={sum(int(r['extreme_accel_outlier']) for r in all_rows)}",
    ))
    (analysis_dir / "horizontal_pulse_analysis_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"analysis_dir={analysis_dir}")


if __name__ == "__main__":
    main()
