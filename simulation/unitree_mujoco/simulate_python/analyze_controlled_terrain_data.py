"""Analyze repeated controlled G1 terrain windows without a classifier."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hil_sensor import HIL_SENSOR_CHANNELS
from terrain_profiles import TERRAIN_PROFILES


SIMULATE_PYTHON_DIR = Path(__file__).resolve().parent
SIMULATION_DIR = SIMULATE_PYTHON_DIR.parents[1]
DEFAULT_DATA_DIR = SIMULATION_DIR / "outputs" / "controlled_terrain_data"
DEFAULT_ANALYSIS_DIR = SIMULATION_DIR / "outputs" / "controlled_terrain_analysis"
DEFAULT_BASELINE_DIR = SIMULATION_DIR / "outputs" / "terrain_data"
TERRAINS = tuple(TERRAIN_PROFILES.keys())
COLORS = {
    "concrete": "tab:gray",
    "marble": "tab:blue",
    "ice": "tab:cyan",
    "sand": "tab:orange",
}
FEATURES = (
    "force_impulse",
    "contact_duration",
    "peak_point_force",
    "peak_force_sum",
    "force_distribution_1",
    "force_distribution_2",
    "force_distribution_3",
    "force_distribution_4",
    "accel_rms",
    "gyro_rms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze controlled repeated terrain windows."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def load_run(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"empty run: {path}")
    timestamps = np.asarray([float(row["timestamp"]) for row in rows])
    sensors = np.asarray(
        [[float(row[channel]) for channel in HIL_SENSOR_CHANNELS] for row in rows]
    )
    body_collision = np.asarray(
        [bool(int(row["body_collision_latched"])) for row in rows]
    )
    valid_foot = np.asarray([bool(int(row["valid_foot_contact"])) for row in rows])
    if sensors.shape != (50, 10):
        raise ValueError(f"expected run sensor shape (50, 10), got {sensors.shape}: {path}")
    if timestamps.shape != (50,) or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"invalid timestamps: {path}")
    if not np.all(np.isfinite(sensors)) or not np.all(np.isfinite(timestamps)):
        raise ValueError(f"NaN or Inf: {path}")
    return timestamps, sensors, body_collision, valid_foot


def save_run_variation(
    rows_by_terrain: dict[str, list[dict[str, str]]], output_path: Path
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("terrain", "feature", "mean", "std", "cv", "min", "max"))
        for terrain in TERRAINS:
            for feature in (*FEATURES, "settling_time"):
                values = np.asarray(
                    [
                        float(row[feature])
                        for row in rows_by_terrain[terrain]
                        if row[feature] != ""
                    ]
                )
                if values.size == 0:
                    writer.writerow((terrain, feature, "", "", "", "", ""))
                    continue
                mean = float(values.mean())
                std = float(values.std())
                cv = std / abs(mean) if abs(mean) > 1e-12 else np.nan
                writer.writerow(
                    (
                        terrain,
                        feature,
                        f"{mean:.9f}",
                        f"{std:.9f}",
                        f"{cv:.9f}",
                        f"{values.min():.9f}",
                        f"{values.max():.9f}",
                    )
                )


def feature_matrix(
    rows: list[dict[str, str]],
) -> np.ndarray:
    return np.asarray([[float(row[feature]) for feature in FEATURES] for row in rows])


def pairwise_separation(
    rows_by_terrain: dict[str, list[dict[str, str]]],
    windows: dict[tuple[str, str], np.ndarray],
    output_path: Path,
) -> list[dict[str, float | str]]:
    all_feature_values = np.concatenate(
        [feature_matrix(rows_by_terrain[terrain]) for terrain in TERRAINS]
    )
    feature_scale = all_feature_values.std(axis=0)
    feature_scale[feature_scale < 1e-12] = 1.0
    all_window_values = np.concatenate(list(windows.values()))
    channel_scale = all_window_values.std(axis=0)
    channel_scale[channel_scale < 1e-12] = 1.0
    results = []

    for first, second in combinations(TERRAINS, 2):
        first_features = feature_matrix(rows_by_terrain[first]) / feature_scale
        second_features = feature_matrix(rows_by_terrain[second]) / feature_scale
        first_centroid = first_features.mean(axis=0)
        second_centroid = second_features.mean(axis=0)
        centroid_distance = float(np.linalg.norm(first_centroid - second_centroid))
        first_spread = np.sqrt(np.mean(np.sum((first_features - first_centroid) ** 2, axis=1)))
        second_spread = np.sqrt(
            np.mean(np.sum((second_features - second_centroid) ** 2, axis=1))
        )
        pooled_spread = float(np.sqrt((first_spread**2 + second_spread**2) / 2.0))
        separation_ratio = centroid_distance / max(pooled_spread, 1e-12)

        first_runs = {row["run_id"] for row in rows_by_terrain[first]}
        second_runs = {row["run_id"] for row in rows_by_terrain[second]}
        paired_distances = []
        for run_id in sorted(first_runs & second_runs):
            difference = (windows[(first, run_id)] - windows[(second, run_id)]) / channel_scale
            paired_distances.append(float(np.sqrt(np.mean(difference**2))))

        results.append(
            {
                "terrain_1": first,
                "terrain_2": second,
                "centroid_distance": centroid_distance,
                "pooled_run_spread": pooled_spread,
                "separation_ratio": separation_ratio,
                "paired_window_rms_mean": float(np.mean(paired_distances)),
                "paired_window_rms_std": float(np.std(paired_distances)),
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    return results


def save_feature_boxplots(
    rows_by_terrain: dict[str, list[dict[str, str]]], output_path: Path
) -> None:
    selected = (
        "force_impulse",
        "contact_duration",
        "peak_point_force",
        "accel_rms",
        "gyro_rms",
        "settling_time",
    )
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    for axis, feature in zip(axes.flat, selected):
        values = []
        for terrain in TERRAINS:
            terrain_values = [
                float(row[feature])
                for row in rows_by_terrain[terrain]
                if row[feature] != ""
            ]
            values.append(terrain_values if terrain_values else [np.nan])
        box = axis.boxplot(values, tick_labels=TERRAINS, patch_artist=True)
        for patch, terrain in zip(box["boxes"], TERRAINS):
            patch.set_facecolor(COLORS[terrain])
            patch.set_alpha(0.65)
        axis.set_title(feature)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Controlled run-to-run feature variation")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_force_distribution(
    rows_by_terrain: dict[str, list[dict[str, str]]], output_path: Path
) -> None:
    x = np.arange(4)
    width = 0.19
    figure, axis = plt.subplots(figsize=(9, 4.5))
    for terrain_index, terrain in enumerate(TERRAINS):
        distribution = np.asarray(
            [
                [float(row[f"force_distribution_{point}"]) for point in range(1, 5)]
                for row in rows_by_terrain[terrain]
            ]
        )
        axis.bar(
            x + (terrain_index - 1.5) * width,
            distribution.mean(axis=0),
            width,
            yerr=distribution.std(axis=0),
            capsize=2,
            label=terrain,
            color=COLORS[terrain],
        )
    axis.set_xticks(x, [f"point {index}" for index in range(1, 5)])
    axis.set_ylabel("fraction of left-foot force impulse")
    axis.set_ylim(0.0, 0.55)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_example_timeseries(
    windows: dict[tuple[str, str], np.ndarray], output_path: Path
) -> None:
    timestamps = np.arange(1, 51) / 50.0
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for terrain in TERRAINS:
        values = windows[(terrain, "run_001")]
        axes[0].plot(timestamps, values[:, :4].sum(axis=1), label=terrain, color=COLORS[terrain])
        axes[1].plot(timestamps, np.linalg.norm(values[:, 4:7], axis=1), color=COLORS[terrain])
        axes[2].plot(timestamps, np.linalg.norm(values[:, 7:10], axis=1), color=COLORS[terrain])
    axes[0].set_ylabel("force sum [N]")
    axes[1].set_ylabel("accel magnitude")
    axes[2].set_ylabel("gyro magnitude")
    axes[2].set_xlabel("simulation time [s]")
    axes[0].legend(ncol=4)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Paired excitation example: run_001")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_window_pca(
    rows_by_terrain: dict[str, list[dict[str, str]]],
    windows: dict[tuple[str, str], np.ndarray],
    output_path: Path,
) -> tuple[float, float]:
    labels = []
    flattened = []
    for terrain in TERRAINS:
        for row in rows_by_terrain[terrain]:
            labels.append(terrain)
            flattened.append(windows[(terrain, row["run_id"])].reshape(-1))
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
        axis.scatter(
            projection[mask, 0],
            projection[mask, 1],
            label=terrain,
            color=COLORS[terrain],
            alpha=0.75,
            s=28,
        )
    axis.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    axis.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    axis.set_title("PCA of controlled 50x10 run windows")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return float(explained[0]), float(explained[1])


def save_separation_heatmap(
    results: list[dict[str, float | str]], output_path: Path
) -> None:
    matrix = np.zeros((len(TERRAINS), len(TERRAINS)))
    for result in results:
        first = TERRAINS.index(str(result["terrain_1"]))
        second = TERRAINS.index(str(result["terrain_2"]))
        value = float(result["separation_ratio"])
        matrix[first, second] = matrix[second, first] = value
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="viridis")
    axis.set_xticks(range(len(TERRAINS)), TERRAINS)
    axis.set_yticks(range(len(TERRAINS)), TERRAINS)
    for row in range(len(TERRAINS)):
        for column in range(len(TERRAINS)):
            axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", color="white")
    axis.set_title("Feature centroid distance / run-to-run spread")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def baseline_extremes(baseline_dir: Path) -> dict[str, tuple[float, float]]:
    results = {}
    for terrain in TERRAINS:
        path = baseline_dir / f"{terrain}.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))
        values = np.asarray(
            [[float(row[channel]) for channel in HIL_SENSOR_CHANNELS] for row in rows]
        )
        results[terrain] = (
            float(values[:, :4].max()),
            float(np.abs(values[:, 4:7]).max()),
        )
    return results


def write_summary(
    rows_by_terrain: dict[str, list[dict[str, str]]],
    all_rows_by_terrain: dict[str, list[dict[str, str]]],
    windows: dict[tuple[str, str], np.ndarray],
    separation: list[dict[str, float | str]],
    pca_explained: tuple[float, float],
    baseline: dict[str, tuple[float, float]],
    output_path: Path,
) -> None:
    lines = [
        "Controlled terrain dataset analysis (no classifier)",
        "",
        "Protocol:",
        "- window=1.0s, rate=50Hz, sensor_shape=(50,10)",
        "- vertical ElasticBand support ratio=0.70",
        "- paired seeded excitation conditions across terrains",
        "",
        "Terrain summaries:",
    ]
    for terrain in TERRAINS:
        rows = rows_by_terrain[terrain]
        all_rows = all_rows_by_terrain[terrain]
        valid_count = len(rows)
        body_count = sum(int(row["body_collision"]) for row in all_rows)
        impulse = np.asarray([float(row["force_impulse"]) for row in rows])
        contact = np.asarray([float(row["contact_duration"]) for row in rows])
        peak = np.asarray([float(row["peak_point_force"]) for row in rows])
        accel = np.asarray([float(row["accel_rms"]) for row in rows])
        gyro = np.asarray([float(row["gyro_rms"]) for row in rows])
        settle = np.asarray([float(row["settling_time"]) for row in rows if row["settling_time"] != ""])
        controlled_force_max = max(float(windows[(terrain, row["run_id"])][:, :4].max()) for row in rows)
        controlled_accel_max = max(float(np.abs(windows[(terrain, row["run_id"])][:, 4:7]).max()) for row in rows)
        baseline_values = baseline.get(terrain)
        baseline_text = (
            "baseline unavailable"
            if baseline_values is None
            else f"baseline_peak_force={baseline_values[0]:.3f}, baseline_abs_accel={baseline_values[1]:.3f}"
        )
        lines.extend(
            (
                f"- {terrain}: valid={valid_count}/{len(all_rows)}, body_collision={body_count}",
                f"  impulse={impulse.mean():.3f}±{impulse.std():.3f} Ns, contact={contact.mean():.3f}±{contact.std():.3f}s",
                f"  peak={peak.mean():.3f}±{peak.std():.3f} N, accel_rms={accel.mean():.3f}±{accel.std():.3f}, gyro_rms={gyro.mean():.3f}±{gyro.std():.3f}",
                f"  settling_available={settle.size}/{len(rows)}" + (f", settling={settle.mean():.3f}±{settle.std():.3f}s" if settle.size else ""),
                f"  controlled_max_force={controlled_force_max:.3f}, controlled_max_abs_accel={controlled_accel_max:.3f}, {baseline_text}",
            )
        )
    lines.extend(("", "Pairwise separation:"))
    for result in sorted(separation, key=lambda item: float(item["separation_ratio"])):
        lines.append(
            f"- {result['terrain_1']} vs {result['terrain_2']}: "
            f"ratio={float(result['separation_ratio']):.3f}, "
            f"paired_window_rms={float(result['paired_window_rms_mean']):.3f}±{float(result['paired_window_rms_std']):.3f}"
        )
    lines.extend(
        (
            "",
            f"PCA explained variance: PC1={pca_explained[0]:.6f}, PC2={pca_explained[1]:.6f}",
            "",
            "Limits:",
            "- Contact profiles remain engineering approximations, not measured materials.",
            "- Excitation ranges are narrow and deterministic; broader repeated sessions are not yet covered.",
            "- PCA and separation ratios are descriptive statistics, not classifier performance.",
        )
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    analysis_dir = args.analysis_dir.resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = load_manifest(data_dir / "run_manifest.csv")
    all_rows_by_terrain = {
        terrain: [row for row in manifest_rows if row["terrain"] == terrain]
        for terrain in TERRAINS
    }
    rows_by_terrain = {
        terrain: [
            row
            for row in all_rows_by_terrain[terrain]
            if int(row["valid_run"]) == 1
        ]
        for terrain in TERRAINS
    }
    if any(not rows for rows in rows_by_terrain.values()):
        raise ValueError("every terrain needs at least one valid run")

    windows = {}
    for terrain in TERRAINS:
        for row in rows_by_terrain[terrain]:
            _, sensors, body_collision, _ = load_run(data_dir / row["csv_path"])
            if np.any(body_collision):
                raise ValueError(f"valid run contains body collision: {row['csv_path']}")
            windows[(terrain, row["run_id"])] = sensors

    save_run_variation(rows_by_terrain, analysis_dir / "run_to_run_variation.csv")
    separation = pairwise_separation(
        rows_by_terrain, windows, analysis_dir / "pairwise_separation.csv"
    )
    save_feature_boxplots(rows_by_terrain, analysis_dir / "feature_boxplots.png")
    save_force_distribution(rows_by_terrain, analysis_dir / "force_distribution.png")
    save_example_timeseries(windows, analysis_dir / "paired_run_001_timeseries.png")
    pca_explained = save_window_pca(
        rows_by_terrain, windows, analysis_dir / "controlled_window_pca.png"
    )
    save_separation_heatmap(
        separation, analysis_dir / "pairwise_separation_heatmap.png"
    )
    baseline = baseline_extremes(args.baseline_dir.resolve())
    write_summary(
        rows_by_terrain,
        all_rows_by_terrain,
        windows,
        separation,
        pca_explained,
        baseline,
        analysis_dir / "controlled_analysis_summary.txt",
    )

    print(f"data_dir={data_dir}")
    print(f"analysis_dir={analysis_dir}")
    for terrain in TERRAINS:
        print(f"terrain={terrain} valid_runs={len(rows_by_terrain[terrain])}")
    closest = min(separation, key=lambda item: float(item["separation_ratio"]))
    print(
        f"closest_pair={closest['terrain_1']}:{closest['terrain_2']} "
        f"separation_ratio={float(closest['separation_ratio']):.6f}"
    )
    print(
        f"pca_explained=PC1:{pca_explained[0]:.6f},PC2:{pca_explained[1]:.6f}"
    )


if __name__ == "__main__":
    main()
