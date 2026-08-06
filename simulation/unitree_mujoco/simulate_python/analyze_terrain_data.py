"""Compare four terrain CSV datasets without training a classifier."""

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
DEFAULT_DATA_DIR = SIMULATION_DIR / "outputs" / "terrain_data"
DEFAULT_PLOT_DIR = SIMULATION_DIR / "outputs" / "terrain_plots"
TERRAIN_NAMES = tuple(TERRAIN_PROFILES.keys())
COLORS = {
    "concrete": "tab:gray",
    "marble": "tab:blue",
    "ice": "tab:cyan",
    "sand": "tab:orange",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot and statistically compare terrain HIL datasets."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    return parser.parse_args()


def load_dataset(path: Path, expected_terrain: str) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        expected_columns = ["timestamp", *HIL_SENSOR_CHANNELS, "terrain"]
        if reader.fieldnames != expected_columns:
            raise ValueError(f"unexpected columns in {path}: {reader.fieldnames}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"empty dataset: {path}")
    terrains = {row["terrain"] for row in rows}
    if terrains != {expected_terrain}:
        raise ValueError(f"terrain label mismatch in {path}: {terrains}")

    timestamps = np.asarray([float(row["timestamp"]) for row in rows])
    values = np.asarray(
        [[float(row[channel]) for channel in HIL_SENSOR_CHANNELS] for row in rows]
    )
    if values.shape[1] != len(HIL_SENSOR_CHANNELS):
        raise ValueError(f"expected 10 channels in {path}, got {values.shape}")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(timestamps)):
        raise ValueError(f"NaN or Inf in {path}")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"non-monotonic timestamps in {path}")
    return timestamps, values


def save_time_series_plot(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    channel_indices: tuple[int, ...],
    title: str,
    output_path: Path,
) -> None:
    rows = len(channel_indices)
    figure, axes = plt.subplots(rows, 1, figsize=(10, 2.5 * rows), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, channel_index in zip(axes, channel_indices):
        for terrain in TERRAIN_NAMES:
            timestamps, values = datasets[terrain]
            axis.plot(
                timestamps,
                values[:, channel_index],
                label=terrain,
                color=COLORS[terrain],
                linewidth=1.2,
            )
        axis.set_ylabel(HIL_SENSOR_CHANNELS[channel_index])
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("simulation time [s]")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_force_sum_plot(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]], output_path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(10, 4))
    for terrain in TERRAIN_NAMES:
        timestamps, values = datasets[terrain]
        axis.plot(
            timestamps,
            values[:, :4].sum(axis=1),
            label=terrain,
            color=COLORS[terrain],
            linewidth=1.3,
        )
    axis.set(title="Left-foot normal-force sum", xlabel="simulation time [s]", ylabel="force [N]")
    axis.grid(alpha=0.25)
    axis.legend(ncol=4)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_mean_std_plots(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]], plot_dir: Path
) -> None:
    groups = {
        "foot_force_mean_std.png": tuple(range(4)),
        "accelerometer_mean_std.png": (4, 5, 6),
        "gyroscope_mean_std.png": (7, 8, 9),
    }
    for filename, indices in groups.items():
        x = np.arange(len(indices))
        width = 0.19
        figure, axis = plt.subplots(figsize=(9, 4.5))
        for terrain_index, terrain in enumerate(TERRAIN_NAMES):
            values = datasets[terrain][1][:, indices]
            axis.bar(
                x + (terrain_index - 1.5) * width,
                values.mean(axis=0),
                width,
                yerr=values.std(axis=0),
                capsize=2,
                label=terrain,
                color=COLORS[terrain],
                alpha=0.85,
            )
        axis.set_xticks(x, [HIL_SENSOR_CHANNELS[index] for index in indices])
        axis.set_ylabel("mean with temporal std")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(ncol=4, fontsize=8)
        figure.tight_layout()
        figure.savefig(plot_dir / filename, dpi=160)
        plt.close(figure)


def write_channel_summary(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]], output_path: Path
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("terrain", "channel", "min", "max", "mean", "std"))
        for terrain in TERRAIN_NAMES:
            values = datasets[terrain][1]
            for index, channel in enumerate(HIL_SENSOR_CHANNELS):
                column = values[:, index]
                writer.writerow(
                    (
                        terrain,
                        channel,
                        f"{column.min():.9f}",
                        f"{column.max():.9f}",
                        f"{column.mean():.9f}",
                        f"{column.std():.9f}",
                    )
                )


def pairwise_signal_distances(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]], output_path: Path
) -> list[tuple[str, str, float, np.ndarray]]:
    reference_timestamps = datasets[TERRAIN_NAMES[0]][0]
    all_values = np.concatenate([datasets[name][1] for name in TERRAIN_NAMES])
    channel_scale = all_values.std(axis=0)
    channel_scale[channel_scale < 1e-12] = 1.0
    results = []

    for first, second in combinations(TERRAIN_NAMES, 2):
        first_time, first_values = datasets[first]
        second_time, second_values = datasets[second]
        if not np.allclose(first_time, reference_timestamps) or not np.allclose(
            second_time, reference_timestamps
        ):
            raise ValueError("pairwise comparison requires aligned timestamps")
        per_channel = np.sqrt(
            np.mean(((first_values - second_values) / channel_scale) ** 2, axis=0)
        )
        overall = float(np.sqrt(np.mean(per_channel**2)))
        results.append((first, second, overall, per_channel))

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("terrain_1", "terrain_2", "overall_rms", *HIL_SENSOR_CHANNELS))
        for first, second, overall, per_channel in results:
            writer.writerow(
                (first, second, f"{overall:.9f}", *(f"{value:.9f}" for value in per_channel))
            )
    return results


def save_pca_plot(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]], output_path: Path
) -> tuple[float, float]:
    combined = np.concatenate([datasets[name][1] for name in TERRAIN_NAMES])
    center = combined.mean(axis=0)
    scale = combined.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (combined - center) / scale
    _, singular_values, components = np.linalg.svd(standardized, full_matrices=False)
    projection = standardized @ components[:2].T
    explained = singular_values**2 / np.sum(singular_values**2)

    figure, axis = plt.subplots(figsize=(7, 6))
    offset = 0
    for terrain in TERRAIN_NAMES:
        count = datasets[terrain][1].shape[0]
        points = projection[offset : offset + count]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=14,
            alpha=0.65,
            label=terrain,
            color=COLORS[terrain],
        )
        offset += count
    axis.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    axis.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    axis.set_title("Unsupervised PCA of raw 10-channel samples")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return float(explained[0]), float(explained[1])


def write_text_report(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    distances: list[tuple[str, str, float, np.ndarray]],
    explained: tuple[float, float],
    output_path: Path,
) -> None:
    closest = min(distances, key=lambda item: item[2])
    lines = [
        "Terrain virtual-sensor comparison (no classifier)",
        "",
        "Dataset:",
    ]
    for terrain in TERRAIN_NAMES:
        timestamps, values = datasets[terrain]
        rate = 1.0 / np.median(np.diff(timestamps))
        lines.append(
            f"- {terrain}: samples={values.shape[0]}, "
            f"time={timestamps[0]:.3f}..{timestamps[-1]:.3f}s, rate={rate:.1f}Hz"
        )
    lines.extend(("", "Pairwise normalized aligned-signal RMS:"))
    for first, second, overall, _ in sorted(distances, key=lambda item: item[2]):
        lines.append(f"- {first} vs {second}: {overall:.6f}")
    lines.extend(
        (
            "",
            f"Closest pair: {closest[0]} vs {closest[1]} ({closest[2]:.6f})",
            f"PCA explained variance: PC1={explained[0]:.6f}, PC2={explained[1]:.6f}",
            "",
            "Interpretation limits:",
            "- Profiles are relative engineering approximations, not measured material properties.",
            "- Samples come from one deterministic passive-drop/initial-slide trajectory.",
            "- PCA visualizes variance only; it is not a classifier or an accuracy result.",
        )
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    plot_dir = args.plot_dir.resolve()
    plot_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        terrain: load_dataset(data_dir / f"{terrain}.csv", terrain)
        for terrain in TERRAIN_NAMES
    }
    counts = {values.shape[0] for _, values in datasets.values()}
    if len(counts) != 1:
        raise ValueError(f"terrain sample counts differ: {counts}")

    save_time_series_plot(
        datasets, (0, 1, 2, 3), "Left-foot point normal forces", plot_dir / "foot_force_timeseries.png"
    )
    save_time_series_plot(
        datasets, (4, 5, 6), "Pelvis IMU accelerometer", plot_dir / "accelerometer_timeseries.png"
    )
    save_time_series_plot(
        datasets, (7, 8, 9), "Pelvis IMU gyroscope", plot_dir / "gyroscope_timeseries.png"
    )
    save_force_sum_plot(datasets, plot_dir / "foot_force_sum_timeseries.png")
    save_mean_std_plots(datasets, plot_dir)
    write_channel_summary(datasets, plot_dir / "channel_summary.csv")
    distances = pairwise_signal_distances(
        datasets, plot_dir / "pairwise_signal_rms.csv"
    )
    explained = save_pca_plot(datasets, plot_dir / "pca_projection.png")
    write_text_report(
        datasets, distances, explained, plot_dir / "terrain_signal_summary.txt"
    )

    closest = min(distances, key=lambda item: item[2])
    print(f"data_dir={data_dir}")
    print(f"plot_dir={plot_dir}")
    print(f"samples_per_terrain={next(iter(counts))}")
    print(
        f"closest_pair={closest[0]}:{closest[1]} "
        f"normalized_aligned_rms={closest[2]:.6f}"
    )
    print(
        f"pca_explained_variance=PC1:{explained[0]:.6f},PC2:{explained[1]:.6f}"
    )


if __name__ == "__main__":
    main()
