"""Analyze pulse-aligned bidirectional sessions without training a classifier."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analyze_controlled_terrain_data import COLORS, TERRAINS
from hil_sensor import HIL_SENSOR_CHANNELS
from pulse_windows import WINDOW_PROFILES, extract_window, window_shapes
from run_horizontal_pulse_dataset import SIMULATION_DIR
from slip_diagnostics import DIAGNOSTIC_CHANNELS


DEFAULT_DATA_DIR = SIMULATION_DIR / "outputs" / "bidirectional_pulse_validation"
DEFAULT_ANALYSIS_DIR = SIMULATION_DIR / "outputs" / "bidirectional_pulse_analysis"
FEATURES = (
    "force_impulse",
    "force_distribution_1",
    "force_distribution_2",
    "force_distribution_3",
    "force_distribution_4",
    "peak_force_sum",
    "accel_rms",
    "gyro_rms",
    "peak_accel_magnitude",
    "peak_gyro_magnitude",
)
PAIR_NAMES = tuple(combinations(TERRAINS, 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--baseline-concrete-marble", type=float, default=0.239)
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def load_run(path: Path, expected_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = load_csv(path)
    timestamps = np.asarray([float(row["timestamp"]) for row in rows])
    sensors = np.asarray(
        [[float(row[channel]) for channel in HIL_SENSOR_CHANNELS] for row in rows]
    )
    diagnostics = np.asarray(
        [[float(row[channel]) for channel in DIAGNOSTIC_CHANNELS] for row in rows]
    )
    if sensors.shape != (expected_samples, len(HIL_SENSOR_CHANNELS)):
        raise ValueError(f"unexpected sensor shape {sensors.shape}: {path}")
    if diagnostics.shape != (expected_samples, len(DIAGNOSTIC_CHANNELS)):
        raise ValueError(f"unexpected diagnostic shape {diagnostics.shape}: {path}")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"timestamps are not strictly monotonic: {path}")
    if not all(np.all(np.isfinite(values)) for values in (timestamps, sensors, diagnostics)):
        raise ValueError(f"NaN or Inf: {path}")
    return timestamps, sensors, diagnostics


def run_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return row["session_id"], row["pulse_direction"], row["terrain"], row["run_id"]


def window_feature_row(
    manifest_row: dict[str, str], sensors: np.ndarray, sample_rate: float, window: str
) -> dict[str, float | str]:
    forces = sensors[:, :4]
    impulse_by_point = forces.sum(axis=0) / sample_rate
    total_impulse = float(impulse_by_point.sum())
    distribution = impulse_by_point / max(total_impulse, 1e-12)
    accel_magnitude = np.linalg.norm(sensors[:, 4:7], axis=1)
    gyro_magnitude = np.linalg.norm(sensors[:, 7:10], axis=1)
    return {
        "window": window,
        "session_id": manifest_row["session_id"],
        "pulse_direction": manifest_row["pulse_direction"],
        "terrain": manifest_row["terrain"],
        "run_id": manifest_row["run_id"],
        "force_impulse": total_impulse,
        **{
            f"force_distribution_{index + 1}": float(value)
            for index, value in enumerate(distribution)
        },
        "peak_force_sum": float(forces.sum(axis=1).max()),
        "accel_rms": float(np.sqrt(np.mean(accel_magnitude**2))),
        "gyro_rms": float(np.sqrt(np.mean(gyro_magnitude**2))),
        "peak_accel_magnitude": float(accel_magnitude.max()),
        "peak_gyro_magnitude": float(gyro_magnitude.max()),
    }


def feature_matrix(rows: list[dict[str, float | str]]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in FEATURES] for row in rows])


def separation(
    rows: list[dict[str, float | str]], scale: np.ndarray
) -> list[dict[str, float | str]]:
    result = []
    for first, second in PAIR_NAMES:
        a_rows = [row for row in rows if row["terrain"] == first]
        b_rows = [row for row in rows if row["terrain"] == second]
        if not a_rows or not b_rows:
            continue
        a = feature_matrix(a_rows) / scale
        b = feature_matrix(b_rows) / scale
        a_centroid, b_centroid = a.mean(axis=0), b.mean(axis=0)
        a_spread = np.sqrt(np.mean(np.sum((a - a_centroid) ** 2, axis=1)))
        b_spread = np.sqrt(np.mean(np.sum((b - b_centroid) ** 2, axis=1)))
        pooled_spread = float(np.sqrt((a_spread**2 + b_spread**2) / 2.0))
        distance = float(np.linalg.norm(a_centroid - b_centroid))
        result.append(
            {
                "terrain_1": first,
                "terrain_2": second,
                "centroid_distance": distance,
                "pooled_run_spread": pooled_spread,
                "separation_ratio": distance / max(pooled_spread, 1e-12),
                "terrain_1_runs": len(a_rows),
                "terrain_2_runs": len(b_rows),
            }
        )
    return result


def write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_separation_rows(
    features_by_window: dict[str, list[dict[str, float | str]]],
    sessions: tuple[str, ...],
    directions: tuple[str, ...],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for window, rows in features_by_window.items():
        scale = feature_matrix(rows).std(axis=0)
        scale[scale < 1e-12] = 1.0
        groups: list[tuple[str, str, str, list[dict[str, float | str]]]] = [
            ("pooled", "all", "all", rows)
        ]
        groups.extend(
            ("session", session, "all", [row for row in rows if row["session_id"] == session])
            for session in sessions
        )
        groups.extend(
            ("direction", "all", direction, [row for row in rows if row["pulse_direction"] == direction])
            for direction in directions
        )
        groups.extend(
            (
                "session_direction",
                session,
                direction,
                [
                    row for row in rows
                    if row["session_id"] == session and row["pulse_direction"] == direction
                ],
            )
            for session in sessions
            for direction in directions
        )
        for scope, session, direction, group_rows in groups:
            for item in separation(group_rows, scale):
                output.append(
                    {
                        "window": window,
                        "scope": scope,
                        "session_id": session,
                        "pulse_direction": direction,
                        **item,
                    }
                )
    return output


def cm_rows(separation_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row for row in separation_rows
        if row["terrain_1"] == "concrete" and row["terrain_2"] == "marble"
    ]


def save_cm_by_window(rows: list[dict[str, object]], path: Path) -> None:
    selected = [row for row in rows if row["scope"] == "session_direction"]
    figure, axis = plt.subplots(figsize=(10, 5))
    labels = [f"{row['session_id'][-2:]}/{'+X' if row['pulse_direction'] == 'positive_x' else '-X'}" for row in selected if row["window"] == "full"]
    x = np.arange(len(labels))
    width = 0.19
    for offset, window in enumerate(WINDOW_PROFILES):
        values = [float(row["separation_ratio"]) for row in selected if row["window"] == window]
        axis.bar(x + (offset - 1.5) * width, values, width, label=window)
    axis.set_xticks(x, labels)
    axis.set_ylabel("concrete-marble separation ratio")
    axis.set_xlabel("session/direction")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_direction_comparison(rows: list[dict[str, object]], path: Path) -> None:
    selected = [row for row in rows if row["scope"] == "direction"]
    labels = [f"{first[:3]}-{second[:3]}" for first, second in PAIR_NAMES]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for axis, window in zip(axes.flat, WINDOW_PROFILES):
        window_rows = [row for row in selected if row["window"] == window]
        x = np.arange(len(PAIR_NAMES))
        for offset, direction in enumerate(("positive_x", "negative_x")):
            values = []
            for pair in PAIR_NAMES:
                row = next(
                    item for item in window_rows
                    if item["pulse_direction"] == direction
                    and (item["terrain_1"], item["terrain_2"]) == pair
                )
                values.append(float(row["separation_ratio"]))
            axis.bar(x + (-0.18 if offset == 0 else 0.18), values, 0.36, label=direction)
        axis.set_title(window)
        axis.set_xticks(x, labels, rotation=25)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend()
    figure.supylabel("separation ratio")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_session_variation(rows: list[dict[str, object]], path: Path) -> None:
    selected = [row for row in rows if row["scope"] == "session"]
    sessions = sorted({str(row["session_id"]) for row in selected})
    figure, axis = plt.subplots(figsize=(9, 5))
    for window in WINDOW_PROFILES:
        values = [
            float(next(row for row in selected if row["window"] == window and row["session_id"] == session)["separation_ratio"])
            for session in sessions
        ]
        axis.plot(sessions, values, marker="o", label=window)
    axis.set_ylabel("concrete-marble separation ratio")
    axis.set_title("Session-to-session variation (directions pooled)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_representative_signals(
    runs: dict[tuple[str, str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    sessions: tuple[str, ...],
    directions: tuple[str, ...],
    pulse_start: float,
    pulse_end: float,
    path: Path,
) -> None:
    session = sessions[0]
    figure, axes = plt.subplots(len(directions), 3, figsize=(13, 7), sharex=True)
    for row_index, direction in enumerate(directions):
        for terrain in TERRAINS:
            time, sensors, _ = runs[(session, direction, terrain, "run_001")]
            axes[row_index, 0].plot(time, sensors[:, :4].sum(axis=1), color=COLORS[terrain], label=terrain)
            axes[row_index, 1].plot(time, np.linalg.norm(sensors[:, 4:7], axis=1), color=COLORS[terrain])
            axes[row_index, 2].plot(time, np.linalg.norm(sensors[:, 7:10], axis=1), color=COLORS[terrain])
        for axis in axes[row_index]:
            axis.axvspan(pulse_start, pulse_end, color="black", alpha=0.08)
            axis.grid(alpha=0.25)
        axes[row_index, 0].set_ylabel(f"{direction}\nforce [N]")
    axes[0, 0].set_title("left-foot normal force sum")
    axes[0, 1].set_title("accelerometer magnitude")
    axes[0, 2].set_title("gyroscope magnitude")
    axes[0, 0].legend(ncol=4, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("simulation time [s]")
    figure.suptitle(f"Representative pulse-aligned response: {session}/run_001")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def validity_rows(
    manifest: list[dict[str, str]], sessions: tuple[str, ...], directions: tuple[str, ...]
) -> list[dict[str, object]]:
    rows = []
    for session in sessions:
        for direction in directions:
            for terrain in TERRAINS:
                group = [
                    row for row in manifest
                    if row["session_id"] == session
                    and row["pulse_direction"] == direction
                    and row["terrain"] == terrain
                ]
                rows.append(
                    {
                        "session_id": session,
                        "pulse_direction": direction,
                        "terrain": terrain,
                        "total_runs": len(group),
                        "valid_runs": sum(int(row["valid_run"]) for row in group),
                        "mean_pulse_contact_fraction": float(np.mean([float(row["pulse_contact_fraction"]) for row in group])),
                        "body_collision_count": sum(int(row["body_collision"]) for row in group),
                        "extreme_force_count": sum(int(row["extreme_force_outlier"]) for row in group),
                        "extreme_accel_count": sum(int(row["extreme_accel_outlier"]) for row in group),
                    }
                )
    return rows


def save_validity(rows: list[dict[str, object]], path: Path) -> None:
    groups = sorted({(str(row["session_id"]), str(row["pulse_direction"])) for row in rows})
    figure, axis = plt.subplots(figsize=(12, 5))
    x = np.arange(len(groups))
    width = 0.19
    for index, terrain in enumerate(TERRAINS):
        values = [
            100.0 * int(next(row for row in rows if row["session_id"] == session and row["pulse_direction"] == direction and row["terrain"] == terrain)["valid_runs"])
            / int(next(row for row in rows if row["session_id"] == session and row["pulse_direction"] == direction and row["terrain"] == terrain)["total_runs"])
            for session, direction in groups
        ]
        axis.bar(x + (index - 1.5) * width, values, width, label=terrain, color=COLORS[terrain])
    axis.set_xticks(x, [f"{session[-2:]}/{'+X' if direction == 'positive_x' else '-X'}" for session, direction in groups])
    axis.set_ylim(0, 105)
    axis.set_ylabel("valid runs [%]")
    axis.set_xlabel("session/direction")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def diagnostic_rows(
    manifest: list[dict[str, str]], directions: tuple[str, ...]
) -> list[dict[str, object]]:
    metrics = (
        "pulse_slip_displacement",
        "pulse_max_foot_speed",
        "pulse_peak_tangential_force",
        "pulse_mean_force_ratio",
        "pulse_contact_fraction",
    )
    output = []
    valid = [row for row in manifest if int(row["valid_run"])]
    for direction in directions:
        for terrain in TERRAINS:
            group = [row for row in valid if row["pulse_direction"] == direction and row["terrain"] == terrain]
            item: dict[str, object] = {
                "pulse_direction": direction,
                "terrain": terrain,
                "valid_runs": len(group),
            }
            for metric in metrics:
                values = np.asarray([float(row[metric]) for row in group])
                item[f"{metric}_mean"] = float(values.mean())
                item[f"{metric}_std"] = float(values.std())
            output.append(item)
    return output


def rank_correlation(first: list[float], second: list[float]) -> float:
    first_rank = np.argsort(np.argsort(np.asarray(first))).astype(float)
    second_rank = np.argsort(np.argsort(np.asarray(second))).astype(float)
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def save_pca(
    window: str,
    valid_manifest: list[dict[str, str]],
    extracted: dict[tuple[str, str, str, str, str], np.ndarray],
    path: Path,
) -> tuple[float, float]:
    matrix = np.asarray([extracted[(*run_key(row), window)].reshape(-1) for row in valid_manifest])
    labels = np.asarray([row["terrain"] for row in valid_manifest])
    scale = matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (matrix - matrix.mean(axis=0)) / scale
    _, singular, components = np.linalg.svd(standardized, full_matrices=False)
    projection = standardized @ components[:2].T
    explained = singular**2 / np.sum(singular**2)
    figure, axis = plt.subplots(figsize=(7, 6))
    for terrain in TERRAINS:
        mask = labels == terrain
        axis.scatter(projection[mask, 0], projection[mask, 1], color=COLORS[terrain], label=terrain, alpha=0.55, s=18)
    axis.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    axis.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    axis.set_title(f"PCA: {window} 10-channel window")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return float(explained[0]), float(explained[1])


def find_cm(
    rows: list[dict[str, object]], window: str, scope: str, session: str, direction: str
) -> float:
    row = next(
        item for item in rows
        if item["window"] == window and item["scope"] == scope
        and item["session_id"] == session and item["pulse_direction"] == direction
        and item["terrain_1"] == "concrete" and item["terrain_2"] == "marble"
    )
    return float(row["separation_ratio"])


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    analysis_dir = args.analysis_dir.resolve()
    if analysis_dir.exists() and any(analysis_dir.iterdir()):
        raise FileExistsError(f"analysis directory is not empty: {analysis_dir}")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_csv(data_dir / "manifest.csv")
    sessions = tuple(sorted({row["session_id"] for row in manifest}))
    directions = tuple(sorted({row["pulse_direction"] for row in manifest}, reverse=True))
    expected_samples = int(manifest[0]["sample_count"])
    duration = expected_samples / args.sample_rate
    pulse_start = float(manifest[0]["pulse_start_time"])
    pulse_end = pulse_start + float(manifest[0]["pulse_duration"])
    shapes = window_shapes(expected_samples, args.sample_rate)
    write_dict_rows(
        analysis_dir / "window_shapes.csv",
        [
            {
                "window": name,
                "start_time": profile.start_time,
                "end_time": "run_end" if profile.end_time is None else profile.end_time,
                "samples": shapes[name][0],
                "channels": shapes[name][1],
                "shape": f"({shapes[name][0]},{shapes[name][1]})",
            }
            for name, profile in WINDOW_PROFILES.items()
        ],
    )

    runs: dict[tuple[str, str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    extracted: dict[tuple[str, str, str, str, str], np.ndarray] = {}
    features_by_window = {name: [] for name in WINDOW_PROFILES}
    valid_manifest = [row for row in manifest if int(row["valid_run"])]
    for row in manifest:
        path = data_dir / row["csv_path"]
        run = load_run(path, expected_samples)
        runs[run_key(row)] = run
        if not int(row["valid_run"]):
            continue
        for name, profile in WINDOW_PROFILES.items():
            _, sensors = extract_window(run[0], run[1], profile, args.sample_rate)
            extracted[(*run_key(row), name)] = sensors
            features_by_window[name].append(window_feature_row(row, sensors, args.sample_rate, name))

    feature_rows = [row for rows in features_by_window.values() for row in rows]
    write_dict_rows(analysis_dir / "window_run_features.csv", feature_rows)
    separation_rows = build_separation_rows(features_by_window, sessions, directions)
    write_dict_rows(analysis_dir / "pairwise_separation_by_group.csv", separation_rows)
    concrete_marble = cm_rows(separation_rows)
    write_dict_rows(analysis_dir / "concrete_marble_reproducibility.csv", concrete_marble)

    validity = validity_rows(manifest, sessions, directions)
    write_dict_rows(analysis_dir / "validity_summary.csv", validity)
    diagnostics = diagnostic_rows(manifest, directions)
    write_dict_rows(analysis_dir / "diagnostic_direction_summary.csv", diagnostics)

    save_cm_by_window(concrete_marble, analysis_dir / "concrete_marble_by_window.png")
    save_direction_comparison(separation_rows, analysis_dir / "direction_separation_comparison.png")
    save_session_variation(concrete_marble, analysis_dir / "session_separation_variation.png")
    save_representative_signals(runs, sessions, directions, pulse_start, pulse_end, analysis_dir / "representative_pulse_signals.png")
    save_validity(validity, analysis_dir / "validity_contact_summary.png")

    pca_rows = []
    for window in WINDOW_PROFILES:
        explained = save_pca(window, valid_manifest, extracted, analysis_dir / f"pca_{window}.png")
        pca_rows.append({"window": window, "pc1_explained": explained[0], "pc2_explained": explained[1]})
    write_dict_rows(analysis_dir / "pca_summary.csv", pca_rows)

    candidate_windows = tuple(name for name in WINDOW_PROFILES if name != "full")
    best_window = max(
        candidate_windows,
        key=lambda name: np.mean([
            float(row["separation_ratio"]) for row in concrete_marble
            if row["window"] == name and row["scope"] == "session_direction"
        ]),
    )
    best_values = np.asarray([
        float(row["separation_ratio"]) for row in concrete_marble
        if row["window"] == best_window and row["scope"] == "session_direction"
    ])
    full_values = np.asarray([
        float(row["separation_ratio"]) for row in concrete_marble
        if row["window"] == "full" and row["scope"] == "session_direction"
    ])
    session_improvements = [
        find_cm(concrete_marble, best_window, "session", session, "all")
        > find_cm(concrete_marble, "full", "session", session, "all")
        for session in sessions
    ]
    direction_improvements = [
        find_cm(concrete_marble, best_window, "direction", "all", direction)
        > find_cm(concrete_marble, "full", "direction", "all", direction)
        for direction in directions
    ]
    slip_by_direction = {
        direction: [
            float(next(row for row in diagnostics if row["pulse_direction"] == direction and row["terrain"] == terrain)["pulse_slip_displacement_mean"])
            for terrain in TERRAINS
        ]
        for direction in directions
    }
    slip_rank_correlation = rank_correlation(
        slip_by_direction[directions[0]], slip_by_direction[directions[1]]
    )
    body_collisions = sum(int(row["body_collision"]) for row in manifest)
    force_outliers = sum(int(row["extreme_force_outlier"]) for row in manifest)
    accel_outliers = sum(int(row["extreme_accel_outlier"]) for row in manifest)
    validity_floor = min(int(row["valid_runs"]) / int(row["total_runs"]) for row in validity)
    improved_groups = int(np.count_nonzero(best_values > full_values))
    pair_gate_rows = []
    for first, second in PAIR_NAMES:
        selected_pair = np.asarray([
            float(row["separation_ratio"]) for row in separation_rows
            if row["window"] == best_window and row["scope"] == "session_direction"
            and row["terrain_1"] == first and row["terrain_2"] == second
        ])
        full_pair = np.asarray([
            float(row["separation_ratio"]) for row in separation_rows
            if row["window"] == "full" and row["scope"] == "session_direction"
            and row["terrain_1"] == first and row["terrain_2"] == second
        ])
        selected_pooled = next(
            float(row["separation_ratio"]) for row in separation_rows
            if row["window"] == best_window and row["scope"] == "pooled"
            and row["terrain_1"] == first and row["terrain_2"] == second
        )
        full_pooled = next(
            float(row["separation_ratio"]) for row in separation_rows
            if row["window"] == "full" and row["scope"] == "pooled"
            and row["terrain_1"] == first and row["terrain_2"] == second
        )
        improvement_count = int(np.count_nonzero(selected_pair > full_pair))
        pair_gate_rows.append(
            {
                "terrain_1": first,
                "terrain_2": second,
                "selected_window": best_window,
                "selected_pooled_separation": selected_pooled,
                "full_pooled_separation": full_pooled,
                "improved_session_direction_groups": improvement_count,
                "total_session_direction_groups": len(selected_pair),
                "reproducibly_improved": int(
                    selected_pooled > full_pooled
                    and improvement_count >= len(selected_pair) - 1
                ),
            }
        )
    write_dict_rows(analysis_dir / "pairwise_window_gate.csv", pair_gate_rows)
    all_pairs_reproducibly_improved = all(
        int(row["reproducibly_improved"]) for row in pair_gate_rows
    )
    dataset_ready = (
        all(session_improvements)
        and all(direction_improvements)
        and improved_groups >= len(best_values) - 1
        and all_pairs_reproducibly_improved
        and slip_rank_correlation >= 0.8
        and body_collisions == 0
        and force_outliers == 0
        and accel_outliers == 0
        and validity_floor >= 0.8
    )

    pooled_rows = [row for row in separation_rows if row["scope"] == "pooled"]
    session_seed_text = ", ".join(
        f"{session}="
        f"{next(row['dataset_seed'] for row in manifest if row['session_id'] == session)}"
        for session in sessions
    )
    summary = [
        "Bidirectional pulse validation (no classifier)",
        "",
        f"Protocol: 80N half-sine, start={pulse_start:.2f}s, end={pulse_end:.2f}s, duration={duration:.2f}s, rate={args.sample_rate:g}Hz",
        f"Sessions: {session_seed_text}",
        f"Runs: total={len(manifest)}, valid={len(valid_manifest)}, body_collisions={body_collisions}",
        f"Extreme flags: force={force_outliers}, accel={accel_outliers}",
        "",
        "Window shapes:",
        *(f"- {name}: {shape}" for name, shape in shapes.items()),
        "",
        "Pooled pairwise separation:",
    ]
    for window in WINDOW_PROFILES:
        summary.append(f"- {window}:")
        for row in pooled_rows:
            if row["window"] == window:
                summary.append(f"  {row['terrain_1']}-{row['terrain_2']}: {float(row['separation_ratio']):.6f}")
    summary.extend(
        (
            "",
            "Concrete-marble reproducibility:",
            f"- historical full-window baseline: {args.baseline_concrete_marble:.6f}",
            f"- selected window: {best_window}",
            f"- session/direction mean±std: {best_values.mean():.6f}±{best_values.std():.6f}",
            f"- session/direction min/max: {best_values.min():.6f}/{best_values.max():.6f}",
            f"- matching full-window mean±std: {full_values.mean():.6f}±{full_values.std():.6f}",
            f"- improved session/direction groups: {improved_groups}/{len(best_values)}",
            f"- all sessions improved when directions pooled: {all(session_improvements)}",
            f"- both directions improved when sessions pooled: {all(direction_improvements)}",
            "",
            f"Four-terrain pair gate for {best_window} versus full:",
            *(
                f"- {row['terrain_1']}-{row['terrain_2']}: "
                f"pooled={float(row['selected_pooled_separation']):.6f} vs "
                f"full={float(row['full_pooled_separation']):.6f}, "
                f"improved groups={row['improved_session_direction_groups']}/"
                f"{row['total_session_direction_groups']}, "
                f"pass={bool(row['reproducibly_improved'])}"
                for row in pair_gate_rows
            ),
            "",
            "Direction diagnostics:",
            f"- slip terrain-order rank correlation (+X vs -X): {slip_rank_correlation:.6f}",
            f"- minimum subgroup valid rate: {validity_floor * 100:.1f}%",
            "",
            f"Recommended window: {best_window}",
            f"Input shape: {shapes[best_window]}",
            f"AI dataset ready: {'YES' if dataset_ready else 'NO'}",
            "Gate uses all-pair improvement across sessions/directions, validity, collision/outlier checks, and direction-consistent slip ordering; it is not a classifier metric.",
        )
    )
    (analysis_dir / "bidirectional_validation_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    print(f"analysis_dir={analysis_dir}")


if __name__ == "__main__":
    main()
