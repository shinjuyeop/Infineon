"""Audit preserved Slip v1 anticipation and Sand v2 temporal-feature feasibility.

This is a read-only planning diagnostic: it never loads a neural model, generates
new simulation data, or evaluates a new test policy.  The Slip portion reads the
already-recorded v1 frozen-test stable-firing timestamps; the Sand portion indexes
only train/validation rows for a compact temporal persistence check.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


TRACE_PRE_MS = 50
ORACLE = {
    "contact": 0,
    "fn": 1,
    "ft": 2,
    "ft_over_fn": 3,
    "foot_velocity_x": 4,
    "foot_velocity_y": 5,
    "foot_velocity_z": 6,
    "horizontal_speed": 13,
}
EXPECTED_TEST_FAMILIES = {"warped_multisine", "smooth_random_patches"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("../../outputs/terrain_fast_reflex_v1_full_corrected_v2"))
    parser.add_argument("--slip-final", type=Path,
                        default=Path("../../outputs/terrain_fast_reflex_slip_final_test_v1"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("../../outputs/terrain_fast_reflex_v2_planning_diagnostic"))
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical_onset(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask[TRACE_PRE_MS:])
    return None if not len(indices) else int(indices[0])


def classify_pre_onset_firing(
    stable_firing_ms: int, slip_onset_ms: int | None, valid_trace: bool
) -> str:
    """Classify by already-existing outcome, without changing the slip label.

    A pre-onset stable firing followed by the preserved canonical onset is an
    incipient-slip *candidate*, not a relabelled canonical positive.  Full valid
    traces without any later canonical onset remain true false alarms.
    """
    if not valid_trace or stable_firing_ms < 0:
        return "ambiguous"
    if slip_onset_ms is None:
        return "true_false_alarm"
    if stable_firing_ms < slip_onset_ms:
        return "incipient_slip_candidate"
    return "ambiguous"


def _frame_values(oracle: np.ndarray, sensors: np.ndarray, endpoint_ms: int) -> dict[str, float]:
    index = TRACE_PRE_MS + int(np.clip(endpoint_ms, 0, 99))
    frame, sensor = oracle[index], sensors[index]
    return {
        "horizontal_speed_mps": float(frame[ORACLE["horizontal_speed"]]),
        "fn_N": float(frame[ORACLE["fn"]]),
        "ft_N": float(frame[ORACLE["ft"]]),
        "ft_over_fn": float(frame[ORACLE["ft_over_fn"]]),
        "contact": int(frame[ORACLE["contact"]] > 0.5),
        "foot_velocity_x_mps": float(frame[ORACLE["foot_velocity_x"]]),
        "foot_velocity_y_mps": float(frame[ORACLE["foot_velocity_y"]]),
        "foot_velocity_z_mps": float(frame[ORACLE["foot_velocity_z"]]),
        "fsr_sum_N": float(np.sum(sensor[:4])),
        "accel_magnitude_mps2": float(np.linalg.norm(sensor[4:7])),
        "gyro_magnitude_rad_s": float(np.linalg.norm(sensor[7:10])),
    }


def audit_pre_onset_firings(
    metrics_rows: list[dict[str, str]], manifest_by_id: dict[str, dict[str, str]],
    sensors: np.ndarray, oracle: np.ndarray, slip: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in metrics_rows:
        if metric["pre_onset_false_alarm"] != "1":
            continue
        run_id = metric["run_id"]
        metadata = manifest_by_id[run_id]
        if metadata["split"] != "test" or metadata["surface_family"] not in EXPECTED_TEST_FAMILIES:
            raise ValueError(f"non-v1-test firing row: {run_id}")
        index = int(metadata["_index"])
        firing = int(metric["first_stable_detection_ms"])
        onset = canonical_onset(slip[index])
        label = classify_pre_onset_firing(firing, onset, bool(int(metadata["valid"])))
        reference = onset if onset is not None else min(99, firing + 20)
        at_firing = _frame_values(oracle[index], sensors[index], firing)
        at_reference = _frame_values(oracle[index], sensors[index], reference)
        row: dict[str, object] = {
            "run_id": run_id,
            "scenario": metadata["scenario"],
            "family": metadata["surface_family"],
            "surface": metadata["surface_index"],
            "run": metadata["run_index"],
            "transition_time_ms": 0,
            "stable_firing_time_ms": firing,
            "canonical_slip_onset_ms": "" if onset is None else onset,
            "lead_time_ms": "" if onset is None else onset - firing,
            "classification": label,
            "reference_time_ms": reference,
        }
        for key, value in at_firing.items():
            row[f"at_firing_{key}"] = value
        for key, value in at_reference.items():
            row[f"at_reference_{key}"] = value
        for key in ("horizontal_speed_mps", "fn_N", "ft_N", "ft_over_fn", "fsr_sum_N",
                    "accel_magnitude_mps2", "gyro_magnitude_rad_s"):
            row[f"delta_{key}"] = at_reference[key] - at_firing[key]
        rows.append(row)
    return rows


def sand_temporal_rows(
    manifest: list[dict[str, str]], sensors: np.ndarray, sink: np.ndarray, tilt: np.ndarray,
) -> list[dict[str, object]]:
    """Small validation-only check of persistence/direction, not a detector search."""
    rows: list[dict[str, object]] = []
    for index, metadata in enumerate(manifest):
        if metadata["split"] != "validation":
            continue
        if metadata["surface_family"] not in {"crosshatch", "rounded_ridges"}:
            raise ValueError("Sand diagnostic encountered non-validation family")
        if not np.any(tilt[index]) or np.any(sink[index]):
            continue
        onset = canonical_onset(tilt[index])
        assert onset is not None
        fsr = sensors[index, TRACE_PRE_MS:, :4]
        total = fsr.sum(axis=1)
        front_rear = (fsr[:, 2:].sum(axis=1) - fsr[:, :2].sum(axis=1)) / np.maximum(total, 1e-6)
        gyro = sensors[index, TRACE_PRE_MS:, 7:9]
        pre = max(0, onset - 5)
        post5, post10 = min(99, onset + 5), min(99, onset + 10)
        onset_change = float(front_rear[onset] - front_rear[pre])
        post5_change = float(front_rear[post5] - front_rear[onset])
        post10_change = float(front_rear[post10] - front_rear[post5])
        gx_on, gy_on = gyro[onset]
        gx_mean, gy_mean = gyro[onset:post5 + 1].mean(axis=0)
        rows.append({
            "run_id": metadata["run_id"], "family": metadata["surface_family"],
            "tilt_onset_ms": onset,
            "front_rear_change_pre5_to_onset": onset_change,
            "front_rear_change_onset_to_post5": post5_change,
            "front_rear_change_post5_to_post10": post10_change,
            "front_rear_direction_consistent_5ms": int(
                onset_change != 0.0 and np.sign(onset_change) == np.sign(post5_change)),
            "front_rear_direction_consistent_10ms": int(
                post5_change != 0.0 and np.sign(post5_change) == np.sign(post10_change)),
            "gyro_x_onset": float(gx_on), "gyro_y_onset": float(gy_on),
            "gyro_xy_magnitude_onset": float(np.linalg.norm(gyro[onset])),
            "gyro_xy_integral_0_to_5ms": float(np.linalg.norm(gyro[onset:post5 + 1], axis=1).sum() * .001),
            "gyro_x_direction_consistent_5ms": int(abs(gx_on) > 1e-5 and np.sign(gx_on) == np.sign(gx_mean)),
            "gyro_y_direction_consistent_5ms": int(abs(gy_on) > 1e-5 and np.sign(gy_on) == np.sign(gy_mean)),
        })
    return rows


def _plot_slip_rows(output: Path, audit: list[dict[str, object]], manifest_by_id: dict[str, dict[str, str]],
                    sensors: np.ndarray, oracle: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    for row in audit:
        index = int(manifest_by_id[str(row["run_id"])]["_index"])
        firing, onset = int(row["stable_firing_time_ms"]), row["canonical_slip_onset_ms"]
        end = min(99, (int(onset) if onset != "" else firing + 20) + 10)
        start = max(0, firing - 20)
        endpoints = np.arange(start, end + 1)
        samples = endpoints + TRACE_PRE_MS
        fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(endpoints, sensors[index, samples, :4]); axes[0].set_ylabel("FSR1..4 [N]")
        axes[1].plot(endpoints, sensors[index, samples, 4:7]); axes[1].set_ylabel("accel XYZ")
        axes[2].plot(endpoints, sensors[index, samples, 7:10]); axes[2].set_ylabel("gyro XYZ")
        axes[3].plot(endpoints, oracle[index, samples, ORACLE["horizontal_speed"]], label="horizontal speed")
        axes[3].plot(endpoints, oracle[index, samples, ORACLE["fn"]] / 100.0, label="Fn / 100")
        axes[3].plot(endpoints, oracle[index, samples, ORACLE["ft"]] / 100.0, label="Ft / 100")
        axes[3].plot(endpoints, oracle[index, samples, ORACLE["ft_over_fn"]], label="Ft/Fn")
        axes[3].set_ylabel("oracle"); axes[3].legend(ncol=4, fontsize=8)
        for axis in axes:
            axis.axvline(firing, color="tab:orange", linestyle="--", label="stable firing")
            if onset != "":
                axis.axvline(int(onset), color="tab:red", linestyle=":", label="canonical onset")
        axes[-1].set_xlabel("ms after transition")
        fig.suptitle(f"{row['classification']}: {row['run_id']}")
        fig.tight_layout()
        fig.savefig(output / f"{row['run_id']}.png", dpi=140)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.execute:
        print("DRY RUN: v2 planning diagnostic only; no model inference or dataset generation. Add --execute.")
        return
    source, slip_final, output = args.source.resolve(), args.slip_final.resolve(), args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    frozen = json.loads((slip_final / "frozen_config.json").read_text(encoding="utf-8"))
    if protocol.get("derived_artifact_revision") != 2 or frozen.get("threshold") != 0.7317748725:
        raise ValueError("expected preserved corrected-v2 data and frozen Slip v1 configuration")
    with (source / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    manifest_by_id = {row["run_id"]: {**row, "_index": str(index)} for index, row in enumerate(manifest)}
    with (slip_final / "test_run_metrics.csv").open(newline="", encoding="utf-8") as stream:
        metrics = list(csv.DictReader(stream))
    with np.load(source / "inputs_fusion10.npz", allow_pickle=False) as data:
        sensors = data["sensors"]
    with np.load(source / "oracle_diagnostics.npz", allow_pickle=False) as data:
        oracle, slip, sink, tilt = data["oracle"], data["slip"], data["sink"], data["tilt"]
    audit = audit_pre_onset_firings(metrics, manifest_by_id, sensors, oracle, slip)
    if len(audit) != 9:
        raise ValueError(f"expected the preserved nine pre-onset firing runs, got {len(audit)}")
    sand_rows = sand_temporal_rows(manifest, sensors, sink, tilt)
    if len(sand_rows) != 15:
        raise ValueError(f"expected 15 validation Tilt-only rows, got {len(sand_rows)}")
    output.mkdir(parents=True)
    _plot_slip_rows(output / "slip_pre_onset_plots", audit, manifest_by_id, sensors, oracle)
    write_csv(output / "slip_pre_onset_audit.csv", audit)
    write_csv(output / "sand_tilt_temporal_persistence.csv", sand_rows)
    classes = {name: sum(row["classification"] == name for row in audit)
               for name in ("true_false_alarm", "incipient_slip_candidate", "ambiguous")}
    incipient = [row for row in audit if row["classification"] == "incipient_slip_candidate"]
    summary = {
        "scope": "read-only v2 planning diagnostic; no detector inference, training, test evaluation, or dataset generation",
        "slip_pre_onset_firing_runs": len(audit), "slip_classification_counts": classes,
        "incipient_lead_ms_median": float(np.median([row["lead_time_ms"] for row in incipient])),
        "incipient_lead_ms_p95": float(np.percentile([row["lead_time_ms"] for row in incipient], 95)),
        "incipient_horizontal_speed_at_firing_median_mps": float(np.median([row["at_firing_horizontal_speed_mps"] for row in incipient])),
        "incipient_horizontal_speed_at_onset_median_mps": float(np.median([row["at_reference_horizontal_speed_mps"] for row in incipient])),
        "sand_validation_only": True, "sand_test_rows_indexed": 0,
        "sand_tilt_only_rows": len(sand_rows),
        "sand_front_rear_direction_consistent_5ms": sum(row["front_rear_direction_consistent_5ms"] for row in sand_rows),
        "sand_front_rear_direction_consistent_10ms": sum(row["front_rear_direction_consistent_10ms"] for row in sand_rows),
        "sand_gyro_x_direction_consistent_5ms": sum(row["gyro_x_direction_consistent_5ms"] for row in sand_rows),
        "sand_gyro_y_direction_consistent_5ms": sum(row["gyro_y_direction_consistent_5ms"] for row in sand_rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "summary.md").write_text(
        "# Fast Reflex v2 planning diagnostic\n\n"
        f"- Preserved Slip v1 pre-onset firing runs: {len(audit)}\n"
        f"- Classification: {classes}\n"
        f"- Incipient-candidate lead median/p95: {summary['incipient_lead_ms_median']:.1f}/{summary['incipient_lead_ms_p95']:.1f} ms\n"
        f"- Tilt-only validation front/rear direction consistency: {summary['sand_front_rear_direction_consistent_5ms']}/15 at 5 ms; {summary['sand_front_rear_direction_consistent_10ms']}/15 at 10 ms\n"
        f"- Tilt-only validation raw gyro direction consistency: gyro_x {summary['sand_gyro_x_direction_consistent_5ms']}/15; gyro_y {summary['sand_gyro_y_direction_consistent_5ms']}/15 over 5 ms\n\n"
        "This diagnostic did not execute a detector or use a new test evaluation.\n",
        encoding="utf-8",
    )
    print(f"FAST_REFLEX_V2_DIAGNOSTIC_COMPLETE output={output}")


if __name__ == "__main__":
    main()
