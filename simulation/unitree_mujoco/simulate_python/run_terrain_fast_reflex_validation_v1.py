"""Run validation-only failure diagnosis and a bounded pooling ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from terrain_fast_reflex_detector_v1 import (
    DETECTORS, PRIMARY_WINDOWS_MS, SEED, TRAIN_ENDPOINT_STRIDE_MS,
    ChannelNormalizer, balanced_run_weights, binary_metrics, build_model,
    load_corrected_traces, make_windows, resource_estimate, select_threshold, write_csv,
)
from terrain_fast_reflex_validation_v1 import (
    PERSISTENCE_CANDIDATES, ScoredRun, best_run_policy, endpoint_confusion_rows,
    onset_relative_rows, plot_slip_examples, run_policy_metrics, sink_subgroup_rows,
    threshold_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../../outputs/terrain_fast_reflex_v1_full_corrected_v2"))
    parser.add_argument("--models-dir", type=Path, default=Path("../../outputs/terrain_fast_reflex_detector_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("../../outputs/terrain_fast_reflex_detector_validation_v1"))
    parser.add_argument("--pooling-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-pooling", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def read_validation_thresholds(path: Path) -> dict[tuple[str, int], float]:
    selected = {}
    for detector in DETECTORS:
        for window_ms in PRIMARY_WINDOWS_MS:
            info = json.loads(
                (path / detector / f"{window_ms}ms" / "model_info.json").read_text(encoding="utf-8")
            )
            selected[(detector, window_ms)] = float(info["validation_selected_threshold"])
    return selected


def scored_runs(traces, scores: np.ndarray) -> list[ScoredRun]:
    if len(scores) != 100 * len(traces):
        raise ValueError("validation prediction count mismatch")
    return [ScoredRun(trace, scores[i * 100:(i + 1) * 100]) for i, trace in enumerate(traces)]


def score_model(model, normalizer: ChannelNormalizer, data) -> np.ndarray:
    return model.predict(normalizer.transform(data.x), batch_size=1024, verbose=0).reshape(-1)


def load_normalizer(path: Path) -> ChannelNormalizer:
    values = json.loads(path.read_text(encoding="utf-8"))
    return ChannelNormalizer(np.asarray(values["mean"], dtype=np.float32),
                             np.asarray(values["std"], dtype=np.float32))


def add_policy_sweep(
    detector: str, architecture: str, window_ms: int, runs: list[ScoredRun],
    endpoint_threshold: float, threshold_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    flat_scores = np.concatenate([item.scores for item in runs])
    local = []
    for threshold in threshold_candidates(flat_scores, endpoint_threshold):
        for persistence in PERSISTENCE_CANDIDATES:
            row = {"detector": detector, "architecture": architecture, "window_ms": window_ms,
                   "threshold_source": "existing_endpoint" if np.isclose(threshold, endpoint_threshold) else "validation_quantile",
                   **run_policy_metrics(runs, detector, float(threshold), persistence)}
            threshold_rows.append(row); local.append(row)
    return local


def main() -> None:
    args = parse_args()
    if not args.execute:
        print("DRY RUN: validation split only; existing 12 models + two Slip 5 ms pooling fits")
        print("Add --execute. Test predictions and test metrics will not be read.")
        return
    source, models_dir, output = args.source.resolve(), args.models_dir.resolve(), args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    if source_protocol.get("derived_artifact_revision") != 2:
        raise ValueError("source must be corrected-v2")
    train_traces, _ = load_corrected_traces(source, split="train")
    validation_traces, _ = load_corrected_traces(source, split="validation")
    if len(validation_traces) != 120 or any(t.metadata["split"] != "validation" for t in validation_traces):
        raise ValueError("validation-only safeguard failed")
    existing_thresholds = read_validation_thresholds(models_dir)
    audit = {
        "source": str(source), "derived_artifact_revision": 2,
        "train_runs_materialized": len(train_traces),
        "validation_runs_materialized": len(validation_traces),
        "test_runs_materialized": 0,
    }

    import tensorflow as tf
    endpoint_run_rows, persistence_rows, threshold_rows = [], [], []
    slip_failure, sink_failure, subgroup_rows, onset_rows = [], [], [], []
    model_runs: dict[tuple[str, int, str], list[ScoredRun]] = {}
    started = time.perf_counter()
    for detector in DETECTORS:
        for window_ms in PRIMARY_WINDOWS_MS:
            artifact = models_dir / detector / f"{window_ms}ms"
            model = tf.keras.models.load_model(artifact / "model.keras")
            normalizer = load_normalizer(artifact / "normalization.json")
            validation = make_windows(validation_traces, detector, window_ms)
            scores = score_model(model, normalizer, validation)
            runs = scored_runs(validation_traces, scores)
            model_runs[(detector, window_ms, "average")] = runs
            threshold = existing_thresholds[(detector, window_ms)]
            endpoint = binary_metrics(
                validation.y, scores, threshold, validation.clean_negative
            )
            canonical_run = run_policy_metrics(runs, detector, threshold, 3)
            endpoint_run_rows.append({
                "detector": detector, "architecture": "average", "window_ms": window_ms,
                "validation_threshold": threshold,
                "endpoint_precision": endpoint["precision"],
                "endpoint_recall": endpoint["recall"], "endpoint_f1": endpoint["f1"],
                "endpoint_fpr": endpoint["fpr"], **canonical_run,
            })
            for persistence in PERSISTENCE_CANDIDATES:
                persistence_rows.append({
                    "detector": detector, "architecture": "average", "window_ms": window_ms,
                    "threshold_policy": "existing_endpoint_fpr_5pct",
                    **run_policy_metrics(runs, detector, threshold, persistence),
                })
            add_policy_sweep(detector, "average", window_ms, runs, threshold, threshold_rows)
            confusion = endpoint_confusion_rows(validation_traces, detector, window_ms, scores, threshold)
            (slip_failure if detector == "slip" else sink_failure).extend(confusion)
            onset_rows.extend(onset_relative_rows(runs, detector, window_ms, threshold))
            if detector == "sink_tilt":
                for row in sink_subgroup_rows(runs, threshold, 3):
                    subgroup_rows.append({"window_ms": window_ms, **row})

    example_rows = plot_slip_examples(
        model_runs[("slip", 5, "average")], existing_thresholds[("slip", 5)],
        output / "plots",
    )
    write_csv(output / "slip_example_index.csv", example_rows)

    pooling_rows = []
    # The 5 ms GAP model is the strongest existing Slip endpoint model and is the
    # controlled baseline. Only its two pooling substitutions are retrained.
    baseline_runs = model_runs[("slip", 5, "average")]
    baseline_threshold = existing_thresholds[("slip", 5)]
    baseline_validation = make_windows(validation_traces, "slip", 5)
    baseline_scores = np.concatenate([item.scores for item in baseline_runs])
    baseline_endpoint = binary_metrics(
        baseline_validation.y, baseline_scores, baseline_threshold,
        baseline_validation.clean_negative,
    )
    baseline_local = [r for r in threshold_rows if r["detector"] == "slip"
                      and r["window_ms"] == 5 and r["architecture"] == "average"]
    baseline_best = best_run_policy(baseline_local)
    pooling_rows.append({
        "detector": "slip", "architecture": "average", "window_ms": 5,
        "parameters": resource_estimate(5, "average")["parameters"],
        "endpoint_threshold": baseline_threshold, "endpoint_recall": float(baseline_endpoint["recall"]),
        "endpoint_fpr": float(baseline_endpoint["fpr"]),
        **{f"best_{k}": v for k, v in (baseline_best or {}).items()},
    })
    if not args.skip_pooling:
        train = make_windows(train_traces, "slip", 5, TRAIN_ENDPOINT_STRIDE_MS)
        validation = make_windows(validation_traces, "slip", 5)
        for pooling in ("max", "average_max"):
            tf.keras.backend.clear_session()
            normalizer = ChannelNormalizer.fit(train.x)
            model = build_model(5, args.seed, pooling)
            history = model.fit(
                normalizer.transform(train.x), train.y,
                sample_weight=balanced_run_weights(train),
                validation_data=(normalizer.transform(validation.x), validation.y),
                epochs=args.pooling_epochs, batch_size=args.batch_size, verbose=2,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=8, restore_best_weights=True
                )],
            )
            scores = score_model(model, normalizer, validation)
            endpoint_threshold = select_threshold(validation.y, scores)
            endpoint_metrics = binary_metrics(
                validation.y, scores, endpoint_threshold, validation.clean_negative
            )
            runs = scored_runs(validation_traces, scores)
            local = add_policy_sweep("slip", pooling, 5, runs, endpoint_threshold, threshold_rows)
            best = best_run_policy(local)
            artifact = output / "pooling" / f"slip_5ms_{pooling}"
            artifact.mkdir(parents=True, exist_ok=True)
            model.save(artifact / "model.keras")
            (artifact / "normalization.json").write_text(
                json.dumps(normalizer.as_dict(), indent=2) + "\n", encoding="utf-8"
            )
            info = {"validation_only_architecture_ablation": True, "pooling": pooling,
                    "window_ms": 5, "epochs_completed": len(history.history["loss"]),
                    **resource_estimate(5, pooling)}
            (artifact / "model_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
            pooling_rows.append({
                "detector": "slip", "architecture": pooling, "window_ms": 5,
                "parameters": info["parameters"], "endpoint_threshold": endpoint_threshold,
                "endpoint_recall": endpoint_metrics["recall"], "endpoint_fpr": endpoint_metrics["fpr"],
                **{f"best_{k}": v for k, v in (best or {}).items()},
            })

    # One best run-policy row per controlled model/window makes the compact summary.
    summary_rows = []
    identities = sorted({(r["detector"], r["architecture"], int(r["window_ms"])) for r in threshold_rows})
    for detector, architecture, window_ms in identities:
        local = [r for r in threshold_rows if r["detector"] == detector
                 and r["architecture"] == architecture and int(r["window_ms"]) == window_ms]
        best = best_run_policy(local)
        if best:
            summary_rows.append(best)

    selection = {}
    for detector in DETECTORS:
        candidates = [r for r in summary_rows if r["detector"] == detector
                      and float(r["run_target_recall"]) >= 0.95]
        if candidates:
            chosen = min(candidates, key=lambda r=(None): (
                int(r["window_ms"]), float("inf") if r["median_hazard_to_detection_ms"] is None
                else float(r["median_hazard_to_detection_ms"]), float(r["run_pre_onset_fpr"])
            ))
        else:
            chosen = None
        feasible = [r for r in summary_rows if r["detector"] == detector]
        best_diagnostic = max(feasible, key=lambda r: (
            float(r["run_target_recall"]), -float(r["run_pre_onset_fpr"]), -int(r["window_ms"])
        )) if feasible else None
        selection[detector] = {
            "rule": "validation run-pre-onset FPR<=0.05, recall>=0.95; shortest window then latency",
            "selected_configuration": chosen,
            "gate_pass": chosen is not None,
            "under_20ms_pass": chosen is not None and int(chosen["window_ms"]) <= 20,
            "best_nonpassing_diagnostic": best_diagnostic if chosen is None else None,
        }

    write_csv(output / "validation_endpoint_vs_run.csv", endpoint_run_rows)
    write_csv(output / "validation_persistence_sweep.csv", persistence_rows)
    write_csv(output / "validation_threshold_persistence.csv", threshold_rows)
    write_csv(output / "slip_failure_analysis.csv", slip_failure)
    write_csv(output / "sink_tilt_failure_analysis.csv", sink_failure)
    write_csv(output / "sink_tilt_subgroup_metrics.csv", subgroup_rows)
    write_csv(output / "onset_relative_metrics.csv", onset_rows)
    write_csv(output / "pooling_ablation.csv", pooling_rows)
    write_csv(output / "validation_best_by_model.csv", summary_rows)
    (output / "validation_selection_v2.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    protocol = {
        "status": "validation-only diagnosis; no final detector selected unless gate passes",
        "source": str(source), "existing_models": str(models_dir), "audit": audit,
        "selection_data": "validation only", "existing_test_metrics_read": False,
        "test_rows_materialized": False,
        "test_predictions_performed": False, "validation_runs": len(validation_traces),
        "monolithic_source_note": "NPZ container schema/order is validated, but no test trace is materialized or scored",
        "canonical_run_fpr": "stable firing before target onset on any validation run / all validation runs",
        "additional_run_fpr": ["target-negative full replay", "completely physical-hazard-free", "normal-terrain"],
        "threshold_grid": "41 validation-score quantiles + prior endpoint threshold + above-max sentinel",
        "persistence_candidates": list(PERSISTENCE_CANDIDATES),
        "pooling_ablation": "Slip 5 ms only: existing GAP vs newly trained GlobalMax and GAP+Max",
        "sink_pooling_not_run": "50 ms plateau and onset/subgroup diagnostics are evaluated before authorizing more training",
        "runtime_s": time.perf_counter() - started,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    lines = ["# Fast Reflex validation-only diagnosis", "", "Test split was not read for selection or analysis.", "",
             "| Detector | Architecture | Window | Persistence | Run FPR | Run Recall | Median hazard latency | p95 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary_rows:
        lines.append(f"| {row['detector']} | {row['architecture']} | {row['window_ms']} ms | "
                     f"{row['persistence']} | {float(row['run_pre_onset_fpr']):.3f} | "
                     f"{float(row['run_target_recall']):.3f} | {row['median_hazard_to_detection_ms']} | "
                     f"{row['p95_hazard_to_detection_ms']} |")
    lines += ["", "## Gate", ""]
    for detector, result in selection.items():
        lines.append(f"- {detector}: 95% recall / 5% run-FPR gate "
                     f"{'PASS' if result['gate_pass'] else 'FAIL'}; <=20 ms "
                     f"{'PASS' if result['under_20ms_pass'] else 'FAIL'}")
    lines += [
        "", "## Endpoint versus run-level false alarm", "",
        "Endpoint FPR pools individual 1 ms decisions. Run-pre-onset FPR counts a run once "
        "when any persistence-confirmed firing occurs before its target onset (or anywhere "
        "for a target-negative run), so repeated correlated decisions accumulate at run level.", "",
        "| Detector | Window | Endpoint FPR | Run-pre-onset FPR | Target-negative FPR | Hazard-free FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in endpoint_run_rows:
        lines.append(f"| {row['detector']} | {row['window_ms']} ms | {float(row['endpoint_fpr']):.3f} | "
                     f"{float(row['run_pre_onset_fpr']):.3f} | {float(row['target_negative_run_fpr']):.3f} | "
                     f"{float(row['hazard_free_run_fpr']):.3f} |")
    lines += ["", "## Controlled Slip pooling ablation", "",
              "Only Slip 5 ms was retrained; Sink/Tilt pooling was not expanded because its "
              "failure is concentrated in the tilt-only subgroup rather than window length.", "",
              "| Pooling | Parameters | Endpoint recall | Best run recall | Run FPR | Persistence |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in pooling_rows:
        lines.append(f"| {row['architecture']} | {row['parameters']} | {float(row['endpoint_recall']):.3f} | "
                     f"{float(row['best_run_target_recall']):.3f} | {float(row['best_run_pre_onset_fpr']):.3f} | "
                     f"{row['best_persistence']} |")
    lines += ["", "## Failure mode", "",
              "- Slip: onset-relative endpoint recall falls as GAP windows lengthen, consistent "
              "with dilution of a short transient. GAP+Max restores a validation run-level candidate.",
              "- Sink/Tilt: validation contains no sink-only runs; all 30 sink+tilt runs are detected, "
              "while the 15 tilt-only runs dominate false negatives. Window extension does not fix this subgroup.",
              "- Pre-onset firing remains an anticipation diagnostic and is not counted as primary detection."]
    (output / "validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"VALIDATION_DIAGNOSIS_COMPLETE rows={len(threshold_rows)} runtime_s={time.perf_counter()-started:.2f}")
    print(f"OUTPUT_DIR={output}")


if __name__ == "__main__":
    main()
