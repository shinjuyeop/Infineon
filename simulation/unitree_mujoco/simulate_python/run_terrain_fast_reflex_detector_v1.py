"""Train/evaluate independent Float Host Fast Reflex binary detectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from terrain_fast_reflex_detector_v1 import (
    DETECTORS, PRIMARY_WINDOWS_MS, SEED, TRAIN_ENDPOINT_STRIDE_MS,
    ChannelNormalizer, audit_corrected_v2, balanced_run_weights, binary_metrics,
    build_model, make_windows, replay_rows, resource_estimate, select_threshold,
    subset_traces, threshold_sensitivity, write_csv,
)


DEFAULT_SOURCE = Path("../../outputs/terrain_fast_reflex_v1_full_corrected_v2")
DEFAULT_OUTPUT = Path("../../outputs/terrain_fast_reflex_detector_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--detectors", nargs="+", choices=DETECTORS, default=list(DETECTORS))
    parser.add_argument("--windows-ms", nargs="+", type=int, choices=PRIMARY_WINDOWS_MS,
                        default=list(PRIMARY_WINDOWS_MS))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-runs-per-scenario-split", type=int)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.execute:
        count = len(args.detectors) * len(args.windows_ms)
        print(f"DRY RUN: {count} detector/window fits, epochs<= {args.epochs}")
        print("Add --execute to train. No files were written.")
        return
    source, output = args.source.resolve(), args.output_dir.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    all_traces, audit = audit_corrected_v2(source)
    source_protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    sensitivity = threshold_sensitivity(all_traces, source_protocol["thresholds"])
    write_csv(output / "threshold_sensitivity.csv", sensitivity)
    if args.audit_only:
        (output / "protocol.json").write_text(json.dumps({
            "status": "dataset/label audit and threshold sensitivity complete; no model training",
            "audit": audit, "test_used_for_threshold_sensitivity": False,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"AUDIT COMPLETE runs={audit['runs']} valid={audit['valid_runs']}")
        print(f"OUTPUT_DIR={output}")
        return

    traces = subset_traces(all_traces, args.max_runs_per_scenario_split)
    by_split = {split: [t for t in traces if t.metadata["split"] == split]
                for split in ("train", "validation", "test")}
    run_owners: dict[str, str] = {}
    for split, selected in by_split.items():
        for trace in selected:
            old = run_owners.setdefault(trace.metadata["run_id"], split)
            if old != split:
                raise ValueError(f"run leakage: {trace.metadata['run_id']}")

    import tensorflow as tf
    metrics_rows, latency_rows = [], []
    validation_records: dict[str, list[dict[str, object]]] = {d: [] for d in args.detectors}
    trained: dict[tuple[str, int], dict[str, object]] = {}
    started = time.perf_counter()
    for detector in args.detectors:
        for window_ms in sorted(set(args.windows_ms)):
            job_start = time.perf_counter()
            train_fit = make_windows(by_split["train"], detector, window_ms, TRAIN_ENDPOINT_STRIDE_MS)
            sets = {split: make_windows(selected, detector, window_ms)
                    for split, selected in by_split.items()}
            normalizer = ChannelNormalizer.fit(train_fit.x)
            model = build_model(window_ms, args.seed)
            callbacks = [tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True
            )]
            history = model.fit(
                normalizer.transform(train_fit.x), train_fit.y,
                sample_weight=balanced_run_weights(train_fit),
                validation_data=(normalizer.transform(sets["validation"].x), sets["validation"].y),
                epochs=args.epochs, batch_size=args.batch_size, verbose=2, callbacks=callbacks,
            )
            scores = {split: model.predict(normalizer.transform(data.x), verbose=0).reshape(-1)
                      for split, data in sets.items()}
            threshold = select_threshold(sets["validation"].y, scores["validation"])
            model_dir = output / detector / f"{window_ms}ms"
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(model_dir / "model.keras")
            (model_dir / "normalization.json").write_text(
                json.dumps(normalizer.as_dict(), indent=2) + "\n", encoding="utf-8"
            )
            model_info = {
                "detector": detector, "window_ms": window_ms, "seed": args.seed,
                "epochs_requested": args.epochs, "epochs_completed": len(history.history["loss"]),
                "training_endpoint_stride_ms": TRAIN_ENDPOINT_STRIDE_MS,
                "validation_selected_threshold": threshold, **resource_estimate(window_ms),
            }
            (model_dir / "model_info.json").write_text(
                json.dumps(model_info, indent=2) + "\n", encoding="utf-8"
            )
            for split, data in sets.items():
                row = {"detector": detector, "window_ms": window_ms, "split": split,
                       **binary_metrics(data.y, scores[split], threshold, data.clean_negative)}
                metrics_rows.append(row)
                if split == "validation": validation_records[detector].append(row)
            for split in ("validation", "test"):
                latency_rows.extend(replay_rows(
                    model, normalizer, by_split[split], detector, window_ms, threshold, split
                ))
            trained[(detector, window_ms)] = {"threshold": threshold}
            print(f"JOB_COMPLETE detector={detector} window_ms={window_ms} "
                  f"seconds={time.perf_counter()-job_start:.2f} threshold={threshold:.8g}")

    selections: dict[str, object] = {}
    for detector, records in validation_records.items():
        passing = [row for row in records if row["fpr"] <= 0.05 and row["recall"] >= 0.95]
        chosen = min(passing, key=lambda row: int(row["window_ms"])) if passing else None
        selections[detector] = {
            "rule": "shortest validation window with all-non-target FPR<=0.05 and recall>=0.95",
            "selected_window_ms": None if chosen is None else chosen["window_ms"],
            "validation_metrics": None if chosen is None else chosen,
            "constraint_satisfied": chosen is not None,
        }
    test_report: dict[str, object] = {
        "selection_source": "validation only", "test_threshold_reselection": False,
        "detectors": {},
    }
    for detector, selection in selections.items():
        window = selection["selected_window_ms"]
        selected = None if window is None else next(
            row for row in metrics_rows
            if row["detector"] == detector and row["window_ms"] == window and row["split"] == "test"
        )
        test_report["detectors"][detector] = {"selected_window_ms": window, "test_metrics": selected}

    write_csv(output / "window_metrics.csv", metrics_rows)
    write_csv(output / "window_latency.csv", latency_rows)
    (output / "validation_selection.json").write_text(
        json.dumps(selections, indent=2) + "\n", encoding="utf-8"
    )
    (output / "test_final_report.json").write_text(
        json.dumps(test_report, indent=2) + "\n", encoding="utf-8"
    )
    protocol = {
        "status": "Float Host detector training/evaluation pipeline output; no INT8/E84 artifact",
        "source": str(source), "source_must_be_corrected_v2": True, "audit": audit,
        "ai_input": "Fusion10 only; oracle used only for endpoint labels/evaluation",
        "primary_task": "current physical hazard detection, not future-hazard anticipation",
        "canonical_negative": "all endpoints without detector target are negative, including other hazards",
        "clean_negative_only": "diagnostic FPR only; not used to fit/select",
        "window_semantics": "sliding [t-L+1,t], endpoints t=0..99 ms, oracle state at t",
        "training_sampling": "2 ms endpoint stride, run-balanced then binary-class-balanced sample weights",
        "evaluation_sampling": "every 1 ms endpoint",
        "normalization": "per detector/window, fit only on train split windows",
        "architecture": "Conv1D(12,k5,same,relu)->Conv1D(16,k3,same,relu)->GAP->Dense(1,sigmoid)",
        "operating_threshold": "validation maximum recall subject to all-non-target FPR<=5%; no test retuning",
        "persistence_samples": 3,
        "latency_semantics": "transition and post-onset latency separate; pre-onset firing is anticipation",
        "split_ownership": {"train": ["multisine", "filtered_random", "sparse_aggregate"],
                            "validation": ["crosshatch", "rounded_ridges"],
                            "test": ["warped_multisine", "smooth_random_patches"]},
        "windows_ms": sorted(set(args.windows_ms)), "detectors": args.detectors,
        "seed": args.seed, "subset_limit_per_scenario_split": args.max_runs_per_scenario_split,
        "total_runtime_s": time.perf_counter() - started,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(f"FAST_REFLEX_DETECTOR_COMPLETE jobs={len(trained)} runtime_s={time.perf_counter()-started:.2f}")
    print(f"OUTPUT_DIR={output}")


if __name__ == "__main__":
    main()
