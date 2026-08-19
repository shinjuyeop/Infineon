"""One-shot, realization-disjoint fresh transition test for the v4 INT8 model."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_terrain_causal_window_v4 import (
    LABEL, OUT, STATIC, T0, ChannelNormalizer, aggregate_transition, load_transition,
    stable, transition_split,
)
from terrain_int8 import predict_tflite


FRESH_FAMILY = "filtered_random"
FRESH_SURFACE_REALIZATION = 8
FRESH_RUN_INDICES = (4, 5, 6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT / "fresh_test")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("fresh test is one-shot; output already exists")
    int8_summary = json.loads((OUT / "int8/summary.json").read_text())
    if not int8_summary["TERRAIN_CAUSAL_WINDOW_INT8_READY"]:
        raise RuntimeError("fresh test requires both strict-INT8 validation gates")
    selected = int8_summary["selected_float"]
    length = int(selected["window_ms"])
    int8_path = Path(int8_summary["manifest"]["int8_path"])
    from run_terrain_transition import CASES, audit, label, run_one

    with np.load(STATIC) as values:
        static_x, static_split = values["X"], values["split"]
    source_rows, _, transition_x, _, transition_split_values = load_transition(length)
    x = np.concatenate((static_x[:, -length:, :], transition_x))
    split = np.concatenate((static_split, transition_split_values))
    normalizer = ChannelNormalizer.fit(x[split == "train"])
    existing_ids = {row["run_id"] for row in source_rows}
    runs = [run_one(case, run_index, FRESH_FAMILY, FRESH_SURFACE_REALIZATION) for case in CASES for run_index in FRESH_RUN_INDICES]
    if any(run.metadata["run_id"] in existing_ids for run in runs):
        raise RuntimeError("fresh transition source-run overlaps prior corpus")
    labels = [label(run.oracle, int(run.metadata["transition_sample"])) for run in runs]
    audit_rows, physical_audit = audit(runs, labels)
    if not all(row["valid"] for row in audit_rows):
        raise RuntimeError("fresh physical corpus audit failed")
    records = []
    for run in runs:
        trace, row = run.fusion10, run.metadata
        endpoints = np.arange(length - 1, len(trace))
        windows = np.asarray([trace[end - length + 1 : end + 1] for end in endpoints], np.float32)
        prediction = np.full(len(trace), -1, np.int8)
        prediction[length - 1 :] = np.argmax(predict_tflite(int8_path, normalizer.transform(windows)), axis=1)
        target, source = LABEL[row["terrain_after"]], LABEL[row["terrain_before"]]
        t1 = stable(prediction, target)
        post = prediction[T0:]
        records.append({
            "run_id": row["run_id"], "case_id": row["case_id"], "detected": t1 is not None,
            "t1_ms": None if t1 is None else t1 - T0, "occupancy": float(np.mean(post == target)),
            "switches": int(np.count_nonzero(np.diff(post) != 0)),
            "pre_source_accuracy": float(np.mean(prediction[550:T0] == source)),
            "post_target_accuracy": float(np.mean(post == target)),
        })
    transition = aggregate_transition(records)
    gates = {case: bool(values["stable_detection_rate"] >= .90 and values["target_occupancy"] >= .80) for case, values in transition.items()}
    out.mkdir(parents=True)
    np.savez_compressed(out / "fresh_transition_runs.npz", fusion10=np.asarray([run.fusion10 for run in runs], np.float32), oracle=np.asarray([run.oracle for run in runs], np.float32), terrain_gt=np.asarray([run.terrain_gt for run in runs]), run_id=np.asarray([run.metadata["run_id"] for run in runs]))
    with (out / "fresh_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0])); writer.writeheader(); writer.writerows(audit_rows)
    summary = {
        "sealed_after_run": True,
        "model": int8_summary["manifest"],
        "reservation": {"family": FRESH_FAMILY, "surface_realization": FRESH_SURFACE_REALIZATION, "run_indices": FRESH_RUN_INDICES, "source_run_overlap": False, "surface_realization_overlap_with_v4_train_or_selection": False},
        "physical_audit": physical_audit, "transition": transition, "transition_gates": gates,
        "FRESH_TRANSITION_TEST_PASS": bool(all(gates.values())),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
