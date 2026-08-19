"""Strict-INT8 gate for the single float-selected Terrain v4 candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run_terrain_causal_window_v4 import (
    OUT, REFERENCE, STATIC, T0, ChannelNormalizer, LABEL, SIM, aggregate_transition,
    class_recalls, load_transition, macro_f1, stable, transition_gates, transition_split,
)
from terrain_int8 import export_full_int8_tflite, predict_tflite


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tflite_transition_metrics(path: Path, normalizer, traces, rows, length: int):
    records = []
    for trace, row in zip(traces, rows):
        endpoints = np.arange(length - 1, len(trace))
        windows = np.asarray([trace[end - length + 1 : end + 1] for end in endpoints], np.float32)
        prediction = np.full(len(trace), -1, np.int8)
        prediction[length - 1 :] = np.argmax(predict_tflite(path, normalizer.transform(windows)), axis=1)
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
    return aggregate_transition(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT / "int8")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    selection = json.loads((OUT / "architecture_selection.json").read_text())
    selected = selection["selected_float"]
    if selected is None or not selected["eligible_float"]:
        raise RuntimeError("no float candidate passed both v4 selection gates")
    import tensorflow as tf

    length = int(selected["window_ms"])
    float_path = Path(selected["path"])
    if sha(float_path) != selected["sha256"]:
        raise RuntimeError("selected float artifact SHA256 changed")
    with np.load(STATIC) as values:
        static_x, static_y, static_split = values["X"], values["y"], values["split"]
    rows, traces, transition_x, transition_y, transition_split_values = load_transition(length)
    x = np.concatenate((static_x[:, -length:, :], transition_x))
    y = np.concatenate((static_y, transition_y))
    split = np.concatenate((static_split, transition_split_values))
    indices = np.arange(len(x))
    train = split == "train"
    static_test = (split == "test") & (indices < len(static_x))
    normalizer = ChannelNormalizer.fit(x[train])
    normalized = normalizer.transform(x)
    model = tf.keras.models.load_model(float_path, compile=False)
    float_prediction = np.argmax(model.predict(normalized[static_test], verbose=0), axis=1)
    rng = np.random.default_rng(20261001)
    calibration_indices = np.concatenate([
        rng.choice(np.flatnonzero(train & (y == label)), min(64, int(np.sum(train & (y == label)))), replace=False)
        for label in range(4)
    ])
    rng.shuffle(calibration_indices)
    out.mkdir(parents=True)
    int8_path = out / f"{float_path.stem}_strict_int8.tflite"
    input_detail, output_detail = export_full_int8_tflite(model, normalized[calibration_indices], int8_path)
    int_prediction = np.argmax(predict_tflite(int8_path, normalized[static_test]), axis=1)
    selection_rows = [index for index, row in enumerate(rows) if transition_split(row) == "architecture_selection"]
    transition = tflite_transition_metrics(int8_path, normalizer, traces[selection_rows], [rows[index] for index in selection_rows], length)
    reference = json.loads(REFERENCE.read_text())
    threshold = reference["int8_test"]["accuracy"] - 0.01
    static = {
        "float_accuracy": float(np.mean(float_prediction == y[static_test])),
        "float_macro_f1": macro_f1(y[static_test], float_prediction),
        "float_per_class_recall": class_recalls(y[static_test], float_prediction),
        "int8_accuracy": float(np.mean(int_prediction == y[static_test])),
        "int8_macro_f1": macro_f1(y[static_test], int_prediction),
        "int8_per_class_recall": class_recalls(y[static_test], int_prediction),
        "float_int8_agreement": float(np.mean(float_prediction == int_prediction)),
    }
    gates = {
        "STATIC_RETENTION_GATE": bool(static["int8_accuracy"] >= threshold),
        "TRANSITION_VALIDATION_GATE": bool(all(transition_gates(transition).values())),
    }
    manifest = {
        "candidate_specific_float_path": str(float_path), "candidate_specific_float_sha256": selected["sha256"],
        "int8_path": str(int8_path), "int8_sha256": sha(int8_path),
        "normalization": "fit on v4 train-only static+transition windows", "calibration": {
            "split": "train only", "count": int(len(calibration_indices)), "indices": calibration_indices.tolist(),
        }, "input": vars(input_detail), "output": vars(output_detail),
    }
    summary = {
        "selected_float": selected, "static_test": static,
        "static_reference_int8_accuracy": reference["int8_test"]["accuracy"], "static_retention_threshold": threshold,
        "transition_selection_int8": transition, "transition_gates": transition_gates(transition), "gates": gates,
        "TERRAIN_CAUSAL_WINDOW_INT8_READY": bool(all(gates.values())), "manifest": manifest,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
