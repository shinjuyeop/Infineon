"""Leakage-safe v3.1 Global+Recent temporal aggregation ablation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from run_terrain_transition_aware_v2 import CASES, LABEL, SEEDS, SIM, STATIC, aggregate, macro_f1, transition_metrics, transition_metrics_int8, transition_windows
from terrain_cnn import ChannelNormalizer, build_compact_1d_cnn, estimate_model_macs, estimate_model_resources
from terrain_int8 import export_full_int8_tflite, predict_tflite


SOURCE = SIM / "outputs/terrain_transition_aware_v2_1_80_20"
DEFAULT_OUT = SIM / "outputs/terrain_global_recent_v3_1"
ARCHITECTURES = (("gap", "gap", 0), ("global_recent_8", "global_recent", 8), ("global_recent_16", "global_recent", 16))
STATIC_GATE = .96098
TRANSITION_GATE = .80
NORMALIZER: ChannelNormalizer


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reserved_transition_split(rows: list[dict[str, str]]) -> tuple[list[int], list[int]]:
    """Reserve run_index 3 from each train family/realization/case for selection."""
    train, selection = [], []
    for index, row in enumerate(rows):
        if row["split"] != "train":
            continue
        (selection if int(row["run_index"]) == 3 else train).append(index)
    train_ids = {rows[index]["run_id"] for index in train}; selection_ids = {rows[index]["run_id"] for index in selection}
    if train_ids & selection_ids or not selection or {rows[index]["case_id"] for index in selection} != set(CASES):
        raise ValueError("architecture-selection reservation is invalid")
    return train, selection


def gates(metrics):
    return {case: metrics[case]["detection_rate"] >= .9 and metrics[case]["occupancy_mean"] >= TRANSITION_GATE for case in CASES}


def load_data():
    with (SOURCE / "split_manifest.csv").open() as stream: rows = list(csv.DictReader(stream))
    with np.load(SOURCE / "transition_runs.npz") as values: traces = values["fusion10"]
    tx, ty, _, run_ids = transition_windows(traces, rows)
    with np.load(STATIC) as values: sx, sy, ss = values["X"], values["y"], values["split"]
    transition_train_rows, selection_rows = reserved_transition_split(rows)
    train_ids = {rows[index]["run_id"] for index in transition_train_rows}; selection_ids = {rows[index]["run_id"] for index in selection_rows}
    transition_split = np.full(len(tx), "excluded", dtype="<U22")
    transition_split[np.isin(run_ids, list(train_ids))] = "train"
    transition_split[np.isin(run_ids, list(selection_ids))] = "architecture_selection"
    x = np.concatenate((sx, tx)); y = np.concatenate((sy, ty)); split = np.concatenate((ss, transition_split))
    return rows, traces, sx, x, y, split, transition_train_rows, selection_rows


def fit_indices(y, split, static_count):
    static = np.flatnonzero((split == "train") & (np.arange(len(y)) < static_count))
    transition = np.flatnonzero((split == "train") & (np.arange(len(y)) >= static_count))
    count = round(len(static) * .225 / .775); labels = sorted(set(y[transition].tolist())); per = count // len(labels)
    rng = np.random.default_rng(20260911); chosen = np.concatenate([rng.choice(transition[y[transition] == label], per, replace=False) for label in labels])
    result = np.concatenate((static, chosen))
    return result, {"requested_static_fraction": .775, "effective_static_windows": int(len(static)), "effective_transition_windows": int(len(chosen)), "effective_static_fraction": float(len(static) / len(result)), "global_inverse_frequency_weighting": False}


def train_one(name, aggregation, recent_k, seed, xn, y, fit, static_val, rows, traces, selection_rows, out):
    import tensorflow as tf
    tf.keras.backend.clear_session(); path = out / f"{name}_seed_{seed}.keras"
    if path.exists(): model = tf.keras.models.load_model(path, compile=False); epochs = None
    else:
        model = build_compact_1d_cnn(10, seed, aggregation=aggregation, recent_k=recent_k); model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        history = model.fit(xn[fit], y[fit], validation_data=(xn[static_val], y[static_val]), epochs=60, batch_size=128, callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)], verbose=0)
        epochs = len(history.history["loss"]); model.save(path)
    prediction = np.argmax(model.predict(xn[static_val], verbose=0), 1); trans = aggregate(transition_metrics(model, NORMALIZER, traces[selection_rows], [rows[index] for index in selection_rows]))
    return {"architecture": name, "aggregation": aggregation, "recent_k": recent_k, "seed": seed, "path": str(path), "float_sha256": sha(path), "epochs": epochs, "static_validation_accuracy": float((prediction == y[static_val]).mean()), "static_validation_macro_f1": macro_f1(y[static_val], prediction), "transition_validation": trans, "transition_gates": gates(trans)}


def select(candidates):
    eligible = [item for item in candidates if all(item["transition_gates"].values())]
    if not eligible: return None
    # Apply the simpler-head tie-breaker only to comparable transition margins.
    # A candidate just above 80% Case-C occupancy is not comparable to one with
    # a materially larger all-direction occupancy margin.
    best_margin = max(min(value["occupancy_mean"] for value in item["transition_validation"].values()) for item in eligible)
    comparable = [item for item in eligible if min(value["occupancy_mean"] for value in item["transition_validation"].values()) >= best_margin - .02]
    simple = {"gap": 0, "global_recent_8": 1, "global_recent_16": 2}
    return max(comparable, key=lambda item: (item["static_validation_accuracy"], item["transition_validation"]["C"]["occupancy_mean"], -item["transition_validation"]["C"]["switches_total"], -item["transition_validation"]["C"]["latency_p95_ms"], -simple[item["architecture"]]))


def int8_evaluation(selected, x, y, split, static_count, rows, traces, selection_rows, out):
    import tensorflow as tf
    train = np.flatnonzero(split == "train"); rng = np.random.default_rng(20260912); calibration = np.concatenate([rng.choice(train[y[train] == label], 64, replace=False) for label in range(4)]); rng.shuffle(calibration)
    model = tf.keras.models.load_model(selected["path"], compile=False); path = out / "int8" / f"{selected['architecture']}_seed_{selected['seed']}_strict_int8.tflite"; inp, outp = export_full_int8_tflite(model, NORMALIZER.transform(x[calibration]), path)
    static_test = (split == "test") & (np.arange(len(split)) < static_count); float_prediction = np.argmax(model.predict(NORMALIZER.transform(x[static_test]), verbose=0), 1); int_prediction = np.argmax(predict_tflite(path, NORMALIZER.transform(x[static_test])), 1)
    transition = aggregate(transition_metrics_int8(path, NORMALIZER, traces[selection_rows], [rows[index] for index in selection_rows]))
    manifest = {"float_sha256": selected["float_sha256"], "int8_sha256": sha(path), "float_candidate_path": selected["path"], "int8_candidate_path": str(path), "normalization": {"fit_split": "static train + reserved transition train only", "train_windows": int(len(train))}, "representative_calibration": {"split": "train only", "seed": 20260912, "count": int(len(calibration)), "indices": calibration.tolist()}, "input": vars(inp), "output": vars(outp)}
    (out / "int8").mkdir(exist_ok=True); (out / "int8" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"path": str(path), "sha256": manifest["int8_sha256"], "model_size_bytes": path.stat().st_size, "static_accuracy": float((int_prediction == y[static_test]).mean()), "static_macro_f1": macro_f1(y[static_test], int_prediction), "float_agreement": float((int_prediction == float_prediction).mean()), "transition_validation": transition, "transition_gates": gates(transition), "static_gate": float((int_prediction == y[static_test]).mean()) >= STATIC_GATE, "manifest": manifest}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT); args = parser.parse_args(); out = args.output_dir.resolve()
    if out.exists() and (out / "summary.json").exists(): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    global NORMALIZER
    rows, traces, sx, x, y, split, transition_train_rows, selection_rows = load_data(); NORMALIZER = ChannelNormalizer.fit(x[split == "train"]); xn = NORMALIZER.transform(x); fit, mixture = fit_indices(y, split, len(sx)); static_val = (split == "validation") & (np.arange(len(split)) < len(sx))
    protocol = {"architecture_selection_transition_reservation": {"source": "original transition train partition", "held_out_run_index": 3, "train_runs": len(transition_train_rows), "selection_runs": len(selection_rows), "directions": {case: sum(rows[index]["case_id"] == case for index in selection_rows) for case in CASES}, "run_leakage": False}, "static_validation": "existing static validation split (no run/realization metadata is present in the static artifact)", "transition_test_excluded": True, "diagnostic_excluded": True, "frozen_training_mixture": mixture, "architectures": [{"name": name, "aggregation": aggregation, "recent_k": recent_k, "parameters": estimate_model_resources(10, aggregation).parameters, "macs": estimate_model_macs(10)} for name, aggregation, recent_k in ARCHITECTURES]}
    candidates = []
    for name, aggregation, recent_k in ARCHITECTURES:
        for seed in SEEDS:
            print(f"training architecture={name} seed={seed}", flush=True); candidates.append(train_one(name, aggregation, recent_k, seed, xn, y, fit, static_val, rows, traces, selection_rows, out))
    selected = select(candidates); summary = {"protocol": protocol, "candidates": candidates, "selected_float": selected}
    if selected:
        import tensorflow as tf
        tf.keras.models.load_model(selected["path"], compile=False).save(out / "selected_model.keras"); summary["selected_int8"] = int8_evaluation(selected, x, y, split, len(sx), rows, traces, selection_rows, out); summary["ready"] = summary["selected_int8"]["static_gate"] and all(summary["selected_int8"]["transition_gates"].values())
    else: summary["ready"] = False
    (out / "normalization.json").write_text(json.dumps(NORMALIZER.as_dict(), indent=2) + "\n"); (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
