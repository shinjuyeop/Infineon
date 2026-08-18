"""Validation-only endpoint-aware temporal aggregation ablation for Terrain v3.

The source transition corpus is frozen before this program runs.  In particular,
the architecture-selection reservation is the existing family-disjoint validation
partition; no transition test or diagnostic trace is read for model selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from run_terrain_transition_aware_v2 import (
    CASES, FAMILIES, LABEL, SEEDS, SIM, STATIC, aggregate, macro_f1,
    transition_metrics, transition_metrics_int8, transition_windows,
)
from terrain_cnn import ChannelNormalizer, build_compact_1d_cnn, estimate_model_macs, estimate_model_resources
from terrain_int8 import export_full_int8_tflite, predict_tflite


SOURCE = SIM / "outputs/terrain_transition_aware_v2_1_80_20"
DEFAULT_OUT = SIM / "outputs/terrain_endpoint_aware_v3"
ARCHITECTURES = (("gap", "gap", 0), ("last_step", "last_step", 0), ("recent_8", "recent", 8))
STATIC_GATE = 0.96098
TRANSITION_GATE = 0.80


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def architecture_gates(metrics: dict[str, dict[str, float]]) -> dict[str, bool]:
    return {
        case: metrics[case]["detection_rate"] >= 0.9
        and metrics[case]["occupancy_mean"] >= TRANSITION_GATE
        for case in CASES
    }


def load_data():
    with (SOURCE / "split_manifest.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with np.load(SOURCE / "transition_runs.npz") as values:
        traces = values["fusion10"]
    tx, ty, ts, _ = transition_windows(traces, rows)
    with np.load(STATIC) as values:
        sx, sy, ss = values["X"], values["y"], values["split"]
    x = np.concatenate((sx, tx)); y = np.concatenate((sy, ty)); split = np.concatenate((ss, ts))
    return rows, traces, sx, x, y, split


def frozen_fit_indices(y: np.ndarray, split: np.ndarray, static_count: int) -> tuple[np.ndarray, dict[str, float]]:
    train = split == "train"
    static = np.flatnonzero(train & (np.arange(len(y)) < static_count))
    transition = np.flatnonzero(train & (np.arange(len(y)) >= static_count))
    # Freeze the 77.5:22.5 data mixture across all architecture candidates.
    desired_transition = round(len(static) * (1.0 - .775) / .775)
    labels = sorted(set(y[transition].tolist())); per = desired_transition // len(labels)
    rng = np.random.default_rng(20260833)
    sampled = np.concatenate([rng.choice(transition[y[transition] == label], per, replace=False) for label in labels])
    indices = np.concatenate((static, sampled))
    return indices, {"requested_static_fraction": .775, "effective_static_windows": int(len(static)), "effective_transition_windows": int(len(sampled)), "effective_static_fraction": float(len(static) / len(indices)), "global_inverse_frequency_weighting": False}


def train_candidate(name, aggregation, recent_k, seed, xn, y, fit, static_validation, rows, traces, out):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    path = out / f"{name}_seed_{seed}.keras"
    if path.exists():
        model = tf.keras.models.load_model(path, compile=False); epochs = None
    else:
        model = build_compact_1d_cnn(10, seed, aggregation=aggregation, recent_k=recent_k)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        history = model.fit(xn[fit], y[fit], validation_data=(xn[static_validation], y[static_validation]), epochs=60, batch_size=128, callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)], verbose=0)
        epochs = len(history.history["loss"]); model.save(path)
    pred = np.argmax(model.predict(xn[static_validation], verbose=0), 1)
    vi = [i for i, row in enumerate(rows) if row["split"] == "validation"]
    transition = aggregate(transition_metrics(model, NORMALIZER, traces[vi], [rows[i] for i in vi]))
    return {"architecture": name, "aggregation": aggregation, "recent_k": recent_k, "seed": seed, "path": str(path), "float_sha256": sha(path), "epochs": epochs, "static_validation_accuracy": float((pred == y[static_validation]).mean()), "static_validation_macro_f1": macro_f1(y[static_validation], pred), "transition_validation": transition, "transition_gates": architecture_gates(transition)}


def select(candidates):
    eligible = [candidate for candidate in candidates if all(candidate["transition_gates"].values())]
    if not eligible:
        return None
    # The architecture ordering encodes the predeclared simplicity tie-breaker.
    order = {"gap": 0, "last_step": 1, "recent_8": 2}
    return max(eligible, key=lambda candidate: (-order[candidate["architecture"]], candidate["static_validation_accuracy"], candidate["transition_validation"]["C"]["occupancy_mean"], -candidate["transition_validation"]["C"]["switches_total"], -candidate["transition_validation"]["C"]["latency_p95_ms"]))


def strict_int8(selected, x, y, split, static_count, rows, traces, out):
    import tensorflow as tf
    train = np.flatnonzero(split == "train"); rng = np.random.default_rng(20260903)
    calibration = np.concatenate([rng.choice(train[y[train] == label], 64, replace=False) for label in range(4)]); rng.shuffle(calibration)
    model = tf.keras.models.load_model(selected["path"], compile=False)
    path = out / "int8" / f"{selected['architecture']}_seed_{selected['seed']}_strict_int8.tflite"
    inp, outp = export_full_int8_tflite(model, NORMALIZER.transform(x[calibration]), path)
    static_test = (split == "test") & (np.arange(len(split)) < static_count)
    float_prediction = np.argmax(model.predict(NORMALIZER.transform(x[static_test]), verbose=0), 1)
    int_prediction = np.argmax(predict_tflite(path, NORMALIZER.transform(x[static_test])), 1)
    vi = [i for i, row in enumerate(rows) if row["split"] == "validation"]
    transition = aggregate(transition_metrics_int8(path, NORMALIZER, traces[vi], [rows[i] for i in vi]))
    manifest = {"float_sha256": selected["float_sha256"], "int8_sha256": sha(path), "candidate_specific_path": str(path.relative_to(out)), "normalization": {"fit_split": "combined static+transition train only", "train_windows": int(len(train))}, "representative_calibration": {"split": "train only", "seed": 20260903, "count": int(len(calibration)), "indices": calibration.tolist()}, "input": vars(inp), "output": vars(outp)}
    (out / "int8" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"path": str(path), "sha256": manifest["int8_sha256"], "static_accuracy": float((int_prediction == y[static_test]).mean()), "static_macro_f1": macro_f1(y[static_test], int_prediction), "float_agreement": float((int_prediction == float_prediction).mean()), "transition_validation": transition, "transition_gates": architecture_gates(transition), "static_gate": float((int_prediction == y[static_test]).mean()) >= STATIC_GATE, "manifest": manifest}


NORMALIZER: ChannelNormalizer


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT); args = parser.parse_args(); out = args.output_dir.resolve()
    if out.exists() and (out / "summary.json").exists(): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    global NORMALIZER
    rows, traces, sx, x, y, split = load_data(); NORMALIZER = ChannelNormalizer.fit(x[split == "train"]); xn = NORMALIZER.transform(x)
    fit, mixture = frozen_fit_indices(y, split, len(sx)); static_validation = (split == "validation") & (np.arange(len(split)) < len(sx))
    protocol = {"architecture_selection_split": "existing family-disjoint transition validation + static validation; source run/family/realization disjoint from training", "transition_test_excluded": True, "diagnostic_excluded": True, "frozen_training_mixture": mixture, "endpoint_labeling": "label(t), causal [t-49,t]", "persistence": 3, "architectures": [{"name": n, "aggregation": a, "recent_k": k, "parameters": estimate_model_resources(10).parameters, "macs": estimate_model_macs(10)} for n, a, k in ARCHITECTURES]}
    candidates = []
    for name, aggregation, recent_k in ARCHITECTURES:
        for seed in SEEDS:
            print(f"training architecture={name} seed={seed}", flush=True)
            candidates.append(train_candidate(name, aggregation, recent_k, seed, xn, y, fit, static_validation, rows, traces, out))
    selected = select(candidates)
    summary = {"protocol": protocol, "candidates": candidates, "selected_float": selected}
    if selected:
        import tensorflow as tf
        model = tf.keras.models.load_model(selected["path"], compile=False); model.save(out / "selected_model.keras")
        summary["selected_int8"] = strict_int8(selected, x, y, split, len(sx), rows, traces, out)
        summary["ready"] = summary["selected_int8"]["static_gate"] and all(summary["selected_int8"]["transition_gates"].values())
    else:
        summary["ready"] = False
    (out / "normalization.json").write_text(json.dumps(NORMALIZER.as_dict(), indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
