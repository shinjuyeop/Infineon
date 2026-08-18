"""Establish the strict-INT8 static reference on the v4 provenance split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run_terrain_transition_aware_v2 import SIM, macro_f1
from terrain_cnn import ChannelNormalizer, build_compact_1d_cnn
from terrain_int8 import export_full_int8_tflite, predict_tflite


DATA = SIM / "outputs/terrain_static_provenance_v4/dataset_noisy_provenance.npz"
OUT = SIM / "outputs/terrain_static_reference_v4"
SEEDS = (20260921, 20260922, 20260923)


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=OUT); args = parser.parse_args(); out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True)
    import tensorflow as tf
    with np.load(DATA) as values: x, y, split = values["X"], values["y"], values["split"]
    train = split == "train"; selection = split == "architecture_selection"; test = split == "test"; norm = ChannelNormalizer.fit(x[train]); xn = norm.transform(x); candidates = []
    for seed in SEEDS:
        tf.keras.backend.clear_session(); model = build_compact_1d_cnn(10, seed, time_steps=50); model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        history = model.fit(xn[train], y[train], validation_data=(xn[selection], y[selection]), epochs=60, batch_size=128, callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)], verbose=0)
        path = out / f"gap_50_seed_{seed}.keras"; model.save(path); pred = np.argmax(model.predict(xn[selection], verbose=0), 1)
        candidates.append({"seed": seed, "path": str(path), "sha256": sha(path), "epochs": len(history.history["loss"]), "selection_accuracy": float((pred == y[selection]).mean()), "selection_macro_f1": macro_f1(y[selection], pred)})
    selected = max(candidates, key=lambda item: (item["selection_accuracy"], item["selection_macro_f1"])); model = tf.keras.models.load_model(selected["path"], compile=False); model.save(out / "selected_model.keras")
    float_pred = np.argmax(model.predict(xn[test], verbose=0), 1); rng = np.random.default_rng(20260924); indices = np.concatenate([rng.choice(np.flatnonzero(train & (y == label)), 64, replace=False) for label in range(4)]); rng.shuffle(indices)
    int8_path = out / "int8" / f"gap_50_seed_{selected['seed']}_strict_int8.tflite"; inp, outp = export_full_int8_tflite(model, xn[indices], int8_path); int_pred = np.argmax(predict_tflite(int8_path, xn[test]), 1)
    int8 = {"path": str(int8_path), "sha256": sha(int8_path), "accuracy": float((int_pred == y[test]).mean()), "macro_f1": macro_f1(y[test], int_pred), "float_agreement": float((int_pred == float_pred).mean()), "input": vars(inp), "output": vars(outp), "calibration": {"split": "train only", "count": int(len(indices)), "indices": indices.tolist()}}
    summary = {"dataset": str(DATA), "architecture": "gap_50_static_reference", "candidates": candidates, "selected": selected, "float_test": {"accuracy": float((float_pred == y[test]).mean()), "macro_f1": macro_f1(y[test], float_pred)}, "int8_test": int8}
    (out / "normalization.json").write_text(json.dumps(norm.as_dict(), indent=2) + "\n"); (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n"); (out / "int8" / "manifest.json").write_text(json.dumps({"float_sha256": selected["sha256"], "int8_sha256": int8["sha256"], "candidate_specific_path": str(int8_path.relative_to(out)), "calibration": int8["calibration"], "input": vars(inp), "output": vars(outp)}, indent=2) + "\n"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
