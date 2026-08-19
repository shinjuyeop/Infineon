"""Leakage-audited float selection sweep for the v4 causal-window ablation.

The legacy transition validation and final test are deliberately not loaded.
Within the original transition-training partition, complete surface
realizations (not merely individual runs) are reserved for selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from terrain_cnn import ChannelNormalizer, build_compact_1d_cnn, estimate_model_macs, estimate_model_resources


SIM = Path(__file__).resolve().parents[2]
STATIC = SIM / "outputs/terrain_static_provenance_v4/dataset_noisy_provenance.npz"
TRANSITION = SIM / "outputs/terrain_transition_aware_v2_1_80_20"
REFERENCE = SIM / "outputs/terrain_static_reference_v4/summary.json"
OUT = SIM / "outputs/terrain_causal_window_v4"
WINDOWS = (20, 30, 50)
FRONTENDS = (("baseline", False), ("causal", True))
T0 = 650
SELECTION_SURFACE_REALIZATION = "2"
CASES = ("A", "B", "C", "D")
LABEL = {"concrete": 0, "marble": 1, "ice": 2, "sand": 3}
SEEDS = (20260821, 20260822, 20260823)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable(prediction: np.ndarray, target: int) -> int | None:
    count = 0
    for index in range(T0, len(prediction)):
        count = count + 1 if prediction[index] == target else 0
        if count == 3:
            return index
    return None


def macro_f1(y_true: np.ndarray, prediction: np.ndarray) -> float:
    scores = []
    for label in range(4):
        tp = np.sum((y_true == label) & (prediction == label))
        fp = np.sum((y_true != label) & (prediction == label))
        fn = np.sum((y_true == label) & (prediction != label))
        scores.append(float(2 * tp / max(2 * tp + fp + fn, 1)))
    return float(np.mean(scores))


def class_recalls(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {name: float(np.mean(prediction[y_true == label] == label)) for name, label in LABEL.items()}


def transition_split(row: dict[str, str]) -> str:
    """Map only original transition training data into v4 train/selection."""
    if row["split"] != "train":
        return "excluded_legacy_validation_or_test"
    return "architecture_selection" if row["surface_realization"] == SELECTION_SURFACE_REALIZATION else "train"


def transition_audit(rows: list[dict[str, str]]) -> dict[str, object]:
    owned_runs: dict[str, str] = {}
    owned_realizations: dict[tuple[str, str], str] = {}
    result: dict[str, object] = {
        "policy": "original transition train only; complete surface_realization=2 reserved for architecture selection; legacy validation/test excluded",
        "source_run_id_leakage": False,
        "surface_realization_leakage": False,
        "family_overlap_expected": True,
        "splits": {},
    }
    for row in rows:
        split = transition_split(row)
        if split not in {"train", "architecture_selection"}:
            continue
        prior = owned_runs.setdefault(row["run_id"], split)
        if prior != split:
            raise ValueError(f"transition source-run leakage: {row['run_id']}")
        key = (row["surface_family"], row["surface_realization"])
        prior = owned_realizations.setdefault(key, split)
        if prior != split:
            raise ValueError(f"transition realization leakage: {key}")
    for split in ("train", "architecture_selection"):
        group = [row for row in rows if transition_split(row) == split]
        per_case = {case: sum(row["case_id"] == case for row in group) for case in CASES}
        if not group or any(count == 0 for count in per_case.values()):
            raise ValueError(f"missing transition direction in {split}")
        result["splits"][split] = {
            "runs": len(group), "directions": per_case,
            "families": sorted({row["surface_family"] for row in group}),
            "surface_realizations": sorted({row["surface_realization"] for row in group}),
        }
    return result


def load_transition(length: int):
    with (TRANSITION / "split_manifest.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with np.load(TRANSITION / "transition_runs.npz") as values:
        traces = values["fusion10"]
    endpoints = np.r_[np.arange(550, 650, 5), np.arange(650, 700, 2), np.arange(700, 800, 5)]
    windows, labels, splits = [], [], []
    for index, row in enumerate(rows):
        windows.extend(traces[index, endpoint - length + 1 : endpoint + 1] for endpoint in endpoints)
        labels.extend(LABEL[row["terrain_before"] if endpoint < T0 else row["terrain_after"]] for endpoint in endpoints)
        splits.extend([transition_split(row)] * len(endpoints))
    return rows, traces, np.asarray(windows, np.float32), np.asarray(labels), np.asarray(splits)


def aggregate_transition(records: list[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    result = {}
    for case in CASES:
        group = [row for row in records if row["case_id"] == case]
        latencies = [row["t1_ms"] for row in group if row["t1_ms"] is not None]
        result[case] = {
            "runs": len(group),
            "stable_detection_rate": float(np.mean([row["detected"] for row in group])),
            "target_occupancy": float(np.mean([row["occupancy"] for row in group])),
            "prediction_switches_total": int(sum(row["switches"] for row in group)),
            "t1_median_ms": None if not latencies else float(np.median(latencies)),
            "t1_p95_ms": None if not latencies else float(np.percentile(latencies, 95)),
            "pre_transition_source_accuracy": float(np.mean([row["pre_source_accuracy"] for row in group])),
            "post_transition_target_accuracy": float(np.mean([row["post_target_accuracy"] for row in group])),
        }
    return result


def transition_metrics(model, normalizer, traces, rows, length: int) -> dict[str, dict[str, float | int]]:
    records = []
    for trace, row in zip(traces, rows):
        endpoints = np.arange(length - 1, len(trace))
        windows = np.asarray([trace[end - length + 1 : end + 1] for end in endpoints], np.float32)
        prediction = np.full(len(trace), -1, np.int8)
        prediction[length - 1 :] = np.argmax(model.predict(normalizer.transform(windows), batch_size=256, verbose=0), axis=1)
        target = LABEL[row["terrain_after"]]
        source = LABEL[row["terrain_before"]]
        t1 = stable(prediction, target)
        post = prediction[T0:]
        records.append({
            "run_id": row["run_id"], "case_id": row["case_id"],
            "detected": t1 is not None, "t1_ms": None if t1 is None else t1 - T0,
            "occupancy": float(np.mean(post == target)), "switches": int(np.count_nonzero(np.diff(post) != 0)),
            "pre_source_accuracy": float(np.mean(prediction[550:T0] == source)),
            "post_target_accuracy": float(np.mean(post == target)),
        })
    return aggregate_transition(records)


def transition_gates(metrics: dict[str, dict[str, float | int]]) -> dict[str, bool]:
    return {case: bool(metrics[case]["stable_detection_rate"] >= 0.90 and metrics[case]["target_occupancy"] >= 0.80) for case in CASES}


def candidate_key(candidate: dict[str, object]) -> tuple[float, ...]:
    """Predeclared selection: simplicity, context, then measured quality."""
    transition = candidate["transition_selection"]
    return (
        int(candidate["front_end"] == "baseline"), candidate["window_ms"],
        candidate["static_selection_accuracy"], transition["C"]["target_occupancy"],
        -transition["C"]["prediction_switches_total"], -transition["C"]["t1_p95_ms"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    with np.load(STATIC) as values:
        static_x, static_y, static_split = values["X"], values["y"], values["split"]
    reference = json.loads(REFERENCE.read_text())
    static_threshold = reference["candidates"][0]["selection_accuracy"] - 0.01
    out.mkdir(parents=True)
    (out / "candidate_results").mkdir()
    all_candidates = []
    for length in WINDOWS:
        rows, traces, transition_x, transition_y, transition_split_values = load_transition(length)
        audit = transition_audit(rows)
        if length == WINDOWS[0]:
            (out / "transition_split_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        x = np.concatenate((static_x[:, -length:, :], transition_x))
        y = np.concatenate((static_y, transition_y))
        split = np.concatenate((static_split, transition_split_values))
        indices = np.arange(len(x))
        static_train = np.flatnonzero((split == "train") & (indices < len(static_x)))
        transition_train = np.flatnonzero((split == "train") & (indices >= len(static_x)))
        transition_count = round(len(static_train) * 0.225 / 0.775)
        rng = np.random.default_rng(20260930 + length)
        labels = sorted(set(y[transition_train].tolist()))
        per_label = transition_count // len(labels)
        fit = np.concatenate((static_train, *[rng.choice(transition_train[y[transition_train] == label], per_label, replace=False) for label in labels]))
        normalizer = ChannelNormalizer.fit(x[split == "train"])
        normalized = normalizer.transform(x)
        static_selection = (split == "architecture_selection") & (indices < len(static_x))
        selection_rows = [index for index, row in enumerate(rows) if transition_split(row) == "architecture_selection"]
        for frontend, causal in FRONTENDS:
            for seed in SEEDS:
                import tensorflow as tf
                tf.keras.backend.clear_session()
                path = out / f"{frontend}_{length}_seed_{seed}.keras"
                model = build_compact_1d_cnn(10, seed, time_steps=length, causal=causal)
                model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                history = model.fit(normalized[fit], y[fit], validation_data=(normalized[static_selection], y[static_selection]), epochs=60, batch_size=128, callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)], verbose=0)
                model.save(path)
                prediction = np.argmax(model.predict(normalized[static_selection], verbose=0), axis=1)
                transition = transition_metrics(model, normalizer, traces[selection_rows], [rows[index] for index in selection_rows], length)
                candidate = {
                    "window_ms": length, "front_end": frontend, "seed": seed, "path": str(path), "sha256": sha(path), "epochs": len(history.history["loss"]),
                    "static_selection_accuracy": float(np.mean(prediction == y[static_selection])), "static_selection_macro_f1": macro_f1(y[static_selection], prediction), "static_selection_per_class_recall": class_recalls(y[static_selection], prediction),
                    "transition_selection": transition, "static_gate_float": bool(np.mean(prediction == y[static_selection]) >= static_threshold), "transition_gates": transition_gates(transition),
                    "parameters": estimate_model_resources(10, time_steps=length).parameters, "macs": estimate_model_macs(10, length),
                }
                candidate["eligible_float"] = bool(candidate["static_gate_float"] and all(candidate["transition_gates"].values()))
                (out / "candidate_results" / f"{frontend}_{length}_seed_{seed}.json").write_text(json.dumps(candidate, indent=2) + "\n")
                all_candidates.append(candidate)
    eligible = [candidate for candidate in all_candidates if candidate["eligible_float"]]
    selected = max(eligible, key=candidate_key) if eligible else None
    summary = {
        "static_reference_selection_accuracy": reference["candidates"][0]["selection_accuracy"], "static_float_retention_threshold": static_threshold,
        "frozen_mixture": {"requested_static_fraction": 0.775, "requested_transition_fraction": 0.225, "global_inverse_frequency_weighting": False},
        "endpoint_label": "window=[t-L+1,...,t], label=terrain_gt(t)", "transition_reservation": transition_audit(rows),
        "candidates": all_candidates, "selected_float": selected,
    }
    (out / "architecture_selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
