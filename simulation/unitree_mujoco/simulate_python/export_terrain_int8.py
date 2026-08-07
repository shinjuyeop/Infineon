"""Export the selected Expanded Dataset v1 Fusion model and verify INT8 parity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from terrain_cnn import (
    FUSION_CHANNEL_NAMES,
    ChannelNormalizer,
    evaluation_rows,
    mutual_pair_confusion,
)
from terrain_int8 import (
    export_full_int8_tflite,
    normalize,
    parity_gate,
    predict_tflite,
    require_tensorflow,
    select_calibration_indices,
)


SIMULATION_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = SIMULATION_DIR / "outputs/terrain_dataset_v1_expanded/dataset_noisy.npz"
CANDIDATE_DIR = (
    SIMULATION_DIR / "outputs/terrain_dataset_v1_expanded_cnn_seed_20260809_e120"
)
MODEL_PATH = CANDIDATE_DIR / "noisy_fusion.keras"
NORMALIZATION_PATH = CANDIDATE_DIR / "normalization.json"
TRAINING_PROTOCOL_PATH = CANDIDATE_DIR / "training_protocol.json"
OUTPUT_DIR = SIMULATION_DIR / "outputs/terrain_dataset_v1_expanded_int8_seed_20260809"
CLASS_NAMES = ("concrete", "marble", "ice", "sand")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--normalization", type=Path, default=NORMALIZATION_PATH)
    parser.add_argument("--training-protocol", type=Path, default=TRAINING_PROTOCOL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--representative-samples", type=int, default=256)
    parser.add_argument("--calibration-seed", type=int, default=20260809)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    model_path = args.model.resolve()
    normalization_path = args.normalization.resolve()
    protocol_path = args.training_protocol.resolve()
    output = args.output_dir.resolve()
    for path in (dataset_path, model_path, normalization_path, protocol_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)

    with np.load(dataset_path) as payload:
        required = {"X", "y", "split", "surface_family"}
        if not required.issubset(payload.files):
            raise ValueError(f"dataset is missing {sorted(required - set(payload.files))}")
        x = np.asarray(payload["X"], dtype=np.float32)
        y = np.asarray(payload["y"], dtype=np.int64)
        split = np.asarray(payload["split"])
        families = np.asarray(payload["surface_family"])
    if x.shape != (len(y), 50, 10) or not np.all(np.isfinite(x)):
        raise ValueError(f"unexpected dataset tensor {x.shape}")

    normalization_payload = json.loads(normalization_path.read_text(encoding="utf-8"))
    try:
        fusion_normalization = normalization_payload["noisy/fusion"]
    except KeyError as exc:
        raise ValueError("normalization file has no noisy/fusion entry") from exc
    if fusion_normalization["channels"] != list(range(10)):
        raise ValueError("candidate normalization is not Fusion10")
    mean = np.asarray(fusion_normalization["mean"], dtype=np.float32)
    std = np.asarray(fusion_normalization["std"], dtype=np.float32)
    recomputed = ChannelNormalizer.fit(x[split == "train"])
    if not np.allclose(mean, recomputed.mean, rtol=1e-6, atol=1e-7) or not np.allclose(
        std, recomputed.std, rtol=1e-6, atol=1e-7
    ):
        raise ValueError("stored normalization does not match the train partition")

    calibration_indices = select_calibration_indices(
        split, y, families, args.representative_samples, args.calibration_seed
    )
    calibration_rows = [
        {
            "sample_index": int(index),
            "split": str(split[index]),
            "terrain_id": int(y[index]),
            "terrain_class": CLASS_NAMES[int(y[index])],
            "surface_family": str(families[index]),
        }
        for index in calibration_indices
    ]
    if {row["split"] for row in calibration_rows} != {"train"}:
        raise RuntimeError("non-train sample entered calibration manifest")
    write_rows(output / "calibration_manifest.csv", calibration_rows)

    tf = require_tensorflow()
    model = tf.keras.models.load_model(model_path)
    if tuple(model.input_shape[1:]) != (50, 10) or tuple(model.output_shape[1:]) != (4,):
        raise ValueError(
            f"unexpected candidate interface input={model.input_shape} output={model.output_shape}"
        )
    representative = normalize(x[calibration_indices], mean, std)
    tflite_path = output / "noisy_fusion_int8.tflite"
    input_spec, output_spec = export_full_int8_tflite(model, representative, tflite_path)

    test_mask = split == "test"
    standardized_test = normalize(x[test_mask], mean, std)
    float_probabilities = np.asarray(
        model.predict(standardized_test, batch_size=64, verbose=0), dtype=np.float32
    )
    int8_probabilities = predict_tflite(tflite_path, standardized_test)
    float_prediction = np.argmax(float_probabilities, axis=1)
    int8_prediction = np.argmax(int8_probabilities, axis=1)
    test_y = y[test_mask]
    test_families = families[test_mask]

    metric_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    summary_values: dict[str, dict[str, float]] = {}
    for model_format, probabilities, prediction in (
        ("float32", float_probabilities, float_prediction),
        ("int8", int8_probabilities, int8_prediction),
    ):
        rows, matrix = evaluation_rows(test_y, probabilities, CLASS_NAMES)
        metric_rows.extend(
            {
                "model_format": model_format,
                "scope": "test",
                "surface_family": "all",
                **row,
            }
            for row in rows
        )
        confusion_rows.extend(
            {"model_format": model_format, **row} for row in matrix
        )
        overall = rows[0]
        pair_count, pair_support, pair_ratio = mutual_pair_confusion(
            test_y, prediction, 0, 1
        )
        pair_rows.append(
            {
                "model_format": model_format,
                "metric": "concrete_marble_mutual_confusion",
                "value": pair_ratio,
                "count": pair_count,
                "support": pair_support,
            }
        )
        summary_values[model_format] = {
            "accuracy": float(overall["accuracy"]),
            "macro_f1": float(overall["f1"]),
            "concrete_marble_mutual_confusion": pair_ratio,
        }
        for family in sorted(set(test_families.tolist())):
            family_mask = test_families == family
            family_rows, _ = evaluation_rows(
                test_y[family_mask], probabilities[family_mask], CLASS_NAMES
            )
            metric_rows.extend(
                {
                    "model_format": model_format,
                    "scope": "test_family",
                    "surface_family": str(family),
                    **row,
                }
                for row in family_rows
            )

    gate = parity_gate(
        summary_values["float32"]["accuracy"],
        summary_values["int8"]["accuracy"],
        summary_values["float32"]["macro_f1"],
        summary_values["int8"]["macro_f1"],
        summary_values["float32"]["concrete_marble_mutual_confusion"],
        summary_values["int8"]["concrete_marble_mutual_confusion"],
    )
    agreement = float(np.mean(float_prediction == int8_prediction))
    write_rows(output / "parity_metrics.csv", metric_rows)
    write_rows(output / "test_confusion.csv", confusion_rows)
    write_rows(output / "hard_surface_metrics.csv", pair_rows)

    training_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    metadata = {
        "model_version": "terrain_dataset_v1_expanded_noisy_fusion_seed20260809_int8_v1",
        "source_model": str(model_path),
        "source_model_sha256": sha256(model_path),
        "source_dataset": str(dataset_path),
        "source_dataset_sha256": sha256(dataset_path),
        "training_protocol": training_protocol,
        "input_shape": [50, 10],
        "channel_order": list(FUSION_CHANNEL_NAMES),
        "class_names": list(CLASS_NAMES),
        "sample_rate_hz": 100,
        "window": {"name": "medium_response", "start_s": 0.15, "end_s": 0.65},
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "calibration": {
            "partition": "train",
            "samples": len(calibration_indices),
            "seed": args.calibration_seed,
            "surface_families": sorted(set(families[calibration_indices].tolist())),
            "terrain_counts": {
                CLASS_NAMES[label]: int(np.sum(y[calibration_indices] == label))
                for label in range(4)
            },
        },
        "tflite": {
            "path": str(tflite_path),
            "sha256": sha256(tflite_path),
            "size_bytes": tflite_path.stat().st_size,
            "strict_builtins_int8": True,
            "input": asdict(input_spec),
            "output": asdict(output_spec),
        },
    }
    (output / "deployment_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "float32": summary_values["float32"],
        "int8": summary_values["int8"],
        "prediction_agreement": agreement,
        "gate": gate,
        "test_samples": int(np.sum(test_mask)),
        "test_surface_families": sorted(set(test_families.tolist())),
    }
    (output / "parity_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"wrote strict INT8 deployment candidate to {output}")
    if not bool(gate["passed"]):
        raise SystemExit("INT8 parity gate failed")


if __name__ == "__main__":
    main()
