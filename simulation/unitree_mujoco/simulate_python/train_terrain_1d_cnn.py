"""Train shared compact 1D-CNN FSR/IMU/Fusion ablations on Expanded Dataset v1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from terrain_cnn import (
    CHANNEL_GROUPS,
    ChannelNormalizer,
    build_compact_1d_cnn,
    estimate_model_resources,
    evaluation_rows,
    mutual_pair_confusion,
)


EXPANDED_SCHEMA_NAME = "terrain_dataset_v1_expanded"
SIMULATION_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = SIMULATION_DIR / "outputs" / EXPANDED_SCHEMA_NAME
OUTPUT_DIR = SIMULATION_DIR / "outputs" / f"{EXPANDED_SCHEMA_NAME}_cnn"
DATASET_SEED = 20260807
TERRAIN_LABELS = {"concrete": 0, "marble": 1, "ice": 2, "sand": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--variants", nargs="+", choices=("clean", "noisy"), default=("clean", "noisy"))
    parser.add_argument(
        "--sensor-groups",
        nargs="+",
        choices=tuple(CHANNEL_GROUPS),
        default=tuple(CHANNEL_GROUPS),
        help="channel ablations to train; rate ablation uses fusion only",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DATASET_SEED)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="train one epoch on deterministic synthetic shapes without reading or writing artifacts",
    )
    return parser.parse_args()


def validate_dataset_arrays(
    x: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
    families: np.ndarray,
) -> None:
    if x.ndim != 3 or x.shape[1:] != (50, 10):
        raise ValueError(f"expected X=(N, 50, 10), got {x.shape}")
    if any(len(values) != len(x) for values in (y, split, families)):
        raise ValueError("dataset arrays have inconsistent lengths")
    if not np.all(np.isfinite(x)):
        raise ValueError("dataset contains NaN/Inf")
    expected_splits = {"train", "validation", "test"}
    if set(split.tolist()) != expected_splits:
        raise ValueError(f"expected splits {expected_splits}, got {set(split.tolist())}")
    owners: dict[str, str] = {}
    for family, split_name in zip(families.tolist(), split.tolist()):
        previous = owners.setdefault(str(family), str(split_name))
        if previous != split_name:
            raise ValueError(f"surface family {family} leaks across splits")
    for split_name in expected_splits:
        observed = set(y[split == split_name].tolist())
        if observed != set(TERRAIN_LABELS.values()):
            raise ValueError(f"split {split_name} does not contain all terrain classes")


def load_variant(dataset_dir: Path, variant: str) -> tuple[np.ndarray, ...]:
    path = dataset_dir / f"dataset_{variant}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as payload:
        required = {"X", "y", "split", "surface_family"}
        if not required.issubset(payload.files):
            raise ValueError(f"{path.name} is missing {sorted(required - set(payload.files))}")
        arrays = tuple(
            np.asarray(payload[name]) for name in ("X", "y", "split", "surface_family")
        )
    validate_dataset_arrays(*arrays)
    return arrays


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required; install simulate_python/requirements-cnn.txt"
        ) from exc
    return tf


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def synthetic_smoke(seed: int) -> None:
    rng = np.random.default_rng(seed)
    samples_per_class = 12
    y = np.repeat(np.arange(4), samples_per_class)
    x = rng.normal(0.0, 0.15, size=(len(y), 50, 10)).astype(np.float32)
    x += y[:, None, None] * np.linspace(0.05, 0.20, 10)[None, None, :]
    for group, channels in CHANNEL_GROUPS.items():
        selected = x[:, :, channels]
        normalizer = ChannelNormalizer.fit(selected)
        normalized = normalizer.transform(selected)
        model = build_compact_1d_cnn(len(channels), seed)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(normalized, y, epochs=1, batch_size=16, verbose=0)
        probabilities = np.asarray(model.predict(normalized[:4], verbose=0))
        if probabilities.shape != (4, 4) or not np.all(np.isfinite(probabilities)):
            raise RuntimeError(f"CNN smoke failed for {group}")
        resources = estimate_model_resources(len(channels))
        if model.count_params() != resources.parameters:
            raise RuntimeError(f"resource estimate mismatch for {group}")
        print(f"{group}: output={probabilities.shape}, parameters={model.count_params()}")


def main() -> None:
    args = parse_args()
    if args.smoke:
        synthetic_smoke(args.seed)
        return
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience < 0:
        raise ValueError("epochs/batch-size must be positive and patience non-negative")
    tf = _tensorflow()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    pair_metric_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    resource_rows: list[dict[str, object]] = []
    normalization: dict[str, object] = {}
    training_times: list[dict[str, object]] = []
    terrain_names = tuple(TERRAIN_LABELS)

    for variant in args.variants:
        x, y, split, families = load_variant(args.dataset_dir.resolve(), variant)
        for group in args.sensor_groups:
            channels = CHANNEL_GROUPS[group]
            tf.keras.backend.clear_session()
            selected = x[:, :, channels]
            train_mask = split == "train"
            validation_mask = split == "validation"
            normalizer = ChannelNormalizer.fit(selected[train_mask])
            normalized = normalizer.transform(selected)
            normalization[f"{variant}/{group}"] = {
                "channels": list(channels),
                **normalizer.as_dict(),
            }
            model = build_compact_1d_cnn(len(channels), args.seed)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=args.patience,
                    restore_best_weights=True,
                )
            ]
            training_start = time.perf_counter()
            history = model.fit(
                normalized[train_mask],
                y[train_mask],
                validation_data=(normalized[validation_mask], y[validation_mask]),
                epochs=args.epochs,
                batch_size=args.batch_size,
                shuffle=True,
                callbacks=callbacks,
                verbose=2,
            )
            training_time_s = time.perf_counter() - training_start
            training_times.append(
                {
                    "variant": variant,
                    "sensor_group": group,
                    "wall_time_s": training_time_s,
                    "epochs_completed": len(history.history["loss"]),
                }
            )
            model.save(output / f"{variant}_{group}.keras")
            model_path = output / f"{variant}_{group}.keras"
            for epoch in range(len(history.history["loss"])):
                history_rows.append(
                    {
                        "variant": variant,
                        "sensor_group": group,
                        "epoch": epoch + 1,
                        **{name: float(values[epoch]) for name, values in history.history.items()},
                    }
                )

            for split_name in ("train", "validation", "test"):
                mask = split == split_name
                probabilities = np.asarray(model.predict(normalized[mask], verbose=0))
                rows, matrix = evaluation_rows(y[mask], probabilities, terrain_names)
                metric_rows.extend(
                    {
                        "variant": variant,
                        "sensor_group": group,
                        "scope": "split",
                        "split": split_name,
                        "surface_family": "all",
                        **row,
                    }
                    for row in rows
                )
                if split_name == "test":
                    confusion_rows.extend(
                        {"variant": variant, "sensor_group": group, **row}
                        for row in matrix
                    )
                    prediction = np.argmax(probabilities, axis=1)
                    concrete = TERRAIN_LABELS["concrete"]
                    marble = TERRAIN_LABELS["marble"]
                    mutual_count, denominator, mutual_ratio = mutual_pair_confusion(
                        y[mask], prediction, concrete, marble
                    )
                    pair_metric_rows.append(
                        {
                            "variant": variant,
                            "sensor_group": group,
                            "split": "test",
                            "metric": "concrete_marble_mutual_confusion",
                            "value": mutual_ratio,
                            "count": mutual_count,
                            "support": denominator,
                        }
                    )

            for family in sorted(set(families[split == "test"].tolist())):
                mask = (split == "test") & (families == family)
                probabilities = np.asarray(model.predict(normalized[mask], verbose=0))
                rows, _ = evaluation_rows(y[mask], probabilities, terrain_names)
                metric_rows.extend(
                    {
                        "variant": variant,
                        "sensor_group": group,
                        "scope": "test_family",
                        "split": "test",
                        "surface_family": str(family),
                        **row,
                    }
                    for row in rows
                )

            estimate = estimate_model_resources(len(channels))
            resource_rows.append(
                {
                    "variant": variant,
                    "sensor_group": group,
                    **vars(estimate),
                    "keras_parameters": int(model.count_params()),
                    "keras_artifact_bytes": model_path.stat().st_size,
                    "estimate_scope": "tensor payload/liveness only; excludes TFLite arena metadata, alignment, and kernel scratch buffers",
                }
            )

    write_rows(output / "metrics.csv", metric_rows)
    write_rows(output / "test_confusion.csv", confusion_rows)
    write_rows(output / "hard_surface_metrics.csv", pair_metric_rows)
    write_rows(output / "training_history.csv", history_rows)
    write_rows(output / "resource_estimates.csv", resource_rows)
    (output / "normalization.json").write_text(
        json.dumps(normalization, indent=2) + "\n", encoding="utf-8"
    )
    dataset_protocol_path = args.dataset_dir.resolve() / "protocol.json"
    dataset_protocol = (
        json.loads(dataset_protocol_path.read_text(encoding="utf-8"))
        if dataset_protocol_path.is_file()
        else None
    )
    (output / "training_protocol.json").write_text(
        json.dumps(
            {
                "architecture": "Conv1D(12,k=5)-Conv1D(16,k=3)-GlobalAveragePooling-Dense(4)",
                "channel_groups": {name: list(values) for name, values in CHANNEL_GROUPS.items()},
                "trained_sensor_groups": list(args.sensor_groups),
                "variants": list(args.variants),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "patience": args.patience,
                "seed": args.seed,
                "source_dataset_dir": str(args.dataset_dir.resolve()),
                "source_dataset_protocol": dataset_protocol,
                "normalization": "per-channel mean/std fitted on train families only",
                "selection": "early stopping on validation-family loss; test families are evaluation-only",
                "training_times": training_times,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote CNN ablation outputs to {output}")


if __name__ == "__main__":
    main()
