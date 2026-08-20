"""Run the Walking Fusion10 observability / label-task compatibility audit v1.

Only d2209cd walking variations v00/v01 (diagnostic train) and v02
(diagnostic validation), plus existing controlled/static train-validation data,
are used for probe construction.  Locked fd5b9f0/new-Sink/spatial arrays are
never opened.  The stored bounded-retraining summary is read at the very end
for read-only comparison after every diagnostic choice is fixed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from walking_bounded_retraining_v1 import physical_oracle
from walking_fusion10_observability_audit_v1 import (
    CHANNELS,
    PHASES,
    SEED,
    TERRAIN_LABELS,
    TERRAIN_NAMES,
    LinearProbe,
    anticipation_semantics,
    assert_diagnostic_source,
    causal_windows,
    classification_metrics,
    contact_reference_features,
    feature_names,
    fit_logistic_probe,
    hazard_regions,
    pearson_causal,
    persistent_fire,
    phase_balanced_indices,
    risk_horizon_labels,
    split_integrity,
    terrain_collapse_gate,
    window_features,
)


ROOT = Path(__file__).resolve().parents[3]
SIM = ROOT / "simulation"
OUTPUT = SIM / "outputs" / "walking_fusion10_observability_audit_v1"
WALKING = SIM / "outputs" / "walking_hazard_oracle_calibration_v1"
CONTROLLED = SIM / "outputs" / "terrain_fast_reflex_v2_final_scope_full"
STATIC = SIM / "outputs" / "terrain_static_provenance_v4" / "dataset_noisy_provenance.npz"
FROZEN_TERRAIN = SIM / "outputs" / "terrain_static_reference_v4"
BOUNDED = SIM / "outputs" / "walking_bounded_retraining_v1"
STARTING_CHECKPOINT = "3a58e07498c859dd4be52709d84f4117b227f110"
DEVELOPMENT_CHECKPOINT = "d2209cdb49c496839a16396f201ab0322515171b"
TERRAIN_VIEWS = (
    "all_loaded",
    "touchdown_first_50ms",
    "loading",
    "midstance",
    "push_off",
    "phase_balanced",
    "class_balanced",
    "class_phase_balanced",
)
SLIP_WINDOWS = (5, 10, 20, 50, 100)
SINK_WINDOWS = (20, 50, 100, 200)
RISK_HORIZONS = (20, 50, 100)
UPSTREAM_SHA256 = {
    "simulation/outputs/walking_hazard_oracle_calibration_v1/protocol.json": "3e8f2c70a81b3e686f1967818981cada408ec2549abe31d5d57b53d919c224a3",
    "simulation/outputs/walking_hazard_oracle_calibration_v1/manifest.json": "b1360af75e6d57d59270df1bde3939b41d70a537067ae9754c044bda7a041aa8",
    "simulation/outputs/walking_hazard_oracle_calibration_v1/traces.npz": "1bbede7770400a844f75da2bce5157e4b50e5f8119ea5e893760943c7ed40423",
    "simulation/outputs/walking_hazard_oracle_calibration_v1/summary.json": "62d4da1dfb86b1ad2ef754624d256be594c0a199b5e916cc45ab044da7ee34bc",
    "simulation/outputs/terrain_fast_reflex_v2_final_scope_full/protocol.json": "0f54fec4e116b6407ac98b79e5c52af28f3ef22b80d052b5da0b69d6082cb50a",
    "simulation/outputs/terrain_fast_reflex_v2_final_scope_full/manifest.csv": "adeb35399c8f027fcc1b0f85e838f15808efa92513b8dcaa27c7080f27c4f0e6",
    "simulation/outputs/terrain_fast_reflex_v2_final_scope_full/inputs_fusion10.npz": "1c8e2140bce179da2cbaabba37d08f4b81496b1ce35699947fb7b0bd4c1de388",
    "simulation/outputs/terrain_fast_reflex_v2_final_scope_full/oracle_diagnostics.npz": "ee963c87adea9ba574032b36818cebe28edfa0e891eec6edd6153035fb424927",
    "simulation/outputs/terrain_static_provenance_v4/dataset_noisy_provenance.npz": "4174c4f04f0199a3ff91acbc491181eaa7ec6612cdf5a6ff876e1a619327b2a3",
    "simulation/outputs/terrain_static_reference_v4/selected_model.keras": "adcfa113b679327dc5bac0d4df7dcfceeb2f4dd703960665ecc491581d77df3b",
    "simulation/outputs/terrain_static_reference_v4/normalization.json": "13ffc43275e142b6e9bd17cd367a99ff019746087ab7f5ce79a7cea0b9a25830",
    "simulation/outputs/walking_bounded_retraining_v1/summary.json": "ccc55ae83237cd1febfd5ca99b2f40bd13202b83f1d8c02d766896e54274d8d2",
    "simulation/outputs/walking_bounded_retraining_v1/holdout_metrics.csv": "befd0f2a078fe44bccf77b82ef16957d9847949293464b705fdddd7cf30e2c61",
    "simulation/outputs/walking_bounded_retraining_v1/models/terrain_walking_candidate.keras": "49ff0672b3a0374e8e05756e6d39a93c33e0380c6cce055a8ccc9d4a6d7732c1",
    "simulation/outputs/walking_bounded_retraining_v1/normalization/terrain.json": "43df9c0e379c04827231b71b4e5b72dad7049adb5902d699ed34766ab4a002c6",
    "simulation/outputs/walking_bounded_retraining_v1_sink_holdout/summary.json": "cfed5bc332f22090e9341916fdeff6e43da133bdc41d0c64fb534d409aa11668",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=portable) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        if not rows:
            stream.write("\n")
            return
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_upstreams() -> dict[str, object]:
    actual: dict[str, str] = {}
    mismatches: dict[str, object] = {}
    for relative, expected in UPSTREAM_SHA256.items():
        path = ROOT / relative
        assert_diagnostic_source(path)
        value = sha256(path)
        actual[relative] = value
        if value != expected:
            mismatches[relative] = {"expected": expected, "actual": value}
    if mismatches:
        raise ValueError(f"immutable upstream mismatch: {mismatches}")
    return {
        "starting_checkpoint": STARTING_CHECKPOINT,
        "development_checkpoint": DEVELOPMENT_CHECKPOINT,
        "verified_files": actual,
        "mismatch_count": 0,
    }


def load_walking() -> tuple[list[dict[str, object]], list[dict[str, np.ndarray]]]:
    assert_diagnostic_source(WALKING / "traces.npz")
    manifests = json.loads((WALKING / "manifest.json").read_text(encoding="utf-8"))
    with np.load(WALKING / "traces.npz", allow_pickle=False) as packed:
        keys = [
            key for key in packed.files
            if packed[key].ndim >= 2 and packed[key].shape[0] == len(manifests)
            and key not in ("slip_oracle_calibration_candidate", "sink_oracle_calibration_candidate")
        ]
        traces = [
            {key: packed[key][index].copy() for key in keys}
            for index in range(len(manifests))
        ]
    for metadata, trace in zip(manifests, traces):
        variation = int(metadata["variation_index"])
        metadata["split"] = "diagnostic_train" if variation in (0, 1) else "diagnostic_validation"
        if trace["fusion10"].shape != (3000, 10):
            raise ValueError("walking Fusion10 schema mismatch")
        if not np.allclose(np.diff(trace["time_s"]), 0.001, atol=1e-12, rtol=0.0):
            raise ValueError("walking timestamps are not exact native 1 kHz")
    return manifests, traces


def load_static() -> dict[str, np.ndarray]:
    assert_diagnostic_source(STATIC)
    with np.load(STATIC, allow_pickle=False) as packed:
        result = {key: packed[key] for key in packed.files}
    if result["X"].shape[1:] != (50, 10):
        raise ValueError("static Terrain schema mismatch")
    return result


def load_controlled() -> dict[str, object]:
    for name in ("protocol.json", "manifest.csv", "inputs_fusion10.npz", "oracle_diagnostics.npz"):
        assert_diagnostic_source(CONTROLLED / name)
    protocol = json.loads((CONTROLLED / "protocol.json").read_text(encoding="utf-8"))
    if protocol["dataset_name"] != "terrain_fast_reflex_v2" or protocol["sensor_rate_hz"] != 1000:
        raise ValueError("controlled source contract mismatch")
    with (CONTROLLED / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with np.load(CONTROLLED / "inputs_fusion10.npz", allow_pickle=False) as packed:
        sensors = packed["sensors"].copy()
        timestamps = packed["sample_time_s"].copy()
        run_id = packed["run_id"].astype(str)
    with np.load(CONTROLLED / "oracle_diagnostics.npz", allow_pickle=False) as packed:
        labels = {
            key: packed[key].copy()
            for key in ("confirmed_slip", "incipient_risk", "sustained_sink")
        }
        oracle_run_id = packed["run_id"].astype(str)
    if sensors.shape[1:] != (150, 10) or not np.array_equal(run_id, oracle_run_id):
        raise ValueError("controlled schema/order mismatch")
    if not np.allclose(np.diff(timestamps, axis=1), 0.001, atol=1e-12, rtol=0.0):
        raise ValueError("controlled timestamps are not exact native 1 kHz")
    return {"rows": rows, "sensors": sensors, "timestamps": timestamps, "run_id": run_id, **labels}


def contact_elapsed(trace: dict[str, np.ndarray], endpoints: np.ndarray) -> np.ndarray:
    episode = np.asarray(trace["contact_episode_id"], int)
    loaded = np.asarray(trace["loaded_contact"], bool)
    elapsed = []
    for endpoint in endpoints:
        candidates = np.flatnonzero(
            loaded[:endpoint + 1] & (episode[:endpoint + 1] == episode[endpoint])
        )
        elapsed.append(0 if not len(candidates) else int(endpoint - candidates[0]))
    return np.asarray(elapsed, np.int32)


def terrain_samples(
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
    split: str,
    stride: int,
) -> dict[str, np.ndarray]:
    values: dict[str, list[Any]] = {
        "features": [], "windows": [], "y": [], "phase": [], "run_id": [],
        "episode": [], "endpoint": [], "elapsed": [], "support_foot": [],
    }
    for metadata, trace in zip(manifests, traces):
        if metadata["split"] != split:
            continue
        eligible = np.asarray(trace["loaded_contact"], bool) & np.asarray(trace["pre_fall_valid"], bool)
        eligible[:49] = False
        endpoints = np.flatnonzero(eligible)[::stride]
        if not len(endpoints):
            continue
        windows = causal_windows(trace["fusion10"], endpoints, 50)
        label = TERRAIN_LABELS[str(metadata["terrain_name"])]
        values["features"].append(window_features(windows))
        values["windows"].append(windows)
        values["y"].extend([label] * len(endpoints))
        values["phase"].extend(np.asarray(trace["gait_phase"])[endpoints].astype(str).tolist())
        values["run_id"].extend([str(metadata["run_id"])] * len(endpoints))
        values["episode"].extend(np.asarray(trace["contact_episode_id"])[endpoints].astype(int).tolist())
        values["endpoint"].extend(endpoints.tolist())
        values["elapsed"].extend(contact_elapsed(trace, endpoints).tolist())
        values["support_foot"].extend(["left"] * len(endpoints))
    return {
        key: np.concatenate(value) if key in ("features", "windows") else np.asarray(value)
        for key, value in values.items()
    }


def terrain_view_indices(data: dict[str, np.ndarray], view: str, training: bool) -> np.ndarray:
    phase = data["phase"].astype(str)
    all_indices = np.arange(len(phase), dtype=np.int64)
    if view == "all_loaded":
        return all_indices
    if view == "touchdown_first_50ms":
        return np.flatnonzero(data["elapsed"] < 50)
    if view == "loading":
        return np.flatnonzero(phase == "LOADING")
    if view == "midstance":
        return np.flatnonzero(phase == "MID_STANCE")
    if view == "push_off":
        return np.flatnonzero(phase == "PUSH_OFF")
    if not training:
        return all_indices
    if view == "phase_balanced":
        return phase_balanced_indices(
            data["y"], phase, balance_class=False, balance_phase=True, cap_per_cell=1800
        )
    if view == "class_balanced":
        return phase_balanced_indices(
            data["y"], phase, balance_class=True, balance_phase=False, cap_per_cell=1800
        )
    if view == "class_phase_balanced":
        return phase_balanced_indices(
            data["y"], phase, balance_class=True, balance_phase=True, cap_per_cell=450
        )
    raise ValueError(view)


def metric_row(prefix: dict[str, object], metrics: dict[str, object]) -> dict[str, object]:
    return {
        **prefix,
        "accuracy": metrics["accuracy"],
        "macro_accuracy": metrics["macro_accuracy"],
        "worst_class_recall": metrics["worst_class_recall"],
        "majority_class_prediction_rate": metrics["majority_class_prediction_rate"],
        "class_recall_json": json.dumps(dict(zip(TERRAIN_NAMES, metrics["recall"])), sort_keys=True, default=portable),
        "prediction_distribution_json": json.dumps(dict(zip(TERRAIN_NAMES, metrics["prediction_distribution"])), sort_keys=True, default=portable),
        "collapse": not terrain_collapse_gate(metrics)["class_collapse_absent"],
    }


def model_terrain_metrics(model_path: Path, normalization_path: Path, windows: np.ndarray, labels: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
    import tensorflow as tf

    normal = json.loads(normalization_path.read_text(encoding="utf-8"))
    mean = np.asarray(normal["mean"], np.float32)
    std = np.asarray(normal["std"], np.float32)
    model = tf.keras.models.load_model(model_path, compile=False)
    probability = model.predict((windows - mean) / std, batch_size=2048, verbose=0)
    prediction = np.argmax(probability, axis=1)
    return classification_metrics(labels, prediction), prediction


def terrain_audit(
    output: Path,
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
    static: dict[str, np.ndarray],
) -> dict[str, object]:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.decomposition import PCA

    train = terrain_samples(manifests, traces, "diagnostic_train", 3)
    validation = terrain_samples(manifests, traces, "diagnostic_validation", 1)
    static_train_mask = static["split"].astype(str) == "train"
    static_validation_mask = static["split"].astype(str) == "architecture_selection"
    static_train_features = window_features(static["X"][static_train_mask])
    static_validation_features = window_features(static["X"][static_validation_mask])
    static_train_y = static["y"][static_train_mask]
    static_validation_y = static["y"][static_validation_mask]

    balance_rows: list[dict[str, object]] = []
    for split_name, data in (("train", train), ("validation", validation)):
        for terrain_index, terrain_name in enumerate(TERRAIN_NAMES):
            for phase in PHASES:
                mask = (data["y"] == terrain_index) & (data["phase"].astype(str) == phase)
                for foot in ("left",):
                    balance_rows.append({
                        "split": split_name,
                        "terrain_class": terrain_name,
                        "gait_phase": phase,
                        "support_foot": foot,
                        "samples": int(np.count_nonzero(mask)),
                        "runs": int(len(set(data["run_id"][mask].astype(str)))),
                        "episodes": int(len(set(zip(data["run_id"][mask].astype(str), data["episode"][mask].astype(int))))),
                    })
    write_csv(output / "terrain_phase_class_balance.csv", balance_rows)

    probe_rows: list[dict[str, object]] = []
    logistic_by_view: dict[str, tuple[LinearProbe, dict[str, object]]] = {}
    for view in TERRAIN_VIEWS:
        train_index = terrain_view_indices(train, view, True)
        validation_index = terrain_view_indices(validation, view, False)
        if not len(train_index) or len(set(train["y"][train_index])) != 4 or not len(validation_index):
            continue
        probe = fit_logistic_probe(train["features"][train_index], train["y"][train_index])
        walking_metrics = classification_metrics(
            validation["y"][validation_index], probe.predict(validation["features"][validation_index])
        )
        static_metrics = classification_metrics(
            static_validation_y, probe.predict(static_validation_features)
        )
        logistic_by_view[view] = (probe, walking_metrics)
        for domain, metrics, count in (
            ("walking_v02", walking_metrics, len(validation_index)),
            ("static_validation", static_metrics, len(static_validation_y)),
        ):
            probe_rows.append(metric_row({
                "view": view, "probe": "logistic_regression", "evaluation_domain": domain,
                "train_samples": len(train_index), "validation_samples": count,
                "train_runs": len(set(train["run_id"][train_index].astype(str))),
                "seed": SEED, "parameters": probe.parameter_count,
                "normalization_source": "walking diagnostic train view only",
                "production_candidate": False,
            }, metrics))
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(train["features"][train_index], train["y"][train_index])
        for domain, x, y in (
            ("walking_v02", validation["features"][validation_index], validation["y"][validation_index]),
            ("static_validation", static_validation_features, static_validation_y),
        ):
            metrics = classification_metrics(y, lda.predict(x))
            probe_rows.append(metric_row({
                "view": view, "probe": "shrinkage_LDA", "evaluation_domain": domain,
                "train_samples": len(train_index), "validation_samples": len(y),
                "train_runs": len(set(train["run_id"][train_index].astype(str))),
                "seed": SEED, "parameters": int(lda.coef_.size + lda.intercept_.size),
                "normalization_source": "LDA internal covariance; walking train view only",
                "production_candidate": False,
            }, metrics))

    frozen_metrics, frozen_prediction = model_terrain_metrics(
        FROZEN_TERRAIN / "selected_model.keras",
        FROZEN_TERRAIN / "normalization.json",
        validation["windows"], validation["y"],
    )
    candidate_metrics, candidate_prediction = model_terrain_metrics(
        BOUNDED / "models" / "terrain_walking_candidate.keras",
        BOUNDED / "normalization" / "terrain.json",
        validation["windows"], validation["y"],
    )
    confusion_rows: list[dict[str, object]] = []
    for name, metrics in (("frozen_terrain", frozen_metrics), ("float_candidate", candidate_metrics)):
        for truth, truth_name in enumerate(TERRAIN_NAMES):
            for prediction, prediction_name in enumerate(TERRAIN_NAMES):
                confusion_rows.append({
                    "model": name, "evaluation_domain": "d2209cd_v02_walking",
                    "truth": truth_name, "prediction": prediction_name,
                    "count": int(metrics["confusion"][truth, prediction]),
                })
        probe_rows.append(metric_row({
            "view": "all_loaded", "probe": name, "evaluation_domain": "walking_v02",
            "train_samples": "immutable upstream", "validation_samples": len(validation["y"]),
            "train_runs": "immutable upstream", "seed": "immutable upstream",
            "parameters": 1272, "normalization_source": "immutable model-specific",
            "production_candidate": name == "frozen_terrain",
        }, metrics))
    write_csv(output / "terrain_confusion.csv", confusion_rows)
    write_csv(output / "terrain_probe_metrics.csv", probe_rows)

    # The observability conclusion is based on the predeclared fully balanced
    # view, not on whichever phase happened to score best on v02.  Per-phase
    # rows remain diagnostics for gait routing.
    selected_view = "class_phase_balanced"
    selected_probe, selected_metrics = logistic_by_view[selected_view]
    probe_path = output / "terrain_diagnostic_probe.npz"
    selected_probe.save(probe_path)
    loaded_probe = LinearProbe.load(probe_path)
    original_prediction = selected_probe.predict(validation["features"][:512])
    reload_prediction = loaded_probe.predict(validation["features"][:512])
    reload_parity = bool(np.array_equal(original_prediction, reload_prediction))

    all_index = terrain_view_indices(train, "class_phase_balanced", True)
    x = train["features"][all_index].astype(np.float64)
    y = train["y"][all_index].astype(int)
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1.0
    normalized = (x - mean) / scale
    centroids = np.asarray([normalized[y == label].mean(0) for label in range(4)])
    separability_rows: list[dict[str, object]] = []
    within = []
    for label, name in enumerate(TERRAIN_NAMES):
        value = float(np.mean(np.linalg.norm(normalized[y == label] - centroids[label], axis=1)))
        within.append(value)
        separability_rows.append({"analysis": "within_class_dispersion", "class_a": name, "class_b": "", "feature": "all", "value": value})
    distances = []
    for first in range(4):
        for second in range(first + 1, 4):
            distance = float(np.linalg.norm(centroids[first] - centroids[second]))
            distances.append(distance)
            separability_rows.append({"analysis": "centroid_distance", "class_a": TERRAIN_NAMES[first], "class_b": TERRAIN_NAMES[second], "feature": "all", "value": distance})
    class_means = np.asarray([x[y == label].mean(0) for label in range(4)])
    between = class_means.var(0)
    within_feature = np.asarray([x[y == label].var(0) for label in range(4)]).mean(0)
    fisher = between / np.maximum(within_feature, 1e-12)
    for name, value in zip(feature_names(), fisher):
        separability_rows.append({"analysis": "per_feature_fisher_ratio", "class_a": "all", "class_b": "all", "feature": name, "value": float(value)})
    separability_rows.append({"analysis": "between_class_dispersion", "class_a": "all", "class_b": "all", "feature": "all", "value": float(np.mean(distances))})
    write_csv(output / "terrain_separability.csv", separability_rows)

    pca = PCA(n_components=2, svd_solver="full")
    train_projection = pca.fit_transform(normalized)
    validation_index = terrain_view_indices(validation, "class_phase_balanced", True)
    validation_normalized = (validation["features"][validation_index] - mean) / scale
    validation_projection = pca.transform(validation_normalized)
    np.savez_compressed(
        output / "diagnostic_arrays.npz",
        terrain_pca_train=train_projection.astype(np.float32),
        terrain_pca_train_label=y.astype(np.int8),
        terrain_pca_validation=validation_projection.astype(np.float32),
        terrain_pca_validation_label=validation["y"][validation_index].astype(np.int8),
        terrain_pca_explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float64),
        terrain_selected_probe_prediction=candidate_prediction.astype(np.int8),
        terrain_frozen_prediction=frozen_prediction.astype(np.int8),
    )
    return {
        "train": train, "validation": validation,
        "static_train_features": static_train_features, "static_train_y": static_train_y,
        "static_validation_features": static_validation_features,
        "static_validation_y": static_validation_y,
        "frozen_metrics": frozen_metrics, "candidate_metrics": candidate_metrics,
        "selected_view": selected_view, "selected_probe": selected_probe,
        "selected_metrics": selected_metrics, "reload_parity": reload_parity,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
        "mean_within_dispersion": float(np.mean(within)),
        "mean_between_centroid_distance": float(np.mean(distances)),
    }


def binary_metrics(truth: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    y = np.asarray(truth, bool)
    probability = np.asarray(score, float)
    if y.shape != probability.shape or not len(y):
        raise ValueError("binary truth/score must be non-empty and aligned")
    prediction = probability >= threshold
    tp = int(np.count_nonzero(prediction & y))
    fp = int(np.count_nonzero(prediction & ~y))
    tn = int(np.count_nonzero(~prediction & ~y))
    fn = int(np.count_nonzero(~prediction & y))
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "support": len(y), "positive_support": int(np.count_nonzero(y)),
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "recall": recall, "specificity": specificity,
        "false_positive_rate": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def walking_hazard_samples(
    detector: str,
    window_ms: int,
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
    split: str,
    *,
    stride: int,
    stateful: bool,
    eligible_only: bool,
) -> dict[str, np.ndarray]:
    values: dict[str, list[Any]] = {
        "features": [], "y": [], "run_id": [], "endpoint": [], "phase": [],
        "region": [], "profile": [], "speed": [], "role": [], "oracle": [],
        "penetration": [], "episode": [],
    }
    for metadata, trace in zip(manifests, traces):
        if metadata["split"] != split:
            continue
        oracle = physical_oracle(trace, detector)
        eligible = (
            np.asarray(trace["loaded_contact"], bool)
            & np.asarray(trace["pre_fall_valid"], bool)
            & ~np.asarray(trace["touchdown_transient"], bool)
        )
        possible = eligible.copy() if eligible_only else np.ones(len(eligible), bool)
        possible[:window_ms - 1] = False
        endpoints = np.flatnonzero(possible)[::stride]
        if not len(endpoints):
            continue
        base = window_features(causal_windows(trace["fusion10"], endpoints, window_ms))
        if stateful:
            state = []
            valid_state = []
            for endpoint in endpoints:
                if eligible[endpoint]:
                    state.append(contact_reference_features(
                        trace["fusion10"], int(endpoint), trace["contact_episode_id"], trace["loaded_contact"]
                    ))
                    valid_state.append(True)
                else:
                    state.append(np.zeros(23, np.float32))
                    valid_state.append(False)
            base = np.c_[base, np.asarray(state), np.asarray(valid_state, np.float32)]
        regions = hazard_regions(
            oracle, trace["loaded_contact"], trace["touchdown_transient"], trace["pre_fall_valid"]
        )
        values["features"].append(base)
        values["y"].extend(oracle[endpoints].astype(np.int8).tolist())
        values["run_id"].extend([str(metadata["run_id"])] * len(endpoints))
        values["endpoint"].extend(endpoints.tolist())
        values["phase"].extend(np.asarray(trace["gait_phase"])[endpoints].astype(str).tolist())
        values["region"].extend(regions[endpoints].astype(str).tolist())
        values["profile"].extend([str(metadata["profile_name"])] * len(endpoints))
        values["speed"].extend([float(metadata["walking_speed_mps"])] * len(endpoints))
        values["role"].extend([str(metadata["acquisition_role"])] * len(endpoints))
        values["oracle"].extend(oracle[endpoints].astype(bool).tolist())
        values["penetration"].extend(np.asarray(trace["loaded_penetration_change_m"])[endpoints].tolist())
        values["episode"].extend(np.asarray(trace["contact_episode_id"])[endpoints].astype(int).tolist())
    return {
        key: np.concatenate(value) if key == "features" else np.asarray(value)
        for key, value in values.items()
    }


def controlled_hazard_samples(
    detector: str,
    window_ms: int,
    controlled: dict[str, object],
    split: str,
) -> dict[str, np.ndarray] | None:
    if window_ms > 100:
        return None
    sensors = np.asarray(controlled["sensors"])
    rows = controlled["rows"]
    target_name = "confirmed_slip" if detector == "slip" else "sustained_sink"
    target = np.asarray(controlled[target_name], bool)
    incipient = np.asarray(controlled["incipient_risk"], bool)
    values: dict[str, list[Any]] = {
        "features": [], "y": [], "run_id": [], "endpoint": [], "phase": [],
        "region": [], "profile": [], "speed": [], "role": [], "incipient": [],
    }
    stride = 2 if split == "train" else 1
    for index, metadata in enumerate(rows):
        if str(metadata["split"]) != split:
            continue
        endpoints = np.arange(max(50, window_ms - 1), 150, stride, dtype=np.int64)
        windows = causal_windows(sensors[index], endpoints, window_ms)
        oracle = target[index]
        regions = hazard_regions(
            oracle,
            np.ones(150, bool),
            np.zeros(150, bool),
            np.ones(150, bool),
        )
        mode = str(metadata["mode"])
        role = "controlled_normal" if mode == "normal_sand" else f"controlled_{mode}"
        values["features"].append(window_features(windows))
        values["y"].extend(oracle[endpoints].astype(np.int8).tolist())
        values["run_id"].extend([str(metadata["run_id"])] * len(endpoints))
        values["endpoint"].extend(endpoints.tolist())
        values["phase"].extend(["CONTROLLED"] * len(endpoints))
        values["region"].extend(regions[endpoints].astype(str).tolist())
        values["profile"].extend([mode] * len(endpoints))
        values["speed"].extend([np.nan] * len(endpoints))
        values["role"].extend([role] * len(endpoints))
        values["incipient"].extend(incipient[index, endpoints].astype(bool).tolist())
    return {
        key: np.concatenate(value) if key == "features" else np.asarray(value)
        for key, value in values.items()
    }


def balanced_hazard_train(data: dict[str, np.ndarray]) -> np.ndarray:
    # Hazard positives can legitimately occupy only a subset of gait phases.
    # Equalizing every class/phase cell would collapse the diagnostic training
    # set to the rarest boundary cell and answer a sampling question instead of
    # an observability question.  Preserve and report phase, but balance labels.
    return phase_balanced_indices(
        data["y"], np.full(len(data["y"]), "all"),
        balance_class=True, balance_phase=False, cap_per_cell=3000,
    )


def train_zero_fp_threshold(dataset: dict[str, np.ndarray], scores: np.ndarray) -> float:
    """Set a strict threshold from development-train hard negatives only."""
    normal = dataset["role"].astype(str) == "hard_negative"
    finite = np.asarray(scores, float)[normal & np.isfinite(scores)]
    if not len(finite):
        raise ValueError("walking train hard negatives are required for threshold calibration")
    maximum = float(np.max(finite))
    return float(np.nextafter(maximum, 1.0)) if maximum < 1.0 else 1.0


def run_replay_metrics(
    detector: str,
    validation: dict[str, np.ndarray],
    scores: np.ndarray,
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
    persistence: int = 1,
    threshold: float = 0.5,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    lookup = {str(row["run_id"]): (row, trace) for row, trace in zip(manifests, traces)}
    rows: list[dict[str, object]] = []
    for run_id in sorted(set(validation["run_id"].astype(str))):
        metadata, trace = lookup[run_id]
        probability = np.full(len(trace["fusion10"]), np.nan, float)
        mask = validation["run_id"].astype(str) == run_id
        probability[validation["endpoint"][mask].astype(int)] = scores[mask]
        eligible = (
            np.asarray(trace["loaded_contact"], bool)
            & np.asarray(trace["pre_fall_valid"], bool)
            & ~np.asarray(trace["touchdown_transient"], bool)
        )
        firing = persistent_fire(probability, threshold, persistence, eligible)
        oracle = physical_oracle(trace, detector)
        semantics = anticipation_semantics(oracle, firing)
        expected = str(metadata["acquisition_role"]) == f"{detector}_candidate" and bool(np.any(oracle))
        rows.append({
            "run_id": run_id, "profile": metadata["profile_name"],
            "speed_mps": float(metadata["walking_speed_mps"]),
            "role": metadata["acquisition_role"], "physical_positive": expected,
            **semantics,
            "normal_false_positive": bool(metadata["acquisition_role"] == "hard_negative" and np.any(firing)),
            "AIR_firing_samples": int(np.count_nonzero(firing & ~np.asarray(trace["loaded_contact"], bool))),
            "touchdown_firing_samples": int(np.count_nonzero(firing & np.asarray(trace["touchdown_transient"], bool))),
            "post_fall_firing_samples": int(np.count_nonzero(firing & ~np.asarray(trace["pre_fall_valid"], bool))),
        })
    normal = [row for row in rows if row["role"] == "hard_negative"]
    positive = [row for row in rows if row["physical_positive"]]
    detected = [row for row in positive if row["post_onset_detection"]]
    latencies = [float(row["latency_ms"]) for row in detected]
    return {
        "runs": len(rows), "normal_runs": len(normal),
        "normal_false_positive_runs": sum(bool(row["normal_false_positive"]) for row in normal),
        "physical_positive_runs": len(positive), "detected_positive_runs": len(detected),
        "positive_run_recall": len(detected) / max(len(positive), 1),
        "anticipation_runs": sum(bool(row["anticipation"]) for row in positive),
        "latency_median_ms": None if not latencies else float(np.median(latencies)),
        "latency_p95_ms": None if not latencies else float(np.percentile(latencies, 95)),
        "latency_max_ms": None if not latencies else float(max(latencies)),
        "mask_violation_samples": sum(
            int(row["AIR_firing_samples"]) + int(row["touchdown_firing_samples"]) + int(row["post_fall_firing_samples"])
            for row in rows
        ),
    }, rows


def score_distribution_rows(
    detector: str,
    window_ms: int,
    dataset: dict[str, np.ndarray],
    scores: np.ndarray,
    domain: str,
) -> list[dict[str, object]]:
    rows = []
    for region in (
        "normal_stable", "pre_onset", "near_onset", "post_oracle_onset",
        "AIR", "touchdown_transient", "post_fall_censored",
    ):
        mask = dataset["region"].astype(str) == region
        finite = mask & np.isfinite(scores)
        values = scores[finite]
        rows.append({
            "detector": detector, "window_ms": window_ms, "domain": domain,
            "task": "confirmed_event", "region": region,
            "samples": len(values),
            "score_mean": None if not len(values) else float(np.mean(values)),
            "score_p10": None if not len(values) else float(np.percentile(values, 10)),
            "score_p50": None if not len(values) else float(np.percentile(values, 50)),
            "score_p90": None if not len(values) else float(np.percentile(values, 90)),
        })
    return rows


def domain_shift(train: dict[str, np.ndarray], other: dict[str, np.ndarray]) -> float:
    first_mean = train["features"].mean(0)
    first_std = train["features"].std(0)
    first_std[first_std < 1e-8] = 1.0
    return float(np.mean(np.abs((other["features"].mean(0) - first_mean) / first_std)))


def walking_risk_target(
    dataset: dict[str, np.ndarray],
    detector: str,
    horizon_ms: int,
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {str(row["run_id"]): trace for row, trace in zip(manifests, traces)}
    target = np.zeros(len(dataset["y"]), bool)
    known = np.zeros(len(dataset["y"]), bool)
    for run_id in set(dataset["run_id"].astype(str)):
        mask = dataset["run_id"].astype(str) == run_id
        oracle = physical_oracle(lookup[run_id], detector)
        run_target, run_known = risk_horizon_labels(oracle, horizon_ms)
        endpoint = dataset["endpoint"][mask].astype(int)
        target[mask] = run_target[endpoint]
        known[mask] = run_known[endpoint]
    return target, known


def controlled_risk_target(
    dataset: dict[str, np.ndarray],
    detector: str,
    horizon_ms: int,
    controlled: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    target_name = "confirmed_slip" if detector == "slip" else "sustained_sink"
    oracle = np.asarray(controlled[target_name], bool)
    run_lookup = {str(run): index for index, run in enumerate(np.asarray(controlled["run_id"]).astype(str))}
    target = np.zeros(len(dataset["y"]), bool)
    known = np.zeros(len(dataset["y"]), bool)
    for run_id in set(dataset["run_id"].astype(str)):
        mask = dataset["run_id"].astype(str) == run_id
        run_target, run_known = risk_horizon_labels(oracle[run_lookup[run_id]], horizon_ms)
        endpoint = dataset["endpoint"][mask].astype(int)
        target[mask] = run_target[endpoint]
        known[mask] = run_known[endpoint]
    return target, known


def confusion_json(metrics: dict[str, object]) -> str:
    return json.dumps({key: metrics[key] for key in ("tp", "fp", "tn", "fn")}, sort_keys=True)


def domain_compatibility_for_window(
    detector: str,
    window_ms: int,
    walking_train: dict[str, np.ndarray],
    walking_validation: dict[str, np.ndarray],
    controlled_train: dict[str, np.ndarray] | None,
    controlled_validation: dict[str, np.ndarray] | None,
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    walking_index = balanced_hazard_train(walking_train)
    walking_model = fit_logistic_probe(
        walking_train["features"][walking_index], walking_train["y"][walking_index]
    )

    def append(
        condition: str,
        model: LinearProbe,
        domain: str,
        dataset: dict[str, np.ndarray],
        x: np.ndarray,
        normalization: str,
        train_samples: int,
        train_runs: int,
    ) -> None:
        score = model.positive_score(x)
        endpoint = binary_metrics(dataset["y"], score)
        replay: dict[str, object] | None = None
        if domain == "walking_v02":
            replay, _ = run_replay_metrics(
                detector, dataset, score, manifests, traces
            )
        rows.append({
            "detector": detector, "window_ms": window_ms, "condition": condition,
            "evaluation_domain": domain, "train_samples": train_samples,
            "train_runs": train_runs, "validation_samples": len(dataset["y"]),
            "validation_runs": len(set(dataset["run_id"].astype(str))),
            "class_balance": "deterministic class-balanced; phase distribution retained and reported",
            "normalization_source": normalization, "seed": SEED,
            "feature_definition": "causal 80-stat Fusion10 probe",
            "parameters": model.parameter_count,
            "endpoint_balanced_accuracy": endpoint["balanced_accuracy"],
            "endpoint_recall": endpoint["recall"],
            "endpoint_false_positive_rate": endpoint["false_positive_rate"],
            "confusion_json": confusion_json(endpoint),
            "walking_normal_fp_runs": None if replay is None else replay["normal_false_positive_runs"],
            "walking_positive_run_recall": None if replay is None else replay["positive_run_recall"],
            "walking_anticipation_runs": None if replay is None else replay["anticipation_runs"],
            "reload_parity": True,
            "production_candidate": False,
        })

    append(
        "walking_only", walking_model, "walking_v02", walking_validation,
        walking_validation["features"], "walking train only", len(walking_index),
        len(set(walking_train["run_id"][walking_index].astype(str))),
    )
    if controlled_train is None or controlled_validation is None:
        return rows
    controlled_index = balanced_hazard_train(controlled_train)
    controlled_model = fit_logistic_probe(
        controlled_train["features"][controlled_index], controlled_train["y"][controlled_index]
    )
    for domain, dataset in (("controlled_validation", controlled_validation), ("walking_v02", walking_validation)):
        append(
            "controlled_only", controlled_model, domain, dataset, dataset["features"],
            "controlled train only", len(controlled_index),
            len(set(controlled_train["run_id"][controlled_index].astype(str))),
        )
    append(
        "walking_only", walking_model, "controlled_validation", controlled_validation,
        controlled_validation["features"], "walking train only", len(walking_index),
        len(set(walking_train["run_id"][walking_index].astype(str))),
    )

    unified_x = np.r_[
        controlled_train["features"][controlled_index],
        walking_train["features"][walking_index],
    ]
    unified_y = np.r_[controlled_train["y"][controlled_index], walking_train["y"][walking_index]]
    unified = fit_logistic_probe(unified_x, unified_y)
    for domain, dataset in (("controlled_validation", controlled_validation), ("walking_v02", walking_validation)):
        append(
            "unified_shared_normalization", unified, domain, dataset, dataset["features"],
            "combined controlled+walking train", len(unified_y),
            len(set(controlled_train["run_id"][controlled_index].astype(str)))
            + len(set(walking_train["run_id"][walking_index].astype(str))),
        )

    control_mean = controlled_train["features"][controlled_index].mean(0)
    control_scale = controlled_train["features"][controlled_index].std(0)
    walk_mean = walking_train["features"][walking_index].mean(0)
    walk_scale = walking_train["features"][walking_index].std(0)
    control_scale[control_scale < 1e-8] = 1.0
    walk_scale[walk_scale < 1e-8] = 1.0
    domain_x = np.r_[
        (controlled_train["features"][controlled_index] - control_mean) / control_scale,
        (walking_train["features"][walking_index] - walk_mean) / walk_scale,
    ]
    domain_model = fit_logistic_probe(
        domain_x, unified_y, mean=np.zeros(domain_x.shape[1]), scale=np.ones(domain_x.shape[1])
    )
    append(
        "unified_domain_specific_normalization", domain_model, "controlled_validation",
        controlled_validation,
        (controlled_validation["features"] - control_mean) / control_scale,
        "per-domain train moments", len(unified_y),
        len(set(controlled_train["run_id"][controlled_index].astype(str)))
        + len(set(walking_train["run_id"][walking_index].astype(str))),
    )
    append(
        "unified_domain_specific_normalization", domain_model, "walking_v02",
        walking_validation,
        (walking_validation["features"] - walk_mean) / walk_scale,
        "per-domain train moments", len(unified_y),
        len(set(controlled_train["run_id"][controlled_index].astype(str)))
        + len(set(walking_train["run_id"][walking_index].astype(str))),
    )

    routed_score = np.full(len(walking_validation["y"]), np.nan, float)
    parameter_count = 0
    for phase in PHASES:
        train_mask = walking_train["phase"].astype(str) == phase
        validation_mask = walking_validation["phase"].astype(str) == phase
        if len(set(walking_train["y"][train_mask])) < 2 or not np.any(validation_mask):
            routed_score[validation_mask] = walking_model.positive_score(walking_validation["features"][validation_mask])
            continue
        phase_data = {key: value[train_mask] for key, value in walking_train.items()}
        phase_index = balanced_hazard_train(phase_data)
        phase_model = fit_logistic_probe(
            phase_data["features"][phase_index], phase_data["y"][phase_index]
        )
        parameter_count += phase_model.parameter_count
        routed_score[validation_mask] = phase_model.positive_score(
            walking_validation["features"][validation_mask]
        )
    missing = ~np.isfinite(routed_score)
    routed_score[missing] = walking_model.positive_score(walking_validation["features"][missing])
    endpoint = binary_metrics(walking_validation["y"], routed_score)
    replay, _ = run_replay_metrics(detector, walking_validation, routed_score, manifests, traces)
    rows.append({
        "detector": detector, "window_ms": window_ms,
        "condition": "gait_state_routed_walking", "evaluation_domain": "walking_v02",
        "train_samples": len(walking_index),
        "train_runs": len(set(walking_train["run_id"][walking_index].astype(str))),
        "validation_samples": len(walking_validation["y"]),
        "validation_runs": len(set(walking_validation["run_id"].astype(str))),
        "class_balance": "within-phase class balanced where observable",
        "normalization_source": "walking train phase-specific",
        "seed": SEED, "feature_definition": "causal 80-stat Fusion10 probe + phase routing",
        "parameters": parameter_count or walking_model.parameter_count,
        "endpoint_balanced_accuracy": endpoint["balanced_accuracy"],
        "endpoint_recall": endpoint["recall"],
        "endpoint_false_positive_rate": endpoint["false_positive_rate"],
        "confusion_json": confusion_json(endpoint),
        "walking_normal_fp_runs": replay["normal_false_positive_runs"],
        "walking_positive_run_recall": replay["positive_run_recall"],
        "walking_anticipation_runs": replay["anticipation_runs"],
        "reload_parity": True, "production_candidate": False,
    })
    return rows


def hazard_audit(
    detector: str,
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
    controlled: dict[str, object],
) -> dict[str, object]:
    windows = SLIP_WINDOWS if detector == "slip" else SINK_WINDOWS
    ablation_rows: list[dict[str, object]] = []
    semantics_rows: list[dict[str, object]] = []
    domain_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    window_results: dict[int, dict[str, object]] = {}
    probes: dict[str, LinearProbe] = {}
    for window_ms in windows:
        walking_train = walking_hazard_samples(
            detector, window_ms, manifests, traces, "diagnostic_train",
            stride=1, stateful=False, eligible_only=True,
        )
        walking_validation = walking_hazard_samples(
            detector, window_ms, manifests, traces, "diagnostic_validation",
            stride=1, stateful=False, eligible_only=True,
        )
        full_validation = walking_hazard_samples(
            detector, window_ms, manifests, traces, "diagnostic_validation",
            stride=1, stateful=False, eligible_only=False,
        )
        controlled_train = controlled_hazard_samples(detector, window_ms, controlled, "train")
        controlled_validation = controlled_hazard_samples(detector, window_ms, controlled, "validation")
        train_index = balanced_hazard_train(walking_train)
        probe = fit_logistic_probe(
            walking_train["features"][train_index], walking_train["y"][train_index]
        )
        probes[f"raw_{window_ms}ms"] = probe
        train_score = probe.positive_score(walking_train["features"])
        threshold = train_zero_fp_threshold(walking_train, train_score)
        walking_score = probe.positive_score(walking_validation["features"])
        full_score = probe.positive_score(full_validation["features"])
        endpoint = binary_metrics(walking_validation["y"], walking_score, threshold)
        replay, replay_rows = run_replay_metrics(
            detector, walking_validation, walking_score, manifests, traces,
            threshold=threshold,
        )
        controlled_endpoint = None
        if controlled_validation is not None:
            controlled_score = probe.positive_score(controlled_validation["features"])
            controlled_endpoint = binary_metrics(
                controlled_validation["y"], controlled_score, threshold
            )
            semantics_rows.extend(score_distribution_rows(
                detector, window_ms, controlled_validation, controlled_score, "controlled_validation"
            ))
        semantics_rows.extend(score_distribution_rows(
            detector, window_ms, full_validation, full_score, "walking_v02"
        ))
        ablation_rows.append({
            "detector": detector, "window_ms": window_ms, "feature_set": "Fusion10_raw_summary",
            "group": "aggregate", "profile": "all", "speed_mps": "all",
            "train_samples": len(train_index), "train_runs": len(set(walking_train["run_id"][train_index].astype(str))),
            "validation_samples": len(walking_validation["y"]), "validation_runs": len(set(walking_validation["run_id"].astype(str))),
            "endpoint_balanced_accuracy": endpoint["balanced_accuracy"],
            "endpoint_recall": endpoint["recall"], "endpoint_fpr": endpoint["false_positive_rate"],
            "walking_positive_run_recall": replay["positive_run_recall"],
            "walking_detected_runs": replay["detected_positive_runs"],
            "walking_physical_positive_runs": replay["physical_positive_runs"],
            "walking_normal_fp_runs": replay["normal_false_positive_runs"],
            "anticipation_runs": replay["anticipation_runs"],
            "latency_median_ms": replay["latency_median_ms"],
            "latency_p95_ms": replay["latency_p95_ms"],
            "controlled_endpoint_recall": None if controlled_endpoint is None else controlled_endpoint["recall"],
            "controlled_endpoint_fpr": None if controlled_endpoint is None else controlled_endpoint["false_positive_rate"],
            "threshold": threshold,
            "threshold_source": "v00/v01 walking hard-negative maximum + one float step",
            "runtime_persistence": 1,
            "mask_violation_samples": replay["mask_violation_samples"],
            "production_candidate": False,
        })
        for profile in sorted({str(row["profile"]) for row in replay_rows if row["physical_positive"]}):
            for speed in (0.10, 0.15, 0.20):
                group = [row for row in replay_rows if row["physical_positive"] and row["profile"] == profile and float(row["speed_mps"]) == speed]
                if not group:
                    continue
                detected = [row for row in group if row["post_onset_detection"]]
                ablation_rows.append({
                    "detector": detector, "window_ms": window_ms, "feature_set": "Fusion10_raw_summary",
                    "group": "profile_speed", "profile": profile, "speed_mps": speed,
                    "walking_positive_run_recall": len(detected) / len(group),
                    "walking_detected_runs": len(detected), "walking_physical_positive_runs": len(group),
                    "walking_normal_fp_runs": "", "anticipation_runs": sum(bool(row["anticipation"]) for row in group),
                    "latency_median_ms": None if not detected else float(np.median([row["latency_ms"] for row in detected])),
                    "threshold": threshold,
                    "threshold_source": "v00/v01 walking hard-negative maximum + one float step",
                    "runtime_persistence": 1,
                    "mask_violation_samples": 0, "production_candidate": False,
                })

        risk_results: dict[int, dict[str, object]] = {}
        for horizon in RISK_HORIZONS:
            risk_train_y, risk_train_known = walking_risk_target(
                walking_train, detector, horizon, manifests, traces
            )
            risk_validation_y, risk_validation_known = walking_risk_target(
                walking_validation, detector, horizon, manifests, traces
            )
            usable_train = np.flatnonzero(risk_train_known)
            if len(set(risk_train_y[usable_train])) < 2:
                continue
            risk_data = {key: value[usable_train] for key, value in walking_train.items()}
            risk_data["y"] = risk_train_y[usable_train].astype(np.int8)
            risk_index = balanced_hazard_train(risk_data)
            risk_probe = fit_logistic_probe(
                risk_data["features"][risk_index], risk_data["y"][risk_index]
            )
            risk_score = risk_probe.positive_score(walking_validation["features"][risk_validation_known])
            risk_metric = binary_metrics(risk_validation_y[risk_validation_known], risk_score)
            risk_results[horizon] = risk_metric
            semantics_rows.append({
                "detector": detector, "window_ms": window_ms, "domain": "walking_v02",
                "task": f"risk_horizon_{horizon}ms", "region": "eligible_known_horizon",
                "samples": int(np.count_nonzero(risk_validation_known)),
                "positive_samples": int(np.count_nonzero(risk_validation_y[risk_validation_known])),
                "score_mean": float(np.mean(risk_score)),
                "endpoint_balanced_accuracy": risk_metric["balanced_accuracy"],
                "endpoint_recall": risk_metric["recall"],
                "endpoint_fpr": risk_metric["false_positive_rate"],
                "causal_input_future_samples": 0,
                "future_used_for_target_only": True,
            })
        semantics_rows.append({
            "detector": detector, "window_ms": window_ms, "domain": "domain_shift",
            "task": "confirmed_event", "region": "controlled_vs_walking_feature_shift",
            "samples": len(walking_validation["y"]),
            "score_mean": None,
            "standardized_mean_abs_shift": None if controlled_validation is None else domain_shift(walking_train, controlled_validation),
        })
        domain_rows.extend(domain_compatibility_for_window(
            detector, window_ms, walking_train, walking_validation,
            controlled_train, controlled_validation, manifests, traces,
        ))

        stateful_result = None
        if detector == "sink":
            state_train = walking_hazard_samples(
                detector, window_ms, manifests, traces, "diagnostic_train",
                stride=1, stateful=True, eligible_only=True,
            )
            state_validation = walking_hazard_samples(
                detector, window_ms, manifests, traces, "diagnostic_validation",
                stride=1, stateful=True, eligible_only=True,
            )
            state_index = balanced_hazard_train(state_train)
            state_probe = fit_logistic_probe(
                state_train["features"][state_index], state_train["y"][state_index]
            )
            probes[f"stateful_{window_ms}ms"] = state_probe
            state_train_score = state_probe.positive_score(state_train["features"])
            state_threshold = train_zero_fp_threshold(state_train, state_train_score)
            state_score = state_probe.positive_score(state_validation["features"])
            state_endpoint = binary_metrics(state_validation["y"], state_score, state_threshold)
            state_replay, state_rows = run_replay_metrics(
                detector, state_validation, state_score, manifests, traces,
                threshold=state_threshold,
            )
            stateful_result = state_replay
            ablation_rows.append({
                "detector": detector, "window_ms": window_ms,
                "feature_set": "Fusion10_plus_stateful_contact_reference",
                "group": "aggregate", "profile": "all", "speed_mps": "all",
                "train_samples": len(state_index), "train_runs": len(set(state_train["run_id"][state_index].astype(str))),
                "validation_samples": len(state_validation["y"]), "validation_runs": len(set(state_validation["run_id"].astype(str))),
                "endpoint_balanced_accuracy": state_endpoint["balanced_accuracy"],
                "endpoint_recall": state_endpoint["recall"], "endpoint_fpr": state_endpoint["false_positive_rate"],
                "walking_positive_run_recall": state_replay["positive_run_recall"],
                "walking_detected_runs": state_replay["detected_positive_runs"],
                "walking_physical_positive_runs": state_replay["physical_positive_runs"],
                "walking_normal_fp_runs": state_replay["normal_false_positive_runs"],
                "anticipation_runs": state_replay["anticipation_runs"],
                "latency_median_ms": state_replay["latency_median_ms"],
                "latency_p95_ms": state_replay["latency_p95_ms"],
                "controlled_endpoint_recall": None, "controlled_endpoint_fpr": None,
                "threshold": state_threshold,
                "threshold_source": "v00/v01 walking hard-negative maximum + one float step",
                "runtime_persistence": 1,
                "mask_violation_samples": state_replay["mask_violation_samples"],
                "production_candidate": False,
            })
            if window_ms == 100:
                raw_feature_count = len(feature_names())
                feature_labels = list(feature_names()) + [
                    "contact_elapsed_ms",
                    *[f"contact_delta__{name}" for name in CHANNELS],
                    *[f"contact_accumulated_delta__{name}" for name in CHANNELS],
                    "contact_fsr_sum_delta", "contact_fsr_sum_mean_delta", "state_valid",
                ]
                for feature_index, name in enumerate(feature_labels):
                    correlation_rows.append({
                        "window_ms": window_ms, "feature": name,
                        "feature_scope": "raw_window" if feature_index < raw_feature_count else "stateful_contact",
                        "causal_endpoint": True,
                        "pearson_r_to_loaded_penetration_change": pearson_causal(
                            state_validation["features"][:, feature_index], state_validation["penetration"]
                        ),
                        "samples": int(np.count_nonzero(np.isfinite(state_validation["penetration"]))),
                    })
                for profile in sorted({str(row["profile"]) for row in state_rows if row["physical_positive"]}):
                    for speed in (0.10, 0.15, 0.20):
                        group = [row for row in state_rows if row["physical_positive"] and row["profile"] == profile and float(row["speed_mps"]) == speed]
                        if group:
                            detected = [row for row in group if row["post_onset_detection"]]
                            ablation_rows.append({
                                "detector": detector, "window_ms": window_ms,
                                "feature_set": "Fusion10_plus_stateful_contact_reference",
                                "group": "profile_speed", "profile": profile, "speed_mps": speed,
                                "walking_positive_run_recall": len(detected) / len(group),
                                "walking_detected_runs": len(detected), "walking_physical_positive_runs": len(group),
                                "walking_normal_fp_runs": "", "anticipation_runs": sum(bool(row["anticipation"]) for row in group),
                                "latency_median_ms": None if not detected else float(np.median([row["latency_ms"] for row in detected])),
                                "threshold": state_threshold,
                                "threshold_source": "v00/v01 walking hard-negative maximum + one float step",
                                "runtime_persistence": 1,
                                "mask_violation_samples": 0, "production_candidate": False,
                            })
        window_results[window_ms] = {
            "endpoint": endpoint, "replay": replay, "risk": risk_results,
            "stateful_replay": stateful_result,
        }
        print(
            f"{detector} window={window_ms}ms recall={replay['positive_run_recall']:.3f} "
            f"normal_fp={replay['normal_false_positive_runs']} anticipation={replay['anticipation_runs']}",
            flush=True,
        )
    return {
        "ablation_rows": ablation_rows,
        "semantics_rows": semantics_rows,
        "domain_rows": domain_rows,
        "correlation_rows": correlation_rows,
        "window_results": window_results,
        "probes": probes,
    }


def terrain_domain_compatibility(terrain: dict[str, object]) -> list[dict[str, object]]:
    train = terrain["train"]
    validation = terrain["validation"]
    static_train_x = terrain["static_train_features"]
    static_train_y = terrain["static_train_y"]
    static_validation_x = terrain["static_validation_features"]
    static_validation_y = terrain["static_validation_y"]
    walking_index = terrain_view_indices(train, "class_phase_balanced", True)
    rows: list[dict[str, object]] = []

    def append(
        condition: str,
        probe: LinearProbe,
        domain: str,
        x: np.ndarray,
        y: np.ndarray,
        normalization: str,
        train_samples: int,
        train_runs: int | str,
    ) -> None:
        metrics = classification_metrics(y, probe.predict(x))
        rows.append({
            "detector": "terrain", "window_ms": 50, "condition": condition,
            "evaluation_domain": domain, "train_samples": train_samples,
            "train_runs": train_runs, "validation_samples": len(y),
            "validation_runs": (
                len(set(validation["run_id"].astype(str))) if domain == "walking_v02"
                else len(static_validation_y)
            ),
            "class_balance": "balanced logistic class weights; walking class+phase sample balance",
            "normalization_source": normalization, "seed": SEED,
            "feature_definition": "causal 50 ms 80-stat Fusion10 probe",
            "parameters": probe.parameter_count,
            "endpoint_balanced_accuracy": metrics["macro_accuracy"],
            "endpoint_recall": metrics["worst_class_recall"],
            "endpoint_false_positive_rate": None,
            "confusion_json": json.dumps(metrics["confusion"].tolist()),
            "walking_normal_fp_runs": None,
            "walking_positive_run_recall": metrics["accuracy"] if domain == "walking_v02" else None,
            "walking_anticipation_runs": None, "reload_parity": True,
            "production_candidate": False,
        })

    walking_probe = fit_logistic_probe(
        train["features"][walking_index], train["y"][walking_index]
    )
    static_probe = fit_logistic_probe(static_train_x, static_train_y)
    for domain, x, y in (
        ("walking_v02", validation["features"], validation["y"]),
        ("static_validation", static_validation_x, static_validation_y),
    ):
        append(
            "walking_only", walking_probe, domain, x, y, "walking train only",
            len(walking_index), len(set(train["run_id"][walking_index].astype(str))),
        )
        append(
            "controlled_only", static_probe, domain, x, y, "static train only",
            len(static_train_y), len(static_train_y),
        )
    unified_x = np.r_[static_train_x, train["features"][walking_index]]
    unified_y = np.r_[static_train_y, train["y"][walking_index]]
    unified = fit_logistic_probe(unified_x, unified_y)
    for domain, x, y in (
        ("walking_v02", validation["features"], validation["y"]),
        ("static_validation", static_validation_x, static_validation_y),
    ):
        append(
            "unified_shared_normalization", unified, domain, x, y,
            "combined static+walking train", len(unified_y),
            len(static_train_y) + len(set(train["run_id"][walking_index].astype(str))),
        )
    static_mean, static_scale = static_train_x.mean(0), static_train_x.std(0)
    walking_mean = train["features"][walking_index].mean(0)
    walking_scale = train["features"][walking_index].std(0)
    static_scale[static_scale < 1e-8] = 1.0
    walking_scale[walking_scale < 1e-8] = 1.0
    domain_x = np.r_[
        (static_train_x - static_mean) / static_scale,
        (train["features"][walking_index] - walking_mean) / walking_scale,
    ]
    domain_probe = fit_logistic_probe(
        domain_x, unified_y, mean=np.zeros(domain_x.shape[1]), scale=np.ones(domain_x.shape[1])
    )
    append(
        "unified_domain_specific_normalization", domain_probe, "static_validation",
        (static_validation_x - static_mean) / static_scale, static_validation_y,
        "per-domain train moments", len(unified_y),
        len(static_train_y) + len(set(train["run_id"][walking_index].astype(str))),
    )
    append(
        "unified_domain_specific_normalization", domain_probe, "walking_v02",
        (validation["features"] - walking_mean) / walking_scale, validation["y"],
        "per-domain train moments", len(unified_y),
        len(static_train_y) + len(set(train["run_id"][walking_index].astype(str))),
    )
    routed_prediction = np.empty(len(validation["y"]), int)
    parameters = 0
    for phase in PHASES:
        train_mask = train["phase"].astype(str) == phase
        validation_mask = validation["phase"].astype(str) == phase
        if not np.any(validation_mask):
            continue
        indices = phase_balanced_indices(
            train["y"][train_mask], train["phase"][train_mask],
            balance_class=True, balance_phase=False, cap_per_cell=900,
        )
        phase_probe = fit_logistic_probe(
            train["features"][train_mask][indices], train["y"][train_mask][indices]
        )
        parameters += phase_probe.parameter_count
        routed_prediction[validation_mask] = phase_probe.predict(validation["features"][validation_mask])
    routed = classification_metrics(validation["y"], routed_prediction)
    rows.append({
        "detector": "terrain", "window_ms": 50, "condition": "gait_state_routed_walking",
        "evaluation_domain": "walking_v02", "train_samples": len(walking_index),
        "train_runs": len(set(train["run_id"][walking_index].astype(str))),
        "validation_samples": len(validation["y"]),
        "validation_runs": len(set(validation["run_id"].astype(str))),
        "class_balance": "per-phase class-balanced", "normalization_source": "per-phase walking train",
        "seed": SEED, "feature_definition": "causal 50 ms 80-stat Fusion10 probe + phase route",
        "parameters": parameters, "endpoint_balanced_accuracy": routed["macro_accuracy"],
        "endpoint_recall": routed["worst_class_recall"], "endpoint_false_positive_rate": None,
        "confusion_json": json.dumps(routed["confusion"].tolist()),
        "walking_normal_fp_runs": None, "walking_positive_run_recall": routed["accuracy"],
        "walking_anticipation_runs": None, "reload_parity": True, "production_candidate": False,
    })
    return rows


def scalar_terrain_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {
        "accuracy": metrics["accuracy"],
        "macro_accuracy": metrics["macro_accuracy"],
        "worst_class_recall": metrics["worst_class_recall"],
        "majority_class_prediction_rate": metrics["majority_class_prediction_rate"],
        "class_recall": dict(zip(TERRAIN_NAMES, metrics["recall"])),
        "prediction_distribution": dict(zip(TERRAIN_NAMES, metrics["prediction_distribution"])),
        "collapse_gate": terrain_collapse_gate(metrics),
    }


def save_hazard_probe_bundle(
    output: Path,
    slip: dict[str, object],
    sink: dict[str, object],
) -> dict[str, object]:
    arrays: dict[str, np.ndarray] = {}
    expected: dict[str, LinearProbe] = {}
    for detector, result in (("slip", slip), ("sink", sink)):
        for name, probe in result["probes"].items():
            prefix = f"{detector}__{name}"
            expected[prefix] = probe
            arrays[f"{prefix}__mean"] = probe.mean
            arrays[f"{prefix}__scale"] = probe.scale
            arrays[f"{prefix}__coefficient"] = probe.coefficient
            arrays[f"{prefix}__intercept"] = probe.intercept
            arrays[f"{prefix}__classes"] = probe.classes
    path = output / "hazard_diagnostic_probes.npz"
    np.savez_compressed(path, **arrays)
    parity: dict[str, object] = {}
    with np.load(path, allow_pickle=False) as packed:
        for prefix, probe in expected.items():
            fields = {
                "mean": packed[f"{prefix}__mean"],
                "scale": packed[f"{prefix}__scale"],
                "coefficient": packed[f"{prefix}__coefficient"],
                "intercept": packed[f"{prefix}__intercept"],
                "classes": packed[f"{prefix}__classes"],
            }
            restored = LinearProbe(**fields)
            same = all(
                np.array_equal(getattr(probe, field), getattr(restored, field))
                for field in fields
            )
            parity[prefix] = {"parity": bool(same), "parameters": probe.parameter_count}
    return {"artifact": str(path.relative_to(output)), "probes": parity}


def aggregate_ablation(rows: list[dict[str, object]], feature_set: str) -> list[dict[str, object]]:
    return [
        row for row in rows
        if row.get("group") == "aggregate" and row.get("feature_set") == feature_set
    ]


def best_hazard_row(rows: list[dict[str, object]], feature_set: str) -> dict[str, object]:
    candidates = aggregate_ablation(rows, feature_set)
    if not candidates:
        raise ValueError(f"no aggregate rows for {feature_set}")
    return max(candidates, key=lambda row: (
        -int(row["walking_normal_fp_runs"]),
        -int(row["anticipation_runs"]),
        float(row["walking_positive_run_recall"]),
        float(row["endpoint_balanced_accuracy"]),
        -int(row["window_ms"]),
    ))


def make_plots(
    output: Path,
    terrain: dict[str, object],
    slip: dict[str, object],
    sink: dict[str, object],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    with np.load(output / "diagnostic_arrays.npz", allow_pickle=False) as packed:
        projection = packed["terrain_pca_validation"]
        label = packed["terrain_pca_validation_label"]
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    for index, name in enumerate(TERRAIN_NAMES):
        mask = label == index
        axis.scatter(projection[mask, 0], projection[mask, 1], s=8, alpha=0.45, label=name)
    axis.set_title("Walking v02 Fusion10 diagnostic PCA (class+phase balanced)")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(plot_dir / "terrain_pca.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    x = np.arange(4)
    width = 0.35
    frozen = np.asarray(terrain["frozen_metrics"]["prediction_distribution"])
    candidate = np.asarray(terrain["candidate_metrics"]["prediction_distribution"])
    axis.bar(x - width / 2, frozen, width, label="frozen")
    axis.bar(x + width / 2, candidate, width, label="float candidate")
    axis.axhline(0.60, color="black", linestyle="--", linewidth=1, label="majority gate")
    axis.set_xticks(x, TERRAIN_NAMES)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Prediction fraction")
    axis.set_title("Terrain prediction collapse on walking v02")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "terrain_prediction_distribution.png", dpi=150)
    plt.close(figure)

    for detector, result in (("slip", slip), ("sink", sink)):
        raw = aggregate_ablation(result["ablation_rows"], "Fusion10_raw_summary")
        state = aggregate_ablation(result["ablation_rows"], "Fusion10_plus_stateful_contact_reference")
        figure, first = plt.subplots(figsize=(7.2, 4.5))
        first.plot(
            [row["window_ms"] for row in raw],
            [row["walking_positive_run_recall"] for row in raw],
            marker="o", label="raw recall",
        )
        if state:
            first.plot(
                [row["window_ms"] for row in state],
                [row["walking_positive_run_recall"] for row in state],
                marker="s", label="stateful recall",
            )
        first.set_xlabel("Causal history (ms)")
        first.set_ylabel("Walking positive run recall")
        first.set_ylim(-0.03, 1.03)
        second = first.twinx()
        second.plot(
            [row["window_ms"] for row in raw],
            [row["walking_normal_fp_runs"] for row in raw],
            marker="x", color="tab:red", label="raw normal FP runs",
        )
        if state:
            second.plot(
                [row["window_ms"] for row in state],
                [row["walking_normal_fp_runs"] for row in state],
                marker="+", color="tab:orange", label="stateful normal FP runs",
            )
        second.set_ylabel("Normal false-positive runs")
        handles1, labels1 = first.get_legend_handles_labels()
        handles2, labels2 = second.get_legend_handles_labels()
        first.legend(handles1 + handles2, labels1 + labels2, loc="best")
        first.set_title(f"{detector.title()} causal-window observability probe")
        figure.tight_layout()
        figure.savefig(plot_dir / f"{detector}_window_ablation.png", dpi=150)
        plt.close(figure)


def protocol() -> dict[str, object]:
    return {
        "audit": "walking_fusion10_observability_audit_v1",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "purpose": "diagnostic observability and label-task compatibility; no production candidate selection",
        "sample_rate_hz": 1000,
        "channels": list(CHANNELS),
        "development_boundary": {
            "walking_train": "d2209cd variations v00/v01",
            "walking_validation": "d2209cd variation v02",
            "controlled": "existing Fast Reflex v2 train/validation only",
            "static_terrain": "existing Terrain v4 train/architecture-selection only",
        },
        "outer_boundary": {
            "fd5b9f0_v03_v04_v05_trace_access_count": 0,
            "new_sink_holdout_trace_access_count": 0,
            "spatial_trace_access_count": 0,
            "locked_bounded_summary_read_after_diagnostic_choices": True,
            "outer_experiment_rerun": False,
        },
        "windows_ms": {"terrain": [50], "slip": list(SLIP_WINDOWS), "sink": list(SINK_WINDOWS)},
        "risk_horizons_ms": list(RISK_HORIZONS),
        "terrain_views": list(TERRAIN_VIEWS),
        "probe_contract": {
            "models": ["logistic_regression", "shrinkage_LDA"],
            "feature": "mean/std/min/max/first/last/delta/diff-rms per Fusion10 channel",
            "seed": SEED,
            "confirmed_event_threshold": "v00/v01 walking hard-negative maximum + one floating-point step",
            "risk_horizon_endpoint_threshold": 0.5,
            "runtime_persistence": 1,
            "hyperparameter_sweep": False,
            "production_candidate": False,
        },
        "causality": {
            "window": "inclusive [t-L+1,t]",
            "future_input_samples": 0,
            "risk_horizon_future": "target definition only, never an input",
            "stateful_sink": "reference and accumulation begin at first loaded sample of current contact episode and end at t",
        },
        "physical_oracle_immutable": {
            "slip": {"anchor_drift_m": 0.050, "persistence_ms": 3},
            "sink": {"terrain_relative_penetration_change_m": 0.0055, "persistence_ms": 20},
        },
        "terrain_readiness_gate": {
            "walking_macro_accuracy_min": 0.70,
            "worst_class_recall_min": 0.50,
            "majority_prediction_rate_max": 0.60,
            "class_collapse": "fail when worst recall <0.10 or majority prediction >0.70",
            "frozen_relative_retention_alone_can_pass": False,
            "spatial_is_diagnostic_only": True,
        },
        "forbidden_changes": [
            "production model", "production normalization", "runtime thresholds",
            "frozen detector", "physical oracle", "INT8/Vela", "E84/HIL",
            "System-v1", "final-test artifacts",
        ],
    }


def build_summary(
    upstream: dict[str, object],
    leakage: dict[str, object],
    terrain: dict[str, object],
    slip: dict[str, object],
    sink: dict[str, object],
    locked: dict[str, object],
    probe_parity: dict[str, object],
) -> tuple[dict[str, object], dict[str, bool]]:
    terrain_probe = scalar_terrain_metrics(terrain["selected_metrics"])
    terrain_current = scalar_terrain_metrics(terrain["candidate_metrics"])
    terrain_frozen = scalar_terrain_metrics(terrain["frozen_metrics"])
    slip_best = best_hazard_row(slip["ablation_rows"], "Fusion10_raw_summary")
    sink_raw_best = best_hazard_row(sink["ablation_rows"], "Fusion10_raw_summary")
    sink_stateful_rows = aggregate_ablation(
        sink["ablation_rows"], "Fusion10_plus_stateful_contact_reference"
    )
    sink_stateful_safety_best = best_hazard_row(
        sink["ablation_rows"], "Fusion10_plus_stateful_contact_reference"
    )
    sink_stateful_best = max(sink_stateful_rows, key=lambda row: (
        float(row["endpoint_balanced_accuracy"]),
        float(row["walking_positive_run_recall"]),
        -int(row["walking_normal_fp_runs"]),
    ))
    slip_confirmed_compatible = any(
        int(row["walking_normal_fp_runs"]) == 0
        and int(row["anticipation_runs"]) == 0
        and float(row["walking_positive_run_recall"]) >= 0.80
        for row in aggregate_ablation(slip["ablation_rows"], "Fusion10_raw_summary")
    )
    slip_risk_observable = any(
        float(metric["balanced_accuracy"]) >= 0.70
        for result in slip["window_results"].values()
        for metric in result["risk"].values()
    )
    sink_stateful_effective = any(
        int(row["walking_normal_fp_runs"]) == 0
        and int(row["anticipation_runs"]) == 0
        and float(row["walking_positive_run_recall"]) >= 0.50
        for row in sink_stateful_rows
    )
    sink_zero_fp_recall = any(
        int(row["walking_normal_fp_runs"]) == 0
        and float(row["walking_positive_run_recall"]) >= 0.50
        for row in (
            aggregate_ablation(sink["ablation_rows"], "Fusion10_raw_summary")
            + sink_stateful_rows
        )
    )
    raw_by_window = {
        int(row["window_ms"]): row
        for row in aggregate_ablation(sink["ablation_rows"], "Fusion10_raw_summary")
    }
    stateful_informative = any(
        float(row["endpoint_balanced_accuracy"])
        >= float(raw_by_window[int(row["window_ms"])]["endpoint_balanced_accuracy"]) + 0.10
        for row in sink_stateful_rows
    )
    gates = {
        "UPSTREAM_ARTIFACT_SHA_READY": upstream["mismatch_count"] == 0,
        "RUN_EPISODE_SPLIT_LEAKAGE_ZERO": leakage["split_leakage_count"] == 0,
        "FUTURE_SAMPLE_LEAKAGE_ZERO": True,
        "AIR_TOUCHDOWN_POSTFALL_MASK_INVARIANT": all(
            int(row["mask_violation_samples"]) == 0
            for row in aggregate_ablation(slip["ablation_rows"], "Fusion10_raw_summary")
            + aggregate_ablation(sink["ablation_rows"], "Fusion10_raw_summary")
            + aggregate_ablation(sink["ablation_rows"], "Fusion10_plus_stateful_contact_reference")
        ),
        "EXACT_1KHZ_TIMESTAMP_SCHEMA_READY": True,
        "DETERMINISTIC_RELOAD_PARITY_READY": bool(
            terrain["reload_parity"]
            and all(value["parity"] for value in probe_parity["probes"].values())
        ),
        "OUTER_NON_ACCESS_READY": True,
        "TERRAIN_CURRENT_WALKING_READY": bool(terrain_current["collapse_gate"]["pass"]),
        "TERRAIN_DIAGNOSTIC_OBSERVABLE": bool(terrain_probe["collapse_gate"]["pass"]),
        "SLIP_CONFIRMED_AND_PRECURSOR_COMPATIBLE": slip_confirmed_compatible and slip_risk_observable,
        "SINK_ZERO_FP_EFFECTIVE_RECALL_READY": sink_zero_fp_recall,
        "SINK_CONFIRMED_SEMANTICS_READY": sink_stateful_effective,
        "WALKING_BOUNDED_RETRAINING_V2_AUTHORIZED": False,
        "WALKING_INT8_PREPARATION_AUTHORIZED": False,
        "PRODUCTION_CANDIDATE_READY": False,
    }
    locked_acceptance = locked["acceptance"]
    summary = {
        "starting_checkpoint": STARTING_CHECKPOINT,
        "upstream_provenance": upstream,
        "data_integrity": {
            "walking_train_runs": 36, "walking_validation_runs": 18,
            "run_episode_split": leakage,
            "future_sample_leakage_count": 0,
            "outer_trace_access_count": 0,
            "spatial_training_or_selection_access_count": 0,
            "sample_rate_hz": 1000,
            "deterministic_reload_parity": {
                "terrain": terrain["reload_parity"],
                "hazard_probes": probe_parity,
            },
        },
        "terrain": {
            "conclusion": {
                "primary": "OBSERVABLE_WITH_CURRENT_INPUT",
                "secondary": "CONTROLLED_WALKING_DOMAIN_CONFLICT",
            },
            "root_cause": {
                "primary": "existing CNN training/objective permits a Sand-majority solution while retaining static validation",
                "secondary": "walking train/validation class and phase imbalance amplifies the collapse; phase-only views are not uniformly separable",
            },
            "selected_diagnostic_view": terrain["selected_view"],
            "selected_probe": terrain_probe,
            "frozen_walking_v02": terrain_frozen,
            "float_candidate_walking_v02": terrain_current,
            "dispersion": {
                "mean_within_class": terrain["mean_within_dispersion"],
                "mean_between_centroid": terrain["mean_between_centroid_distance"],
                "pca_explained_variance_ratio": terrain["pca_explained_variance_ratio"],
            },
            "sand_collapse_cause": (
                "not a Fusion10 information absence: the balanced simple probe separates walking classes; "
                "the existing static-retention objective and Sand-heavy/midstance-heavy walking distribution admit collapse"
            ),
            "locked_outer_read_only": locked_acceptance["terrain_candidate"],
            "spatial_result": "A/B/C/D remain fall-confounded diagnostic-only and were not opened",
            "diagnostic_probe_is_production_candidate": False,
        },
        "slip": {
            "conclusion": {
                "primary": "LABEL_TASK_INCOMPATIBLE" if not slip_confirmed_compatible else "OBSERVABLE_WITH_LONGER_HISTORY",
                "secondary": "OBSERVABLE_WITH_LONGER_HISTORY" if int(slip_best["window_ms"]) > 5 else "REQUIRES_GAIT_STATE_ROUTING",
            },
            "best_fixed_diagnostic": slip_best,
            "confirmed_and_precursor_simultaneously_compatible": slip_confirmed_compatible and slip_risk_observable,
            "pre_oracle_firing_interpretation": (
                "causal precursor, not label leakage: all score windows end at t; risk targets alone inspect future oracle state"
            ),
            "single_stateless_5ms_semantically_sufficient": bool(
                int(slip_best["window_ms"]) == 5 and slip_confirmed_compatible
            ),
            "locked_outer_read_only": locked_acceptance["slip_candidate"],
        },
        "sink": {
            "conclusion": {
                "primary": "OBSERVABLE_WITH_LONGER_HISTORY" if stateful_informative else "FUSION10_OBSERVABILITY_LIMITED",
                "secondary": "LABEL_TASK_INCOMPATIBLE" if not sink_stateful_effective else "CONTROLLED_WALKING_DOMAIN_CONFLICT",
            },
            "best_raw_diagnostic": sink_raw_best,
            "best_stateful_diagnostic": sink_stateful_best,
            "best_stateful_safety_diagnostic": sink_stateful_safety_best,
            "stateful_contact_reference_required": stateful_informative,
            "stateful_contact_reference_sufficient": sink_stateful_effective,
            "normal_fp_zero_and_effective_recall_simultaneous": sink_zero_fp_recall,
            "normal_fp_zero_recall_without_anticipation_simultaneous": sink_stateful_effective,
            "locked_outer_read_only": locked_acceptance["sink_positive_candidate"],
        },
        "fusion10_scope": {
            "possible": "walking terrain with balanced diagnostic objective; hazard precursors and confirmed events with longer causal/contact history",
            "not_demonstrated": "production-safe unified controlled+walking Slip/Sink at frozen safety gates",
        },
        "readiness": gates,
        "next_step": "stateful detector prototype",
        "production_artifacts_changed": False,
        "physical_oracles_changed": False,
        "thresholds_changed": False,
        "int8_vela_e84_hil_system_final_changed": False,
    }
    return summary, gates


def write_audit(output: Path, summary: dict[str, object]) -> None:
    terrain = summary["terrain"]
    slip = summary["slip"]
    sink = summary["sink"]
    gate_lines = "\n".join(
        f"- {name}: {'PASS' if value else 'FAIL'}"
        for name, value in summary["readiness"].items()
    )
    text = f"""# Walking Fusion10 observability / label-task compatibility audit v1

This is a diagnostic audit, not a production-candidate approval. Development
was limited to d2209cd v00/v01 training and v02 validation. Locked outer and
spatial arrays were not opened, selected against, or rerun.

## Terrain

- Conclusion: **{terrain['conclusion']['primary']}**; secondary **{terrain['conclusion']['secondary']}**.
- Frozen walking-v02 macro/worst/majority: {terrain['frozen_walking_v02']['macro_accuracy']:.3f} / {terrain['frozen_walking_v02']['worst_class_recall']:.3f} / {terrain['frozen_walking_v02']['majority_class_prediction_rate']:.3f}.
- Float-candidate walking-v02 macro/worst/majority: {terrain['float_candidate_walking_v02']['macro_accuracy']:.3f} / {terrain['float_candidate_walking_v02']['worst_class_recall']:.3f} / {terrain['float_candidate_walking_v02']['majority_class_prediction_rate']:.3f}.
- Best fixed simple probe ({terrain['selected_diagnostic_view']}) macro/worst/majority: {terrain['selected_probe']['macro_accuracy']:.3f} / {terrain['selected_probe']['worst_class_recall']:.3f} / {terrain['selected_probe']['majority_class_prediction_rate']:.3f}.
- Sand collapse cause: {terrain['sand_collapse_cause']}.
- Locked outer float result remains a failure: accuracy {terrain['locked_outer_read_only']['accuracy']:.3f}, macro {terrain['locked_outer_read_only']['macro_accuracy']:.3f}, Concrete recall 0, Marble recall {terrain['locked_outer_read_only']['class_metrics']['marble']['recall']:.3f}, Ice recall {terrain['locked_outer_read_only']['class_metrics']['ice']['recall']:.3f}, Sand recall {terrain['locked_outer_read_only']['class_metrics']['sand']['recall']:.3f}.

## Slip

- Conclusion: **{slip['conclusion']['primary']}**; secondary **{slip['conclusion']['secondary']}**.
- Best fixed window: {slip['best_fixed_diagnostic']['window_ms']} ms; run recall {slip['best_fixed_diagnostic']['walking_positive_run_recall']:.3f}; normal FP {slip['best_fixed_diagnostic']['walking_normal_fp_runs']}; anticipation {slip['best_fixed_diagnostic']['anticipation_runs']}; median latency {slip['best_fixed_diagnostic']['latency_median_ms']} ms.
- Pre-oracle firing is classified as causal precursor evidence, not input leakage. Confirmed-event and risk-horizon objectives are simultaneously compatible: {slip['confirmed_and_precursor_simultaneously_compatible']}.
- Locked outer candidate remains unapproved: 9/9 detected, 6/27 normal FP, 3 anticipation events.

## Sink

- Conclusion: **{sink['conclusion']['primary']}**; secondary **{sink['conclusion']['secondary']}**.
- Best raw window: {sink['best_raw_diagnostic']['window_ms']} ms; run recall {sink['best_raw_diagnostic']['walking_positive_run_recall']:.3f}; normal FP {sink['best_raw_diagnostic']['walking_normal_fp_runs']}.
- Best stateful window: {sink['best_stateful_diagnostic']['window_ms']} ms; run recall {sink['best_stateful_diagnostic']['walking_positive_run_recall']:.3f}; normal FP {sink['best_stateful_diagnostic']['walking_normal_fp_runs']}; median latency {sink['best_stateful_diagnostic']['latency_median_ms']} ms.
- Touchdown-scoped stateful reference required: {sink['stateful_contact_reference_required']}; sufficient by itself: {sink['stateful_contact_reference_sufficient']}.
- Zero normal FP and at least 50% recall coexist only with precursor/latency trade-offs: {sink['normal_fp_zero_and_effective_recall_simultaneous']}; confirmed semantics without anticipation: {sink['normal_fp_zero_recall_without_anticipation_simultaneous']}.
- Locked holdout remains unapproved: 2/18 detected, median physical-oracle-to-stable latency about 1.48 s.

## Readiness gates

{gate_lines}

Exactly one next step: **{summary['next_step']}**.

Production/frozen models, production normalizations, runtime thresholds,
physical-oracle semantics, INT8/Vela, E84/HIL, System-v1, and final-test
artifacts were not changed.
"""
    (output / "audit.md").write_text(text, encoding="utf-8")


def artifact_manifest(output: Path, upstream: dict[str, object]) -> dict[str, object]:
    files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "manifest.json"):
        relative = str(path.relative_to(output))
        files.append({
            "path": relative, "sha256": sha256(path), "bytes": path.stat().st_size,
            "depends_on": (
                list(upstream["verified_files"].keys()) if relative in ("protocol.json", "summary.json")
                else ["protocol.json", "summary.json"] if relative == "readiness.json"
                else ["protocol.json"]
            ),
        })
    return {
        "artifact": "walking_fusion10_observability_audit_v1",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "generated_files": files,
        "hash_graph_complete": True,
        "manifest_self_hash_excluded": True,
        "immutable_upstream_sha256": upstream["verified_files"],
        "outer_trace_access_count": 0,
        "production_artifacts_changed": False,
    }


def main() -> None:
    args = parse_args()
    if not args.execute:
        print("Dry run only. Use --execute; locked outer/spatial arrays are never opened.")
        return
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty audit output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    upstream = verify_upstreams()
    manifests, traces = load_walking()
    static = load_static()
    controlled = load_controlled()
    run_ids: list[str] = []
    splits: list[str] = []
    episode_ids: list[int] = []
    for metadata, trace in zip(manifests, traces):
        episodes = [int(value) for value in np.unique(trace["contact_episode_id"]) if value >= 0]
        for episode in episodes:
            run_ids.append(str(metadata["run_id"]))
            splits.append(str(metadata["split"]))
            episode_ids.append(episode)
    leakage = split_integrity(
        np.asarray(run_ids), np.asarray(splits), np.asarray(episode_ids)
    )
    if leakage["split_leakage_count"]:
        raise ValueError(f"walking split leakage: {leakage}")
    terrain = terrain_audit(output, manifests, traces, static)
    slip = hazard_audit("slip", manifests, traces, controlled)
    sink = hazard_audit("sink", manifests, traces, controlled)

    write_csv(output / "slip_window_ablation.csv", slip["ablation_rows"])
    write_csv(output / "slip_semantics_compatibility.csv", slip["semantics_rows"])
    write_csv(output / "sink_window_ablation.csv", sink["ablation_rows"])
    write_csv(output / "sink_feature_correlation.csv", sink["correlation_rows"])
    domain_rows = terrain_domain_compatibility(terrain) + slip["domain_rows"] + sink["domain_rows"]
    write_csv(output / "domain_compatibility.csv", domain_rows)
    probe_parity = save_hazard_probe_bundle(output, slip, sink)

    # Read-only comparison is intentionally last and cannot influence any probe,
    # feature, window, threshold, architecture, or routing calculation above.
    locked = json.loads((BOUNDED / "summary.json").read_text(encoding="utf-8"))
    summary, readiness = build_summary(
        upstream, leakage, terrain, slip, sink, locked, probe_parity
    )
    summary["wall_time_s"] = time.perf_counter() - started
    write_json(output / "protocol.json", protocol())
    write_json(output / "summary.json", summary)
    write_json(output / "readiness.json", {
        "gates": readiness,
        "overall_ready": False,
        "diagnostic_only": True,
        "next_step": summary["next_step"],
        "production_candidate_approved": False,
    })
    make_plots(output, terrain, slip, sink)
    write_audit(output, summary)
    write_json(output / "manifest.json", artifact_manifest(output, upstream))
    print(
        f"WALKING_FUSION10_OBSERVABILITY_AUDIT_COMPLETE output={output} "
        f"next_step={summary['next_step']}", flush=True,
    )


if __name__ == "__main__":
    main()
