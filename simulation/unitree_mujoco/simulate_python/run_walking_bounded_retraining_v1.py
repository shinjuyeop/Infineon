"""Build, train, select, and holdout-test bounded walking float candidates.

Selection reads only existing static/controlled validation and d2209cd v02.
The fd5b9f0 v03-v05 outer traces and new Sink holdout are accessed only after
all three selection JSON files have been written and hashed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from run_terrain_fast_reflex_v2_detector import model as build_hazard_model
from run_walking_hazard_ground_truth_v1 import (
    DEFAULT_POLICY,
    PHYSICS_TIMESTEP_S,
    sink_candidate_profiles,
)
from run_walking_hazard_oracle_calibration_v1 import (
    RobustnessCondition,
    Variation,
    collect_run,
)
from terrain_cnn import FUSION_CHANNEL_NAMES, build_compact_1d_cnn
from walking_bounded_retraining_v1 import (
    HAZARD_PERSISTENCE,
    MIXTURE_RATIOS,
    TERRAIN_LABELS,
    TERRAIN_NAMES,
    TRAINING_SEEDS,
    Normalizer,
    causal_windows,
    classification_metrics,
    controlled_replay,
    physical_oracle,
    select_hazard,
    select_terrain,
    split_audit,
    stable_fire,
    walking_hazard_replay,
    weighted_normalizer,
)


ROOT = Path(__file__).resolve().parents[3]
SIM = ROOT / "simulation"
OUTPUT = SIM / "outputs" / "walking_bounded_retraining_v1"
SINK_HOLDOUT_OUTPUT = SIM / "outputs" / "walking_bounded_retraining_v1_sink_holdout"
STARTING_CHECKPOINT = "fd5b9f0baada107fb2d0811e04c29b3fbc2b412e"
DEVELOPMENT_CHECKPOINT = "d2209cdb49c496839a16396f201ab0322515171b"
DEV = SIM / "outputs" / "walking_hazard_oracle_calibration_v1"
OUTER = SIM / "outputs" / "walking_hazard_slip_nested_calibration_v2"
STATIC_DATA = SIM / "outputs" / "terrain_static_provenance_v4" / "dataset_noisy_provenance.npz"
STATIC_REFERENCE = SIM / "outputs" / "terrain_static_reference_v4"
HAZARD_DATA = SIM / "outputs" / "terrain_fast_reflex_v2_detector_dataset"
HAZARD_MODELS = {
    "slip": SIM / "outputs" / "terrain_fast_reflex_v2_detector_training_slip" / "slip_5ms" / "model.keras",
    "sink": SIM / "outputs" / "terrain_fast_reflex_v2_detector_training_sink" / "sink_20ms" / "model.keras",
}
SPATIAL = SIM / "outputs" / "walking_terrain_transition_v1_pilot"
WINDOWS = {"terrain": 50, "slip": 5, "sink": 20}
OLD_HAZARD_CONFIG = {
    "slip": {"threshold": 0.9719217419624332, "persistence": 3, "recall": 0.8235294117647058},
    "sink": {"threshold": 0.9836072999238968, "persistence": 1, "recall": 1.0},
}
IMMUTABLE_SHA256 = {
    "simulation/outputs/walking_hazard_ground_truth_v1_pilot/protocol.json": "04f8302cd8b8232de0bec2ef742287a9fd4ad4422ac8eb8a6cad9d49d76aac88",
    "simulation/outputs/walking_hazard_ground_truth_v1_pilot/summary.json": "96a4c29ce495aea1e7785ae95124652bf1017ac3d9618727d4ef17c9bc37aa10",
    "simulation/outputs/walking_hazard_ground_truth_v1_pilot/traces.npz": "cba2cf5f4ce915135a8607ee384e5b9b04d76a4ab2bbc0e3608b27e0b8d946d7",
    "simulation/outputs/walking_hazard_oracle_calibration_v1/protocol.json": "3e8f2c70a81b3e686f1967818981cada408ec2549abe31d5d57b53d919c224a3",
    "simulation/outputs/walking_hazard_oracle_calibration_v1/manifest.json": "b1360af75e6d57d59270df1bde3939b41d70a537067ae9754c044bda7a041aa8",
    "simulation/outputs/walking_hazard_oracle_calibration_v1/traces.npz": "1bbede7770400a844f75da2bce5157e4b50e5f8119ea5e893760943c7ed40423",
    "simulation/outputs/walking_hazard_oracle_calibration_v1/summary.json": "62d4da1dfb86b1ad2ef754624d256be594c0a199b5e916cc45ab044da7ee34bc",
    "simulation/outputs/walking_hazard_slip_nested_calibration_v2/protocol.json": "6d61335d2c861981f87b51fecb54061b7ad3b8157fd09687ca62e9fdd06aac92",
    "simulation/outputs/walking_hazard_slip_nested_calibration_v2/nested_selection.json": "72330c16397dc9aa2f416beb933518e22ce20f307f87e833c0399c198928896f",
    "simulation/outputs/walking_hazard_slip_nested_calibration_v2/outer_validation_manifest.json": "e04db778a49575ad7afbc6ee7c5f35a5b85a5ce0d7a7ef9664f4933020a138fd",
    "simulation/outputs/walking_hazard_slip_nested_calibration_v2/outer_validation_traces.npz": "4b7876eae0e7caa9411bcd106f01eedfeee6e342cde5e551e56ea52f5d6de1d3",
    "simulation/outputs/walking_hazard_slip_nested_calibration_v2/summary.json": "7a476799d72c40081d64a531f6ddd7e2591eb18d93d269fa74114bfa4f24970b",
    "simulation/outputs/terrain_static_provenance_v4/dataset_noisy_provenance.npz": "4174c4f04f0199a3ff91acbc491181eaa7ec6612cdf5a6ff876e1a619327b2a3",
    "simulation/outputs/terrain_static_reference_v4/selected_model.keras": "adcfa113b679327dc5bac0d4df7dcfceeb2f4dd703960665ecc491581d77df3b",
    "simulation/outputs/terrain_static_reference_v4/normalization.json": "13ffc43275e142b6e9bd17cd367a99ff019746087ab7f5ce79a7cea0b9a25830",
    "simulation/outputs/terrain_static_reference_v4/summary.json": "0569d9db3046fc3a80b85ddb40e6c266d697b223014e4a4e75ec1aace20d2fb4",
    "simulation/outputs/terrain_static_reference_v4/int8/gap_50_seed_20260921_strict_int8.tflite": "27d4da4d30c012307c895ea73636c6a17fa2bbb36d9507c0293d2a7fd7f4c943",
    "simulation/outputs/terrain_fast_reflex_v2_detector_dataset/slip_5ms/train.npz": "a76fb9fb52198f37d295f8684a83b2d0b0327ef727173fde1aab025afd10f7d4",
    "simulation/outputs/terrain_fast_reflex_v2_detector_dataset/slip_5ms/validation.npz": "e191843a5dc64ac1660f9da91d92992c77b5b2d1f9f6f864797e65e3271696fd",
    "simulation/outputs/terrain_fast_reflex_v2_detector_dataset/slip_5ms/normalization.json": "67f8f6256626d55fe0dd33a1631836234216dbcafe473a66df304554033a65df",
    "simulation/outputs/terrain_fast_reflex_v2_detector_dataset/sink_20ms/train.npz": "ec7d3228ee2ac579dc5b12b9a245bffaaf2d358c4be27eae24dddf13dfaf94a2",
    "simulation/outputs/terrain_fast_reflex_v2_detector_dataset/sink_20ms/validation.npz": "9a0a7f477dda3d0cb225f4704a9bbd5de71587cbdc496ef9cde040d80f6fda65",
    "simulation/outputs/terrain_fast_reflex_v2_detector_dataset/sink_20ms/normalization.json": "5949d27423d81bbe6ccfe586b35d6f9727717d5eb35bc4f07c91750a793848f1",
    "simulation/outputs/terrain_fast_reflex_v2_detector_training_slip/slip_5ms/model.keras": "4751c73be3047ad968caa16574f4bac7537369392435e928c7c8d2ad0051d377",
    "simulation/outputs/terrain_fast_reflex_v2_detector_training_sink/sink_20ms/model.keras": "8447152ffe17dcf75943aca6008fbab19b2b64a28c041ba3cc6573e329833453",
    "simulation/outputs/terrain_fast_reflex_v2_int8/slip/model_int8.tflite": "c414ef1135fc0d44f139960faba55027843d7a4d9574318478e2f49eae3db66d",
    "simulation/outputs/terrain_fast_reflex_v2_int8/sink/model_int8.tflite": "4dcccf97d5175876b6c5e6fb31dd2c74f67e4dd001042fd29d861c1aa8101948",
    "simulation/outputs/terrain_fast_reflex_v2_vela_e84/slip/model_int8_vela.tflite": "9e300a92e4d603c6c8b474ce236b3a7fab437796371b9680a65195dc5e0e9a4e",
    "simulation/outputs/terrain_fast_reflex_v2_vela_e84/sink/model_int8_vela.tflite": "8c57aabf196b116bbee9f527b84857ddc07e237c7bb8118c46adbbd53fbe0c7d",
    "simulation/outputs/terrain_fast_reflex_system_v1/protocol.json": "ae20439e86d6b6728a5226b2b241f25761d95a16f942b46f61d42a906a22b8be",
    "simulation/outputs/terrain_fast_reflex_system_v1/summary.json": "01cf0cb90f64751ce3f32de5ee2b89b843e10591d606443ca8e7f6250b88178e",
    "simulation/outputs/walking_domain_failure_audit_v1/protocol.json": "d10ed4141d65d7daa20cecc5e1ced049f211ece10ac99a9a49f4cb2ed2306e67",
    "simulation/outputs/walking_domain_failure_audit_v1/summary.json": "46b988d14034e55272528d1472904b40619bc479d59949ee79ad623508772649",
    "simulation/outputs/walking_terrain_transition_v1_pilot/protocol.json": "32572f39ee1e68c85704a272c721690c326c668f0f72610e5f7fd89855867a75",
    "simulation/outputs/walking_terrain_transition_v1_pilot/transition_traces.npz": "1b9ef299612bba42833203c91c780621c5d672e59fc57dbb4b09070853eea887",
    "simulation/outputs/walking_terrain_transition_v1_pilot/summary.json": "b05300449415923862ccaf69ffb4ebcd09e3f6848d5d12d256aa3b1e08a3ebd9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--sink-holdout-dir", type=Path, default=SINK_HOLDOUT_OUTPUT)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=keys, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def portable_candidate_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Remove ephemeral training-directory names from persistent audit tables."""
    portable = []
    for row in rows:
        item = dict(row)
        if item.get("candidate_model_path"):
            item["candidate_model_path"] = (
                f"training_workspace/{Path(str(item['candidate_model_path'])).name}"
            )
        if item.get("best_runtime_config_json"):
            config = json.loads(str(item["best_runtime_config_json"]))
            if config and config.get("candidate_model_path"):
                config["candidate_model_path"] = (
                    "training_workspace/"
                    f"{Path(str(config['candidate_model_path'])).name}"
                )
            item["best_runtime_config_json"] = json.dumps(
                config, sort_keys=True, default=str
            )
        portable.append(item)
    return portable


def verify_sources() -> dict[str, object]:
    actual = {name: sha256(ROOT / name) for name in IMMUTABLE_SHA256}
    mismatches = {
        name: {"expected": IMMUTABLE_SHA256[name], "actual": value}
        for name, value in actual.items() if value != IMMUTABLE_SHA256[name]
    }
    if mismatches:
        raise ValueError(f"immutable source mismatch: {mismatches}")
    return {
        "starting_checkpoint": STARTING_CHECKPOINT,
        "verified_files": actual,
        "mismatch_count": 0,
        "production_artifacts_modified": False,
    }


def load_trace_set(
    trace_path: Path, manifest_path: Path, expected_runs: int
) -> tuple[list[dict[str, object]], list[dict[str, np.ndarray]]]:
    manifest_value = json.loads(manifest_path.read_text())
    manifests = manifest_value["runs"] if isinstance(manifest_value, dict) else manifest_value
    with np.load(trace_path, allow_pickle=False) as packed:
        keys = [
            key for key in packed.files
            if packed[key].ndim >= 2 and packed[key].shape[0] == expected_runs
            and key not in {
                "slip_oracle_calibration_candidate", "sink_oracle_calibration_candidate",
                "slip_label_oracle_frozen_candidate",
            }
        ]
        traces = [
            {key: packed[key][index].copy() for key in keys}
            for index in range(expected_runs)
        ]
    if len(manifests) != expected_runs:
        raise ValueError("trace/manifest run count mismatch")
    return manifests, traces


def load_development() -> tuple[list[dict[str, object]], list[dict[str, np.ndarray]]]:
    manifests, traces = load_trace_set(DEV / "traces.npz", DEV / "manifest.json", 54)
    for row, trace in zip(manifests, traces):
        row["split"] = "walking_train" if int(row["variation_index"]) in (0, 1) else "walking_validation"
        row["contact_episode_ids"] = [
            int(value) for value in np.unique(trace["contact_episode_id"]) if value >= 0
        ]
    return manifests, traces


def _sample_balanced(endpoints: np.ndarray, keys: np.ndarray, cap: int = 64) -> np.ndarray:
    selected = []
    for key in np.unique(keys):
        group = endpoints[keys == key]
        if len(group) <= cap:
            selected.extend(group.tolist())
        else:
            positions = np.linspace(0, len(group) - 1, cap).round().astype(int)
            selected.extend(group[positions].tolist())
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def walking_windows(
    detector: str,
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
    split: str,
    sampled_train: bool,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    window = WINDOWS[detector]
    arrays: dict[str, list[object]] = {
        "x": [], "y": [], "run_id": [], "endpoint_sample": [], "variation_index": [],
        "speed": [], "terrain": [], "profile": [], "episode": [], "gait_phase": [],
        "pre_fall_valid": [], "physical_oracle_state": [], "source_checkpoint": [],
        "endpoint_time_s": [], "split_ownership": [],
    }
    aggregate: list[dict[str, object]] = []
    for metadata, trace in zip(manifests, traces):
        if metadata["split"] != split:
            continue
        prefall = np.asarray(trace["pre_fall_valid"], bool)
        transient = np.asarray(trace["touchdown_transient"], bool)
        if detector == "terrain":
            eligible = np.asarray(trace["loaded_contact"], bool) & prefall
            label = np.full(len(prefall), TERRAIN_LABELS[str(metadata["terrain_name"])], np.int8)
        else:
            eligible = prefall & ~transient
            label = physical_oracle(trace, detector).astype(np.int8)
        eligible[:window - 1] = False
        endpoints = np.flatnonzero(eligible)
        if sampled_train:
            phase = np.asarray(trace["gait_phase"]).astype(str)
            group_keys = np.asarray([
                f"{phase[value]}::{int(label[value])}" for value in endpoints
            ])
            endpoints = _sample_balanced(endpoints, group_keys, 64)
        if not endpoints.size:
            continue
        windows = causal_windows(trace["fusion10"], endpoints, window)
        arrays["x"].extend(windows)
        arrays["y"].extend(label[endpoints].tolist())
        for endpoint in endpoints:
            arrays["run_id"].append(metadata["run_id"])
            arrays["endpoint_sample"].append(int(endpoint))
            arrays["variation_index"].append(int(metadata["variation_index"]))
            arrays["speed"].append(float(metadata["walking_speed_mps"]))
            arrays["terrain"].append(str(metadata["terrain_name"]))
            arrays["profile"].append(str(metadata["profile_name"]))
            arrays["episode"].append(int(trace["contact_episode_id"][endpoint]))
            arrays["gait_phase"].append(str(trace["gait_phase"][endpoint]))
            arrays["pre_fall_valid"].append(bool(prefall[endpoint]))
            arrays["physical_oracle_state"].append(bool(label[endpoint]))
            arrays["source_checkpoint"].append(DEVELOPMENT_CHECKPOINT)
            arrays["endpoint_time_s"].append(float(trace["time_s"][endpoint]))
            arrays["split_ownership"].append(split)
        for phase_name in sorted(set(np.asarray(trace["gait_phase"])[endpoints].astype(str))):
            mask = np.asarray(trace["gait_phase"])[endpoints].astype(str) == phase_name
            aggregate.append({
                "detector": detector,
                "source_checkpoint": DEVELOPMENT_CHECKPOINT,
                "run_id": metadata["run_id"],
                "variation": metadata["variation_index"],
                "speed_mps": metadata["walking_speed_mps"],
                "terrain": metadata["terrain_name"],
                "profile": metadata["profile_name"],
                "gait_phase": phase_name,
                "split_ownership": split,
                "window_count": int(np.count_nonzero(mask)),
                "positive_windows": int(np.sum(label[endpoints][mask])),
                "window_ms": window,
                "window_semantics": "causal [t-L+1,t]",
            })
    result = {key: np.asarray(value) for key, value in arrays.items()}
    result["x"] = result["x"].astype(np.float32)
    result["y"] = result["y"].astype(np.int8)
    return result, aggregate


def raw_controlled(detector: str, split: str) -> dict[str, np.ndarray]:
    directory = HAZARD_DATA / f"{detector}_{WINDOWS[detector]}ms"
    with np.load(directory / f"{split}.npz", allow_pickle=False) as packed:
        result = {key: packed[key] for key in packed.files}
    normalizer = json.loads((directory / "normalization.json").read_text())
    mean = np.asarray(normalizer["mean"], np.float32)
    std = np.asarray(normalizer["std"], np.float32)
    result["raw_x"] = (result["x"] * std + mean).astype(np.float32)
    return result


def old_normalizer(path: Path) -> Normalizer:
    value = json.loads(path.read_text())
    return Normalizer(np.asarray(value["mean"], np.float32), np.asarray(value["std"], np.float32))


def adapt_first_convolution_normalization(
    model: object, previous: Normalizer, current: Normalizer
) -> None:
    """Preserve the first convolution's raw-signal transform after renormalizing."""
    layer = next(value for value in model.layers if value.__class__.__name__ == "Conv1D")
    kernel, bias = layer.get_weights()
    scale = current.std / previous.std
    offset = (current.mean - previous.mean) / previous.std
    adjusted_kernel = kernel * scale[None, :, None]
    adjusted_bias = bias + np.sum(kernel * offset[None, :, None], axis=(0, 1))
    layer.set_weights([adjusted_kernel, adjusted_bias])


def model_trace_scores(model: object, normalizer: Normalizer, trace: dict[str, np.ndarray], window: int) -> np.ndarray:
    endpoints = np.arange(window - 1, len(trace["fusion10"]), dtype=np.int64)
    values = normalizer.transform(causal_windows(trace["fusion10"], endpoints, window))
    predicted = np.asarray(model.predict(values, batch_size=2048, verbose=0))
    probability = predicted.reshape(-1) if predicted.shape[-1] == 1 else predicted
    output_shape = (len(trace["fusion10"]),) if probability.ndim == 1 else (len(trace["fusion10"]), probability.shape[1])
    output = np.full(output_shape, np.nan, dtype=np.float32)
    output[endpoints] = probability
    return output


def sample_weights(static_count: int, walking_count: int, ratio: float, labels: np.ndarray) -> np.ndarray:
    weights = np.r_[
        np.full(static_count, 1.0 / static_count),
        np.full(walking_count, ratio / walking_count),
    ]
    for label in np.unique(labels):
        mask = labels == label
        weights[mask] *= (1.0 + ratio) / (len(np.unique(labels)) * weights[mask].sum())
    return weights.astype(np.float32)


def protocol(epochs: int) -> dict[str, object]:
    return {
        "dataset": "walking_bounded_retraining_v1",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "scope": "bounded float candidates only; production artifacts remain immutable",
        "source_boundaries": {
            "model_development": "existing static/controlled train+validation and d2209cd v00/v01 train, v02 validation",
            "holdout": "fd5b9f0 v03/v04/v05 plus new Sink v03/v04/v05 after selection lock",
            "test_or_final_used": False,
            "holdout_used_for_normalization_training_balancing_threshold_or_persistence": False,
        },
        "fusion_contract": {"sample_rate_hz": 1000, "channels": list(FUSION_CHANNEL_NAMES)},
        "architectures": {
            "terrain": "existing Terrain v4 50x10 Conv12/Conv16/GAP/4-way softmax",
            "slip": "existing Fast Reflex v2 5x10 Conv12/Conv16/GAP+GlobalMax/sigmoid",
            "sink": "existing Fast Reflex v2 20x10 Conv12/Conv16/GAP/sigmoid",
        },
        "walking_labels": {
            "terrain": "homogeneous physical material at loaded-contact endpoint; AIR creates no label",
            "slip": "0.050 m contact-anchor drift, 3 ms, loaded/post-touchdown/pre-fall",
            "sink": "0.0055 m loaded penetration change, 20 ms, loaded/post-touchdown/pre-fall",
            "scenario_or_terrain_hazard_labels": False,
            "causal_window": "label at t uses only Fusion10 [t-L+1,t]",
            "air_negative": True,
            "post_fall_excluded": True,
            "touchdown_transient_excluded": True,
        },
        "mixture_grid": {
            "walking_effective_weight_relative_to_existing_train": list(MIXTURE_RATIOS),
            "candidate_count_per_detector": len(MIXTURE_RATIOS),
            "walking_sampling": "at most 64 unique endpoints per run/gait-phase/label; no duplication",
        },
        "training": {
            "seeds": list(TRAINING_SEEDS),
            "epochs": epochs,
            "batch_size": 256,
            "early_stopping": "validation loss patience 6, restore best weights",
            "terrain_epoch_checkpoint_rule": (
                "compare normalization-adapted epoch 0 with the early-stopped final "
                "checkpoint; among checkpoints retaining static accuracy within 1pp, "
                "maximize walking validation macro accuracy, then static accuracy"
            ),
            "normalization": "combined train-only weighted moments per mixture",
            "class_balance": "train-only effective class sample weights",
            "initialization": "corresponding immutable deployed float weights; bounded fine-tuning",
            "learning_rate": {"terrain": 0.00001, "slip": 0.001, "sink": 0.001},
        },
        "hazard_selection_grid": {
            "thresholds": "41 quantiles of controlled+d2209cd-v02 validation scores plus 0.5 and frozen baseline threshold",
            "persistence_samples": list(HAZARD_PERSISTENCE),
            "declared_before_holdout": True,
        },
        "selection_priority": [
            "all safety/integrity gates", "walking normal FPR", "walking positive run recall",
            "static/controlled validation retention", "p95 stable-firing latency",
            "smaller unchanged architecture", "mixture then seed deterministic tie-break",
        ],
        "no_full-pass_fallback": (
            "still freeze exactly one diagnostic float candidate by maximum mandatory-gate "
            "count, then minimum walking FP/anticipation, recall, retention, latency; keep "
            "candidate/readiness gates false and never promote to production"
        ),
        "selection_gates": {
            "terrain_static_accuracy_drop_max_percentage_point": 1.0,
            "hazard_controlled_overall_causal_run_fpr_max": 0.05,
            "hazard_controlled_recall_drop_max": 0.05,
            "walking_normal_false_positive_runs": 0,
            "walking_anticipation_runs": 0,
            "invalid_mask_firing_samples": 0,
        },
        "holdout_failure_policy": "never reselect or adjust model, mixture, threshold, or persistence",
        "forbidden_changes": [
            "production/frozen model", "production probability threshold", "INT8", "Vela/E84/U55",
            "board flash/HIL", "System-v1", "one-shot final test", "physical oracle",
        ],
    }


def terrain_validation_details(
    model: object,
    normalizer: Normalizer,
    static_x: np.ndarray,
    static_y: np.ndarray,
    walking: dict[str, np.ndarray],
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    static_prediction = np.argmax(
        model.predict(normalizer.transform(static_x), batch_size=2048, verbose=0), axis=1
    )
    walking_prediction = np.argmax(
        model.predict(normalizer.transform(walking["x"]), batch_size=2048, verbose=0), axis=1
    )
    static_metrics = classification_metrics(static_y, static_prediction)
    walking_metrics = classification_metrics(walking["y"], walking_prediction)
    phases = []
    for phase in sorted(set(walking["gait_phase"].astype(str))):
        mask = walking["gait_phase"].astype(str) == phase
        phases.append({
            "gait_phase": phase,
            "support": int(np.count_nonzero(mask)),
            "accuracy": float(np.mean(walking_prediction[mask] == walking["y"][mask])),
        })
    return static_metrics, walking_metrics, phases


def train_terrain_candidates(
    temporary: Path,
    epochs: int,
    static_train_x: np.ndarray,
    static_train_y: np.ndarray,
    static_validation_x: np.ndarray,
    static_validation_y: np.ndarray,
    walking_train: dict[str, np.ndarray],
    walking_validation: dict[str, np.ndarray],
    baseline_accuracy: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[tuple[float, int], Normalizer]]:
    import tensorflow as tf

    rows = []
    seed_rows = []
    normalizers = {}
    validation_x_raw = np.concatenate([static_validation_x, walking_validation["x"]])
    validation_y = np.concatenate([static_validation_y, walking_validation["y"]])
    frozen = tf.keras.models.load_model(STATIC_REFERENCE / "selected_model.keras", compile=False)
    previous_normalizer = old_normalizer(STATIC_REFERENCE / "normalization.json")
    for ratio in MIXTURE_RATIOS:
        normalizer = weighted_normalizer((static_train_x, walking_train["x"]), (1.0, ratio))
        train_x = normalizer.transform(np.concatenate([static_train_x, walking_train["x"]]))
        train_y = np.concatenate([static_train_y, walking_train["y"]])
        weights = sample_weights(len(static_train_y), len(walking_train["y"]), ratio, train_y)
        validation_x = normalizer.transform(validation_x_raw)
        for seed in TRAINING_SEEDS:
            tf.keras.backend.clear_session()
            candidate = build_compact_1d_cnn(10, seed, time_steps=50)
            candidate.set_weights(frozen.get_weights())
            adapt_first_convolution_normalization(candidate, previous_normalizer, normalizer)
            candidate.compile(
                optimizer=tf.keras.optimizers.Adam(1e-5),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            initial_static, initial_walking, initial_phases = terrain_validation_details(
                candidate, normalizer, static_validation_x, static_validation_y,
                walking_validation,
            )
            best_weights = candidate.get_weights()
            best_epoch = 0
            best_key = (
                float(initial_walking["macro_accuracy"]),
                float(initial_static["accuracy"]),
                0,
            ) if float(initial_static["accuracy"]) >= baseline_accuracy - 0.01 else None
            best_metrics = (initial_static, initial_walking, initial_phases)
            history = candidate.fit(
                train_x, train_y, sample_weight=weights,
                validation_data=(validation_x, validation_y),
                epochs=epochs, batch_size=256, verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=6, restore_best_weights=True
                )],
            )
            epochs_completed = len(history.history["loss"])
            current_static, current_walking, current_phases = terrain_validation_details(
                candidate, normalizer, static_validation_x, static_validation_y,
                walking_validation,
            )
            current_key = (
                float(current_walking["macro_accuracy"]),
                float(current_static["accuracy"]),
                -epochs_completed,
            )
            if (
                float(current_static["accuracy"]) >= baseline_accuracy - 0.01
                and (best_key is None or current_key > best_key)
            ):
                best_weights = candidate.get_weights()
                best_epoch = epochs_completed
                best_metrics = (current_static, current_walking, current_phases)
            candidate.set_weights(best_weights)
            model_path = temporary / f"terrain_r{ratio:.2f}_s{seed}.keras"
            candidate.save(model_path)
            static_metrics, walking_metrics, phases = best_metrics
            coverage = all(
                walking_metrics["class_metrics"][name]["support"] > 0 for name in TERRAIN_NAMES
            )
            gate = bool(
                float(static_metrics["accuracy"]) >= baseline_accuracy - 0.01
                and coverage
            )
            row = {
                "detector": "terrain", "mixture_ratio": ratio, "training_seed": seed,
                "parameters": int(candidate.count_params()), "epochs_completed": epochs_completed,
                "selected_epoch": best_epoch,
                "static_validation_accuracy": static_metrics["accuracy"],
                "static_validation_macro_accuracy": static_metrics["macro_accuracy"],
                "static_baseline_accuracy": baseline_accuracy,
                "static_accuracy_delta": float(static_metrics["accuracy"]) - baseline_accuracy,
                "walking_validation_accuracy": walking_metrics["accuracy"],
                "walking_validation_macro_accuracy": walking_metrics["macro_accuracy"],
                "walking_class_recall_json": json.dumps(walking_metrics["class_metrics"], sort_keys=True),
                "walking_phase_accuracy_json": json.dumps(phases, sort_keys=True),
                "all_terrain_classes_covered": coverage,
                "candidate_gate_pass": gate,
                "candidate_model_path": str(model_path),
                "candidate_model_sha256": sha256(model_path),
            }
            rows.append(row)
            seed_rows.append({
                **row,
                "train_existing_windows": len(static_train_y),
                "train_walking_windows": len(walking_train["y"]),
                "walking_effective_weight": ratio,
                "train_class_count_json": json.dumps({
                    name: int(np.count_nonzero(train_y == label))
                    for name, label in TERRAIN_LABELS.items()
                }, sort_keys=True),
            })
            normalizers[(ratio, seed)] = normalizer
            print(
                f"terrain ratio={ratio:.2f} seed={seed} static={static_metrics['accuracy']:.4f} "
                f"walking_macro={walking_metrics['macro_accuracy']:.4f} gate={gate}", flush=True,
            )
    return rows, seed_rows, normalizers


def hazard_thresholds(scores: np.ndarray, frozen: float) -> list[float]:
    finite = np.asarray(scores, float)
    finite = finite[np.isfinite(finite)]
    values = np.quantile(finite, np.linspace(0.0, 1.0, 41))
    return sorted(set(float(value) for value in np.r_[values, 0.5, frozen]))


def score_walking_traces(
    model: object,
    normalizer: Normalizer,
    traces: list[dict[str, np.ndarray]],
    window: int,
) -> tuple[list[np.ndarray], float]:
    started = time.perf_counter()
    result = [model_trace_scores(model, normalizer, trace, window) for trace in traces]
    milliseconds = (time.perf_counter() - started) * 1000.0 / sum(
        len(trace["fusion10"]) - window + 1 for trace in traces
    )
    return result, milliseconds


def positive_profile_gate(detector: str, metrics: dict[str, object]) -> bool:
    profiles = metrics["profile_results"]
    if detector == "slip":
        return bool(
            profiles and all(
                int(value["physical_positive_runs"]) == int(value["detected_runs"])
                and value["detected_speeds_mps"] == [0.1, 0.15, 0.2]
                for value in profiles.values()
            )
        )
    return any(
        int(value["physical_positive_runs"]) == int(value["detected_runs"])
        and value["detected_speeds_mps"] == [0.1, 0.15, 0.2]
        for value in profiles.values()
    )


def train_hazard_candidates(
    detector: str,
    temporary: Path,
    epochs: int,
    controlled_train: dict[str, np.ndarray],
    controlled_validation: dict[str, np.ndarray],
    walking_train: dict[str, np.ndarray],
    walking_validation: dict[str, np.ndarray],
    validation_manifests: list[dict[str, object]],
    validation_traces: list[dict[str, np.ndarray]],
    controlled_baseline: dict[str, object],
) -> tuple[
    list[dict[str, object]], list[dict[str, object]],
    dict[tuple[float, int], Normalizer], dict[tuple[float, int], float]
]:
    import tensorflow as tf

    rows = []
    seed_rows = []
    normalizers = {}
    compute_times = {}
    window = WINDOWS[detector]
    frozen = tf.keras.models.load_model(HAZARD_MODELS[detector], compile=False)
    previous_normalizer = old_normalizer(
        HAZARD_DATA / f"{detector}_{window}ms" / "normalization.json"
    )
    for ratio in MIXTURE_RATIOS:
        normalizer = weighted_normalizer(
            (controlled_train["raw_x"], walking_train["x"]), (1.0, ratio)
        )
        train_x = normalizer.transform(np.concatenate([
            controlled_train["raw_x"], walking_train["x"]
        ]))
        train_y = np.concatenate([controlled_train["y"], walking_train["y"]])
        weights = sample_weights(
            len(controlled_train["y"]), len(walking_train["y"]), ratio, train_y
        )
        validation_x = normalizer.transform(np.concatenate([
            controlled_validation["raw_x"], walking_validation["x"]
        ]))
        validation_y = np.concatenate([
            controlled_validation["y"], walking_validation["y"]
        ])
        for seed in TRAINING_SEEDS:
            tf.keras.backend.clear_session()
            candidate = build_hazard_model(
                window, "average_max" if detector == "slip" else "average", seed
            )
            candidate.set_weights(frozen.get_weights())
            adapt_first_convolution_normalization(candidate, previous_normalizer, normalizer)
            candidate.compile(
                optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy"
            )
            history = candidate.fit(
                train_x, train_y, sample_weight=weights,
                validation_data=(validation_x, validation_y),
                epochs=epochs, batch_size=256, verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=6, restore_best_weights=True
                )],
            )
            model_path = temporary / f"{detector}_r{ratio:.2f}_s{seed}.keras"
            candidate.save(model_path)
            control_scores = candidate.predict(
                normalizer.transform(controlled_validation["raw_x"]),
                batch_size=2048, verbose=0,
            ).reshape(-1)
            walking_scores, compute_ms = score_walking_traces(
                candidate, normalizer, validation_traces, window
            )
            thresholds = hazard_thresholds(
                np.r_[control_scores, np.concatenate(walking_scores)],
                OLD_HAZARD_CONFIG[detector]["threshold"],
            )
            valid_config_count = 0
            best_model_config: dict[str, object] | None = None
            for threshold in thresholds:
                for persistence in HAZARD_PERSISTENCE:
                    _, control = controlled_replay(
                        controlled_validation["run_id"], controlled_validation["endpoint_ms"],
                        controlled_validation["y"], control_scores, threshold, persistence,
                    )
                    _, walking = walking_hazard_replay(
                        detector, validation_manifests, validation_traces,
                        walking_scores, threshold, persistence,
                    )
                    gate = bool(
                        float(control["overall_causal_run_fpr"]) <= 0.05
                        and float(control["run_recall"]) >= float(controlled_baseline["run_recall"]) - 0.05
                        and int(walking["normal_false_positive_runs"]) == 0
                        and int(walking["anticipation_runs"]) == 0
                        and int(walking["label_mask_violation_samples"]) == 0
                        and positive_profile_gate(detector, walking)
                    )
                    profile_gate = positive_profile_gate(detector, walking)
                    row = {
                        "detector": detector, "mixture_ratio": ratio, "training_seed": seed,
                        "parameters": int(candidate.count_params()),
                        "probability_threshold": threshold,
                        "runtime_persistence": persistence,
                        "controlled_validation_causal_run_fpr": control["overall_causal_run_fpr"],
                        "controlled_validation_run_recall": control["run_recall"],
                        "controlled_baseline_run_recall": controlled_baseline["run_recall"],
                        "walking_normal_false_positive_runs": walking["normal_false_positive_runs"],
                        "walking_positive_run_recall": walking["positive_run_recall"],
                        "walking_detected_positive_runs": walking["detected_positive_runs"],
                        "walking_physical_positive_runs": walking["physical_positive_runs"],
                        "walking_anticipation_runs": walking["anticipation_runs"],
                        "walking_invalid_firing_samples": walking["label_mask_violation_samples"],
                        "walking_p95_stable_latency_ms": walking["latency_p95_ms"],
                        "walking_profile_results_json": json.dumps(walking["profile_results"], sort_keys=True),
                        "positive_profile_gate_pass": profile_gate,
                        "inference_compute_ms_per_window": compute_ms,
                        "candidate_gate_pass": gate,
                        "candidate_model_path": str(model_path),
                        "candidate_model_sha256": sha256(model_path),
                    }
                    rows.append(row)
                    if gate:
                        valid_config_count += 1
                        if best_model_config is None or select_hazard([best_model_config, row]) is row:
                            best_model_config = row
            seed_rows.append({
                "detector": detector, "mixture_ratio": ratio, "training_seed": seed,
                "parameters": int(candidate.count_params()), "epochs_completed": len(history.history["loss"]),
                "train_existing_windows": len(controlled_train["y"]),
                "train_walking_windows": len(walking_train["y"]),
                "existing_train_positive_windows": int(np.sum(controlled_train["y"])),
                "walking_train_positive_windows": int(np.sum(walking_train["y"])),
                "walking_effective_weight": ratio,
                "passing_runtime_configs": valid_config_count,
                "best_runtime_config_json": json.dumps(best_model_config, sort_keys=True, default=str),
                "candidate_model_sha256": sha256(model_path),
            })
            normalizers[(ratio, seed)] = normalizer
            compute_times[(ratio, seed)] = compute_ms
            print(
                f"{detector} ratio={ratio:.2f} seed={seed} configs={len(thresholds)*len(HAZARD_PERSISTENCE)} "
                f"passing={valid_config_count}", flush=True,
            )
    return rows, seed_rows, normalizers, compute_times


def diagnostic_hazard_candidate(
    detector: str, rows: list[dict[str, object]], controlled_baseline: dict[str, object]
) -> dict[str, object]:
    def key(row: dict[str, object]) -> tuple[object, ...]:
        checks = (
            float(row["controlled_validation_causal_run_fpr"]) <= 0.05,
            float(row["controlled_validation_run_recall"])
            >= float(controlled_baseline["run_recall"]) - 0.05,
            int(row["walking_normal_false_positive_runs"]) == 0,
            int(row["walking_anticipation_runs"]) == 0,
            int(row["walking_invalid_firing_samples"]) == 0,
            bool(row["positive_profile_gate_pass"]),
        )
        latency = row["walking_p95_stable_latency_ms"]
        return (
            sum(checks),
            -int(row["walking_normal_false_positive_runs"]),
            -int(row["walking_anticipation_runs"]),
            float(row["walking_positive_run_recall"]),
            float(row["controlled_validation_run_recall"]),
            -float(row["controlled_validation_causal_run_fpr"]),
            -(float(latency) if latency is not None else 1e12),
            -float(row["mixture_ratio"]),
            -int(row["training_seed"]),
            float(row["probability_threshold"]),
            -int(row["runtime_persistence"]),
        )
    selected = max(rows, key=key)
    selected["diagnostic_fallback_selected"] = True
    selected["detector"] = detector
    return selected


def baseline_replay(
    dev_manifests: list[dict[str, object]],
    dev_traces: list[dict[str, np.ndarray]],
    terrain_validation: dict[str, np.ndarray],
    controlled_validation: dict[str, dict[str, np.ndarray]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    import tensorflow as tf

    rows: list[dict[str, object]] = []
    aggregate: dict[str, object] = {}
    terrain_model = tf.keras.models.load_model(STATIC_REFERENCE / "selected_model.keras", compile=False)
    terrain_norm = old_normalizer(STATIC_REFERENCE / "normalization.json")
    terrain_prediction = np.argmax(
        terrain_model.predict(terrain_norm.transform(terrain_validation["x"]), batch_size=2048, verbose=0), axis=1
    )
    terrain_metrics = classification_metrics(terrain_validation["y"], terrain_prediction)
    aggregate["terrain_walking_validation"] = terrain_metrics
    for run_id in sorted(set(terrain_validation["run_id"].astype(str))):
        mask = terrain_validation["run_id"].astype(str) == run_id
        rows.append({
            "detector": "terrain", "split": "walking_validation", "run_id": run_id,
            "endpoints": int(np.count_nonzero(mask)),
            "accuracy": float(np.mean(terrain_prediction[mask] == terrain_validation["y"][mask])),
            "normal_false_positive": "", "detected": "", "anticipation": "",
        })
    validation_pairs = [
        (row, trace) for row, trace in zip(dev_manifests, dev_traces)
        if row["split"] == "walking_validation"
    ]
    validation_manifests = [value[0] for value in validation_pairs]
    validation_traces = [value[1] for value in validation_pairs]
    for detector in ("slip", "sink"):
        model = tf.keras.models.load_model(HAZARD_MODELS[detector], compile=False)
        normalizer = old_normalizer(
            HAZARD_DATA / f"{detector}_{WINDOWS[detector]}ms" / "normalization.json"
        )
        walking_scores, _ = score_walking_traces(
            model, normalizer, validation_traces, WINDOWS[detector]
        )
        replay_rows, walking_metrics = walking_hazard_replay(
            detector, validation_manifests, validation_traces, walking_scores,
            float(OLD_HAZARD_CONFIG[detector]["threshold"]),
            int(OLD_HAZARD_CONFIG[detector]["persistence"]),
        )
        control_scores = model.predict(
            controlled_validation[detector]["x"], batch_size=2048, verbose=0
        ).reshape(-1)
        _, controlled_metrics = controlled_replay(
            controlled_validation[detector]["run_id"],
            controlled_validation[detector]["endpoint_ms"],
            controlled_validation[detector]["y"], control_scores,
            float(OLD_HAZARD_CONFIG[detector]["threshold"]),
            int(OLD_HAZARD_CONFIG[detector]["persistence"]),
        )
        aggregate[f"{detector}_walking_validation"] = walking_metrics
        aggregate[f"{detector}_controlled_validation"] = controlled_metrics
        for row in replay_rows:
            rows.append({
                "detector": detector, "split": "walking_validation", "run_id": row["run_id"],
                "endpoints": 3000, "accuracy": "",
                "normal_false_positive": row["normal_false_positive"],
                "detected": row["model_detected_post_onset"],
                "anticipation": row["anticipation_false_alarm"],
            })
    return rows, aggregate


def selected_payload(
    detector: str,
    selected: dict[str, object],
    normalizer: Normalizer,
    output: Path,
) -> dict[str, object]:
    model_directory = output / "models"
    normalization_directory = output / "normalization"
    model_directory.mkdir(exist_ok=True)
    normalization_directory.mkdir(exist_ok=True)
    model_path = model_directory / f"{detector}_walking_candidate.keras"
    normalization_path = normalization_directory / f"{detector}.json"
    shutil.copy2(Path(str(selected["candidate_model_path"])), model_path)
    write_json(normalization_path, normalizer.as_dict())
    payload = {
        "detector": detector,
        "selection_data": "existing validation + d2209cd v02 only",
        "holdout_runs_accessed_before_selection": 0,
        "architecture_unchanged": True,
        "window_ms": WINDOWS[detector],
        "mixture_ratio": selected["mixture_ratio"],
        "training_seed": selected["training_seed"],
        "model_path": str(model_path.relative_to(output)),
        "model_sha256": sha256(model_path),
        "normalization_path": str(normalization_path.relative_to(output)),
        "normalization_sha256": sha256(normalization_path),
        "parameters": selected["parameters"],
        "candidate_gate_pass": selected["candidate_gate_pass"],
        "selection_rationale": (
            "all mandatory gates, then walking normal FPR, walking positive recall, "
            "static/controlled retention, p95 latency, unchanged model size, deterministic ties"
        ),
        "production_artifact_replaced": False,
    }
    if detector == "terrain":
        payload.update({
            "selected_probability_threshold": None,
            "selected_runtime_persistence": 3,
            "static_validation_accuracy": selected["static_validation_accuracy"],
            "walking_validation_accuracy": selected["walking_validation_accuracy"],
            "walking_validation_macro_accuracy": selected["walking_validation_macro_accuracy"],
            "walking_class_recall_json": selected["walking_class_recall_json"],
        })
    else:
        payload.update({
            "selected_probability_threshold": selected["probability_threshold"],
            "selected_runtime_persistence": selected["runtime_persistence"],
            "controlled_validation_causal_run_fpr": selected["controlled_validation_causal_run_fpr"],
            "controlled_validation_run_recall": selected["controlled_validation_run_recall"],
            "walking_normal_false_positive_runs": selected["walking_normal_false_positive_runs"],
            "walking_positive_run_recall": selected["walking_positive_run_recall"],
            "walking_anticipation_runs": selected["walking_anticipation_runs"],
            "walking_p95_stable_latency_ms": selected["walking_p95_stable_latency_ms"],
            "diagnostic_fallback_selected": bool(
                selected.get("diagnostic_fallback_selected", False)
            ),
            "validation_gate_failure_reasons": ([] if selected["candidate_gate_pass"] else [
                "no runtime configuration for this trained candidate passed every mandatory validation gate"
            ]),
        })
    return payload


def save_window_provenance(
    path: Path, datasets: dict[str, dict[str, np.ndarray]]
) -> dict[str, object]:
    packed = {}
    for name, data in datasets.items():
        for key, value in data.items():
            if key != "x":
                packed[f"{name}__{key}"] = value
    np.savez_compressed(path, **packed)
    return {
        "path": path.name,
        "sha256": sha256(path),
        "arrays": {
            name: {key: list(value.shape) for key, value in data.items() if key != "x"}
            for name, data in datasets.items()
        },
        "every_walking_window_provenance": [
            "source_checkpoint", "run_id", "variation_index", "speed", "terrain", "profile",
            "episode", "gait_phase", "endpoint_sample", "endpoint_time_s", "pre_fall_valid",
            "physical_oracle_state", "split_ownership",
        ],
    }


def load_candidate(output: Path, detector: str) -> tuple[object, Normalizer, dict[str, object]]:
    import tensorflow as tf

    selection = json.loads((output / f"{detector}_selection.json").read_text())
    model = tf.keras.models.load_model(output / selection["model_path"], compile=False)
    normalizer = old_normalizer(output / selection["normalization_path"])
    return model, normalizer, selection


def terrain_holdout(
    model: object,
    normalizer: Normalizer,
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
    source: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    all_truth = []
    all_prediction = []
    for metadata, trace in zip(manifests, traces):
        endpoint = np.flatnonzero(
            np.asarray(trace["loaded_contact"], bool)
            & np.asarray(trace["pre_fall_valid"], bool)
        )
        endpoint = endpoint[endpoint >= WINDOWS["terrain"] - 1]
        windows = normalizer.transform(causal_windows(
            trace["fusion10"], endpoint, WINDOWS["terrain"]
        ))
        prediction = np.argmax(model.predict(windows, batch_size=2048, verbose=0), axis=1)
        truth = np.full(len(endpoint), TERRAIN_LABELS[str(metadata["terrain_name"])], np.int8)
        all_truth.extend(truth.tolist())
        all_prediction.extend(prediction.tolist())
        for phase in sorted(set(np.asarray(trace["gait_phase"])[endpoint].astype(str))):
            mask = np.asarray(trace["gait_phase"])[endpoint].astype(str) == phase
            rows.append({
                "detector": "terrain", "source": source, "run_id": metadata["run_id"],
                "terrain": metadata["terrain_name"], "profile": metadata["profile_name"],
                "speed_mps": metadata["walking_speed_mps"], "variation": metadata["variation_index"],
                "gait_phase": phase, "endpoints": int(np.count_nonzero(mask)),
                "accuracy": float(np.mean(prediction[mask] == truth[mask])),
            })
    return rows, classification_metrics(
        np.asarray(all_truth, np.int8), np.asarray(all_prediction, np.int8)
    )


def hazard_holdout(
    detector: str,
    model: object,
    normalizer: Normalizer,
    selection: dict[str, object],
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
    included_roles: set[str],
    source: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    selected_pairs = [
        (row, trace) for row, trace in zip(manifests, traces)
        if str(row["acquisition_role"]) in included_roles
    ]
    selected_manifests = [value[0] for value in selected_pairs]
    selected_traces = [value[1] for value in selected_pairs]
    scores, compute_ms = score_walking_traces(
        model, normalizer, selected_traces, WINDOWS[detector]
    )
    replay_rows, metrics = walking_hazard_replay(
        detector, selected_manifests, selected_traces, scores,
        float(selection["selected_probability_threshold"]),
        int(selection["selected_runtime_persistence"]),
    )
    latency_rows = []
    endpoint_truth = []
    endpoint_threshold_prediction = []
    endpoint_stable_prediction = []
    for metadata, trace, probability, base in zip(
        selected_manifests, selected_traces, scores, replay_rows
    ):
        eligible = (
            np.asarray(trace["loaded_contact"], bool)
            & np.asarray(trace["pre_fall_valid"], bool)
            & ~np.asarray(trace["touchdown_transient"], bool)
        )
        fire = stable_fire(
            probability, float(selection["selected_probability_threshold"]),
            int(selection["selected_runtime_persistence"]), eligible,
        )
        physical = physical_oracle(trace, detector)
        endpoint_truth.extend(physical[eligible].astype(bool).tolist())
        endpoint_threshold_prediction.extend(
            (np.asarray(probability)[eligible] >= float(selection["selected_probability_threshold"])).tolist()
        )
        endpoint_stable_prediction.extend(fire[eligible].tolist())
        model_indices = np.flatnonzero(fire)
        physical_indices = np.flatnonzero(physical)
        physical_fire = None if not physical_indices.size else int(physical_indices[0])
        first_model = None if not model_indices.size else int(model_indices[0])
        post_model = None if physical_fire is None else next(
            (int(value) for value in model_indices if value >= physical_fire), None
        )
        chosen = first_model if physical_fire is None else post_model
        threshold_crossing = (
            None if chosen is None else chosen - int(selection["selected_runtime_persistence"]) + 1
        )
        episode = None if physical_fire is None else int(trace["contact_episode_id"][physical_fire])
        physical_valid_onset = None
        touchdown = None
        if episode is not None:
            valid_name = f"{detector}_calibration_valid"
            valid_indices = np.flatnonzero(
                trace[valid_name] & (trace["contact_episode_id"] == episode)
            )
            contact_indices = np.flatnonzero(
                trace["left_contact"] & (trace["contact_episode_id"] == episode)
            )
            physical_valid_onset = None if not valid_indices.size else int(valid_indices[0])
            touchdown = None if not contact_indices.size else int(contact_indices[0])
        latency_rows.append({
            "detector": detector, "source": source, "run_id": metadata["run_id"],
            "profile": metadata["profile_name"], "speed_mps": metadata["walking_speed_mps"],
            "variation": metadata["variation_index"],
            "physical_episode_valid_onset_sample": physical_valid_onset,
            "physical_oracle_fire_sample": physical_fire,
            "model_probability_threshold_crossing_sample": threshold_crossing,
            "model_stable_fire_sample": chosen,
            "touchdown_sample": touchdown,
            "physical_valid_onset_to_oracle_fire_ms": (
                None if physical_fire is None else physical_fire - int(physical_valid_onset)
            ),
            "physical_oracle_fire_to_model_positive_ms": (
                None if post_model is None else post_model - int(physical_fire)
            ),
            "probability_crossing_to_stable_fire_ms": (
                None if chosen is None else chosen - int(threshold_crossing) + 1
            ),
            "touchdown_to_model_stable_fire_ms": (
                None if chosen is None or touchdown is None else chosen - touchdown
            ),
            "model_inference_compute_ms_per_window": compute_ms,
            "anticipation_false_alarm": base["anticipation_false_alarm"],
            "fall_occurred": metadata["fall_occurred"],
            "pre_fall_only_evaluation": True,
            "fast_reflex_target_ms": 20,
        })
    for row in replay_rows:
        row["source"] = source
        row["model_inference_compute_ms_per_window"] = compute_ms
    metrics["model_inference_compute_ms_per_window"] = compute_ms
    truth = np.asarray(endpoint_truth, bool)
    for name, prediction_values in (
        ("probability_threshold", endpoint_threshold_prediction),
        ("stable_persistence", endpoint_stable_prediction),
    ):
        prediction = np.asarray(prediction_values, bool)
        true_positive = int(np.count_nonzero(prediction & truth))
        false_positive = int(np.count_nonzero(prediction & ~truth))
        false_negative = int(np.count_nonzero(~prediction & truth))
        true_negative = int(np.count_nonzero(~prediction & ~truth))
        metrics[f"{name}_endpoint_metrics"] = {
            "support": len(truth),
            "positive_support": int(np.count_nonzero(truth)),
            "precision": true_positive / max(true_positive + false_positive, 1),
            "recall": true_positive / max(true_positive + false_negative, 1),
            "false_positive_rate": false_positive / max(false_positive + true_negative, 1),
        }
    return replay_rows, latency_rows, metrics


def sink_holdout_conditions() -> list[RobustnessCondition]:
    profiles = {
        profile.name: profile for profile in sink_candidate_profiles()
        if profile.name in {
            "sand_solref_interpolation_1of3", "sand_solref_interpolation_2of3"
        }
    }
    variations = (
        Variation(3, 202608193, 1.0 / 6.0, 0.060),
        Variation(4, 202608194, 1.0 / 2.0, 0.080),
        Variation(5, 202608195, 5.0 / 6.0, 0.100),
    )
    return [
        RobustnessCondition(name, "sand", profiles[name], "sink_candidate", speed, variation)
        for variation in variations
        for speed in (0.10, 0.15, 0.20)
        for name in sorted(profiles)
    ]


def collect_sink_holdout(
    output: Path,
    policy_path: Path,
    selection_hashes: dict[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, np.ndarray]]]:
    conditions = sink_holdout_conditions()
    output.mkdir(parents=True)
    write_json(output / "protocol.json", {
        "dataset": "walking_bounded_retraining_v1_sink_holdout",
        "created_after_all_detector_selection_artifacts": True,
        "selection_artifact_sha256_before_collection": selection_hashes,
        "selection_adjustment_after_holdout_forbidden": True,
        "sample_rate_hz": 1000,
        "duration_s": 3.0,
        "physics_timestep_s": PHYSICS_TIMESTEP_S,
        "matrix": "sand 1of3/2of3 x 0.10/0.15/0.20 x v03/v04/v05 = 18",
        "physical_oracle": "loaded_penetration_change_m >=0.0055m for 20ms, loaded/post-touchdown/pre-fall",
        "conditions": [{
            "run_id": item.run_id,
            "profile_name": item.profile.name,
            "speed_mps": item.walking_speed_mps,
            "variation": asdict(item.variation),
        } for item in conditions],
    })
    manifests = []
    traces = []
    started = time.perf_counter()
    for index, condition in enumerate(conditions):
        trace, metadata, _ = collect_run(condition, policy_path, 3.0)
        metadata["split"] = "model_holdout"
        manifests.append(metadata)
        traces.append(trace)
        print(
            f"sink holdout [{index + 1}/18] {condition.run_id} "
            f"fall={metadata['fall_occurred']} elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )
    packed = {key: np.asarray([trace[key] for trace in traces]) for key in traces[0]}
    packed.update({
        "run_id": np.asarray([row["run_id"] for row in manifests]),
        "profile_name": np.asarray([row["profile_name"] for row in manifests]),
        "walking_speed_mps": np.asarray([row["walking_speed_mps"] for row in manifests]),
        "variation_index": np.asarray([row["variation_index"] for row in manifests]),
        "sink_physical_label_oracle": np.asarray([
            physical_oracle(trace, "sink") for trace in traces
        ]),
    })
    np.savez_compressed(output / "traces.npz", **packed)
    write_json(output / "manifest.json", manifests)
    write_json(output / "source_lock.json", {
        "selection_artifact_sha256_before_collection": selection_hashes,
        "traces_sha256": sha256(output / "traces.npz"),
        "runs": len(manifests),
    })
    return manifests, traces


def spatial_diagnostic(
    terrain_model: object, terrain_normalizer: Normalizer
) -> tuple[list[dict[str, object]], dict[str, object]]:
    with (SPATIAL / "audit.csv").open(newline="", encoding="utf-8") as stream:
        audit = list(csv.DictReader(stream))
    with np.load(SPATIAL / "transition_traces.npz", allow_pickle=False) as packed:
        fusion = packed["fusion10"]
        run_ids = packed["run_id"].astype(str)
    rows = []
    for index, base in enumerate(audit):
        if run_ids[index] != base["run_id"]:
            raise ValueError("spatial manifest ordering mismatch")
        t0 = int(base["T0"])
        endpoints = np.arange(t0, t0 + 3, dtype=np.int64)
        values = terrain_normalizer.transform(causal_windows(
            fusion[index], endpoints, WINDOWS["terrain"]
        ))
        prediction = np.argmax(
            terrain_model.predict(values, batch_size=3, verbose=0), axis=1
        )
        expected = TERRAIN_LABELS[base["terrain_after"]]
        candidate_correct = bool(np.all(prediction == expected))
        rows.append({
            "run_id": base["run_id"], "case_id": base["case_id"],
            "terrain_before": base["terrain_before"], "terrain_after": base["terrain_after"],
            "fall_confounded": base["fall_occurred"] == "True",
            "frozen_case_correct": base["case_correct"] == "True",
            "candidate_predictions": "|".join(TERRAIN_NAMES[int(value)] for value in prediction),
            "candidate_three_endpoint_correct": candidate_correct,
            "candidate_improved": candidate_correct and base["case_correct"] != "True",
            "candidate_regressed": not candidate_correct and base["case_correct"] == "True",
            "read_only_diagnostic_not_training_or_selection": True,
        })
    by_case = {}
    for case in "ABCD":
        group = [row for row in rows if row["case_id"] == case]
        by_case[case] = {
            "runs": len(group),
            "frozen_correct": sum(bool(row["frozen_case_correct"]) for row in group),
            "candidate_correct": sum(bool(row["candidate_three_endpoint_correct"]) for row in group),
            "delta": sum(bool(row["candidate_three_endpoint_correct"]) for row in group)
            - sum(bool(row["frozen_case_correct"]) for row in group),
        }
    return rows, {
        "all_runs_fall_confounded": all(bool(row["fall_confounded"]) for row in rows),
        "read_only_after_selection": True,
        "case_results": by_case,
        "a_or_d_improved": by_case["A"]["delta"] > 0 or by_case["D"]["delta"] > 0,
    }


def mixture_rows(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for detector in ("terrain", "slip", "sink"):
        for ratio in MIXTURE_RATIOS:
            group = [
                row for row in seed_rows
                if row["detector"] == detector and np.isclose(float(row["mixture_ratio"]), ratio)
            ]
            rows.append({
                "detector": detector, "mixture_ratio": ratio,
                "training_seeds": "|".join(str(row["training_seed"]) for row in group),
                "existing_train_windows": group[0]["train_existing_windows"],
                "walking_train_windows": group[0]["train_walking_windows"],
                "walking_effective_weight": ratio,
                "passing_seed_count": sum(
                    bool(row.get("candidate_gate_pass", False))
                    or int(row.get("passing_runtime_configs", 0)) > 0 for row in group
                ),
                "walking_windows_duplicated": False,
            })
    return rows


def save_plots(
    output: Path,
    validation_rows: list[dict[str, object]],
    holdout_rows: list[dict[str, object]],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = output / "plots"
    directory.mkdir()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    terrain = [row for row in validation_rows if row["detector"] == "terrain"]
    for ratio in MIXTURE_RATIOS:
        group = [row for row in terrain if np.isclose(float(row["mixture_ratio"]), ratio)]
        axes[0].scatter(
            [row["static_validation_accuracy"] for row in group],
            [row["walking_validation_macro_accuracy"] for row in group],
            label=f"walking weight {ratio}",
        )
    axes[0].set(xlabel="static validation accuracy", ylabel="walking validation macro accuracy")
    axes[0].legend(fontsize=7)
    for detector, marker in (("slip", "o"), ("sink", "x")):
        group = [
            row for row in validation_rows
            if row["detector"] == detector and row["candidate_gate_pass"]
        ]
        axes[1].scatter(
            [row["controlled_validation_causal_run_fpr"] for row in group],
            [row["walking_positive_run_recall"] for row in group],
            marker=marker, label=detector, alpha=0.5,
        )
    axes[1].axvline(0.05, color="red", linestyle="--")
    axes[1].set(xlabel="controlled causal run FPR", ylabel="walking positive run recall")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(directory / "validation_selection.png", dpi=150)
    plt.close(figure)

    hazard = [row for row in holdout_rows if row.get("detector") in {"slip", "sink"}]
    figure, axis = plt.subplots(figsize=(8, 4))
    for detector, marker in (("slip", "o"), ("sink", "x")):
        group = [row for row in hazard if row["detector"] == detector and row.get("physical_source_valid")]
        axis.scatter(
            [row["walking_speed_mps"] for row in group],
            [row["physical_to_model_stable_latency_ms"] for row in group],
            marker=marker, label=detector,
        )
    axis.axhline(20, color="red", linestyle="--", label="20 ms target")
    axis.set(xlabel="walking speed [m/s]", ylabel="physical oracle to model stable [ms]")
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "holdout_physical_to_model_latency.png", dpi=150)
    plt.close(figure)


def audit_markdown(summary: dict[str, object]) -> str:
    return f"""# Walking-domain bounded float retraining v1

The existing deployable Terrain 50 ms, Slip 5 ms, and Sink 20 ms architectures
were retained.  Development used existing train/validation sources plus only
d2209cd v00/v01 training and v02 walking validation.  All candidate selections
were written and hashed before fd5b9f0 v03/v04/v05 or the new Sink holdout was
read or generated.

Selected candidates:

```json
{json.dumps(summary['selected_configs'], indent=2)}
```

Validation and holdout results:

```json
{json.dumps(summary['acceptance'], indent=2)}
```

Spatial A/B/C/D replay is read-only and every source run is fall-confounded:

```json
{json.dumps(summary['spatial_transition_diagnostic'], indent=2)}
```

No production model, normalization, threshold, persistence, INT8, Vela/E84,
U55, board, final-test, or System-v1 artifact was changed.

## Readiness gates

""" + "\n".join(
        f"- {name}={str(value).lower()}" for name, value in summary["gates"].items()
    )


def main() -> None:
    args = parse_args()
    planned = protocol(args.epochs)
    if not args.execute:
        print(json.dumps(planned, indent=2))
        return
    output = args.output_dir.resolve()
    sink_output = args.sink_holdout_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output: {output}")
    if sink_output.exists() and any(sink_output.iterdir()):
        raise FileExistsError(f"refusing non-empty Sink holdout: {sink_output}")
    if not args.policy_path.is_file():
        raise FileNotFoundError(args.policy_path)
    sources = verify_sources()
    output.mkdir(parents=True)
    write_json(output / "protocol.json", planned)
    write_json(output / "source_hashes.json", sources)

    dev_manifests, dev_traces = load_development()
    train_pairs = [
        (row, trace) for row, trace in zip(dev_manifests, dev_traces)
        if row["split"] == "walking_train"
    ]
    validation_pairs = [
        (row, trace) for row, trace in zip(dev_manifests, dev_traces)
        if row["split"] == "walking_validation"
    ]
    validation_manifests = [value[0] for value in validation_pairs]
    validation_traces = [value[1] for value in validation_pairs]
    audit = split_audit(dev_manifests)
    audit.update({
        "walking_train_runs": len(train_pairs),
        "walking_validation_runs": len(validation_pairs),
        "train_variations": [0, 1],
        "validation_variations": [2],
        "outer_variations_excluded": [3, 4, 5],
        "outer_run_count_accessed_before_selection": 0,
        "sample_or_episode_random_split": False,
    })
    if audit["split_leakage_count"]:
        raise ValueError(f"walking split leakage: {audit}")
    write_json(output / "split_audit.json", audit)

    window_sets = {}
    manifest_rows: list[dict[str, object]] = []
    for detector in ("terrain", "slip", "sink"):
        for split_name, sampled in (("walking_train", True), ("walking_validation", False)):
            data, rows = walking_windows(
                detector, dev_manifests, dev_traces, split_name, sampled
            )
            window_sets[f"{detector}_{split_name}"] = data
            manifest_rows.extend(rows)
    write_csv(output / "dataset_manifest.csv", manifest_rows)
    provenance = save_window_provenance(
        output / "walking_window_provenance.npz", window_sets
    )
    write_json(output / "dataset_manifest.json", {
        "aggregate_csv": "dataset_manifest.csv",
        "aggregate_csv_sha256": sha256(output / "dataset_manifest.csv"),
        "window_level_provenance": provenance,
        "existing_source_window_provenance": {
            "terrain": str(STATIC_DATA),
            "slip": str(HAZARD_DATA / "slip_5ms"),
            "sink": str(HAZARD_DATA / "sink_20ms"),
        },
        "run_and_episode_split_integrity": audit,
    })

    with np.load(STATIC_DATA, allow_pickle=False) as packed:
        static_x = packed["X"].astype(np.float32)
        static_y = packed["y"].astype(np.int64)
        static_split = packed["split"].astype(str)
    static_train = static_split == "train"
    static_validation = static_split == "architecture_selection"
    controlled = {
        detector: {
            split_name: raw_controlled(detector, split_name)
            for split_name in ("train", "validation")
        } for detector in ("slip", "sink")
    }

    baseline_rows, baseline = baseline_replay(
        dev_manifests, dev_traces, window_sets["terrain_walking_validation"],
        {name: value["validation"] for name, value in controlled.items()},
    )
    write_csv(output / "baseline_walking_replay.csv", baseline_rows)
    write_json(output / "baseline_walking_replay_summary.json", baseline)
    static_model_baseline = json.loads((STATIC_REFERENCE / "summary.json").read_text())
    baseline_static_accuracy = float(static_model_baseline["candidates"][0]["selection_accuracy"])

    dataset_statistics = {
        "sources": {
            "terrain_static_train_windows": int(np.count_nonzero(static_train)),
            "terrain_static_validation_windows": int(np.count_nonzero(static_validation)),
            "terrain_static_train_class_counts": {
                name: int(np.count_nonzero(static_y[static_train] == label))
                for name, label in TERRAIN_LABELS.items()
            },
            "terrain_static_validation_class_counts": {
                name: int(np.count_nonzero(static_y[static_validation] == label))
                for name, label in TERRAIN_LABELS.items()
            },
            "slip_controlled_train_windows": len(controlled["slip"]["train"]["y"]),
            "slip_controlled_validation_windows": len(controlled["slip"]["validation"]["y"]),
            "sink_controlled_train_windows": len(controlled["sink"]["train"]["y"]),
            "sink_controlled_validation_windows": len(controlled["sink"]["validation"]["y"]),
            "slip_controlled_train_class_counts": {
                "negative": int(np.count_nonzero(controlled["slip"]["train"]["y"] == 0)),
                "positive": int(np.count_nonzero(controlled["slip"]["train"]["y"] == 1)),
            },
            "slip_controlled_validation_class_counts": {
                "negative": int(np.count_nonzero(controlled["slip"]["validation"]["y"] == 0)),
                "positive": int(np.count_nonzero(controlled["slip"]["validation"]["y"] == 1)),
            },
            "sink_controlled_train_class_counts": {
                "negative": int(np.count_nonzero(controlled["sink"]["train"]["y"] == 0)),
                "positive": int(np.count_nonzero(controlled["sink"]["train"]["y"] == 1)),
            },
            "sink_controlled_validation_class_counts": {
                "negative": int(np.count_nonzero(controlled["sink"]["validation"]["y"] == 0)),
                "positive": int(np.count_nonzero(controlled["sink"]["validation"]["y"] == 1)),
            },
            "walking_train_runs": len(train_pairs),
            "walking_validation_runs": len(validation_pairs),
        },
        "walking_windows": {
            name: {
                "windows": len(data["y"]),
                "class_counts": ({
                    terrain: int(np.count_nonzero(data["y"] == label))
                    for terrain, label in TERRAIN_LABELS.items()
                } if name.startswith("terrain_") else {
                    "negative": int(np.count_nonzero(data["y"] == 0)),
                    "positive": int(np.count_nonzero(data["y"] == 1)),
                }),
                "unique_runs": len(set(data["run_id"].astype(str))),
                "gait_phase_counts": {
                    phase: int(np.count_nonzero(data["gait_phase"].astype(str) == phase))
                    for phase in sorted(set(data["gait_phase"].astype(str)))
                },
            } for name, data in window_sets.items()
        },
        "mixture_grid": list(MIXTURE_RATIOS),
        "training_seeds": list(TRAINING_SEEDS),
        "walking_sample_duplication": False,
    }
    write_json(output / "dataset_statistics.json", dataset_statistics)

    all_validation_rows: list[dict[str, object]] = []
    all_seed_rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="walking-bounded-v1-") as temporary_name:
        temporary = Path(temporary_name)
        terrain_rows, terrain_seed_rows, terrain_norms = train_terrain_candidates(
            temporary, args.epochs,
            static_x[static_train], static_y[static_train],
            static_x[static_validation], static_y[static_validation],
            window_sets["terrain_walking_train"], window_sets["terrain_walking_validation"],
            baseline_static_accuracy,
        )
        all_validation_rows.extend(terrain_rows)
        all_seed_rows.extend(terrain_seed_rows)
        terrain_selected = select_terrain(terrain_rows)
        hazard_selected = {}
        hazard_norms = {}
        for detector in ("slip", "sink"):
            rows, seed_rows, norms, _ = train_hazard_candidates(
                detector, temporary, args.epochs,
                controlled[detector]["train"], controlled[detector]["validation"],
                window_sets[f"{detector}_walking_train"],
                window_sets[f"{detector}_walking_validation"],
                validation_manifests, validation_traces,
                baseline[f"{detector}_controlled_validation"],
            )
            all_validation_rows.extend(rows)
            all_seed_rows.extend(seed_rows)
            hazard_selected[detector] = select_hazard(rows)
            if hazard_selected[detector] is None:
                hazard_selected[detector] = diagnostic_hazard_candidate(
                    detector, rows, baseline[f"{detector}_controlled_validation"]
                )
            hazard_norms[detector] = norms
        if terrain_selected is None:
            write_csv(
                output / "validation_metrics.csv",
                portable_candidate_rows(all_validation_rows),
            )
            write_csv(
                output / "seed_metrics.csv", portable_candidate_rows(all_seed_rows)
            )
            raise RuntimeError("Terrain had no static-retention-gate-passing candidate")

        selections = {
            "terrain": selected_payload(
                "terrain", terrain_selected,
                terrain_norms[(float(terrain_selected["mixture_ratio"]), int(terrain_selected["training_seed"]))],
                output,
            ),
            **{
                detector: selected_payload(
                    detector, selected,
                    hazard_norms[detector][(
                        float(selected["mixture_ratio"]), int(selected["training_seed"])
                    )], output,
                ) for detector, selected in hazard_selected.items()
            },
        }
        for detector, selection in selections.items():
            write_json(output / f"{detector}_selection.json", selection)
        selection_hashes = {
            detector: sha256(output / f"{detector}_selection.json")
            for detector in selections
        }
        write_json(output / "selection_lock.json", {
            "all_three_selections_completed": True,
            "selection_artifact_sha256": selection_hashes,
            "fd5b9f0_outer_trace_access_count_before_lock": 0,
            "new_sink_holdout_runs_before_lock": 0,
            "locked_before_holdout": True,
        })

    write_csv(
        output / "validation_metrics.csv",
        portable_candidate_rows(all_validation_rows),
    )
    write_csv(output / "seed_metrics.csv", portable_candidate_rows(all_seed_rows))
    write_csv(output / "mixture_candidates.csv", mixture_rows(all_seed_rows))

    # Holdout phase begins only after selection JSON files and their lock exist.
    terrain_model, terrain_norm, terrain_selection = load_candidate(output, "terrain")
    slip_model, slip_norm, slip_selection = load_candidate(output, "slip")
    sink_model, sink_norm, sink_selection = load_candidate(output, "sink")
    outer_manifests, outer_traces = load_trace_set(
        OUTER / "outer_validation_traces.npz",
        OUTER / "outer_validation_manifest.json", 36,
    )
    for row in outer_manifests:
        row["split"] = "model_holdout"
    sink_manifests, sink_traces = collect_sink_holdout(
        sink_output, args.policy_path.resolve(), selection_hashes
    )

    holdout_rows: list[dict[str, object]] = []
    latency_rows: list[dict[str, object]] = []
    terrain_rows_holdout, terrain_metrics = terrain_holdout(
        terrain_model, terrain_norm, outer_manifests, outer_traces, "fd5b9f0_outer"
    )
    holdout_rows.extend(terrain_rows_holdout)
    slip_rows, slip_latency, slip_metrics = hazard_holdout(
        "slip", slip_model, slip_norm, slip_selection,
        outer_manifests, outer_traces, {"hard_negative", "slip_candidate"}, "fd5b9f0_outer",
    )
    holdout_rows.extend(slip_rows)
    latency_rows.extend(slip_latency)
    sink_normal_rows, sink_normal_latency, sink_normal_metrics = hazard_holdout(
        "sink", sink_model, sink_norm, sink_selection,
        outer_manifests, outer_traces, {"hard_negative"}, "fd5b9f0_outer_normal",
    )
    sink_positive_rows, sink_positive_latency, sink_positive_metrics = hazard_holdout(
        "sink", sink_model, sink_norm, sink_selection,
        sink_manifests, sink_traces, {"sink_candidate"}, "new_sink_holdout",
    )
    holdout_rows.extend(sink_normal_rows + sink_positive_rows)
    latency_rows.extend(sink_normal_latency + sink_positive_latency)

    # Frozen baselines are also read-only and evaluated only after the lock.
    import tensorflow as tf
    frozen_terrain = tf.keras.models.load_model(STATIC_REFERENCE / "selected_model.keras", compile=False)
    _, frozen_terrain_metrics = terrain_holdout(
        frozen_terrain, old_normalizer(STATIC_REFERENCE / "normalization.json"),
        outer_manifests, outer_traces, "fd5b9f0_outer_frozen_baseline",
    )
    frozen_holdout = {"terrain": frozen_terrain_metrics}
    for detector, manifests, traces, roles in (
        ("slip", outer_manifests, outer_traces, {"hard_negative", "slip_candidate"}),
        ("sink", outer_manifests + sink_manifests, outer_traces + sink_traces, {"hard_negative", "sink_candidate"}),
    ):
        model = tf.keras.models.load_model(HAZARD_MODELS[detector], compile=False)
        norm = old_normalizer(HAZARD_DATA / f"{detector}_{WINDOWS[detector]}ms" / "normalization.json")
        selection = {
            "selected_probability_threshold": OLD_HAZARD_CONFIG[detector]["threshold"],
            "selected_runtime_persistence": OLD_HAZARD_CONFIG[detector]["persistence"],
        }
        _, _, metrics = hazard_holdout(
            detector, model, norm, selection, manifests, traces, roles,
            f"{detector}_frozen_baseline_holdout",
        )
        frozen_holdout[detector] = metrics

    spatial_rows, spatial_metrics = spatial_diagnostic(terrain_model, terrain_norm)
    write_csv(output / "holdout_metrics.csv", holdout_rows)
    write_csv(output / "latency_metrics.csv", latency_rows)
    write_csv(output / "spatial_transition_diagnostic.csv", spatial_rows)

    reload_parity = {}
    for detector, data in (
        ("terrain", window_sets["terrain_walking_validation"]["x"][:32]),
        ("slip", window_sets["slip_walking_validation"]["x"][:32]),
        ("sink", window_sets["sink_walking_validation"]["x"][:32]),
    ):
        first_model, norm, _ = load_candidate(output, detector)
        second_model, _, _ = load_candidate(output, detector)
        values = norm.transform(data)
        first = first_model.predict(values, verbose=0)
        second = second_model.predict(values, verbose=0)
        reload_parity[detector] = {
            "samples": len(values),
            "max_abs_difference": float(np.max(np.abs(first - second))),
            "parity": bool(np.array_equal(first, second)),
        }

    sink_profiles = sink_positive_metrics["profile_results"]
    sink_complete_profiles = [
        name for name, value in sink_profiles.items()
        if int(value["physical_positive_runs"]) == 9
        and int(value["detected_runs"]) == 9
        and value["detected_speeds_mps"] == [0.1, 0.15, 0.2]
    ]
    terrain_ready = bool(
        float(terrain_selection["static_validation_accuracy"]) >= baseline_static_accuracy - 0.01
        and all(
            terrain_metrics["class_metrics"][name]["support"] > 0 for name in TERRAIN_NAMES
        )
    )
    slip_ready = bool(
        int(slip_metrics["normal_false_positive_runs"]) == 0
        and int(slip_metrics["detected_positive_runs"]) == 9
        and int(slip_metrics["physical_positive_runs"]) == 9
        and int(slip_metrics["anticipation_runs"]) == 0
        and int(slip_metrics["label_mask_violation_samples"]) == 0
    )
    sink_ready = bool(
        int(sink_normal_metrics["normal_false_positive_runs"]) == 0
        and sink_complete_profiles
        and int(sink_positive_metrics["anticipation_runs"]) == 0
        and int(sink_positive_metrics["label_mask_violation_samples"]) == 0
    )
    dataset_ready = bool(
        len(train_pairs) == 36 and len(validation_pairs) == 18
        and all(len(data["y"]) for data in window_sets.values())
    )
    non_access_ready = bool(
        all(value["holdout_runs_accessed_before_selection"] == 0 for value in selections.values())
        and json.loads((output / "selection_lock.json").read_text())[
            "fd5b9f0_outer_trace_access_count_before_lock"
        ] == 0
    )
    candidate_ready = {
        name: bool(value["candidate_gate_pass"]) for name, value in selections.items()
    }
    gates = {
        "WALKING_RETRAIN_DATASET_READY": dataset_ready,
        "WALKING_RETRAIN_SPLIT_INTEGRITY_READY": audit["split_leakage_count"] == 0,
        "WALKING_TERRAIN_FLOAT_CANDIDATE_READY": candidate_ready["terrain"],
        "WALKING_SLIP_FLOAT_CANDIDATE_READY": candidate_ready["slip"],
        "WALKING_SINK_FLOAT_CANDIDATE_READY": candidate_ready["sink"],
        "WALKING_MODEL_HOLDOUT_NON_ACCESS_READY": non_access_ready,
        "WALKING_TERRAIN_HOLDOUT_READY": terrain_ready,
        "WALKING_SLIP_HOLDOUT_READY": slip_ready,
        "WALKING_SINK_HOLDOUT_READY": sink_ready,
    }
    complete = all(gates.values())
    gates["WALKING_BOUNDED_FLOAT_RETRAINING_READY"] = complete
    gates["WALKING_INT8_PREPARATION_AUTHORIZED"] = complete
    summary = {
        "starting_checkpoint": STARTING_CHECKPOINT,
        "development_sources": {
            "walking_train_runs": 36, "walking_validation_runs": 18,
            "static_train_windows": int(np.count_nonzero(static_train)),
            "static_validation_windows": int(np.count_nonzero(static_validation)),
            "controlled_train_windows_per_hazard": 6750,
            "controlled_validation_windows_per_hazard": 9000,
        },
        "holdout_sources": {
            "fd5b9f0_outer_runs": 36,
            "new_sink_runs": 18,
            "spatial_diagnostic_runs": 12,
        },
        "selected_configs": selections,
        "selection_artifact_sha256": selection_hashes,
        "baseline_validation": baseline,
        "frozen_holdout_baseline": frozen_holdout,
        "acceptance": {
            "terrain_candidate": terrain_metrics,
            "terrain_frozen_baseline": frozen_terrain_metrics,
            "terrain_air_transitions": 0,
            "slip_candidate": slip_metrics,
            "sink_normal_candidate": sink_normal_metrics,
            "sink_positive_candidate": sink_positive_metrics,
            "sink_complete_profiles": sink_complete_profiles,
        },
        "spatial_transition_diagnostic": spatial_metrics,
        "model_reload_parity": reload_parity,
        "falls": {
            "fd5b9f0_outer": sum(bool(row["fall_occurred"]) for row in outer_manifests),
            "new_sink_holdout": sum(bool(row["fall_occurred"]) for row in sink_manifests),
            "evaluation_semantics": "pre-fall only",
        },
        "production_artifacts_changed": False,
        "physical_oracles_changed": False,
        "final_test_used": False,
        "gates": gates,
        **gates,
    }
    write_json(output / "summary.json", summary)
    write_json(sink_output / "summary.json", {
        "runs": len(sink_manifests),
        "falls": sum(bool(row["fall_occurred"]) for row in sink_manifests),
        "pre_fall_only_evaluation": True,
        "selection_artifact_sha256_before_collection": selection_hashes,
        "candidate_metrics": sink_positive_metrics,
        "complete_profiles": sink_complete_profiles,
        "WALKING_SINK_HOLDOUT_READY": sink_ready,
        "selection_or_threshold_adjusted_after_holdout": False,
    })
    (output / "audit.md").write_text(audit_markdown(summary), encoding="utf-8")
    save_plots(output, all_validation_rows, holdout_rows)
    print(json.dumps({"selected": selections, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
