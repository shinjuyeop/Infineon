"""Run the frozen Slip detector on the held-out test split exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from terrain_fast_reflex_detector_v1 import ChannelNormalizer, load_corrected_traces, make_windows, write_csv
from terrain_fast_reflex_slip_final_test_v1 import (
    FROZEN_ARCHITECTURE, FROZEN_PERSISTENCE, FROZEN_THRESHOLD, FROZEN_WINDOW_MS,
    evaluate_frozen_scores, validate_test_ownership,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../../outputs/terrain_fast_reflex_v1_full_corrected_v2"))
    parser.add_argument("--candidate-dir", type=Path, default=Path("../../outputs/terrain_fast_reflex_detector_validation_v1/pooling/slip_5ms_average_max"))
    parser.add_argument("--output-dir", type=Path, default=Path("../../outputs/terrain_fast_reflex_slip_final_test_v1"))
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not args.execute:
        print("DRY RUN: frozen Slip test; add --execute for the single permitted test inference")
        return
    source = args.source.resolve(); candidate = args.candidate_dir.resolve(); output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to rerun or overwrite non-empty final output: {output}")
    model_path = candidate / "model.keras"
    normalization_path = candidate / "normalization.json"
    info = json.loads((candidate / "model_info.json").read_text(encoding="utf-8"))
    expected_info = {"pooling": FROZEN_ARCHITECTURE, "window_ms": FROZEN_WINDOW_MS, "parameters": 1237,
                     "conv_dense_macs_per_inference": 5912}
    for key, expected in expected_info.items():
        if info.get(key) != expected:
            raise ValueError(f"frozen candidate {key} mismatch: {info.get(key)!r} != {expected!r}")
    source_protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    if source_protocol.get("derived_artifact_revision") != 2:
        raise ValueError("source must be corrected-v2")

    output.mkdir(parents=True, exist_ok=True)
    frozen = {
        "status": "frozen before test split materialization and inference",
        "candidate_selection_source": "validation only",
        "architecture": "Conv1D(12,k=5,relu,same) -> Conv1D(16,k=3,relu,same) -> GAP + GlobalMax concat -> Dense(1,sigmoid)",
        "pooling": FROZEN_ARCHITECTURE, "window_ms": FROZEN_WINDOW_MS,
        "threshold": FROZEN_THRESHOLD, "persistence": FROZEN_PERSISTENCE,
        "normalization": "loaded unchanged from frozen candidate artifact",
        "seed": 20260812, "retraining": False, "parameters": 1237, "macs": 5912,
        "test_families": ["smooth_random_patches", "warped_multisine"],
        "model_sha256": sha256(model_path), "normalization_sha256": sha256(normalization_path),
        "post_test_tuning_permitted": False,
    }
    (output / "frozen_config.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")

    test_traces, _ = load_corrected_traces(source, split="test")
    test_traces = validate_test_ownership(test_traces)
    windows = make_windows(test_traces, "slip", FROZEN_WINDOW_MS)
    values = json.loads(normalization_path.read_text(encoding="utf-8"))
    normalizer = ChannelNormalizer(np.asarray(values["mean"], dtype=np.float32), np.asarray(values["std"], dtype=np.float32))
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    # The only model inference in this final-test program. All metrics reuse this array.
    scores = model.predict(normalizer.transform(windows.x), batch_size=1024, verbose=0).reshape(-1)
    result = evaluate_frozen_scores(test_traces, scores)
    metrics = {"frozen_config": frozen, **result.metrics}
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "test_run_metrics.csv", result.run_rows)
    write_csv(output / "test_latency.csv", result.latency_rows)
    endpoint, run, gate = result.metrics["endpoint"], result.metrics["run_level"], result.metrics["gate"]
    summary = f"""# Frozen Slip Final Test\n\n{gate['classification']}\n\n- Test runs: {run['run_count']} (`warped_multisine`, `smooth_random_patches`)\n- Endpoint precision / recall / F1 / FPR: {endpoint['precision']:.6f} / {endpoint['recall']:.6f} / {endpoint['f1']:.6f} / {endpoint['fpr']:.6f}\n- Run-level target recall: {run['run_target_detected']} / {run['target_runs']} = {run['run_level_target_recall']:.6f}\n- Pre-onset run FPR: {run['pre_onset_false_alarm_runs']} / {run['run_count']} = {run['pre_onset_run_fpr']:.6f}\n- Completely hazard-free run FPR: {run['completely_hazard_free_run_fpr']}\n- Normal-terrain run FPR: {run['normal_terrain_run_fpr']}\n- Transition to stable detection median / p95 / max (ms): {run['median_transition_to_stable_detection_ms']} / {run['p95_transition_to_stable_detection_ms']} / {run['max_transition_to_stable_detection_ms']}\n- Hazard onset to stable detection median / p95 / max (ms): {run['median_hazard_onset_to_stable_detection_ms']} / {run['p95_hazard_onset_to_stable_detection_ms']} / {run['max_hazard_onset_to_stable_detection_ms']}\n- Anticipation firing runs / endpoints: {run['anticipation_firing_runs']} / {run['anticipation_firing_count']}\n- Anticipation lead median / p95 / max (ms): {run['median_anticipation_lead_ms']} / {run['p95_anticipation_lead_ms']} / {run['max_anticipation_lead_ms']}\n\nThe candidate remains frozen. No test-driven tuning or retraining was performed.\n"""
    (output / "test_summary.md").write_text(summary, encoding="utf-8")
    print(gate["classification"])


if __name__ == "__main__":
    main()
