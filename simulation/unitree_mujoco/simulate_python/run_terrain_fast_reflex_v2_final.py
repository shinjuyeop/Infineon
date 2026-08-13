"""Fail-closed, one-shot final host evaluation for frozen Fast Reflex v2 detectors.

Without --execute-final-test this command is a read-only preflight.  The
explicit execution path materializes exactly one reserved final dataset and
immediately evaluates only the two already-frozen detector configurations.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, subprocess, sys
from pathlib import Path
from typing import Any
import numpy as np

from terrain_fast_reflex_v2 import TRACE_PRE_MS, V2Calibration, final_scope_full_configs, label_v2, validate_state_order
from terrain_fast_reflex_v2_detector import Normalizer
from run_terrain_fast_reflex_v2_validation import endpoint_metrics, replay, subgroup_rows, write_csv, save_representative_replays

ROOT = Path("../..")
FINAL_OUTPUT = ROOT / "outputs/terrain_fast_reflex_v2_final_test"
EVALUATION_OUTPUT = ROOT / "outputs/terrain_fast_reflex_v2_final_evaluation"
SELECTION_OUTPUT = ROOT / "outputs/terrain_fast_reflex_v2_detector_validation_selection"
SOURCE_OUTPUT = ROOT / "outputs/terrain_fast_reflex_v2_final_scope_full"
DATASET = ROOT / "outputs/terrain_fast_reflex_v2_detector_dataset"
FAMILIES = ("final_fresh_crosshatch", "final_fresh_rounded_ridges")
PHYSICAL_FAMILIES = ("crosshatch", "rounded_ridges")
SURFACE_INDICES = tuple(range(9110, 9115))
RUNS_PER_SURFACE = 3
FINAL_SESSION_SEED = 20260903
FINAL_EXCITATION_OFFSET = 921000
FROZEN = {
    "slip": {"target": "confirmed_slip", "window_ms": 5, "threshold": 0.9719217419624332, "persistence": 3, "pooling": "GAP + GlobalMax", "parameters": 1237, "model": ROOT / "outputs/terrain_fast_reflex_v2_detector_training_slip/slip_5ms/model.keras", "normalization": DATASET / "slip_5ms/normalization.json", "model_sha256": "4751c73be3047ad968caa16574f4bac7537369392435e928c7c8d2ad0051d377", "normalization_sha256": "67f8f6256626d55fe0dd33a1631836234216dbcafe473a66df304554033a65df"},
    "sink": {"target": "sustained_sink", "window_ms": 20, "threshold": 0.9836072999238968, "persistence": 1, "pooling": "GAP", "parameters": 1221, "model": ROOT / "outputs/terrain_fast_reflex_v2_detector_training_sink/sink_20ms/model.keras", "normalization": DATASET / "sink_20ms/normalization.json", "model_sha256": "8447152ffe17dcf75943aca6008fbab19b2b64a28c041ba3cc6573e329833453", "normalization_sha256": "5949d27423d81bbe6ccfe586b35d6f9727717d5eb35bc4f07c91750a793848f1"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def planned_runs() -> int: return len(final_scope_full_configs()) * len(FAMILIES) * len(SURFACE_INDICES) * RUNS_PER_SURFACE


def frozen_public() -> dict[str, Any]:
    return {d: {k: (str(v.resolve()) if isinstance(v, Path) else v) for k, v in c.items()} for d, c in FROZEN.items()}


def verify_selected_values(detector: str, actual: dict[str, Any]) -> None:
    for key in ("window_ms", "threshold", "persistence"):
        if actual.get(key) != FROZEN[detector][key]:
            raise ValueError(f"{detector} selected {key} differs from frozen protocol")


def verify_preflight(final_output: Path, evaluation_output: Path | None = None) -> dict[str, Any]:
    selection = json.loads((SELECTION_OUTPUT / "protocol.json").read_text())
    if selection.get("FINAL_TEST_READY") is not True: raise ValueError("FINAL_TEST_READY is not true")
    if final_output.exists(): raise FileExistsError(f"final-test output already exists: {final_output}")
    if evaluation_output is not None and evaluation_output.exists(): raise FileExistsError(f"final evaluation output already exists: {evaluation_output}")
    hashes = {}
    for detector, config in FROZEN.items():
        selected = json.loads((SELECTION_OUTPUT / f"{detector}_selected.json").read_text())
        actual = selected.get("selected") or {}
        verify_selected_values(detector, actual)
        for key in ("model", "normalization"):
            value = sha256(config[key]); expected = config[f"{key}_sha256"]
            if value != expected: raise ValueError(f"{detector} {key} SHA256 mismatch")
            hashes[f"{detector}_{key}"] = value
    return {"FINAL_TEST_READY": True, "final_test_materialized": 0, "output_must_not_exist": str(final_output.resolve()), "evaluation_output_must_not_exist": None if evaluation_output is None else str(evaluation_output.resolve()), "planned_runs": planned_runs(), "families": FAMILIES, "physical_morphology_realizations": PHYSICAL_FAMILIES, "surface_index_reservation": [min(SURFACE_INDICES), max(SURFACE_INDICES)], "session_seed": FINAL_SESSION_SEED, "excitation_seed_offset": FINAL_EXCITATION_OFFSET, "modes": ["normal_sand", "slip_risk_dominant", "sink_dominant"], "frozen_hashes": hashes, "detectors": frozen_public()}


def final_rows() -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    # MuJoCo is required only for the explicitly authorized one-shot run, not
    # for preflight/hash verification or its unit tests.
    from run_terrain_fast_reflex_v2 import run_one
    source_protocol = json.loads((SOURCE_OUTPUT / "protocol.json").read_text())
    calibration = V2Calibration(**{k: v for k, v in source_protocol["calibration"].items() if k in V2Calibration.__dataclass_fields__})
    raw = []
    for config in final_scope_full_configs():
        for family, physical in zip(FAMILIES, PHYSICAL_FAMILIES):
            for index in SURFACE_INDICES:
                for run_index in range(RUNS_PER_SURFACE):
                    item = run_one(config, physical, index, run_index)
                    item.metadata.update({"split": "final_test", "surface_family": family, "surface_seed": index, "session_id": f"v2_final_{FINAL_SESSION_SEED}_{family}_{index}", "reserved_surface_index": index, "excitation_seed_offset": FINAL_EXCITATION_OFFSET})
                    raw.append(item)
    labels = [label_v2(item.oracle, calibration) for item in raw]
    for value in labels: validate_state_order(value)
    return raw, labels, calibration.as_dict()


def materialize(final_output: Path) -> None:
    from run_terrain_fast_reflex_v2 import _row, save
    final_output.mkdir(parents=True, exist_ok=False)
    raw, labels, calibration = final_rows()
    rows = [_row(item, label) for item, label in zip(raw, labels)]
    protocol = {"dataset_name": "terrain_fast_reflex_v2", "schema_version": 2, "final_test": {"materialized": True, "one_shot": True, "marker": "FINAL_TEST_MATERIALIZED"}, "status": "untouched final reservation materialized exactly once", "split": "final_test", "modes": ["normal_sand", "slip_risk_dominant", "sink_dominant"], "families": FAMILIES, "physical_morphology_realizations": PHYSICAL_FAMILIES, "surface_index_reservation": list(SURFACE_INDICES), "session_seed": FINAL_SESSION_SEED, "excitation_seed_offset": FINAL_EXCITATION_OFFSET, "candidate_runs": len(raw), "calibration": calibration, "frozen_detector_configs": frozen_public(), "v1_test_families_ineligible": ["warped_multisine", "smooth_random_patches"]}
    save(final_output, raw, labels, rows, protocol)
    (final_output / "frozen_detector_configs.json").write_text(json.dumps(frozen_public(), indent=2) + "\n")
    (final_output / "FINAL_TEST_MATERIALIZED").write_text("one-shot final materialization completed; regeneration forbidden\n")


def final_windows(sensors: np.ndarray, labels: dict[str, np.ndarray], rows: list[dict[str, str]], detector: str) -> dict[str, np.ndarray]:
    c = FROZEN[detector]; x=[]; y=[]; run=[]; endpoint=[]; family=[]; mode=[]
    for i, row in enumerate(rows):
        for ms in range(100):
            end = TRACE_PRE_MS + ms; start = end - c["window_ms"] + 1
            x.append(sensors[i, start:end + 1]); y.append(labels[c["target"]][i, end]); run.append(row["run_id"]); endpoint.append(ms); family.append(row["surface_family"]); mode.append(row["mode"])
    normal = json.loads(c["normalization"].read_text()); norm = Normalizer(np.asarray(normal["mean"], np.float32), np.asarray(normal["std"], np.float32))
    return {"x": norm.transform(np.asarray(x)), "y": np.asarray(y, np.int8), "run_id": np.asarray(run), "endpoint_ms": np.asarray(endpoint, np.int16), "family": np.asarray(family), "mode": np.asarray(mode)}


def evaluate(final_output: Path, evaluation_output: Path) -> None:
    if not (final_output / "FINAL_TEST_MATERIALIZED").is_file(): raise ValueError("missing FINAL_TEST_MATERIALIZED marker")
    if evaluation_output.exists(): raise FileExistsError(f"final evaluation output exists: {evaluation_output}")
    with (final_output / "manifest.csv").open() as f: rows = list(csv.DictReader(f))
    if any(r["split"] != "final_test" or r["surface_family"] not in FAMILIES for r in rows): raise ValueError("unexpected final split/family")
    with np.load(final_output / "inputs_fusion10.npz", allow_pickle=False) as z: sensors = z["sensors"]
    with np.load(final_output / "oracle_diagnostics.npz", allow_pickle=False) as z: labels = {k: z[k] for k in ("confirmed_slip", "sustained_sink")}
    import tensorflow as tf
    evaluation_output.mkdir(parents=True); plots = evaluation_output / "plots"; plots.mkdir()
    gate = {}
    for detector, config in FROZEN.items():
        data = final_windows(sensors, labels, rows, detector)
        model = tf.keras.models.load_model(config["model"], compile=False); scores = model.predict(data["x"], batch_size=1024, verbose=0).reshape(-1)
        replay_rows, metrics = replay(data, scores, config["threshold"], config["persistence"])
        report = {"detector": detector, "frozen_config": frozen_public()[detector], "endpoint": endpoint_metrics(data["y"], scores, config["threshold"]), "run": metrics, "primary_host_gate": "overall causal run FPR <= 5%", "host_gate": "PASS" if metrics["overall_causal_run_fpr"] <= .05 else "FAIL"}
        (evaluation_output / f"{detector}_final_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        write_csv(evaluation_output / f"{detector}_family_metrics.csv", subgroup_rows(replay_rows, "family")); write_csv(evaluation_output / f"{detector}_mode_metrics.csv", subgroup_rows(replay_rows, "mode")); write_csv(evaluation_output / f"{detector}_final_replay.csv", replay_rows); save_representative_replays(plots, detector, data, scores, config, replay_rows); gate[detector] = report["host_gate"]
    ready = all(value == "PASS" for value in gate.values())
    (evaluation_output / "final_summary.md").write_text("# Frozen Fast Reflex v2 final host evaluation\n\n" + "\n".join(f"- {d.upper()}_FINAL_HOST_GATE={v}" for d, v in gate.items()) + f"\n- INT8_READY={'true' if ready else 'false'}\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("action", choices=("preflight", "run", "evaluate", "audit"), nargs="?", default="preflight"); p.add_argument("--final-output", type=Path, default=FINAL_OUTPUT); p.add_argument("--evaluation-output", type=Path, default=EVALUATION_OUTPUT); p.add_argument("--evaluation-python", type=Path, default=Path("../../../Infineon_HIL/.venv/bin/python")); p.add_argument("--execute-final-test", action="store_true"); a = p.parse_args()
    final_output, evaluation_output = a.final_output.resolve(), a.evaluation_output.resolve()
    if a.action == "preflight": print(json.dumps(verify_preflight(final_output, evaluation_output), indent=2)); return
    if a.action == "run":
        if not a.execute_final_test: raise ValueError("run is fail-closed; require --execute-final-test")
        print(json.dumps(verify_preflight(final_output, evaluation_output), indent=2)); materialize(final_output)
        evaluator = a.evaluation_python.resolve()
        if not evaluator.is_file(): raise FileNotFoundError(f"evaluation Python is unavailable: {evaluator}")
        subprocess.run([str(evaluator), str(Path(__file__).resolve()), "evaluate", "--final-output", str(final_output), "--evaluation-output", str(evaluation_output)], check=True)
        print(f"FINAL_TEST_ONE_SHOT_COMPLETE final={final_output} evaluation={evaluation_output}"); return
    if a.action == "evaluate":
        evaluate(final_output, evaluation_output); print(f"FINAL_TEST_EVALUATION_COMPLETE evaluation={evaluation_output}"); return
    if not (evaluation_output / "final_summary.md").is_file(): raise FileNotFoundError("final evaluation summary does not exist")
    print((evaluation_output / "final_summary.md").read_text(), end="")


if __name__ == "__main__": main()
