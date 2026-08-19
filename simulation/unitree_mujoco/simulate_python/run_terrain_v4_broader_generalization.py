"""Reserve then execute the frozen Terrain v4 broader-domain INT8 test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from run_terrain_causal_window_v4 import LABEL, OUT as V4, STATIC, T0, ChannelNormalizer, aggregate_transition, load_transition, stable
from terrain_int8 import predict_tflite


SIM = Path(__file__).resolve().parents[2]
OUT = SIM / "outputs/terrain_v4_broader_generalization"
FAMILIES = ("crosshatch", "rounded_ridges", "warped_multisine")
REALIZATIONS = (8, 9)
RUN_INDICES = (4, 5)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance() -> dict[str, object]:
    int8 = json.loads((V4 / "int8/summary.json").read_text())
    manifest = int8["manifest"]
    normalizer = V4 / "normalization.json"
    if not normalizer.exists():
        # v4 stores the source-of-truth normalizer through the selected INT8 manifest.
        normalizer = V4 / "int8/manifest.json"
    return {"float_path": manifest["candidate_specific_float_path"], "float_sha256": manifest["candidate_specific_float_sha256"], "strict_int8_path": manifest["int8_path"], "strict_int8_sha256": manifest["int8_sha256"], "normalization_source": str(normalizer), "normalization_sha256": sha(normalizer), "input_shape": [1, 50, 10], "input_dtype": "int8", "output_dtype": "int8", "class_mapping": LABEL, "sample_rate_hz": 1000, "window_ms": 50, "seed": int8["selected_float"]["seed"], "t1_persistence": 3}


def reservation() -> dict[str, object]:
    return {"families": FAMILIES, "surface_realizations": REALIZATIONS, "run_indices": RUN_INDICES, "directions": ("A", "B", "C", "D"), "runs": len(FAMILIES) * len(REALIZATIONS) * len(RUN_INDICES) * 4, "freeze_before_results": True, "rationale": "all seven available procedural families occur in historical static or transition work; reserve unused s08/s09 realizations and r004/r005 source runs from three morphologically distinct families"}


def protocol() -> dict[str, object]:
    return {"dataset": "terrain_v4_broader_generalization", "frozen_model": provenance(), "reservation": reservation(), "metrics": ["stable_detection_rate", "target_occupancy", "prediction_switches", "t1_median_p95_max", "pre_transition_source_accuracy", "post_transition_target_accuracy"], "gates": {"each_direction_stable_detection_rate": 0.90, "case_c_target_occupancy": 0.80, "severe_persistent_oscillation": "absent"}, "no_training_or_model_policy_change": True}


def known_audit() -> dict[str, object]:
    def realization(value: object) -> str:
        return str(int(str(value).split("s")[-1]))

    known: set[tuple[str, str, str]] = set()
    known_family_realizations: set[tuple[str, str]] = set()
    source_runs: set[str] = set()
    with (SIM / "outputs/terrain_static_provenance_v4/static_split_manifest.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = (row["surface_family"], realization(row["surface_realization"]), row["source_run_id"])
            known.add(key); known_family_realizations.add(key[:2]); source_runs.add(row["source_run_id"])
    with (SIM / "outputs/terrain_transition_aware_v2_1_80_20/split_manifest.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = (row["surface_family"], realization(row["surface_realization"]), row["run_id"])
            known.add(key); known_family_realizations.add(key[:2]); source_runs.add(row["run_id"])
    for path in (V4 / "fresh_test/fresh_manifest.csv", SIM / "outputs/terrain_transition_v1_pilot/manifest.csv"):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                key = (row["surface_family"], realization(row["surface_realization"]), row["run_id"])
                known.add(key); known_family_realizations.add(key[:2]); source_runs.add(row["run_id"])
    proposed = {(family, str(surface), f"transition_{case}_{family}_{surface:02d}_{run:02d}") for family in FAMILIES for surface in REALIZATIONS for run in RUN_INDICES for case in "ABCD"}
    proposed_family_realizations = {item[:2] for item in proposed}
    return {"known_source_runs": len(source_runs), "known_family_realization_run_records": len(known), "known_family_realizations": len(known_family_realizations), "source_run_leakage": bool({item[2] for item in proposed} & source_runs), "family_realization_leakage": bool(proposed_family_realizations & known_family_realizations), "family_realization_source_run_leakage": bool(proposed & known), "family_overlap_unavoidable": True, "new_families_available": False, "unseen_realizations": sorted({f"{family}:s{int(surface):02d}" for family, surface, _ in proposed})}


def host_metrics(path: Path, normalizer, runs) -> tuple[dict[str, object], list[dict[str, object]]]:
    records = []
    for run in runs:
        trace, row = run.fusion10, run.metadata
        windows = np.asarray([trace[end - 49:end + 1] for end in range(49, 800)], np.float32)
        prediction = np.full(800, -1, np.int8); prediction[49:] = np.argmax(predict_tflite(path, normalizer.transform(windows)), axis=1)
        target, source = LABEL[row["terrain_after"]], LABEL[row["terrain_before"]]
        t1 = stable(prediction, target); post = prediction[T0:]
        records.append({"run_id": row["run_id"], "case_id": row["case_id"], "detected": t1 is not None, "t1_ms": None if t1 is None else t1-T0, "occupancy": float(np.mean(post == target)), "switches": int(np.count_nonzero(np.diff(post) != 0)), "pre_source_accuracy": float(np.mean(prediction[550:T0] == source)), "post_target_accuracy": float(np.mean(post == target)), "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest()})
    aggregate = aggregate_transition(records)
    for case in aggregate:
        latencies = [item["t1_ms"] for item in records if item["case_id"] == case and item["t1_ms"] is not None]
        aggregate[case]["t1_max_ms"] = None if not latencies else float(max(latencies))
    return aggregate, records


def execute(out: Path, frozen: dict[str, object]) -> None:
    from run_terrain_transition import CASES, audit, label, run_one
    if not (out / "protocol.json").exists():
        raise RuntimeError("reserve the protocol before generating results")
    audit_record = known_audit()
    if audit_record["source_run_leakage"] or audit_record["family_realization_leakage"] or audit_record["family_realization_source_run_leakage"]:
        raise RuntimeError(f"broader reservation leaks: {audit_record}")
    # Persist the successful disjointness audit before any new physical run.
    (out / "reservation_audit.json").write_text(json.dumps(audit_record, indent=2) + "\n")
    with np.load(STATIC) as values:
        static_x, static_split = values["X"], values["split"]
    _, _, transition_x, _, transition_split = load_transition(50)
    normalizer = ChannelNormalizer.fit(np.concatenate((static_x, transition_x))[np.concatenate((static_split, transition_split)) == "train"])
    runs = [run_one(case, run, family, surface) for family in FAMILIES for surface in REALIZATIONS for case in CASES for run in RUN_INDICES]
    labels = [label(run.oracle, int(run.metadata["transition_sample"])) for run in runs]
    physical_rows, physical = audit(runs, labels)
    if not all(row["valid"] for row in physical_rows):
        raise RuntimeError("invalid broader physical run")
    metrics, records = host_metrics(Path(frozen["strict_int8_path"]), normalizer, runs)
    gates = {case: bool(item["stable_detection_rate"] >= .90) for case, item in metrics.items()}
    gates["C"] = bool(gates["C"] and metrics["C"]["target_occupancy"] >= .80)
    severe = any(item["prediction_switches_total"] > 4 * item["runs"] for item in metrics.values())
    passed = bool(all(gates.values()) and not severe)
    np.savez_compressed(out / "broader_transition_runs.npz", fusion10=np.asarray([run.fusion10 for run in runs], np.float32), oracle=np.asarray([run.oracle for run in runs], np.float32), terrain_gt=np.asarray([run.terrain_gt for run in runs]), run_id=np.asarray([run.metadata["run_id"] for run in runs]))
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(physical_rows[0])); writer.writeheader(); writer.writerows(physical_rows)
    (out / "leakage_audit.json").write_text(json.dumps(audit_record, indent=2) + "\n")
    summary = {"frozen_model": frozen, "physical_audit": physical, "transition": metrics, "records": records, "gates": gates, "severe_persistent_oscillation": severe, "TERRAIN_BROADER_GENERALIZATION_GATE": "PASS" if passed else "FAIL"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=OUT); parser.add_argument("--reserve", action="store_true"); parser.add_argument("--execute", action="store_true"); args = parser.parse_args(); out = args.output_dir.resolve()
    if args.reserve == args.execute: raise ValueError("choose exactly one of --reserve or --execute")
    if args.reserve:
        if out.exists() and any(out.iterdir()): raise FileExistsError(out)
        out.mkdir(parents=True); payload = protocol(); (out / "protocol.json").write_text(json.dumps(payload, indent=2) + "\n"); (out / "reservation_audit.json").write_text(json.dumps(known_audit(), indent=2) + "\n"); print(json.dumps(payload, indent=2)); return
    execute(out, provenance())


if __name__ == "__main__": main()
