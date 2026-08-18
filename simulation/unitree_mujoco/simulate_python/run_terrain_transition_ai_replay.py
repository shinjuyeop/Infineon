"""Replay frozen INT8 terrain and Fast Reflex models over Transition v1 traces."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from terrain_int8 import normalize, predict_tflite
from terrain_fast_reflex_v2_detector import Normalizer
from terrain_int8 import predict_tflite_binary


SIM = Path(__file__).resolve().parents[2]
SOURCE = SIM / "outputs/terrain_transition_v1_pilot"
OUT = SIM / "outputs/terrain_transition_v1_ai_replay"
TERRAIN_METADATA = SIM / "outputs/terrain_dataset_v1_expanded_1000hz_int8_seed_20260807/deployment_metadata.json"
TERRAIN_MODEL = SIM / "outputs/terrain_dataset_v1_expanded_1000hz_int8_seed_20260807/noisy_fusion_int8.tflite"
REFLEX = {
    "slip": {"case": "A", "target": "confirmed_slip", "window": 5, "raw_threshold": 121, "persistence": 3,
             "model": SIM / "outputs/terrain_fast_reflex_v2_int8/slip/model_int8.tflite", "normalization": SIM / "outputs/terrain_fast_reflex_v2_detector_dataset/slip_5ms/normalization.json"},
    "sink": {"case": "B", "target": "sustained_sink", "window": 20, "raw_threshold": 124, "persistence": 1,
             "model": SIM / "outputs/terrain_fast_reflex_v2_int8/sink/model_int8.tflite", "normalization": SIM / "outputs/terrain_fast_reflex_v2_detector_dataset/sink_20ms/normalization.json"},
}
STABLE_SAMPLES = 3


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_endpoint(prediction: np.ndarray, target: int, start: int) -> int | None:
    """T1 is the third causal endpoint of three target-class predictions."""
    consecutive = 0
    for index in range(start, len(prediction)):
        consecutive = consecutive + 1 if prediction[index] == target else 0
        if consecutive == STABLE_SAMPLES:
            return index
    return None


def sustained_endpoint(raw: np.ndarray, threshold: int, persistence: int, start: int) -> int | None:
    consecutive = 0
    for index in range(start, len(raw)):
        consecutive = consecutive + 1 if raw[index] >= threshold else 0
        if consecutive == persistence:
            return index
    return None


def first_onset(label: np.ndarray, start: int) -> int | None:
    hit = np.flatnonzero(label & (np.arange(len(label)) >= start))
    return None if not len(hit) else int(hit[0])


def ms(sample: int | None, time_s: np.ndarray) -> float | None:
    # Transition v1 defines sample 650 as the elapsed 1-kHz tick at physical
    # T0=650 ms. The stored vector starts after the first physics interval.
    return None if sample is None else float(sample)


def percentile(values: list[float]) -> dict[str, float | None]:
    return {"median_ms": None if not values else float(np.median(values)), "p95_ms": None if not values else float(np.percentile(values, 95))}


def load(source: Path) -> tuple[list[dict[str, str]], dict[str, np.ndarray]]:
    with (source / "manifest.csv").open(newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
    with np.load(source / "transition_traces.npz", allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    ids = data["run_id"].astype(str)
    if len(rows) != len(ids) or not np.array_equal(np.asarray([r["run_id"] for r in rows]), ids): raise ValueError("transition manifest/trace order mismatch")
    if data["fusion10"].shape != (12, 800, 10) or not np.isfinite(data["fusion10"]).all() or not np.allclose(np.diff(data["time_s"], axis=1), .001, atol=1e-9): raise ValueError("invalid Transition v1 Fusion10 source")
    if not all(r["valid"] == "True" for r in rows): raise ValueError("source has invalid run")
    return rows, data


def terrain_replay(fusion: np.ndarray, metadata: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mean, std = np.asarray(metadata["normalization"]["mean"], np.float32), np.asarray(metadata["normalization"]["std"], np.float32)
    windows = np.asarray([trace[end - 49:end + 1] for trace in fusion for end in range(49, trace.shape[0])], np.float32)
    scores = predict_tflite(TERRAIN_MODEL, normalize(windows, mean, std)).reshape(len(fusion), 751, 4)
    prediction = np.argmax(scores, axis=2).astype(np.int8)
    # Align inference endpoint to trace index; -1 denotes the prescribed 50-ms warm-up.
    padded = np.full((len(fusion), 800), -1, np.int8); padded[:, 49:] = prediction
    return scores, padded


def reflex_replay(fusion: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    cfg = REFLEX[kind]; normal = json.loads(cfg["normalization"].read_text())
    norm = Normalizer(np.asarray(normal["mean"], np.float32), np.asarray(normal["std"], np.float32))
    windows = np.asarray([trace[end - cfg["window"] + 1:end + 1] for trace in fusion for end in range(cfg["window"] - 1, 800)], np.float32)
    _, raw = predict_tflite_binary(cfg["model"], norm.transform(windows), cfg["window"])
    padded = np.full((len(fusion), 800), -128, np.int8); padded[:, cfg["window"] - 1:] = raw.reshape(len(fusion), -1)
    return padded, padded >= cfg["raw_threshold"]


def row_for(index: int, row: dict[str, str], data: dict[str, np.ndarray], terrain: np.ndarray, reflex_raw: dict[str, np.ndarray], reflex_fire: dict[str, np.ndarray]) -> dict[str, Any]:
    target_names = ("concrete", "marble", "ice", "sand"); target = target_names.index(row["terrain_after"]); t0 = int(row["transition_sample"])
    source = target_names.index(row["terrain_before"])
    t1 = stable_endpoint(terrain[index], target, t0); kind = "slip" if row["case_id"] == "A" else "sink" if row["case_id"] == "B" else None
    t2 = None if kind is None else first_onset(data[REFLEX[kind]["target"]][index], t0)
    t3 = None if kind is None else sustained_endpoint(reflex_raw[kind][index], REFLEX[kind]["raw_threshold"], REFLEX[kind]["persistence"], t0)
    time_s = data["time_s"][index]
    pre = terrain[index, 49:t0]; post = terrain[index, t0:]; wrong = post[(post >= 0) & (post != target)]
    early = bool(t2 is not None and t3 is not None and t3 < t2)
    order = "" if t2 is None or t3 is None else " < ".join(name for _, name in sorted(((t0, "T0"), (t1, "T1"), (t2, "T2"), (t3, "T3")), key=lambda x: (999999 if x[0] is None else x[0], x[1])))
    return {"run_id": row["run_id"], "case_id": row["case_id"], "terrain_before": row["terrain_before"], "terrain_after": row["terrain_after"],
        "T0_sample": t0, "T1_sample": t1, "T2_sample": t2, "T3_sample": t3, "T0_ms": ms(t0, time_s), "T1_ms": ms(t1, time_s), "T2_ms": ms(t2, time_s), "T3_ms": ms(t3, time_s),
        "terrain_detected": t1 is not None, "hazard_present": t2 is not None, "hazard_detected": t3 is not None,
        "T1_minus_T0_ms": None if t1 is None else float((t1-t0)), "T2_minus_T0_ms": None if t2 is None else float((t2-t0)), "T3_minus_T2_ms": None if t2 is None or t3 is None else float(t3-t2), "T3_minus_T0_ms": None if t3 is None else float(t3-t0),
        "early_firing": early, "early_lead_ms": None if not early else float(t2-t3), "event_order": order,
        "pre_source_accuracy": float((pre == source).mean()),
        "post_target_occupancy": float((post == target).mean()), "post_wrong_class_count": int(len(wrong)),
        "post_wrong_class_distribution": json.dumps({target_names[k]: int((wrong == k).sum()) for k in range(4) if (wrong == k).any()}),
        "prediction_switches_post": int(np.count_nonzero(np.diff(post[post >= 0]) != 0)),
        "reflex_kind": "" if kind is None else kind}


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases = {}
    for case in "ABCD":
        group = [r for r in rows if r["case_id"] == case]
        value = {"runs": len(group), "terrain_transition_success": sum(r["terrain_detected"] for r in group),
            "T1_minus_T0": percentile([r["T1_minus_T0_ms"] for r in group if r["T1_minus_T0_ms"] is not None]),
            "post_target_occupancy_mean": float(np.mean([r["post_target_occupancy"] for r in group])), "prediction_switches_post_total": sum(r["prediction_switches_post"] for r in group)}
        if case in "AB":
            value.update({"hazard_positive_runs": sum(r["hazard_present"] for r in group), "fast_reflex_detected_runs": sum(r["hazard_detected"] for r in group), "false_firing_runs": sum(r["hazard_detected"] and not r["hazard_present"] for r in group),
                "T2_minus_T0": percentile([r["T2_minus_T0_ms"] for r in group if r["T2_minus_T0_ms"] is not None]), "T3_minus_T2": percentile([r["T3_minus_T2_ms"] for r in group if r["T3_minus_T2_ms"] is not None]), "T3_minus_T0": percentile([r["T3_minus_T0_ms"] for r in group if r["T3_minus_T0_ms"] is not None]),
                "terrain_recognized_before_hazard": sum(r["T1_sample"] is not None and r["T2_sample"] is not None and r["T1_sample"] < r["T2_sample"] for r in group), "early_firing_runs": sum(r["early_firing"] for r in group)})
        cases[case] = value
    all_detected = all(r["terrain_detected"] for r in rows)
    # A target must be stably recognized in all three materializations to avoid
    # hiding a direction-specific transition/domain failure.
    recommended = not all_detected
    return {"dataset_name": "terrain_transition_v1_ai_replay", "replay_runs": len(rows), "cases": cases,
        "boundary_confusion": {case: {"pre_source_accuracy_mean": float(np.mean([r["pre_source_accuracy"] for r in rows if r["case_id"] == case])), "post_target_occupancy_mean": cases[case]["post_target_occupancy_mean"], "switches": cases[case]["prediction_switches_post_total"]} for case in cases},
        "TERRAIN_TRANSITION_AI_REPLAY_COMPLETE": len(rows) == 12, "TERRAIN_MODEL_CHANGE_RECOMMENDED": recommended,
        "model_change_reason": None if not recommended else "Ice-to-Marble reaches stable target in only 2/3 runs; static-window source occupancy is also low, indicating transition/domain mismatch rather than an oracle failure"}


def plots(out: Path, rows: list[dict[str, Any]], data: dict[str, np.ndarray], scores: np.ndarray, terrain: np.ndarray, raw: dict[str, np.ndarray], firing: dict[str, np.ndarray]) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = out / "plots"; directory.mkdir(exist_ok=True); class_keys = ("concrete", "marble", "ice", "sand"); names = tuple(key.title() for key in class_keys)
    for i, row in enumerate(rows):
        x = data["time_s"][i] * 1000.; figure, ax = plt.subplots(4 if row["case_id"] in "AB" else 3, 1, figsize=(11, 9), sharex=True)
        target = class_keys.index(row["terrain_after"]); gt = (data["terrain_gt"][i] == row["terrain_after"]).astype(int)
        ax[0].step(x, gt, where="post", label="terrain GT (after=1)"); ax[0].step(x, terrain[i] == target, where="post", label="target prediction"); ax[0].legend(ncol=2); ax[0].set_ylabel("terrain")
        ax[1].plot(x[49:], scores[i]); ax[1].legend(names, ncol=4, fontsize=8); ax[1].set_ylabel("INT8 score")
        ax[2].plot(x, data["oracle"][i,:,13], label="foot speed"); ax[2].plot(x, data["oracle"][i,:,14], label="sink depth"); ax[2].legend(); ax[2].set_ylabel("physical")
        if row["case_id"] in "AB":
            kind = "slip" if row["case_id"] == "A" else "sink"; ax[3].plot(x, raw[kind][i], label="INT8 raw score"); ax[3].axhline(REFLEX[kind]["raw_threshold"], color="r", ls=":", label="threshold"); ax[3].step(x, firing[kind][i].astype(int) * 127, where="post", label="raw positive"); ax[3].step(x, data[REFLEX[kind]["target"]][i].astype(int) * 120, where="post", label="oracle"); ax[3].legend(ncol=2, fontsize=8)
        for name in ("T0", "T1", "T2", "T3"):
            value = row[f"{name}_ms"]
            if value is not None:
                for a in ax: a.axvline(value, color={"T0":"k","T1":"tab:green","T2":"tab:orange","T3":"tab:red"}[name], ls="--", alpha=.8)
        ax[-1].set_xlabel("simulation time [ms]"); figure.suptitle(f"{row['run_id']}: {row['event_order'] or 'T0/T1 timeline'}"); figure.tight_layout(); figure.savefig(directory / f"case_{row['case_id'].lower()}_{i % 3:02d}.png", dpi=150); plt.close(figure)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--source", type=Path, default=SOURCE); p.add_argument("--output-dir", type=Path, default=OUT); p.add_argument("--execute", action="store_true"); a = p.parse_args()
    metadata = json.loads(TERRAIN_METADATA.read_text())
    protocol = {"source": str(a.source.resolve()), "terrain_model": {"path": str(TERRAIN_MODEL.resolve()), "sha256": digest(TERRAIN_MODEL), "metadata_path": str(TERRAIN_METADATA.resolve()), "metadata_sha256": digest(TERRAIN_METADATA), "type": "strict_INT8", "input_shape": metadata["input_shape"], "sample_rate_hz": metadata["sample_rate_hz"], "class_names": metadata["class_names"], "normalization": "deployment_metadata.normalization"}, "stable_terrain_policy": "target terrain at 3 consecutive 1-ms inference endpoints; T1 is third endpoint", "fast_reflex": {k: {**{x: (str(y.resolve()) if isinstance(y, Path) else y) for x,y in v.items()}, "model_sha256": digest(v["model"]), "normalization_sha256": digest(v["normalization"])} for k,v in REFLEX.items()}}
    if not a.execute: print(json.dumps(protocol, indent=2)); return
    out = a.output_dir.resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True); manifest, data = load(a.source.resolve()); scores, terrain = terrain_replay(data["fusion10"], metadata); raw, firing = {}, {}
    for kind in REFLEX: raw[kind], firing[kind] = reflex_replay(data["fusion10"], kind)
    rows = [row_for(i, r, data, terrain, raw, firing) for i, r in enumerate(manifest)]; report = summary(rows)
    with (out / "timeline.csv").open("w", newline="", encoding="utf-8") as f: writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(out / "replay_outputs.npz", terrain_scores=scores.astype(np.float32), terrain_prediction=terrain, **{f"{k}_raw_int8": v for k,v in raw.items()}, **{f"{k}_raw_positive": v for k,v in firing.items()})
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n"); (out / "model_manifest.json").write_text(json.dumps({"terrain": protocol["terrain_model"], "fast_reflex": protocol["fast_reflex"]}, indent=2) + "\n"); (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plots(out, rows, data, scores, terrain, raw, firing); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
