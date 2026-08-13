"""Validation-only threshold/persistence selection for frozen Fast Reflex v2 models.

This tool reads only validation windows and existing Keras artifacts.  It never
builds a dataset, trains, writes models, or reads the sealed final-test split.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

PERSISTENCE = (1, 2, 3, 5, 8)
PRIMARY = {"slip": (5, 10), "sink": (10, 20, 30)}
POOLING = {"slip": "GAP + GlobalMax", "sink": "GAP"}
PARAMETERS = {"slip": 1237, "sink": 1221}
SEED = 20260812
DEFAULT_DATASET = Path("../../outputs/terrain_fast_reflex_v2_detector_dataset")
DEFAULT_MODELS = {"slip": Path("../../outputs/terrain_fast_reflex_v2_detector_training_slip"), "sink": Path("../../outputs/terrain_fast_reflex_v2_detector_training_sink")}
DEFAULT_OUTPUT = Path("../../outputs/terrain_fast_reflex_v2_detector_validation_selection")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--slip-model-dir", type=Path, default=DEFAULT_MODELS["slip"])
    p.add_argument("--sink-model-dir", type=Path, default=DEFAULT_MODELS["sink"])
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def stable_starts(scores: np.ndarray, threshold: float, persistence: int) -> list[int]:
    """Return starts of all non-overlapping stable firing segments at 1 ms stride."""
    starts: list[int] = []
    run = 0
    for i, above in enumerate(scores >= threshold):
        run = run + 1 if above else 0
        if run == persistence:
            starts.append(i - persistence + 1)
    return starts


def endpoint_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= threshold
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
    return {"precision": tp / max(tp + fp, 1), "endpoint_recall": tp / max(tp + fn, 1),
            "endpoint_f1": 2 * tp / max(2 * tp + fp + fn, 1), "endpoint_fpr": fp / max(fp + tn, 1)}


def percentile(values: list[int], q: float) -> float | None:
    return None if not values else float(np.percentile(values, q))


def replay(data: dict[str, np.ndarray], scores: np.ndarray, threshold: float, persistence: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Causal per-run replay; a pre-onset stable start is a false alarm."""
    rows: list[dict[str, Any]] = []
    for run_id in np.unique(data["run_id"]):
        mask = data["run_id"] == run_id
        order = np.argsort(data["endpoint_ms"][mask])
        y, q, ms = data["y"][mask][order], scores[mask][order], data["endpoint_ms"][mask][order]
        onset_i = int(np.flatnonzero(y == 1)[0]) if np.any(y == 1) else None
        starts = stable_starts(q, threshold, persistence)
        first = starts[0] if starts else None
        post = None if onset_i is None else next((s for s in starts if s >= onset_i), None)
        rows.append({"run_id": str(run_id), "family": str(data["family"][mask][0]), "mode": str(data["mode"][mask][0]),
                     "target_onset_ms": None if onset_i is None else int(ms[onset_i]),
                     "stable_firing_ms": None if first is None else int(ms[first]),
                     "post_onset_stable_ms": None if post is None else int(ms[post]),
                     "negative_only": onset_i is None,
                     "pre_onset_firing": onset_i is not None and first is not None and first < onset_i,
                     "post_onset_detection": post is not None})
    neg, pos = [r for r in rows if r["negative_only"]], [r for r in rows if not r["negative_only"]]
    neg_fires = sum(r["stable_firing_ms"] is not None for r in neg)
    pre_fires = sum(r["pre_onset_firing"] for r in pos)
    detected = sum(r["post_onset_detection"] for r in pos)
    latency = [r["post_onset_stable_ms"] - r["target_onset_ms"] for r in pos if r["post_onset_detection"]]
    leads = [r["target_onset_ms"] - r["stable_firing_ms"] for r in pos if r["pre_onset_firing"]]
    metrics = {"negative_only_runs": len(neg), "positive_runs": len(pos),
               "negative_only_run_fpr": neg_fires / max(len(neg), 1),
               "pre_onset_false_alarm_rate": pre_fires / max(len(pos), 1),
               "overall_causal_run_fpr": (neg_fires + pre_fires) / max(len(rows), 1),
               "run_recall": detected / max(len(pos), 1), "latency_median_ms": percentile(latency, 50),
               "latency_p95_ms": percentile(latency, 95), "latency_max_ms": None if not latency else max(latency),
               "anticipation_runs": len(leads), "anticipation_lead_median_ms": percentile(leads, 50),
               "anticipation_lead_p95_ms": percentile(leads, 95), "anticipation_lead_max_ms": None if not leads else max(leads)}
    return rows, metrics


def threshold_candidates(scores: np.ndarray) -> list[float]:
    # Dense empirical validation-score quantiles avoid an arbitrary 0.5 grid.
    # The sentinel boundaries also explicitly represent always-fire/never-fire.
    values = np.unique(np.quantile(scores.astype(float), np.linspace(0, 1, 101)))
    return [float(np.nextafter(values[0], -np.inf)), *values.tolist(), float(np.nextafter(values[-1], np.inf))]


def select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    allowed = [r for r in rows if r["overall_causal_run_fpr"] <= .05]
    if not allowed:
        return None
    # Frozen ordering: recall, finite shorter p95, then shorter observation window.
    return max(allowed, key=lambda r: (r["run_recall"], -(r["latency_p95_ms"] if r["latency_p95_ms"] is not None else 1e9), -r["window_ms"], r["threshold"]))


def pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if not any((o["run_recall"] >= r["run_recall"] and o["overall_causal_run_fpr"] <= r["overall_causal_run_fpr"] and (o["run_recall"] > r["run_recall"] or o["overall_causal_run_fpr"] < r["overall_causal_run_fpr"])) for o in rows)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def save_plots(directory: Path, detector: str, rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    threshold = np.array([r["threshold"] for r in rows]); recall = np.array([r["run_recall"] for r in rows]); fpr = np.array([r["overall_causal_run_fpr"] for r in rows])
    p95 = np.array([np.nan if r["latency_p95_ms"] is None else r["latency_p95_ms"] for r in rows])
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].scatter(threshold, recall, s=8); ax[0, 0].set(ylabel="run recall", xlabel="threshold")
    ax[0, 1].scatter(threshold, fpr, s=8); ax[0, 1].axhline(.05, color="r", ls="--"); ax[0, 1].set(ylabel="overall causal run FPR", xlabel="threshold")
    ax[1, 0].scatter(fpr, recall, c=threshold, s=10); ax[1, 0].axvline(.05, color="r", ls="--"); ax[1, 0].set(xlabel="run FPR", ylabel="run recall")
    ax[1, 1].scatter(threshold, p95, s=8); ax[1, 1].set(ylabel="p95 latency (ms)", xlabel="threshold")
    fig.suptitle(f"{detector}: validation-only threshold × persistence sweep"); fig.tight_layout(); fig.savefig(directory / f"{detector}_sweep.png", dpi=140); plt.close(fig)


def save_representative_replays(directory: Path, detector: str, data: dict[str, np.ndarray], scores: np.ndarray, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Save at most one true detection, negative false alarm, and anticipation trace."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases = [("true_detection", next((r for r in rows if r["post_onset_detection"]), None)),
             ("negative_false_alarm", next((r for r in rows if r["negative_only"] and r["stable_firing_ms"] is not None), None)),
             ("anticipation", next((r for r in rows if r["pre_onset_firing"]), None))]
    for title, row in cases:
        if row is None:
            continue
        mask = data["run_id"] == row["run_id"]; order = np.argsort(data["endpoint_ms"][mask])
        ms, q, y = data["endpoint_ms"][mask][order], scores[mask][order], data["y"][mask][order]
        fig, ax = plt.subplots(figsize=(8, 3)); ax.plot(ms, q, label="score"); ax.axhline(config["threshold"], color="tab:red", ls="--", label="threshold")
        if np.any(y): ax.axvline(ms[np.flatnonzero(y)[0]], color="tab:green", label="target onset")
        if row["stable_firing_ms"] is not None: ax.axvline(row["stable_firing_ms"], color="tab:purple", label="stable firing")
        ax.set(title=f"{detector}: {title} ({row['run_id']})", xlabel="endpoint time (ms)", ylabel="score", ylim=(-.02, 1.02)); ax.legend(loc="best"); fig.tight_layout(); fig.savefig(directory / f"{detector}_{title}.png", dpi=140); plt.close(fig)


def subgroup_rows(replay_rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    result = []
    for value in sorted({r[key] for r in replay_rows}):
        rs = [r for r in replay_rows if r[key] == value]; pos = [r for r in rs if not r["negative_only"]]
        lat = [r["post_onset_stable_ms"] - r["target_onset_ms"] for r in pos if r["post_onset_detection"]]
        result.append({key: value, "runs": len(rs), "target_positive_runs": len(pos), "run_recall": sum(r["post_onset_detection"] for r in pos) / max(len(pos), 1), "stable_firing_runs": sum(r["stable_firing_ms"] is not None for r in rs), "negative_or_pre_onset_false_firing_runs": sum((r["negative_only"] and r["stable_firing_ms"] is not None) or r["pre_onset_firing"] for r in rs), "latency_median_ms": percentile(lat, 50)})
    return result


def main() -> None:
    args = parse_args()
    if not args.execute:
        print("Dry run only. Existing validation models only; final test remains sealed."); return
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing non-empty {out}")
    out.mkdir(parents=True); plots = out / "plots"; plots.mkdir()
    dataset = args.dataset_dir.resolve(); roots = {"slip": args.slip_model_dir.resolve(), "sink": args.sink_model_dir.resolve()}
    import tensorflow as tf
    selected: dict[str, dict[str, Any]] = {}
    for detector in PRIMARY:
        sweep: list[dict[str, Any]] = []
        for window in PRIMARY[detector]:
            source = dataset / f"{detector}_{window}ms" / "validation.npz"
            with np.load(source, allow_pickle=False) as loaded:
                data = {key: loaded[key] for key in loaded.files}
            model_path = roots[detector] / f"{detector}_{window}ms" / "model.keras"
            model = tf.keras.models.load_model(model_path, compile=False)
            scores = model.predict(data["x"], batch_size=1024, verbose=0).reshape(-1)
            for threshold in threshold_candidates(scores):
                endpoint = endpoint_metrics(data["y"], scores, threshold)
                for persistence in PERSISTENCE:
                    _, run = replay(data, scores, threshold, persistence)
                    sweep.append({"detector": detector, "window_ms": window, "threshold": threshold, "persistence": persistence, **endpoint, **run})
        write_csv(out / f"{detector}_sweep.csv", sweep); write_csv(out / f"{detector}_pareto.csv", pareto(sweep)); save_plots(plots, detector, sweep)
        best = select_candidate(sweep)
        if best is not None:
            window = best["window_ms"]
            with np.load(dataset / f"{detector}_{window}ms" / "validation.npz", allow_pickle=False) as loaded: data = {key: loaded[key] for key in loaded.files}
            model = tf.keras.models.load_model(roots[detector] / f"{detector}_{window}ms" / "model.keras", compile=False)
            scores = model.predict(data["x"], batch_size=1024, verbose=0).reshape(-1)
            selected_replay, _ = replay(data, scores, best["threshold"], best["persistence"])
            write_csv(out / f"{detector}_selected_replay.csv", selected_replay)
            write_csv(out / f"{detector}_family_results.csv", subgroup_rows(selected_replay, "family")); write_csv(out / f"{detector}_mode_results.csv", subgroup_rows(selected_replay, "mode"))
            save_representative_replays(plots, detector, data, scores, best, selected_replay)
        result = {"status": f"{detector.upper()}_V2_VALIDATION_SELECTED" if best else f"{detector.upper()}_VALIDATION_GATE_FAIL", "validation_only": True, "final_test_materialized": 0, "selection_rule": "overall causal run FPR <=5%; then maximum recall; then shorter p95 onset-to-stable latency; then shorter window", "selected": best, "model_path": None if best is None else str(roots[detector] / f"{detector}_{best['window_ms']}ms/model.keras"), "normalization_path": None if best is None else str(dataset / f"{detector}_{best['window_ms']}ms/normalization.json"), "pooling": POOLING[detector], "parameters": PARAMETERS[detector], "seed": SEED, "training_provenance": "user-completed frozen 40-epoch training artifact; read-only model load"}
        (out / f"{detector}_selected.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); selected[detector] = result
    ready = all(v["selected"] is not None for v in selected.values())
    protocol = {"validation_families": ["crosshatch", "rounded_ridges"], "validation_only": True, "train_used_for_selection": False, "final_test_materialized": 0, "final_test_inference": 0, "persistence_samples": PERSISTENCE, "threshold_candidates": "101 empirical validation prediction-score quantiles plus strict below-minimum/above-maximum boundaries", "overall_causal_run_fpr_denominator": "all validation runs; numerator is a negative-only run with any stable firing plus a target-positive run with a stable firing before onset", "post_onset_policy": "a positive run recalls only when a stable firing starts at or after target onset; pre-onset-only firing remains false alarm/anticipation", "FINAL_TEST_READY": ready}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    summary = "# Fast Reflex v2 validation-only selection\n\n" + "\n".join(f"- {d}: {v['status']}" for d, v in selected.items()) + f"\n- FINAL_TEST_READY={'true' if ready else 'false'}\n- Final test materialized: 0\n"
    (out / "validation_summary.md").write_text(summary, encoding="utf-8")
    print(f"V2_VALIDATION_SELECTION_COMPLETE final_test_ready={str(ready).lower()} output={out}")


if __name__ == "__main__":
    main()
