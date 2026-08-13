"""Validation-only physical-feature analysis for the Fast Reflex tilt failure mode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import numpy as np

TRACE_PRE_MS = 50


ALLOWED_SPLITS = frozenset({"train", "validation"})
TEST_FAMILIES = frozenset({"warped_multisine", "smooth_random_patches"})
FSR_MAPPING = {
    "rear_left": 0, "rear_right": 1, "front_left": 2, "front_right": 3,
}
FEATURE_NAMES = (
    "fsr_sum", "front_minus_rear", "left_minus_right", "norm_front_rear",
    "norm_left_right", "fsr_variance", "fsr_cv", "fsr_range",
    "d_fsr_sum", "d_norm_front_rear", "d_norm_left_right", "gyro_x",
    "gyro_y", "gyro_xy_magnitude", "gyro_xy_integral", "accel_magnitude",
)


@dataclass(frozen=True)
class FastReflexTrace:
    metadata: dict[str, str]
    timestamps_s: np.ndarray
    sensors: np.ndarray
    oracle: np.ndarray
    slip: np.ndarray
    sink: np.ndarray
    tilt: np.ndarray
    valid: bool
    invalid_reason: str

    @property
    def sink_or_tilt(self):
        return self.sink | self.tilt


def validate_trace(trace: FastReflexTrace):
    if trace.sensors.shape != (150, 10) or trace.oracle.shape != (150, 16):
        raise ValueError("unexpected corrected-v2 trace shape")
    if any(mask.shape != (150,) for mask in (trace.slip, trace.sink, trace.tilt)):
        raise ValueError("unexpected oracle mask shape")
    if not trace.valid or not np.all(np.isfinite(trace.sensors)) or not np.all(np.isfinite(trace.oracle)):
        raise ValueError("invalid/non-finite corrected-v2 trace")


def assert_validation_only(rows: list[dict[str, str]], splits: set[str]) -> list[int]:
    """Return permitted row indices, failing closed on a test split/family request."""
    if not splits or not splits <= ALLOWED_SPLITS:
        raise ValueError(f"Sand analysis permits train/validation only, got {sorted(splits)}")
    selected = [i for i, row in enumerate(rows) if row["split"] in splits]
    if any(rows[i]["surface_family"] in TEST_FAMILIES for i in selected):
        raise ValueError("Sand test-family materialization safeguard failed")
    expected = {"train": {"multisine", "filtered_random", "sparse_aggregate"},
                "validation": {"crosshatch", "rounded_ridges"}}
    for split in splits:
        actual = {rows[i]["surface_family"] for i in selected if rows[i]["split"] == split}
        if actual != expected[split]:
            raise ValueError(f"{split} family ownership mismatch: {sorted(actual)}")
    return selected


def load_validation_only(source: Path, splits: set[str]) -> tuple[list[FastReflexTrace], list[dict[str, str]]]:
    """Load only trace objects whose manifest ownership is train/validation.

    The source arrays are monolithic NPZ containers, so their central arrays must be opened,
    but no test row is indexed, copied into a trace, returned, scored, or inferred.
    """
    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("derived_artifact_revision") != 2:
        raise ValueError("source must be corrected-v2")
    with (source / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    indices = assert_validation_only(rows, splits)
    with np.load(source / "inputs_fusion10.npz", allow_pickle=False) as fusion, \
         np.load(source / "oracle_diagnostics.npz", allow_pickle=False) as diagnostic:
        fusion_ids, oracle_ids = fusion["run_id"].astype(str), diagnostic["run_id"].astype(str)
        manifest_ids = np.asarray([row["run_id"] for row in rows])
        if not (np.array_equal(manifest_ids, fusion_ids) and np.array_equal(manifest_ids, oracle_ids)):
            raise ValueError("manifest/input/oracle run ordering does not match")
        traces = []
        selected_rows = []
        for index in indices:
            row = rows[index]
            trace = FastReflexTrace(
                metadata=dict(row), timestamps_s=np.asarray(fusion["sample_time_s"][index]),
                sensors=np.asarray(fusion["sensors"][index]), oracle=np.asarray(diagnostic["oracle"][index]),
                slip=np.asarray(diagnostic["slip"][index], dtype=bool),
                sink=np.asarray(diagnostic["sink"][index], dtype=bool),
                tilt=np.asarray(diagnostic["tilt"][index], dtype=bool),
                valid=bool(int(row["valid"])), invalid_reason=row["invalid_reason"],
            )
            validate_trace(trace)
            traces.append(trace); selected_rows.append(row)
    if any(t.metadata["surface_family"] in TEST_FAMILIES for t in traces):
        raise AssertionError("test family escaped validation-only loader")
    return traces, selected_rows


def physical_features(sensors: np.ndarray, window_ms: int) -> np.ndarray:
    """Causal E84-computable features for each endpoint of one or more traces."""
    values = np.asarray(sensors, dtype=np.float64)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[2] != 10 or window_ms < 1:
        raise ValueError("expected (runs,time,10) Fusion10 data and positive window")
    result = np.zeros((*values.shape[:2], len(FEATURE_NAMES)), dtype=np.float64)
    for endpoint in range(values.shape[1]):
        start = max(0, endpoint - window_ms + 1)
        frame, first = values[:, endpoint], values[:, start]
        fsr, first_fsr = frame[:, :4], first[:, :4]
        total, first_total = fsr.sum(1), first_fsr.sum(1)
        rear, front = fsr[:, :2].sum(1), fsr[:, 2:4].sum(1)
        left, right = fsr[:, [0, 2]].sum(1), fsr[:, [1, 3]].sum(1)
        denom = np.maximum(np.abs(total), 1e-6)
        nfr, nlr = (front - rear) / denom, (left - right) / denom
        frear, ffront = first_fsr[:, :2].sum(1), first_fsr[:, 2:4].sum(1)
        fleft, fright = first_fsr[:, [0, 2]].sum(1), first_fsr[:, [1, 3]].sum(1)
        fdenom = np.maximum(np.abs(first_total), 1e-6)
        duration = max(1, endpoint - start)
        window = values[:, start:endpoint + 1]
        gyro_xy = window[:, :, 7:9]
        result[:, endpoint] = np.column_stack((
            total, front - rear, left - right, nfr, nlr, fsr.var(1),
            fsr.std(1) / np.maximum(np.abs(fsr.mean(1)), 1e-6), np.ptp(fsr, axis=1),
            (total - first_total) / duration,
            (nfr - (ffront - frear) / fdenom) / duration,
            (nlr - (fleft - fright) / fdenom) / duration,
            frame[:, 7], frame[:, 8], np.linalg.norm(frame[:, 7:9], axis=1),
            np.sum(np.linalg.norm(gyro_xy, axis=2), axis=1) * 0.001,
            np.linalg.norm(frame[:, 4:7], axis=1),
        ))
    return result[0] if np.asarray(sensors).ndim == 2 else result


def stable_endpoints(scores: np.ndarray, threshold: float, persistence: int) -> np.ndarray:
    active = np.asarray(scores) >= threshold
    if persistence < 1:
        raise ValueError("persistence must be positive")
    if persistence == 1:
        return np.flatnonzero(active)
    hits = np.convolve(active.astype(np.int8), np.ones(persistence, dtype=np.int8), "valid")
    return np.flatnonzero(hits == persistence) + persistence - 1


def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg): return float("nan")
    return float(np.mean(pos[:, None] > neg[None, :]) + .5 * np.mean(pos[:, None] == neg[None, :]))


def pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    if not labels.any(): return float("nan")
    order = np.argsort(-scores, kind="stable"); y = labels[order]
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float(np.sum(precision[y]) / np.sum(y))


@dataclass(frozen=True)
class StandardLogistic:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float

    def score(self, values: np.ndarray) -> np.ndarray:
        z = ((values - self.mean) / self.scale) @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def fit_logistic(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> StandardLogistic:
    from scipy.optimize import minimize
    x, y = np.asarray(x, float), np.asarray(y, float)
    mean, scale = x.mean(0), x.std(0); scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
    weights = weights / weights.mean()
    def objective(params):
        logits = z @ params[:-1] + params[-1]
        loss = np.logaddexp(0.0, logits) - y * logits
        return float(np.mean(weights * loss) + 1e-3 * np.sum(params[:-1] ** 2))
    fit = minimize(objective, np.zeros(z.shape[1] + 1), method="L-BFGS-B", options={"maxiter": 300})
    if not fit.success: raise RuntimeError(f"logistic fit failed: {fit.message}")
    return StandardLogistic(mean, scale, fit.x[:-1], float(fit.x[-1]))
