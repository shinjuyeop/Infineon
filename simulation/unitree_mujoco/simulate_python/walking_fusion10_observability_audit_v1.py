"""Pure helpers for the Walking Fusion10 observability diagnostic audit.

The helpers in this module deliberately build diagnostic, causal probes.  They
do not expose any production model, normalization, or runtime-threshold write
path.  A label at endpoint ``t`` may consume Fusion10 samples no later than
``t``; future samples are used only to define the explicitly diagnostic risk
horizon target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


SEED = 20260820
CHANNELS = (
    "foot_force_1", "foot_force_2", "foot_force_3", "foot_force_4",
    "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
)
TERRAIN_NAMES = ("concrete", "marble", "ice", "sand")
TERRAIN_LABELS = {name: index for index, name in enumerate(TERRAIN_NAMES)}
PHASES = ("TOUCHDOWN", "LOADING", "MID_STANCE", "PUSH_OFF")
FORBIDDEN_OUTER_PARTS = (
    "walking_hazard_slip_nested_calibration_v2/outer_validation_traces.npz",
    "walking_bounded_retraining_v1_sink_holdout/traces.npz",
    "walking_terrain_transition_v1_pilot/transition_traces.npz",
)


def assert_diagnostic_source(path: Path | str) -> None:
    """Fail closed if a diagnostic path names locked outer/spatial arrays."""
    portable = str(path).replace("\\", "/")
    if any(value in portable for value in FORBIDDEN_OUTER_PARTS):
        raise PermissionError(f"outer/spatial diagnostic arrays are read-only and locked: {path}")


def causal_windows(
    values: np.ndarray, endpoints: np.ndarray, window_ms: int
) -> np.ndarray:
    """Return native-1 kHz windows with inclusive endpoint and no future data."""
    trace = np.asarray(values)
    endpoint = np.asarray(endpoints, dtype=np.int64)
    if trace.ndim != 2 or trace.shape[1] != len(CHANNELS) or window_ms <= 0:
        raise ValueError("expected (samples,10) Fusion10 and a positive window")
    if endpoint.ndim != 1:
        raise ValueError("endpoints must be one-dimensional")
    if np.any(endpoint < window_ms - 1) or np.any(endpoint >= len(trace)):
        raise ValueError("endpoint cannot provide a complete causal window")
    return np.asarray(
        [trace[item - window_ms + 1:item + 1] for item in endpoint],
        dtype=np.float32,
    )


def feature_names() -> tuple[str, ...]:
    groups = ("mean", "std", "minimum", "maximum", "first", "last", "delta", "diff_rms")
    return tuple(f"{group}__{channel}" for group in groups for channel in CHANNELS)


def window_features(windows: np.ndarray) -> np.ndarray:
    """Fixed 80-value summary probe; every value is causal within its window."""
    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != len(CHANNELS) or not len(values):
        raise ValueError("expected non-empty (windows,time,10) input")
    difference = np.diff(values, axis=1)
    diff_rms = (
        np.sqrt(np.mean(np.square(difference), axis=1))
        if values.shape[1] > 1 else np.zeros((len(values), len(CHANNELS)))
    )
    result = np.concatenate(
        (
            values.mean(1), values.std(1), values.min(1), values.max(1),
            values[:, 0], values[:, -1], values[:, -1] - values[:, 0], diff_rms,
        ),
        axis=1,
    )
    if result.shape[1] != len(feature_names()) or not np.isfinite(result).all():
        raise ValueError("invalid diagnostic window features")
    return result.astype(np.float32)


def phase_balanced_indices(
    labels: np.ndarray,
    phases: np.ndarray,
    *,
    balance_class: bool,
    balance_phase: bool,
    cap_per_cell: int | None = None,
) -> np.ndarray:
    """Deterministically take equal counts across requested class/phase cells."""
    y = np.asarray(labels)
    gait = np.asarray(phases).astype(str)
    if y.shape != gait.shape or y.ndim != 1 or not len(y):
        raise ValueError("labels/phases must be non-empty and aligned")
    keys = np.asarray([
        f"{int(label) if balance_class else '*'}::{phase if balance_phase else '*'}"
        for label, phase in zip(y, gait)
    ])
    groups = [np.flatnonzero(keys == key) for key in sorted(set(keys))]
    groups = [group for group in groups if len(group)]
    target = min(len(group) for group in groups)
    if cap_per_cell is not None:
        if cap_per_cell <= 0:
            raise ValueError("cap_per_cell must be positive")
        target = min(target, cap_per_cell)
    selected: list[int] = []
    for group in groups:
        positions = np.linspace(0, len(group) - 1, target).round().astype(int)
        selected.extend(group[positions].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def split_integrity(
    run_ids: np.ndarray,
    split: np.ndarray,
    episode_ids: np.ndarray | None = None,
) -> dict[str, object]:
    """Audit explicit run and ``(run, contact-episode)`` split ownership."""
    runs = np.asarray(run_ids).astype(str)
    ownership = np.asarray(split).astype(str)
    if runs.shape != ownership.shape:
        raise ValueError("run/split arrays must align")
    episodes = (
        np.zeros(len(runs), dtype=int)
        if episode_ids is None else np.asarray(episode_ids, dtype=int)
    )
    if episodes.shape != runs.shape:
        raise ValueError("episode array must align")
    leaks = sorted({run for run in runs if len(set(ownership[runs == run])) != 1})
    episode_leaks = sorted({
        f"{run}:{episode}"
        for run, episode in zip(runs, episodes)
        if len(set(ownership[(runs == run) & (episodes == episode)])) != 1
    })
    return {
        "run_leakage_count": len(leaks),
        "episode_leakage_count": len(episode_leaks),
        "leaking_runs": leaks,
        "leaking_episodes": episode_leaks,
        "split_leakage_count": len(leaks) + len(episode_leaks),
    }


@dataclass(frozen=True)
class LinearProbe:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: np.ndarray
    classes: np.ndarray

    @property
    def parameter_count(self) -> int:
        return int(self.coefficient.size + self.intercept.size)

    def decision(self, values: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(values, np.float64) - self.mean) / self.scale
        return normalized @ self.coefficient.T + self.intercept

    def predict(self, values: np.ndarray) -> np.ndarray:
        score = self.decision(values)
        if score.shape[1] == 1:
            return np.where(score[:, 0] >= 0.0, self.classes[1], self.classes[0])
        return self.classes[np.argmax(score, axis=1)]

    def positive_score(self, values: np.ndarray) -> np.ndarray:
        score = self.decision(values)
        if score.shape[1] != 1 or len(self.classes) != 2:
            raise ValueError("positive_score is binary-only")
        clipped = np.clip(score[:, 0], -40.0, 40.0)
        return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float64)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            mean=self.mean,
            scale=self.scale,
            coefficient=self.coefficient,
            intercept=self.intercept,
            classes=self.classes,
        )

    @classmethod
    def load(cls, path: Path) -> "LinearProbe":
        with np.load(path, allow_pickle=False) as packed:
            return cls(**{key: packed[key] for key in packed.files})


def fit_logistic_probe(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> LinearProbe:
    """Fit the single predeclared deterministic logistic diagnostic probe."""
    from sklearn.linear_model import LogisticRegression

    x = np.asarray(values, np.float64)
    y = np.asarray(labels)
    if x.ndim != 2 or len(x) != len(y) or len(np.unique(y)) < 2:
        raise ValueError("probe needs aligned samples and at least two classes")
    fitted_mean = x.mean(0) if mean is None else np.asarray(mean, np.float64)
    fitted_scale = x.std(0) if scale is None else np.asarray(scale, np.float64)
    fitted_scale = fitted_scale.copy()
    fitted_scale[fitted_scale < 1e-8] = 1.0
    normalized = (x - fitted_mean) / fitted_scale
    estimator = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=SEED,
        solver="lbfgs",
        tol=1e-7,
    )
    estimator.fit(normalized, y)
    return LinearProbe(
        mean=fitted_mean,
        scale=fitted_scale,
        coefficient=np.asarray(estimator.coef_, np.float64),
        intercept=np.asarray(estimator.intercept_, np.float64),
        classes=np.asarray(estimator.classes_),
    )


def classification_metrics(
    truth: np.ndarray, prediction: np.ndarray, class_count: int = 4
) -> dict[str, object]:
    y = np.asarray(truth, dtype=int)
    pred = np.asarray(prediction, dtype=int)
    if y.shape != pred.shape or not len(y):
        raise ValueError("truth/prediction must be non-empty and aligned")
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(confusion, (y, pred), 1)
    support = confusion.sum(1)
    recall = np.divide(
        np.diag(confusion), support, out=np.zeros(class_count, float), where=support > 0
    )
    distribution = np.bincount(pred, minlength=class_count) / len(pred)
    return {
        "accuracy": float(np.mean(y == pred)),
        "macro_accuracy": float(recall[support > 0].mean()),
        "worst_class_recall": float(recall[support > 0].min()),
        "majority_class_prediction_rate": float(distribution.max()),
        "recall": recall,
        "prediction_distribution": distribution,
        "confusion": confusion,
    }


def terrain_collapse_gate(metrics: dict[str, object]) -> dict[str, bool]:
    """Walking gate that cannot pass through frozen-relative retention alone."""
    macro = float(metrics["macro_accuracy"])
    worst = float(metrics["worst_class_recall"])
    majority = float(metrics["majority_class_prediction_rate"])
    collapsed = worst < 0.10 or majority > 0.70
    return {
        "walking_macro_accuracy_at_least_0p70": macro >= 0.70,
        "worst_class_recall_at_least_0p50": worst >= 0.50,
        "majority_prediction_rate_at_most_0p60": majority <= 0.60,
        "class_collapse_absent": not collapsed,
        "pass": macro >= 0.70 and worst >= 0.50 and majority <= 0.60 and not collapsed,
    }


def persistent_fire(scores: np.ndarray, threshold: float, persistence: int, eligible: np.ndarray) -> np.ndarray:
    probability = np.asarray(scores, float)
    allowed = np.asarray(eligible, bool)
    if probability.shape != allowed.shape or persistence <= 0:
        raise ValueError("invalid persistence inputs")
    result = np.zeros(len(probability), bool)
    count = 0
    for index, active in enumerate(allowed & np.isfinite(probability) & (probability >= threshold)):
        count = count + 1 if active else 0
        result[index] = count >= persistence
    return result


def hazard_regions(
    oracle: np.ndarray,
    loaded: np.ndarray,
    touchdown: np.ndarray,
    pre_fall: np.ndarray,
) -> np.ndarray:
    """Partition each sample with post-fall/AIR/touchdown masks taking priority."""
    physical = np.asarray(oracle, bool)
    contact = np.asarray(loaded, bool)
    transient = np.asarray(touchdown, bool)
    valid = np.asarray(pre_fall, bool)
    if not (physical.shape == contact.shape == transient.shape == valid.shape):
        raise ValueError("hazard region inputs must align")
    region = np.full(len(physical), "normal_stable", dtype="<U20")
    onset_values = np.flatnonzero(physical)
    if onset_values.size:
        onset = int(onset_values[0])
        region[max(0, onset - 100):max(0, onset - 20)] = "pre_onset"
        region[max(0, onset - 20):onset] = "near_onset"
        region[onset:] = "post_oracle_onset"
    region[transient] = "touchdown_transient"
    region[~contact] = "AIR"
    region[~valid] = "post_fall_censored"
    return region


def risk_horizon_labels(oracle: np.ndarray, horizon_ms: int) -> tuple[np.ndarray, np.ndarray]:
    """Return future-risk target and the endpoints whose full horizon is observed."""
    physical = np.asarray(oracle, bool)
    if physical.ndim != 1 or horizon_ms <= 0:
        raise ValueError("expected one-dimensional oracle and positive horizon")
    target = np.zeros(len(physical), bool)
    known = np.arange(len(physical)) + horizon_ms < len(physical)
    positive = np.flatnonzero(physical)
    for onset in positive:
        target[max(0, onset - horizon_ms):onset + 1] = True
    return target, known


def anticipation_semantics(oracle: np.ndarray, firing: np.ndarray) -> dict[str, object]:
    physical = np.flatnonzero(np.asarray(oracle, bool))
    detected = np.flatnonzero(np.asarray(firing, bool))
    onset = None if not physical.size else int(physical[0])
    first = None if not detected.size else int(detected[0])
    post = None if onset is None else next((int(value) for value in detected if value >= onset), None)
    return {
        "oracle_onset": onset,
        "first_firing": first,
        "anticipation": bool(onset is not None and first is not None and first < onset),
        "post_onset_detection": post is not None,
        "latency_ms": None if post is None else post - int(onset),
    }


def contact_reference_features(
    fusion10: np.ndarray,
    endpoint: int,
    episode_id: np.ndarray,
    loaded: np.ndarray,
) -> np.ndarray:
    """Causal state from the first loaded sample of the endpoint's contact episode."""
    values = np.asarray(fusion10, float)
    episodes = np.asarray(episode_id, int)
    contact = np.asarray(loaded, bool)
    if endpoint < 0 or endpoint >= len(values) or not contact[endpoint] or episodes[endpoint] < 0:
        raise ValueError("endpoint must be inside a loaded contact episode")
    candidates = np.flatnonzero(
        (np.arange(len(values)) <= endpoint)
        & contact
        & (episodes == episodes[endpoint])
    )
    if not len(candidates):
        raise ValueError("missing causal contact reference")
    reference = int(candidates[0])
    history = values[reference:endpoint + 1]
    delta = values[endpoint] - values[reference]
    accumulated_delta = history.mean(0) - values[reference]
    fsr = history[:, :4].sum(1)
    result = np.r_[
        endpoint - reference,
        delta,
        accumulated_delta,
        fsr[-1] - fsr[0],
        fsr.mean() - fsr[0],
    ]
    if result.shape != (23,) or not np.isfinite(result).all():
        raise ValueError("invalid contact-reference feature")
    return result.astype(np.float32)


def pearson_causal(feature: np.ndarray, physical: np.ndarray) -> float | None:
    x = np.asarray(feature, float)
    y = np.asarray(physical, float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 3 or np.std(x[finite]) < 1e-12 or np.std(y[finite]) < 1e-12:
        return None
    return float(np.corrcoef(x[finite], y[finite])[0, 1])
