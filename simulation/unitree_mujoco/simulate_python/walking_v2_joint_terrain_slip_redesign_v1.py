"""Pure contracts and runtime helpers for walking-v2 Terrain/Slip redesign v1.

Only virtual, hardware-reproducible bilateral Fusion10 and contact telemetry are
accepted by the runtime feature path.  Physical Slip arrays are consumed only
by explicitly offline label/evaluation helpers.  There is no Sink runtime head.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression

from walking_v2_bilateral_bounded_training import (
    evaluation_invalid_firing_count,
    originating_risk_window_detections,
)


SAMPLE_RATE_HZ = 1000
ENDPOINT_STRIDE_MS = 10
TRAINING_SEEDS = (202608211, 202608212, 202608213)
TERRAIN_ARCHITECTURES = ("T1", "T2", "T3")
SLIP_ARCHITECTURES = ("S1", "S2", "S3")
TERRAIN_NAMES = ("concrete", "marble", "ice", "sand")
TERRAIN_LABELS = {name: index for index, name in enumerate(TERRAIN_NAMES)}
PHASE_NAMES = {0: "AIR", 1: "TOUCHDOWN", 2: "LOADING", 3: "MID_STANCE", 4: "PUSH_OFF"}
HAZARD_BIN_EDGES_MS = (20, 40, 60, 80, 100)

RUNTIME_INPUT_FIELDS = frozenset({
    "bilateral_canonical", "force_loaded", "contact_age", "gait_phase_code",
})
FORBIDDEN_RUNTIME_FIELDS = frozenset({
    "terrain_name", "terrain_id", "profile_name", "speed_mps", "run_id",
    "variation_index", "foot_world_xyz", "contact_penetration", "fall_flag",
    "pre_fall_valid", "slip_physical_active", "slip_risk_target",
    "sink_physical_active", "sink_risk_target",
})
FORBIDDEN_PATH_TOKENS = ("outer", "holdout", "final_test", "final-test", "spatial")

RESOURCE_CEILINGS = {
    "parameter_count": 30_000,
    "macs_per_tick": 60_000,
    "history_bytes": 32 * 1024,
    "persistent_state_bytes": 4 * 1024,
    "basis": (
        "bounded below the repository's previously verified approximately "
        "59.2k-MAC deployable runtime envelope; no INT8/E84 work is performed"
    ),
}

ARCHITECTURE_SPECS: dict[str, dict[str, object]] = {
    "T1": {
        "task": "terrain", "history_ms": 200, "encoder": "full_stats_200ms",
        "statistics": ["mean", "std", "min", "max", "last", "delta", "mean_abs_diff", "rms"],
        "aggregation": "symmetric_plus_own_minus_other", "projection_width": 0,
        "head": "weighted_multinomial_logistic",
    },
    "T2": {
        "task": "terrain", "history_ms": 200, "encoder": "full_stats_50ms_plus_200ms",
        "statistics": ["mean", "std", "min", "max", "last", "delta", "mean_abs_diff", "rms"],
        "aggregation": "dual_timescale_symmetric_plus_own_minus_other", "projection_width": 24,
        "head": "fixed_relu_projection_plus_weighted_multinomial_logistic",
    },
    "T3": {
        "task": "terrain", "history_ms": 200, "encoder": "depthwise_causal_conv_bank",
        "kernels": [5, 20, 50, 100, 200, 2],
        "aggregation": "symmetric_plus_own_minus_other", "projection_width": 32,
        "head": "fixed_relu_projection_plus_weighted_multinomial_logistic",
    },
    "S1": {
        "task": "slip", "history_ms": 100, "encoder": "full_stats_100ms",
        "target": ["no_or_far", "actionable_0_100ms", "active_evidence"],
        "projection_width": 48, "decision": "argmax_nonzero", "persistence_endpoints": 1,
    },
    "S2": {
        "task": "slip", "history_ms": 200, "encoder": "full_stats_200ms",
        "target": ["no_or_far", "tte_0_20", "tte_21_40", "tte_41_60", "tte_61_80", "tte_81_100", "active_evidence"],
        "projection_width": 64, "decision": "argmax_nonzero", "persistence_endpoints": 1,
    },
    "S3": {
        "task": "slip", "history_ms": 200, "encoder": "full_stats_50ms_plus_200ms",
        "target": ["actionable_risk_head", "active_evidence_head"],
        "projection_width": 32, "risk_threshold": 0.5, "active_threshold": 0.5,
        "risk_persistence_endpoints": 1, "active_persistence_endpoints": 1,
        "timer_only_promotion": False,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ArtifactAccessGuard:
    """Exact-allowlist, forbidden-token barrier with a durable read ledger."""

    def __init__(self, repo_root: Path, allowed_relative_paths: Iterable[str], log_path: Path):
        self.repo_root = Path(repo_root).resolve()
        self.allowed = tuple(sorted(set(str(value) for value in allowed_relative_paths)))
        self.log_path = Path(log_path)
        self.events: list[dict[str, object]] = []
        for relative in self.allowed:
            self._validate_relative(relative, require_allowlisted=False)
        self._write_log()

    @staticmethod
    def _contains_forbidden(relative: str) -> str | None:
        lowered = relative.lower().replace("\\", "/")
        return next((token for token in FORBIDDEN_PATH_TOKENS if token in lowered), None)

    def _validate_relative(self, relative: str, *, require_allowlisted: bool = True) -> Path:
        normalized = Path(relative).as_posix()
        token = self._contains_forbidden(normalized)
        if token is not None:
            raise PermissionError(f"forbidden artifact namespace {token!r}: {normalized}")
        if Path(normalized).is_absolute() or ".." in Path(normalized).parts:
            raise PermissionError(f"artifact path escapes repository: {normalized}")
        if require_allowlisted and normalized not in self.allowed:
            raise PermissionError(f"artifact is not allowlisted: {normalized}")
        resolved = (self.repo_root / normalized).resolve()
        if self.repo_root not in resolved.parents:
            raise PermissionError(f"artifact resolves outside repository: {normalized}")
        return resolved

    def _write_log(self) -> None:
        payload = {
            "version": "walking_v2_artifact_access_log_v1",
            "fail_closed": True,
            "forbidden_tokens": list(FORBIDDEN_PATH_TOKENS),
            "read_event_count": len(self.events),
            "events": self.events,
        }
        self.log_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _start(self, relative: str, operation: str, purpose: str) -> tuple[Path, int]:
        try:
            path = self._validate_relative(relative)
        except PermissionError as error:
            self.events.append({
                "relative_path": Path(relative).as_posix(), "operation": operation,
                "purpose": purpose, "status": "blocked", "error": str(error),
            })
            self._write_log()
            raise
        if not path.is_file():
            raise FileNotFoundError(relative)
        self.events.append({
            "relative_path": Path(relative).as_posix(), "operation": operation,
            "purpose": purpose, "status": "started",
        })
        self._write_log()
        return path, len(self.events) - 1

    def _finish(self, index: int, path: Path, digest: str, byte_count: int) -> None:
        self.events[index].update({
            "status": "completed", "sha256": digest, "byte_count": int(byte_count),
        })
        self._write_log()

    def hash_input(self, relative: str, purpose: str) -> str:
        path, index = self._start(relative, "sha256", purpose)
        digest = hashlib.sha256()
        byte_count = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                byte_count += len(block)
                digest.update(block)
        value = digest.hexdigest()
        self._finish(index, path, value, byte_count)
        return value

    def read_json(self, relative: str, purpose: str) -> object:
        path, index = self._start(relative, "read_json", purpose)
        raw = path.read_bytes()
        self._finish(index, path, hashlib.sha256(raw).hexdigest(), len(raw))
        return json.loads(raw.decode("utf-8"))

    def load_npz(self, relative: str, purpose: str) -> dict[str, np.ndarray]:
        path, index = self._start(relative, "load_npz", purpose)
        with np.load(path, allow_pickle=False) as archive:
            values = {key: archive[key].copy() for key in archive.files}
        digest = hashlib.sha256()
        byte_count = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                byte_count += len(block)
                digest.update(block)
        self._finish(index, path, digest.hexdigest(), byte_count)
        return values

    def assert_complete(self) -> None:
        incomplete = [event for event in self.events if event["status"] != "completed"]
        if incomplete:
            raise RuntimeError(f"artifact barrier has incomplete/blocked events: {incomplete}")


def causal_endpoints(sample_count: int, history_ms: int) -> np.ndarray:
    if history_ms <= 0 or sample_count < history_ms:
        return np.empty(0, dtype=np.int64)
    return np.arange(history_ms - 1, sample_count, ENDPOINT_STRIDE_MS, dtype=np.int64)


def _full_statistics(window: np.ndarray) -> np.ndarray:
    values = np.asarray(window, np.float64)
    difference = np.diff(values, axis=0)
    return np.stack((
        values.mean(axis=0), values.std(axis=0), values.min(axis=0), values.max(axis=0),
        values[-1], values[-1] - values[0], np.mean(np.abs(difference), axis=0),
        np.sqrt(np.mean(np.square(values), axis=0)),
    ), axis=1).reshape(-1).astype(np.float32)


def _causal_conv_statistics(window: np.ndarray) -> np.ndarray:
    values = np.asarray(window, np.float64)
    outputs = [values[-width:].mean(axis=0) for width in (5, 20, 50, 100, 200)]
    outputs.append(values[-1] - values[-2])
    return np.stack(outputs, axis=1).reshape(-1).astype(np.float32)


def encoder_fingerprint(architecture: str) -> str:
    spec = ARCHITECTURE_SPECS[architecture]
    return sha256_json({
        "architecture": architecture, "history_ms": spec["history_ms"],
        "encoder": spec["encoder"], "canonical_order": ["left", "right"],
        "aggregation": spec.get("aggregation", "own_other_contact"),
    })


def runtime_feature(
    architecture: str,
    side: int,
    endpoint: int,
    bilateral_canonical: np.ndarray,
    force_loaded: np.ndarray,
    contact_age: np.ndarray,
    gait_phase_code: np.ndarray,
    *,
    include_asymmetry: bool = True,
    history_override_ms: int | None = None,
) -> np.ndarray:
    """Return an endpoint-only causal feature vector from runtime observables."""
    if architecture not in ARCHITECTURE_SPECS or side not in (0, 1):
        raise ValueError((architecture, side))
    spec = ARCHITECTURE_SPECS[architecture]
    history = int(spec["history_ms"] if history_override_ms is None else history_override_ms)
    if history <= 1 or history > 200:
        raise ValueError("diagnostic history override must be within 2..200 ms")
    start = endpoint - history + 1
    trace = np.asarray(bilateral_canonical, np.float64)
    loaded = np.asarray(force_loaded, bool)
    age = np.asarray(contact_age, int)
    phase = np.asarray(gait_phase_code, int)
    if endpoint < 0 or trace.ndim != 2 or trace.shape[1] != 20:
        raise ValueError("invalid endpoint or bilateral trace")
    if loaded.shape != (len(trace), 2) or age.shape != loaded.shape or phase.shape != loaded.shape:
        raise ValueError("contact telemetry must align with trace")
    available_start = max(0, start)
    left_window = trace[available_start:endpoint + 1, :10]
    right_window = trace[available_start:endpoint + 1, 10:]
    if start < 0:
        # Startup padding repeats the first already-observed sample.  It retains
        # the full eligible population without reading future samples.
        padding = -start
        left_window = np.vstack((np.repeat(left_window[:1], padding, axis=0), left_window))
        right_window = np.vstack((np.repeat(right_window[:1], padding, axis=0), right_window))
    encoder = str(spec["encoder"])
    if encoder.startswith("full_stats_50ms_plus"):
        left = np.concatenate((_full_statistics(left_window[-50:]), _full_statistics(left_window)))
        right = np.concatenate((_full_statistics(right_window[-50:]), _full_statistics(right_window)))
    elif encoder == "depthwise_causal_conv_bank":
        left, right = _causal_conv_statistics(left_window), _causal_conv_statistics(right_window)
    else:
        left, right = _full_statistics(left_window), _full_statistics(right_window)
    own, other = (left, right) if side == 0 else (right, left)
    symmetric = 0.5 * (own + other)
    parts = [symmetric]
    if include_asymmetry:
        parts.append(own - other)
    force = trace[endpoint, (0, 1, 2, 3, 10, 11, 12, 13)].reshape(2, 4).sum(axis=1)
    total = float(force.sum())
    ratio = float(force[side] / total) if total > 1e-9 else 0.5
    parts.append(np.asarray((
        ratio, 1.0 - ratio, loaded[endpoint, side], loaded[endpoint, 1 - side],
    ), np.float32))
    if str(spec["task"]) == "slip":
        parts.append(np.asarray((
            min(age[endpoint, side], 2000) / 2000.0,
            min(age[endpoint, 1 - side], 2000) / 2000.0,
        ), np.float32))
        parts.append(np.eye(5, dtype=np.float32)[int(phase[endpoint, side])])
        parts.append(np.eye(5, dtype=np.float32)[int(phase[endpoint, 1 - side])])
    feature = np.concatenate(parts).astype(np.float32)
    if not np.all(np.isfinite(feature)):
        raise ValueError("non-finite runtime feature")
    return feature


def raking_weights(factors: np.ndarray, iterations: int = 30) -> np.ndarray:
    """Deterministically equalize all declared marginal groups without dropping rows."""
    groups = np.asarray(factors)
    if groups.ndim != 2 or not len(groups):
        raise ValueError("raking factors must be a nonempty matrix")
    weight = np.ones(len(groups), np.float64)
    for _ in range(iterations):
        for column in range(groups.shape[1]):
            levels, inverse = np.unique(groups[:, column], return_inverse=True)
            target = float(weight.sum()) / len(levels)
            mass = np.bincount(inverse, weights=weight, minlength=len(levels))
            weight *= (target / mass)[inverse]
    weight *= len(weight) / weight.sum()
    return weight


def episode_balanced_weights(target: np.ndarray, balance_unit: np.ndarray) -> np.ndarray:
    """Give every label equal mass and every episode/contact equal mass within label."""
    labels = np.asarray(target)
    units = np.asarray(balance_unit)
    if labels.shape != units.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("episode balance arrays must align")
    weight = np.zeros(len(labels), np.float64)
    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_rows = np.flatnonzero(labels == label)
        label_units, inverse, counts = np.unique(
            units[label_rows], return_inverse=True, return_counts=True
        )
        if not len(label_units):
            raise ValueError("label without a balance unit")
        weight[label_rows] = 1.0 / (len(label_units) * counts[inverse])
    weight *= len(weight) / weight.sum()
    return weight


@dataclass(frozen=True)
class ProjectedLinearModel:
    architecture: str
    seed: int
    mean: np.ndarray
    scale: np.ndarray
    projection: np.ndarray
    projection_bias: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    classes: np.ndarray
    encoder_sha256: str

    def transformed(self, features: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(features, np.float64) - self.mean) / self.scale
        if self.projection.shape[1] == 0:
            return normalized
        hidden = np.maximum(0.0, normalized @ self.projection + self.projection_bias)
        return np.concatenate((normalized, hidden), axis=1)

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        logits = self.transformed(features) @ self.coefficients.T + self.intercept
        if len(self.classes) == 2 and logits.shape[1] == 1:
            positive = 1.0 / (1.0 + np.exp(-np.clip(logits[:, 0], -60.0, 60.0)))
            return np.column_stack((1.0 - positive, positive))
        logits -= logits.max(axis=1, keepdims=True)
        exponential = np.exp(logits)
        return exponential / exponential.sum(axis=1, keepdims=True)

    def predictions(self, features: np.ndarray) -> np.ndarray:
        return self.classes[np.argmax(self.probabilities(features), axis=1)]

    @property
    def parameter_count(self) -> int:
        return int(self.projection.size + self.projection_bias.size + self.coefficients.size + self.intercept.size)

    @property
    def macs(self) -> int:
        return int(self.projection.size + self.coefficients.size)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, kind=np.asarray("projected_linear"), architecture=np.asarray(self.architecture),
            seed=np.asarray(self.seed), mean=self.mean, scale=self.scale,
            projection=self.projection, projection_bias=self.projection_bias,
            coefficients=self.coefficients, intercept=self.intercept, classes=self.classes,
            encoder_sha256=np.asarray(self.encoder_sha256),
        )

    @classmethod
    def load(cls, path: Path) -> "ProjectedLinearModel":
        with np.load(path, allow_pickle=False) as values:
            if str(values["kind"]) != "projected_linear":
                raise ValueError("wrong model kind")
            return cls(
                str(values["architecture"]), int(values["seed"]), values["mean"], values["scale"],
                values["projection"], values["projection_bias"], values["coefficients"],
                values["intercept"], values["classes"], str(values["encoder_sha256"]),
            )


def _weighted_normalization(features: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, np.float64)
    mass = np.asarray(weight, np.float64)
    mean = np.average(values, axis=0, weights=mass)
    variance = np.average(np.square(values - mean), axis=0, weights=mass)
    scale = np.sqrt(variance)
    scale[scale < 1e-12] = 1.0
    return mean, scale


def _fixed_projection(dimension: int, width: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if width == 0:
        return np.empty((dimension, 0), np.float64), np.empty(0, np.float64)
    rng = np.random.default_rng(seed)
    return (
        rng.normal(0.0, 1.0 / np.sqrt(dimension), size=(dimension, width)),
        rng.uniform(-0.25, 0.25, size=width),
    )


def fit_projected_linear(
    architecture: str,
    seed: int,
    features: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    *,
    projection_width_override: int | None = None,
) -> tuple[ProjectedLinearModel, dict[str, object]]:
    values = np.asarray(features, np.float64)
    labels = np.asarray(target)
    weight = np.asarray(sample_weight, np.float64)
    mean, scale = _weighted_normalization(values, weight)
    width = int(
        ARCHITECTURE_SPECS[architecture]["projection_width"]
        if projection_width_override is None else projection_width_override
    )
    projection, bias = _fixed_projection(values.shape[1], width, seed)
    normalized = (values - mean) / scale
    transformed = normalized if not projection.shape[1] else np.concatenate((
        normalized, np.maximum(0.0, normalized @ projection + bias),
    ), axis=1)
    estimator = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=600, tol=1e-6, random_state=seed,
        multi_class="auto",
    ).fit(transformed, labels, sample_weight=weight)
    model = ProjectedLinearModel(
        architecture, seed, mean, scale, projection, bias,
        estimator.coef_.copy(), estimator.intercept_.copy(), estimator.classes_.copy(),
        encoder_fingerprint(architecture),
    )
    health = {
        "optimizer": "LBFGS", "learning_rate": "strong_wolfe_line_search",
        "max_iterations": 600, "early_stop_rule": "gradient_tolerance_1e-6_or_iteration_cap",
        "iterations": int(np.max(estimator.n_iter_)),
        "converged": bool(np.max(estimator.n_iter_) < 600),
    }
    return model, health


@dataclass(frozen=True)
class DualHeadModel:
    architecture: str
    seed: int
    mean: np.ndarray
    scale: np.ndarray
    projection: np.ndarray
    projection_bias: np.ndarray
    risk_coefficients: np.ndarray
    risk_intercept: np.ndarray
    active_coefficients: np.ndarray
    active_intercept: np.ndarray
    encoder_sha256: str

    def transformed(self, features: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(features, np.float64) - self.mean) / self.scale
        hidden = np.maximum(0.0, normalized @ self.projection + self.projection_bias)
        return np.concatenate((normalized, hidden), axis=1)

    @staticmethod
    def _positive(transformed: np.ndarray, coefficients: np.ndarray, intercept: np.ndarray) -> np.ndarray:
        logits = transformed @ coefficients.T + intercept
        return 1.0 / (1.0 + np.exp(-np.clip(logits[:, 0], -60.0, 60.0)))

    def scores(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transformed = self.transformed(features)
        return (
            self._positive(transformed, self.risk_coefficients, self.risk_intercept),
            self._positive(transformed, self.active_coefficients, self.active_intercept),
        )

    @property
    def parameter_count(self) -> int:
        return int(
            self.projection.size + self.projection_bias.size + self.risk_coefficients.size
            + self.risk_intercept.size + self.active_coefficients.size + self.active_intercept.size
        )

    @property
    def macs(self) -> int:
        return int(self.projection.size + self.risk_coefficients.size + self.active_coefficients.size)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, kind=np.asarray("dual_head"), architecture=np.asarray(self.architecture),
            seed=np.asarray(self.seed), mean=self.mean, scale=self.scale,
            projection=self.projection, projection_bias=self.projection_bias,
            risk_coefficients=self.risk_coefficients, risk_intercept=self.risk_intercept,
            active_coefficients=self.active_coefficients, active_intercept=self.active_intercept,
            encoder_sha256=np.asarray(self.encoder_sha256),
        )

    @classmethod
    def load(cls, path: Path) -> "DualHeadModel":
        with np.load(path, allow_pickle=False) as values:
            if str(values["kind"]) != "dual_head":
                raise ValueError("wrong model kind")
            return cls(
                str(values["architecture"]), int(values["seed"]), values["mean"], values["scale"],
                values["projection"], values["projection_bias"], values["risk_coefficients"],
                values["risk_intercept"], values["active_coefficients"], values["active_intercept"],
                str(values["encoder_sha256"]),
            )


def fit_dual_head(
    seed: int,
    features: np.ndarray,
    risk_target: np.ndarray,
    active_target: np.ndarray,
    risk_weight: np.ndarray,
    active_weight: np.ndarray,
) -> tuple[DualHeadModel, dict[str, object]]:
    values = np.asarray(features, np.float64)
    combined_weight = 0.5 * (np.asarray(risk_weight) + np.asarray(active_weight))
    mean, scale = _weighted_normalization(values, combined_weight)
    width = int(ARCHITECTURE_SPECS["S3"]["projection_width"])
    projection, bias = _fixed_projection(values.shape[1], width, seed)
    normalized = (values - mean) / scale
    transformed = np.concatenate((normalized, np.maximum(0.0, normalized @ projection + bias)), axis=1)
    fitted = []
    iterations = []
    for target, weight in ((risk_target, risk_weight), (active_target, active_weight)):
        estimator = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=600, tol=1e-6, random_state=seed,
        ).fit(transformed, target, sample_weight=weight)
        fitted.append((estimator.coef_.copy(), estimator.intercept_.copy()))
        iterations.append(int(np.max(estimator.n_iter_)))
    model = DualHeadModel(
        "S3", seed, mean, scale, projection, bias,
        fitted[0][0], fitted[0][1], fitted[1][0], fitted[1][1], encoder_fingerprint("S3"),
    )
    return model, {
        "optimizer": "LBFGS", "learning_rate": "strong_wolfe_line_search",
        "max_iterations": 600, "early_stop_rule": "gradient_tolerance_1e-6_or_iteration_cap",
        "iterations": max(iterations), "converged": max(iterations) < 600,
    }


def contact_scoped_state(
    raw_activation: np.ndarray,
    endpoints: np.ndarray,
    force_loaded: np.ndarray,
    contact_age: np.ndarray,
    touchdown_transient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mask and own instantaneous evidence separately for each foot/contact.

    No timer can promote evidence.  Every loss of contact and every touchdown
    resets that foot only; therefore previous-contact latch carryover is absent.
    """
    activation = np.asarray(raw_activation, bool)
    endpoint_values = np.asarray(endpoints, int)
    loaded = np.asarray(force_loaded, bool)
    age = np.asarray(contact_age, int)
    touchdown = np.asarray(touchdown_transient, bool)
    if activation.shape != (len(endpoint_values), 2) or loaded.shape != age.shape or loaded.shape != touchdown.shape:
        raise ValueError("state arrays do not align")
    output = np.zeros_like(activation)
    resets = np.full(activation.shape, "none", dtype="<U20")
    owner = np.full(activation.shape, -1, np.int64)
    contact_counter = np.zeros(2, np.int64)
    previous_loaded = np.zeros(2, bool)
    previous_age = np.zeros(2, int)
    for row, endpoint in enumerate(endpoint_values):
        for side in (0, 1):
            is_loaded = bool(loaded[endpoint, side])
            new_touchdown = bool(
                is_loaded and (
                    not previous_loaded[side]
                    or age[endpoint, side] <= previous_age[side]
                    or touchdown[endpoint, side]
                )
            )
            eligible = bool(is_loaded and age[endpoint, side] > 10 and not touchdown[endpoint, side])
            if not is_loaded:
                resets[row, side] = "contact_loss"
            elif new_touchdown or not eligible:
                if new_touchdown:
                    contact_counter[side] += 1
                resets[row, side] = "new_touchdown"
            else:
                output[row, side] = bool(activation[row, side])
                owner[row, side] = contact_counter[side]
            previous_loaded[side] = is_loaded
            previous_age[side] = int(age[endpoint, side])
    return output, resets, owner


def authoritative_owned_detections(
    firing: np.ndarray,
    endpoints: np.ndarray,
    candidate_indices: np.ndarray,
    risk_start_sample: int,
    endpoint_pre_fall: np.ndarray,
) -> np.ndarray:
    """Corrected R4 attribution: pre-fall only and rising edge owned in-window."""
    indices = np.asarray(candidate_indices, int)
    valid = np.asarray(endpoint_pre_fall, bool)
    indices = indices[valid[indices]]
    return originating_risk_window_detections(
        np.asarray(firing, bool), np.asarray(endpoints, int), indices, risk_start_sample,
    )


def invalid_firing_count(air: int, touchdown: int, post_fall_outputs: int) -> int:
    return evaluation_invalid_firing_count(
        air, touchdown, post_fall_outputs, strict_first_fall_censor=True,
    )


def terrain_gate(metrics: dict[str, object]) -> bool:
    ceilings = RESOURCE_CEILINGS
    return bool(
        float(metrics["overall_accuracy"]) >= 0.85
        and float(metrics["macro_recall"]) >= 0.85
        and float(metrics["worst_class_recall"]) >= 0.70
        and float(metrics["sand_recall"]) >= 0.70
        and float(metrics["majority_class_prediction_rate"]) < 0.60
        and float(metrics["minimum_speed_accuracy"]) >= 0.80
        and float(metrics["left_right_accuracy_difference_pp"]) <= 10.0
        and not bool(metrics["class_collapse"])
        and int(metrics["air_terrain_transitions"]) == 0
        and int(metrics["invalid_firings"]) == 0
        and bool(metrics["causal_check_pass"])
        and bool(metrics["bilateral_parity_pass"])
        and int(metrics["parameter_count"]) <= ceilings["parameter_count"]
        and int(metrics["macs_per_tick"]) <= ceilings["macs_per_tick"]
        and int(metrics["history_bytes"]) <= ceilings["history_bytes"]
        and int(metrics["persistent_state_bytes"]) <= ceilings["persistent_state_bytes"]
    )


def slip_gate(metrics: dict[str, object]) -> bool:
    ceilings = RESOURCE_CEILINGS
    return bool(
        float(metrics["actionable_episode_recall"]) >= 0.80
        and float(metrics["minimum_speed_actionable_recall"]) >= 0.70
        and float(metrics["physical_episode_recall"]) >= 0.80
        and float(metrics["affected_foot_accuracy"]) >= 0.90
        and int(metrics["normal_run_fp"]) == 0
        and int(metrics["normal_contact_episode_fp"]) == 0
        and int(metrics["too_early_activations"]) == 0
        and int(metrics["air_firings"]) == 0
        and int(metrics["touchdown_transient_firings"]) == 0
        and int(metrics["invalid_firings"]) == 0
        and int(metrics["post_fall_positive_attributions"]) == 0
        and int(metrics["previous_episode_latch_carryover"]) == 0
        and int(metrics["cross_foot_state_ownership_violations"]) == 0
        and int(metrics["timer_only_promotions"]) == 0
        and float(metrics["median_warning_margin_ms"]) >= 20.0
        and float(metrics["pre_onset_detection_fraction"]) >= 0.80
        and bool(metrics["both_affected_feet_covered"])
        and bool(metrics["causal_check_pass"])
        and bool(metrics["bilateral_parity_pass"])
        and int(metrics["parameter_count"]) <= ceilings["parameter_count"]
        and int(metrics["macs_per_tick"]) <= ceilings["macs_per_tick"]
        and int(metrics["history_bytes"]) <= ceilings["history_bytes"]
        and int(metrics["persistent_state_bytes"]) <= ceilings["persistent_state_bytes"]
    )


def deterministic_terrain_selection(rows: list[dict[str, object]]) -> dict[str, object] | None:
    passing = [row for row in rows if bool(row["gate_pass"])]
    if not passing:
        return None
    return min(passing, key=lambda row: (
        -float(row["macro_recall"]), -float(row["worst_class_recall"]),
        -float(row["overall_accuracy"]), int(row["macs_per_tick"]),
        TERRAIN_ARCHITECTURES.index(str(row["architecture"])), int(row["seed"]),
    ))


def deterministic_slip_selection(rows: list[dict[str, object]]) -> dict[str, object] | None:
    passing = [row for row in rows if bool(row["gate_pass"])]
    if not passing:
        return None
    return min(passing, key=lambda row: (
        -float(row["actionable_episode_recall"]),
        -float(row["minimum_speed_actionable_recall"]),
        -float(row["affected_foot_accuracy"]), -float(row["physical_episode_recall"]),
        -float(row["median_warning_margin_ms"]), int(row["macs_per_tick"]),
        SLIP_ARCHITECTURES.index(str(row["architecture"])), int(row["seed"]),
    ))


def terrain_diagnostic_fallback(rows: list[dict[str, object]]) -> dict[str, object]:
    return min(rows, key=lambda row: (
        -float(row["macro_recall"]), -float(row["worst_class_recall"]),
        -float(row["overall_accuracy"]), int(row["macs_per_tick"]),
        TERRAIN_ARCHITECTURES.index(str(row["architecture"])), int(row["seed"]),
    ))


def slip_diagnostic_fallback(rows: list[dict[str, object]]) -> dict[str, object]:
    return min(rows, key=lambda row: (
        int(row["normal_run_fp"]), int(row["normal_contact_episode_fp"]),
        int(row["too_early_activations"]), -float(row["actionable_episode_recall"]),
        -float(row["minimum_speed_actionable_recall"]), -float(row["affected_foot_accuracy"]),
        SLIP_ARCHITECTURES.index(str(row["architecture"])), int(row["seed"]),
    ))
