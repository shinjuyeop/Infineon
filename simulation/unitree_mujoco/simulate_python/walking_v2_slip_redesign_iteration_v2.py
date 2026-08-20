"""Pure helpers for nested-development walking-v2 Slip redesign iteration v2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from walking_v2_joint_terrain_slip_redesign_v1 import sha256_json


STATE_NAMES = (
    "NORMAL_NO_EVENT", "EARLY_PRECURSOR", "ACTIONABLE_RISK", "PHYSICAL_ACTIVE_EVIDENCE",
)
NORMAL_NO_EVENT, EARLY_PRECURSOR, ACTIONABLE_RISK, PHYSICAL_ACTIVE_EVIDENCE = range(4)
FAMILIES = ("S4-A", "S4-B", "S4-C")
SEEDS = (202608221, 202608222, 202608223)
EARLY_PRECURSOR_MAX_MS = 500
ACTIONABLE_HORIZON_MS = 100

FAMILY_SPECS: dict[str, dict[str, object]] = {
    "S4-A": {
        "history_ms": 200, "base_feature_architecture": "S2",
        "feature": "full_200ms_stats_plus_contact", "projection_width": 32,
        "state_threshold": 0.65, "proposal_threshold": 0.65,
        "early_margin": 0.15, "normal_margin": 0.15, "foot_threshold": 0.50,
        "persistence_endpoints": 2, "hysteresis": 0.05,
        "two_stage": False,
    },
    "S4-B": {
        "history_ms": 200, "base_feature_architecture": "S3",
        "feature": "full_50ms_plus_200ms_stats_plus_contact", "projection_width": 32,
        "state_threshold": 0.70, "proposal_threshold": 0.70,
        "early_margin": 0.18, "normal_margin": 0.18, "foot_threshold": 0.55,
        "persistence_endpoints": 2, "hysteresis": 0.05,
        "two_stage": False,
    },
    "S4-C": {
        "history_ms": 200, "base_feature_architecture": "S3",
        "feature": "dual_timescale_proposal_plus_causal_verifier", "projection_width": 40,
        "state_threshold": 0.65, "proposal_threshold": 0.65,
        "early_margin": 0.25, "normal_margin": 0.20, "foot_threshold": 0.60,
        "persistence_endpoints": 2, "hysteresis": 0.05,
        "two_stage": True,
    },
}

RESOURCE_CEILINGS = {
    "parameters": 30_000, "macs_per_tick": 60_000,
    "history_bytes": 32 * 1024, "state_bytes": 4 * 1024,
}


@dataclass(frozen=True)
class RuntimeStateConfig:
    state_threshold: float
    proposal_threshold: float
    early_margin: float
    normal_margin: float
    foot_threshold: float
    persistence_endpoints: int
    hysteresis: float
    two_stage: bool

    @classmethod
    def from_family(cls, family: str) -> "RuntimeStateConfig":
        spec = FAMILY_SPECS[family]
        return cls(*(
            spec[key] for key in (
                "state_threshold", "proposal_threshold", "early_margin", "normal_margin",
                "foot_threshold", "persistence_endpoints", "hysteresis", "two_stage",
            )
        ))


def make_nested_fold_manifest(metadata: tuple[dict[str, object], ...]) -> dict[str, object]:
    """Create three outer variation groups and two inner mining folds per outer fold."""
    variations = sorted({int(row["variation_index"]) for row in metadata})
    if len(variations) < 3:
        raise ValueError("at least three variation groups are required")
    outer_rows = []
    for outer_index in range(3):
        validation_variations = variations[outer_index::3]
        training_variations = [value for value in variations if value not in validation_variations]
        validation_runs = [
            str(row["run_id"]) for row in metadata
            if int(row["variation_index"]) in validation_variations
        ]
        training_runs = [
            str(row["run_id"]) for row in metadata
            if int(row["variation_index"]) in training_variations
        ]
        inner_rows = []
        for inner_index in range(2):
            mining_validation_variations = training_variations[inner_index::2]
            mining_fit_variations = [
                value for value in training_variations if value not in mining_validation_variations
            ]
            if not mining_fit_variations or not mining_validation_variations:
                raise ValueError("inner mining split has an empty side")
            inner_rows.append({
                "inner_fold": inner_index,
                "fit_variations": mining_fit_variations,
                "mining_variations": mining_validation_variations,
                "fit_run_ids": [
                    str(row["run_id"]) for row in metadata
                    if int(row["variation_index"]) in mining_fit_variations
                ],
                "mining_run_ids": [
                    str(row["run_id"]) for row in metadata
                    if int(row["variation_index"]) in mining_validation_variations
                ],
            })
        outer_rows.append({
            "outer_fold": outer_index,
            "training_variations": training_variations,
            "validation_variations": validation_variations,
            "training_run_ids": training_runs,
            "validation_run_ids": validation_runs,
            "inner_mining_folds": inner_rows,
        })
    manifest = {
        "version": "walking_v2_slip_nested_development_folds_v2",
        "development_corpus_runs": len(metadata), "outer_fold_count": 3,
        "inner_mining_fold_count": 2, "group_key": "variation_index",
        "old_72_48_blind_claim": False, "variations": variations, "outer_folds": outer_rows,
    }
    validate_nested_fold_manifest(manifest)
    return manifest


def validate_nested_fold_manifest(manifest: dict[str, object]) -> None:
    folds = manifest["outer_folds"]
    validation_union: set[str] = set()
    for fold in folds:
        train = set(fold["training_run_ids"])
        validation = set(fold["validation_run_ids"])
        train_variations = set(fold["training_variations"])
        validation_variations = set(fold["validation_variations"])
        if train & validation or train_variations & validation_variations:
            raise ValueError("outer run/variation leakage")
        validation_union.update(validation)
        for inner in fold["inner_mining_folds"]:
            fit = set(inner["fit_run_ids"])
            mining = set(inner["mining_run_ids"])
            if fit & mining or mining & validation or fit & validation:
                raise ValueError("inner mining or outer validation leakage")
            if set(inner["fit_variations"]) & set(inner["mining_variations"]):
                raise ValueError("inner variation leakage")
    if len(validation_union) != int(manifest["development_corpus_runs"]):
        raise ValueError("outer validation folds do not cover the development corpus exactly once")


def weighted_normalization(features: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, np.float64)
    mass = np.asarray(weight, np.float64)
    mean = np.average(values, axis=0, weights=mass)
    variance = np.average(np.square(values - mean), axis=0, weights=mass)
    scale = np.sqrt(variance)
    scale[scale < 1e-12] = 1.0
    return mean, scale


def fixed_projection(dimension: int, width: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (
        rng.normal(0.0, 1.0 / np.sqrt(dimension), size=(dimension, width)),
        rng.uniform(-0.25, 0.25, size=width),
    )


@dataclass(frozen=True)
class SlipV2Model:
    family: str
    seed: int
    mean: np.ndarray
    scale: np.ndarray
    projection: np.ndarray
    projection_bias: np.ndarray
    state_coefficients: np.ndarray
    state_intercept: np.ndarray
    state_classes: np.ndarray
    foot_coefficients: np.ndarray
    foot_intercept: np.ndarray
    proposal_coefficients: np.ndarray
    proposal_intercept: np.ndarray
    feature_sha256: str

    def transformed(self, features: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(features, np.float64) - self.mean) / self.scale
        hidden = np.maximum(0.0, normalized @ self.projection + self.projection_bias)
        return np.concatenate((normalized, hidden), axis=1)

    @staticmethod
    def _binary(transformed: np.ndarray, coefficients: np.ndarray, intercept: np.ndarray) -> np.ndarray:
        value = transformed @ coefficients.T + intercept
        return 1.0 / (1.0 + np.exp(-np.clip(value[:, 0], -60.0, 60.0)))

    def scores(self, features: np.ndarray) -> dict[str, np.ndarray]:
        transformed = self.transformed(features)
        logits = transformed @ self.state_coefficients.T + self.state_intercept
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(axis=1, keepdims=True)
        ordered = np.zeros((len(probability), 4), np.float64)
        for source, state_class in enumerate(self.state_classes):
            ordered[:, int(state_class)] = probability[:, source]
        foot = self._binary(transformed, self.foot_coefficients, self.foot_intercept)
        proposal = self._binary(transformed, self.proposal_coefficients, self.proposal_intercept)
        return {
            "normal": ordered[:, NORMAL_NO_EVENT], "early": ordered[:, EARLY_PRECURSOR],
            "actionable": ordered[:, ACTIONABLE_RISK],
            "active": ordered[:, PHYSICAL_ACTIVE_EVIDENCE], "foot": foot, "proposal": proposal,
        }

    @property
    def parameter_count(self) -> int:
        return int(sum(value.size for value in (
            self.projection, self.projection_bias, self.state_coefficients, self.state_intercept,
            self.foot_coefficients, self.foot_intercept,
            self.proposal_coefficients, self.proposal_intercept,
        )))

    @property
    def macs(self) -> int:
        return int(
            self.projection.size + self.state_coefficients.size
            + self.foot_coefficients.size + self.proposal_coefficients.size
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, family=np.asarray(self.family), seed=np.asarray(self.seed),
            mean=self.mean, scale=self.scale, projection=self.projection,
            projection_bias=self.projection_bias, state_coefficients=self.state_coefficients,
            state_intercept=self.state_intercept, state_classes=self.state_classes,
            foot_coefficients=self.foot_coefficients, foot_intercept=self.foot_intercept,
            proposal_coefficients=self.proposal_coefficients,
            proposal_intercept=self.proposal_intercept, feature_sha256=np.asarray(self.feature_sha256),
        )

    @classmethod
    def load(cls, path: Path) -> "SlipV2Model":
        with np.load(path, allow_pickle=False) as values:
            return cls(
                str(values["family"]), int(values["seed"]), values["mean"], values["scale"],
                values["projection"], values["projection_bias"], values["state_coefficients"],
                values["state_intercept"], values["state_classes"], values["foot_coefficients"],
                values["foot_intercept"], values["proposal_coefficients"],
                values["proposal_intercept"], str(values["feature_sha256"]),
            )


def fit_slip_v2_model(
    family: str,
    seed: int,
    features: np.ndarray,
    state_target: np.ndarray,
    foot_target: np.ndarray,
    state_weight: np.ndarray,
    foot_weight: np.ndarray,
) -> tuple[SlipV2Model, dict[str, object]]:
    values = np.asarray(features, np.float64)
    combined = 0.75 * np.asarray(state_weight) + 0.25 * np.asarray(foot_weight)
    mean, scale = weighted_normalization(values, combined)
    projection, bias = fixed_projection(
        values.shape[1], int(FAMILY_SPECS[family]["projection_width"]), seed,
    )
    normalized = (values - mean) / scale
    transformed = np.concatenate((normalized, np.maximum(0.0, normalized @ projection + bias)), axis=1)
    common = {"C": 1.0, "solver": "lbfgs", "max_iter": 400, "tol": 1e-6, "random_state": seed}
    state = LogisticRegression(multi_class="auto", **common).fit(
        transformed, state_target, sample_weight=state_weight,
    )
    foot = LogisticRegression(**common).fit(transformed, foot_target, sample_weight=foot_weight)
    proposal_target = (np.asarray(state_target) == ACTIONABLE_RISK).astype(int)
    proposal = LogisticRegression(**common).fit(
        transformed, proposal_target, sample_weight=state_weight,
    )
    model = SlipV2Model(
        family, seed, mean, scale, projection, bias, state.coef_.copy(), state.intercept_.copy(),
        state.classes_.copy(), foot.coef_.copy(), foot.intercept_.copy(),
        proposal.coef_.copy(), proposal.intercept_.copy(),
        sha256_json({"family": family, "feature": FAMILY_SPECS[family]["feature"]}),
    )
    iterations = [int(np.max(value.n_iter_)) for value in (state, foot, proposal)]
    return model, {
        "optimizer": "LBFGS", "learning_rate": "strong_wolfe_line_search",
        "max_iterations": 400, "iterations": max(iterations),
        "converged": max(iterations) < 400,
    }


@dataclass(frozen=True)
class StatefulOutput:
    firing: np.ndarray
    reset_reason: np.ndarray
    owner_id: np.ndarray
    persistence_count: np.ndarray
    simultaneous_crossings: int
    selected_left: int
    selected_right: int
    score_differences: tuple[float, ...]


def contact_scoped_runtime_state(
    scores: dict[str, np.ndarray],
    endpoints: np.ndarray,
    force_loaded: np.ndarray,
    contact_age: np.ndarray,
    touchdown_transient: np.ndarray,
    config: RuntimeStateConfig,
) -> StatefulOutput:
    """Apply per-foot persistence/hysteresis, then choose exactly one foot on ties."""
    endpoint_values = np.asarray(endpoints, int)
    loaded = np.asarray(force_loaded, bool)
    age = np.asarray(contact_age, int)
    touchdown = np.asarray(touchdown_transient, bool)
    expected = (len(endpoint_values), 2)
    if any(np.asarray(scores[key]).shape != expected for key in scores):
        raise ValueError("all score arrays must be endpoint x foot")
    if loaded.shape != age.shape or loaded.shape != touchdown.shape:
        raise ValueError("contact arrays must align")
    internal = np.zeros(expected, bool)
    firing = np.zeros(expected, bool)
    reset = np.full(expected, "none", dtype="<U24")
    owner = np.full(expected, -1, np.int64)
    count = np.zeros(expected, np.int16)
    counters = np.zeros(2, np.int64)
    previous_loaded = np.zeros(2, bool)
    previous_age = np.zeros(2, int)
    active = np.zeros(2, bool)
    current_count = np.zeros(2, int)
    differences: list[float] = []
    selected_left = selected_right = simultaneous = 0
    for row, endpoint in enumerate(endpoint_values):
        for side in (0, 1):
            is_loaded = bool(loaded[endpoint, side])
            new_touchdown = bool(
                is_loaded and (
                    not previous_loaded[side] or age[endpoint, side] <= previous_age[side]
                    or touchdown[endpoint, side]
                )
            )
            eligible = bool(is_loaded and age[endpoint, side] > 10 and not touchdown[endpoint, side])
            if not is_loaded:
                active[side] = False
                current_count[side] = 0
                reset[row, side] = "contact_loss"
            elif new_touchdown or not eligible:
                if new_touchdown:
                    counters[side] += 1
                active[side] = False
                current_count[side] = 0
                reset[row, side] = "new_touchdown"
            else:
                action = float(scores["actionable"][row, side])
                early = float(scores["early"][row, side])
                normal = float(scores["normal"][row, side])
                foot = float(scores["foot"][row, side])
                proposal = float(scores["proposal"][row, side])
                threshold_shift = config.hysteresis if active[side] else 0.0
                condition = bool(
                    action >= config.state_threshold - threshold_shift
                    and action - early >= config.early_margin - threshold_shift
                    and action - normal >= config.normal_margin - threshold_shift
                    and foot >= config.foot_threshold - threshold_shift
                    and (not config.two_stage or proposal >= config.proposal_threshold - threshold_shift)
                )
                if condition:
                    current_count[side] += 1
                    active[side] = current_count[side] >= config.persistence_endpoints
                else:
                    if active[side]:
                        reset[row, side] = "score_recovery"
                    active[side] = False
                    current_count[side] = 0
                internal[row, side] = active[side]
                owner[row, side] = counters[side]
            count[row, side] = current_count[side]
            previous_loaded[side] = is_loaded
            previous_age[side] = int(age[endpoint, side])
        if internal[row, 0] and internal[row, 1]:
            simultaneous += 1
            evidence = [
                float(scores["actionable"][row, side] - scores["early"][row, side]
                      + 0.25 * (scores["foot"][row, side] - 0.5))
                for side in (0, 1)
            ]
            selected = 0 if evidence[0] >= evidence[1] else 1
            firing[row, selected] = True
            differences.append(abs(evidence[0] - evidence[1]))
            selected_left += int(selected == 0)
            selected_right += int(selected == 1)
        else:
            firing[row] = internal[row]
    return StatefulOutput(
        firing, reset, owner, count, simultaneous, selected_left, selected_right,
        tuple(differences),
    )


def deterministic_selection(rows: list[dict[str, object]]) -> dict[str, object] | None:
    passing = [row for row in rows if bool(row["gate_pass"])]
    if not passing:
        return None
    return min(passing, key=lambda row: (
        -float(row["actionable_episode_recall"]),
        -float(row["minimum_speed_recall"]), -float(row["affected_foot_accuracy"]),
        -float(row["median_warning_margin_ms"]), int(row["macs_per_tick"]),
        FAMILIES.index(str(row["family"])), int(row["seed"]),
    ))


def diagnostic_fallback(rows: list[dict[str, object]]) -> dict[str, object]:
    return min(rows, key=lambda row: (
        int(row["normal_run_fp"]), int(row["normal_contact_episode_fp"]),
        int(row["too_early_activations"]), -float(row["actionable_episode_recall"]),
        -float(row["affected_foot_accuracy"]), FAMILIES.index(str(row["family"])), int(row["seed"]),
    ))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
