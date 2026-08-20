"""Pure contracts, causal features, and selection helpers for walking-v2.

The runtime path accepts only canonical bilateral Fusion10 and force-derived
contact state.  Physical Slip arrays enter only the explicitly offline episode
and evaluation helpers in this module.  Sink has no runtime output or head.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


SAMPLE_RATE_HZ = 1000
TERRAIN_WINDOW_MS = 50
SLIP_WINDOW_MS = 100
ENDPOINT_STRIDE_MS = 10
TERRAIN_ARCHITECTURES = ("T1", "T2")
SLIP_ARCHITECTURES = ("S1", "S2")
TRAINING_SEEDS = (202608211, 202608212, 202608213)
TERRAIN_NAMES = ("concrete", "marble", "ice", "sand")
TERRAIN_LABELS = {name: index for index, name in enumerate(TERRAIN_NAMES)}
PHASE_NAMES = {0: "AIR", 1: "TOUCHDOWN", 2: "LOADING", 3: "MID_STANCE", 4: "PUSH_OFF"}
SLIP_THRESHOLD_GRID = (0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999)
SLIP_PERSISTENCE_GRID = (1, 2, 3)
SLIP_HYSTERESIS_GRID = (0.05, 0.10)
PHYSICAL_EPISODE_MERGE_GAP_MS = 50
ACTIONABLE_EVENT_COOLDOWN_MS = 100

RUNTIME_INPUT_FIELDS = frozenset({
    "bilateral_canonical", "force_loaded", "contact_age", "gait_phase_code",
})
FORBIDDEN_RUNTIME_FIELDS = frozenset({
    "sink_risk", "sink_detected", "sink_physical_active", "sink_risk_target",
    "contact_penetration", "penetration_change_label_only", "terrain_name",
    "profile_name", "speed_mps", "run_id", "foot_world_xyz", "fall_flag",
    "slip_physical_active", "slip_risk_target",
})


def runtime_scope_contract() -> dict[str, object]:
    return {
        "version": "walking_v2_bilateral_runtime_scope_v1",
        "walking_specific": True,
        "outputs": [
            "terrain_class", "sand_terrain_caution", "left_slip_risk",
            "right_slip_risk", "affected_foot", "slip_evidence_persistent",
            "contact_state", "reset_reason",
        ],
        "terrain_classes": list(TERRAIN_NAMES),
        "inputs": sorted(RUNTIME_INPUT_FIELDS),
        "excluded_inputs": [
            "pelvis_imu", "joint_kinematics", "actuator_effort",
            "tracking_residual", "terrain_id", "speed_profile_run_identity",
            "MuJoCo_privileged_ground_truth", "physical_Sink_input",
        ],
        "sink_runtime_outputs": [],
        "sand_semantics": "SAND_TERRAIN_CAUTION; never Sink detection",
        "controlled_detector_replacement": False,
        "production_change": False,
    }


def input_contract() -> dict[str, object]:
    return {
        "version": "walking_v2_bilateral_input_contract_v1",
        "native_rate_hz": SAMPLE_RATE_HZ,
        "canonical_input_order": ["left Fusion10", "right Fusion10"],
        "per_foot_channels": [
            "fsr_rear_lateral", "fsr_rear_medial", "fsr_front_lateral",
            "fsr_front_medial", "accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z",
        ],
        "runtime_contact_features": [
            "per-foot contact state", "time since touchdown",
            "bilateral load ratio", "single/double support",
        ],
        "terrain_history_ms": TERRAIN_WINDOW_MS,
        "slip_history_ms": SLIP_WINDOW_MS,
        "future_samples": 0,
        "shared_encoder": "one fixed weight object applied to both feet",
        "raw_frame_allowed": False,
        "hardware_reproducible_virtual_signals_only": True,
    }


def sink_deferral_contract() -> dict[str, object]:
    return {
        "status": "SINK_RUNTIME_DETECTION_DEFERRED",
        "runtime_head_created": False,
        "runtime_outputs_forbidden": ["sink_risk", "sink_detected"],
        "physical_oracle_retained_offline": True,
        "dataset_provenance_retained": True,
        "sand_prediction_is_sink": False,
        "sand_runtime_meaning": "SAND_TERRAIN_CAUTION",
    }


def episode_semantics_contract() -> dict[str, object]:
    return {
        "raw_oracle_crossing": "false-to-true transition of offline physical Slip active",
        "physical_episode": (
            "merge active segments in one contact episode when the inactive gap is "
            f"<= {PHYSICAL_EPISODE_MERGE_GAP_MS} ms"
        ),
        "new_episode": "contact loss/new touchdown, or re-entry beyond the merge gap",
        "first_actionable_event": (
            "first physical episode in a contact episode; a later episode is independently "
            f"actionable only after {ACTIONABLE_EVENT_COOLDOWN_MS} ms cooldown"
        ),
        "risk_window": "0..100 ms before physical onset or physical active",
        "too_early": "firing more than 100 ms before the next onset in the same contact episode",
        "post_reflex_counterfactual": "reported separately; no recovery intervention was simulated",
        "threshold_chatter_double_counted": False,
    }


@dataclass(frozen=True)
class SharedCausalFootEncoder:
    """Fixed compact depthwise causal Conv1D/GAP features shared by identity."""

    window_ms: int

    def __post_init__(self) -> None:
        if self.window_ms not in (TERRAIN_WINDOW_MS, SLIP_WINDOW_MS):
            raise ValueError("walking-v2 permits only the frozen 50/100 ms windows")

    @property
    def output_size(self) -> int:
        return 40

    @property
    def parameter_count(self) -> int:
        # 10 depthwise channels x (mean5/mean20/full/difference kernel taps).
        return 10 * (5 + 20 + self.window_ms + 2)

    @property
    def macs(self) -> int:
        return self.parameter_count

    @property
    def fingerprint(self) -> str:
        text = f"walking-v2-shared-depthwise-causal-conv-gap:{self.window_ms}:mean5,mean20,full,diff1"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def encode(self, window: np.ndarray) -> np.ndarray:
        values = np.asarray(window, dtype=np.float64)
        if values.shape != (self.window_ms, 10):
            raise ValueError(f"expected ({self.window_ms},10), got {values.shape}")
        # Each statistic is the current causal output/GAP of a fixed depthwise
        # temporal filter. No sample after the endpoint is visible.
        encoded = np.stack((
            values[-5:].mean(axis=0),
            values[-20:].mean(axis=0),
            values.mean(axis=0),
            values[-1] - values[-2],
        ), axis=1).reshape(-1).astype(np.float32)
        if encoded.shape != (40,) or not np.all(np.isfinite(encoded)):
            raise ValueError("invalid shared foot encoding")
        return encoded


def causal_endpoints(sample_count: int, window_ms: int) -> np.ndarray:
    if window_ms <= 0 or sample_count < window_ms:
        return np.empty(0, dtype=np.int64)
    return np.arange(window_ms - 1, sample_count, ENDPOINT_STRIDE_MS, dtype=np.int64)


def runtime_feature(
    architecture: str,
    side: int,
    endpoint: int,
    bilateral_canonical: np.ndarray,
    force_loaded: np.ndarray,
    contact_age: np.ndarray,
    gait_phase_code: np.ndarray,
    encoder: SharedCausalFootEncoder,
) -> np.ndarray:
    """Build one T1/T2/S1/S2 runtime vector from current/past observables."""
    if architecture not in (*TERRAIN_ARCHITECTURES, *SLIP_ARCHITECTURES):
        raise ValueError(architecture)
    if side not in (0, 1):
        raise ValueError(side)
    values = np.asarray(bilateral_canonical, dtype=np.float64)
    loaded = np.asarray(force_loaded, dtype=bool)
    age = np.asarray(contact_age, dtype=np.int32)
    phase = np.asarray(gait_phase_code, dtype=np.int8)
    if values.ndim != 2 or values.shape[1] != 20:
        raise ValueError("bilateral canonical trace must be (samples,20)")
    if loaded.shape != (len(values), 2) or age.shape != loaded.shape or phase.shape != loaded.shape:
        raise ValueError("contact arrays must align with bilateral trace")
    start = endpoint - encoder.window_ms + 1
    if start < 0:
        raise ValueError("endpoint has incomplete causal history")
    left = encoder.encode(values[start:endpoint + 1, :10])
    right = encoder.encode(values[start:endpoint + 1, 10:])
    own, other = (left, right) if side == 0 else (right, left)
    base = np.concatenate((own, other))
    if architecture == "T1":
        return base
    forces = values[:, (0, 1, 2, 3, 10, 11, 12, 13)].reshape(len(values), 2, 4).sum(axis=2)
    total = forces.sum(axis=1)
    ratio = np.divide(
        forces[:, side], total, out=np.full(len(values), 0.5), where=total > 1e-9
    )
    support = np.asarray((
        loaded[endpoint, side] and not loaded[endpoint, 1 - side],
        loaded[endpoint, 1 - side] and not loaded[endpoint, side],
        loaded[endpoint, side] and loaded[endpoint, 1 - side],
    ), np.float32)
    common = np.concatenate((base, np.asarray((
        ratio[endpoint], 1.0 - ratio[endpoint],
        min(age[endpoint, side], 1000) / 1000.0,
        min(age[endpoint, 1 - side], 1000) / 1000.0,
        *support,
    ), np.float32)))
    if architecture == "T2":
        own_phase = np.eye(5, dtype=np.float32)[int(phase[endpoint, side])]
        other_phase = np.eye(5, dtype=np.float32)[int(phase[endpoint, 1 - side])]
        return np.concatenate((common, own_phase, other_phase))
    if architecture == "S1":
        return common
    touchdown = max(0, endpoint - max(0, int(age[endpoint, side]) - 1))
    own_slice = slice(side * 10, (side + 1) * 10)
    touchdown_delta = values[endpoint, own_slice] - values[touchdown, own_slice]
    ratio_window = ratio[start:endpoint + 1]
    transfer = np.asarray((
        ratio_window.mean(), ratio_window.std(), ratio_window[-1] - ratio_window[0],
        forces[endpoint, side] - forces[touchdown, side],
    ), np.float32)
    return np.concatenate((common, touchdown_delta.astype(np.float32), transfer))


def balanced_indices(groups: np.ndarray, seed: int) -> np.ndarray:
    """Select the same count from every declared group, deterministically."""
    values = np.asarray(groups)
    if values.ndim != 2 or not len(values):
        raise ValueError("groups must be a non-empty 2D array")
    unique, inverse, counts = np.unique(values, axis=0, return_inverse=True, return_counts=True)
    if not len(unique) or np.min(counts) <= 0:
        raise ValueError("empty balance group")
    count = int(np.min(counts))
    rng = np.random.default_rng(seed)
    selected = []
    for group_index in range(len(unique)):
        candidates = np.flatnonzero(inverse == group_index)
        selected.extend(rng.choice(candidates, count, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


@dataclass(frozen=True)
class LinearFloatModel:
    architecture: str
    seed: int
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    classes: np.ndarray
    encoder_fingerprint: str

    @property
    def parameter_count(self) -> int:
        return int(self.coefficients.size + self.intercept.size)

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        values = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        logits = values @ self.coefficients.T + self.intercept
        if len(self.classes) == 2 and logits.shape[1] == 1:
            positive = 1.0 / (1.0 + np.exp(-np.clip(logits[:, 0], -60, 60)))
            return np.column_stack((1.0 - positive, positive))
        logits = logits - logits.max(axis=1, keepdims=True)
        exponential = np.exp(logits)
        return exponential / exponential.sum(axis=1, keepdims=True)

    def predictions(self, features: np.ndarray) -> np.ndarray:
        return self.classes[np.argmax(self.probabilities(features), axis=1)]

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            architecture=np.asarray(self.architecture), seed=np.asarray(self.seed),
            mean=self.mean, scale=self.scale, coefficients=self.coefficients,
            intercept=self.intercept, classes=self.classes,
            encoder_fingerprint=np.asarray(self.encoder_fingerprint),
        )

    @classmethod
    def load(cls, path: Path) -> "LinearFloatModel":
        with np.load(path, allow_pickle=False) as values:
            return cls(
                str(values["architecture"]), int(values["seed"]),
                values["mean"], values["scale"], values["coefficients"],
                values["intercept"], values["classes"],
                str(values["encoder_fingerprint"]),
            )


def fit_linear_float(
    architecture: str,
    seed: int,
    features: np.ndarray,
    target: np.ndarray,
    encoder_fingerprint: str,
) -> LinearFloatModel:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target)
    scaler = StandardScaler().fit(x)
    normalized = scaler.transform(x)
    estimator = LogisticRegression(
        C=1.0, max_iter=500, random_state=seed,
        solver="liblinear" if len(np.unique(y)) == 2 else "lbfgs",
        multi_class="auto",
    ).fit(normalized, y)
    return LinearFloatModel(
        architecture, seed, scaler.mean_.copy(), scaler.scale_.copy(),
        estimator.coef_.copy(), estimator.intercept_.copy(), estimator.classes_.copy(),
        encoder_fingerprint,
    )


@dataclass(frozen=True)
class PhysicalSlipEpisode:
    contact_episode_id: int
    start: int
    end_exclusive: int
    raw_crossings: int


def physical_slip_episodes(
    active: np.ndarray,
    contact_episode_id: np.ndarray,
    valid: np.ndarray,
    merge_gap_ms: int = PHYSICAL_EPISODE_MERGE_GAP_MS,
) -> list[PhysicalSlipEpisode]:
    """Merge threshold chatter without crossing contact boundaries."""
    values = np.asarray(active, bool) & np.asarray(valid, bool)
    contact = np.asarray(contact_episode_id, int)
    if values.shape != contact.shape or values.ndim != 1 or merge_gap_ms < 0:
        raise ValueError("episode arrays must align")
    segments: list[PhysicalSlipEpisode] = []
    for contact_id in np.unique(contact[values]):
        if contact_id < 0:
            continue
        in_contact = values & (contact == contact_id)
        edges = np.diff(np.r_[False, in_contact, False].astype(np.int8))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            segments.append(PhysicalSlipEpisode(int(contact_id), int(start), int(end), 1))
    segments.sort(key=lambda value: (value.start, value.end_exclusive, value.contact_episode_id))
    output: list[PhysicalSlipEpisode] = []
    for segment in segments:
        if (
            output
            and output[-1].contact_episode_id == segment.contact_episode_id
            and segment.start - output[-1].end_exclusive <= merge_gap_ms
        ):
            previous = output.pop()
            output.append(PhysicalSlipEpisode(
                segment.contact_episode_id, previous.start, segment.end_exclusive,
                previous.raw_crossings + 1,
            ))
        else:
            output.append(segment)
    return output


def raw_slip_crossings(
    active: np.ndarray,
    contact_episode_id: np.ndarray,
    valid: np.ndarray,
) -> list[PhysicalSlipEpisode]:
    """Return each unmerged physical-oracle false-to-true crossing."""
    return physical_slip_episodes(active, contact_episode_id, valid, merge_gap_ms=0)


def risk_firing_is_too_early(
    sample: int,
    contact_episode_id: int,
    episodes: list[PhysicalSlipEpisode],
    horizon_ms: int = SLIP_WINDOW_MS,
) -> bool:
    """Whether a firing precedes the next same-contact onset by over the horizon."""
    future_starts = [
        value.start for value in episodes
        if value.contact_episode_id == contact_episode_id and value.start > sample
    ]
    return bool(future_starts and sample < min(future_starts) - horizon_ms)


def first_actionable_events(
    episodes: list[PhysicalSlipEpisode],
    cooldown_ms: int = ACTIONABLE_EVENT_COOLDOWN_MS,
) -> list[PhysicalSlipEpisode]:
    if cooldown_ms < 0:
        raise ValueError("cooldown must be nonnegative")
    output: list[PhysicalSlipEpisode] = []
    previous_by_contact: dict[int, PhysicalSlipEpisode] = {}
    for episode in episodes:
        previous = previous_by_contact.get(episode.contact_episode_id)
        if previous is None or episode.start - previous.end_exclusive > cooldown_ms:
            output.append(episode)
        previous_by_contact[episode.contact_episode_id] = episode
    return output


@dataclass(frozen=True)
class SlipStateConfig:
    threshold: float
    persistence_endpoints: int
    hysteresis: float

    @property
    def exit_threshold(self) -> float:
        return max(0.0, self.threshold - self.hysteresis)


def stateful_slip_firing(
    scores: np.ndarray,
    endpoints: np.ndarray,
    force_loaded: np.ndarray,
    contact_age: np.ndarray,
    config: SlipStateConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal per-foot state with contact-loss/new-touchdown reset."""
    probability = np.asarray(scores, float)
    endpoint_values = np.asarray(endpoints, int)
    loaded = np.asarray(force_loaded, bool)
    age = np.asarray(contact_age, int)
    if probability.shape != endpoint_values.shape or loaded.shape != age.shape:
        raise ValueError("state arrays must align")
    firing = np.zeros(len(probability), bool)
    reset = np.full(len(probability), "none", dtype="<U16")
    active = False
    count = 0
    previous_age = 0
    for index, endpoint in enumerate(endpoint_values):
        eligible = bool(loaded[endpoint] and age[endpoint] > 10)
        new_touchdown = bool(age[endpoint] <= previous_age and age[endpoint] > 0)
        if not eligible or new_touchdown:
            active = False
            count = 0
            reset[index] = "contact_loss" if not loaded[endpoint] else "new_touchdown"
        elif active:
            active = bool(probability[index] >= config.exit_threshold)
            if not active:
                count = 0
                reset[index] = "score_recovery"
        else:
            count = count + 1 if probability[index] >= config.threshold else 0
            active = count >= config.persistence_endpoints
        firing[index] = active and eligible
        previous_age = int(age[endpoint])
    return firing, reset


def affected_foot_correct(
    actual_foot: int,
    detection_endpoint_index: int,
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    left_firing: np.ndarray,
    right_firing: np.ndarray,
) -> bool:
    if actual_foot not in (0, 1):
        raise ValueError(actual_foot)
    scores = (left_scores, right_scores)
    firing = (left_firing, right_firing)
    return bool(
        firing[actual_foot][detection_endpoint_index]
        and scores[actual_foot][detection_endpoint_index]
        >= scores[1 - actual_foot][detection_endpoint_index]
    )


def originating_risk_window_detections(
    firing: np.ndarray,
    endpoints: np.ndarray,
    detected_indices: np.ndarray,
    risk_start_sample: int,
) -> np.ndarray:
    """Reject a latched detection whose activation began before this risk window.

    This is offline episode attribution only. It does not alter detector state or
    use an oracle as a runtime input.
    """
    state = np.asarray(firing, bool)
    endpoint_values = np.asarray(endpoints, int)
    selected = np.asarray(detected_indices, int)
    if state.shape != endpoint_values.shape or selected.ndim != 1:
        raise ValueError("detection attribution arrays must align")
    rising = state & ~np.r_[False, state[:-1]]
    rising_indices = np.flatnonzero(rising)
    owned = []
    for index in selected:
        previous = rising_indices[rising_indices <= index]
        if len(previous) and endpoint_values[previous[-1]] >= risk_start_sample:
            owned.append(int(index))
    return np.asarray(owned, dtype=np.int64)


def terrain_gate(metrics: dict[str, object]) -> bool:
    return bool(
        float(metrics["overall_accuracy"]) >= 0.85
        and float(metrics["macro_accuracy"]) >= 0.85
        and float(metrics["worst_class_recall"]) >= 0.70
        and float(metrics["majority_class_prediction_rate"]) < 0.60
        and float(metrics["minimum_speed_accuracy"]) >= 0.80
        and float(metrics["left_right_accuracy_difference_pp"]) <= 10.0
        and not bool(metrics["class_collapse"])
        and int(metrics["air_terrain_transitions"]) == 0
        and int(metrics["invalid_firings"]) == 0
    )


def slip_gate(metrics: dict[str, object]) -> bool:
    return bool(
        float(metrics["valid_ice_run_coverage"]) == 1.0
        and float(metrics["first_actionable_event_recall"]) == 1.0
        and float(metrics["physical_episode_recall"]) >= 0.80
        and float(metrics["affected_foot_accuracy"]) >= 0.90
        and int(metrics["normal_risk_run_fp"]) == 0
        and int(metrics["normal_physical_episode_fp"]) == 0
        and int(metrics["too_early_firings"]) == 0
        and int(metrics["invalid_firings"]) == 0
        and bool(metrics["all_speed_coverage"])
        and bool(metrics["both_affected_feet_coverage"])
        and float(metrics["median_warning_margin_ms"]) >= 20.0
        and float(metrics["pre_onset_detection_fraction"]) >= 0.80
        and bool(metrics["reset_invariant_pass"])
    )


def evaluation_invalid_firing_count(
    air_firings: int,
    touchdown_firings: int,
    post_fall_state_outputs: int,
    *,
    strict_first_fall_censor: bool,
) -> int:
    """Count evaluable invalid firings without erasing post-fall diagnostics.

    A first-fall-censored sample is outside the evaluation population.  The
    detector may still produce a counterfactual state there because fall state
    is not a runtime input, but that output must be reported separately instead
    of being added to the invalid-evaluation numerator.
    """
    if min(air_firings, touchdown_firings, post_fall_state_outputs) < 0:
        raise ValueError("firing counts must be nonnegative")
    return int(
        air_firings
        + touchdown_firings
        + (0 if strict_first_fall_censor else post_fall_state_outputs)
    )


def deterministic_terrain_selection(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("no Terrain candidates")
    return min(rows, key=lambda row: (
        int(row["invalid_firings"]), int(row["class_collapse"]),
        -int(bool(row["gate_pass"])), -float(row["macro_accuracy"]),
        -float(row["worst_class_recall"]),
        float(row["left_right_accuracy_difference_pp"]),
        int(row["parameter_count"]), int(row["macs"]),
        TERRAIN_ARCHITECTURES.index(str(row["architecture"])), int(row["seed"]),
    ))


def deterministic_slip_selection(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("no Slip candidates")
    return min(rows, key=lambda row: (
        int(row["invalid_firings"]), int(row["normal_risk_run_fp"]),
        int(row["too_early_firings"]), -int(bool(row["gate_pass"])),
        -float(row["first_actionable_event_recall"]),
        -float(row["physical_episode_recall"]),
        -float(row["median_warning_margin_ms"]),
        int(row["parameter_count"]), int(row["macs"]),
        SLIP_ARCHITECTURES.index(str(row["architecture"])), int(row["seed"]),
        float(row["threshold"]), int(row["persistence_endpoints"]),
        float(row["hysteresis"]),
    ))


def holdout_authorized(
    terrain_ready: bool,
    slip_ready: bool,
    selection_lock_exists: bool,
) -> bool:
    return bool(terrain_ready and slip_ready and selection_lock_exists)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
