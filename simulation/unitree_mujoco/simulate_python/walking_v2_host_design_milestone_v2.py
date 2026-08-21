"""Frozen contracts for the Walking-v2 host design milestone v2.

This module contains only causal runtime helpers and preregistered development
conditions.  MuJoCo terrain identity, transition labels, Slip ground truth and
fall state are deliberately absent from every runtime function.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np


SAMPLE_RATE_HZ = 1000
PHYSICS_TIMESTEP_S = 0.0005
PHYSICS_STEPS_PER_SAMPLE = 2
TERRAIN_NAMES = ("concrete", "marble", "ice", "sand")
TERRAIN_CODE = {name: index for index, name in enumerate(TERRAIN_NAMES)}
HARD_TERRAINS = frozenset(("concrete", "marble"))
SPEEDS_MPS = (0.10, 0.15, 0.20)
SIDES = ("left", "right")


@dataclass(frozen=True)
class Variation:
    index: int
    phase_fraction: float
    command_delay_s: float
    lateral_offset_m: float
    friction_scale_before: float
    friction_scale_after: float
    solref_scale_before: float
    solref_scale_after: float


VARIATIONS = (
    Variation(0, 0.07, 0.000, -0.006, 1.00, 1.00, 1.00, 1.00),
    Variation(1, 0.23, 0.020, -0.003, 0.94, 1.06, 0.96, 1.04),
    Variation(2, 0.41, 0.040, 0.000, 1.06, 0.94, 1.04, 0.96),
    Variation(3, 0.67, 0.060, 0.003, 0.90, 1.02, 1.06, 0.94),
    Variation(4, 0.83, 0.080, 0.006, 1.02, 0.90, 0.94, 1.06),
)


# Case N is an explicit hard-to-hard transition negative.  Case S is a
# material-preserving seam crossing and provides precursor-like normal contact.
SCENARIOS = (
    ("A_marble_ice", "A", "marble", "ice"),
    ("A_concrete_ice", "A", "concrete", "ice"),
    ("N_marble_concrete", "N", "marble", "concrete"),
    ("N_concrete_marble", "N", "concrete", "marble"),
    ("B_marble_sand", "B", "marble", "sand"),
    ("B_concrete_sand", "B", "concrete", "sand"),
    ("C_ice_marble", "C", "ice", "marble"),
    ("C_ice_concrete", "C", "ice", "concrete"),
    ("D_sand_marble", "D", "sand", "marble"),
    ("D_sand_concrete", "D", "sand", "concrete"),
    ("S_concrete", "S", "concrete", "concrete"),
    ("S_marble", "S", "marble", "marble"),
    ("S_ice", "S", "ice", "ice"),
    ("S_sand", "S", "sand", "sand"),
)


@dataclass(frozen=True)
class RunCondition:
    run_id: str
    pair_id: str
    source_id: str
    scenario_id: str
    case_id: str
    terrain_before: str
    terrain_after: str
    speed_mps: float
    variation_index: int
    phase_fraction: float
    command_delay_s: float
    lateral_offset_m: float
    friction_scale_before: float
    friction_scale_after: float
    solref_scale_before: float
    solref_scale_after: float
    boundary_x_m: float
    replicate_index: int
    seed: int
    duration_s: float

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def acquisition_matrix(duration_s: float = 1.2) -> tuple[RunCondition, ...]:
    """Return the preregistered 420-run, five-fold development matrix."""
    if duration_s < 1.0:
        raise ValueError("duration must preserve pre/post-transition evidence")
    rows: list[RunCondition] = []
    index = 0
    for variation in VARIATIONS:
        for scenario_id, case_id, before, after in SCENARIOS:
            for speed in SPEEDS_MPS:
                for replicate in range(2):
                    tag = (
                        f"{scenario_id}_{speed:.2f}_v{variation.index}_r{replicate}"
                    ).replace(".", "p")
                    rows.append(RunCondition(
                        run_id=f"hostv2_{tag}",
                        pair_id=f"hostv2_pair_{index:03d}",
                        source_id=f"hostv2_source_{index:03d}",
                        scenario_id=scenario_id, case_id=case_id,
                        terrain_before=before, terrain_after=after,
                        speed_mps=speed, variation_index=variation.index,
                        phase_fraction=variation.phase_fraction,
                        command_delay_s=variation.command_delay_s,
                        lateral_offset_m=variation.lateral_offset_m,
                        friction_scale_before=variation.friction_scale_before,
                        friction_scale_after=variation.friction_scale_after,
                        solref_scale_before=variation.solref_scale_before,
                        solref_scale_after=variation.solref_scale_after,
                        boundary_x_m=0.10 if variation.index == 2 else 0.075,
                        replicate_index=replicate,
                        seed=2026082100 + index, duration_s=duration_s,
                    ))
                    index += 1
    validate_matrix(rows)
    return tuple(rows)


def validate_matrix(rows: list[RunCondition] | tuple[RunCondition, ...]) -> None:
    values = list(rows)
    if len(values) != 420 or len({row.run_id for row in values}) != 420:
        raise ValueError("matrix must contain 420 unique runs")
    for field in ("pair_id", "source_id", "seed"):
        if len({getattr(row, field) for row in values}) != len(values):
            raise ValueError(f"{field} must be isolated per run")
    cells = {
        (row.scenario_id, row.speed_mps, row.variation_index, row.replicate_index)
        for row in values
    }
    if len(cells) != len(SCENARIOS) * len(SPEEDS_MPS) * len(VARIATIONS) * 2:
        raise ValueError("scenario/speed/variation coverage is incomplete")
    if {row.case_id for row in values} != {"A", "B", "C", "D", "N", "S"}:
        raise ValueError("transition case coverage is incomplete")


def causal_contact_telemetry(
    loaded: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive causal touchdown, contact age, and runtime episode per foot."""
    values = np.asarray(loaded, bool)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("loaded must have shape [sample, 2]")
    touchdown = values & ~np.vstack((np.zeros((1, 2), bool), values[:-1]))
    age = np.zeros(values.shape, np.uint16)
    episode = np.full(values.shape, -1, np.int32)
    counters = [-1, -1]
    for sample in range(len(values)):
        for foot in (0, 1):
            if touchdown[sample, foot]:
                counters[foot] += 1
            if values[sample, foot]:
                age[sample, foot] = (
                    1 if sample == 0 or not values[sample - 1, foot]
                    else min(65535, int(age[sample - 1, foot]) + 1)
                )
                episode[sample, foot] = counters[foot]
    return touchdown, age, episode


def causal_g0_owner(
    loaded: np.ndarray, age: np.ndarray, finite: np.ndarray, minimum_age_ms: int = 5,
) -> np.ndarray:
    """Return the unique eligible contact owner, or -1 when ownership is ambiguous."""
    values = np.asarray(loaded, bool)
    ages = np.asarray(age, int)
    quality = np.asarray(finite, bool)
    eligible = values & (ages >= minimum_age_ms) & quality[:, None]
    owner = np.full(len(values), -1, np.int8)
    unique = eligible.sum(axis=1) == 1
    owner[unique] = np.argmax(eligible[unique], axis=1).astype(np.int8)
    return owner


@dataclass(frozen=True)
class TransitionPolicy:
    confidence_threshold: float
    dwell_endpoints: int
    minimum_contact_age_ms: int

    @property
    def config_id(self) -> str:
        return (
            f"p{self.confidence_threshold:.2f}_d{self.dwell_endpoints}"
            f"_a{self.minimum_contact_age_ms}"
        )


@dataclass(frozen=True)
class LinearTerrainModel:
    """Portable normalized multinomial-linear Terrain model artifact."""

    architecture: str
    history_ms: int
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    classes: np.ndarray

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        values = (np.asarray(features, np.float64) - self.mean) / self.scale
        logits = values @ self.coefficients.T + self.intercept
        logits -= logits.max(axis=1, keepdims=True)
        exponential = np.exp(np.clip(logits, -60.0, 60.0))
        return exponential / exponential.sum(axis=1, keepdims=True)

    @property
    def parameter_count(self) -> int:
        return int(self.mean.size + self.scale.size + self.coefficients.size + self.intercept.size)

    @property
    def macs_per_endpoint(self) -> int:
        return int(self.coefficients.size)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path, architecture=np.asarray(self.architecture),
            history_ms=np.asarray(self.history_ms, np.int32), mean=self.mean,
            scale=self.scale, coefficients=self.coefficients,
            intercept=self.intercept, classes=self.classes,
        )

    @classmethod
    def load(cls, path: str) -> "LinearTerrainModel":
        with np.load(path, allow_pickle=False) as archive:
            return cls(
                str(archive["architecture"]), int(archive["history_ms"]),
                archive["mean"].copy(), archive["scale"].copy(),
                archive["coefficients"].copy(), archive["intercept"].copy(),
                archive["classes"].copy(),
            )


def transition_policy_matrix() -> tuple[TransitionPolicy, ...]:
    """Frozen before evaluation; deliberately small, bounded policy search."""
    return tuple(
        TransitionPolicy(threshold, dwell, age)
        for threshold in (0.50, 0.65, 0.80, 0.90)
        for dwell in (1, 2, 3)
        for age in (5, 10)
    )


@dataclass
class TransitionTimeline:
    stable_state: np.ndarray
    case_a_advisory: np.ndarray
    direct_reflex: np.ndarray
    owner: np.ndarray
    emissions: list[dict[str, int | float | bool]]


def case_a_transition_timeline(
    probabilities: np.ndarray,
    endpoints: np.ndarray,
    loaded: np.ndarray,
    age: np.ndarray,
    runtime_episode: np.ndarray,
    finite: np.ndarray,
    policy: TransitionPolicy,
    *,
    direct_authorized: bool = False,
) -> TransitionTimeline:
    """Causally infer hard-to-Ice transitions with retained across-air terrain state."""
    probability = np.asarray(probabilities, float)
    endpoint_values = np.asarray(endpoints, int)
    if probability.shape != (len(endpoint_values), 2, 4):
        raise ValueError("probability shape must be [endpoint, foot, 4]")
    owner_samples = causal_g0_owner(loaded, age, finite, policy.minimum_contact_age_ms)
    owner = owner_samples[endpoint_values]
    stable = np.full((len(endpoint_values), 2), -1, np.int8)
    advisory = np.zeros((len(endpoint_values), 2), bool)
    candidate = [-1, -1]
    count = [0, 0]
    stable_value = [-1, -1]
    pending_episode = [-1, -1]
    emitted: set[tuple[int, int]] = set()
    emissions: list[dict[str, int | float | bool]] = []
    hard_codes = {TERRAIN_CODE["concrete"], TERRAIN_CODE["marble"]}
    ice = TERRAIN_CODE["ice"]
    for row, sample in enumerate(endpoint_values):
        for foot in (0, 1):
            if not loaded[sample, foot] or age[sample, foot] < policy.minimum_contact_age_ms:
                stable[row, foot] = stable_value[foot]
                continue
            predicted = int(np.argmax(probability[row, foot]))
            confident = float(probability[row, foot, predicted]) >= policy.confidence_threshold
            if not confident:
                candidate[foot] = -1; count[foot] = 0
                stable[row, foot] = stable_value[foot]
                continue
            if candidate[foot] == predicted:
                count[foot] += 1
            else:
                candidate[foot] = predicted; count[foot] = 1
            if count[foot] >= policy.dwell_endpoints and stable_value[foot] != predicted:
                previous = stable_value[foot]
                stable_value[foot] = predicted
                episode = int(runtime_episode[sample, foot])
                key = (foot, episode)
                if previous in hard_codes and predicted == ice and key not in emitted:
                    pending_episode[foot] = episode
            episode = int(runtime_episode[sample, foot])
            key = (foot, episode)
            if pending_episode[foot] == episode and owner[row] == foot and key not in emitted:
                emitted.add(key); pending_episode[foot] = -1
                emissions.append({
                    "endpoint_row": row, "sample": int(sample), "foot": foot,
                    "runtime_episode_id": episode,
                    "terrain_probability": float(probability[row, foot, ice]),
                    "unique_g0_owner": True,
                    "data_quality_valid": bool(finite[sample]),
                })
            stable[row, foot] = stable_value[foot]
        for output in emissions:
            if int(output["endpoint_row"]) == row:
                foot = int(output["foot"])
                advisory[row, foot] = True
    direct = advisory & (owner[:, None] == np.arange(2)[None, :]) if direct_authorized else np.zeros_like(advisory)
    return TransitionTimeline(stable, advisory, direct, owner, emissions)


def runtime_contract() -> dict[str, object]:
    return {
        "version": "walking_v2_host_design_runtime_contract_v2",
        "architecture_id": "CASE_A_TRANSITION_MONITOR_PLUS_SLIP_ADVISORY",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "inference_rate_hz": 100,
        "history_ms": 200,
        "runtime_inputs": [
            "bilateral canonical Fusion20", "causal force-loaded contact",
            "causal touchdown", "causal contact age", "current finite-data flag",
        ],
        "prohibited_runtime_inputs": [
            "terrain ground truth", "surface/profile/friction identity", "Case label",
            "transition timestamp", "physical Slip oracle", "first-fall/prefall oracle",
            "speed label", "foot target label", "variation/source/run/pair/seed identity",
            "future sample or future onset",
        ],
        "responsibilities": {
            "Terrain_transition_model": "four-class contact terrain score and Case-A advisory",
            "Slip_S4C": "unchanged broad risk advisory; diagnostic foot only",
            "G0": "causal loaded-contact state, age, runtime episode and unique-owner telemetry",
            "deterministic_state": "terrain dwell, across-air prior state, reset, one-shot dedup and authority firewall",
            "Sink": "deferred; Case B is evaluation context only",
        },
        "authority": {
            "case_a_transition": "monitoring/advisory only",
            "slip": "monitoring/advisory only",
            "direct_reflex": False,
            "recovery_actuation": False,
        },
    }
