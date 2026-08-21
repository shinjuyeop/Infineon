"""Causal runtime helpers for the Walking-v2 Fast Reflex host prototype.

The module intentionally separates signal production from authority.  Terrain
and Slip may emit development advisory telemetry, but ``direct_reflex`` is a
hard false until a direct-authority candidate passes its safety gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


TERRAIN_NAMES = ("concrete", "marble", "ice", "sand")
HARD_TERRAIN = frozenset((0, 1))
ENDPOINT_STRIDE_MS = 10


@dataclass(frozen=True)
class TerrainVerifierConfig:
    """Contact-local confidence and dwell policy for Case-A advisory context."""

    hard_dwell_endpoints: int
    ice_dwell_endpoints: int
    confidence_threshold: float
    pending_until_unique_owner: bool = True

    @property
    def config_id(self) -> str:
        return (
            f"contact_h{self.hard_dwell_endpoints}_i{self.ice_dwell_endpoints}"
            f"_p{self.confidence_threshold:.2f}_pending{int(self.pending_until_unique_owner)}"
        )


def terrain_candidate_matrix() -> tuple[TerrainVerifierConfig, ...]:
    """Return the bounded matrix frozen before the v1 host replay."""
    return tuple(
        TerrainVerifierConfig(hard, ice, confidence)
        for hard in (2, 3, 5)
        for ice in (3, 5, 8)
        for confidence in (0.70, 0.80, 0.90)
    )


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    """Return ``None`` for a zero denominator; never manufacture 0% evidence."""
    return None if denominator == 0 else float(numerator) / float(denominator)


def causal_data_quality_mask(canonical: np.ndarray) -> np.ndarray:
    """A current-sample-only data-quality signal; it is not a fall oracle."""
    values = np.asarray(canonical)
    if values.ndim != 2 or values.shape[1] != 20:
        raise ValueError("Fusion20 trace must have shape [sample, 20]")
    return np.all(np.isfinite(values), axis=1)


def unique_g0_owner(
    endpoints: np.ndarray,
    loaded: np.ndarray,
    contact_age: np.ndarray,
    touchdown: np.ndarray,
    data_quality_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return endpoint eligibility, unique-owner availability and owner index."""
    endpoint_values = np.asarray(endpoints, int)
    loaded_values = np.asarray(loaded, bool)
    age_values = np.asarray(contact_age, int)
    touchdown_values = np.asarray(touchdown, bool)
    quality = np.asarray(data_quality_valid, bool)
    if loaded_values.shape != age_values.shape or loaded_values.shape != touchdown_values.shape:
        raise ValueError("contact telemetry must align")
    if loaded_values.shape[0] != len(quality) or loaded_values.shape[1] != 2:
        raise ValueError("data-quality mask must align")
    eligible = (
        loaded_values[endpoint_values]
        & (age_values[endpoint_values] > 10)
        & ~touchdown_values[endpoint_values]
        & quality[endpoint_values, None]
    )
    unique = np.sum(eligible, axis=1) == 1
    owner = np.full(len(endpoint_values), -1, np.int8)
    owner[unique] = np.argmax(eligible[unique], axis=1).astype(np.int8)
    return eligible, unique, owner


@dataclass
class TerrainTimeline:
    case_a_active: np.ndarray
    stable_class: np.ndarray
    emissions: list[dict[str, Any]]
    eligibility: np.ndarray
    unique_owner: np.ndarray
    owner: np.ndarray


def terrain_case_a_timeline(
    probabilities: np.ndarray,
    endpoints: np.ndarray,
    loaded: np.ndarray,
    contact_age: np.ndarray,
    touchdown: np.ndarray,
    runtime_episode: np.ndarray,
    physical_episode: np.ndarray,
    data_quality_valid: np.ndarray,
    config: TerrainVerifierConfig,
) -> TerrainTimeline:
    """Build a contact-local, causal, one-shot predicted Case-A timeline.

    The physical episode is copied into emission records only for offline
    attribution.  It never changes the state machine or an output.
    """
    endpoint_values = np.asarray(endpoints, int)
    probability = np.asarray(probabilities, np.float64)
    if probability.shape != (len(endpoint_values), 2, 4):
        raise ValueError("Terrain probability shape must be [endpoint, foot, class]")
    eligible, unique, owner_now = unique_g0_owner(
        endpoint_values, loaded, contact_age, touchdown, data_quality_valid,
    )
    episodes = np.asarray(runtime_episode, int)
    physical = np.asarray(physical_episode, int)
    stable = np.full((len(endpoint_values), 2), -1, np.int8)
    active = np.zeros((len(endpoint_values), 2), bool)
    stable_value: list[int | None] = [None, None]
    candidate: list[int | None] = [None, None]
    count = [0, 0]
    owned_episode = [-1, -1]
    pending_episode: list[int | None] = [None, None]
    emitted: set[tuple[int, int]] = set()
    emissions: list[dict[str, Any]] = []

    for row, endpoint in enumerate(endpoint_values):
        unique_foot = int(owner_now[row])
        for foot in (0, 1):
            episode = int(episodes[endpoint, foot])
            if not eligible[row, foot]:
                candidate[foot] = None
                count[foot] = 0
                if not loaded[endpoint, foot] or episode != owned_episode[foot]:
                    stable_value[foot] = None
                    pending_episode[foot] = None
                owned_episode[foot] = episode if loaded[endpoint, foot] else -1
                continue
            if episode != owned_episode[foot]:
                stable_value[foot] = None
                candidate[foot] = None
                pending_episode[foot] = None
                count[foot] = 0
            owned_episode[foot] = episode
            predicted = int(np.argmax(probability[row, foot]))
            confident = float(probability[row, foot, predicted]) >= config.confidence_threshold
            if confident:
                if candidate[foot] == predicted:
                    count[foot] += 1
                else:
                    candidate[foot] = predicted
                    count[foot] = 1
                required = (
                    config.ice_dwell_endpoints if predicted == 2
                    else config.hard_dwell_endpoints
                )
                if count[foot] >= required and stable_value[foot] != predicted:
                    prior = stable_value[foot]
                    stable_value[foot] = predicted
                    if prior in HARD_TERRAIN and predicted == 2:
                        pending_episode[foot] = episode
            else:
                candidate[foot] = None
                count[foot] = 0

            pending = pending_episode[foot]
            can_emit = bool(
                pending == episode
                and (not config.pending_until_unique_owner or unique_foot == foot)
                and (foot, episode) not in emitted
            )
            if can_emit:
                emitted.add((foot, episode))
                pending_episode[foot] = None
                emissions.append({
                    "endpoint_row": row,
                    "sample": int(endpoint),
                    "foot": foot,
                    "runtime_episode_id": episode,
                    "physical_episode_id": int(physical[endpoint, foot]),
                    "transition_case": "A",
                    "terrain_probability": float(probability[row, foot, 2]),
                    "unique_g0_owner": unique_foot == foot,
                    "data_quality_valid": bool(data_quality_valid[endpoint]),
                })
            stable[row, foot] = -1 if stable_value[foot] is None else stable_value[foot]
            if (foot, episode) in emitted:
                active[row, foot] = True

    return TerrainTimeline(active, stable, emissions, eligible, unique, owner_now)


def direct_authority_gate(metrics: dict[str, Any]) -> bool:
    """Strict development gate for any output that could actuate."""
    zero_fields = (
        "normal_contact_fp", "too_early", "invalid_output", "post_fall_output",
        "wrong_case_output", "duplicate_output", "latch_carryover",
        "contact_owner_mismatch",
    )
    speed = metrics.get("speed_recall_within_20ms", {})
    return bool(
        all(int(metrics.get(name, 0)) == 0 for name in zero_fields)
        and float(metrics.get("recall_within_20ms", 0.0)) >= 0.80
        and speed
        and min(float(value) for value in speed.values()) >= 0.70
        and int(metrics.get("output_count", 0)) > 0
    )


def select_terrain_advisory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select a non-actuating context signal by alert F1, then precision/recall."""
    usable = [
        row for row in rows
        if int(row["invalid_output"]) == 0
        and int(row["wrong_case_output"]) == 0
        and int(row["duplicate_output"]) == 0
        and int(row["normal_run_fp"]) == 0
        and int(row["output_count"]) > 0
    ]
    if not usable:
        raise RuntimeError("no usable Terrain advisory candidate")
    return min(usable, key=lambda row: (
        -float(row["alert_f1"]), -float(row["precision"]),
        -float(row["event_recall"]),
        float(row["p95_late_latency_ms"] if row["p95_late_latency_ms"] is not None else 1e9),
        row["config_id"],
    ))


def causal_runtime_contract() -> dict[str, Any]:
    return {
        "version": "walking_v2_fast_reflex_host_contract_v1",
        "architecture_id": "MONITORING_ONLY_TERRAIN_AND_SLIP_ADVISORY",
        "sample_rate_hz": 1000,
        "inference_endpoint_stride_ms": ENDPOINT_STRIDE_MS,
        "history_ms": 200,
        "runtime_inputs": [
            "bilateral Fusion20", "causal G0 force-loaded contact",
            "causal contact age", "causal gait phase", "current finite-data-quality",
        ],
        "prohibited_runtime_inputs": [
            "terrain_name/id", "M1 profile or friction", "physical Slip oracle",
            "future onset", "pre_fall_valid or fall oracle", "run/source/seed identity",
        ],
        "authority": {
            "Terrain_T2": "Case-A context advisory only; no actuation",
            "Slip_S4C": "risk advisory only; learned foot is diagnostic only",
            "G0": "causal unique-owner tag; sole permitted future foot owner",
            "M1": "development simulation provenance only; no runtime input",
            "deterministic_state": (
                "owns current-data mask, contact reset, persistence, one-shot dedup, "
                "and the hard direct-actuation firewall"
            ),
            "direct_reflex": "constant false",
            "Sink": "SINK_RUNTIME_DETECTION_DEFERRED",
        },
        "outputs": {
            "terrain_case_a_advisory": "telemetry only",
            "slip_risk_advisory": "telemetry/operator-study only",
            "direct_reflex": False,
            "recovery_actuation": False,
        },
    }


def config_payload(config: TerrainVerifierConfig) -> dict[str, Any]:
    return {**asdict(config), "config_id": config.config_id}
