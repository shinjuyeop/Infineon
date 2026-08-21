"""Pure contracts for Walking-v2 Slip risk-scope reduction v5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from walking_v2_slip_corrected_targeted_retraining_v4 import (
    ENDPOINTS, HEAD_NAMES, OperatingConfig, corrected_v4_state,
)


CONTRACTS = (
    "C0_BROAD_PREDICTIVE_BASELINE",
    "C1_CASE_A_UNIQUE_OWNER_PREDICTIVE",
    "C2_CASE_A_UNIQUE_OWNER_REACTIVE_CONFIRMATION",
    "C3_TERRAIN_CASE_A_REFLEX_ONLY",
    "C4_STOP_WALKING_V2_SLIP_RUNTIME",
)
DEADLINES_MS = (10, 20, 30, 50)
ORIGINAL_EPISODE_COUNT = 471


def causal_signal_inventory() -> list[dict[str, Any]]:
    """Authoritative availability claims; unknown evidence is never promoted."""
    rows = [
        (
            "predicted_terrain_state", "locked Walking-v2 T2 Terrain model",
            "current causal 200 ms bilateral history endpoint", "10 ms endpoint cadence; compute latency unbenchmarked",
            True, False, True, "conditional on a completed 200 ms history", "conditional",
            "Model prediction; never terrain_id/name/oracle.",
        ),
        (
            "predicted_transition_case", "portable transition case mapper over two stable predicted Terrain states",
            "endpoint at which a different stable predicted Terrain state becomes available",
            "unbounded in this corpus: transition persistence/timeline not stored", True, False, True,
            "not evidenced for any of the 471 episodes", "not evidenced",
            "Allowed in principle, but unavailable for C1/C2 accounting without a prior predicted state.",
        ),
        (
            "transition_state_availability_timestamp", "Terrain transition state machine",
            "same causal endpoint as predicted transition recognition", "not recorded", False, False, False,
            "no", "no", "Missing authoritative per-run transition timeline.",
        ),
        (
            "touchdown_timestamp", "causal force-loaded rising edge", "current 1 kHz sample", "<=1 ms",
            False, False, True, "yes", "yes", "Runtime-observable virtual FSR contact edge.",
        ),
        (
            "causal_touchdown_age", "counter since causal force-loaded rising edge", "current 1 kHz sample",
            "<=1 ms", False, False, True, "yes", "yes", "Reconstructed exactly against legacy telemetry.",
        ),
        (
            "current_G0_contact_state", "bilateral virtual FSR/contact telemetry", "current 1 kHz sample",
            "<=1 ms", False, False, True, "yes", "yes", "G0 geometry is fixed; contact state is causal telemetry.",
        ),
        (
            "current_unique_contact_owner", "exactly-one-foot predicate over valid G0 contact state",
            "current 1 kHz sample", "<=1 ms", False, False, True, "yes", "yes",
            "Permitted actuation owner; no learned-foot substitution.",
        ),
        (
            "single_support_dual_support_state", "bilateral G0 contact-state reducer", "current 1 kHz sample",
            "<=1 ms", False, False, True, "yes", "yes", "Derived without future contact.",
        ),
        (
            "foot_identity", "fixed left/right sensor wiring and G0 owner", "current sample", "0 ms",
            False, False, True, "yes", "yes", "Both feet remain representable.",
        ),
        (
            "current_speed_command", "locomotion command source", "current controller tick", "0 ms",
            False, False, True, "yes", "yes", "Inventory only; C1/C2 do not select one speed.",
        ),
        (
            "measured_causal_speed", "encoder/IMU speed estimator", "current estimator tick", "not frozen",
            False, False, False, "unknown", "unknown", "No frozen estimator contract; not used as a gate.",
        ),
        (
            "M1_state", "locked simulation friction profile", "environment configuration", "not a runtime measurement",
            False, True, False, "no", "no", "Configuration/oracle context; prohibited as a runtime scope gate.",
        ),
        (
            "raw_state_head_outputs", "stored S4-C OOF state head", "10 ms endpoint after 200 ms history",
            "10 ms cadence; compute latency unbenchmarked", True, False, False, "yes", "yes",
            "Detector evidence, explicitly prohibited from selecting the operational scope.",
        ),
        (
            "raw_proposal_head_output", "stored S4-C OOF proposal head", "10 ms endpoint after 200 ms history",
            "10 ms cadence; compute latency unbenchmarked", True, False, False, "yes", "yes",
            "Detector evidence, not a scope selector.",
        ),
        (
            "raw_foot_head_output", "stored S4-C OOF foot head", "10 ms endpoint after 200 ms history",
            "10 ms cadence; compute latency unbenchmarked", True, False, False, "yes", "yes",
            "Diagnostic in C2; G0 unique owner controls actuation ownership.",
        ),
        (
            "causal_physical_active_evidence", "S4-C physical-active state probability/argmax",
            "10 ms endpoint after 200 ms history", "10 ms cadence; compute latency unbenchmarked",
            True, False, True, "may be predicted early", "yes",
            "Model-predicted evidence, not the physical oracle; pre-onset firing remains a violation.",
        ),
        (
            "first_fall_boundary", "offline first-fall oracle in frozen traces", "first invalid physical sample",
            "offline only", False, True, False, "no", "no",
            "Evaluation/reset audit only; no locked causal fall estimator exists.",
        ),
        (
            "invalid_mask", "runtime data-quality monitor plus offline fall censor", "current sample",
            "<=1 ms for data quality; fall part offline", False, False, True, "partial", "partial",
            "Causal data-quality component allowed; oracle fall component evaluation-only.",
        ),
        (
            "AIR_mask", "causal bilateral force-loaded state", "current 1 kHz sample", "<=1 ms",
            False, False, True, "yes", "yes", "Runtime allowed.",
        ),
        (
            "touchdown_transient_mask", "causal touchdown age <=10 ms", "current 1 kHz sample", "<=1 ms",
            False, False, True, "yes", "yes", "Runtime allowed.",
        ),
    ]
    fields = (
        "signal", "producer", "timestamp_reference", "latency", "model_predicted",
        "oracle_derived", "allowed_runtime_gate", "available_before_physical_onset",
        "available_within_20ms_after_onset", "qualification",
    )
    return [dict(zip(fields, row)) for row in rows]


def scope_matrix_payload() -> dict[str, Any]:
    return {
        "version": "walking_v2_slip_risk_scope_matrix_v5",
        "frozen_before_scope_metrics": True,
        "contracts": [
            {
                "contract_id": CONTRACTS[0], "selectable": False,
                "role": "diagnostic baseline", "horizon_ms": 100,
                "scope_selectors": [], "denominator": "original frozen 471 episodes",
                "reason": "already failed the corrected-v4 mandatory gates",
            },
            {
                "contract_id": CONTRACTS[1], "selectable": True,
                "role": "foot-specific predictive warning", "horizon_ms": 100,
                "scope_selectors": [
                    "locked predicted Case A", "exactly one current G0 owner",
                    "valid loaded contact", "touchdown transient passed", "before first fall",
                ],
                "forbidden_selectors": [
                    "run/source/variation/seed", "observed error", "model score",
                    "speed-only", "foot-only", "future severity/duration/peak", "oracle Terrain",
                ],
                "physical_active_evidence_required": False,
            },
            {
                "contract_id": CONTRACTS[2], "selectable": True,
                "role": "reactive confirmation/recovery escalation only", "selectable_deadline_ms": 20,
                "diagnostic_deadlines_ms": [10, 30, 50],
                "scope_selectors": [
                    "locked predicted Case A", "exactly one current G0 owner",
                    "valid loaded contact", "touchdown transient passed", "before first fall",
                ],
                "causal_confirmation_gate": "physical-active state is argmax; no new threshold",
                "actuation_owner": "current unique G0 owner", "learned_foot_head": "diagnostic only",
                "initial_reflex_authority": False,
            },
            {
                "contract_id": CONTRACTS[3], "selectable": True,
                "role": "Terrain predicted Case-A transition reflex only",
                "learned_slip_inference_authority": "offline diagnostic only",
                "learned_slip_actuation_authority": False,
                "requires_later_validation": "Terrain-only Case-A System validation",
            },
            {
                "contract_id": CONTRACTS[4], "selectable": True,
                "role": "no learned Slip inference or actuation", "terrain_preserved": True,
                "sink": "SINK_RUNTIME_DETECTION_DEFERRED",
            },
        ],
        "decision_order": list(CONTRACTS[1:]),
        "gates_lowered": False, "new_contract_after_results": False,
    }


def contact_scope_signals(
    loaded: np.ndarray, valid: np.ndarray, touchdown: np.ndarray, prefall: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return causal unique-owner availability and owner identity."""
    contact = (
        np.asarray(loaded, bool) & np.asarray(valid, bool)
        & ~np.asarray(touchdown, bool) & np.asarray(prefall, bool)[:, None]
    )
    unique = np.sum(contact, axis=1) == 1
    owner = np.full(len(contact), -1, np.int8)
    owner[unique] = np.argmax(contact[unique], axis=1).astype(np.int8)
    return unique, owner


def event_scope_eligibility(
    endpoint_samples: np.ndarray, onset: int, foot: int, unique_owner: np.ndarray,
    owner: np.ndarray, predicted_case_a: np.ndarray, mode: str,
) -> dict[str, Any]:
    """Scope eligibility uses only preregistered causal gates, never scores/IDs."""
    endpoints = np.asarray(endpoint_samples, int)
    if mode == "predictive":
        window = (endpoints >= onset - 100) & (endpoints <= onset)
    elif mode == "reactive":
        window = (endpoints >= onset) & (endpoints <= onset + 20)
    else:
        raise ValueError(mode)
    unique = np.asarray(unique_owner, bool)[endpoints] & window
    matching = unique & (np.asarray(owner, int)[endpoints] == int(foot))
    case = matching & np.asarray(predicted_case_a, bool)
    return {
        "window_endpoint_count": int(np.sum(window)),
        "unique_owner_available": bool(np.any(unique)),
        "matching_owner_available": bool(np.any(matching)),
        "predicted_case_a_available": bool(np.any(case)),
        "eligible": bool(np.any(case)),
        "exclusion_reason": (
            "IN_SCOPE" if np.any(case)
            else "PREDICTED_CASE_A_TRANSITION_UNAVAILABLE"
            if np.any(matching) else "UNIQUE_MATCHING_G0_OWNER_UNAVAILABLE"
        ),
    }


def scoped_scores(
    scores: dict[str, np.ndarray], allowed: np.ndarray, owner: np.ndarray,
    reactive: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Apply causal scope gates before V4 persistence; never use physical onset."""
    output = {name: np.array(scores[name], copy=True) for name in HEAD_NAMES}
    gate = np.asarray(allowed, bool)
    if gate.shape != output["proposal"].shape:
        raise ValueError("scope gate shape mismatch")
    active_argmax = np.argmax(
        np.stack([output[name] for name in HEAD_NAMES[:4]], axis=-1), axis=-1
    ) == 3
    if reactive:
        gate &= active_argmax
        owner_values = np.asarray(owner, int)
        output["foot"][:] = 0.0
        for row in range(len(owner_values)):
            if owner_values[row] >= 0:
                output["foot"][row, owner_values[row]] = 1.0
    output["proposal"][~gate] = 0.0
    return output, active_argmax


def replay_scoped_state(
    scores: dict[str, np.ndarray], loaded: np.ndarray, valid: np.ndarray,
    prefall: np.ndarray, touchdown: np.ndarray, physical_episode: np.ndarray,
    predicted_case_a: np.ndarray, config: OperatingConfig, reactive: bool,
):
    unique, owner = contact_scope_signals(loaded, valid, touchdown, prefall)
    endpoint_owner = owner[ENDPOINTS]
    allowed = np.zeros((len(ENDPOINTS), 2), bool)
    for row, foot in enumerate(endpoint_owner):
        if foot >= 0 and predicted_case_a[row]:
            allowed[row, foot] = True
    transformed, active_argmax = scoped_scores(scores, allowed, endpoint_owner, reactive)
    state = corrected_v4_state(
        transformed, loaded, valid, prefall, touchdown, physical_episode, config,
    )
    return state, active_argmax, unique, owner


@dataclass(frozen=True)
class ContractDecision:
    chosen_contract: str
    next_step: str


def deterministic_contract_decision(
    c1_pass: bool, c1_evidence: bool, c2_pass: bool, c2_evidence: bool,
    c3_valid: bool,
) -> ContractDecision:
    if c1_pass and c1_evidence:
        return ContractDecision(CONTRACTS[1], "TRAIN_SCOPED_PREDICTIVE_SLIP_MODEL")
    if c2_pass and c2_evidence:
        return ContractDecision(CONTRACTS[2], "TRAIN_SCOPED_REACTIVE_SLIP_CONFIRMATION")
    if c3_valid:
        return ContractDecision(CONTRACTS[3], "VALIDATE_TERRAIN_ONLY_CASE_REFLEX")
    return ContractDecision(CONTRACTS[4], "STOP_WALKING_V2_DEPLOYMENT")
