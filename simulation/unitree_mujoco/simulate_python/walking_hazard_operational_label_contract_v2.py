"""Versioned walking hazard label and reflex-contract definitions.

All physical quantities in this module are label-only.  Runtime-capable fields
are contract names for causal Fusion10 estimators and never receive a physical
oracle, simulator phase, run identity, terrain identity, or future sample.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from walking_bounded_retraining_v1 import physical_oracle


SAMPLE_RATE_HZ = 1000
SLIP_RISK_HORIZON_MS = 100
SINK_PENETRATION_THRESHOLD_M = 0.0055
SINK_PENETRATION_PERSISTENCE_MS = 20
SUPPORT_DROP_THRESHOLD_M = 0.010
RECOVERY_DROP_THRESHOLD_M = 0.020
DOWNWARD_SPEED_THRESHOLD_MPS = 0.050
SUPPORT_PERSISTENCE_MS = 20

OFFLINE_ONLY_FIELDS = (
    "slip_physical_onset",
    "slip_physical_active",
    "sink_physical_onset",
    "sink_physical_active",
    "fall_censor",
    "valid_loaded_contact",
)
RUNTIME_CAPABLE_FIELDS = (
    "slip_risk",
    "slip_evidence_persistent",
    "sink_support_degradation_risk",
    "sink_evidence_persistent",
    "contact_state",
    "recovery_required",
)
DEPRECATED_FIELDS = ("slip", "sink", "hazard_confirmed")


def onset_pulses(active: np.ndarray) -> np.ndarray:
    values = np.asarray(active, bool)
    if values.ndim != 1:
        raise ValueError("active label must be one-dimensional")
    return values & ~np.r_[False, values[:-1]]


def first_fall_censor(pre_fall_valid: np.ndarray) -> np.ndarray:
    valid = np.asarray(pre_fall_valid, bool)
    if valid.ndim != 1:
        raise ValueError("pre_fall_valid must be one-dimensional")
    output = np.zeros(len(valid), bool)
    invalid = np.flatnonzero(~valid)
    if invalid.size:
        output[int(invalid[0]):] = True
    return output


def valid_loaded_contact(trace: dict[str, np.ndarray]) -> np.ndarray:
    return (
        np.asarray(trace["loaded_contact"], bool)
        & ~np.asarray(trace["touchdown_transient"], bool)
        & ~first_fall_censor(trace["pre_fall_valid"])
    )


def risk_target_from_physical(
    physical_active: np.ndarray,
    valid: np.ndarray,
    episode_id: np.ndarray,
    horizon_ms: int,
) -> np.ndarray:
    """Create an offline risk target; future onset is never a runtime input."""
    active = np.asarray(physical_active, bool)
    allowed = np.asarray(valid, bool)
    episode = np.asarray(episode_id, int)
    if not (active.shape == allowed.shape == episode.shape) or horizon_ms < 0:
        raise ValueError("risk target inputs must align and horizon must be nonnegative")
    output = active & allowed
    for onset in np.flatnonzero(onset_pulses(active)):
        current_episode = int(episode[onset])
        start = max(0, int(onset) - int(horizon_ms))
        indices = np.arange(start, int(onset) + 1)
        eligible = allowed[indices] & (episode[indices] == current_episode)
        output[indices[eligible]] = True
    return output


def _episode_pelvis_drop(trace: dict[str, np.ndarray], valid: np.ndarray) -> np.ndarray:
    episode = np.asarray(trace["contact_episode_id"], int)
    pelvis_z = np.asarray(trace["pelvis_xyz"], float)[:, 2]
    output = np.zeros(len(valid), np.float64)
    reference: dict[int, float] = {}
    for index in range(len(valid)):
        item = int(episode[index])
        if not valid[index] or item < 0:
            continue
        if item not in reference:
            reference[item] = float(pelvis_z[index])
        output[index] = max(0.0, reference[item] - float(pelvis_z[index]))
    return output


def sustained_within_episode(
    condition: np.ndarray,
    valid: np.ndarray,
    episode_id: np.ndarray,
    persistence_ms: int,
) -> np.ndarray:
    selected = np.asarray(condition, bool)
    allowed = np.asarray(valid, bool)
    episode = np.asarray(episode_id, int)
    if not (selected.shape == allowed.shape == episode.shape) or persistence_ms <= 0:
        raise ValueError("invalid sustained-label inputs")
    output = np.zeros(len(selected), bool)
    count = 0
    previous_episode = -1
    for index, active in enumerate(selected & allowed):
        current_episode = int(episode[index])
        if current_episode != previous_episode:
            count = 0
        count = count + 1 if active and current_episode >= 0 else 0
        output[index] = count >= persistence_ms
        previous_episode = current_episode
    return output


def sink_candidate_labels(trace: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return the three fixed, label-only physical Sink candidates."""
    valid = valid_loaded_contact(trace)
    episode = np.asarray(trace["contact_episode_id"], int)
    penetration = physical_oracle(trace, "sink") & valid
    pelvis_drop = _episode_pelvis_drop(trace, valid)
    downward_speed = np.maximum(
        0.0, -np.asarray(trace["pelvis_velocity_xyz"], float)[:, 2]
    )
    support_condition = penetration & (
        (pelvis_drop >= SUPPORT_DROP_THRESHOLD_M)
        | (downward_speed >= DOWNWARD_SPEED_THRESHOLD_MPS)
    )
    support = sustained_within_episode(
        support_condition, valid, episode, SUPPORT_PERSISTENCE_MS
    )
    recovery_condition = support & (
        pelvis_drop >= RECOVERY_DROP_THRESHOLD_M
    ) & (downward_speed >= DOWNWARD_SPEED_THRESHOLD_MPS)
    recovery = sustained_within_episode(
        recovery_condition, valid, episode, SUPPORT_PERSISTENCE_MS
    )
    return {
        "A_penetration_risk": penetration,
        "B_support_degradation_risk": support,
        "C_recovery_required_sink": recovery,
        "pelvis_drop_m": pelvis_drop,
        "downward_speed_mps": downward_speed,
    }


def operational_offline_labels(trace: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    valid = valid_loaded_contact(trace)
    slip_active = physical_oracle(trace, "slip") & valid
    sink_active = physical_oracle(trace, "sink") & valid
    sink_candidates = sink_candidate_labels(trace)
    return {
        "slip_physical_onset": onset_pulses(slip_active),
        "slip_physical_active": slip_active,
        "slip_risk_target": risk_target_from_physical(
            slip_active, valid, trace["contact_episode_id"], SLIP_RISK_HORIZON_MS
        ),
        "sink_physical_onset": onset_pulses(sink_active),
        "sink_physical_active": sink_active,
        "fall_censor": first_fall_censor(trace["pre_fall_valid"]),
        "valid_loaded_contact": valid,
        **sink_candidates,
    }


def transition_count(values: np.ndarray) -> int:
    active = np.asarray(values, bool)
    return int(np.count_nonzero(active[1:] != active[:-1])) if len(active) > 1 else 0


def pulse_lengths(values: np.ndarray) -> list[int]:
    active = np.asarray(values, bool)
    padded = np.r_[False, active, False].astype(np.int8)
    change = np.diff(padded)
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1)
    return (ends - starts).astype(int).tolist()


def _entry(
    namespace: str,
    purpose: str,
    generation: str,
    allowed_information: list[str],
    future_information: str,
    runtime_available: bool,
    valid_region: str,
    reset_condition: str,
    invalid_handling: str,
    reflex_relation: str,
    false_positive: str,
    early_warning: str,
    latency_anchor: str,
) -> dict[str, Any]:
    return {
        "namespace": namespace,
        "purpose": purpose,
        "generation_criterion": generation,
        "allowed_information": allowed_information,
        "future_information": future_information,
        "runtime_available": runtime_available,
        "valid_region": valid_region,
        "reset_condition": reset_condition,
        "AIR_touchdown_post_fall": invalid_handling,
        "fast_reflex_relation": reflex_relation,
        "false_positive_definition": false_positive,
        "early_warning_definition": early_warning,
        "latency_anchor": latency_anchor,
    }


def label_contract_v2() -> dict[str, Any]:
    offline = {
        "slip_physical_onset": _entry(
            "offline-only", "mark first physical Slip sample per episode",
            "rising edge of locked 50 mm anchor-drift/3 ms physical oracle",
            ["simulator anchor drift", "loaded-contact validity"], "none", False,
            "valid loaded pre-fall contact", "new episode/contact loss/fall censor",
            "always false", "evaluation anchor only; never triggers reflex",
            "not applicable", "not applicable", "this onset sample",
        ),
        "slip_physical_active": _entry(
            "offline-only", "measure physically active Slip",
            "locked anchor-drift oracle active after persistence",
            ["simulator anchor drift", "loaded-contact validity"], "none", False,
            "valid loaded pre-fall contact", "contact loss/new episode/fall censor",
            "always false", "evaluation only",
            "runtime firing without risk-target support", "not applicable",
            "slip_physical_onset",
        ),
        "sink_physical_onset": _entry(
            "offline-only", "mark first physical penetration event sample",
            "rising edge of locked 5.5 mm/20 ms penetration-change oracle",
            ["simulator penetration", "loaded-contact validity"], "none", False,
            "valid loaded pre-fall contact", "new episode/contact loss/fall censor",
            "always false", "evaluation anchor only",
            "not applicable", "not applicable", "this onset sample",
        ),
        "sink_physical_active": _entry(
            "offline-only", "measure sustained physical penetration",
            "locked penetration-change oracle active after persistence",
            ["simulator penetration", "loaded-contact validity"], "none", False,
            "valid loaded pre-fall contact", "contact loss/new episode/fall censor",
            "always false", "physical oracle only; no production trigger",
            "not applicable", "not applicable", "sink_physical_onset",
        ),
        "fall_censor": _entry(
            "offline-only", "exclude samples at and after first fall",
            "first false pre_fall_valid sample through run end",
            ["simulator fall diagnostic"], "none; first fall only", False,
            "whole trace", "new run", "defines post-fall rather than being excluded",
            "never triggers reflex", "not applicable", "not applicable", "first fall",
        ),
        "valid_loaded_contact": _entry(
            "offline-only", "define calibration/evaluation support",
            "loaded_contact AND not touchdown transient AND before first fall",
            ["simulator contact", "FSR load", "fall censor"], "none", False,
            "stable loaded pre-fall contact", "contact loss/new episode/fall censor",
            "false", "gates offline metrics only",
            "not applicable", "not applicable", "touchdown/loading completion",
        ),
    }
    runtime = {
        "slip_risk": _entry(
            "runtime-capable", "causal Fast Reflex trigger candidate",
            "locked D 100 ms causal Fusion10 score/state machine",
            ["current/past Fusion10", "causal contact/touchdown state"],
            "runtime: none; offline target may look up to 100 ms to physical onset",
            True, "causally inferred stable contact", "contact loss/new touchdown/fall censor",
            "must be false", "may trigger Slip reflex without physical confirmation",
            "normal-run firing or >100 ms pre-onset firing",
            "0..100 ms before onset is valid; active/post-onset firing is detection",
            "slip_physical_onset for offline margin only",
        ),
        "slip_evidence_persistent": _entry(
            "runtime-capable", "report sustained causal Slip evidence",
            "independent persistent score/hysteresis state; not physical truth",
            ["current/past Fusion10", "prior score/state"], "none", True,
            "causally inferred stable contact", "contact loss/new touchdown/fall censor",
            "must be false", "may maintain/latch a reflex already triggered by slip_risk",
            "persistence on normal run", "not an oracle-anticipation label",
            "slip_risk entry",
        ),
        "sink_support_degradation_risk": _entry(
            "runtime-capable-proposal", "predict physically grounded support loss",
            "future estimator target is candidate B; no approved runtime estimator exists",
            ["future implementation: current/past deployable sensors only"],
            "none at runtime; simulator physical quantities are label-only", False,
            "causally inferred stable contact", "contact loss/new touchdown/fall censor",
            "must be false", "production trigger prohibited until readiness",
            "normal walking firing", "not approved", "candidate physical onset",
        ),
        "sink_evidence_persistent": _entry(
            "runtime-capable-proposal", "report sustained Sink evidence",
            "independent causal persistence; current prototype is not ready",
            ["future implementation: current/past deployable sensors only"], "none", False,
            "causally inferred stable contact", "contact loss/new touchdown/fall censor",
            "must be false", "production trigger prohibited",
            "persistence during normal gait", "not approved", "sink risk entry",
        ),
        "contact_state": _entry(
            "runtime-capable", "scope detector state causally",
            "FSR hysteresis and touchdown elapsed state",
            ["current/past FSR"], "none", True, "whole stream",
            "contact loss/new touchdown/explicit reset", "AIR/LOADING are explicit states",
            "gates but never proves hazard", "not applicable", "not applicable",
            "FSR contact-on",
        ),
        "recovery_required": _entry(
            "runtime-capable-proposal", "request controller recovery arbitration",
            "evidence-persistent plus controller policy; no recovery benefit is claimed",
            ["approved hazard state", "controller latch/state"], "none", False,
            "contact episode", "contact loss/new touchdown/cooldown completion",
            "must be false", "controller request, not physical oracle",
            "request without approved operational evidence", "not applicable",
            "request transition",
        ),
    }
    deprecated = {
        "slip": _entry(
            "deprecated-ambiguous", "legacy overloaded boolean",
            "historically dataset oracle or runtime detector firing depending context",
            ["context dependent"], "ambiguous", False, "ambiguous", "ambiguous",
            "ambiguous", "replace runtime use with slip_risk and offline use with slip_physical_active",
            "ambiguous", "ambiguous", "ambiguous",
        ),
        "sink": _entry(
            "deprecated-ambiguous", "legacy overloaded boolean",
            "historically penetration/tilt label or runtime detector firing",
            ["context dependent"], "ambiguous", False, "ambiguous", "ambiguous",
            "ambiguous", "replace offline use with sink_physical_active; no runtime trigger yet",
            "ambiguous", "ambiguous", "ambiguous",
        ),
        "hazard_confirmed": _entry(
            "deprecated-ambiguous", "legacy confirmation name",
            "could mean physical oracle or persistent sensor evidence",
            ["context dependent"], "ambiguous", False, "ambiguous", "ambiguous",
            "ambiguous", "replace with hazard-specific physical/evidence fields",
            "ambiguous", "ambiguous", "ambiguous",
        ),
    }
    return {
        "version": 2,
        "offline_only": offline,
        "runtime_capable": runtime,
        "deprecated_or_ambiguous": deprecated,
        "namespace_unique": len(set(offline) | set(runtime) | set(deprecated)) == 15,
        "physical_oracle_is_runtime_contract": False,
    }


def reflex_activation_contract_v2() -> dict[str, Any]:
    common = {
        "contact_loss": "immediate reset to NO_HAZARD and clear all latches",
        "new_touchdown": "start a new episode in NO_HAZARD; no prior evidence carries over",
        "fall_censor": "offline evaluation terminates; runtime safety supervisor is outside this proposal",
        "duplicate_reflex": "one trigger latch per contact episode; suppress retrigger through COOLDOWN",
        "simultaneous_slip_sink": "Slip is the only enabled hazard; Sink remains diagnostic until separately ready",
        "terrain_priority": "slip_risk is not suppressed by Terrain context; Terrain transition is contextual metadata",
    }
    states = {
        "NO_HAZARD": {
            "entry": "reset, cooldown completion, or stable contact without evidence",
            "maintain": "no approved risk output",
            "release": "slip_risk true",
            "minimum_dwell_ms": 0, "hysteresis": "risk-on threshold is owned by locked detector",
        },
        "RISK": {
            "entry": "slip_risk true in stable contact",
            "maintain": "risk above hysteresis-off or minimum dwell incomplete",
            "release": "evidence persistent -> EVIDENCE_PERSISTENT; evidence clears -> COOLDOWN",
            "minimum_dwell_ms": 20, "hysteresis": "locked on/off score thresholds",
        },
        "EVIDENCE_PERSISTENT": {
            "entry": "independent slip_evidence_persistent true after RISK",
            "maintain": "persistent evidence or recovery request latched",
            "release": "controller accepts request -> RECOVERY_REQUIRED; evidence clears -> COOLDOWN",
            "minimum_dwell_ms": 20, "hysteresis": "independent persistence/off threshold",
        },
        "RECOVERY_REQUIRED": {
            "entry": "approved controller policy accepts persistent evidence",
            "maintain": "single reflex command latch",
            "release": "command completion, evidence clear, or contact reset -> COOLDOWN/NO_HAZARD",
            "minimum_dwell_ms": 50, "hysteresis": "controller-owned latch; no repeated command",
        },
        "COOLDOWN": {
            "entry": "risk/evidence release or reflex completion",
            "maintain": "250 ms or remainder of contact episode",
            "release": "cooldown elapsed with no evidence; contact reset immediately clears",
            "minimum_dwell_ms": 250, "hysteresis": "new risk cannot duplicate the latched reflex",
        },
    }
    return {
        "version": 2,
        "proposal_only": True,
        "production_implementation_changed": False,
        "states": states,
        "common_reset_and_arbitration": common,
        "trigger_semantics": {
            "slip": "slip_risk may trigger; never wait for slip_physical_active",
            "sink": "production trigger prohibited until operational-label readiness",
            "physical_oracle_input": False,
        },
    }


def legacy_mapping_rows() -> list[dict[str, str]]:
    return [
        {"legacy_field": "slip", "legacy_semantics": "dataset physical label OR runtime detector firing", "v2_proposed_mapping": "offline: slip_physical_active; runtime: slip_risk", "status": "ambiguous_deprecated"},
        {"legacy_field": "sink", "legacy_semantics": "penetration/tilt label OR runtime detector firing", "v2_proposed_mapping": "offline: sink_physical_active; runtime: disabled pending sink_support_degradation_risk readiness", "status": "ambiguous_deprecated"},
        {"legacy_field": "HAZARD_REFLEX_REQUIRED", "legacy_semantics": "slip OR sink firing independent of terrain", "v2_proposed_mapping": "enabled slip_risk OR future approved sink risk", "status": "proposal_only"},
        {"legacy_field": "CASE_REFLEX_REQUIRED", "legacy_semantics": "case A+slip OR case B+sink", "v2_proposed_mapping": "terrain-context match tag; must not suppress approved slip_risk", "status": "proposal_only"},
        {"legacy_field": "RECOVERY_REQUIRED", "legacy_semantics": "Terrain transition case C/D symbolic recovery", "v2_proposed_mapping": "rename terrain_transition_recovery_required; reserve recovery_required for controller arbitration", "status": "name_collision"},
        {"legacy_field": "mismatch/dual-hazard flags", "legacy_semantics": "derived from ambiguous slip/sink booleans", "v2_proposed_mapping": "derive only from explicit *_risk fields and readiness mask", "status": "proposal_only"},
    ]


def readiness_dependency_graph(flags: dict[str, bool]) -> dict[str, bool]:
    """Validate the fixed authorization dependencies without mutating flags."""
    required = {
        "WALKING_OPERATIONAL_LABEL_DATA_READY",
        "WALKING_OPERATIONAL_LABEL_SPLIT_INTEGRITY_READY",
        "WALKING_OPERATIONAL_LABEL_CAUSAL_CONTRACT_READY",
        "WALKING_SLIP_RISK_LABEL_READY",
        "WALKING_SLIP_REFLEX_CONTRACT_READY",
        "WALKING_SLIP_NEW_BLIND_HOLDOUT_AUTHORIZED",
        "WALKING_SINK_PHYSICAL_LABEL_READY",
        "WALKING_SINK_OPERATIONAL_LABEL_READY",
        "WALKING_SINK_REFLEX_CONTRACT_READY",
        "WALKING_SYSTEM_SCHEMA_MIGRATION_AUTHORIZED",
        "WALKING_BOUNDED_RETRAINING_V2_AUTHORIZED",
        "WALKING_INT8_PREPARATION_AUTHORIZED",
    }
    if set(flags) != required:
        raise ValueError("readiness flag schema mismatch")
    slip_dependencies = bool(
        flags["WALKING_OPERATIONAL_LABEL_DATA_READY"]
        and flags["WALKING_OPERATIONAL_LABEL_SPLIT_INTEGRITY_READY"]
        and flags["WALKING_OPERATIONAL_LABEL_CAUSAL_CONTRACT_READY"]
        and flags["WALKING_SLIP_RISK_LABEL_READY"]
        and flags["WALKING_SLIP_REFLEX_CONTRACT_READY"]
    )
    return {
        "schema_complete": True,
        "slip_holdout_dependency_pass": (
            not flags["WALKING_SLIP_NEW_BLIND_HOLDOUT_AUTHORIZED"]
            or slip_dependencies
        ),
        "sink_reflex_dependency_pass": (
            not flags["WALKING_SINK_REFLEX_CONTRACT_READY"]
            or flags["WALKING_SINK_OPERATIONAL_LABEL_READY"]
        ),
        "forced_false_authorizations_pass": not any((
            flags["WALKING_SYSTEM_SCHEMA_MIGRATION_AUTHORIZED"],
            flags["WALKING_BOUNDED_RETRAINING_V2_AUTHORIZED"],
            flags["WALKING_INT8_PREPARATION_AUTHORIZED"],
        )),
    }
