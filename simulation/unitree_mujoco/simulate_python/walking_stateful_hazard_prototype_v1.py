"""Causal Fusion10 runtime state for the walking hazard prototype.

This module is deployable-input clean: it consumes one Fusion10 sample at a
time and optional explicit reset requests.  Labels, simulator diagnostics,
terrain identity, run metadata, and future samples are intentionally absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from walking_fusion10_observability_audit_v1 import LinearProbe, window_features


CONTACT_ON_N = 5.0
CONTACT_OFF_N = 2.5
LOADING_SAMPLES = 30
FLOAT_BYTES = 4


class DetectorState(str, Enum):
    AIR = "AIR"
    LOADING = "LOADING"
    STABLE_CONTACT = "STABLE_CONTACT"
    HAZARD_RISK = "HAZARD_RISK"
    HAZARD_CONFIRMED = "HAZARD_CONFIRMED"
    RECOVERY = "RECOVERY"
    RESET = "RESET"


@dataclass(frozen=True)
class MachineConfig:
    detector: str
    candidate_id: str
    history_ms: int
    use_touchdown_reference: bool
    risk_enabled: bool
    confirmed_enabled: bool
    require_risk_before_confirm: bool
    risk_on: float
    risk_off: float
    confirmed_on: float
    confirmed_off: float
    risk_persistence: int
    confirmed_persistence: int
    recovery_persistence: int
    confirmation_delta_min: float
    contact_on_n: float = CONTACT_ON_N
    contact_off_n: float = CONTACT_OFF_N
    loading_samples: int = LOADING_SAMPLES


@dataclass(frozen=True)
class RuntimeOutput:
    slip_risk: bool
    slip_confirmed: bool
    sink_risk: bool
    sink_confirmed: bool
    contact_state: str
    time_since_touchdown: int
    reset_reason: str
    risk_score: float
    confirmed_score: float


class ContactPhaseTracker:
    """Infer loaded contact phase using only the first four FSR channels."""

    def __init__(
        self,
        contact_on_n: float = CONTACT_ON_N,
        contact_off_n: float = CONTACT_OFF_N,
        loading_samples: int = LOADING_SAMPLES,
    ) -> None:
        if not 0.0 <= contact_off_n < contact_on_n or loading_samples <= 0:
            raise ValueError("invalid contact hysteresis")
        self.contact_on_n = float(contact_on_n)
        self.contact_off_n = float(contact_off_n)
        self.loading_samples = int(loading_samples)
        self.in_contact = False
        self.elapsed = -1
        self.peak_fsr = 0.0
        self.previous_fsr = 0.0
        self.ema_fsr = 0.0
        self._reset_pulse = False

    def reset(self) -> None:
        self.in_contact = False
        self.elapsed = -1
        self.peak_fsr = 0.0
        self.previous_fsr = 0.0
        self.ema_fsr = 0.0
        self._reset_pulse = True

    def update(self, sample: np.ndarray) -> tuple[DetectorState, int, str]:
        values = np.asarray(sample, dtype=float)
        if values.shape != (10,) or not np.isfinite(values).all():
            raise ValueError("expected one finite Fusion10 sample")
        fsr = float(np.sum(values[:4]))
        if self._reset_pulse:
            self._reset_pulse = False
            return DetectorState.RESET, -1, "explicit_reset"
        if not self.in_contact:
            if fsr < self.contact_on_n:
                return DetectorState.AIR, -1, "none"
            self.in_contact = True
            self.elapsed = 0
            self.peak_fsr = fsr
            self.previous_fsr = fsr
            self.ema_fsr = fsr
            return DetectorState.LOADING, 0, "touchdown"
        if fsr <= self.contact_off_n:
            self.in_contact = False
            self.elapsed = -1
            self.peak_fsr = 0.0
            self.previous_fsr = fsr
            self.ema_fsr = fsr
            return DetectorState.RESET, -1, "contact_loss"
        self.elapsed += 1
        self.peak_fsr = max(self.peak_fsr, fsr)
        self.ema_fsr = 0.95 * self.ema_fsr + 0.05 * fsr
        declining = fsr < self.previous_fsr and fsr < 0.60 * max(self.peak_fsr, 1e-6)
        self.previous_fsr = fsr
        if self.elapsed < self.loading_samples:
            return DetectorState.LOADING, self.elapsed, "none"
        if declining:
            return DetectorState.RECOVERY, self.elapsed, "none"
        return DetectorState.STABLE_CONTACT, self.elapsed, "none"


class CausalFeatureState:
    """Bounded history, touchdown reference, EMA, and cumulative state."""

    def __init__(self, history_ms: int) -> None:
        if history_ms <= 0:
            raise ValueError("history_ms must be positive")
        self.history_ms = int(history_ms)
        self.history: list[np.ndarray] = []
        self.reference: np.ndarray | None = None
        self.cumulative = np.zeros(10, dtype=np.float64)
        self.count = 0
        self.ema = np.zeros(10, dtype=np.float64)

    def reset(self) -> None:
        self.history.clear()
        self.reference = None
        self.cumulative.fill(0.0)
        self.count = 0
        self.ema.fill(0.0)

    def update(self, sample: np.ndarray, touchdown: bool) -> None:
        values = np.asarray(sample, dtype=np.float64)
        if values.shape != (10,) or not np.isfinite(values).all():
            raise ValueError("expected one finite Fusion10 sample")
        if touchdown or self.reference is None:
            self.reference = values.copy()
            self.cumulative.fill(0.0)
            self.count = 0
            self.ema = values.copy()
        self.history.append(values.copy())
        if len(self.history) > self.history_ms:
            self.history.pop(0)
        self.count += 1
        self.cumulative += values
        self.ema = 0.95 * self.ema + 0.05 * values

    def vector(self, use_touchdown_reference: bool) -> np.ndarray:
        if not self.history:
            raise ValueError("feature history is empty")
        padded = self.history
        if len(padded) < self.history_ms:
            padded = [padded[0]] * (self.history_ms - len(padded)) + padded
        raw = window_features(np.asarray([padded], dtype=np.float32))[0]
        if not use_touchdown_reference:
            return raw
        if self.reference is None or self.count <= 0:
            state = np.zeros(44, dtype=np.float32)
        else:
            current = self.history[-1]
            mean = self.cumulative / self.count
            variance = np.var(np.asarray(self.history), axis=0)
            fsr_current = float(np.sum(current[:4]))
            fsr_reference = float(np.sum(self.reference[:4]))
            state = np.r_[
                self.count - 1,
                current - self.reference,
                mean - self.reference,
                self.ema - self.reference,
                variance,
                fsr_current - fsr_reference,
                float(np.sum(mean[:4])) - fsr_reference,
                1.0,
            ].astype(np.float32)
        if state.shape != (44,):
            raise ValueError("stateful feature schema mismatch")
        return np.r_[raw, state].astype(np.float32)


class StatefulHazardMachine:
    """Independent risk and confirmation evidence with contact-scoped reset."""

    def __init__(
        self,
        config: MachineConfig,
        risk_probe: LinearProbe,
        confirmed_probe: LinearProbe,
    ) -> None:
        if config.detector not in ("slip", "sink"):
            raise ValueError("detector must be slip or sink")
        if config.risk_persistence <= 0 or config.confirmed_persistence <= 0:
            raise ValueError("persistence must be positive")
        self.config = config
        self.risk_probe = risk_probe
        self.confirmed_probe = confirmed_probe
        self.phase = ContactPhaseTracker(
            config.contact_on_n, config.contact_off_n, config.loading_samples
        )
        self.features = CausalFeatureState(config.history_ms)
        self.state = DetectorState.AIR
        self.risk_count = 0
        self.confirmed_count = 0
        self.recovery_count = 0
        self.risk_entry_confirmed_score = 0.0
        self.pending_reset_reason = "none"

    def request_reset(self, reason: str) -> None:
        if reason not in ("episode_boundary", "fall_censor", "operator_reset"):
            raise ValueError("unsupported reset reason")
        self.pending_reset_reason = reason

    def _clear_hazard(self) -> None:
        self.risk_count = 0
        self.confirmed_count = 0
        self.recovery_count = 0
        self.risk_entry_confirmed_score = 0.0

    def _output(
        self,
        state: DetectorState,
        elapsed: int,
        reason: str,
        risk_score: float = 0.0,
        confirmed_score: float = 0.0,
    ) -> RuntimeOutput:
        risk = state in (DetectorState.HAZARD_RISK, DetectorState.HAZARD_CONFIRMED)
        confirmed = state == DetectorState.HAZARD_CONFIRMED
        return RuntimeOutput(
            slip_risk=bool(risk and self.config.detector == "slip"),
            slip_confirmed=bool(confirmed and self.config.detector == "slip"),
            sink_risk=bool(risk and self.config.detector == "sink"),
            sink_confirmed=bool(confirmed and self.config.detector == "sink"),
            contact_state=state.value,
            time_since_touchdown=int(elapsed),
            reset_reason=reason,
            risk_score=float(risk_score),
            confirmed_score=float(confirmed_score),
        )

    def step(self, fusion10_sample: np.ndarray) -> RuntimeOutput:
        if self.pending_reset_reason != "none":
            reason = self.pending_reset_reason
            self.pending_reset_reason = "none"
            self.phase = ContactPhaseTracker(
                self.config.contact_on_n,
                self.config.contact_off_n,
                self.config.loading_samples,
            )
            self.features.reset()
            self._clear_hazard()
            self.state = DetectorState.RESET
            return self._output(DetectorState.RESET, -1, reason)
        phase, elapsed, reason = self.phase.update(fusion10_sample)
        if phase in (DetectorState.AIR, DetectorState.RESET):
            self.features.reset()
            self._clear_hazard()
            self.state = phase
            return self._output(phase, elapsed, reason)
        touchdown = reason == "touchdown"
        self.features.update(fusion10_sample, touchdown)
        vector = self.features.vector(self.config.use_touchdown_reference)[None, :]
        risk_score = float(self.risk_probe.positive_score(vector)[0])
        confirmed_score = float(self.confirmed_probe.positive_score(vector)[0])
        if phase == DetectorState.LOADING:
            self._clear_hazard()
            self.state = DetectorState.LOADING
            return self._output(self.state, elapsed, reason, risk_score, confirmed_score)
        if self.state not in (
            DetectorState.HAZARD_RISK,
            DetectorState.HAZARD_CONFIRMED,
            DetectorState.RECOVERY,
        ):
            self.state = phase
        risk_active = self.config.risk_enabled and risk_score >= self.config.risk_on
        self.risk_count = self.risk_count + 1 if risk_active else 0
        if self.state in (DetectorState.STABLE_CONTACT, DetectorState.RECOVERY):
            if self.risk_count >= self.config.risk_persistence:
                self.state = DetectorState.HAZARD_RISK
                self.risk_entry_confirmed_score = confirmed_score
                self.confirmed_count = 0
        can_confirm = (
            self.config.confirmed_enabled
            and confirmed_score >= self.config.confirmed_on
            and (
                not self.config.require_risk_before_confirm
                or self.state == DetectorState.HAZARD_RISK
            )
            and (
                not self.config.require_risk_before_confirm
                or confirmed_score - self.risk_entry_confirmed_score
                >= self.config.confirmation_delta_min
            )
        )
        self.confirmed_count = self.confirmed_count + 1 if can_confirm else 0
        if self.confirmed_count >= self.config.confirmed_persistence:
            self.state = DetectorState.HAZARD_CONFIRMED
        low_evidence = (
            risk_score < self.config.risk_off
            and confirmed_score < self.config.confirmed_off
        )
        if self.state in (DetectorState.HAZARD_RISK, DetectorState.HAZARD_CONFIRMED):
            self.recovery_count = self.recovery_count + 1 if low_evidence else 0
            if self.recovery_count >= self.config.recovery_persistence:
                self.state = DetectorState.RECOVERY
                self._clear_hazard()
        elif phase == DetectorState.RECOVERY:
            self.state = DetectorState.RECOVERY
        return self._output(self.state, elapsed, reason, risk_score, confirmed_score)


def estimate_resources(
    config: MachineConfig,
    risk_probe: LinearProbe,
    confirmed_probe: LinearProbe,
) -> dict[str, object]:
    feature_count = 124 if config.use_touchdown_reference else 80
    parameters = risk_probe.parameter_count + confirmed_probe.parameter_count
    history_bytes = config.history_ms * 10 * FLOAT_BYTES
    persistent_floats = 10 + 10 + 10 + 10 + feature_count + 8
    state_bytes = persistent_floats * FLOAT_BYTES + 32
    feature_macs = config.history_ms * 10 * 8
    linear_macs = 2 * feature_count
    return {
        "parameter_count": parameters,
        "persistent_state_bytes": int(state_bytes),
        "history_buffer_bytes": int(history_bytes),
        "estimated_macs_per_1khz_sample": int(feature_macs + linear_macs),
        "feature_count": feature_count,
        "e84_u55_compatible": True,
        "vela_unsupported_ops_expected": False,
        "int8_or_vela_executed": False,
    }


def selection_key(row: dict[str, object]) -> tuple[object, ...]:
    latency = row.get("confirmed_latency_median_ms")
    return (
        -int(row["invalid_firing_samples"]),
        -int(row["normal_confirmed_fp_runs"]),
        -int(row["pre_onset_confirmed_runs"]),
        float(row["confirmed_run_recall"]),
        float(row["risk_run_coverage"]),
        -(float(latency) if latency is not None else 1e12),
        -int(row["parameter_count"]),
        -int(row["persistent_state_bytes"]),
        -int(row["candidate_order"]),
    )


def select_candidate(rows: list[dict[str, object]]) -> tuple[dict[str, object], bool]:
    if not rows:
        raise ValueError("candidate rows are empty")
    passing = [row for row in rows if bool(row["mandatory_gate_pass"])]
    pool = passing if passing else rows
    return max(pool, key=selection_key), bool(passing)
