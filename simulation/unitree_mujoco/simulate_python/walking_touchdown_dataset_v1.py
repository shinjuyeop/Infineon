"""Touchdown-aligned event schema and extraction for walking terrain data."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from hil_sensor import HIL_SENSOR_CHANNELS


SCHEMA_NAME = "terrain_walking_touchdown_v1"
SCHEMA_VERSION = 1
PHYSICS_TIMESTEP_S = 0.0005
SENSOR_RATE_HZ = 1000
PHYSICS_STEPS_PER_SAMPLE = 2
PRE_SAMPLES = 10
POST_SAMPLES = 50
EVENT_SAMPLES = PRE_SAMPLES + POST_SAMPLES
RELATIVE_TIME_MS = np.arange(-PRE_SAMPLES, POST_SAMPLES, dtype=np.int16)
FSR_THRESHOLD_N = 5.0
MIN_AIR_SAMPLES = 40
CONTACT_CONFIRMATION_SAMPLES = 4
OBSERVATION_CANDIDATES_MS = (1, 2, 5, 10, 15, 20, 30, 50)


@dataclass(frozen=True)
class SensorSample:
    timestamp_s: float
    sensor: np.ndarray
    contact: bool


@dataclass(frozen=True)
class TouchdownEvent:
    touchdown_time_s: float
    timestamps_s: np.ndarray
    sensors: np.ndarray
    contacts: np.ndarray
    fsr_sum: np.ndarray
    fsr_threshold_crossing_time_s: float | None
    valid: bool
    invalid_reason: str


class TouchdownEventCollector:
    """Accept debounced AIR->CONTACT transitions and retain [-10,+50) ms."""

    def __init__(
        self,
        *,
        min_air_samples: int = MIN_AIR_SAMPLES,
        contact_confirmation_samples: int = CONTACT_CONFIRMATION_SAMPLES,
        fsr_threshold_n: float = FSR_THRESHOLD_N,
    ) -> None:
        if min_air_samples < PRE_SAMPLES:
            raise ValueError("minimum AIR duration must cover the pre-contact window")
        if contact_confirmation_samples <= 0 or fsr_threshold_n <= 0.0:
            raise ValueError("confirmation and FSR threshold must be positive")
        self.min_air_samples = min_air_samples
        self.contact_confirmation_samples = contact_confirmation_samples
        self.fsr_threshold_n = fsr_threshold_n
        self.samples: list[SensorSample] = []
        self.events: list[TouchdownEvent] = []
        self._previous_contact = False
        self._air_samples = 0
        self._pending_touchdown: int | None = None
        self._accepted_touchdowns: deque[int] = deque()
        self.rejected_short_air = 0
        self.rejected_contact_chatter = 0
        self.incomplete_at_end = 0

    def append(
        self, timestamp_s: float, sensor: np.ndarray, contact: bool, *, enabled: bool = True
    ) -> None:
        vector = np.asarray(sensor, dtype=np.float64)
        if vector.shape != (len(HIL_SENSOR_CHANNELS),):
            raise ValueError(f"expected sensor shape (10,), got {vector.shape}")
        index = len(self.samples)
        if self.samples:
            spacing = timestamp_s - self.samples[-1].timestamp_s
            if not np.isclose(spacing, 0.001, rtol=0.0, atol=1e-9):
                raise ValueError(f"non-native 1 kHz timestamp spacing: {spacing}")
        self.samples.append(SensorSample(float(timestamp_s), vector.copy(), bool(contact)))

        transition = contact and not self._previous_contact
        if transition and enabled:
            if self._air_samples >= self.min_air_samples and index >= PRE_SAMPLES:
                self._pending_touchdown = index
            else:
                self.rejected_short_air += 1

        if self._pending_touchdown is not None:
            start = self._pending_touchdown
            confirmation_end = start + self.contact_confirmation_samples
            if not contact and index < confirmation_end:
                self._pending_touchdown = None
                self.rejected_contact_chatter += 1
            elif index + 1 >= confirmation_end:
                self._accepted_touchdowns.append(start)
                self._pending_touchdown = None

        if contact:
            self._air_samples = 0
        else:
            self._air_samples += 1
        self._previous_contact = bool(contact)
        self._finalize_ready(index)

    def _finalize_ready(self, latest_index: int) -> None:
        while self._accepted_touchdowns:
            touchdown = self._accepted_touchdowns[0]
            if latest_index < touchdown + POST_SAMPLES - 1:
                break
            self._accepted_touchdowns.popleft()
            start, stop = touchdown - PRE_SAMPLES, touchdown + POST_SAMPLES
            window = self.samples[start:stop]
            timestamps = np.asarray([sample.timestamp_s for sample in window])
            sensors = np.asarray([sample.sensor for sample in window], dtype=np.float64)
            contacts = np.asarray([sample.contact for sample in window], dtype=bool)
            relative = (timestamps - self.samples[touchdown].timestamp_s) * 1000.0
            reasons: list[str] = []
            if sensors.shape != (EVENT_SAMPLES, len(HIL_SENSOR_CHANNELS)):
                reasons.append("incomplete_window")
            if not np.all(np.isfinite(sensors)):
                reasons.append("nan_or_inf")
            if not np.allclose(relative, RELATIVE_TIME_MS, rtol=0.0, atol=1e-6):
                reasons.append("timestamp_alignment")
            if np.any(contacts[:PRE_SAMPLES]) or not contacts[PRE_SAMPLES]:
                reasons.append("contact_alignment")
            fsr_sum = sensors[:, :4].sum(axis=1)
            threshold = fsr_sum >= self.fsr_threshold_n
            crossings = np.flatnonzero(threshold & ~np.r_[False, threshold[:-1]])
            crossing_time = None if crossings.size == 0 else float(timestamps[crossings[0]])
            self.events.append(
                TouchdownEvent(
                    touchdown_time_s=float(self.samples[touchdown].timestamp_s),
                    timestamps_s=timestamps,
                    sensors=sensors,
                    contacts=contacts,
                    fsr_sum=fsr_sum,
                    fsr_threshold_crossing_time_s=crossing_time,
                    valid=not reasons,
                    invalid_reason="|".join(reasons),
                )
            )

    def finish(self) -> None:
        self.incomplete_at_end = len(self._accepted_touchdowns) + int(
            self._pending_touchdown is not None
        )


def event_manifest_row(
    event: TouchdownEvent, event_id: str, metadata: dict[str, Any]
) -> dict[str, Any]:
    crossing = event.fsr_threshold_crossing_time_s
    return {
        "event_id": event_id,
        **metadata,
        "foot": "left",
        "touchdown_sim_time_s": event.touchdown_time_s,
        "fsr_threshold_N": FSR_THRESHOLD_N,
        "fsr_threshold_crossing_time_s": "" if crossing is None else crossing,
        "fsr_touchdown_delta_ms": ""
        if crossing is None
        else (crossing - event.touchdown_time_s) * 1000.0,
        "sample_count": EVENT_SAMPLES,
        "pre_contact_samples": PRE_SAMPLES,
        "post_contact_samples": POST_SAMPLES,
        "valid": int(event.valid),
        "invalid_reason": event.invalid_reason,
    }


def validate_event_split_integrity(rows: list[dict[str, Any]]) -> None:
    """Forbid event-level random ownership across family/session/run groups."""
    family_owners: dict[str, str] = {}
    for row in rows:
        family, split = str(row["surface_family"]), str(row["split"])
        previous = family_owners.setdefault(family, split)
        if previous != split:
            raise ValueError(f"surface family {family} leaks across splits")
    for key in ("surface_seed", "session_id", "run_id"):
        owners: dict[str, str] = {}
        for row in rows:
            value, split = str(row[key]), str(row["split"])
            previous = owners.setdefault(value, split)
            if previous != split:
                raise ValueError(f"{key} leaks across {previous} and {split}")


def stack_events(events: list[TouchdownEvent]) -> dict[str, np.ndarray]:
    if not events:
        return {
            "sensors": np.empty((0, EVENT_SAMPLES, 10), dtype=np.float32),
            "contact": np.empty((0, EVENT_SAMPLES), dtype=bool),
            "fsr_sum": np.empty((0, EVENT_SAMPLES), dtype=np.float32),
            "sample_relative_time_ms": RELATIVE_TIME_MS.copy(),
        }
    return {
        "sensors": np.asarray([event.sensors for event in events], dtype=np.float32),
        "contact": np.asarray([event.contacts for event in events], dtype=bool),
        "fsr_sum": np.asarray([event.fsr_sum for event in events], dtype=np.float32),
        "sample_relative_time_ms": RELATIVE_TIME_MS.copy(),
    }
