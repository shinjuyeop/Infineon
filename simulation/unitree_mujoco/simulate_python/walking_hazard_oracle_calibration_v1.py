"""Pure helpers for walking physical-oracle persistence and split audits."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from itertools import combinations

import numpy as np


PERSISTENCE_GRID_MS = (3, 5, 10, 20)
SLIP_THRESHOLD_GRID_M = (0.010, 0.015, 0.020, 0.030, 0.050, 0.075, 0.100)
SINK_THRESHOLD_GRID_M = (0.0040, 0.0045, 0.0050, 0.0055, 0.0060, 0.0070, 0.0080)
DIVERSITY_TRACE_KEYS = (
    "fusion10",
    "left_contact",
    "loaded_contact",
    "foot_xyz",
    "foot_velocity_xyz",
    "pelvis_xyz",
    "pelvis_velocity_xyz",
    "tangential_anchor_drift_m",
    "loaded_penetration_change_m",
    "max_contact_penetration_m",
    "pre_fall_valid",
)


def persistent_oracle(
    observable: np.ndarray,
    valid: np.ndarray,
    contact_episode_id: np.ndarray,
    threshold_m: float,
    persistence_ms: int,
) -> np.ndarray:
    """Apply consecutive 1-kHz persistence without crossing invalid/episode edges."""
    values = np.asarray(observable, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    episodes = np.asarray(contact_episode_id, dtype=np.int64)
    if values.shape != mask.shape or values.shape != episodes.shape:
        raise ValueError("oracle arrays must have the same one-dimensional shape")
    if values.ndim != 1 or threshold_m < 0.0 or persistence_ms <= 0:
        raise ValueError("invalid oracle threshold or persistence")
    result = np.zeros(len(values), dtype=bool)
    count = 0
    previous_episode = -1
    for sample, (value, is_valid, episode) in enumerate(zip(values, mask, episodes)):
        passes = bool(is_valid and episode >= 0 and np.isfinite(value) and value >= threshold_m)
        if not passes:
            count = 0
            previous_episode = int(episode)
            continue
        if int(episode) != previous_episode:
            count = 1
        else:
            count += 1
        previous_episode = int(episode)
        result[sample] = count >= persistence_ms
    return result


def trace_digest(trace: dict[str, np.ndarray]) -> str:
    """Hash the complete selected endpoint state with dtype/shape framing."""
    digest = hashlib.sha256()
    for key in DIVERSITY_TRACE_KEYS:
        value = np.ascontiguousarray(trace[key])
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _max_abs_difference(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return 0.0
    return float(np.max(np.abs(a[finite] - b[finite])))


def duplicate_trace_audit(
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
) -> list[dict[str, object]]:
    """Compare every same-condition replicate pair for exact duplication."""
    if len(manifests) != len(traces):
        raise ValueError("manifest/trace count mismatch")
    grouped: dict[tuple[str, str, float], list[int]] = defaultdict(list)
    for index, row in enumerate(manifests):
        grouped[(
            str(row["condition_name"]),
            str(row["profile_name"]),
            float(row["walking_speed_mps"]),
        )].append(index)
    digests = [trace_digest(trace) for trace in traces]
    rows: list[dict[str, object]] = []
    for key, indices in grouped.items():
        for first, second in combinations(indices, 2):
            endpoint_identical = all(
                np.array_equal(
                    np.asarray(traces[first][name]),
                    np.asarray(traces[second][name]),
                    equal_nan=True,
                )
                for name in DIVERSITY_TRACE_KEYS
            )
            rows.append({
                "condition_name": key[0],
                "profile_name": key[1],
                "walking_speed_mps": key[2],
                "run_id_a": manifests[first]["run_id"],
                "run_id_b": manifests[second]["run_id"],
                "variation_index_a": manifests[first]["variation_index"],
                "variation_index_b": manifests[second]["variation_index"],
                "trace_sha256_a": digests[first],
                "trace_sha256_b": digests[second],
                "byte_identical": digests[first] == digests[second],
                "endpoint_identical": endpoint_identical,
                "fusion10_max_abs_difference": _max_abs_difference(
                    traces[first]["fusion10"], traces[second]["fusion10"]
                ),
                "foot_xyz_max_abs_difference_m": _max_abs_difference(
                    traces[first]["foot_xyz"], traces[second]["foot_xyz"]
                ),
                "contact_mismatch_samples": int(np.count_nonzero(
                    traces[first]["left_contact"] != traces[second]["left_contact"]
                )),
                "duplicate": bool(digests[first] == digests[second] or endpoint_identical),
            })
    return rows


def assign_calibration_splits(
    manifests: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Assign whole variation runs: indices 0/1 train and index 2 validation."""
    grouped: dict[tuple[str, str, float], set[int]] = defaultdict(set)
    rows = []
    for manifest in manifests:
        key = (
            str(manifest["condition_name"]),
            str(manifest["profile_name"]),
            float(manifest["walking_speed_mps"]),
        )
        variation_index = int(manifest["variation_index"])
        grouped[key].add(variation_index)
        split = "calibration_validation" if variation_index % 3 == 2 else "calibration_train"
        rows.append({
            "run_id": manifest["run_id"],
            "condition_name": manifest["condition_name"],
            "profile_name": manifest["profile_name"],
            "acquisition_role": manifest["acquisition_role"],
            "walking_speed_mps": manifest["walking_speed_mps"],
            "variation_index": variation_index,
            "variation_seed": manifest["variation_seed"],
            "split": split,
            "ownership": "whole run and all of its contact episodes",
        })
    missing = {key: sorted(values) for key, values in grouped.items() if not {0, 1, 2} <= values}
    if missing:
        raise ValueError(f"each condition requires variation indices 0,1,2: {missing}")
    return rows


def split_integrity(
    split_rows: list[dict[str, object]],
    episode_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Audit whole-run ownership and contact-episode leakage."""
    split_by_run = {str(row["run_id"]): str(row["split"]) for row in split_rows}
    if len(split_by_run) != len(split_rows):
        raise ValueError("duplicate run_id in split manifest")
    train_runs = {run for run, split in split_by_run.items() if split == "calibration_train"}
    validation_runs = {
        run for run, split in split_by_run.items() if split == "calibration_validation"
    }
    unknown_episode_runs = sorted({
        str(row["run_id"]) for row in episode_rows if str(row["run_id"]) not in split_by_run
    })
    episode_owners: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in episode_rows:
        run_id = str(row["run_id"])
        if run_id in split_by_run:
            episode_owners[(run_id, int(row["contact_episode_id"]))].add(split_by_run[run_id])
    leaking_episodes = [
        f"{run_id}:{episode}" for (run_id, episode), owners in episode_owners.items()
        if len(owners) != 1
    ]
    return {
        "train_run_count": len(train_runs),
        "validation_run_count": len(validation_runs),
        "run_overlap_count": len(train_runs & validation_runs),
        "episode_overlap_count": len(leaking_episodes),
        "unknown_episode_run_count": len(unknown_episode_runs),
        "leaking_episodes": leaking_episodes,
        "unknown_episode_runs": unknown_episode_runs,
        "split_leakage_count": (
            len(train_runs & validation_runs)
            + len(leaking_episodes)
            + len(unknown_episode_runs)
        ),
    }
