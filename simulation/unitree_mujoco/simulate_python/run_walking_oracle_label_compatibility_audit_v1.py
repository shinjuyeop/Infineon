"""Audit whether frozen Slip/Sink oracle labels mean the same thing in walking.

The frozen Fast Reflex v2 oracle was calibrated on controlled excitation.  It
is intentionally left unchanged here.  This analysis asks whether its absolute
foot-motion proxies remain valid when a locomotion policy produces ordinary
touchdown, loading, stance and push-off motion.

No model, normalization, threshold, persistence or training corpus is changed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SIM = Path(__file__).resolve().parents[2]
DEFAULT_HOMOGENEOUS = SIM / "outputs/walking_domain_failure_audit_v1/homogeneous_traces.npz"
DEFAULT_TRANSITION = SIM / "outputs/walking_terrain_transition_v1_pilot/transition_traces.npz"
DEFAULT_TRANSITION_MANIFEST = SIM / "outputs/walking_terrain_transition_v1_pilot/manifest.json"
DEFAULT_OUTPUT = SIM / "outputs/walking_oracle_label_compatibility_audit_v1"
LOAD_N = 5.0
TOUCHDOWN_TRANSIENT_MS = 10
MIN_EPISODE_MS = 3
NORMAL_WALKING_TERRAINS = frozenset(("marble", "concrete", "sand"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--homogeneous-traces", type=Path, default=DEFAULT_HOMOGENEOUS)
    parser.add_argument("--transition-traces", type=Path, default=DEFAULT_TRANSITION)
    parser.add_argument("--transition-manifest", type=Path, default=DEFAULT_TRANSITION_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def contact_episodes(mask: np.ndarray, minimum_samples: int = MIN_EPISODE_MS) -> list[tuple[int, int]]:
    """Return half-open loaded-contact episodes of at least ``minimum_samples``."""
    values = np.asarray(mask, bool)
    edge = np.diff(np.r_[False, values, False].astype(np.int8))
    starts = np.flatnonzero(edge == 1)
    ends = np.flatnonzero(edge == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)
            if end - start >= minimum_samples]


def gait_phase(force_N: np.ndarray) -> np.ndarray:
    """Reproduce the v1 audit's deterministic FSR-only phase assignment."""
    loaded = np.asarray(force_N) >= LOAD_N
    result = np.full(len(loaded), "AIR", dtype="<U10")
    for start, end in contact_episodes(loaded, 1):
        result[start:min(start + 10, end)] = "TOUCHDOWN"
        result[min(start + 10, end):min(start + 30, end)] = "LOADING"
        result[max(start + 30, end - 10):end] = "PUSH_OFF"
        if end - start > 40:
            result[start + 30:end - 10] = "MID_STANCE"
    return result


def homogeneous_terrain(run_id: str) -> str:
    parts = run_id.split("_")
    if len(parts) < 3 or parts[0] != "homogeneous":
        raise ValueError(f"unexpected homogeneous run id: {run_id}")
    return parts[1]


def transition_case(run_id: str) -> str:
    parts = run_id.split("_")
    if len(parts) < 3 or parts[0] != "walking" or parts[1] not in "ABCD":
        raise ValueError(f"unexpected transition run id: {run_id}")
    return parts[1]


def episode_rows(
    run_id: str,
    domain: str,
    role: str,
    terrain: str,
    force_N: np.ndarray,
    left_contact: np.ndarray,
    foot_xyz: np.ndarray,
    foot_velocity_xyz: np.ndarray,
    frozen_slip: np.ndarray,
    frozen_sink: np.ndarray,
    first_allowed_sample: int = 0,
) -> list[dict[str, object]]:
    loaded = np.asarray(left_contact, bool) & (np.asarray(force_N) >= LOAD_N)
    rows: list[dict[str, object]] = []
    for episode_index, (start, end) in enumerate(contact_episodes(loaded)):
        selected_start = max(start, first_allowed_sample)
        if selected_start >= end:
            continue
        xyz = np.asarray(foot_xyz[selected_start:end], float)
        velocity = np.asarray(foot_velocity_xyz[selected_start:end], float)
        post = min(TOUCHDOWN_TRANSIENT_MS, len(xyz))
        anchor_drift = np.linalg.norm(xyz[:, :2] - xyz[0, :2], axis=1)
        rows.append({
            "run_id": run_id,
            "domain": domain,
            "role": role,
            "terrain": terrain,
            "episode_index": episode_index,
            "start_sample": selected_start,
            "end_sample_exclusive": end,
            "duration_ms": end - selected_start,
            "max_anchor_drift_m": float(anchor_drift.max(initial=0.0)),
            "net_anchor_drift_m": float(np.linalg.norm(xyz[-1, :2] - xyz[0, :2])),
            "max_horizontal_speed_post_touchdown_mps": (
                float(np.linalg.norm(velocity[post:, :2], axis=1).max(initial=0.0))
                if post < len(velocity) else 0.0
            ),
            # Diagnostic only: ankle-body descent is explicitly not Sink GT.
            "max_ankle_drop_from_touchdown_m": float(np.maximum(0.0, xyz[0, 2] - xyz[:, 2]).max(initial=0.0)),
            "frozen_slip_samples": int(np.count_nonzero(frozen_slip[selected_start:end])),
            "frozen_sink_samples": int(np.count_nonzero(frozen_sink[selected_start:end])),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values, float), q))


def main() -> None:
    args = parse_args()
    dry_protocol = {
        "analysis_only": True,
        "frozen_assets_unchanged": True,
        "normal_walking_controls": sorted(NORMAL_WALKING_TERRAINS),
        "slip_semantic_question": "loaded stance-foot tangential drift relative to its contact anchor",
        "sink_semantic_question": "loaded sole penetration relative to the local terrain surface",
    }
    if not args.execute:
        print(json.dumps(dry_protocol, indent=2))
        return

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True)

    with np.load(args.homogeneous_traces, allow_pickle=False) as source:
        homogeneous = {key: source[key] for key in source.files}
    with np.load(args.transition_traces, allow_pickle=False) as source:
        transition = {key: source[key] for key in source.files}
    transition_manifest = json.loads(args.transition_manifest.read_text(encoding="utf-8"))
    manifest_by_id = {str(row["run_id"]): row for row in transition_manifest}

    required = ("run_id", "fusion10", "left_contact", "foot_xyz", "foot_vel_xyz",
                "confirmed_slip", "sustained_sink")
    for name, data in (("homogeneous", homogeneous), ("transition", transition)):
        missing = sorted(set(required) - set(data))
        if missing:
            raise ValueError(f"{name} traces missing {missing}")

    episodes: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []

    for index, raw_run_id in enumerate(homogeneous["run_id"].astype(str)):
        terrain = homogeneous_terrain(raw_run_id)
        if terrain in NORMAL_WALKING_TERRAINS:
            role = "normal_walking_control"
        elif terrain == "ice":
            role = "slip_positive_evidence"
        else:
            raise ValueError(f"unrouted homogeneous terrain: {terrain}")
        force = homogeneous["fusion10"][index, :, :4].sum(axis=1)
        loaded = homogeneous["left_contact"][index] & (force >= LOAD_N)
        slip = homogeneous["confirmed_slip"][index]
        sink = homogeneous["sustained_sink"][index]
        phases = gait_phase(force)
        current = episode_rows(raw_run_id, "homogeneous", role, terrain, force,
                               homogeneous["left_contact"][index], homogeneous["foot_xyz"][index],
                               homogeneous["foot_vel_xyz"][index], slip, sink)
        episodes.extend(current)
        run_rows.append({
            "run_id": raw_run_id,
            "domain": "homogeneous",
            "role": role,
            "terrain_or_case": terrain,
            "fall_occurred": False,
            "loaded_samples": int(np.count_nonzero(loaded)),
            "loaded_episodes": len(current),
            "frozen_slip_positive_samples": int(np.count_nonzero(slip & loaded)),
            "frozen_sink_positive_samples": int(np.count_nonzero(sink & loaded)),
            "frozen_slip_run_positive": bool(np.any(slip & loaded)),
            "frozen_sink_run_positive": bool(np.any(sink & loaded)),
        })
        for phase_name in ("AIR", "TOUCHDOWN", "LOADING", "MID_STANCE", "PUSH_OFF"):
            selected = phases == phase_name
            phase_rows.append({
                "run_id": raw_run_id,
                "domain": "homogeneous",
                "role": role,
                "terrain_or_case": terrain,
                "phase": phase_name,
                "samples": int(np.count_nonzero(selected)),
                "frozen_slip_positive_samples": int(np.count_nonzero(slip & selected)),
                "frozen_sink_positive_samples": int(np.count_nonzero(sink & selected)),
            })

    # Spatial traces are retained as confounded diagnostics: all cases fell and
    # Sand solref was replaced by the hard walking-support value.  They must not
    # become positive or negative label ground truth in this audit.
    for index, raw_run_id in enumerate(transition["run_id"].astype(str)):
        case = transition_case(raw_run_id)
        metadata = manifest_by_id[raw_run_id]
        t0 = metadata.get("T0")
        force = transition["fusion10"][index, :, :4].sum(axis=1)
        loaded = transition["left_contact"][index] & (force >= LOAD_N)
        slip = transition["confirmed_slip"][index]
        sink = transition["sustained_sink"][index]
        phases = gait_phase(force)
        role = "confounded_falling_transition"
        current = episode_rows(raw_run_id, "spatial_transition", role, case, force,
                               transition["left_contact"][index], transition["foot_xyz"][index],
                               transition["foot_vel_xyz"][index], slip, sink,
                               0 if t0 is None else int(t0))
        episodes.extend(current)
        run_rows.append({
            "run_id": raw_run_id,
            "domain": "spatial_transition",
            "role": role,
            "terrain_or_case": case,
            "fall_occurred": bool(metadata["fall_occurred"]),
            "loaded_samples": int(np.count_nonzero(loaded)),
            "loaded_episodes": len(current),
            "frozen_slip_positive_samples": int(np.count_nonzero(slip & loaded)),
            "frozen_sink_positive_samples": int(np.count_nonzero(sink & loaded)),
            "frozen_slip_run_positive": bool(np.any(slip & loaded)),
            "frozen_sink_run_positive": bool(np.any(sink & loaded)),
        })
        for phase_name in ("AIR", "TOUCHDOWN", "LOADING", "MID_STANCE", "PUSH_OFF"):
            selected = phases == phase_name
            phase_rows.append({
                "run_id": raw_run_id,
                "domain": "spatial_transition",
                "role": role,
                "terrain_or_case": case,
                "phase": phase_name,
                "samples": int(np.count_nonzero(selected)),
                "frozen_slip_positive_samples": int(np.count_nonzero(slip & selected)),
                "frozen_sink_positive_samples": int(np.count_nonzero(sink & selected)),
            })

    normal_runs = [row for row in run_rows if row["role"] == "normal_walking_control"]
    slip_evidence_runs = [row for row in run_rows if row["role"] == "slip_positive_evidence"]
    normal_episodes = [row for row in episodes if row["role"] == "normal_walking_control"]
    ice_episodes = [row for row in episodes if row["role"] == "slip_positive_evidence"]
    normal_drift = [float(row["max_anchor_drift_m"]) for row in normal_episodes]
    ice_drift = [float(row["max_anchor_drift_m"]) for row in ice_episodes]
    normal_run_drift = [max(float(row["max_anchor_drift_m"]) for row in normal_episodes
                            if row["run_id"] == run["run_id"]) for run in normal_runs]
    ice_run_drift = [max(float(row["max_anchor_drift_m"]) for row in ice_episodes
                         if row["run_id"] == run["run_id"]) for run in slip_evidence_runs]

    all_normal_slip_false = sum(bool(row["frozen_slip_run_positive"]) for row in normal_runs)
    all_normal_sink_false = sum(bool(row["frozen_sink_run_positive"]) for row in normal_runs)
    all_transition_falls = all(bool(row["fall_occurred"]) for row in run_rows
                               if row["domain"] == "spatial_transition")
    slip_motion_separated = bool(normal_run_drift and ice_run_drift
                                 and min(ice_run_drift) > max(normal_run_drift))

    compatibility = [
        {
            "detector": "slip",
            "frozen_oracle_semantic": "loaded absolute foot horizontal speed above 0.0448878 m/s for 3 ms",
            "walking_target_semantic": "loaded stance-foot tangential drift relative to that contact episode's anchor, after touchdown transient",
            "normal_control_runs": len(normal_runs),
            "normal_false_positive_runs": all_normal_slip_false,
            "positive_evidence_runs": len(slip_evidence_runs),
            "positive_evidence_profiles": 1,
            "positive_motion_separated_from_normal": slip_motion_separated,
            "compatibility": "INCOMPATIBLE",
        },
        {
            "detector": "sink",
            "frozen_oracle_semantic": "ankle z drop from a single trace/T0 reference plus downward ankle speed",
            "walking_target_semantic": "loaded sole penetration relative to the local terrain surface after stable contact",
            "normal_control_runs": len(normal_runs),
            "normal_false_positive_runs": all_normal_sink_false,
            "positive_evidence_runs": 0,
            "positive_evidence_profiles": 0,
            "positive_motion_separated_from_normal": False,
            "compatibility": "INCOMPATIBLE_AND_POSITIVE_GT_MISSING",
        },
    ]

    semantic_contract = {
        "version": "walking_slip_sink_semantics_v1",
        "terrain_name_never_labels_hazard": True,
        "air_samples_are_always_hazard_negative": True,
        "slip": {
            "event": "excess loaded stance-foot tangential motion relative to the contact anchor",
            "minimum_context": "one loaded-contact episode with a separately represented touchdown transient",
            "required_physical_observable": "foot-to-ground tangential relative displacement or an equivalent contact-point motion signal",
            "forbidden_proxy": "absolute world-frame foot speed alone",
            "current_evidence": "homogeneous Ice contact-anchor drift is separated from Marble/Concrete/hardened-Sand normal controls",
            "status": "SEMANTICS_READY_THRESHOLD_NOT_CALIBRATED",
        },
        "sink": {
            "event": "continued downward sole penetration relative to the local terrain surface during loaded contact",
            "minimum_context": "stable loaded-contact reference followed by terrain-relative penetration",
            "required_physical_observable": "local ground height/deformation or sole-ground contact penetration",
            "forbidden_proxy": "ankle body z relative to trace start or touchdown",
            "current_evidence": "none; walking-support-v1 replaces Sand solref with Concrete solref",
            "status": "SEMANTICS_READY_POSITIVE_GROUND_TRUTH_MISSING",
        },
        "spatial_transition_use": {
            "status": "DIAGNOSTIC_ONLY",
            "reason": "all A/B/C/D runs fell and first-fall sample is not available for censoring",
        },
    }

    retraining_gate = {
        "hard_negative_source_ready": True,
        "hard_negative_sources": [
            "homogeneous Marble loaded gait phases",
            "homogeneous Concrete loaded gait phases",
            "homogeneous hardened-Sand loaded gait phases",
        ],
        "slip_positive_source_available": slip_motion_separated,
        "sink_positive_source_available": False,
        "terrain_bounded_retraining_authorized": False,
        "slip_bounded_retraining_authorized": False,
        "sink_bounded_retraining_authorized": False,
        "reason": "freeze retraining until contact-relative Slip threshold calibration and a real terrain-relative Sink positive acquisition are approved",
    }

    summary = {
        "normal_control_runs": len(normal_runs),
        "normal_control_terrain_profiles": len(NORMAL_WALKING_TERRAINS),
        "deterministic_replicates_per_homogeneous_profile": 3,
        "frozen_slip_false_positive_runs": all_normal_slip_false,
        "frozen_sink_false_positive_runs": all_normal_sink_false,
        "normal_contact_anchor_drift_run_max_m": max(normal_run_drift),
        "ice_contact_anchor_drift_run_min_m": min(ice_run_drift),
        "normal_episode_anchor_drift_p95_m": percentile(normal_drift, 95),
        "ice_episode_anchor_drift_p95_m": percentile(ice_drift, 95),
        "slip_motion_separated_from_normal": slip_motion_separated,
        "all_spatial_transition_runs_fell": all_transition_falls,
        "WALKING_FROZEN_SLIP_ORACLE_COMPATIBLE": False,
        "WALKING_FROZEN_SINK_ORACLE_COMPATIBLE": False,
        "WALKING_SLIP_LABEL_SEMANTICS_READY": True,
        "WALKING_SINK_LABEL_SEMANTICS_READY": True,
        "WALKING_SINK_POSITIVE_GROUND_TRUTH_READY": False,
        "WALKING_BOUNDED_RETRAINING_AUTHORIZED": False,
        "WALKING_ORACLE_LABEL_COMPATIBILITY_AUDIT_READY": True,
    }

    write_csv(output / "contact_episode_metrics.csv", episodes)
    write_csv(output / "run_compatibility.csv", run_rows)
    write_csv(output / "phase_label_counts.csv", phase_rows)
    write_csv(output / "compatibility_matrix.csv", compatibility)
    (output / "semantic_contract.json").write_text(json.dumps(semantic_contract, indent=2) + "\n", encoding="utf-8")
    (output / "bounded_retraining_gate.json").write_text(json.dumps(retraining_gate, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "protocol.json").write_text(json.dumps(dry_protocol, indent=2) + "\n", encoding="utf-8")
    report = f"""# Walking Slip/Sink oracle-label compatibility audit v1

The frozen controlled-excitation oracles are **not compatible** with normal
walking labels.  Frozen Slip fired in {all_normal_slip_false}/{len(normal_runs)}
normal Marble/Concrete/hardened-Sand runs, and frozen Sink fired in
{all_normal_sink_false}/{len(normal_runs)} normal runs.

The three runs per homogeneous terrain are deterministic replicates, so these
counts cover three normal physical profiles rather than nine independent
random trials.

Slip now means contact-anchor-relative tangential stance motion, not absolute
world-frame foot speed.  The largest normal-control run drift was
{max(normal_run_drift):.6f} m while the smallest homogeneous-Ice run maximum
was {min(ice_run_drift):.6f} m.  This supports the semantic distinction but
does not freeze a detector threshold.

Sink now means sole penetration relative to the local terrain surface.  The
current ankle-z proxy is rejected.  No valid positive walking Sink ground
truth exists because walking-support-v1 hardens Sand's solref.  A new physical
observable and positive acquisition are required before retraining.

All spatial A/B/C/D runs fell, and the stored traces do not contain a
first-fall sample suitable for censoring.  They remain diagnostic-only.

**Bounded retraining is not authorized by this checkpoint.**  Normal walking
hard negatives are ready, but Slip threshold calibration and real Sink
positive ground truth must be completed first.
"""
    (output / "audit.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
