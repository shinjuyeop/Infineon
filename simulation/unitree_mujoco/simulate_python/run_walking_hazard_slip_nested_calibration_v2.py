"""Nested-calibrate a walking Slip label-oracle, then run blind validation v2.

This script uses only the immutable d2209cd development pool for selection.
It writes the fixed protocol and selection artifacts before collecting any
v03/v04/v05 outer-validation trace.  It does not train a model or change a
runtime, INT8, E84, detector, or System-v1 threshold.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from itertools import combinations
from pathlib import Path
import subprocess
import time

import numpy as np

from g1_upstream_locomotion import TESTED_POLICY_SHA256, UPSTREAM_REVISION
from hil_sensor import HIL_SENSOR_CHANNELS
from terrain_profiles import TERRAIN_PROFILES
from walking_hazard_ground_truth_v1 import SENSOR_RATE_HZ
from walking_hazard_oracle_calibration_v1 import (
    DIVERSITY_TRACE_KEYS,
    PERSISTENCE_GRID_MS,
    SLIP_THRESHOLD_GRID_M,
    trace_digest,
)
from walking_hazard_slip_nested_calibration_v2 import (
    TARGET_SPEEDS,
    development_fold_manifest,
    episode_latency_rows,
    nested_candidate_metrics,
    run_max_m,
    select_nested_candidate,
    slip_fire,
)
from run_walking_hazard_ground_truth_v1 import (
    DEFAULT_POLICY,
    PHYSICS_TIMESTEP_S,
    write_csv,
)
from run_walking_hazard_oracle_calibration_v1 import (
    RobustnessCondition,
    Variation,
    collect_run,
    hardened_sand_profile,
)


SIMULATION_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = SIMULATION_DIR / "outputs" / "walking_hazard_slip_nested_calibration_v2"
DEVELOPMENT_OUTPUT = SIMULATION_DIR / "outputs" / "walking_hazard_oracle_calibration_v1"
STARTING_CHECKPOINT = "d2209cdb49c496839a16396f201ab0322515171b"
DEFAULT_DURATION_S = 3.0
VARIATIONS = (
    Variation(3, 202608193, 1.0 / 6.0, 0.060),
    Variation(4, 202608194, 1.0 / 2.0, 0.080),
    Variation(5, 202608195, 5.0 / 6.0, 0.100),
)
EXPECTED_FIRST_COMMAND_OBSERVATIONS_S = (0.080, 0.100, 0.120)
IMMUTABLE_SHA256 = {
    "simulation/unitree_mujoco/simulate_python/walking_hazard_oracle_calibration_v1.py":
        "5056222ae11d1760e01118d76be4e53207682e8a86c8295d319478159f663506",
    "simulation/unitree_mujoco/simulate_python/run_walking_hazard_oracle_calibration_v1.py":
        "310911ff0a9579e7caad1419c17f800c1aa7f713be067d85a9d19ee2188e6ba6",
    "simulation/outputs/walking_hazard_oracle_calibration_v1/protocol.json":
        "3e8f2c70a81b3e686f1967818981cada408ec2549abe31d5d57b53d919c224a3",
    "simulation/outputs/walking_hazard_oracle_calibration_v1/manifest.json":
        "b1360af75e6d57d59270df1bde3939b41d70a537067ae9754c044bda7a041aa8",
    "simulation/outputs/walking_hazard_oracle_calibration_v1/traces.npz":
        "1bbede7770400a844f75da2bce5157e4b50e5f8119ea5e893760943c7ed40423",
    "simulation/outputs/walking_hazard_oracle_calibration_v1/split_manifest.json":
        "9def8445c67366ac4b1417b458f675fef983c152e80654e794b4b2405871d8bc",
    "simulation/outputs/walking_hazard_oracle_calibration_v1/summary.json":
        "62d4da1dfb86b1ad2ef754624d256be594c0a199b5e916cc45ab044da7ee34bc",
    "simulation/unitree_mujoco/simulate_python/walking_hazard_ground_truth_v1.py":
        "e12cdb8699f70d448766de04f3dcc5e952ac43155e09540512428b408dba2b81",
    "simulation/unitree_mujoco/simulate_python/run_walking_hazard_ground_truth_v1.py":
        "c7e53baa44f9cbd6241949f2c797c370a6185be872247da820168b4cb8e0a503",
    "simulation/outputs/walking_hazard_ground_truth_v1_pilot/protocol.json":
        "04f8302cd8b8232de0bec2ef742287a9fd4ad4422ac8eb8a6cad9d49d76aac88",
    "simulation/outputs/walking_hazard_ground_truth_v1_pilot/summary.json":
        "96a4c29ce495aea1e7785ae95124652bf1017ac3d9618727d4ef17c9bc37aa10",
    "simulation/outputs/walking_hazard_ground_truth_v1_pilot/traces.npz":
        "cba2cf5f4ce915135a8607ee384e5b9b04d76a4ab2bbc0e3608b27e0b8d946d7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="finalize already saved 36-run outer traces without reacquisition",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def immutable_hash_audit() -> dict[str, object]:
    actual = {name: sha256(REPOSITORY_ROOT / name) for name in IMMUTABLE_SHA256}
    mismatches = {
        name: {"expected": IMMUTABLE_SHA256[name], "actual": value}
        for name, value in actual.items() if value != IMMUTABLE_SHA256[name]
    }
    if mismatches:
        raise ValueError(f"immutable input hash mismatch: {mismatches}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    checkpoint_is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", STARTING_CHECKPOINT, head],
        cwd=REPOSITORY_ROOT,
    ).returncode == 0
    if not checkpoint_is_ancestor:
        raise ValueError(f"{STARTING_CHECKPOINT} is not an ancestor of {head}")
    return {
        "starting_checkpoint": STARTING_CHECKPOINT,
        "execution_head": head,
        "checkpoint_is_ancestor": checkpoint_is_ancestor,
        "verified_sha256": actual,
        "mismatch_count": 0,
    }


def outer_conditions(duration_s: float) -> list[RobustnessCondition]:
    if not np.isclose(duration_s, 3.0):
        raise ValueError("the blind outer protocol requires exactly 3.0 seconds")
    conditions = []
    for variation in VARIATIONS:
        for speed in sorted(TARGET_SPEEDS):
            for name, terrain, profile, role in (
                ("marble_native", "marble", TERRAIN_PROFILES["marble"], "hard_negative"),
                ("concrete_native", "concrete", TERRAIN_PROFILES["concrete"], "hard_negative"),
                ("sand_hardened", "sand", hardened_sand_profile(), "hard_negative"),
                ("ice_native", "ice", TERRAIN_PROFILES["ice"], "slip_candidate"),
            ):
                conditions.append(RobustnessCondition(
                    name, terrain, profile, role, float(speed), variation
                ))
    if len(conditions) != 36:
        raise AssertionError("outer matrix must contain 36 runs")
    return conditions


def protocol(
    duration_s: float,
    conditions: list[RobustnessCondition],
    hashes: dict[str, object],
) -> dict[str, object]:
    return {
        "dataset": "walking_hazard_slip_nested_calibration_v2",
        "purpose": "nested development-only Slip calibration plus new blind outer validation",
        "starting_checkpoint": STARTING_CHECKPOINT,
        "immutable_input_hash_audit": hashes,
        "data_boundaries": {
            "selection_source": "d2209cd v00/v01/v02 development pool only",
            "outer_source": "new v03/v04/v05 blind physical simulation",
            "test_or_final_used": False,
            "sink_recollected_or_reselected": False,
            "outer_data_access_during_selection": False,
            "selection_artifacts_written_before_outer_collection": True,
        },
        "sample_rate_hz": SENSOR_RATE_HZ,
        "duration_s": duration_s,
        "physics_timestep_s": PHYSICS_TIMESTEP_S,
        "candidate_grid": {
            "declared_before_outer_validation": True,
            "refined_after_development": False,
            "slip_threshold_grid_m": list(SLIP_THRESHOLD_GRID_M),
            "persistence_grid_ms": list(PERSISTENCE_GRID_MS),
            "hardcoded_50mm_selection": False,
        },
        "nested_folds": [
            {"fold_id": 0, "selection_variations": [1, 2], "internal_validation_variation": 0},
            {"fold_id": 1, "selection_variations": [0, 2], "internal_validation_variation": 1},
            {"fold_id": 2, "selection_variations": [0, 1], "internal_validation_variation": 2},
        ],
        "selection_rule": {
            "mandatory_first": [
                "zero normal false-positive runs and samples",
                "all physical-positive runs and every Slip speed detected",
                "zero AIR/post-fall/touchdown-transient positives",
                "pass every development fold selection and internal validation partition",
            ],
            "robustness_then_latency": (
                "retain candidates within both 80% and 5 mm of the best worst-fold "
                "envelope margin, then minimize worst-fold p95 and mean physical-onset latency"
            ),
            "deterministic_ties_only": "lower persistence, then lower threshold",
            "longer_persistence_preferred": False,
        },
        "outer_variation_design": {
            "declared_before_execution": True,
            "kind": "bounded deterministic locomotion phase and command-onset combinations",
            "rationale": (
                "new phase fractions and three additional 50-Hz command ticks create "
                "first command observations at 80/100/120 ms, distinct from prior "
                "20/40/60 ms; no sensor, actuator, contact, or label noise"
            ),
            "controller_update_period_s": 0.020,
            "expected_first_nonzero_command_observations_s": list(
                EXPECTED_FIRST_COMMAND_OBSERVATIONS_S
            ),
            "variations": [asdict(value) for value in VARIATIONS],
        },
        "outer_matrix": {
            "normal": "Marble/Concrete/hardened Sand x 0.10/0.15/0.20 x v03/v04/v05",
            "slip": "Ice x 0.10/0.15/0.20 x v03/v04/v05",
            "normal_runs": 27,
            "slip_runs": 9,
            "total_runs": len(conditions),
        },
        "outer_acceptance": {
            "normal_false_positive_runs": 0,
            "ice_physical_source_valid_runs": 9,
            "detected_per_speed": "3/3",
            "label_mask_violations": 0,
            "duplicate_pairs": 0,
            "leakage": 0,
            "adjust_after_outer_failure": False,
        },
        "latency_semantics": {
            "physical_onset_to_fire": (
                "elapsed sample indices from first post-touchdown slip-valid sample "
                "to persistence fire, at 1 kHz"
            ),
            "threshold_crossing_to_persistence_completion": (
                "inclusive sample-count duration from the first crossing in the firing "
                "streak to completion, at 1 kHz"
            ),
            "model_inference": "not measured; this is a physical label-oracle, not model inference",
        },
        "freeze_scope_if_pass": "walking Slip physical label-oracle for dataset generation only",
        "runtime_or_model_changes": False,
        "controller": {
            "upstream_revision": UPSTREAM_REVISION,
            "tested_policy_sha256": TESTED_POLICY_SHA256,
        },
        "channels": list(HIL_SENSOR_CHANNELS),
        "conditions": [{
            "run_id": item.run_id,
            "condition_name": item.condition_name,
            "terrain_name": item.terrain_name,
            "profile_name": item.profile.name,
            "acquisition_role": item.acquisition_role,
            "walking_speed_mps": item.walking_speed_mps,
            "variation_index": item.variation.index,
            "variation_seed": item.variation.seed,
            "initial_locomotion_phase_fraction": item.variation.initial_locomotion_phase_fraction,
            "command_onset_delay_s": item.variation.command_onset_delay_s,
            "expected_first_nonzero_policy_command_time_s":
                EXPECTED_FIRST_COMMAND_OBSERVATIONS_S[item.variation.index - 3],
            "friction": item.profile.friction,
            "solref": item.profile.solref,
            "solimp": item.profile.solimp,
        } for item in conditions],
        "overwrite_policy": "refuse any non-empty output directory",
    }


def load_development() -> tuple[list[dict[str, object]], list[dict[str, np.ndarray]]]:
    all_manifest = json.loads((DEVELOPMENT_OUTPUT / "manifest.json").read_text())
    keep = [
        i for i, row in enumerate(all_manifest)
        if row["acquisition_role"] in {"hard_negative", "slip_candidate"}
    ]
    manifests = [all_manifest[i] for i in keep]
    with np.load(DEVELOPMENT_OUTPUT / "traces.npz", allow_pickle=False) as packed:
        trace_keys = [
            key for key in packed.files
            if packed[key].ndim >= 2 and packed[key].shape[:2] == (54, 3000)
            and key not in {
                "slip_oracle_calibration_candidate", "sink_oracle_calibration_candidate"
            }
        ]
        traces = [
            {key: packed[key][index].copy() for key in trace_keys}
            for index in keep
        ]
    if len(manifests) != 36 or len(traces) != 36:
        raise ValueError("expected 36 Slip-development runs")
    return manifests, traces


def _max_abs(first: np.ndarray, second: np.ndarray) -> float:
    a, b = np.asarray(first), np.asarray(second)
    if a.dtype.kind in "bUS" or b.dtype.kind in "bUS":
        return float(np.count_nonzero(a != b))
    finite = np.isfinite(a) & np.isfinite(b)
    return 0.0 if not np.any(finite) else float(np.max(np.abs(a[finite] - b[finite])))


def duplicate_rows(
    manifests: list[dict[str, object]], traces: list[dict[str, np.ndarray]]
) -> list[dict[str, object]]:
    rows = []
    for condition in sorted({str(row["condition_name"]) for row in manifests}):
        for speed in sorted(TARGET_SPEEDS):
            indices = [
                i for i, row in enumerate(manifests)
                if row["condition_name"] == condition
                and np.isclose(float(row["walking_speed_mps"]), speed)
            ]
            for first, second in combinations(indices, 2):
                digest_a, digest_b = trace_digest(traces[first]), trace_digest(traces[second])
                differences = {
                    key: _max_abs(traces[first][key], traces[second][key])
                    for key in DIVERSITY_TRACE_KEYS
                }
                rows.append({
                    "condition_name": condition,
                    "walking_speed_mps": speed,
                    "run_id_a": manifests[first]["run_id"],
                    "run_id_b": manifests[second]["run_id"],
                    "variation_index_a": manifests[first]["variation_index"],
                    "variation_index_b": manifests[second]["variation_index"],
                    "full_endpoint_sha256_a": digest_a,
                    "full_endpoint_sha256_b": digest_b,
                    "byte_identical": digest_a == digest_b,
                    "endpoint_identical": all(value == 0.0 for value in differences.values()),
                    "duplicate": digest_a == digest_b or all(
                        value == 0.0 for value in differences.values()
                    ),
                    "key_channel_max_abs_or_mismatch_json": json.dumps(
                        differences, sort_keys=True, separators=(",", ":")
                    ),
                })
    return rows


def latency_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    groups = {(float(row["walking_speed_mps"]), int(row["variation_index"])) for row in rows}
    for speed, variation in sorted(groups):
        group = [
            row for row in rows
            if np.isclose(float(row["walking_speed_mps"]), speed)
            and int(row["variation_index"]) == variation and row["detected"]
        ]
        label = np.asarray([row["physical_onset_to_fire_latency_ms"] for row in group], float)
        persistence = np.asarray([
            row["threshold_crossing_to_persistence_completion_ms"] for row in group
        ], float)
        item: dict[str, object] = {
            "walking_speed_mps": speed,
            "variation_index": variation,
            "detected_episode_count": len(group),
            "model_inference_latency": "not_applicable",
        }
        for name, values in (("physical_onset_to_fire", label), ("threshold_to_completion", persistence)):
            item.update({
                f"{name}_mean_ms": None if not values.size else float(np.mean(values)),
                f"{name}_median_ms": None if not values.size else float(np.median(values)),
                f"{name}_p95_ms": None if not values.size else float(np.percentile(values, 95)),
                f"{name}_max_ms": None if not values.size else float(np.max(values)),
            })
        result.append(item)
    return result


def evaluate_outer(
    manifests: list[dict[str, object]],
    traces: list[dict[str, np.ndarray]],
    threshold_m: float,
    persistence_ms: int,
) -> tuple[list[dict[str, object]], np.ndarray, list[dict[str, object]]]:
    fires = np.asarray([
        slip_fire(trace, threshold_m, persistence_ms) for trace in traces
    ])
    normal_max = max(
        float(run_max_m(traces[i])) for i, row in enumerate(manifests)
        if row["acquisition_role"] == "hard_negative" and run_max_m(traces[i]) is not None
    )
    run_rows = []
    episodes = []
    for i, (metadata, trace) in enumerate(zip(manifests, traces)):
        role = str(metadata["acquisition_role"])
        maximum = run_max_m(trace)
        physical_valid = bool(
            role == "hard_negative" or (
                metadata["stable_loaded_contact_pre_fall"]
                and maximum is not None and maximum > normal_max
            )
        )
        current_episodes = episode_latency_rows(
            str(metadata["run_id"]), float(metadata["walking_speed_mps"]),
            int(metadata["variation_index"]), trace, fires[i], persistence_ms,
        )
        for row in current_episodes:
            row.update({
                "condition_name": metadata["condition_name"],
                "acquisition_role": role,
            })
        episodes.extend(current_episodes)
        detected_latencies = [
            float(row["physical_onset_to_fire_latency_ms"])
            for row in current_episodes if row["detected"]
        ]
        first = np.flatnonzero(fires[i])
        run_rows.append({
            "run_id": metadata["run_id"],
            "condition_name": metadata["condition_name"],
            "profile_name": metadata["profile_name"],
            "acquisition_role": role,
            "walking_speed_mps": metadata["walking_speed_mps"],
            "variation_index": metadata["variation_index"],
            "physical_source_valid": physical_valid,
            "observable_run_max_m": maximum,
            "outer_normal_envelope_max_m": normal_max,
            "threshold_m": threshold_m,
            "persistence_ms": persistence_ms,
            "detected": bool(first.size),
            "false_positive": bool(role == "hard_negative" and first.size),
            "positive_missed": bool(role == "slip_candidate" and physical_valid and not first.size),
            "positive_samples": int(np.count_nonzero(fires[i])),
            "first_detection_sample": None if not first.size else int(first[0]),
            "first_detection_time_s": None if not first.size else float(trace["time_s"][first[0]]),
            "detected_episode_count": len(detected_latencies),
            "mean_physical_onset_to_fire_latency_ms": (
                None if not detected_latencies else float(np.mean(detected_latencies))
            ),
            "max_physical_onset_to_fire_latency_ms": (
                None if not detected_latencies else float(np.max(detected_latencies))
            ),
            "air_positive_count": int(np.count_nonzero(fires[i] & ~trace["left_contact"])),
            "post_fall_positive_count": int(np.count_nonzero(fires[i] & ~trace["pre_fall_valid"])),
            "touchdown_transient_positive_count": int(
                np.count_nonzero(fires[i] & trace["touchdown_transient"])
            ),
            "fall_occurred": metadata["fall_occurred"],
            "first_fall_sample": metadata["first_fall_sample"],
        })
    return run_rows, fires, episodes


def fall_censor_rows(
    manifests: list[dict[str, object]], traces: list[dict[str, np.ndarray]], fires: np.ndarray
) -> list[dict[str, object]]:
    return [{
        "run_id": row["run_id"],
        "acquisition_role": row["acquisition_role"],
        "walking_speed_mps": row["walking_speed_mps"],
        "variation_index": row["variation_index"],
        "fall_occurred": row["fall_occurred"],
        "first_fall_sample": row["first_fall_sample"],
        "first_fall_time_s": row["first_fall_time_s"],
        "pre_fall_valid_samples": int(np.count_nonzero(trace["pre_fall_valid"])),
        "post_fall_censored_samples": int(np.count_nonzero(~trace["pre_fall_valid"])),
        "air_samples": int(np.count_nonzero(~trace["left_contact"])),
        "touchdown_transient_samples": int(np.count_nonzero(trace["touchdown_transient"])),
        "selected_slip_air_positives": int(np.count_nonzero(fire & ~trace["left_contact"])),
        "selected_slip_post_fall_positives": int(np.count_nonzero(fire & ~trace["pre_fall_valid"])),
        "selected_slip_touchdown_positives": int(np.count_nonzero(fire & trace["touchdown_transient"])),
    } for row, trace, fire in zip(manifests, traces, fires)]


def save_outer(
    path: Path, traces: list[dict[str, np.ndarray]],
    manifests: list[dict[str, object]], fires: np.ndarray,
) -> None:
    packed = {key: np.asarray([trace[key] for trace in traces]) for key in traces[0]}
    packed.update({
        "run_id": np.asarray([row["run_id"] for row in manifests]),
        "terrain_name": np.asarray([row["terrain_name"] for row in manifests]),
        "profile_name": np.asarray([row["profile_name"] for row in manifests]),
        "acquisition_role": np.asarray([row["acquisition_role"] for row in manifests]),
        "walking_speed_mps": np.asarray([row["walking_speed_mps"] for row in manifests]),
        "variation_index": np.asarray([row["variation_index"] for row in manifests]),
        "variation_seed": np.asarray([row["variation_seed"] for row in manifests]),
        "slip_label_oracle_frozen_candidate": fires,
    })
    np.savez_compressed(path, **packed)


def make_plots(
    output: Path, manifests: list[dict[str, object]], traces: list[dict[str, np.ndarray]],
    fires: np.ndarray, threshold_m: float, latency_rows: list[dict[str, object]],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = output / "plots"
    directory.mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    for role, marker in (("hard_negative", "o"), ("slip_candidate", "x")):
        xs, ys = [], []
        for i, row in enumerate(manifests):
            if row["acquisition_role"] == role:
                xs.append(float(row["walking_speed_mps"]) + 0.003 * (int(row["variation_index"]) - 4))
                ys.append(float(run_max_m(traces[i])) * 1000)
        axis.scatter(xs, ys, marker=marker, label=role)
    axis.axhline(threshold_m * 1000, color="black", linestyle="--", label="selected threshold")
    axis.set_xlabel("walking speed [m/s]")
    axis.set_ylabel("Slip observable run maximum [mm]")
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "outer_envelope_by_speed_variation.png", dpi=150)
    plt.close(figure)

    index = next(i for i, row in enumerate(manifests) if row["acquisition_role"] == "slip_candidate")
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(traces[index]["time_s"], traces[index]["tangential_anchor_drift_m"] * 1000)
    axis.step(
        traces[index]["time_s"], fires[index] * threshold_m * 1000,
        where="post", label="label-oracle fire",
    )
    axis.axhline(threshold_m * 1000, color="black", linestyle="--")
    axis.set_xlabel("simulation time [s]")
    axis.set_ylabel("tangential anchor drift [mm]")
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "outer_ice_timeline.png", dpi=150)
    plt.close(figure)

    detected = [row for row in latency_rows if row["detected"]]
    figure, axis = plt.subplots(figsize=(9, 5))
    for variation in (3, 4, 5):
        values = [row for row in detected if int(row["variation_index"]) == variation]
        axis.scatter(
            [row["walking_speed_mps"] for row in values],
            [row["physical_onset_to_fire_latency_ms"] for row in values],
            label=f"v{variation:02d}", alpha=0.7,
        )
    axis.set_xlabel("walking speed [m/s]")
    axis.set_ylabel("physical onset to fire [ms]")
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / "outer_episode_latency.png", dpi=150)
    plt.close(figure)


def audit_markdown(summary: dict[str, object]) -> str:
    return f"""# Walking Slip nested calibration + blind validation v2

The fixed grid was selected with three leave-one-variation-out folds over the
immutable d2209cd v00/v01/v02 development pool.  Whole runs and their contact
episodes stayed together.  No test/final data or v03/v04/v05 outer trace was
available to the selector.  Selection completed before outer acquisition.

The selected dataset-generation physical label-oracle is
`{summary['selected_candidate']['threshold_m']} m / {summary['selected_candidate']['persistence_ms']} ms`.
All-fold-pass candidates first met zero normal FP, all-speed/all-run physical
positive detection, and zero AIR/post-fall/touchdown violations.  A sufficient
margin band then admitted candidates within both 80% and 5 mm of the best
worst-fold margin; worst-fold p95 and mean latency decided next.  Longer
persistence received no preference.

The blind outer matrix contains {summary['outer_run_count']} new 3-second,
1-kHz runs: 27 normal and 9 Ice.  v03/v04/v05 use new phase fractions and
command onsets, with observed first commands recorded in the manifest.

- Outer normal false-positive runs: {summary['outer_normal_false_positive_runs']}
- Outer valid Ice physical-source runs: {summary['outer_ice_physical_source_valid_runs']}/9
- Outer Ice detections by speed: {summary['outer_ice_detections_by_speed']}
- AIR/post-fall/touchdown violations: {summary['outer_label_violation_count']}
- Duplicate pairs: {summary['outer_duplicate_pair_count']}
- Leakage: {summary['leakage_count']}
- Failure reasons: {summary['failure_reasons']}

Latency is reported per contact episode and by speed/variation.  Physical
label onset is distinct from model inference: no model inference occurred.

## Readiness gates

""" + "\n".join(
        f"- {name}={str(value).lower()}" for name, value in summary["gates"].items()
    ) + """

The Slip oracle is frozen only as a physical label-oracle for dataset
generation.  The d2209cd Sink candidate is carried forward without selection
or recollection.  No runtime threshold, trained model, INT8/E84 artifact,
frozen detector, or System-v1 behavior changed.
"""


def main() -> None:
    args = parse_args()
    conditions = outer_conditions(args.duration_s)
    hashes = immutable_hash_audit()
    planned = protocol(args.duration_s, conditions, hashes)
    if not args.execute:
        print(json.dumps(planned, indent=2))
        return
    output = args.output_dir.resolve()
    if args.finalize_existing:
        required = output / "outer_validation_traces.npz"
        if not required.is_file():
            raise FileNotFoundError(required)
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    if not args.policy_path.is_file():
        raise FileNotFoundError(args.policy_path)
    policy_path = args.policy_path.resolve()
    policy_hash = sha256(policy_path)
    if policy_hash != TESTED_POLICY_SHA256:
        raise ValueError(f"policy hash mismatch: {policy_hash}")
    output.mkdir(parents=True, exist_ok=True)
    planned["controller"].update({"policy_path": str(policy_path), "policy_sha256": policy_hash})

    # Selection phase: only immutable development inputs exist in this scope.
    dev_manifests, dev_traces = load_development()
    folds = development_fold_manifest(dev_manifests)
    candidates = nested_candidate_metrics(
        dev_manifests, dev_traces, SLIP_THRESHOLD_GRID_M, PERSISTENCE_GRID_MS
    )
    selected = select_nested_candidate(candidates)
    if selected is None:
        raise RuntimeError("no Slip candidate passed every development fold")
    selection = {
        **selected,
        "development_source_sha256": IMMUTABLE_SHA256[
            "simulation/outputs/walking_hazard_oracle_calibration_v1/traces.npz"
        ],
        "outer_trace_count_at_selection": 0,
        "outer_data_access_during_selection": False,
        "selection_completed_before_outer_collection": True,
        "production_runtime_threshold_changed": False,
    }
    if not args.finalize_existing:
        (output / "protocol.json").write_text(json.dumps(planned, indent=2) + "\n")
        (output / "development_fold_manifest.json").write_text(json.dumps(folds, indent=2) + "\n")
        write_csv(output / "nested_candidate_metrics.csv", candidates)
        (output / "nested_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    selection_hash = sha256(output / "nested_selection.json")
    started = time.perf_counter()
    if args.finalize_existing:
        outer_saved = json.loads((output / "outer_validation_manifest.json").read_text())
        manifests = [
            {key: value for key, value in row.items() if key != "full_endpoint_sha256"}
            for row in outer_saved["runs"]
        ]
        with np.load(output / "outer_validation_traces.npz", allow_pickle=False) as packed:
            trace_keys = [
                key for key in packed.files
                if packed[key].ndim >= 2 and packed[key].shape[:2] == (36, 3000)
                and key != "slip_label_oracle_frozen_candidate"
            ]
            traces = [
                {key: packed[key][index].copy() for key in trace_keys}
                for index in range(36)
            ]
        print(
            "finalizing 36 previously saved blind outer traces without reacquisition",
            flush=True,
        )
    else:
        print(
            f"selected from development only: {selected['threshold_m']:.3f}m/"
            f"{selected['persistence_ms']}ms; beginning blind outer acquisition",
            flush=True,
        )
        traces = []
        manifests = []
        for index, condition in enumerate(conditions):
            trace, metadata, _ = collect_run(condition, policy_path, args.duration_s)
            traces.append(trace)
            manifests.append(metadata)
            elapsed = time.perf_counter() - started
            eta = elapsed / (index + 1) * (len(conditions) - index - 1)
            print(
                f"[{index + 1}/{len(conditions)}] {condition.run_id} "
                f"phase={condition.variation.initial_locomotion_phase_fraction:.3f} "
                f"onset={condition.variation.command_onset_delay_s * 1000:.0f}ms "
                f"command_observed={float(metadata['first_nonzero_policy_command_time_s']) * 1000:.0f}ms "
                f"fall={metadata['fall_occurred']} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    metrics, fires, episodes = evaluate_outer(
        manifests, traces, float(selected["threshold_m"]), int(selected["persistence_ms"])
    )
    duplicates = duplicate_rows(manifests, traces)
    latency_groups = latency_summary([
        row for row in episodes if row["acquisition_role"] == "slip_candidate"
    ])
    dev_ids = {str(row["run_id"]) for row in dev_manifests}
    outer_ids = {str(row["run_id"]) for row in manifests}
    overlap = sorted(dev_ids & outer_ids)
    command_observed = sorted({
        round(float(row["first_nonzero_policy_command_time_s"]), 9) for row in manifests
    })
    command_diversity_ready = bool(
        len(command_observed) == 3
        and np.allclose(command_observed, EXPECTED_FIRST_COMMAND_OBSERVATIONS_S, atol=1e-9)
    )
    normal_fp = sum(bool(row["false_positive"]) for row in metrics)
    ice_valid = sum(
        row["acquisition_role"] == "slip_candidate" and bool(row["physical_source_valid"])
        for row in metrics
    )
    detections_by_speed = {
        f"{speed:.2f}": int(sum(
            row["acquisition_role"] == "slip_candidate"
            and np.isclose(float(row["walking_speed_mps"]), speed)
            and bool(row["detected"])
            for row in metrics
        ))
        for speed in sorted(TARGET_SPEEDS)
    }
    violations = sum(
        int(row["air_positive_count"]) + int(row["post_fall_positive_count"])
        + int(row["touchdown_transient_positive_count"])
        for row in metrics
    )
    duplicate_count = sum(bool(row["duplicate"]) for row in duplicates)
    leakage_count = len(overlap) + int(folds["run_fold_role_leakage_count"])
    nested_ready = bool(selected["all_folds_pass"] and folds["run_fold_role_leakage_count"] == 0)
    diversity_ready = bool(command_diversity_ready and duplicate_count == 0)
    outer_ready = bool(
        normal_fp == 0 and ice_valid == 9
        and all(value == 3 for value in detections_by_speed.values())
        and violations == 0 and duplicate_count == 0 and leakage_count == 0
    )
    sink_summary = json.loads((DEVELOPMENT_OUTPUT / "summary.json").read_text())
    sink_ready = bool(sink_summary["gates"]["WALKING_SINK_ORACLE_CALIBRATION_READY"])
    freeze_ready = bool(nested_ready and diversity_ready and outer_ready)
    gates = {
        "WALKING_SLIP_NESTED_CALIBRATION_READY": nested_ready,
        "WALKING_SLIP_OUTER_VARIATION_DIVERSITY_READY": diversity_ready,
        "WALKING_SLIP_OUTER_VALIDATION_READY": outer_ready,
        "WALKING_SLIP_LABEL_ORACLE_FREEZE_READY": freeze_ready,
        "WALKING_ORACLE_ROBUSTNESS_READY": bool(freeze_ready and sink_ready),
        "WALKING_BOUNDED_RETRAINING_READY": bool(freeze_ready and sink_ready),
    }
    failure_reasons = []
    if normal_fp:
        failure_reasons.append(f"{normal_fp} outer normal false-positive runs")
    if ice_valid != 9:
        failure_reasons.append(f"only {ice_valid}/9 Ice physical sources valid")
    if any(value != 3 for value in detections_by_speed.values()):
        failure_reasons.append(f"Ice detections by speed were {detections_by_speed}")
    if violations:
        failure_reasons.append(f"{violations} invalid-mask positives")
    if duplicate_count:
        failure_reasons.append(f"{duplicate_count} duplicate outer pairs")
    if leakage_count:
        failure_reasons.append(f"{leakage_count} leakage findings")
    if not command_diversity_ready:
        failure_reasons.append(f"unexpected command observations {command_observed}")
    summary = {
        "starting_checkpoint": STARTING_CHECKPOINT,
        "outer_run_count": len(manifests),
        "outer_run_counts": {
            "hard_negative": sum(row["acquisition_role"] == "hard_negative" for row in manifests),
            "slip_candidate": sum(row["acquisition_role"] == "slip_candidate" for row in manifests),
        },
        "selected_candidate": {
            "threshold_m": selected["threshold_m"],
            "persistence_ms": selected["persistence_ms"],
            "freeze_scope": "physical Slip label-oracle for dataset generation only",
            "production_runtime_changed": False,
        },
        "selection_artifact_sha256_before_outer_collection": selection_hash,
        "outer_data_access_during_selection": False,
        "development_fold_leakage_count": folds["run_fold_role_leakage_count"],
        "development_outer_run_overlap": overlap,
        "observed_first_nonzero_command_times_s": command_observed,
        "outer_normal_false_positive_runs": normal_fp,
        "outer_ice_physical_source_valid_runs": ice_valid,
        "outer_ice_detections_by_speed": detections_by_speed,
        "outer_label_violation_count": violations,
        "outer_duplicate_pair_count": duplicate_count,
        "leakage_count": leakage_count,
        "latency_by_speed_variation": latency_groups,
        "sink_carried_forward": {
            "source_checkpoint": STARTING_CHECKPOINT,
            "threshold_m": sink_summary["selected_candidates"]["sink"]["threshold_m"],
            "persistence_ms": sink_summary["selected_candidates"]["sink"]["persistence_ms"],
            "recollected": False,
            "reselected": False,
            "ready": sink_ready,
        },
        "models_retrained": False,
        "runtime_thresholds_changed": False,
        "int8_e84_system_v1_or_frozen_detector_changed": False,
        "failure_reasons": failure_reasons,
        "wall_time_s": time.perf_counter() - started,
        "gates": gates,
        **gates,
    }
    outer_manifest = {
        "role": "blind outer validation; never selection",
        "selection_artifact_sha256_before_outer_collection": selection_hash,
        "outer_data_access_during_selection": False,
        "development_outer_run_overlap_count": len(overlap),
        "sample_rate_hz": SENSOR_RATE_HZ,
        "expected_sample_count_per_run": 3000,
        "runs": [{
            **row,
            "full_endpoint_sha256": trace_digest(trace),
        } for row, trace in zip(manifests, traces)],
    }
    (output / "outer_validation_manifest.json").write_text(
        json.dumps(outer_manifest, indent=2) + "\n"
    )
    write_csv(output / "outer_validation_metrics.csv", metrics)
    write_csv(output / "outer_episode_latency.csv", episodes)
    write_csv(output / "outer_latency_by_speed_variation.csv", latency_groups)
    write_csv(output / "duplicate_trace_audit.csv", duplicates)
    write_csv(output / "fall_censor_audit.csv", fall_censor_rows(manifests, traces, fires))
    save_outer(output / "outer_validation_traces.npz", traces, manifests, fires)
    make_plots(
        output, manifests, traces, fires, float(selected["threshold_m"]),
        [row for row in episodes if row["acquisition_role"] == "slip_candidate"],
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "audit.md").write_text(audit_markdown(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
