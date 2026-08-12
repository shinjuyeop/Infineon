"""Plan or collect native-1-kHz, left-touchdown-aligned walking events."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from controlled_excitation import find_allowed_foot_geom_ids, has_nonfoot_floor_contact
from expanded_terrain_dataset_v1 import (
    ExpandedSurfaceParameters,
    SURFACE_FAMILIES,
    family_for_name,
    make_expanded_run_specification,
    make_expanded_surface_parameters,
    normalized_family_surface,
)
from g1_upstream_locomotion import (
    TESTED_POLICY_SHA256,
    UPSTREAM_REVISION,
    UnitreeG1PretrainedController,
)
from hil_sensor import G1HilSensorReader, HIL_SENSOR_CHANNELS
from run_surface_sampling_rate_study import SIMULATION_DIR, write_dict_rows
from surface_profiles import HFIELD_NAME, SURFACE_FLOOR_NAME
from terrain_dataset_v1 import TERRAIN_LABELS
from terrain_profiles import TERRAIN_PROFILES, apply_terrain_profile
from walking_touchdown_dataset_v1 import (
    CONTACT_CONFIRMATION_SAMPLES,
    EVENT_SAMPLES,
    FSR_THRESHOLD_N,
    MIN_AIR_SAMPLES,
    OBSERVATION_CANDIDATES_MS,
    PHYSICS_STEPS_PER_SAMPLE,
    PHYSICS_TIMESTEP_S,
    POST_SAMPLES,
    PRE_SAMPLES,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    SENSOR_RATE_HZ,
    TouchdownEventCollector,
    TouchdownEvent,
    event_manifest_row,
    stack_events,
    validate_event_split_integrity,
)


SCENE_PATH = (
    SIMULATION_DIR
    / "unitree_mujoco"
    / "unitree_robots"
    / "g1"
    / "scene_walking_surface.xml"
)
OUTPUT_DIR = SIMULATION_DIR / "outputs" / SCHEMA_NAME
DEFAULT_DURATION_S = 2.4
DEFAULT_SETTLING_S = 0.6
DEFAULT_FORWARD_SPEED_MPS = 0.20
CONTACT_MODELS = ("full-body", "foot-spheres-only")
CONTACT_PARAMETER_MODES = (
    "terrain-native",
    "walking-support-v1",
    "hard-solref-audit",
    "hard-solimp-audit",
    "hard-reference-audit",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--terrains", nargs="+", choices=tuple(TERRAIN_LABELS), default=["concrete"])
    parser.add_argument(
        "--families", nargs="+", choices=tuple(family.name for family in SURFACE_FAMILIES),
        default=["multisine"],
    )
    parser.add_argument("--surfaces-per-family", type=int, default=1)
    parser.add_argument("--runs-per-surface", type=int, default=1)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--settling-s", type=float, default=DEFAULT_SETTLING_S)
    parser.add_argument("--walking-speed", type=float, default=DEFAULT_FORWARD_SPEED_MPS)
    parser.add_argument(
        "--contact-model", choices=CONTACT_MODELS, default="full-body",
        help="retain the upstream collision model or restrict surface support to foot spheres",
    )
    parser.add_argument(
        "--contact-parameters",
        choices=CONTACT_PARAMETER_MODES,
        default="terrain-native",
        help="use terrain parameters or hard-contact values for diagnosis only",
    )
    parser.add_argument("--plot", action="store_true", help="save a first-event diagnostic plot")
    parser.add_argument("--gui", action="store_true", help="render the single-run smoke test")
    parser.add_argument(
        "--stop-on-fall",
        action="store_true",
        help="stop a run at the first fall/collision gate and record diagnostics",
    )
    parser.add_argument("--execute", action="store_true", help="run collection; otherwise print plan")
    return parser.parse_args()


def candidate_count(args: argparse.Namespace) -> int:
    return (
        len(args.terrains) * len(args.families) * args.surfaces_per_family * args.runs_per_surface
    )


def protocol(args: argparse.Namespace) -> dict[str, object]:
    if args.surfaces_per_family <= 0 or args.runs_per_surface <= 0:
        raise ValueError("surface and run counts must be positive")
    if args.duration_s <= args.settling_s + POST_SAMPLES / SENSOR_RATE_HZ:
        raise ValueError("duration must leave room after settling for an event window")
    for family in args.families:
        family_for_name(family)
    runs = candidate_count(args)
    return {
        "dataset_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "walking touchdown acquisition foundation; no classifier training",
        "controller": {
            "name": "unitreerobotics/unitree_rl_mjlab pretrained G1-29DOF velocity policy",
            "artifact": "deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx (user supplied; not vendored)",
            "upstream_revision": UPSTREAM_REVISION,
            "tested_policy_sha256": TESTED_POLICY_SHA256,
            "command_mps": [args.walking_speed, 0.0, 0.0],
            "control_rate_hz": 50,
            "controlled_joints": "all 29 joints of the repository's full-body G1",
        },
        "terrain_label_definition": "terrain contacted by the named left sole spheres at touchdown",
        "label_mapping": TERRAIN_LABELS,
        "terrains": args.terrains,
        "surface_families": [vars(family_for_name(name)) for name in args.families],
        "surfaces_per_family": args.surfaces_per_family,
        "runs_per_surface": args.runs_per_surface,
        "candidate_runs": runs,
        "duration_s": args.duration_s,
        "settling_s": args.settling_s,
        "stop_on_fall": args.stop_on_fall,
        "contact_model": args.contact_model,
        "contact_parameters": args.contact_parameters,
        "physics_timestep_s": PHYSICS_TIMESTEP_S,
        "physics_rate_hz": 2000,
        "sensor_rate_hz": SENSOR_RATE_HZ,
        "physics_steps_per_sample": PHYSICS_STEPS_PER_SAMPLE,
        "native_sampling": True,
        "interpolation": False,
        "channels": HIL_SENSOR_CHANNELS,
        "units": ["N", "N", "N", "N", "m/s^2", "m/s^2", "m/s^2", "rad/s", "rad/s", "rad/s"],
        "event_window": {
            "interval_ms": [-PRE_SAMPLES, POST_SAMPLES],
            "interval_semantics": "left-closed/right-open: -10..+49 ms",
            "shape": [EVENT_SAMPLES, len(HIL_SENSOR_CHANNELS)],
            "classifier_input": "only samples at t >= 0",
            "candidate_observation_ms": OBSERVATION_CANDIDATES_MS,
        },
        "touchdown": {
            "ground_truth": "sampled MuJoCo AIR->CONTACT transition for any named left sole sphere against surface_floor",
            "minimum_preceding_air_ms": MIN_AIR_SAMPLES,
            "contact_confirmation_ms": CONTACT_CONFIRMATION_SAMPLES,
            "timestamp": "first CONTACT sample, backdated after confirmation",
            "fsr_diagnostic_threshold_N": FSR_THRESHOLD_N,
        },
        "split_policy": {
            "train": [family.name for family in SURFACE_FAMILIES if family.split == "train"],
            "validation": [family.name for family in SURFACE_FAMILIES if family.split == "validation"],
            "test": [family.name for family in SURFACE_FAMILIES if family.split == "test"],
            "ownership": "whole surface family, surface seed, session and run; never random touchdown event",
        },
        "estimated_touchdowns": [runs * 2, runs * 4],
        "estimated_uncompressed_event_bytes": runs * 3 * EVENT_SAMPLES * 10 * 4,
        "overwrite_policy": "refuse any non-empty output directory",
    }


def configure_walking_surface(
    model: mujoco.MjModel,
    terrain: str,
    family: str,
    surface_index: int,
) -> tuple[int, ExpandedSurfaceParameters]:
    surface = make_expanded_surface_parameters(terrain, family, surface_index)
    hfield_id = model.hfield(HFIELD_NAME).id
    nrow, ncol = int(model.hfield_nrow[hfield_id]), int(model.hfield_ncol[hfield_id])
    if (nrow - 1) % 80 or (ncol - 1) % 120:
        raise ValueError("walking hfield must tile the canonical 81x121 surface grid")
    canonical = normalized_family_surface(81, 121, surface)
    core = canonical[:-1, :-1]
    tiled = np.tile(core, ((nrow - 1) // 80, (ncol - 1) // 120))
    values = np.empty((nrow, ncol), dtype=np.float64)
    values[:-1, :-1] = tiled
    values[-1, :-1] = tiled[0]
    values[:, -1] = values[:, 0]
    model.hfield_size[hfield_id, 2] = surface.peak_to_valley_m
    surface_geom_id = model.geom(SURFACE_FLOOR_NAME).id
    model.geom_pos[surface_geom_id, 2] = -0.5 * surface.peak_to_valley_m
    address = int(model.hfield_adr[hfield_id])
    model.hfield_data[address : address + values.size] = (0.5 * (values + 1.0)).ravel()
    floor_id = apply_terrain_profile(
        model, TERRAIN_PROFILES[terrain], SURFACE_FLOOR_NAME
    )
    return floor_id, surface


def configure_contact_model(
    model: mujoco.MjModel,
    floor_id: int,
    contact_model: str,
) -> tuple[str, ...]:
    """Optionally isolate terrain support to the explicit sole spheres.

    The source G1 XML is left unchanged.  Disabling happens on the per-run model
    instance, and fall validity remains guarded by base height and orientation.
    """
    if contact_model == "full-body":
        return ()
    if contact_model != "foot-spheres-only":
        raise ValueError(f"unknown contact model {contact_model!r}")
    allowed = find_allowed_foot_geom_ids(model)
    disabled = []
    for geom_id in range(model.ngeom):
        if geom_id == floor_id or geom_id in allowed:
            continue
        if model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0:
            continue
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        disabled.append(geom_name or f"{body_name or f'body_{body_id}'}/geom_{geom_id}")
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0
    return tuple(disabled)


def _run_one(
    args: argparse.Namespace,
    policy_path: Path,
    terrain: str,
    family: str,
    surface_index: int,
    run_index: int,
) -> tuple[list[TouchdownEvent], dict[str, object]]:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    model.opt.timestep = PHYSICS_TIMESTEP_S
    data = mujoco.MjData(model)
    floor_id, surface = configure_walking_surface(
        model, terrain, family, surface_index
    )
    contact_model = getattr(args, "contact_model", "full-body")
    disabled_collision_geoms = configure_contact_model(model, floor_id, contact_model)
    spec = make_expanded_run_specification(terrain, family, surface_index, run_index)
    # Apply this run's deterministic friction (configure_walking_surface used r0
    # only to construct the first profile before this explicit replacement).
    contact_parameters = getattr(args, "contact_parameters", "terrain-native")
    run_profile = replace(TERRAIN_PROFILES[terrain], friction=spec.friction)
    if contact_parameters == "walking-support-v1" and terrain == "sand":
        run_profile = replace(
            run_profile,
            solref=TERRAIN_PROFILES["concrete"].solref,
        )
    elif contact_parameters in (
        "hard-solref-audit", "hard-solimp-audit", "hard-reference-audit"
    ):
        hard_reference = TERRAIN_PROFILES["concrete"]
        run_profile = replace(
            run_profile,
            solref=(
                hard_reference.solref
                if contact_parameters in ("hard-solref-audit", "hard-reference-audit")
                else run_profile.solref
            ),
            solimp=(
                hard_reference.solimp
                if contact_parameters in ("hard-solimp-audit", "hard-reference-audit")
                else run_profile.solimp
            ),
        )
    elif contact_parameters not in ("terrain-native", "walking-support-v1"):
        raise ValueError(f"unknown contact parameter mode {contact_parameters!r}")
    apply_terrain_profile(model, run_profile, SURFACE_FLOOR_NAME)
    mujoco.mj_forward(model, data)
    controller = UnitreeG1PretrainedController(model, data, policy_path, args.walking_speed)
    reader = G1HilSensorReader(model, data)
    collector = TouchdownEventCollector()
    allowed_feet = find_allowed_foot_geom_ids(model)
    surface_ids = frozenset((floor_id,))
    viewer = None
    if args.gui:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)

    total_steps = int(round(args.duration_s / PHYSICS_TIMESTEP_S))
    min_base_height = float("inf")
    min_upright_z = float("inf")
    nonfoot_collision = False
    failure_time_s: float | None = None
    failure_reason = ""
    last_stable_base_speed_mps = 0.0
    last_stable_left_foot_speed_mps = 0.0
    last_stable_left_contact = False
    failure_base_speed_mps = 0.0
    failure_left_foot_speed_mps = 0.0
    failure_left_contact = False
    failure_contact_geoms = ""
    failure_contact_position = np.full(3, np.nan, dtype=np.float64)
    failure_contact_distance_m: float | None = None
    failure_contact_normal_force_N: float | None = None
    pelvis_velocity = np.zeros(6, dtype=np.float64)
    foot_velocity = np.zeros(6, dtype=np.float64)
    pelvis_id = model.body("pelvis").id
    left_foot_body_id = model.body("left_ankle_roll_link").id
    start_xy = data.qpos[:2].copy()
    start = time.perf_counter()
    try:
        for physics_step in range(1, total_steps + 1):
            controller.apply()
            mujoco.mj_step(model, data)
            controller.update_after_step()
            sampled_contact: bool | None = None
            if physics_step % PHYSICS_STEPS_PER_SAMPLE == 0:
                sensor = reader.read_vector()
                sampled_contact = reader.has_left_foot_contact(surface_ids)
                collector.append(
                    data.time,
                    sensor,
                    sampled_contact,
                    enabled=data.time >= args.settling_s,
                )
            if data.time >= args.settling_s:
                min_base_height = min(min_base_height, float(data.qpos[2]))
                upright_z = float(data.xmat[pelvis_id, 8])
                min_upright_z = min(min_upright_z, upright_z)
                collision_now = has_nonfoot_floor_contact(
                    data, floor_id, allowed_feet
                )
                nonfoot_collision |= collision_now
                mujoco.mj_objectVelocity(
                    model,
                    data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    pelvis_id,
                    pelvis_velocity,
                    0,
                )
                mujoco.mj_objectVelocity(
                    model,
                    data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    left_foot_body_id,
                    foot_velocity,
                    0,
                )
                base_speed = float(np.linalg.norm(pelvis_velocity[3:5]))
                foot_speed = float(np.linalg.norm(foot_velocity[3:5]))
                left_contact = (
                    reader.has_left_foot_contact(surface_ids)
                    if sampled_contact is None
                    else sampled_contact
                )
                failure_reasons = []
                if data.qpos[2] < 0.55:
                    failure_reasons.append("fallen_base_height")
                if upright_z < 0.55:
                    failure_reasons.append("fallen_orientation")
                if collision_now:
                    failure_reasons.append("nonfoot_surface_contact")
                if not np.all(np.isfinite(data.qpos)) or not np.all(
                    np.isfinite(data.qvel)
                ):
                    failure_reasons.append("nan_or_inf")
                if failure_reasons and failure_time_s is None:
                    failure_time_s = float(data.time)
                    failure_reason = "|".join(failure_reasons)
                    failure_base_speed_mps = base_speed
                    failure_left_foot_speed_mps = foot_speed
                    failure_left_contact = left_contact
                    if collision_now:
                        offending_contacts = []
                        for contact_id in range(data.ncon):
                            contact = data.contact[contact_id]
                            if floor_id not in (contact.geom1, contact.geom2):
                                continue
                            other_geom = int(
                                contact.geom2
                                if contact.geom1 == floor_id
                                else contact.geom1
                            )
                            if other_geom in allowed_feet:
                                continue
                            geom_name = mujoco.mj_id2name(
                                model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
                            )
                            body_id = int(model.geom_bodyid[other_geom])
                            body_name = mujoco.mj_id2name(
                                model, mujoco.mjtObj.mjOBJ_BODY, body_id
                            )
                            offending_contacts.append(
                                (
                                    geom_name
                                    or f"{body_name or f'body_{body_id}'}/geom_{other_geom}",
                                    contact_id,
                                    contact,
                                )
                            )
                        failure_contact_geoms = "|".join(
                            dict.fromkeys(name for name, _, _ in offending_contacts)
                        )
                        if offending_contacts:
                            first_contact = offending_contacts[0][2]
                            failure_contact_position = np.asarray(
                                first_contact.pos, dtype=np.float64
                            ).copy()
                            failure_contact_distance_m = float(first_contact.dist)
                            contact_force = np.zeros(6, dtype=np.float64)
                            normal_forces = []
                            for _, contact_id, _ in offending_contacts:
                                mujoco.mj_contactForce(
                                    model, data, contact_id, contact_force
                                )
                                normal_forces.append(abs(float(contact_force[0])))
                            failure_contact_normal_force_N = max(normal_forces)
                    if args.stop_on_fall:
                        break
                elif not failure_reasons:
                    last_stable_base_speed_mps = base_speed
                    last_stable_left_foot_speed_mps = foot_speed
                    last_stable_left_contact = left_contact
            if viewer is not None and physics_step % 20 == 0:
                viewer.sync()
    finally:
        if viewer is not None:
            viewer.close()
    collector.finish()
    wall_time = time.perf_counter() - start
    executed_duration_s = float(data.time)
    displacement_xy = data.qpos[:2] - start_xy
    touchdown_times = np.asarray(
        [event.touchdown_time_s for event in collector.events], dtype=np.float64
    )
    intervals = np.diff(touchdown_times)
    event_sensor_stack = (
        np.asarray([event.sensors for event in collector.events], dtype=np.float64)
        if collector.events else np.empty((0, EVENT_SAMPLES, 10), dtype=np.float64)
    )
    finite = bool(
        np.all(np.isfinite(data.qpos))
        and np.all(np.isfinite(data.qvel))
        and all(np.all(np.isfinite(event.sensors)) for event in collector.events)
    )
    run_valid = finite and failure_time_s is None
    if not run_valid:
        reasons = []
        if not finite: reasons.append("nan_or_inf")
        if min_base_height < 0.55: reasons.append("fallen_base_height")
        if min_upright_z < 0.55: reasons.append("fallen_orientation")
        if nonfoot_collision: reasons.append("nonfoot_surface_contact")
        collector.events = [
            replace(event, valid=False, invalid_reason="|".join(filter(None, (event.invalid_reason, *reasons))))
            for event in collector.events
        ]
    run_row = {
        "terrain_name": terrain,
        "terrain_class": TERRAIN_LABELS[terrain],
        "surface_family": family,
        "surface_seed": surface.surface_seed,
        "surface_index": surface_index,
        "run_index": run_index,
        "split": family_for_name(family).split,
        "session_id": spec.session_id,
        "run_id": spec.run_id,
        "walking_speed_mps": args.walking_speed,
        "locomotion_condition": f"forward_{args.walking_speed:.3f}_mps",
        "contact_model": contact_model,
        "contact_parameters": contact_parameters,
        "effective_solref": "|".join(f"{value:.9g}" for value in run_profile.solref),
        "effective_solimp": "|".join(f"{value:.9g}" for value in run_profile.solimp),
        "disabled_collision_geom_count": len(disabled_collision_geoms),
        "duration_s": args.duration_s,
        "executed_duration_s": executed_duration_s,
        "stopped_early": int(executed_duration_s + PHYSICS_TIMESTEP_S < args.duration_s),
        "touchdown_events": len(collector.events),
        "valid_events": sum(event.valid for event in collector.events),
        "rejected_short_air": collector.rejected_short_air,
        "rejected_contact_chatter": collector.rejected_contact_chatter,
        "incomplete_at_end": collector.incomplete_at_end,
        "min_base_height_m": min_base_height,
        "min_upright_z": min_upright_z,
        "forward_displacement_m": float(displacement_xy[0]),
        "lateral_displacement_m": float(displacement_xy[1]),
        "touchdown_interval_mean_s": "" if intervals.size == 0 else float(intervals.mean()),
        "touchdown_interval_min_s": "" if intervals.size == 0 else float(intervals.min()),
        "touchdown_interval_max_s": "" if intervals.size == 0 else float(intervals.max()),
        "event_fsr_sum_peak_N": "" if not collector.events else float(event_sensor_stack[:, :, :4].sum(axis=2).max()),
        "event_abs_accel_peak_m_s2": "" if not collector.events else float(np.abs(event_sensor_stack[:, :, 4:7]).max()),
        "event_abs_gyro_peak_rad_s": "" if not collector.events else float(np.abs(event_sensor_stack[:, :, 7:10]).max()),
        "failure_time_s": "" if failure_time_s is None else failure_time_s,
        "failure_reason": failure_reason,
        "last_stable_base_speed_mps": last_stable_base_speed_mps,
        "last_stable_left_foot_speed_mps": last_stable_left_foot_speed_mps,
        "last_stable_left_contact": int(last_stable_left_contact),
        "failure_base_speed_mps": "" if failure_time_s is None else failure_base_speed_mps,
        "failure_left_foot_speed_mps": "" if failure_time_s is None else failure_left_foot_speed_mps,
        "failure_left_contact": "" if failure_time_s is None else int(failure_left_contact),
        "failure_contact_geoms": failure_contact_geoms,
        "failure_contact_x_m": "" if not failure_contact_geoms else float(failure_contact_position[0]),
        "failure_contact_y_m": "" if not failure_contact_geoms else float(failure_contact_position[1]),
        "failure_contact_z_m": "" if not failure_contact_geoms else float(failure_contact_position[2]),
        "failure_contact_distance_m": "" if failure_contact_distance_m is None else failure_contact_distance_m,
        "failure_contact_normal_force_N": "" if failure_contact_normal_force_N is None else failure_contact_normal_force_N,
        "terminal_base_x_m": float(data.qpos[0]),
        "terminal_base_y_m": float(data.qpos[1]),
        "terminal_base_z_m": float(data.qpos[2]),
        "terminal_quat_w": float(data.qpos[3]),
        "terminal_quat_x": float(data.qpos[4]),
        "terminal_quat_y": float(data.qpos[5]),
        "terminal_quat_z": float(data.qpos[6]),
        "nonfoot_surface_contact": int(nonfoot_collision),
        "finite": int(finite),
        "run_valid": int(run_valid),
        "wall_time_s": wall_time,
    }
    return collector.events, run_row


def _plot_first_event(output: Path, event: TouchdownEvent) -> None:
    import matplotlib.pyplot as plt
    relative = np.arange(-PRE_SAMPLES, POST_SAMPLES)
    figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(relative, event.fsr_sum)
    axes[0].axvline(0, color="black", linestyle="--")
    axes[0].set_ylabel("FSR sum [N]")
    axes[1].plot(relative, event.sensors[:, 4:7])
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set(xlabel="relative time [ms]", ylabel="accel [m/s^2]")
    axes[1].legend(("X", "Y", "Z"))
    figure.tight_layout()
    figure.savefig(output / "touchdown_diagnostic.png", dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    payload = protocol(args)
    if not args.execute:
        print(json.dumps(payload, indent=2))
        print("Dry run only. Pass --execute and --policy-path to collect events.")
        return
    if args.policy_path is None:
        raise ValueError("--policy-path is required with --execute")
    policy_path = args.policy_path.resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(policy_path)
    output = (args.output_dir or OUTPUT_DIR).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    policy_sha256 = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    if policy_sha256 != TESTED_POLICY_SHA256:
        raise ValueError(
            f"unverified Unitree G1 policy sha256 {policy_sha256}; "
            f"expected {TESTED_POLICY_SHA256} from {UPSTREAM_REVISION}"
        )
    payload["controller"]["policy_path"] = str(policy_path)
    payload["controller"]["policy_sha256"] = policy_sha256

    all_events: list[TouchdownEvent] = []
    manifest: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    for terrain in args.terrains:
        for family in args.families:
            for surface_index in range(args.surfaces_per_family):
                for run_index in range(args.runs_per_surface):
                    events, run_row = _run_one(
                        args, policy_path, terrain, family, surface_index, run_index
                    )
                    for event in events:
                        event_id = f"event_{len(all_events):06d}"
                        manifest.append(event_manifest_row(event, event_id, run_row))
                        all_events.append(event)
                    run_rows.append(run_row)
                    print(
                        f"run={run_row['run_id']} terrain={terrain} "
                        f"touchdowns={len(events)} valid={run_row['valid_events']} "
                        f"base_min={run_row['min_base_height_m']:.3f}m"
                    )
                    if events:
                        event = events[0]
                        before = float(np.mean(event.fsr_sum[:PRE_SAMPLES]))
                        after = float(np.mean(event.fsr_sum[PRE_SAMPLES:PRE_SAMPLES + 10]))
                        print(
                            f"  touchdown={event.touchdown_time_s:.3f}s pre={PRE_SAMPLES} "
                            f"post={POST_SAMPLES} fsr_before={before:.3f}N "
                            f"fsr_after={after:.3f}N shape={event.sensors.shape} "
                            f"valid={event.valid}"
                        )

    if not all_events:
        raise ValueError("no touchdown events were collected")
    validate_event_split_integrity(manifest)
    arrays = stack_events(all_events)
    if not np.all(np.isfinite(arrays["sensors"])):
        raise ValueError("NaN/Inf in walking event tensor")
    np.savez_compressed(
        output / "events.npz",
        **arrays,
        event_id=np.asarray([row["event_id"] for row in manifest]),
        terrain_class=np.asarray([row["terrain_class"] for row in manifest], dtype=np.int8),
        valid=np.asarray([row["valid"] for row in manifest], dtype=bool),
    )
    write_dict_rows(output / "manifest.csv", manifest)
    write_dict_rows(output / "runs.csv", run_rows)
    payload["measured_generation"] = {
        "runs": len(run_rows),
        "events": len(all_events),
        "valid_events": sum(event.valid for event in all_events),
        "wall_time_s": sum(float(row["wall_time_s"]) for row in run_rows),
    }
    (output / "protocol.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = [
        f"{SCHEMA_NAME} schema_version={SCHEMA_VERSION}",
        f"runs={len(run_rows)} events={len(all_events)} valid={sum(event.valid for event in all_events)}",
        f"sensors={arrays['sensors'].shape} contact={arrays['contact'].shape}",
        "native_sampling=1000Hz physics=2000Hz steps_per_sample=2 timestamp_spacing=1ms",
        "event_window=[-10,+50)ms classifier_candidate=[0,+50)ms",
        "split_unit=surface_family/surface_seed/session/run (no random event split)",
    ]
    (output / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    if args.plot and all_events:
        _plot_first_event(output, all_events[0])
    print("\n".join(summary))


if __name__ == "__main__":
    main()
