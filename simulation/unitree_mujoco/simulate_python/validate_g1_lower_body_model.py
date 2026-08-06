"""Audit and compare reference-pose mass properties of full and reduced G1 models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from build_g1_lower_body_model import composite_mass_properties, subtree_body_ids
from controlled_excitation import (
    ExcitationCondition,
    VerticalElasticBandSupport,
    apply_excitation_condition,
    find_allowed_foot_geom_ids,
    has_nonfoot_floor_contact,
)
from hil_sensor import G1HilSensorReader, HIL_SENSOR_CHANNELS, LEFT_FOOT_CONTACT_GEOM_NAMES
from run_horizontal_pulse_dataset import SIMULATION_DIR
from terrain_profiles import TERRAIN_PROFILES, apply_terrain_profile


FULL_SCENE = SIMULATION_DIR / "unitree_mujoco" / "unitree_robots" / "g1" / "scene.xml"
LOWER_SCENE = SIMULATION_DIR / "models" / "g1_lower_body" / "scene.xml"
DEFAULT_OUTPUT = SIMULATION_DIR / "outputs" / "g1_lower_body_validation" / "model_validation.json"
LEFT_LEG_ROOT = "left_hip_pitch_link"
RIGHT_LEG_ROOT = "right_hip_pitch_link"
UPPER_ROOT = "waist_yaw_link"


def name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    result = mujoco.mj_id2name(model, object_type, object_id)
    return "" if result is None else result


def body_names(model: mujoco.MjModel, body_ids: tuple[int, ...]) -> list[str]:
    return [name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) for body_id in body_ids]


def inertia_record(matrix: np.ndarray) -> dict[str, object]:
    return {
        "matrix": matrix.tolist(),
        "principal_values": np.linalg.eigvalsh(matrix).tolist(),
        "trace": float(np.trace(matrix)),
        "frobenius_norm": float(np.linalg.norm(matrix)),
    }


def model_record(path: Path) -> tuple[mujoco.MjModel, mujoco.MjData, dict[str, object]]:
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    all_ids = tuple(range(1, model.nbody))
    mass, center, inertia = composite_mass_properties(model, data, all_ids)
    pelvis_id = model.body("pelvis").id
    imu_site_id = model.site("imu").id
    sensor_records = {}
    for sensor_name in ("imu_acc", "imu_gyro"):
        sensor_id = model.sensor(sensor_name).id
        sensor_records[sensor_name] = {
            "sensor_id": int(sensor_id),
            "object_type": int(model.sensor_objtype[sensor_id]),
            "site": name(model, mujoco.mjtObj.mjOBJ_SITE, int(model.sensor_objid[sensor_id])),
            "body": name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.site_bodyid[imu_site_id])),
        }
    joint_positions = {}
    for side in ("left", "right"):
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"):
            joint_name = f"{side}_{joint}_joint"
            joint_id = model.joint(joint_name).id
            joint_positions[joint_name] = float(model.qpos0[model.jnt_qposadr[joint_id]])
    foot_positions = {
        geom_name: data.geom_xpos[model.geom(geom_name).id].tolist()
        for geom_name in LEFT_FOOT_CONTACT_GEOM_NAMES
    }
    record = {
        "path": str(path),
        "body_count_excluding_world": model.nbody - 1,
        "joint_count": model.njnt,
        "generalized_velocity_dof": model.nv,
        "actuated_dof": model.nu,
        "total_mass": mass,
        "com_world": center.tolist(),
        "inertia_about_com_world": inertia_record(inertia),
        "pelvis_mass": float(model.body_mass[pelvis_id]),
        "pelvis_position_world": data.xpos[pelvis_id].tolist(),
        "pelvis_quaternion_qpos0": model.qpos0[3:7].tolist(),
        "leg_joint_qpos0": joint_positions,
        "left_foot_contact_positions_world": foot_positions,
        "imu_site": {
            "name": "imu",
            "body": name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.site_bodyid[imu_site_id])),
            "local_position": model.site_pos[imu_site_id].tolist(),
            "world_position": data.site_xpos[imu_site_id].tolist(),
        },
        "imu_sensors": sensor_records,
        "hil_channel_count": len(HIL_SENSOR_CHANNELS),
    }
    return model, data, record


def foot_sphere_ids(model: mujoco.MjModel, body_name: str) -> tuple[int, ...]:
    body_id = model.body(body_name).id
    return tuple(
        geom_id for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] == body_id
        and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
    )


def contact_normal_sum(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    floor_id: int,
    geom_ids: tuple[int, ...],
) -> float:
    allowed = set(geom_ids)
    wrench = np.zeros(6, dtype=np.float64)
    total = 0.0
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        pair = {int(contact.geom1), int(contact.geom2)}
        if floor_id not in pair or not pair.intersection(allowed):
            continue
        wrench.fill(0.0)
        mujoco.mj_contactForce(model, data, contact_id, wrench)
        total += max(0.0, float(wrench[0]))
    return total


def static_contact_validation(path: Path) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_path(str(path))
    model.opt.timestep = 0.005
    floor_id = apply_terrain_profile(model, TERRAIN_PROFILES["concrete"])
    data = mujoco.MjData(model)
    condition = ExcitationCondition("nominal", 0.0, 0.0, 0.0, 0.0, 0.0)
    qpos_address, dof_address = apply_excitation_condition(model, data, condition)
    support = VerticalElasticBandSupport(
        model, data, qpos_address, dof_address, 0.70,
        application_body_name="equivalent_upper_body",
        application_site_name="support_point",
    )
    reader = G1HilSensorReader(model, data)
    allowed = find_allowed_foot_geom_ids(model)
    right_ids = foot_sphere_ids(model, "right_ankle_roll_link")
    samples = []
    right_loads = []
    collision = False
    next_sample = 0.02
    while data.time + 1e-12 < 0.20:
        support.apply()
        mujoco.mj_step(model, data)
        collision |= has_nonfoot_floor_contact(data, floor_id, allowed)
        if data.time + 1e-12 >= next_sample:
            samples.append(reader.read_vector())
            right_loads.append(contact_normal_sum(model, data, floor_id, right_ids))
            next_sample += 0.02
    values = np.asarray(samples)
    right = np.asarray(right_loads)
    if values.shape != (10, 10) or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid static 10-channel sequence: {values.shape}")
    return {
        "duration": 0.20,
        "sample_rate": 50.0,
        "shape": list(values.shape),
        "support_ratio": 0.70,
        "support_target_force": support.target_support_force,
        "first_10_channel_vector": values[0].tolist(),
        "last_10_channel_vector": values[-1].tolist(),
        "left_foot_normal_load_first": float(values[0, :4].sum()),
        "right_foot_normal_load_first": float(right[0]),
        "left_foot_normal_load_mean": float(values[:, :4].sum(axis=1).mean()),
        "right_foot_normal_load_mean": float(right.mean()),
        "nonfoot_body_collision": int(collision),
        "finite": bool(np.all(np.isfinite(values))),
        "right_foot_sphere_geom_ids": list(right_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-scene", type=Path, default=FULL_SCENE)
    parser.add_argument("--lower-scene", type=Path, default=LOWER_SCENE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    full_model, full_data, full = model_record(args.full_scene.resolve())
    lower_model, lower_data, lower = model_record(args.lower_scene.resolve())
    left_ids = subtree_body_ids(full_model, LEFT_LEG_ROOT)
    right_ids = subtree_body_ids(full_model, RIGHT_LEG_ROOT)
    upper_ids = subtree_body_ids(full_model, UPPER_ROOT)
    upper_mass, upper_com, upper_inertia = composite_mass_properties(
        full_model, full_data, upper_ids
    )
    pelvis_rotation = full_data.xmat[full_model.body("pelvis").id].reshape(3, 3)
    upper_com_pelvis = pelvis_rotation.T @ (
        upper_com - full_data.xpos[full_model.body("pelvis").id]
    )
    upper_inertia_pelvis = pelvis_rotation.T @ upper_inertia @ pelvis_rotation
    torso_id = full_model.body("torso_link").id
    full_pelvis_position = full_data.xpos[full_model.body("pelvis").id]
    lower_support_site = lower_model.site("support_point").id
    lower_pulse_site = lower_model.site("pulse_point").id
    secondary_imu_site = full_model.site("secondary_imu").id
    full_inertia = np.asarray(full["inertia_about_com_world"]["matrix"])
    lower_inertia = np.asarray(lower["inertia_about_com_world"]["matrix"])
    mass_difference = float(lower["total_mass"]) - float(full["total_mass"])
    com_difference = np.asarray(lower["com_world"]) - np.asarray(full["com_world"])
    inertia_difference = lower_inertia - full_inertia
    foot_geometry_comparison = {}
    max_foot_geometry_difference = 0.0
    for geom_name in LEFT_FOOT_CONTACT_GEOM_NAMES:
        full_geom_id = full_model.geom(geom_name).id
        lower_geom_id = lower_model.geom(geom_name).id
        position_difference = (
            lower_model.geom_pos[lower_geom_id] - full_model.geom_pos[full_geom_id]
        )
        size_difference = (
            lower_model.geom_size[lower_geom_id] - full_model.geom_size[full_geom_id]
        )
        max_foot_geometry_difference = max(
            max_foot_geometry_difference,
            float(np.max(np.abs(position_difference))),
            float(np.max(np.abs(size_difference))),
        )
        foot_geometry_comparison[geom_name] = {
            "full_local_position": full_model.geom_pos[full_geom_id].tolist(),
            "lower_local_position": lower_model.geom_pos[lower_geom_id].tolist(),
            "full_size": full_model.geom_size[full_geom_id].tolist(),
            "lower_size": lower_model.geom_size[lower_geom_id].tolist(),
            "position_difference": position_difference.tolist(),
            "size_difference": size_difference.tolist(),
            "same_geom_type": bool(
                full_model.geom_type[full_geom_id] == lower_model.geom_type[lower_geom_id]
            ),
        }

    audit = {
        "full_body": full,
        "lower_body": lower,
        "full_body_groups": {
            "left_leg": {
                "bodies": body_names(full_model, left_ids),
                "mass": float(full_model.body_mass[list(left_ids)].sum()),
            },
            "right_leg": {
                "bodies": body_names(full_model, right_ids),
                "mass": float(full_model.body_mass[list(right_ids)].sum()),
            },
            "removed_upper_body": {
                "bodies": body_names(full_model, upper_ids),
                "mass": upper_mass,
                "com_world": upper_com.tolist(),
                "com_relative_to_pelvis": upper_com_pelvis.tolist(),
                "inertia_about_removed_com_in_pelvis": inertia_record(upper_inertia_pelvis),
            },
        },
        "application_points": {
            "full_body_support_body": "torso_link",
            "full_body_pulse_body": "torso_link",
            "full_body_force_point_world_body_com": full_data.xipos[torso_id].tolist(),
            "full_body_force_point_relative_to_pelvis": (
                pelvis_rotation.T @ (full_data.xipos[torso_id] - full_pelvis_position)
            ).tolist(),
            "lower_body_support_body": "equivalent_upper_body",
            "lower_body_support_site": "support_point",
            "lower_body_support_point_world": lower_data.site_xpos[lower_support_site].tolist(),
            "lower_body_pulse_body": "equivalent_upper_body",
            "lower_body_pulse_site": "pulse_point",
            "lower_body_pulse_point_world": lower_data.site_xpos[lower_pulse_site].tolist(),
        },
        "full_body_secondary_imu": {
            "site": "secondary_imu",
            "body": name(
                full_model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(full_model.site_bodyid[secondary_imu_site]),
            ),
            "local_position": full_model.site_pos[secondary_imu_site].tolist(),
        },
        "difference_lower_minus_full": {
            "mass_absolute": mass_difference,
            "mass_relative": mass_difference / float(full["total_mass"]),
            "com_xyz": com_difference.tolist(),
            "com_norm": float(np.linalg.norm(com_difference)),
            "inertia_matrix": inertia_difference.tolist(),
            "inertia_frobenius_norm": float(np.linalg.norm(inertia_difference)),
            "inertia_relative_frobenius": float(np.linalg.norm(inertia_difference) / np.linalg.norm(full_inertia)),
        },
        "left_foot_contact_geometry_comparison": foot_geometry_comparison,
        "left_foot_contact_max_absolute_difference": max_foot_geometry_difference,
        "lower_body_static_concrete_validation": static_contact_validation(
            args.lower_scene.resolve()
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    print(f"Full mass={float(full['total_mass']):.12f} kg")
    print(f"Lower mass={float(lower['total_mass']):.12f} kg")
    print(f"Mass difference={mass_difference:.12e} kg")
    print(f"Full COM={np.asarray(full['com_world'])}")
    print(f"Lower COM={np.asarray(lower['com_world'])}")
    print(f"COM difference={com_difference} m")
    print(f"Inertia Frobenius difference={np.linalg.norm(inertia_difference):.12e} kg*m^2")
    print(f"Removed upper mass={upper_mass:.12f} kg")
    print(f"Removed upper COM relative pelvis={upper_com_pelvis}")
    print(f"Full IMU body={full['imu_site']['body']}; lower IMU body={lower['imu_site']['body']}")
    print(f"Lower actuated DOF={lower_model.nu}")
    static = audit["lower_body_static_concrete_validation"]
    print(
        f"Static sensor shape={tuple(static['shape'])}; "
        f"left/right first load={static['left_foot_normal_load_first']:.6f}/"
        f"{static['right_foot_normal_load_first']:.6f} N; "
        f"nonfoot collision={static['nonfoot_body_collision']}"
    )
    print(f"output={output}")

    if lower_model.nu != 12:
        raise ValueError("lower-body model does not have 12 actuators")
    if abs(mass_difference) > 1e-10 or np.linalg.norm(com_difference) > 1e-10:
        raise ValueError("mass/COM preservation check failed")
    if np.linalg.norm(inertia_difference) > 1e-9:
        raise ValueError("reference-pose inertia preservation check failed")
    if max_foot_geometry_difference != 0.0:
        raise ValueError("named left-foot contact geometry changed")


if __name__ == "__main__":
    main()
