"""Build the project-owned G1 lower-body MJCF from the upstream 29-DOF model."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


SIMULATE_PYTHON_DIR = Path(__file__).resolve().parent
SIMULATION_DIR = SIMULATE_PYTHON_DIR.parents[1]
DEFAULT_SOURCE = SIMULATION_DIR / "unitree_mujoco" / "unitree_robots" / "g1" / "g1_29dof.xml"
DEFAULT_OUTPUT = SIMULATION_DIR / "models" / "g1_lower_body" / "g1_lower_body.xml"
UPPER_ROOT = "waist_yaw_link"
LEG_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
)
PRIMARY_SENSOR_NAMES = {
    "imu_quat", "imu_gyro", "imu_acc", "frame_pos", "frame_vel",
}


def subtree_body_ids(model: mujoco.MjModel, root_name: str) -> tuple[int, ...]:
    root_id = model.body(root_name).id
    descendants = []
    for body_id in range(1, model.nbody):
        cursor = body_id
        while cursor > 0:
            if cursor == root_id:
                descendants.append(body_id)
                break
            cursor = int(model.body_parentid[cursor])
    return tuple(descendants)


def composite_mass_properties(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_ids: tuple[int, ...],
) -> tuple[float, np.ndarray, np.ndarray]:
    masses = model.body_mass[list(body_ids)]
    mass = float(masses.sum())
    if mass <= 0.0:
        raise ValueError("composite body set has zero mass")
    positions = data.xipos[list(body_ids)]
    center = np.sum(masses[:, None] * positions, axis=0) / mass
    inertia = np.zeros((3, 3), dtype=np.float64)
    identity = np.eye(3)
    for body_id in body_ids:
        rotation = data.ximat[body_id].reshape(3, 3)
        body_inertia = rotation @ np.diag(model.body_inertia[body_id]) @ rotation.T
        offset = data.xipos[body_id] - center
        inertia += body_inertia + model.body_mass[body_id] * (
            np.dot(offset, offset) * identity - np.outer(offset, offset)
        )
    return mass, center, inertia


def fmt(values: np.ndarray | tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def full_inertia_values(matrix: np.ndarray) -> tuple[float, ...]:
    return (
        float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[2, 2]),
        float(matrix[0, 1]), float(matrix[0, 2]), float(matrix[1, 2]),
    )


def find_body(root: ET.Element, name: str) -> ET.Element:
    body = root.find(f".//body[@name='{name}']")
    if body is None:
        raise ValueError(f"body not found in source XML: {name}")
    return body


def retained_mesh_names(pelvis: ET.Element) -> set[str]:
    return {
        geom.attrib["mesh"]
        for geom in pelvis.iter("geom")
        if "mesh" in geom.attrib
    }


def build(source_path: Path, output_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(source_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    removed_ids = subtree_body_ids(model, UPPER_ROOT)
    removed_mass, removed_com_world, removed_inertia_world = composite_mass_properties(
        model, data, removed_ids
    )
    pelvis_id = model.body("pelvis").id
    pelvis_rotation = data.xmat[pelvis_id].reshape(3, 3)
    removed_com_pelvis = pelvis_rotation.T @ (
        removed_com_world - data.xpos[pelvis_id]
    )
    removed_inertia_pelvis = pelvis_rotation.T @ removed_inertia_world @ pelvis_rotation
    torso_id = model.body("torso_link").id
    torso_com_pelvis = pelvis_rotation.T @ (
        data.xipos[torso_id] - data.xpos[pelvis_id]
    )
    pulse_point_in_equivalent = torso_com_pelvis - removed_com_pelvis

    tree = ET.parse(source_path)
    root = tree.getroot()
    root.set("model", "g1_lower_body_12dof")
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("source XML has no compiler element")
    compiler.set("meshdir", "../../unitree_mujoco/unitree_robots/g1/meshes")

    pelvis = find_body(root, "pelvis")
    upper_body = next(
        (child for child in pelvis.findall("body") if child.get("name") == UPPER_ROOT),
        None,
    )
    if upper_body is None:
        raise ValueError(f"{UPPER_ROOT} is not a direct pelvis child")
    pelvis.remove(upper_body)

    equivalent = ET.Element(
        "body", {"name": "equivalent_upper_body", "pos": fmt(removed_com_pelvis)}
    )
    ET.SubElement(
        equivalent,
        "inertial",
        {
            "pos": "0 0 0",
            "mass": f"{removed_mass:.12g}",
            "fullinertia": fmt(full_inertia_values(removed_inertia_pelvis)),
        },
    )
    ET.SubElement(
        equivalent,
        "geom",
        {
            "name": "equivalent_upper_body_marker",
            "type": "sphere",
            "size": "0.025",
            "contype": "0",
            "conaffinity": "0",
            "group": "3",
            "density": "0",
            "rgba": "0.9 0.3 0.1 0.5",
        },
    )
    ET.SubElement(equivalent, "site", {"name": "support_point", "pos": "0 0 0", "size": "0.012"})
    ET.SubElement(
        equivalent,
        "site",
        {"name": "pulse_point", "pos": fmt(pulse_point_in_equivalent), "size": "0.012"},
    )
    pelvis.append(equivalent)

    retained_meshes = retained_mesh_names(pelvis)
    asset = root.find("asset")
    if asset is None:
        raise ValueError("source XML has no asset element")
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name") not in retained_meshes:
            asset.remove(mesh)

    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("source XML has no actuator element")
    for element in list(actuator):
        if element.get("joint") not in LEG_JOINTS:
            actuator.remove(element)

    sensor = root.find("sensor")
    if sensor is None:
        raise ValueError("source XML has no sensor element")
    for element in list(sensor):
        joint_name = element.get("joint")
        sensor_name = element.get("name")
        if joint_name in LEG_JOINTS or sensor_name in PRIMARY_SENSOR_NAMES:
            continue
        sensor.remove(element)

    custom = ET.Element("custom")
    ET.SubElement(custom, "numeric", {"name": "removed_upper_body_mass", "data": f"{removed_mass:.12g}"})
    ET.SubElement(custom, "numeric", {"name": "removed_upper_body_com_in_pelvis", "data": fmt(removed_com_pelvis)})
    ET.SubElement(custom, "numeric", {"name": "removed_upper_body_inertia_in_pelvis", "data": fmt(full_inertia_values(removed_inertia_pelvis))})
    ET.SubElement(custom, "numeric", {"name": "original_torso_com_in_pelvis", "data": fmt(torso_com_pelvis)})
    root.append(custom)

    ET.indent(tree, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="unicode", xml_declaration=False)
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write("\n")

    reduced = mujoco.MjModel.from_xml_path(str(output_path))
    if reduced.nu != len(LEG_JOINTS):
        raise ValueError(f"expected {len(LEG_JOINTS)} actuators, got {reduced.nu}")
    print(f"source={source_path}")
    print(f"output={output_path}")
    print(f"removed_bodies={len(removed_ids)}")
    print(f"removed_mass={removed_mass:.12f}")
    print(f"removed_com_pelvis={fmt(removed_com_pelvis)}")
    print(f"removed_inertia_pelvis={fmt(full_inertia_values(removed_inertia_pelvis))}")
    print(f"original_torso_com_pelvis={fmt(torso_com_pelvis)}")
    print(f"lower_body_actuators={reduced.nu}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
