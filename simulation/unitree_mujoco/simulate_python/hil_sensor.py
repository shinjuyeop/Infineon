"""G1 virtual sensors used by the DeepET HIL prototype."""

from __future__ import annotations

import mujoco
import numpy as np


LEFT_FOOT_CONTACT_GEOM_NAMES = (
    "left_foot_contact_1",
    "left_foot_contact_2",
    "left_foot_contact_3",
    "left_foot_contact_4",
)

HIL_SENSOR_CHANNELS = (
    "foot_force_1",
    "foot_force_2",
    "foot_force_3",
    "foot_force_4",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)


class G1HilSensorReader:
    """Read the first 10-channel HIL vector from a G1 MuJoCo state."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self._accelerometer_sensor_id = model.sensor("imu_acc").id
        self._gyroscope_sensor_id = model.sensor("imu_gyro").id

        ankle_body_id = model.body("left_ankle_roll_link").id
        self.left_foot_geom_ids = tuple(
            model.geom(name).id for name in LEFT_FOOT_CONTACT_GEOM_NAMES
        )
        for name, geom_id in zip(
            LEFT_FOOT_CONTACT_GEOM_NAMES, self.left_foot_geom_ids
        ):
            if model.geom_bodyid[geom_id] != ankle_body_id:
                raise ValueError(
                    f"{name} is not attached to the G1 left_ankle_roll_link"
                )
            if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_SPHERE:
                raise ValueError(f"{name} is not a sphere collision geom")

        self._foot_slot_by_geom_id = {
            geom_id: slot for slot, geom_id in enumerate(self.left_foot_geom_ids)
        }
        self._contact_wrench = np.zeros(6, dtype=np.float64)

    def _sensor_data(self, sensor_id: int) -> np.ndarray:
        address = self.model.sensor_adr[sensor_id]
        dimension = self.model.sensor_dim[sensor_id]
        return self.data.sensordata[address : address + dimension].copy()

    def read_left_foot_normal_forces(self) -> np.ndarray:
        """Sum normal-force magnitudes for each named left-foot contact sphere."""
        forces = np.zeros(4, dtype=np.float64)

        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            contacted_slots = {
                self._foot_slot_by_geom_id[geom_id]
                for geom_id in (int(contact.geom1), int(contact.geom2))
                if geom_id in self._foot_slot_by_geom_id
            }
            if not contacted_slots:
                continue

            self._contact_wrench.fill(0.0)
            mujoco.mj_contactForce(
                self.model, self.data, contact_id, self._contact_wrench
            )
            normal_force = max(0.0, float(self._contact_wrench[0]))
            for slot in contacted_slots:
                forces[slot] += normal_force

        return forces

    def read_vector(self) -> np.ndarray:
        """Return [4 foot forces, accelerometer XYZ, gyroscope XYZ]."""
        foot_forces = self.read_left_foot_normal_forces()
        accelerometer = self._sensor_data(self._accelerometer_sensor_id)
        gyroscope = self._sensor_data(self._gyroscope_sensor_id)
        return np.concatenate((foot_forces, accelerometer, gyroscope))


def format_hil_vector(vector: np.ndarray) -> str:
    if vector.shape != (len(HIL_SENSOR_CHANNELS),):
        raise ValueError(f"expected 10 HIL channels, got shape {vector.shape}")
    values = ", ".join(f"{value: .6f}" for value in vector)
    return f"[{values}]"
