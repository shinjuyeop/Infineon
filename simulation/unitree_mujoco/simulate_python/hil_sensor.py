"""G1 virtual sensors used by the Infineon HIL prototype."""

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
    """Read the 10-channel left-foot HIL vector from a G1 MuJoCo state."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self._accelerometer_sensor_id = model.sensor("left_foot_imu_acc").id
        self._gyroscope_sensor_id = model.sensor("left_foot_imu_gyro").id
        self._pelvis_accelerometer_sensor_id = model.sensor("imu_acc").id
        self._pelvis_gyroscope_sensor_id = model.sensor("imu_gyro").id

        ankle_body_id = model.body("left_ankle_roll_link").id
        left_foot_imu_site_id = model.site("left_foot_imu").id
        if model.site_bodyid[left_foot_imu_site_id] != ankle_body_id:
            raise ValueError(
                "left_foot_imu is not attached to the G1 left_ankle_roll_link"
            )
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

    def has_left_foot_contact(
        self, allowed_other_geom_ids: frozenset[int] | None = None
    ) -> bool:
        """Return MuJoCo contact ground truth for a named left sole sphere.

        This deliberately does not use the force values or an FSR threshold.
        When ``allowed_other_geom_ids`` is supplied, self-collisions and contacts
        with unrelated geometry are excluded from the terrain-contact state.
        """
        left_ids = self._foot_slot_by_geom_id
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in left_ids:
                other = geom2
            elif geom2 in left_ids:
                other = geom1
            else:
                continue
            if allowed_other_geom_ids is None or other in allowed_other_geom_ids:
                return True
        return False

    def read_vector(self) -> np.ndarray:
        """Return [4 foot forces, left-foot accel XYZ, left-foot gyro XYZ]."""
        foot_forces = self.read_left_foot_normal_forces()
        accelerometer = self._sensor_data(self._accelerometer_sensor_id)
        gyroscope = self._sensor_data(self._gyroscope_sensor_id)
        return np.concatenate((foot_forces, accelerometer, gyroscope))

    def read_pelvis_diagnostic_vector(self) -> np.ndarray:
        """Return the legacy pelvis-based vector for matched diagnostics only."""
        foot_forces = self.read_left_foot_normal_forces()
        accelerometer = self._sensor_data(self._pelvis_accelerometer_sensor_id)
        gyroscope = self._sensor_data(self._pelvis_gyroscope_sensor_id)
        return np.concatenate((foot_forces, accelerometer, gyroscope))


def format_hil_vector(vector: np.ndarray) -> str:
    if vector.shape != (len(HIL_SENSOR_CHANNELS),):
        raise ValueError(f"expected 10 HIL channels, got shape {vector.shape}")
    values = ", ".join(f"{value: .6f}" for value in vector)
    return f"[{values}]"
