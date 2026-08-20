"""Versioned bilateral virtual-sensor contract for the G1 walking twin.

The legacy :class:`hil_sensor.G1HilSensorReader` remains the source of the
left-foot v1 API.  This additive reader exposes the mirrored right foot,
pelvis IMU, per-foot contact state, and a fixed runtime-reproducible frame
canonicalization for a weight-shared foot encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from hil_sensor import G1HilSensorReader, HIL_SENSOR_CHANNELS


SIDES = ("left", "right")
FOOT_VECTOR_SIZE = len(HIL_SENSOR_CHANNELS)
BILATERAL_VECTOR_SIZE = 2 * FOOT_VECTOR_SIZE
PELVIS_CHANNELS = (
    "pelvis_accel_x", "pelvis_accel_y", "pelvis_accel_z",
    "pelvis_gyro_x", "pelvis_gyro_y", "pelvis_gyro_z",
)
RIGHT_FOOT_CONTACT_GEOM_NAMES = tuple(
    f"right_foot_contact_{index}" for index in range(1, 5)
)
FOOT_CONTACT_GEOM_NAMES = {
    "left": tuple(f"left_foot_contact_{index}" for index in range(1, 5)),
    "right": RIGHT_FOOT_CONTACT_GEOM_NAMES,
}

# Reflection in the sagittal plane.  Acceleration is a polar vector and gyro
# is an axial vector, so gyro transforms with det(R)R.  Contact slots exchange
# their medial/lateral positions.  These constants are the predeclared v2
# transform and do not depend on acquired labels or validation results.
RIGHT_FSR_CANONICAL_ORDER = np.asarray((1, 0, 3, 2), dtype=np.int64)
RIGHT_ACCEL_CANONICAL_SIGN = np.asarray((1.0, -1.0, 1.0))
RIGHT_GYRO_CANONICAL_SIGN = np.asarray((-1.0, 1.0, -1.0))


@dataclass(frozen=True)
class FootContactState:
    """Causal force-derived per-foot state available to a runtime detector."""

    loaded: bool
    total_force_n: float
    age_samples: int
    transition: str


class G1BilateralSensorReaderV2:
    """Read raw and canonical bilateral Fusion10 vectors at a MuJoCo endpoint."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        load_on_n: float = 5.0,
        load_off_n: float = 2.5,
    ) -> None:
        if not 0.0 <= load_off_n <= load_on_n:
            raise ValueError("contact thresholds require 0 <= off <= on")
        self.model = model
        self.data = data
        self._legacy = G1HilSensorReader(model, data)
        self.load_on_n = float(load_on_n)
        self.load_off_n = float(load_off_n)
        self._sensor_ids = {
            side: (
                model.sensor(f"{side}_foot_imu_acc").id,
                model.sensor(f"{side}_foot_imu_gyro").id,
            )
            for side in SIDES
        }
        self._pelvis_sensor_ids = (
            model.sensor("imu_acc").id,
            model.sensor("imu_gyro").id,
        )
        self.foot_geom_ids: dict[str, tuple[int, ...]] = {}
        self._slot_by_geom_id: dict[str, dict[int, int]] = {}
        for side in SIDES:
            ankle_id = model.body(f"{side}_ankle_roll_link").id
            site_id = model.site(f"{side}_foot_imu").id
            if int(model.site_bodyid[site_id]) != ankle_id:
                raise ValueError(f"{side}_foot_imu is not rigidly attached to ankle roll")
            ids = tuple(model.geom(name).id for name in FOOT_CONTACT_GEOM_NAMES[side])
            for name, geom_id in zip(FOOT_CONTACT_GEOM_NAMES[side], ids):
                if int(model.geom_bodyid[geom_id]) != ankle_id:
                    raise ValueError(f"{name} is not attached to {side} ankle roll")
                if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_SPHERE):
                    raise ValueError(f"{name} is not a sphere collision geom")
            self.foot_geom_ids[side] = ids
            self._slot_by_geom_id[side] = {
                geom_id: slot for slot, geom_id in enumerate(ids)
            }
        self.left_foot_geom_ids = self.foot_geom_ids["left"]
        self.right_foot_geom_ids = self.foot_geom_ids["right"]
        self._wrench = np.zeros(6, dtype=np.float64)
        self._loaded = {side: False for side in SIDES}
        self._contact_age = {side: 0 for side in SIDES}

    def _check_side(self, side: str) -> None:
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")

    def _sensor_data(self, sensor_id: int) -> np.ndarray:
        address = int(self.model.sensor_adr[sensor_id])
        dimension = int(self.model.sensor_dim[sensor_id])
        return self.data.sensordata[address:address + dimension].copy()

    def read_foot_normal_forces(self, side: str) -> np.ndarray:
        """Sum normal contact force into the four physical sole slots."""
        self._check_side(side)
        forces = np.zeros(4, dtype=np.float64)
        slots = self._slot_by_geom_id[side]
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            contacted = {
                slots[geom_id]
                for geom_id in (int(contact.geom1), int(contact.geom2))
                if geom_id in slots
            }
            if not contacted:
                continue
            self._wrench.fill(0.0)
            mujoco.mj_contactForce(self.model, self.data, contact_id, self._wrench)
            force = max(0.0, float(self._wrench[0]))
            for slot in contacted:
                forces[slot] += force
        return forces

    def read_foot_vector(self, side: str) -> np.ndarray:
        """Return raw physical-frame [FSR4, accel XYZ, gyro XYZ]."""
        self._check_side(side)
        accelerometer_id, gyroscope_id = self._sensor_ids[side]
        return np.concatenate((
            self.read_foot_normal_forces(side),
            self._sensor_data(accelerometer_id),
            self._sensor_data(gyroscope_id),
        ))

    def read_left_foot_vector(self) -> np.ndarray:
        return self.read_foot_vector("left")

    def read_right_foot_vector(self) -> np.ndarray:
        return self.read_foot_vector("right")

    def read_vector(self) -> np.ndarray:
        """Preserve the legacy left-foot behavior exactly."""
        return self._legacy.read_vector()

    def read_bilateral_vector(self) -> np.ndarray:
        """Return raw [left Fusion10, right Fusion10]."""
        return np.concatenate((self.read_left_foot_vector(), self.read_right_foot_vector()))

    def read_pelvis_vector(self) -> np.ndarray:
        """Return pelvis [accel XYZ, gyro XYZ] from the existing virtual IMU."""
        return np.concatenate(tuple(self._sensor_data(value) for value in self._pelvis_sensor_ids))

    @staticmethod
    def canonicalize_foot_vector(side: str, vector: np.ndarray) -> np.ndarray:
        """Map a raw foot vector to the common left-foot model frame."""
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (FOOT_VECTOR_SIZE,):
            raise ValueError(f"expected Fusion10 shape (10,), got {values.shape}")
        if side == "left":
            return values.copy()
        return np.concatenate((
            values[:4][RIGHT_FSR_CANONICAL_ORDER],
            values[4:7] * RIGHT_ACCEL_CANONICAL_SIGN,
            values[7:10] * RIGHT_GYRO_CANONICAL_SIGN,
        ))

    def read_canonical_foot_vector(self, side: str) -> np.ndarray:
        return self.canonicalize_foot_vector(side, self.read_foot_vector(side))

    def read_canonical_bilateral_vector(self) -> np.ndarray:
        return np.concatenate(tuple(self.read_canonical_foot_vector(side) for side in SIDES))

    def has_foot_contact(
        self, side: str, allowed_other_geom_ids: frozenset[int] | None = None
    ) -> bool:
        """Offline MuJoCo contact source; never used as a detector feature."""
        self._check_side(side)
        ids = self._slot_by_geom_id[side]
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in ids:
                other = geom2
            elif geom2 in ids:
                other = geom1
            else:
                continue
            if allowed_other_geom_ids is None or other in allowed_other_geom_ids:
                return True
        return False

    def update_contact_state(self, side: str, forces: np.ndarray | None = None) -> FootContactState:
        """Advance one endpoint of independent hysteretic contact state."""
        self._check_side(side)
        values = self.read_foot_normal_forces(side) if forces is None else np.asarray(forces)
        if values.shape != (4,):
            raise ValueError("per-foot forces must have shape (4,)")
        total = float(np.sum(values))
        before = self._loaded[side]
        after = total >= (self.load_off_n if before else self.load_on_n)
        if after:
            self._contact_age[side] = self._contact_age[side] + 1 if before else 1
        else:
            self._contact_age[side] = 0
        self._loaded[side] = after
        transition = "steady"
        if after and not before:
            transition = "touchdown"
        elif before and not after:
            transition = "liftoff"
        return FootContactState(after, total, self._contact_age[side], transition)

    def reset_contact_state(self, side: str | None = None) -> None:
        """Reset one foot or both feet without consulting simulator ground truth."""
        selected = SIDES if side is None else (side,)
        for value in selected:
            self._check_side(value)
            self._loaded[value] = False
            self._contact_age[value] = 0
