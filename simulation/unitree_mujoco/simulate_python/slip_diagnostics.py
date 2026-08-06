"""MuJoCo-only slip diagnostics kept outside the 10-channel HIL input."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from hil_sensor import LEFT_FOOT_CONTACT_GEOM_NAMES


DIAGNOSTIC_CHANNELS = (
    "pelvis_velocity_x",
    "pelvis_velocity_y",
    "left_foot_velocity_x",
    "left_foot_velocity_y",
    "foot_slip_displacement",
    "contact_tangential_force",
    "contact_normal_force",
    "contact_force_ratio",
    "force_pulse_active",
)


@dataclass(frozen=True)
class SlipDiagnosticSample:
    values: np.ndarray
    left_floor_contact: bool


class G1SlipDiagnosticReader:
    """Read world-frame body velocity and left-foot/floor contact diagnostics."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, floor_id: int) -> None:
        self.model = model
        self.data = data
        self.floor_id = floor_id
        self.pelvis_id = model.body("pelvis").id
        self.left_foot_body_id = model.body("left_ankle_roll_link").id
        self.left_foot_geom_ids = frozenset(
            model.geom(name).id for name in LEFT_FOOT_CONTACT_GEOM_NAMES
        )
        self._pelvis_velocity = np.zeros(6, dtype=np.float64)
        self._foot_velocity = np.zeros(6, dtype=np.float64)
        self._contact_wrench = np.zeros(6, dtype=np.float64)
        self.slip_displacement = 0.0

    def _body_velocity(self, body_id: int, result: np.ndarray) -> np.ndarray:
        # mj_objectVelocity returns [angular, linear] at the body origin.
        # flg_local=0 keeps the orientation in the world frame.
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body_id, result, 0
        )
        return result[3:6]

    def _left_floor_wrench(self) -> tuple[float, float, bool]:
        normal_force = 0.0
        tangential_force = 0.0
        has_contact = False
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom_pair = {int(contact.geom1), int(contact.geom2)}
            if self.floor_id not in geom_pair:
                continue
            if not (geom_pair & self.left_foot_geom_ids):
                continue
            self._contact_wrench.fill(0.0)
            mujoco.mj_contactForce(
                self.model, self.data, contact_id, self._contact_wrench
            )
            normal_force += max(0.0, float(self._contact_wrench[0]))
            tangential_force += float(np.linalg.norm(self._contact_wrench[1:3]))
            has_contact = True
        return normal_force, tangential_force, has_contact

    def advance_slip(self, timestep: float, accumulate: bool) -> None:
        """Integrate planar left-foot path length at physics rate while in contact."""
        _, _, has_contact = self._left_floor_wrench()
        foot_velocity = self._body_velocity(
            self.left_foot_body_id, self._foot_velocity
        )
        if accumulate and has_contact:
            self.slip_displacement += float(np.linalg.norm(foot_velocity[:2])) * timestep

    def read(self, pulse_active: bool) -> SlipDiagnosticSample:
        pelvis_velocity = self._body_velocity(self.pelvis_id, self._pelvis_velocity)
        foot_velocity = self._body_velocity(
            self.left_foot_body_id, self._foot_velocity
        )
        normal_force, tangential_force, has_contact = self._left_floor_wrench()
        force_ratio = tangential_force / max(normal_force, 1e-12)
        values = np.asarray(
            (
                pelvis_velocity[0],
                pelvis_velocity[1],
                foot_velocity[0],
                foot_velocity[1],
                self.slip_displacement,
                tangential_force,
                normal_force,
                force_ratio,
                float(pulse_active),
            ),
            dtype=np.float64,
        )
        return SlipDiagnosticSample(values=values, left_floor_contact=has_contact)
