"""Repeatable G1 excitation and support helpers for controlled HIL datasets."""

from __future__ import annotations

from dataclasses import dataclass
import os

import mujoco
import numpy as np

from hil_sensor import LEFT_FOOT_CONTACT_GEOM_NAMES


os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
from unitree_sdk2py_bridge import ElasticBand  # noqa: E402


@dataclass(frozen=True)
class ExcitationCondition:
    run_id: str
    initial_velocity_x: float
    initial_velocity_y: float
    base_height_offset: float
    base_roll_deg: float
    base_pitch_deg: float


@dataclass(frozen=True)
class HorizontalPulse:
    """Smooth horizontal force pulse applied identically in every terrain."""

    start_time: float
    duration: float
    magnitude: float
    direction_x: float = 1.0
    direction_y: float = 0.0

    def __post_init__(self) -> None:
        if self.start_time < 0.0 or self.duration <= 0.0 or self.magnitude < 0.0:
            raise ValueError("invalid horizontal pulse timing or magnitude")
        direction = np.asarray((self.direction_x, self.direction_y), dtype=np.float64)
        if not np.all(np.isfinite(direction)) or np.linalg.norm(direction) <= 0.0:
            raise ValueError("horizontal pulse direction must be finite and non-zero")

    def force_at(self, time: float) -> tuple[np.ndarray, bool]:
        phase = (time - self.start_time) / self.duration
        if phase < 0.0 or phase >= 1.0:
            return np.zeros(3, dtype=np.float64), False
        direction = np.asarray((self.direction_x, self.direction_y), dtype=np.float64)
        direction /= np.linalg.norm(direction)
        amplitude = self.magnitude * np.sin(np.pi * phase)
        return np.asarray((*direction * amplitude, 0.0), dtype=np.float64), True


def generate_excitation_conditions(
    run_count: int, seed: int
) -> tuple[ExcitationCondition, ...]:
    if run_count <= 0:
        raise ValueError("run_count must be greater than zero")
    rng = np.random.default_rng(seed)
    conditions = []
    for run_index in range(1, run_count + 1):
        conditions.append(
            ExcitationCondition(
                run_id=f"run_{run_index:03d}",
                initial_velocity_x=float(rng.uniform(0.25, 0.40)),
                initial_velocity_y=float(rng.uniform(-0.06, 0.06)),
                base_height_offset=float(rng.uniform(0.0, 0.008)),
                base_roll_deg=float(rng.uniform(-0.4, 0.4)),
                base_pitch_deg=float(rng.uniform(-0.4, 0.4)),
            )
        )
    return tuple(conditions)


def generate_pulse_conditions(
    run_count: int, seed: int
) -> tuple[ExcitationCondition, ...]:
    """Generate paired pose perturbations without preloading horizontal velocity."""
    if run_count <= 0:
        raise ValueError("run_count must be greater than zero")
    rng = np.random.default_rng(seed)
    return tuple(
        ExcitationCondition(
            run_id=f"run_{run_index:03d}",
            initial_velocity_x=0.0,
            initial_velocity_y=0.0,
            base_height_offset=float(rng.uniform(0.0, 0.008)),
            base_roll_deg=float(rng.uniform(-0.4, 0.4)),
            base_pitch_deg=float(rng.uniform(-0.4, 0.4)),
        )
        for run_index in range(1, run_count + 1)
    )


def _roll_pitch_quaternion(roll_deg: float, pitch_deg: float) -> np.ndarray:
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    return np.asarray((cr * cp, sr * cp, cr * sp, -sr * sp), dtype=np.float64)


def apply_excitation_condition(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    condition: ExcitationCondition,
) -> tuple[int, int]:
    joint_id = model.joint("floating_base_joint").id
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("G1 floating_base_joint is not a free joint")
    qpos_address = int(model.jnt_qposadr[joint_id])
    dof_address = int(model.jnt_dofadr[joint_id])

    data.qpos[qpos_address + 2] += condition.base_height_offset
    data.qpos[qpos_address + 3 : qpos_address + 7] = _roll_pitch_quaternion(
        condition.base_roll_deg, condition.base_pitch_deg
    )
    data.qvel[dof_address] = condition.initial_velocity_x
    data.qvel[dof_address + 1] = condition.initial_velocity_y
    mujoco.mj_forward(model, data)
    return qpos_address, dof_address


class VerticalElasticBandSupport:
    """Use Unitree's ElasticBand force law with a horizontal tracking anchor."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        qpos_address: int,
        dof_address: int,
        support_ratio: float = 0.70,
        anchor_height: float = 2.0,
    ) -> None:
        if not 0.0 < support_ratio < 1.0:
            raise ValueError("support_ratio must be between zero and one")
        self.model = model
        self.data = data
        self.qpos_address = qpos_address
        self.dof_address = dof_address
        self.support_ratio = support_ratio
        self.anchor_height = anchor_height
        self.torso_body_id = model.body("torso_link").id

        self.band = ElasticBand()
        base_position = data.qpos[qpos_address : qpos_address + 3]
        self.band.point = base_position.copy()
        self.band.point[2] += anchor_height
        robot_weight = float(model.body_mass.sum()) * abs(float(model.opt.gravity[2]))
        target_support_force = support_ratio * robot_weight
        self.band.length = anchor_height - target_support_force / self.band.stiffness
        if self.band.length <= 0.0:
            raise ValueError("computed elastic-band length is not positive")
        self.target_support_force = target_support_force

    def apply(self) -> np.ndarray:
        base_position = self.data.qpos[
            self.qpos_address : self.qpos_address + 3
        ]
        base_velocity = self.data.qvel[self.dof_address : self.dof_address + 3]
        self.band.point[:2] = base_position[:2]
        force = self.band.Advance(base_position, base_velocity)
        self.data.xfrc_applied[self.torso_body_id, :3] = force
        return force


class HorizontalPulseExciter:
    """Add a world-frame half-sine force pulse to an existing body force."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        pulse: HorizontalPulse,
        body_name: str = "torso_link",
    ) -> None:
        self.data = data
        self.pulse = pulse
        self.body_id = model.body(body_name).id

    def apply(self, time: float) -> tuple[np.ndarray, bool]:
        force, active = self.pulse.force_at(time)
        self.data.xfrc_applied[self.body_id, :3] += force
        return force, active


def find_allowed_foot_geom_ids(model: mujoco.MjModel) -> frozenset[int]:
    geom_ids = {model.geom(name).id for name in LEFT_FOOT_CONTACT_GEOM_NAMES}
    for body_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        body_id = model.body(body_name).id
        geom_ids.update(
            geom_id
            for geom_id in range(model.ngeom)
            if model.geom_bodyid[geom_id] == body_id
            and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
        )
    return frozenset(geom_ids)


def has_nonfoot_floor_contact(
    data: mujoco.MjData, floor_id: int, allowed_foot_geom_ids: frozenset[int]
) -> bool:
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        if floor_id not in (contact.geom1, contact.geom2):
            continue
        other_geom = int(
            contact.geom2 if contact.geom1 == floor_id else contact.geom1
        )
        if other_geom not in allowed_foot_geom_ids:
            return True
    return False
