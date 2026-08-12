"""Adapter for Unitree's official MuJoCo-trained G1 29-DOF velocity policy.

Constants and the observation/action layout match the BSD-3-Clause
``unitreerobotics/unitree_rl_mjlab`` G1 deployment configuration. The ONNX
artifact remains user supplied and is not vendored in this repository.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


ACTUATOR_NAMES = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll", "right_hip_pitch",
    "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch",
    "right_ankle_roll", "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)
DEFAULT_ANGLES = np.asarray(
    (-.1, 0., 0., .3, -.2, 0., -.1, 0., 0., .3, -.2, 0., 0., 0., 0.,
     .35, .18, 0., .87, 0., 0., 0., .35, -.18, 0., .87, 0., 0., 0.),
    dtype=np.float64,
)
KPS = np.asarray(
    (40.2, 99.1, 40.2, 99.1, 28.5, 28.5, 40.2, 99.1, 40.2, 99.1,
     28.5, 28.5, 40.2, 28.5, 28.5, 14.3, 14.3, 14.3, 14.3, 14.3,
     16.8, 16.8, 14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8),
    dtype=np.float64,
)
KDS = np.asarray(
    (2.6, 6.3, 2.6, 6.3, 1.8, 1.8, 2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
     2.6, 1.8, 1.8, .9, .9, .9, .9, .9, 1.1, 1.1, .9, .9, .9, .9,
     .9, 1.1, 1.1),
    dtype=np.float64,
)
ACTION_SCALE = np.asarray(
    (.55, .35, .55, .35, .44, .44, .55, .35, .55, .35, .44, .44,
     .55, .44, .44, .44, .44, .44, .44, .44, .07, .07, .44, .44,
     .44, .44, .44, .07, .07),
    dtype=np.float64,
)
POLICY_PERIOD_S = 0.6
CONTROL_PERIOD_S = 0.02
UPSTREAM_REVISION = "1425b15f73bd4095f0df53709d7c389c3eb9e790"
TESTED_POLICY_SHA256 = "2a66ca6336eadb3c0b34b557763f3e06d01ff8fcf6260dd4cedbd69d6093fc28"


def gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    """Rotate world gravity unit vector into the body frame (wxyz input)."""
    qw, qx, qy, qz = quaternion
    return np.asarray(
        (
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ),
        dtype=np.float64,
    )


class UnitreeG1PretrainedController:
    """Run Unitree RL MjLab's official full-body G1 velocity policy."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        policy_path: Path,
        forward_speed_mps: float,
    ) -> None:
        if not 0.1 <= forward_speed_mps <= 0.5:
            raise ValueError("forward speed must be in [0.1, 0.5] m/s for gait mode")
        actual = tuple(model.actuator(index).name for index in range(model.nu))
        if actual != ACTUATOR_NAMES:
            raise ValueError(f"G1 29-DOF actuator layout changed: {actual}")
        steps_float = CONTROL_PERIOD_S / float(model.opt.timestep)
        self.control_decimation = int(round(steps_float))
        if not np.isclose(steps_float, self.control_decimation, atol=1e-12, rtol=0.0):
            raise ValueError("physics timestep must divide the 20 ms policy period")
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "onnxruntime is required; install requirements-walking-touchdown-v1.txt"
            ) from exc
        if not policy_path.is_file():
            raise FileNotFoundError(f"Unitree G1 policy not found: {policy_path}")
        self.session = ort.InferenceSession(
            str(policy_path), providers=("CPUExecutionProvider",)
        )
        if self.session.get_inputs()[0].shape != [1, 98]:
            raise ValueError("expected Unitree G1 policy input [1,98]")
        if self.session.get_outputs()[0].shape != [1, 29]:
            raise ValueError("expected Unitree G1 policy output [1,29]")
        self.model, self.data = model, data
        self.command = np.asarray((forward_speed_mps, 0.0, 0.0), dtype=np.float64)
        self.action = np.zeros(29, dtype=np.float32)
        self.target_position = DEFAULT_ANGLES.copy()
        self.step_count = 0
        self.global_phase = 0.0
        gyro_id = model.sensor("imu_gyro").id
        self._gyro_address = int(model.sensor_adr[gyro_id])
        self._gyro_dimension = int(model.sensor_dim[gyro_id])

        # The official FSM first reaches this fixed stand pose. Starting the
        # simulation at that same pose avoids an unrelated two-second startup
        # transient in every short dataset run.
        data.qpos[7:] = DEFAULT_ANGLES
        mujoco.mj_forward(model, data)

    def apply(self) -> None:
        torque = (
            (self.target_position - self.data.qpos[7:]) * KPS
            - self.data.qvel[6:] * KDS
        )
        self.data.ctrl[:] = np.clip(
            torque, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
        )

    def update_after_step(self) -> None:
        self.step_count += 1
        if self.step_count % self.control_decimation:
            return
        self.global_phase = (self.global_phase + CONTROL_PERIOD_S / POLICY_PERIOD_S) % 1.0
        gyro = self.data.sensordata[
            self._gyro_address : self._gyro_address + self._gyro_dimension
        ]
        observation = np.concatenate(
            (
                gyro,
                gravity_orientation(self.data.qpos[3:7]),
                self.command,
                (np.sin(2.0 * np.pi * self.global_phase), np.cos(2.0 * np.pi * self.global_phase)),
                self.data.qpos[7:] - DEFAULT_ANGLES,
                self.data.qvel[6:],
                self.action,
            )
        ).astype(np.float32)
        if observation.shape != (98,) or not np.all(np.isfinite(observation)):
            raise ValueError("invalid 98-element G1 policy observation")
        input_name = self.session.get_inputs()[0].name
        self.action = self.session.run(None, {input_name: observation[None, :]})[0].squeeze()
        if self.action.shape != (29,) or not np.all(np.isfinite(self.action)):
            raise ValueError("invalid 29-element G1 policy action")
        self.target_position = self.action * ACTION_SCALE + DEFAULT_ANGLES
