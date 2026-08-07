"""Regression tests for the G1 HIL sensor location and 10-channel API."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np


SIMULATE_PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMULATE_PYTHON_DIR))

import config  # noqa: E402
from hil_sensor import G1HilSensorReader, HIL_SENSOR_CHANNELS  # noqa: E402


class G1HilSensorReaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        scene_path = (SIMULATE_PYTHON_DIR / config.ROBOT_SCENE).resolve()
        cls.model = mujoco.MjModel.from_xml_path(str(scene_path))
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)
        cls.reader = G1HilSensorReader(cls.model, cls.data)

    def test_left_foot_imu_is_rigidly_attached_to_ankle_roll(self) -> None:
        site_id = self.model.site("left_foot_imu").id
        ankle_id = self.model.body("left_ankle_roll_link").id
        self.assertEqual(int(self.model.site_bodyid[site_id]), ankle_id)
        np.testing.assert_allclose(
            self.model.site_pos[site_id], (0.035, 0.0, -0.03)
        )

    def test_named_foot_and_legacy_pelvis_sensors_exist(self) -> None:
        for sensor_name in (
            "left_foot_imu_acc",
            "left_foot_imu_gyro",
            "imu_acc",
            "imu_gyro",
        ):
            self.assertGreaterEqual(self.model.sensor(sensor_name).id, 0)

    def test_hil_and_pelvis_diagnostic_vectors_keep_shape(self) -> None:
        hil = self.reader.read_vector()
        pelvis = self.reader.read_pelvis_diagnostic_vector()
        self.assertEqual(hil.shape, (len(HIL_SENSOR_CHANNELS),))
        self.assertEqual(pelvis.shape, (len(HIL_SENSOR_CHANNELS),))
        np.testing.assert_array_equal(hil[:4], pelvis[:4])


if __name__ == "__main__":
    unittest.main()
