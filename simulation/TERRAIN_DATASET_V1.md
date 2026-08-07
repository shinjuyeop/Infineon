# MuJoCo Terrain Dataset v1 Pilot

This pipeline is separate from the legacy synthetic `(50, 5)` dataset under
`Infineon_HIL`. Its schema is `(N, 50, 10)` with four FSR channels followed by
left-foot accelerometer XYZ and gyroscope XYZ.

```text
Terrain parameters
      ↓
MuJoCo full-body G1
      ↓
FSR4 + left-foot IMU6
      ↓
Domain variation
      ↓
Sensor imperfections
      ↓
Canonical medium-response window
      ↓
Dataset v1 pilot
      ↓
RandomForest baseline
```

MuJoCo physics ground truth consists of friction, slip, normal load, load
distribution, CoP-related response, surface geometry response, and low-frequency
foot dynamics. The virtual sensor input is FSR4 plus the ankle-mounted foot IMU6.
The pelvis IMU and raw contact diagnostics are diagnostic-only and are stored
outside the AI tensors. Sensor imperfections are reproducible pilot engineering
assumptions, not measured KIT_PSE84_AI/BMI270 calibration values.

The AI input deliberately excludes timestep-sensitive raw high-frequency
vibration PSD, exact dominant vibration frequency, and micro-contact spectral
features. Clean and noisy tensors are saved separately, and train/validation/test
surface realizations and sessions are disjoint.

Run from `simulation/unitree_mujoco/simulate_python`:

```bash
/d/shin/Infineon/simulation/venv/bin/python -m pip install -r requirements-dataset-v1.txt
/d/shin/Infineon/simulation/venv/bin/python run_terrain_dataset_v1_pilot.py
```

Generated data is written to the gitignored
`simulation/outputs/terrain_dataset_v1_pilot/` directory. The runner refuses to
overwrite a non-empty output directory.
