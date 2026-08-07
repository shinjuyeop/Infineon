# MuJoCo Terrain Dataset v1 Pilot

Status: **current production candidate for simulation/host validation**. It has
not been calibrated against real sensors and has not been deployed to E84.

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

## Pilot result

| Item | Result |
|---|---:|
| Candidate runs | 1,200 |
| Valid windows | 1,189 |
| Tensor | `(1189, 50, 10)` |
| FSR-only unseen-surface test accuracy | 84.81% |
| IMU-only unseen-surface test accuracy | 97.47% |
| Fusion unseen-surface test accuracy | 96.20% |
| Concrete recall, fusion | 96.7% |
| Marble recall, fusion | 90.0% |
| Concrete-Marble mutual confusion, fusion | 5.8% |

Fusion is better than FSR-only, but **the pilot does not establish that Fusion is
better than IMU-only**. The next milestone must repeat FSR/IMU/Fusion ablation
with unseen procedural surface families and a shared small 1D-CNN architecture.

## Status boundaries

- **Current/locked:** four terrain labels, FSR4 + foot IMU6, `medium_response`,
  `(50, 10)`, surface-disjoint splits, and low-frequency MuJoCo responses.
- **Diagnostic-only:** pelvis IMU, slip/contact traces, collision flags, and raw
  clean simulation logs.
- **Deprecated/superseded for Dataset v1:** pelvis IMU as AI input, direct use of
  raw MuJoCo high-frequency vibration/PSD, and mixing the synthetic `(50, 5)`
  schema with MuJoCo `(50, 10)` data.
- **Optional/deferred:** virtual high-frequency vibration sensor model, exact PSD
  features, gait-based collection, and real sensor calibration.
