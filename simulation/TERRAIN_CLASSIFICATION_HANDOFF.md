# Terrain Classification Handoff

This is the canonical reset handoff for the MuJoCo terrain-classification work.
Read it together with `TERRAIN_DATASET_V1.md` and `NEXT_MILESTONE.md` before
starting another experiment.

## Current pipeline

```text
Terrain parameters and procedural surface
                    ↓
            MuJoCo full-body G1
                    ↓
      FSR4 + left-foot/ankle IMU6
                    ↓
    domain variation + sensor imperfections
                    ↓
 medium_response [0.15, 0.65) at 100 Hz
                    ↓
                  (50, 10)
                    ↓
 surface/session-disjoint train/validation/test
                    ↓
      host classifier → future KIT_PSE84_AI
```

The operational MuJoCo timestep for low-frequency Dataset v1 dynamics is 0.5 ms
(2 kHz). This is not a claim that raw high-frequency contact vibration converged.

## Design decisions

### LOCKED / CURRENT

- AI channels: `foot_force_1..4`, accelerometer XYZ, gyroscope XYZ.
- IMU site: left foot/ankle, rigidly attached to `left_ankle_roll_link`.
- Classes: concrete=0, marble=1, ice=2, sand=3.
- Canonical window: `medium_response`, `[0.15, 0.65)`.
- Current input shape: `(50, 10)` at 100 Hz.
- Split unit: surface seed, session, and run group; never random windows.
- Test data uses unseen surface realizations.
- Use MuJoCo friction, slip, load distribution, CoP-related response, surface
  geometry response, and low-frequency foot dynamics.

### DIAGNOSTIC-ONLY

- Pelvis IMU (`imu_acc`, `imu_gyro`).
- Pelvis/foot velocity, slip, tangential/normal contact force, collision and
  contact-validity traces.
- Clean raw simulation CSVs and high-rate convergence artifacts.

### DEFERRED / OPTIONAL

- Separate high-frequency vibration sensor model.
- Exact PSD, dominant-frequency, and micro-contact spectral features.
- Gait-based dataset and locomotion-controller integration.
- Real FSR/BMI270 gain, bias, orientation, bandwidth, and noise calibration.

### SUPERSEDED FOR THE CURRENT DATASET

- Pelvis IMU as an AI input.
- Raw MuJoCo high-frequency vibration as classifier evidence.
- Treating 50/100/200 Hz logging differences as physics convergence.
- Mixing the legacy synthetic `(50, 5)` terrain schema with MuJoCo `(50, 10)`.

The synthetic pipeline under `Infineon_HIL` is preserved for its original host,
quantization, and HIL work. It is not silently converted or joined to Dataset v1.

## Dataset v1 pilot result

| Metric | Result |
|---|---:|
| Candidate / valid | 1,200 / 1,189 |
| Clean/noisy tensor | `(1189, 50, 10)` |
| FSR-only test accuracy | 84.81% |
| IMU-only test accuracy | 97.47% |
| Fusion test accuracy | 96.20% |
| Concrete recall, fusion | 96.7% |
| Marble recall, fusion | 90.0% |
| Concrete-Marble mutual confusion | 5.8% |

These are unseen-surface results within one procedural surface family. Fusion is
better than FSR-only, but it is not yet valid to claim that Fusion is superior to
IMU-only.

## Expanded milestone startup status

The next milestone implementation scaffold now exists without changing the
pilot or historical experiments:

- seven procedural surface families with train/validation/test family splits;
- a dry-run-first 4,480-candidate generator and pilot-based cost estimate;
- one shared compact Conv1D architecture for FSR4, IMU6, and Fusion10;
- train-family-only normalization and pooled/per-family evaluation;
- deterministic leakage, balance, morphology, resource, and CNN smoke tests.

A 28-run MuJoCo integration smoke completed with 28/28 valid windows, and all
three CNN input variants completed a one-epoch pipeline smoke. These are wiring
checks, not classifier evidence. The full expanded dataset and substantive CNN
ablation have not been executed. See `NEXT_MILESTONE.md` for exact allocation,
cost, commands, and gates.

## Experiment history

| Experiment | Purpose | Conclusion | Retained in current pipeline | Superseded/deferred |
|---|---|---|---|---|
| Passive terrain | Establish initial four-terrain contact response | Unsupported/passive runs exposed force/acceleration instability and weak hard-surface separation | Terrain profiles and failure checks | Passive data is not a training baseline |
| Controlled excitation | Stabilize comparisons with matched initial conditions and 70% support | Removed the worst passive artifacts; concrete-marble remained difficult | Matched seeds, support, validity/outlier rules | Narrow deterministic dataset was insufficient |
| Horizontal pulse | Add controlled friction/slip excitation | 80 N pulse exposed terrain slip ordering but did not improve full-window concrete-marble separation | 80 N-class pulse and slip/contact diagnostics | Full-window separation as the design target |
| Bidirectional validation | Check +X/-X asymmetry and response windows over 480 runs | Slip ordering was direction-consistent; `medium_response` was the best practical window, but all-pair gate failed | Both pulse directions and canonical medium window | One-direction training and full window |
| Lower-body validation | Test a reduced model while preserving mass/COM/inertia and contact layout | Equivalent lower-body model was validated as a research option | Validation utilities and reference model | Current Dataset v1 continues to use full-body G1 |
| Foot IMU A/B | Compare pelvis versus foot/ankle IMU | Foot location did not improve aggregate concrete-marble separation at flat 50 Hz, but Accel-X sensitivity increased | Foot IMU fixed as AI input; pelvis retained diagnostically | Pelvis IMU AI input |
| Surface/sampling study | Test flat/surface-aware terrain and 50/200 Hz logging | Roughness and higher logging changed separation; apparent spectral peaks were not yet physics-validated | Native hfield representation and low-frequency response | Treating 200 Hz peaks as final vibration features |
| Friction × roughness factorization | Separate material factors | FSR response was roughness-dominant; foot IMU response was friction/slip-dominant | Sensor complementarity and domain-variation design | Claiming simulated response is measured material physics |
| Timestep convergence | Audit 5/2/1 ms and common-band vibration | Previously reported 60–70 Hz behavior was timestep-sensitive | Requirement to separate low-frequency dynamics from raw vibration | 5 ms/200 Hz spectral conclusions |
| Final vibration limitation | Compare 1/0.5 ms and finally 0.5/0.25 ms | Aggregate force/slip converged sufficiently; vibration waveform/PSD and micro-contact switching did not | 0.5 ms operational physics for low-frequency Dataset v1 | Further timestep search and raw high-frequency PSD |
| Dataset v1 pilot | Add domain variation, noise, leakage-safe splits, and classifier evidence | 1,189 valid; IMU 97.47%, Fusion 96.20%; unseen surfaces feasible | Current `(50,10)` pilot and next-milestone baseline | Claim that Fusion already beats IMU-only |

## Repository map and cleanup classification

### Current source

- `terrain_dataset_v1.py`: schema, variation, sensor model, split checks, features,
  and RandomForest evaluation.
- `run_terrain_dataset_v1_pilot.py`: current pilot entry point.
- `hil_sensor.py`, `terrain_profiles.py`, `surface_profiles.py`,
  `controlled_excitation.py`, `run_horizontal_pulse_dataset.py`, and
  `pulse_windows.py`: current simulation foundations.

### Tests

- `test/test_terrain_dataset_v1.py`: seed/noise pairing, leakage, balance,
  filtering, shape, and classifier smoke tests.
- `test/test_hil_sensor_reader.py`, `test/test_surface_sampling_study.py`, and
  `test/test_timestep_convergence.py`: retained regression coverage.
- `gamepad_test.py` and `hil_sensor_test.py` are manual/hardware-oriented scripts,
  not part of the automatic `test*.py` unittest suite.

### Documentation

- `TERRAIN_DATASET_V1.md`: current pilot protocol and result.
- This file: design/history/reset handoff.
- `NEXT_MILESTONE.md`: implementation gate and E84 checklist.
- `models/g1_lower_body/README.md`: historical reduced-model validation.

### Generated output

- Everything under `simulation/outputs/` is generated and gitignored.
- `terrain_dataset_v1_pilot/` is the current evidence directory.
- Directories with `smoke`, `rerun`, `analysis`, or convergence names are
  preserved experiment artifacts, not current training inputs.

### Historical but still useful experiments

- Bidirectional validation, IMU-location A/B, friction×roughness factorization,
  lower-body validation, and final timestep convergence.

### Superseded experiment paths

- Initial passive terrain analysis and old pelvis-labelled analysis scripts.
- Early 5 ms/200 Hz spectral interpretation and intermediate rerun directories.
- These are retained for provenance; do not import their outputs into Dataset v1.

## Code-health audit notes

- No source deletion or refactor was required for this handoff.
- `analyze_terrain_data.py` contains pelvis-specific labels and belongs to the
  initial passive experiment, not the current foot-IMU pipeline.
- CSV loading/writing, separation, and summary helpers are duplicated across
  research runners. Consolidation may be considered later, but changing them now
  would create unnecessary regression risk.
- Final timestep reporting relies on a wrapper that replaces the generic
  assessment with the 0.25 ms terminal assessment. It works, but the coupling is
  worth removing only during a dedicated cleanup.
- Dataset dependencies are in `requirements-dataset-v1.txt`; the simulator as a
  whole still lacks one consolidated environment lockfile.

## Reset recovery

After a full reset, first read this file, then `TERRAIN_DATASET_V1.md`, then
`NEXT_MILESTONE.md`. Do not rerun the historical physics studies unless a model
or contact representation is intentionally changed.
