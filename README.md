# Infineon

Workspace for KIT_PSE84_AI host/HIL experiments and Unitree G1 MuJoCo
simulation.

## Structure

- `Infineon_HIL/`: host-side terrain and deployment validation pipeline.
- `Infineon_test/`: KIT_PSE84_AI and ModusToolbox project sources.
- `simulation/unitree_mujoco/`: Unitree MuJoCo simulator with G1 virtual
  sensors, terrain logging, controlled excitation, and analysis tools.
- `simulation/unitree_sdk2_python/`: Unitree SDK2 Python dependency.

Local documents, virtual environments, downloaded ModusToolbox dependencies,
build products, generated datasets, plots, and model artifacts are intentionally
excluded from version control. Recreate them using the component README and
dependency files.

The simulator terrain profiles are engineering approximations for controlled
signal-separation experiments. They are not measured material properties or
real sensor data.

Third-party license and attribution files remain with their respective source
trees. This repository does not replace or relicense those upstream projects.

The leakage-safe MuJoCo `(N, 50, 10)` terrain pilot and its separation from the
legacy synthetic `(50, 5)` pipeline are documented in
[`simulation/TERRAIN_DATASET_V1.md`](simulation/TERRAIN_DATASET_V1.md).

## G1 horizontal-pulse preview

After preparing the simulation virtual environment:

```bash
cd simulation/unitree_mujoco/simulate_python
python run_horizontal_pulse_dataset.py \
  --terrain ice \
  --runs 1 \
  --duration 1.0 \
  --sample-rate 50 \
  --seed 20260805 \
  --support-ratio 0.70 \
  --pulse-start 0.25 \
  --pulse-duration 0.20 \
  --pulse-magnitude 80 \
  --gui \
  --realtime-factor 0.25 \
  --gui-hold-seconds 5
```
