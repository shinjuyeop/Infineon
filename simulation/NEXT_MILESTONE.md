# Next Milestone: Expanded Dataset v1 and Small 1D-CNN

Status: **expanded dataset, three-seed CNN gate, and host INT8 parity complete;
E84 compatibility is next**.

The implementation start adds:

- `expanded_terrain_dataset_v1.py`: seven bounded surface families, deterministic
  family assignment, leakage checks, and pilot-based execution-cost estimates;
- `run_expanded_terrain_dataset_v1.py`: dry-run-first planning and a separate,
  overwrite-safe expanded dataset generator;
- `terrain_cnn.py`: split-safe channel normalization, the shared compact model,
  evaluation helpers, and resource estimates;
- `train_terrain_1d_cnn.py`: clean/noisy FSR/IMU/Fusion training with identical
  architecture/training policy and per-family test metrics;
- `test_expanded_terrain_dataset_v1.py` and `test_terrain_cnn.py`: family leakage,
  balance, boundedness, normalization, resource, metric, and model smoke tests.

The completed 1,200-run pilot and all historical experiment outputs remain
unchanged. Expanded outputs use `simulation/outputs/terrain_dataset_v1_expanded*`
and are gitignored. Final host-side results are recorded in
`EXPANDED_DATASET_V1_RESULTS.md`.

## Objective and gate

Expand Dataset v1 for unseen **surface-family** generalization, then compare
FSR4, IMU6, and Fusion10 using the same embedded-friendly 1D-CNN family. A larger
sample count alone is not success; the held-out procedural family is the gate.

## A. Dataset diversity

- Add multiple bounded procedural surface families, for example multisine,
  filtered random field, sparse shallow aggregate, and smooth low-amplitude
  family variants.
- Keep each family within documented engineering ranges; do not tune ranges to
  classifier results or claim they are measured material distributions.
- Retain clean/noisy pairing, four labels, foot sensor schema, medium window,
  validity filtering, bidirectional pulse coverage, and diagnostic separation.
- Review 4,000–6,000 valid windows before execution; exact count is not locked.

### Implemented family allocation

| Split | Procedural family | Nominal spatial scale |
|---|---|---:|
| Train | `multisine` | 36–121 mm |
| Train | `filtered_random` | 80–250 mm |
| Train | `sparse_aggregate` | 50–120 mm |
| Validation | `crosshatch` | 58–121 mm |
| Validation | `rounded_ridges` | 58–363 mm including modulation |
| Test | `warped_multisine` | 36–363 mm including coordinate warp |
| Test | `smooth_random_patches` | 75–180 mm |

Each normalized morphology is deterministic, zero-centered, and bounded to
`[-1, 1]`. Terrain-specific Dataset v1 peak-to-valley amplitude ranges and
friction ranges remain unchanged; the new family definitions do not enlarge
roughness solely to improve classification.
The 10 mm hfield grid resolves the smallest 36 mm component with more than three
grid intervals. These are bounded engineering morphology ranges, not measured
material spectra or claims about real concrete, marble, ice, or sand.

The default design is 4 terrains x 7 families x 8 surfaces x 20 runs = 4,480
candidates. Applying the pilot valid rate gives an estimate of about 4,439 valid
windows, within the requested 4k–6k range.

## B. Leakage-safe split

- Assign entire surface families to train, validation, or test.
- Within a family, keep surface seeds, sessions, and run groups in one split.
- Test must contain at least one family never used for fitting, normalization,
  hyperparameter selection, or INT8 calibration.
- Write automated family-level leakage and balance tests before generation.

## C. FSR/IMU/Fusion neural ablation

Use one architecture family and training protocol for all inputs:

1. FSR4: channels 0–3.
2. IMU6: channels 4–9.
3. Fusion10: channels 0–9.

Only the input channel count may differ. Report parameter count for every model
and do not give Fusion extra depth or training budget.

## D. Embedded-friendly 1D-CNN candidate

```text
Input (50, C)
  → Conv1D
  → Conv1D
  → GlobalAveragePooling1D
  → Dense(4, softmax)
```

Before fixing widths/kernel sizes, estimate parameter count, activation peak RAM,
model flash, and operator compatibility. Prefer a compact topology suitable for
Cortex-M55/Ethos-U55 rather than maximizing host accuracy.

The implemented first candidate fixes widths at 12 and 16 with kernels 5 and 3:

| Input | Parameters | Float parameter bytes | Estimated float activation working set |
|---|---:|---:|---:|
| FSR4 | 912 | 3,648 | 5,600 bytes |
| IMU6 | 1,032 | 4,128 | 5,600 bytes |
| Fusion10 | 1,272 | 5,088 | 5,600 bytes |

The activation estimate is tensor liveness only. TFLite arena metadata,
alignment, and kernel scratch buffers must be measured after conversion; it is
not a deployment RAM claim.

## E. Evaluation

- Overall accuracy and macro F1.
- Per-class precision, recall, and F1.
- Confusion matrix and Concrete↔Marble mutual confusion.
- Clean/noisy comparison.
- FSR/IMU/Fusion comparison with identical family splits.
- Per-family generalization, not only pooled test performance.
- RandomForest pilot retained as a reference baseline.

## F. Decision sequence

If the CNN generalizes to unseen surface families:

```text
CNN gate → INT8 conversion → host-side quantized parity
         → KIT_PSE84_AI deployment work
```

If it fails, investigate domain coverage, family shift, split design,
normalization, model capacity, and data quality first. Do **not** jump directly to
a vibration model merely because a classifier fails.

## Entry points and execution cost

These entry points now exist:

- `run_expanded_terrain_dataset_v1.py`: generation and manifests.
- `train_terrain_1d_cnn.py`: shared FSR/IMU/Fusion training.
- `export_terrain_int8.py`: train-family calibration, strict INT8 export, and
  unseen-family float/INT8 parity.

Family-disjoint metrics and reports are currently integrated into
`train_terrain_1d_cnn.py`; a separate `evaluate_terrain_1d_cnn.py` remains
deferred until model serialization and INT8 evaluation requirements are fixed.

The completed pilot took approximately 756.5 seconds and 110,186,372 bytes for
1,200 candidates on this host. Linear scaling estimates the default expanded run
at about 47.1 minutes and 392.3 MiB. This is a planning estimate, not a guarantee.

Dry-run review (default; writes nothing):

```bash
cd simulation/unitree_mujoco/simulate_python
../../venv/bin/python run_expanded_terrain_dataset_v1.py
```

Full generation must be explicitly authorized with `--execute`:

```bash
../../venv/bin/python run_expanded_terrain_dataset_v1.py --execute
```

After successful generation, install `requirements-cnn.txt` in a suitable
training environment and run:

```bash
python train_terrain_1d_cnn.py
```

The trainer refuses to overwrite non-empty output directories, fits
normalization on train families only, uses validation families for early
stopping, and reserves test families for final evaluation.

### Startup smoke evidence

A 28-run integration smoke (4 terrains x 7 families x 1 surface x 1 run)
completed with 28/28 valid windows and `(28, 50, 10)` clean/noisy tensors. A
one-epoch noisy FSR/IMU/Fusion CNN smoke also completed, confirming dataset load,
train-only normalization, model save, and split/per-family evaluation paths.
Its 25% accuracy is not a model result: the smoke is intentionally balanced,
tiny, and trained for only one epoch. The subsequent full result is recorded in
`EXPANDED_DATASET_V1_RESULTS.md`.

## E84 readiness checklist

Nothing below is complete merely because the Dataset v1 pilot exists.

- [x] Freeze the current simulation deployment-candidate input tensor shape,
  channel order, units, and sample timing.
- [x] Freeze host normalization metadata; embedded reproduction remains pending.
- [x] Define representative, split-safe INT8 calibration data.
- [x] Confirm strict INT8 conversion and float/INT8 host parity.
- [ ] Verify TFLite/LiteRT and DEEPCRAFT import/operator compatibility.
- [ ] Measure model parameter count, flash size, arena/activation RAM.
- [ ] Measure inference latency and sustained timing budget.
- [ ] Confirm Cortex-M55 and Ethos-U55 execution path and tool versions.
- [ ] Specify UART framing, byte order, versioning, CRC, and error handling.
- [ ] Implement and verify Host → E84 input transfer.
- [ ] Implement and verify E84 → Host class/score result transfer.
- [ ] Define HIL buffering, window cadence, timeout, and end-to-end timing.
- [ ] Calibrate real FSR and BMI270 orientation, gain, bias, range, and noise.

## First three actions after reset

1. Read `EXPANDED_DATASET_V1_RESULTS.md` and verify the selected Keras, noisy
   dataset, and TFLite hashes before using local ignored artifacts.
2. Test the 7,048-byte strict INT8 model with the intended DEEPCRAFT/TFLite Micro
   importer and record supported operators/tool versions.
3. Measure arena/scratch RAM and inference latency on the Cortex-M55/Ethos-U55
   path before designing UART/HIL timing around unmeasured assumptions.
