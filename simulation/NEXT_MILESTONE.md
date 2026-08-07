# Next Milestone: Expanded Dataset v1 and Small 1D-CNN

Status: **design only; not implemented or executed**.

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

## Planned entry points

These names are candidates and do not exist yet:

- `run_expanded_terrain_dataset_v1.py`: generation and manifests.
- `train_terrain_1d_cnn.py`: shared FSR/IMU/Fusion training.
- `evaluate_terrain_1d_cnn.py`: family-disjoint metrics and reports.

Start implementation only after reviewing the family definitions, valid-sample
budget, storage estimate, and expected runtime.

## E84 readiness checklist

Nothing below is complete merely because the Dataset v1 pilot exists.

- [ ] Freeze final input tensor shape, channel order, units, and sample timing.
- [ ] Freeze normalization and reproduce it in host and embedded code.
- [ ] Define representative, split-safe INT8 calibration data.
- [ ] Confirm strict INT8 conversion and float/INT8 host parity.
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

1. Define and unit-test at least three bounded procedural surface families plus a
   family-level split manifest, without generating the full dataset.
2. Estimate 4k–6k run time/storage and approve the valid-window budget and family
   allocation.
3. Implement a tiny synthetic-shape smoke test for the shared FSR/IMU/Fusion
   Conv1D architecture before any large simulation run.
