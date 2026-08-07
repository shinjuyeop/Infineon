# Expanded Dataset v1, CNN ablation, and INT8 parity

Status: **host-side milestone gate passed; E84 deployment not started**.

This document records the generated-result evidence. Raw simulation data,
Keras models, and TFLite artifacts remain under gitignored `simulation/outputs/`.
They are not measured sensor data and are not firmware deployment evidence.

## Expanded dataset

| Item | Result |
|---|---:|
| Candidates / valid / invalid | 4,480 / 4,453 / 27 |
| Clean/noisy shape | `(4453, 50, 10)` |
| Train / validation / test | 1,909 / 1,271 / 1,273 |
| Classes | Concrete, Marble, Ice, Sand |
| Sampling/window | 100 Hz, `medium_response`, `[0.15, 0.65)` |
| AI channels | FSR4 + left-foot IMU6 |

Surface-family ownership remained disjoint:

- train: `multisine`, `filtered_random`, `sparse_aggregate`;
- validation: `crosshatch`, `rounded_ridges`;
- test: `warped_multisine`, `smooth_random_patches`.

## Three-seed noisy CNN gate

All variants used the same architecture and training budget:

```text
Conv1D(12,k=5) -> Conv1D(16,k=3) -> GAP -> Dense(4)
maximum 120 epochs, patience 12, batch size 64
```

| Sensor group | Test accuracy, mean +/- sample SD | Macro F1, mean +/- sample SD | Concrete-Marble confusion |
|---|---:|---:|---:|
| FSR-only | 97.41 +/- 0.47% | 97.42 +/- 0.47% | 1.82 +/- 0.45% |
| IMU-only | 96.60 +/- 0.09% | 96.61 +/- 0.09% | 5.99 +/- 0.09% |
| Fusion | **98.95 +/- 0.40%** | **98.96 +/- 0.39%** | **1.35 +/- 0.48%** |

Fusion exceeded IMU-only accuracy in all three seeds by 2.04–2.83 percentage
points. Mean Fusion recall was 98.85% for Concrete and 98.02% for Marble. Mean
per-family Fusion accuracy was 99.37% on `smooth_random_patches` and 98.54% on
`warped_multisine`.

Seed `20260809` was selected by validation loss, not by test performance. Its
Fusion model reached best validation loss at epoch 79 and was restored after
early stopping at epoch 91.

## Strict full-INT8 host parity

Source candidate:

```text
simulation/outputs/terrain_dataset_v1_expanded_cnn_seed_20260809_e120/noisy_fusion.keras
```

Calibration used 256 noisy windows exclusively from the train partition:

| Calibration dimension | Distribution |
|---|---|
| Split | train=256, validation=0, test=0 |
| Families | filtered_random=86, multisine=85, sparse_aggregate=85 |
| Classes | Concrete=66, Marble=64, Ice=63, Sand=63 |

The small two-sample class imbalance follows equal allocation across 12
class/family strata and does not include held-out families.

| Metric | Float32 | Full INT8 | INT8 delta |
|---|---:|---:|---:|
| Unseen-family test accuracy | 99.372% | 99.293% | -0.079 percentage points |
| Macro F1 | 99.375% | 99.297% | -0.078 percentage points |
| Concrete-Marble confusion | 0.938% | 1.094% | +0.156 percentage points |
| Prediction agreement | — | 99.607% | — |

INT8 per-family accuracy was 99.53% on `smooth_random_patches` and 99.06% on
`warped_multisine`. INT8 Concrete recall was 99.38% and Marble recall was 98.44%.

Parity gates all passed:

- accuracy delta at least -1 percentage point;
- macro-F1 delta at least -1 percentage point;
- Concrete-Marble confusion increase at most +1 percentage point.

The exported model is 7,048 bytes and contains only builtin integer operators,
with no Flex or floating-point tensors:

| Tensor | Interface | Scale | Zero point |
|---|---|---:|---:|
| Input | `int8 (1, 50, 10)` | 0.1095911264 | 22 |
| Output | `int8 (1, 4)` | 0.00390625 | -128 |

Reproducibility hashes:

- Keras source: `a10f8fd28226e54c0476bf81124d15058b7e18c21774dcd7c0a684121cc17405`
- Noisy dataset: `ee84861570a1ae1244c54957219b9896c29a1756be97d4d02353f9699ac264ec`
- TFLite model: `f5d34f48b765d89f61cebe7e15eb9ee27a78b1c823950bca67043b6d4fa96df4`

## Reproduction

```bash
cd simulation/unitree_mujoco/simulate_python
/d/shin/Infineon/Infineon_HIL/.venv/bin/python export_terrain_int8.py
```

The exporter refuses to overwrite a non-empty output directory. It recomputes
train-only normalization, validates family ownership, writes the exact
calibration manifest, inspects strict INT8 tensor/operator state, and evaluates
the full held-out test partition.

## Remaining deployment gates

This result is sufficient to begin E84 compatibility work, but it does not prove
embedded deployability. Still required:

- DEEPCRAFT/TFLite importer and operator compatibility;
- measured TFLite Micro arena and peak activation/scratch RAM;
- measured Cortex-M55/Ethos-U55 flash placement and inference latency;
- embedded reproduction of normalization and INT8 input quantization;
- UART framing and Host-to-E84/E84-to-Host HIL timing;
- real FSR and BMI270 calibration and real-surface validation.
