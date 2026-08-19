# Terrain Dataset v1 Expanded Results (Historical)

상태: **historical host-side milestone**. 이 기록의 original Dataset v1 model은
후속 v4 frozen model과 deployment 판단을 대체하지 않는다.

이 문서는 generated-result evidence를 기록한다. Raw simulation data, Keras
model, TFLite artifact는 gitignored `simulation/outputs/` 아래에 유지한다. 이
결과는 measured sensor data가 아니며 firmware deployment evidence도 아니다.

## Expanded dataset

| 항목 | 결과 |
|---|---:|
| Candidates / valid / invalid | 4,480 / 4,453 / 27 |
| Clean/noisy shape | `(4453, 50, 10)` |
| Train / validation / test | 1,909 / 1,271 / 1,273 |
| Classes | Concrete, Marble, Ice, Sand |
| Sampling/window | 100 Hz, `medium_response`, `[0.15, 0.65)` |
| AI channels | FSR4 + left-foot IMU6 |

Surface-family ownership는 다음과 같이 완전히 분리됐다.

- train: `multisine`, `filtered_random`, `sparse_aggregate`
- validation: `crosshatch`, `rounded_ridges`
- test: `warped_multisine`, `smooth_random_patches`

## 3-seed noisy CNN gate

모든 variant에 동일한 architecture와 training budget을 적용했다.

```text
Conv1D(12,k=5) -> Conv1D(16,k=3) -> GAP -> Dense(4)
maximum 120 epochs, patience 12, batch size 64
```

| Sensor group | Test accuracy, 평균 +/- sample SD | Macro F1, 평균 +/- sample SD | Concrete-Marble confusion |
|---|---:|---:|---:|
| FSR-only | 97.41 +/- 0.47% | 97.42 +/- 0.47% | 1.82 +/- 0.45% |
| IMU-only | 96.60 +/- 0.09% | 96.61 +/- 0.09% | 5.99 +/- 0.09% |
| Fusion | **98.95 +/- 0.40%** | **98.96 +/- 0.39%** | **1.35 +/- 0.48%** |

Fusion accuracy는 세 seed 모두에서 IMU-only보다 2.04–2.83 percentage point
높았다. Fusion 평균 recall은 Concrete 98.85%, Marble 98.02%였다. Family별
Fusion 평균 accuracy는 `smooth_random_patches` 99.37%, `warped_multisine`
98.54%였다.

Seed `20260809`는 test performance가 아니라 validation loss를 기준으로
선택했다. Fusion model은 epoch 79에서 best validation loss에 도달했고 epoch
91에서 early stopping된 뒤 best weight가 복원됐다.

## Strict full-INT8 host parity

Source candidate:

```text
simulation/outputs/terrain_dataset_v1_expanded_cnn_seed_20260809_e120/noisy_fusion.keras
```

Calibration은 train partition의 noisy window 256개만 사용했다.

| Calibration 구분 | 분포 |
|---|---|
| Split | train=256, validation=0, test=0 |
| Families | filtered_random=86, multisine=85, sparse_aggregate=85 |
| Classes | Concrete=66, Marble=64, Ice=63, Sand=63 |

Class 수의 작은 차이는 12개 class/family stratum에 최대한 균등하게 배분한
결과이며 held-out family는 포함하지 않는다.

| Metric | Float32 | Full INT8 | INT8 delta |
|---|---:|---:|---:|
| Unseen-family test accuracy | 99.372% | 99.293% | -0.079 percentage points |
| Macro F1 | 99.375% | 99.297% | -0.078 percentage points |
| Concrete-Marble confusion | 0.938% | 1.094% | +0.156 percentage points |
| Prediction agreement | — | 99.607% | — |

INT8 family별 accuracy는 `smooth_random_patches` 99.53%,
`warped_multisine` 99.06%였다. INT8 recall은 Concrete 99.38%, Marble
98.44%였다.

모든 parity gate를 통과했다.

- accuracy delta가 -1 percentage point 이상
- macro-F1 delta가 -1 percentage point 이상
- Concrete-Marble confusion 증가가 +1 percentage point 이하

Exported model은 7,048 bytes이며 Flex 또는 floating-point tensor 없이 builtin
integer operator만 포함한다.

| Tensor | Interface | Scale | Zero point |
|---|---|---:|---:|
| Input | `int8 (1, 50, 10)` | 0.1095911264 | 22 |
| Output | `int8 (1, 4)` | 0.00390625 | -128 |

재현성 hash:

- Keras source: `a10f8fd28226e54c0476bf81124d15058b7e18c21774dcd7c0a684121cc17405`
- Noisy dataset: `ee84861570a1ae1244c54957219b9896c29a1756be97d4d02353f9699ac264ec`
- TFLite model: `f5d34f48b765d89f61cebe7e15eb9ee27a78b1c823950bca67043b6d4fa96df4`

## 재현 명령

```bash
cd simulation/unitree_mujoco/simulate_python
/d/shin/Infineon/Infineon_HIL/.venv/bin/python export_terrain_int8.py
```

Exporter는 비어 있지 않은 output directory를 덮어쓰지 않는다. Train-only
normalization 재계산, family ownership 검증, 정확한 calibration manifest 기록,
strict INT8 tensor/operator 검사, 전체 held-out test partition 평가를 수행한다.

## 남은 deployment gate

이 결과는 E84 compatibility 작업을 시작하기에 충분하지만 embedded
deployability를 입증하지는 않는다. 다음 항목이 남아 있다.

- DEEPCRAFT/TFLite importer 및 operator compatibility
- TFLite Micro arena와 peak activation/scratch RAM 실측
- Cortex-M55/Ethos-U55 flash placement 및 inference latency 실측
- Embedded normalization과 INT8 input quantization 재현
- UART framing과 Host-to-E84/E84-to-Host HIL timing
- Real FSR/BMI270 calibration 및 real-surface validation
