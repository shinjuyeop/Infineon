# E84 Deployment 및 Continuous HIL 검증

상태: **expanded dataset, 3-seed CNN gate, host INT8 parity 및
KIT_PSE84_AI E84 deployment/UART window HIL 완료; continuous sample-stream
HIL이 다음 단계**.

완료된 host pipeline:

- `expanded_terrain_dataset_v1.py`: bounded surface family 7개, deterministic
  split, leakage check, cost estimate
- `run_expanded_terrain_dataset_v1.py`: overwrite-safe expanded generator
- `terrain_cnn.py`, `train_terrain_1d_cnn.py`: 동일 architecture의
  FSR/IMU/Fusion ablation과 train-only normalization
- `terrain_int8.py`, `export_terrain_int8.py`: train-family calibration, strict
  INT8 export, unseen-family parity
- 관련 unit/integration test

기존 1,200-run pilot와 historical output은 변경하지 않았다. Expanded output은
`simulation/outputs/terrain_dataset_v1_expanded*`에 있으며 gitignored이다. 최종
host 결과는 `EXPANDED_DATASET_V1_RESULTS.md`에 기록한다.

## 완료된 Dataset 설계

### Surface-family allocation

| Split | Procedural family | Nominal spatial scale |
|---|---|---:|
| Train | `multisine` | 36–121 mm |
| Train | `filtered_random` | 80–250 mm |
| Train | `sparse_aggregate` | 50–120 mm |
| Validation | `crosshatch` | 58–121 mm |
| Validation | `rounded_ridges` | 58–363 mm including modulation |
| Test | `warped_multisine` | 36–363 mm including coordinate warp |
| Test | `smooth_random_patches` | 75–180 mm |

각 morphology는 deterministic, zero-centered, `[-1, 1]` bounded이다.
Terrain별 Dataset v1 peak-to-valley amplitude와 friction range는 변경하지
않았다. 10 mm hfield grid는 최소 36 mm component를 세 grid interval 이상으로
표현한다. 이 범위는 engineering morphology이며 measured material spectrum이
아니다.

기본 설계는 4 terrains x 7 families x 8 surfaces x 20 runs = 4,480
candidates이고 실제로 4,453 valid window를 얻었다.

### Leakage-safe split

- Surface family 전체를 train, validation, test 중 하나에만 할당
- Surface seed, session, run group도 하나의 split에만 유지
- Test family는 fitting, normalization, hyperparameter selection, INT8
  calibration에 사용하지 않음
- Automated family-level leakage와 balance test 적용

## 완료된 CNN ablation

세 input 모두 동일한 architecture family와 training protocol을 사용했다.

1. FSR4: channels 0–3
2. IMU6: channels 4–9
3. Fusion10: channels 0–9

```text
Input (50, C)
  → Conv1D(12, kernel=5)
  → Conv1D(16, kernel=3)
  → GlobalAveragePooling1D
  → Dense(4, softmax)
```

| Input | Parameters | Float parameter bytes | Estimated float activation working set |
|---|---:|---:|---:|
| FSR4 | 912 | 3,648 | 5,600 bytes |
| IMU6 | 1,032 | 4,128 | 5,600 bytes |
| Fusion10 | 1,272 | 5,088 | 5,600 bytes |

Activation estimate는 tensor liveness만 반영한다. TFLite arena metadata,
alignment, kernel scratch buffer를 포함한 deployment RAM 실측값이 아니다.

평가 항목:

- Overall accuracy와 macro F1
- Per-class precision, recall, F1
- Confusion matrix와 Concrete↔Marble mutual confusion
- Clean/noisy 및 FSR/IMU/Fusion 비교
- Pooled test와 family별 generalization
- RandomForest reference baseline

3-seed noisy Fusion은 98.95 +/- 0.40%였고 모든 seed에서 IMU-only보다
우수했다. 상세 결과는 `EXPANDED_DATASET_V1_RESULTS.md`를 참조한다.

## 완료된 INT8 host gate

Validation loss로 선택한 seed `20260809` noisy Fusion model을 사용했다.

- Calibration: train-family noisy window 256개
- Interface: `int8 (1, 50, 10) → int8 (1, 4)`
- Model size: 7,048 bytes
- Float/INT8 accuracy: 99.372% / 99.293%
- Prediction agreement: 99.607%
- Flex 및 floating-point tensor 없음

Host parity gate 이후 동일 artifact를 KIT_PSE84_AI에서 Cortex-M55 TFLM과
Ethos-U55-128 양쪽으로 실행했다. 고정 test tensor와 UART full-window HIL에서
Host/E84 raw INT8 output 및 class parity를 확인했다.

## Entry point

Expanded dataset dry-run:

```bash
cd simulation/unitree_mujoco/simulate_python
../../venv/bin/python run_expanded_terrain_dataset_v1.py
```

Full generation은 `--execute`를 명시해야 한다. 기존 non-empty output을
덮어쓰지 않는다.

CNN training:

```bash
/d/shin/Infineon/Infineon_HIL/.venv/bin/python train_terrain_1d_cnn.py
```

INT8 export/parity:

```bash
/d/shin/Infineon/Infineon_HIL/.venv/bin/python export_terrain_int8.py
```

## E84 readiness checklist

- [x] Current simulation deployment-candidate tensor shape, channel order, unit,
  sample timing 고정
- [x] Host normalization metadata 고정; PC HIL preprocessing에서 재현 확인
- [x] Split-safe representative INT8 calibration 정의
- [x] Strict INT8 conversion과 float/INT8 host parity 확인
- [x] TFLite/TFLM operator compatibility 확인; Vela 4.2.0 U55 전체 graph
  mapping, CPU fallback operator 0
- [x] TFLite Micro arena/scratch RAM 실측; CPU arena 3,236 B, U55 arena
  2,180 B, Vela scratch 1,600 B
- [x] Model 포함/flash placement와 inference latency 실측; raw model 7,048 B,
  CPU 약 564 us, U55 약 90 us one-shot
- [x] Cortex-M55 TFLM 및 Ethos-U55-128 execution path 실제 보드 확인
- [x] UART framing, little-endian length, version magic, CRC32, 기본 error
  handling 정의 (`TRN1`)
- [x] Host → E84 input transfer 구현 및 검증; 1 Mbaud KitProg3 UART로
  INT8 `(50,10)` 500-byte window 전송
- [x] E84 → Host class/raw output transfer 구현 및 검증; sample 878에서
  `[35,-35,-128,-128]`, class 0 Host parity
- [x] HIL buffering, window cadence, timeout, end-to-end timing 검증; E84
  50x10 ring buffer, 100 Hz/stride 1에서 1,000 samples와 951 inferences,
  drop/timeout/deadline miss 0, RTT 1.657 +/- 0.070 ms (p95 1.790 ms)
- [ ] Real FSR/BMI270 orientation, gain, bias, range, noise calibration
  (deferred; current Digital Twin/UART HIL milestone에는 불필요)

## 다음 실행 순서

1. MuJoCo live loop가 생성하는 physical-unit 10-channel sample을 검증된
   `TRN2` client API에 연결한다.
2. Digital Twin session/run boundary에서 sequence 0으로 E84 ring을 명시적으로
   reset한다.
3. Real FSR/BMI270 calibration은 실제 센서 milestone이 승인될 때까지
   deferred 상태로 유지한다.

Importer 또는 on-device gate가 실패하면 vibration model로 돌아가지 않는다.
먼저 unsupported operator, quantization convention, memory arena, preprocessing,
toolchain 문제를 분석한다.
