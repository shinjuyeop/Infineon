# 다음 Milestone: E84 Compatibility 및 On-device 검증

상태: **expanded dataset, 3-seed CNN gate, host INT8 parity 완료;
E84 compatibility가 다음 단계**.

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

Host parity gate는 모두 통과했다. 이는 E84 deployment 완료가 아니라 다음
compatibility 검증을 시작할 수 있다는 의미이다.

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
- [x] Host normalization metadata 고정; embedded 재현은 미완료
- [x] Split-safe representative INT8 calibration 정의
- [x] Strict INT8 conversion과 float/INT8 host parity 확인
- [ ] TFLite/LiteRT와 DEEPCRAFT importer/operator compatibility 확인
- [ ] TFLite Micro arena, activation, scratch RAM 실측
- [ ] Model flash placement와 inference latency 실측
- [ ] Cortex-M55/Ethos-U55 execution path와 tool version 확인
- [ ] UART framing, byte order, version, CRC, error handling 정의
- [ ] Host → E84 input transfer 구현 및 검증
- [ ] E84 → Host class/score transfer 구현 및 검증
- [ ] HIL buffering, window cadence, timeout, end-to-end timing 정의
- [ ] Real FSR/BMI270 orientation, gain, bias, range, noise calibration

## 다음 실행 순서

1. Local ignored artifact를 사용하기 전에 `EXPANDED_DATASET_V1_RESULTS.md`의
   Keras, noisy dataset, TFLite hash를 검증한다.
2. 7,048-byte strict INT8 model을 대상 DEEPCRAFT/TFLite Micro importer에 넣어
   supported operator와 tool version을 기록한다.
3. Cortex-M55/Ethos-U55 path에서 arena/scratch RAM과 inference latency를
   실측한다.
4. 측정된 resource/timing을 기준으로 UART와 HIL protocol을 설계한다.

Importer 또는 on-device gate가 실패하면 vibration model로 돌아가지 않는다.
먼저 unsupported operator, quantization convention, memory arena, preprocessing,
toolchain 문제를 분석한다.
