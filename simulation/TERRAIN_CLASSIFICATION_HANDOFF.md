# Terrain Classification Handoff

MuJoCo terrain-classification 작업의 canonical reset handoff 문서이다. 다음
작업 전 `TERRAIN_DATASET_V1.md`, `EXPANDED_DATASET_V1_RESULTS.md`,
`NEXT_MILESTONE.md`와 함께 읽는다.

## 현재 pipeline

```text
Terrain parameters and procedural surface
                    ↓
            MuJoCo full-body G1
                    ↓
      FSR4 + left-foot/ankle IMU6
                    ↓
    domain variation + sensor imperfections
                    ↓
 canonical baseline: medium_response [0.15, 0.65) at 100 Hz
                    ↓
                  (50, 10)
                    ↓
 surface/session-disjoint train/validation/test
                    ↓
 strict INT8 classifier → KIT_PSE84_AI CM55/U55
```

Low-frequency Dataset v1 dynamics의 operational MuJoCo timestep은 0.5 ms
(2 kHz)이다. Raw high-frequency contact vibration이 수렴했다는 의미는 아니다.

Sampling-rate/observation-window ablation은 이 baseline을 보존하고 native
500 Hz/100 ms와 1000 Hz/50 ms를 비교했다. Full 3-seed와 strict INT8 host
gate를 통과한 fast candidate는 1000 Hz, 50 samples, 50 ms이며 상세 결과는
`TERRAIN_RATE_ABLATION.md`에 있다. 이후 별도 `TERRAIN_FAST1000_CPU` 및
`TERRAIN_FAST1000_U55` artifact를 생성했고 E84/U55 fixed regression과 Host
golden HIL parity를 통과했다.

## 설계 결정

### LOCKED / CURRENT

- AI channel: `foot_force_1..4`, accelerometer XYZ, gyroscope XYZ
- IMU site: `left_ankle_roll_link`에 rigidly attached된 left foot/ankle
- Class: concrete=0, marble=1, ice=2, sand=3
- Canonical window: `medium_response`, `[0.15, 0.65)`
- Input shape: `(50, 10)` at 100 Hz
- Split unit: surface family/seed, session, run group; random window split 금지
- Test data는 training에 사용하지 않은 surface family/realization 사용
- MuJoCo friction, slip, load distribution, CoP-related response, surface
  geometry response, low-frequency foot dynamics 사용

위 100 Hz 항목은 canonical deployed baseline으로 계속 LOCKED이다. Fast host
candidate는 별도 artifact로 `pulse_onset_rate_ablation [0.25,0.30)` at 1000 Hz,
`(50,10)`을 사용한다.

### DIAGNOSTIC-ONLY

- Pelvis IMU (`imu_acc`, `imu_gyro`)
- Pelvis/foot velocity, slip, tangential/normal contact force, collision 및
  contact-validity trace
- Clean raw simulation CSV와 high-rate convergence artifact

### DEFERRED / OPTIONAL

- 별도의 high-frequency vibration sensor model
- Exact PSD, dominant-frequency, micro-contact spectral feature
- Gait-based dataset과 locomotion-controller integration
- Real FSR/BMI270 gain, bias, orientation, bandwidth, noise calibration

### SUPERSEDED FOR CURRENT DATASET

- Pelvis IMU를 AI 입력으로 사용
- Raw MuJoCo high-frequency vibration을 classifier evidence로 사용
- 50/100/200 Hz logging 차이를 physics convergence로 해석
- Legacy synthetic `(50, 5)`와 MuJoCo `(50, 10)` 혼합

`Infineon_HIL` synthetic pipeline은 기존 host, quantization, HIL 작업을 위해
보존한다. Dataset v1과 자동 변환하거나 결합하지 않는다.

## Dataset v1 pilot 결과

| Metric | 결과 |
|---|---:|
| Candidate / valid | 1,200 / 1,189 |
| Clean/noisy tensor | `(1189, 50, 10)` |
| FSR-only test accuracy | 84.81% |
| IMU-only test accuracy | 97.47% |
| Fusion test accuracy | 96.20% |
| Concrete recall, fusion | 96.7% |
| Marble recall, fusion | 90.0% |
| Concrete-Marble mutual confusion | 5.8% |

Pilot은 하나의 procedural surface family 안에서 unseen-surface를 평가했다.
Fusion은 FSR-only보다 우수하지만 IMU-only보다 우수하다고 결론낼 수 없었다.

## Expanded milestone 현재 상태

기존 pilot와 historical experiment를 변경하지 않고 다음을 완료했다.

- Train 3 / validation 2 / test 2의 procedural surface family 7개
- 4,480-candidate dry-run-first generator와 cost estimate
- FSR4, IMU6, Fusion10에 동일한 compact Conv1D architecture
- Train-family-only normalization과 pooled/per-family evaluation
- Leakage, balance, morphology, resource, CNN/INT8 test

Expanded dataset은 4,453 valid window를 생성했다. 3-seed noisy Fusion accuracy는
98.95 +/- 0.40%였다. Validation 기준으로 선택한 seed의 strict full-INT8 host
accuracy는 99.29%였고 float 대비 delta는 -0.079 percentage point였다. 전체
근거는 `EXPANDED_DATASET_V1_RESULTS.md`, E84 deployment 및 continuous HIL
결과와 남은 live-integration 항목은 `NEXT_MILESTONE.md`에 기록한다.

## E84 deployment gate

7,048-byte strict INT8 noisy Fusion model은 실제 KIT_PSE84_AI에서
Cortex-M55 TFLM 및 Vela 4.2.0 Ethos-U55-128 실행을 통과했다. U55 CPU
fallback operator는 0이며, 고정 tensor와 1 Mbaud UART `TRN1` full-window
HIL 모두 Host와 raw output/class가 일치했다. CPU는 약 564 us, U55는 약
90 us one-shot이었고 상세 재현 절차와 arena/scratch 실측값은
`../Infineon_test/projects/terrain_e84_deploy/deployment/README.md`에 기록한다.
이후 별도 `TRN2` sample protocol과 E84 50x10 ring buffer를 실제 보드에서
100 Hz, stride 1, 1,000 samples로 검증했다. 951 inference 동안
drop/timeout/deadline miss는 0이었고 전체 request/response RTT는 평균
1.657 ms, p95 1.790 ms였다. Real sensor calibration만 current Digital
Twin/UART HIL 범위 밖의 deferred 항목으로 남는다.

이 gate는 기존 100 Hz model에 대한 evidence이다. 1000 Hz selected strict INT8은
별도 deployment path에서 Vela U55-128 변환, E84 fixed regression 및 Host golden
HIL exact parity를 통과했다. Board raw output은 `[114,-114,-128,-128]`, class 0이며
fixed/HIL 모두 host와 일치했다. 현재 synchronous TRN2 RTT 1.5–1.7 ms는 1 kHz
sample period 1 ms보다 길어 async 1 kHz transport는 여전히 별도 과제다.

동일한 `TRN2` endpoint를 기존 Dataset-v1 `run_window()`에 최소 callback으로
연결했다. 0.5 ms physics step 20회마다 `G1HilSensorReader.read_vector()`가
생성한 clean physical-unit FSR4 + ankle IMU6 sample 한 개를 보낸다. 네
terrain의 deterministic `multisine` run에서 400 samples, 204 inference,
Host reference INT8/E84 raw 및 class parity 204/204, deadline/drop/timeout/error
0을 실제 probe `13070E98012D2400`에서 확인했다. `[0.15,0.65)`에 정확히
정렬된 네 window는 4/4였지만 모든 exploratory sliding window accuracy는
78.43%였다. 후자는 arbitrary continuous trajectory 결과이며 canonical
Dataset-v1 test accuracy를 대체하지 않는다.

## 실험 이력

| 실험 | 목적 | 결론 | 현재 반영 | 폐기/보류 |
|---|---|---|---|---|
| Passive terrain | 초기 4-terrain contact response 확인 | Unsupported/passive run에서 force/acceleration instability와 weak hard-surface separation 확인 | Terrain profile, failure check | Passive data를 training baseline으로 사용하지 않음 |
| Controlled excitation | Matched initial condition과 70% support로 비교 안정화 | 심한 passive artifact 제거, concrete-marble은 여전히 어려움 | Matched seed, support, validity/outlier rule | 좁은 deterministic dataset |
| Horizontal pulse | Controlled friction/slip excitation 추가 | 80 N pulse가 slip ordering을 드러냈지만 full-window separation은 개선하지 못함 | 80 N-class pulse, slip/contact diagnostic | Full-window separation target |
| Bidirectional validation | +X/-X asymmetry와 response window 검증 | Slip ordering은 direction-consistent, `medium_response`가 가장 실용적 | 양방향 pulse, canonical medium window | One-direction training, full window |
| Lower-body validation | Mass/COM/inertia와 contact layout을 보존한 reduced model 평가 | Research option으로 유효하지만 production은 full-body 유지 | Validation utility와 reference model | Dataset v1은 full-body G1 유지 |
| Foot IMU A/B | Pelvis와 foot/ankle IMU 비교 | Flat 50 Hz에서 aggregate separation은 개선하지 않았지만 Accel-X sensitivity 증가 | Foot IMU를 AI 입력으로 고정 | Pelvis IMU AI 입력 |
| Surface/sampling study | Flat/surface-aware와 50/200 Hz logging 비교 | Roughness와 logging rate가 separation에 영향, spectral peak는 physics 미검증 | Native hfield와 low-frequency response | 200 Hz peak를 vibration feature로 사용 |
| Friction x roughness | Material factor 분리 | FSR은 roughness-dominant, foot IMU는 friction/slip-dominant | Sensor complementarity와 domain variation | Measured material physics 주장 |
| Timestep convergence | 5/2/1 ms와 common-band vibration audit | 기존 60–70 Hz behavior는 timestep-sensitive | Low-frequency와 raw vibration 분리 | 5 ms/200 Hz spectral conclusion |
| Final vibration limitation | 1/0.5 ms와 0.5/0.25 ms 비교 | Aggregate force/slip은 충분히 수렴, waveform/PSD와 micro-contact는 미수렴 | Dataset v1은 0.5 ms 사용 | 추가 timestep 탐색과 raw high-frequency PSD |
| Dataset v1 pilot | Domain variation, noise, leakage-safe split, classifier evidence | 1,189 valid; IMU 97.47%, Fusion 96.20% | `(50,10)` pilot baseline | Fusion 우위 주장 |
| Expanded Dataset v1 | Unseen surface-family generalization 및 CNN ablation | 4,453 valid; 3-seed Fusion 98.95%; INT8 parity 통과 | 현재 deployment candidate | Real-world generalization 주장 |

## Repository map

### 현재 source

- `terrain_dataset_v1.py`: pilot schema, variation, sensor model, split check,
  RandomForest evaluation
- `expanded_terrain_dataset_v1.py`: surface-family design, manifest, cost model
- `run_expanded_terrain_dataset_v1.py`: expanded generator
- `terrain_cnn.py`, `train_terrain_1d_cnn.py`: CNN ablation과 normalization
- `terrain_int8.py`, `export_terrain_int8.py`: strict INT8 export와 parity
- `prepare_rate_ablation_common_valid.py`: rate간 동일 valid-run subset 정렬
- `hil_sensor.py`, `terrain_profiles.py`, `surface_profiles.py`,
  `controlled_excitation.py`, `pulse_windows.py`: simulation foundation
- `run_live_terrain_hil.py`: existing runner callback을 사용하는 physical E84
  live adapter, Host shadow parity, timing/CSV logger
- `terrain_e84_deploy/tools/terrain_preprocessing.py`, `terrain_shadow.py`,
  `terrain_stream_client.py`: shared preprocessing, reference INT8 shadow,
  reusable `TerrainStreamLink`

### Test

- `test/test_terrain_dataset_v1.py`, `test/test_expanded_terrain_dataset_v1.py`
- `test/test_terrain_cnn.py`, `test/test_terrain_int8.py`
- 기존 sensor/surface/timestep regression test
- `gamepad_test.py`, `hil_sensor_test.py`는 automated unittest가 아닌 manual
  hardware-oriented script

### Generated output

- `simulation/outputs/` 전체는 generated이며 gitignored
- Dataset, Keras, TFLite artifact는 output 아래에만 존재
- Historical result는 삭제하거나 current dataset과 혼합하지 않음
- Rate ablation output은 `terrain_dataset_v1_expanded_rate_quick_*` 및
  `terrain_dataset_v1_expanded_1000hz_*`로 100 Hz artifact와 분리

## Code-health 참고

- Historical runner 일부가 current utility를 제공하므로 임의 삭제하지 않는다.
- `analyze_terrain_data.py`의 pelvis label은 초기 passive experiment 기록이다.
- CSV/separation helper 중복은 향후 parity test를 준비한 뒤 통합한다.
- Dataset dependency는 `requirements-dataset-v1.txt`, CNN dependency는
  `requirements-cnn.txt`에 있다.

## Reset recovery

Full reset 후 이 문서, `EXPANDED_DATASET_V1_RESULTS.md`,
`NEXT_MILESTONE.md`, `CLEANUP_AUDIT.md` 순서로 읽는다. Model/contact
representation을 의도적으로 바꾸지 않는 한 historical physics study를 다시
실행하지 않는다.
