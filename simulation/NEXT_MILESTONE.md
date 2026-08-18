# Terrain Classification Milestone Status

## Terrain Transition v1 pilot

상태: **TERRAIN_TRANSITION_V1_PILOT_READY=true**. Fast Reflex v2의 frozen
dataset/model/INT8/E84/HIL 결과와 분리된 `terrain_transition_v1` physical-trace
foundation을 추가했다. 하나의 2 kHz MuJoCo run에서 qpos/qvel reset 없이 T0=650 ms에
두 adjacent ground geom의 contact profile을 동시에 temporal switch하고, native 1 kHz
Fusion10 및 v2의 frozen physical oracle을 전 구간에 저장한다.

초기 hard terrain은 Marble이며, pilot은 각 3회씩 A Marble→Ice, B Marble→Sand,
C Ice→Marble, D Sand→Marble을 생성했다. T0는 elapsed native-sensor tick 650으로
manifest에 보존되며 `transition_traces.npz`의 `sample_time_s`가 array time의
authoritative source다. 다음 단계는 이 raw trace를 기존 terrain normalization과
sliding classifier에 연결해 T1 stable terrain detection을 계산하고, frozen Fast
Reflex v2 replay로 T3를 계산하여 T0/T1/T2/T3 timeline을 완성하는 것이다.

## Terrain Transition v1 AI replay

상태: **TERRAIN_TRANSITION_AI_REPLAY_COMPLETE=true**. Frozen strict-INT8
`fast1000` Terrain classifier (50 samples / 1 kHz / Fusion10)와 frozen v2
strict-INT8 Slip 5 ms 및 Sink 20 ms detector를 12개 continuous trace에 read-only
replay했다. T1은 새 terrain class가 3개의 1 ms causal endpoint에서 연속된 세 번째
endpoint이며, T3는 detector의 raw INT8 threshold와 frozen persistence endpoint다.

A는 Terrain T1-T0 median 43 ms, Slip T2-T0 median 69 ms, T3-T2 median 5 ms였다.
B는 각각 2 ms, 19 ms, 42 ms였다. A/B 모두 hazard-positive run에서 T1이 T2보다
앞섰고 early firing은 없었다. 단 A non-hazard run 하나는 detector firing을 보여
transition-integration false firing으로 별도 기록했다. C Ice→Marble은 2/3만 stable
Marble에 도달했고 pre/post static-window class occupancy도 낮아,
**TERRAIN_MODEL_CHANGE_RECOMMENDED=true**다. 다음 최소 개선은 architecture 변경이
아니라 frozen static training distribution을 보완하는 transition-aware 1 kHz
windows와 direction-balanced continuous traces 추가다.

## Transition-Aware Terrain Classifier v2

336개의 새 continuous transition run(방향별 84; train/validation/test
=144/96/96)을 기존 static 1 kHz dataset과 source-run/family-disjoint하게
결합했다. 동일 Conv1D architecture의 transition-aware candidate는 fresh
transition test에서 네 방향 모두 stable detection 100%를 달성했고, diagnostic
Ice→Marble도 2/3에서 3/3, occupancy 38.7%에서 75.6%로 개선됐다. 다만 strict INT8
static test accuracy는 95.76%로 기존 97.10%보다 1.33pp 낮아 predeclared 1.0pp
retention gate를 통과하지 못했다. 따라서 `TERRAIN_TRANSITION_AWARE_V2_READY=false`,
`TERRAIN_ARCHITECTURE_CHANGE_NEEDED=false`: architecture가 아니라 static/transition
sampling mixture와 static retention을 다음 data-level iteration에서 조정해야 한다.

## Transition-Aware Terrain Classifier v2.1 mixture result

Static-heavy mixtures (75:25, 77.5:22.5, and 80:20) and one bounded
static-Marble contribution correction were evaluated with the endpoint label,
Fusion10, 1 kHz / 50 ms window, T1 persistence 3, train-only normalization,
and train-only strict-INT8 calibration unchanged. No mixture satisfied both
the 96.098% static INT8 retention gate and the frozen per-direction transition
gate. In particular, Ice→Marble (Case C) occupancy remained below 80%; the
77.5:22.5 + Marble 1.10 correction reached 79.75% in Float validation but
only 74.83% after INT8 and also missed static retention. Existing fresh
transition tests and the 12-run diagnostic were excluded from these choices.

상태: **TERRAIN_ARCHITECTURE_CHANGE_NEEDED=true**. This is evidence for a
minimal temporal-aggregation change, not a larger CNN, RNN, Transformer, or a
window/sample-rate change.

## Terrain v3 endpoint-aware aggregation ablation

Terrain v3 freezes the v2.1 77.5:22.5 data mixture (1,916 static and 555
transition training windows; no global inverse-frequency weighting) and uses
the existing family/source-run-disjoint transition validation partition as an
architecture-selection reservation. The transition test partition and the
12-run diagnostic are not read for selection.

The Conv1D(12,k=5) → Conv1D(16,k=3) front end is unchanged. Three equal-cost
aggregation heads were compared across seeds 20260821/22/23: full GAP,
last-step feature, and recent-8-ms average. The recent-8-ms head reached
Float Case-C occupancy 84.64% and strict-INT8 occupancy 84.72% with stable
detection in all directions, validating the endpoint-aware aggregation
hypothesis. However, its selected strict-INT8 static accuracy was 92.627%,
below the 96.098% retention gate. It is therefore not frozen, and no new fresh
transition test or diagnostic replay was opened.

상태: **TERRAIN_ENDPOINT_AWARE_ARCHITECTURE_READY=false**,
**TERRAIN_ENDPOINT_AWARE_INT8_READY=false**,
**TERRAIN_LARGER_ARCHITECTURE_NEEDED=true**. The next milestone should first
separate the observed static-domain regression from aggregation choice (for
example a leakage-safe static/domain training reservation or a causal
front-end/window study), before considering a materially larger model.

## Fast Reflex v2 E84/U55 audit

Separate frozen-artifact Vela outputs use the existing E84 fast1000 command:
U55-128, Performance, Sram_Only, arena cache 2936012, and Vela's default
High_End_Embedded system. Slip has one CPU `Max` fallback and must prove its
mixed delegate path with a fixed board golden test; Sink is fully NPU delegated.
No flash, board inference, or HIL was performed.

## Fast Reflex v2 validation selection

The 225-run Fast Reflex v2 dataset and seven frozen user-trained detector
artifacts are ready for a validation-only threshold/persistence sweep. The
selection tool evaluates Slip 5/10 ms and Sink 10/20/30 ms using only the
crosshatch and rounded-ridges validation families. It freezes a candidate only
when overall causal run FPR is at most 5%, then ranks recall, p95 latency, and
shorter window. Final test remains sealed until both selected JSON artifacts
report a valid candidate; no final-test data may be materialized or inferred.

The next approved user action is only the frozen final-host one-shot command.
Slip is fixed at 5 ms / threshold 0.971921742 / persistence 3; Sink at 20 ms /
threshold 0.983607300 / persistence 1. Preflight verifies SHA256 integrity,
selection values, `FINAL_TEST_READY=true`, and nonexistent final output before
any materialization. The final test remains untouched until that explicit user
command; it is not part of this preparation milestone.

The original reserved final command encountered an output-directory creation
defect after simulation but before any data, marker, inference, or metrics were
persisted. Its reservation is retired and must not be repeated. The replacement
one-shot reservation is indices 9110--9114, session 20260903, and excitation
offset 921000; it keeps all frozen detector values unchanged and creates the
empty final directory before simulation so a partial run remains fail-closed.

## Fast Reflex v2 strict INT8 preparation

Strict INT8 TFLite artifacts were converted from the frozen Float Slip 5 ms and
Sink 20 ms models using only deterministic train representative windows. Their
validation-only Float↔INT8 endpoint/replay parity passed, so `VELA_READY=true`.
Final-test artifacts were not read for INT8 inference. The next approval is
Vela optimization for Ethos-U55-128 only; E84 build/flash/inference and HIL
remain out of scope.

상태: **expanded dataset, host INT8 parity, KIT_PSE84_AI E84 deployment,
TRN1 full-window HIL, TRN2 continuous sample-stream HIL 및 실제 MuJoCo
live-loop → physical E84 integration, native 1 kHz/50 ms feasibility 및
fast1000 E84/U55 fixed regression 완료**.

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

## 완료된 MuJoCo live HIL gate

기존 `run_window()`의 0.5 ms physics loop에서 매 20 step마다 생성되는 clean
physical-unit FSR4 + left-foot/ankle IMU6 sample을 callback으로 받아 100 Hz
`TRN2`로 실제 probe `13070E98012D2400`에 전송했다. 각 terrain/run 시작은
sequence 0으로 E84 ring을 reset한다. Dataset v1의 sensor imperfection은
50-sample tensor 추출 후 적용되는 offline augmentation이므로 live 값에
재적용하지 않았다.

`multisine`, surface 0, run 0의 concrete/marble/ice/sand 네 deterministic
run에서 총 400 live samples와 204 sliding-window inference를 확인했다.

| Gate | 결과 |
|---|---:|
| Host reference INT8 ↔ E84 raw/class exact parity | 204 / 204 |
| Sequence drop / timeout / device error / deadline miss | 0 / 0 / 0 / 0 |
| Send period | mean 10.001 ms, p95 10.163 ms |
| UART request/response RTT | mean 1.574 ms, p95 1.724 ms |
| CPU cycles, inferred | mean 5,836.2, p95 5,847 |
| NPU cycles, inferred | mean 7,144.1, p95 7,160 |
| `[0.15,0.65)` aligned ground-truth accuracy | 4 / 4 |
| 모든 exploratory sliding window accuracy | 78.43% |

Host optimized XNNPACK kernel은 일부 arbitrary input에서 E84와 raw output이
몇 LSB 달랐으므로 exact transport gate는 같은 canonical TFLite의 reference
INT8 kernels를 사용한다. Class는 바뀌지 않았지만 raw parity를 느슨하게
처리하지 않았다. Timing benchmark는 headless이며 GUI는 demonstration
옵션이다. 78.43%는 training distribution 밖의 arbitrary continuous window를
포함하므로 Dataset-v1 99.29% test accuracy와 같은 의미로 해석하지 않는다.

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
- [x] 실제 MuJoCo 2 kHz loop → 100 Hz TRN2 → physical E84/U55 live HIL;
  4 terrains, 400 samples, 204 inference, Host/E84 exact parity 100%,
  drop/timeout/error/deadline miss 0
- [ ] Real FSR/BMI270 orientation, gain, bias, range, noise calibration
  (deferred; current Digital Twin/UART HIL milestone에는 불필요)

## 현재 우선순위

### Primary: Terrain Transition Fast Reflex HIL

별도 `terrain_fast_reflex_v1` pipeline에서 Marble→Ice slip과 Marble→Sand
sink/tilt onset을 연구한다. 기존 4-class classifier와 deployment artifact는
유지한다.

- [x] continuous qpos/qvel temporal contact-parameter switch
- [x] native 1 kHz Fusion10 `[-50,+100) ms` transition trace
- [x] train-normal-calibrated slip/sink/tilt MuJoCo oracle ground truth
- [x] transition/hazard aligned 1/2/5/10/15/20/30/50 ms prefix export
- [x] HIL replay artifact와 6-run smoke/diagnostic plot
- [x] 사용자 pilot 36/full 420-run generation과 physical validity 검토
- [x] terrain-independent physical hazard metadata revision 2와 split별 separation
- [x] leakage-safe Float Host detector training/evaluation pipeline 및 1-epoch smoke
- [x] 사용자 full 5/10/15/20/30/50 ms 학습과 validation-only failure diagnosis
- [x] Slip 5 ms GAP/Max/GAP+Max 최소 ablation; GAP+Max validation gate 통과
- [x] Slip validation candidate의 one-shot 고정 test 평가; recall 98.46%,
  pre-onset run FPR 7.50%로 **Host Final Gate FAIL**, test 기반 재튜닝 없음
- [x] Sink/Tilt Tilt-only 15-run validation audit, sensor observability와
  physical rule/logistic/combined replay; test 미사용, Sand gate 미달
- [x] Fast Reflex v2 experiment planning: frozen Slip pre-onset physical audit,
  fresh final-holdout policy, Sand sustained-hazard/failure-mode/boundary-step
  protocol; no new dataset, training, or test evaluation
- [x] Fast Reflex v2 schema/dataset foundation and train-only six-mode smoke:
  two-pass train-normal labels, fail-closed final-test reservation, and real
  Marble/Sand front/rear plus left/right boundary geoms; full generation pending
- [x] Fast Reflex v2 bounded train-only scenario-physics calibration: explicit
  config serialization, Marble-to-Ice slip mode, vertical-load/seam sweeps,
  coverage/selection artifacts, and preserved non-ready physical outcome;
  thresholds, v1 records, final-test rows, training, and INT8/E84 untouched
- [ ] Fast Reflex v2 approved bounded redesign for loaded front/rear tilt
  coverage, then a fresh train-only calibration; do not generate a pilot until
  `scenario_selection.json` reports `pilot_ready=true`
- [x] Fast Reflex v2 front/rear pitch-torque rejection audit: orientation was
  produced only by reducing loaded contact, so the seven-run train-only design
  is isolated from defaults and does not authorize pilot generation
- [x] Fast Reflex v2 localized-compliance front/rear rejection audit: bilateral
  contact was retained, but soft-rear rotation was too small and soft-front
  Tilt was contaminated by preceding Sink; left/right was correctly not swept
- [x] Fast Reflex v2 terminal bounded hard-backed-layer/height audit:
  `SAND_TILT_PHYSICAL_DESIGN_REJECTED`; isolated Sand Tilt-only is not
  physically validated in the current Digital Twin and must not drive pilot or
  deployment scope
- [x] Fast Reflex v2 deployment-scope decision: Slip Risk and Sand Sink Hazard
  only; isolated Sand Tilt-only excluded as a bounded Digital Twin limitation
- [x] Fast Reflex v2 pre-pilot target-policy freeze: Slip Hazard is confirmed
  slip and Incipient is diagnostic-only; revised train-only scope gate passes
  without changing a threshold, label, or historical artifact
- [x] Fast Reflex v2 100-row final-scope physical pilot: shared frozen-policy
  audit PASS; legacy generation summary gate was diagnostic/tilt based and is
  preserved only as historical output
- [x] Fast Reflex v2 target-centred 225-run physical dataset: frozen-scope
  audit PASS; final test sealed
- [x] Fast Reflex v2 causal detector-dataset foundation and 1-epoch small
  smoke: confirmed-Slip/Sustained-Sink endpoint targets, train-only
  normalization, causal windows, model save/load, and replay foundation
- [ ] User-run Fast Reflex v2 validation-only detector training: Slip 5/10/20
  ms and Sink 10/20/30/50 ms, threshold/persistence frozen only after a
  separately approved validation analysis; final test remains sealed
- [ ] 후속 승인 뒤 selected detector INT8/E84, reflex rule 및 E2E 측정

다음 milestone은 frozen Slip 결과를 그대로 종료 상태로 보존하고, Sand v2를
별도 experiment로 설계하는 것이다. 우선 Tilt-only 경계의 physical significance와
run-level로 지속되는 FSR spatial signal을 재검토하고, 필요하면 label/task를
사전에 재정의한 새 train/validation protocol을 만든다. 현재 결과로 INT8/E84나
firmware 단계로 이동하지 않는다.

Fast Layer KPI는 transition→hazard detection `<50 ms`, 우선 연구 목표는
`<=20 ms` observation 가능성이다. Reflex command 변경 `<30 ms`는 아직
측정하지 않는다. 상세 protocol과 실행 명령은
`TERRAIN_FAST_REFLEX_V1.md`를 참조한다.

### Deferred

- Ice low-friction locomotion-policy 재학습
- Full walking touchdown classifier generation/training
- 1 kHz asynchronous E84 transport와 reflex firmware/control rule

기존 Walking Touchdown foundation은 삭제하지 않고 후속 locomotion/sim-to-real
고도화 작업으로 보존한다.

## 이전 milestone

### Native 1 kHz / 50 ms terrain classification feasibility

- [x] 2 kHz MuJoCo physics에서 2 step마다 native 1 kHz sensor sample 취득
- [x] `pulse_onset_rate_ablation [0.25,0.30)`의 `(50,10)` window 생성
- [x] 4 terrains × 7 families full dataset 및 split/leakage/statistics 검증
- [x] 동일 compact CNN architecture의 3-seed 평가와 train-only normalization
- [x] strict full-INT8 host gate와 100 Hz/500 ms baseline 비교
- [x] 별도 `TERRAIN_FAST1000_CPU`/`TERRAIN_FAST1000_U55` artifact 및 E84 fixed
  tensor/Host-golden exact parity

### Walking Touchdown-Aligned Terrain Classification (진행 중)

핵심 연구 질문:

> 1 kHz FSR4 + left-foot IMU6를 이용할 때, 보행 중 touchdown 이후 최소 몇
> ms의 관측으로 terrain을 안정적으로 분류할 수 있는가?

현재 단계는 full walking dataset이나 accuracy 결과가 아니라 acquisition
foundation이다. MuJoCo left-foot AIR→CONTACT를 canonical touchdown으로 삼아
`[-10,+50) ms` physical-unit trajectory를 한 번 저장하고, 후속 단계에서 같은
event의 `t >= 0` prefix를 재사용한다.

- candidate observation: **1 / 2 / 5 / 10 / 15 / 20 / 30 / 50 ms**
- 주력 후보: 5 / 10 / 15 / 20 / 30 / 50 ms
- 우선 목표: **touchdown 이후 ≤ 20 ms**
- event random split 금지; surface family/realization, session, run ownership 유지
- terrain label은 몸 위치가 아니라 touchdown 순간 왼발이 접촉한 terrain
- 1/2 ms는 lower-bound exploratory candidate이며 아직 accuracy 결과 없음

Concrete/Marble/Ice/Sand pilot은 31 events 중 16 valid였다. 후속 contact audit는
원본 XML을 유지한 채 terrain support를 명시적인 sole sphere로 격리했다.
Sand native soft contact의 수직 침하 원인은 `solref`였으며, native `solimp`와
friction/roughness를 유지하고 `solref=(0.015,1.0)`만 적용한
`walking-support-v1`에서는 0.10/0.15/0.20 m/s가 모두 2/2 통과했다.

Ice는 전진 추종을 포함한 최종 gate에서 모든 속도 0/2였다. Upstream policy의
학습 foot-friction 범위 `[0.3,1.6]`에 비해 Ice domain `[0.03,0.12]`는 명백한
OOD이다. 다음 gate는 full generation이 아니라 low-friction domain randomization
정책의 학습·export와 동일 12-run sweep 재통과다. 그 전에는 walking CNN
training, INT8 export 또는 E84 deployment를 완료 처리하지 않는다.

### 그 밖의 후속 작업

#### Prepared: Fast Reflex v2 Sink continuous HIL

Use `FAST_REFLEX_SINK_V2_U55` with `TERRAIN_MODE=frv2_sink_hil` and the
`tools/fast_reflex_sink_hil.py` client.  Its six deterministic source traces
are validation-only and oracle-labelled (`sustained_sink`, never scenario
names); final-test identifiers are fail-closed.  This is preparation and local
firmware-build scope only, not a board HIL PASS claim.  The next action is a
user-run Sink flash and smoke/validation replay, followed by parity/timing gate
review before any Slip continuous-HIL work.

#### Complete: Fast Reflex v2 E84 continuous HIL

Sink and Slip continuous replay completed on the authoritative ACM1 board.
The current deployment scope is complete: no further detector tuning or HIL
work is required. Preserve final-host and HIL results as separate gates.

1. 1 kHz에 맞는 batched/asynchronous UART와 E84 ring-buffer cadence를 별도
   milestone로 설계한다. 기존 TRN1/TRN2 100 Hz path는 유지한다.
2. Async transport 전에는 fixed regression 결과를 real-time 1 kHz streaming
   evidence로 해석하지 않는다.
3. Real FSR/BMI270 calibration은 실제 센서 milestone이 승인될 때까지
   deferred 상태로 유지한다.

Importer 또는 on-device gate가 실패하면 vibration model로 돌아가지 않는다.
먼저 unsupported operator, quantization convention, memory arena, preprocessing,
toolchain 문제를 분석한다.
