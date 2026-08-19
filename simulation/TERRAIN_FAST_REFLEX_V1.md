# Terrain Transition Fast Reflex v1 (Historical Foundation)

상태: **continuous-state temporal transition schema, physical hazard ground
truth, native 1 kHz trace/window export와 6-run smoke 완료; neural-network
training, INT8, E84 firmware 및 reflex rule 미실행**.

## 목적과 계층 분리

기존 Concrete/Marble/Ice/Sand 4-class classifier와 E84 TRN1/TRN2 artifact는
변경하지 않는다. Fast Reflex는 별도 `terrain_fast_reflex_v1` pipeline이며
질문은 terrain identity가 아니라 실제 slip 또는 sink/tilt onset이다.

```text
Fusion10 @ 1 kHz
  ├─ Fast Reflex: slip / sink / tilt hazard, 5..50 ms
  └─ 기존 Terrain Adaptation: 4-class terrain, next-step adaptation
```

AI 입력은 실제 E84에서 확보 가능한 다음 10개 physical-unit channel뿐이다.

```text
FSR1..4 [N], Accel XYZ [m/s^2], Gyro XYZ [rad/s]
```

MuJoCo-only contact `Fn/Ft`, `Ft/Fn`, foot world linear/angular velocity,
orientation, vertical position/depth는 `oracle_diagnostics.npz`에 분리하며 AI
입력에 포함하지 않는다.

## Transition scenario

`run_terrain_fast_reflex_v1.py`는 기존 full-body G1, 70% vertical elastic
support, 80 N/100 ms horizontal pulse와 explicit sole sphere contact를 재사용한다.
Physics는 2 kHz, sensor read는 매 2 physics step의 native 1 kHz이다.

- `t0=0.250 s`
- `transition_type=temporal_parameter_switch`
- `t0`에서 floor friction/`solref`/`solimp`만 pre profile에서 post profile로 교체
- qpos/qvel reset, signal concatenation, simulation restart 없음
- 전환 전 surface morphology/hfield는 run 전체에서 고정
- pulse도 `t0`에 시작하고 run parity에 따라 ±X 방향을 교대
- raw trace는 `[-50,+100) ms`, 150 samples를 한 번 저장

Scenario는 `marble_to_ice`, `marble_to_sand`, normal negative
`marble_to_marble`, `concrete_to_concrete`이다. Sand는 기존 compliance 기반
engineering approximation이며 plastic deformation 또는 material history를
모델링하지 않는다.

## Physical hazard ground truth

Threshold는 test family를 보지 않고 train-family normal transition의
`[0,+100) ms` pulse/contact distribution만 사용한다. 각 channel의 normal
99 percentile과 `median + 6*MAD` 중 큰 값을 upper threshold로 사용하고,
minimum load는 loaded normal-force distribution의 10 percentile이다. 조건은
3개의 연속 1 kHz sample에서 유지되어야 한다.

- Slip: left sole contact AND `Fn >= minimum_load` AND horizontal foot world
  speed가 normal-calibrated threshold 초과
- Sink: loaded contact AND transition foot height 대비 sink depth 및 downward
  velocity가 각각 normal threshold 초과
- Tilt: loaded contact AND transition 대비 foot roll/pitch vector 변화가 normal
  threshold 초과
- Fast combined Sand target: `sink_or_tilt = sink OR tilt`

`Ft/Fn`은 validation diagnostic으로 저장하지만 canonical slip label 조건에는
사용하지 않는다. Terrain이 Ice/Sand라는 사실 자체는 hazard label이 아니다.

각 manifest에는 transition, slip/sink/tilt/hazard onset timestamp를 별도로
보존한다. Transition-aligned prefix는 HIL injection latency, hazard-aligned
prefix는 detector signal separation용이다.

## Artifact schema

- `inputs_fusion10.npz`: `(runs,150,10)` AI input과 sample/relative timestamp
- `oracle_diagnostics.npz`: `(runs,150,16)` oracle와 slip/sink/tilt mask
- `manifest.csv`: scenario intent와 분리된 actual physical hazard type,
  transition/any-hazard/target-hazard onset, split ownership, validity
- `transition_aligned_XXms.npz`: 1/2/5/10/15/20/30/50 ms prefix
- `hazard_aligned_XXms.npz`: onset이 있고 trace tail이 충분한 prefix만 저장
- `hil_replay/*.npz`: sequence/timestamp/Fusion10 compact physical replay
- `window_separation.csv`: split별 model-free FSR/IMU centroid separation과
  onset coverage; physical-hazard가 없는 normal trace만 negative로 사용
- `protocol.json`, `summary.txt`: threshold provenance, KPI, run summary

Split은 기존 family ownership을 유지하며 surface realization/session/run을
event 단위로 나누지 않는다. Threshold calibration에는 train family normal만
사용하고 test family는 threshold/window 선택에 사용하지 않는다.

## Smoke result

Output:
`simulation/outputs/terrain_fast_reflex_v1_smoke_verified_20260812`

```text
multisine/train, surface 0
Marble→Ice 2 + Marble→Sand 2 + Marble→Marble 2 = 6 runs
physics=2000 Hz, sensor=1000 Hz, spacing=1 ms
valid=6/6, NaN/Inf=0
transition qpos/qvel max absolute delta=0/0 for all runs
normal oracle hazard=0/2
```

Normal-calibrated smoke thresholds:

| Threshold | Value |
|---|---:|
| Minimum load | 56.1666 N |
| Horizontal slip speed | 0.000864 m/s |
| Sink depth | 0.003606 mm |
| Downward speed | 0.001512 m/s |
| Tilt change | 0.000151 rad |

작은 threshold는 supported quasi-static normal trace의 variation에서 나온 PoC
값이며 실제 로봇 threshold가 아니다. Pilot/full의 더 넓은 train normal
distribution으로 다시 calibration되어야 한다.

| Scenario/run | Diagnostic onset | Transition→onset |
|---|---|---:|
| Marble→Ice r0 | slip | 28 ms |
| Marble→Ice r1 | slip | 11 ms |
| Marble→Sand r0 | sink and tilt | 28 ms |
| Marble→Sand r1 | sink and tilt | 30 ms |
| Marble→Marble r0/r1 | none | — |

Model-free separation score는 2+2 run의 작은 smoke statistic이므로 accuracy가
아니다. 20 ms transition window에서 Ice는 IMU 1.346 / FSR 1.580이고 onset
coverage는 1/2였다. Sand는 IMU 1.824 / FSR 1.656이지만 onset coverage는
0/2였다. 따라서 Fusion10 변화는 0–20 ms에도 관찰되지만, physical hazard가
아직 발생하지 않은 Sand trace를 조기 hazard로 간주할 수 없고 ≤20 ms detector
가능성은 pilot 전까지 미확정이다.

Diagnostic plot:

- `.../marble_to_ice_diagnostic.png`
- `.../marble_to_sand_diagnostic.png`

## User-run pilot and full generation

Pilot은 4 scenarios × 3 train families × 1 surface × 3 runs = **36 runs**다.
Smoke 실측 비율 기준 약 5–20초, compressed output 약 3–5 MiB를 예상한다.

```bash
cd /d/shin/Infineon/simulation/unitree_mujoco/simulate_python
../../venv/bin/python run_terrain_fast_reflex_v1.py \
  --execute \
  --output-dir ../../outputs/terrain_fast_reflex_v1_pilot \
  --scenarios marble_to_ice marble_to_sand marble_to_marble concrete_to_concrete \
  --families multisine filtered_random sparse_aggregate \
  --surfaces-per-family 1 --runs-per-surface 3 --plot
```

Full은 4 scenarios × 7 families × 3 surfaces × 5 runs = **420 runs**다.
Smoke 실측과 파일 수 overhead를 고려해 약 1–7분, compressed output 약
25–50 MiB를 예상한다.

```bash
cd /d/shin/Infineon/simulation/unitree_mujoco/simulate_python
../../venv/bin/python run_terrain_fast_reflex_v1.py \
  --execute \
  --output-dir ../../outputs/terrain_fast_reflex_v1_full \
  --scenarios marble_to_ice marble_to_sand marble_to_marble concrete_to_concrete \
  --families multisine filtered_random sparse_aggregate crosshatch rounded_ridges warped_multisine smooth_random_patches \
  --surfaces-per-family 3 --runs-per-surface 5 --plot
```

정상 종료 마지막 log 형식:

```text
terrain_fast_reflex_v1 schema_version=1
runs=420 valid=...
native_sampling=1000Hz physics=2000Hz spacing=1ms
thresholds={...}
marble_to_ice: runs=105 onsets=... latency_ms=...
marble_to_sand: runs=105 onsets=... latency_ms=...
marble_to_marble: runs=105 onsets=... latency_ms=none
concrete_to_concrete: runs=105 onsets=... latency_ms=none
```

향후 Fast Layer KPI는 transition→hazard detection `<50 ms`, 우선 연구 목표는
`<=20 ms` observation의 유효성이다. Hazard detection→reflex command `<30 ms`
E2E KPI는 이번 milestone에서 측정하지 않는다.

## Pilot/full generation과 derived revision 2

사용자가 36-run pilot과 420-run full generation을 완료했다. Raw Fusion10,
oracle, timestamp, split 및 transition state는 모두 유효했다. 최초 derived
artifact는 `expected_hazard=normal`인 normal-terrain scenario에서 실제
oracle slip/tilt가 발생해도 combined hazard metadata에서 숨기는 문제가 있었다.
Terrain/scenario 이름은 hazard label이 아니라는 원칙에 따라 revision 2에서
다음을 수정했다.

- `hazard_type`, `hazard_onset_time_s`: scenario와 무관한 실제
  `slip|sink|tilt` mask 기준
- `target_hazard_onset_time_s`: Ice slip/Sand sink-or-tilt 연구 target을 별도 보존
- hazard-aligned export: 모든 physical hazard에 동일 규칙 적용
- separation: train/validation/test별 계산, physical-hazard가 없는 normal만 negative

원본 simulation을 다시 실행하지 않고 보존된 raw trace에서 다음 authoritative
artifact를 만들었다.

- `simulation/outputs/terrain_fast_reflex_v1_pilot_corrected_v2`
- `simulation/outputs/terrain_fast_reflex_v1_full_corrected_v2`

Full revision 2 결과:

| Scenario | Runs | Physical hazard runs | Expected target onset |
|---|---:|---:|---:|
| Marble→Ice | 105 | 105 | slip 105 |
| Marble→Sand | 105 | 105 | sink-or-tilt 105 |
| Marble→Marble | 105 | slip 19 | n/a |
| Concrete→Concrete | 105 | tilt 16 | n/a |

Normal-terrain scenario의 35 physical-hazard run은 false positive가 아니라 실제
oracle-positive다. 실제 hazard가 없는 normal negative는 train 78,
validation 45, test 52, 총 175개다. Corrected full은 420/420 valid,
physical hazard 245개, hazard-aligned 245개이며 raw Fusion10/timestamp는 원본과
동일하다. 기존 non-corrected pilot/full의 derived manifest/window는 superseded다.

기존 output을 다시 변환해야 할 경우에만 다음을 사용한다. 현재 위 두 corrected
directory는 이미 생성되었으므로 다시 실행할 필요가 없다.

```bash
../../venv/bin/python repair_terrain_fast_reflex_v1.py \
  --input-dir ../../outputs/terrain_fast_reflex_v1_full \
  --output-dir ../../outputs/terrain_fast_reflex_v1_full_corrected_v2 --plot
```

## Float Host binary-detector pipeline (ready, full result pending)

`terrain_fast_reflex_detector_v1.py`와
`run_terrain_fast_reflex_detector_v1.py`가 corrected-v2 raw 150-sample trace에서
독립적인 `slip` 및 `sink_tilt` detector dataset을 만든다. Primary label은
미래 scenario intent가 아니라 각 sliding window `[t-L+1,t]` 끝점의 physical
oracle state다. 따라서 hazard onset 전 transition prefix는 음성이고 anticipation
결과와 detection 결과를 섞지 않는다. 기존 transition/hazard-aligned prefix
artifact는 audit/diagnostic에는 쓸 수 있지만 primary detector의 run label로
그대로 학습하지 않는다.

Canonical negative는 all-non-target-negative다. 즉 slip detector에는 slip이 없는
sink/tilt endpoint도 음성이고, sink/tilt detector에는 sink/tilt가 없는 slip
endpoint도 음성이다. 다른 hazard까지 없는 clean-negative-only FPR은 진단 열로만
기록한다. Train은 endpoint 0–99 ms를 2 ms stride로 subsample하고 run-balanced 뒤
binary class-balanced weight를 적용한다. Validation/test와 offline replay는 1 ms
stride다. 모든 derived window는 원 run의 family split을 그대로 유지하며,
normalization은 detector/window별 train window에서만 계산한다.

모든 5/10/15/20/30/50 ms 모델은 다음 동일 family를 사용한다.

```text
Input(L,10) -> Conv1D(12,k5,same,relu) -> Conv1D(16,k3,same,relu)
            -> GlobalAveragePooling1D -> Dense(1,sigmoid)
```

모델은 1,221 parameters이며 convolution+dense MAC은 `1,176*L+16`이다.
Decision threshold는 validation all-non-target FPR 5% 이하에서 recall 최대,
동률이면 더 낮은 FPR와 높은 threshold 순으로 고정한다. Test에서 재조정하지
않는다. Offline replay는 1 kHz, 3회 연속 positive의 세 번째 endpoint를 stable
detection으로 기록한다. transition→detection과 hazard onset→detection을 분리하고,
onset 전 firing은 별도 anticipation lead로 기록한다.

Threshold sensitivity는 test family를 제외하고 canonical p99/MAD6/3 samples
주변의 percentile 99.5/99.9, MAD 4/8, confirmation 2/5를 비교한다. Train+validation
target coverage는 Ice 98.67–100%, Sand 100%, median onset은 Ice 26–28 ms와 Sand
19 ms로 유지됐다. Ice p95는 54.3–68.7 ms로 tail sensitivity가 있으나 target
존재/중앙 onset 결론은 뒤집히지 않아 revision-2 canonical threshold를 유지한다.

1-epoch smoke는 corrected-v2의 scenario/split당 2 runs, 두 detector의 10 ms
window로 통과했다. 이는 pipeline/shape/inference 검증일 뿐 detector 성능이나
최종 window 결과가 아니다. Full training 전까지 상태는 **pipeline ready;
performance pending**다. Float Host 결과가 확정되기 전에는 INT8/Vela/E84 또는
reflex firmware로 진행하지 않는다.

## Full detector result와 validation-only failure diagnosis

사용자가 seed `20260812`로 두 detector × 5/10/15/20/30/50 ms Float Host full
training을 완료했다. 기존 endpoint 기준 `FPR<=5%, recall>=95%` gate는 Slip
5 ms recall 84.63%, Sink/Tilt 50 ms recall 87.01%가 최고여서 모두 실패했다.
Test ablation은 기존 runner가 산출했지만 후속 model/window/threshold/persistence/
pooling 선택에는 읽거나 사용하지 않았다.

`run_terrain_fast_reflex_validation_v1.py`는 validation 120 runs만 대상으로 기존
prediction을 재생성하고 다음 false-alarm 정의를 분리한다.

- endpoint FPR: 모든 target-negative 1 ms endpoint 중 positive 비율
- run-pre-onset FPR: target onset 전(무-target run은 전체 replay) stable firing이
  한 번이라도 있는 validation run / 전체 validation run
- target-negative, 완전 physical-hazard-free, normal-terrain run FPR: 각각의
  명시된 denominator에 대한 별도 diagnostic

Persistence 1/2/3/5/8과 validation-score 41 quantiles를 제한적으로 결합했다.
기존 Slip 5 ms GAP는 endpoint threshold/persistence 3에서 run recall 92.19%,
run FPR 5.00%였다. Run-level policy의 최선은 recall 93.75%, FPR 2.50%로 95%
gate에 미달했다. Onset 직후 0–2 ms endpoint recall은 5 ms에서 66.15%지만
50 ms GAP에서 10.94%로 감소해 짧은 slip transient의 평균 희석과 일치했다.

허용된 최소 architecture ablation으로 Slip 5 ms의 pooling만 비교했다.
GlobalMax는 1,221 parameters, GAP+Max는 1,237 parameters다. Validation-only
run-level 선택 결과 GAP+Max, threshold `0.7317748725`, persistence 3이 64 target
runs 중 62개(96.875%)를 검출하고 pre-onset false alarm 6/120(5.00%)이었다.
Hazard-onset→detection median/p95는 3.0/12.85 ms다. 이 구성은 validation gate와
`<=20 ms` observation 목표를 통과했지만 test는 아직 실행하지 않은 candidate다.

Sink/Tilt validation target 45 runs는 sink-only 0, tilt-only 15, sink+tilt 30으로
구성된다. 기존 모델은 모든 window에서 sink+tilt run recall 100%인 반면 tilt-only
run recall은 최대 13.33%였다. Run FPR 5% 이하 threshold×persistence 조건에서는
최고 recall이 30/45=66.67%에 머물렀다. 이는 단순 observation/GAP 문제보다
heterogeneous target 및 tilt-only observability 문제가 지배적이라는 근거이므로
Sink/Tilt pooling 재학습은 확대하지 않았다. Sink/Tilt는 gate 실패 상태이며
INT8/E84 대상이 아니다.

상세 artifact는
`simulation/outputs/terrain_fast_reflex_detector_validation_v1`의 summary/CSV/plot에
있다. 다음 단계는 Slip candidate의 고정 test 평가와 Sink/Tilt task/physical
feature/oracle-label 설계 검토이며, 그 승인 전 INT8/E84로 진행하지 않는다.

## Frozen Slip one-shot test와 Sand Tilt validation 분석

Slip validation candidate는 test materialization 전에 다음 값으로 동결했다.

```text
Conv1D(12,k5) -> Conv1D(16,k3) -> GAP+GlobalMax -> Dense(1)
window=5 ms, threshold=0.7317748725, persistence=3
parameters=1,237, MAC=5,912, seed=20260812
```

`warped_multisine`와 `smooth_random_patches` 소유 test 120 runs에 대해 모델
inference를 정확히 한 번 실행했다. Target recall은 64/65=98.46%였지만,
pre-onset false alarm은 9/120=7.50%여서 사전 gate의 5% 한계를 넘었다. 따라서
판정은 **Slip Host Final Gate FAIL**이다. Hazard onset부터 stable detection까지
median/p95/max는 2/10.25/42 ms였다. Test 결과를 본 뒤 threshold, persistence,
window, pooling, normalization, seed 또는 모델을 변경하거나 재학습하지 않았다.
최종 artifact는 `simulation/outputs/terrain_fast_reflex_slip_final_test_v1`에 있다.

| Metric | Validation | Test |
|---|---:|---:|
| Window | 5 ms | 5 ms |
| Run recall | 96.875% | 98.462% |
| Pre-onset run FPR | 5.00% | 7.50% |
| Median onset latency | 3.0 ms | 2.0 ms |
| p95 onset latency | 12.85 ms | 10.25 ms |

```text
Slip FINAL HOST GATE: FAIL
```

Sand는 test split을 전혀 사용하지 않고 train 180 + validation 120 runs만
분석했다. FSR contact 위치는 모델 XML의 local coordinate를 근거로 rear=(1,2),
front=(3,4), left=(1,3), right=(2,4)로 매핑했다. IMU site에는 별도 quaternion이
없지만 실제 보드 축 보정은 아직 없으므로 분석과 후보 입력에는 raw
`gyro_x/gyro_y` 명칭을 유지한다.

Validation Tilt-only 15 runs의 oracle maximum tilt는 0.000381--0.000595 rad,
median 0.000445 rad로 canonical 0.000347 rad threshold의 1.10--1.72배에 불과했다.
그럼에도 onset-relative 5 ms에서 normalized front/rear FSR imbalance는
Tilt-only와 completely hazard-free Normal 사이 ROC-AUC/PR-AUC 1.000,
Cohen's d 5.00이었고, 5 ms gyro XY integral도 ROC-AUC 0.905였다. 즉 raw
Fusion10에 정보가 전혀 없다는 가설은 지지되지 않지만, 신호가 매우 작은 oracle
boundary 부근이고 기존 joint CNN이 강한 Sink+Tilt mode에 치우친 representation/
multi-mode 문제가 함께 존재한다.

Train-score threshold 후보와 validation `run FPR<=5%` 선택만 사용한 경량 rule과
logistic baseline을 평가했다. 최종 authoritative 결과는
`simulation/outputs/terrain_fast_reflex_sand_tilt_final_validation_v1`에 저장한다.
단일 feature separation이 높아도 전체 100 ms replay에서 false alarm을 억제하면
지속적인 Tilt-only run 검출로 이어지지 않았으며, 기존 CNN과 lightweight path의
OR 조합도 95% recall gate를 통과하지 못했다. Sand candidate는 아직 확보되지
않았고 Sand test, INT8, Vela, E84, UART 및 reflex firmware는 실행하지 않았다.

| Detector | Window | Sand run recall | Tilt-only recall | Run FPR | Median latency |
|---|---:|---:|---:|---:|---:|
| Existing CNN | 20 ms | 66.67% | 0.00% | 0.83% | 12.0 ms |
| Physical rule | 50 ms | 44.44% | 26.67% | 5.00% | 17.5 ms |
| Logistic | 50 ms | 60.00% | 13.33% | 4.17% | 10.0 ms |
| Combined OR | 50 ms | 75.56% | 26.67% | 5.00% | 11.0 ms |

Sand validation gates: recall `>=95%` **FAIL**, run FPR `<=5%` **PASS**,
observation `<=20 ms` **FAIL**. Small MLP는 rule/logistic의 causal replay가
onset-aligned separation을 재현하지 못해 추가 capacity의 근거가 없으므로 실행하지
않았다.
