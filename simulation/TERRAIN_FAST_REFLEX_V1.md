# Terrain Transition Fast Reflex v1

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
