# Terrain Sampling-Rate / Observation-Window Ablation

상태: **Quick 500/1000 Hz 비교, 1000 Hz full 3-seed, selected candidate strict
INT8 host gate 및 E84/U55 fixed/Host-golden HIL 검증 완료. Async 1 kHz UART는
미구현**.

이 문서는 기존 100 Hz canonical result를 보존한 채, 2 kHz MuJoCo physics에서
native read한 500/1000 Hz FSR4 + left-foot/ankle IMU6의 50-sample terrain
classification 결과를 기록한다. 실제 센서 또는 real-world high-frequency
vibration 검증이 아니다.

## 결론

500 Hz/100 ms와 1000 Hz/50 ms 모두 quick unseen-family gate에서 네 class를
의미 있게 분류했다. 500 Hz가 quick accuracy에서 1.57 percentage point
높았지만, 1000 Hz도 95.60%, 두 test family 모두 95% 이상이었고 모든 class
recall이 91.25% 이상이었다. 따라서 transition-latency 우선 규칙에 따라
**1000 Hz/50 ms를 fast candidate로 선정**했다.

1000 Hz full 3-seed noisy Fusion은 `97.57 +/- 0.36%` accuracy와
`97.58 +/- 0.35%` macro F1을 얻었다. Validation loss로 선택한 seed
`20260807`의 strict INT8 accuracy는 `97.10%`이며 float 대비 `-0.157 pp`,
prediction agreement는 `99.69%`이다.

| Rate | Samples | Window | Float Acc | INT8 Acc | Macro F1 | 비고 |
|---:|---:|---:|---:|---:|---:|---|
| 100 Hz | 50 | 500 ms | 99.372% | 99.293% | 99.297% INT8 | existing canonical baseline |
| 500 Hz | 50 | 100 ms | 97.170% | 미실행 | 97.207% | quick, common-valid subset |
| 1000 Hz | 50 | 50 ms | 95.597% | 미실행 | 95.627% | quick, common-valid subset |
| 1000 Hz | 50 | 50 ms | 97.255% | 97.098% | 97.113% INT8 | full selected seed 20260807 |

100 Hz 행은 기존 artifact/result를 읽은 값이며 dataset/model을 다시 만들지
않았다. Full 1000 Hz의 3-seed float mean은 아래 별도 표에 기록한다.

## Sampling pipeline

기존 `run_window()`는 physics step 직후 `G1HilSensorReader.read_vector()`로
physical-unit FSR4 + ankle IMU6를 읽고 CSV/window extraction에 전달한다.
이번 변경은 rate가 2 kHz physics rate의 정수 약수인지 검증한 뒤 physics step
counter로 native read cadence를 고정했다.

| Sensor rate | Physics timestep | Physics steps/sample | Raw sample period |
|---:|---:|---:|---:|
| 100 Hz | 0.5 ms | 20 | 10 ms |
| 500 Hz | 0.5 ms | 4 | 2 ms |
| 1000 Hz | 0.5 ms | 2 | 1 ms |

Interpolation, 기존 100 Hz CSV/tensor upsampling, duplication은 사용하지 않았다.
Rate-ablation window는 fast decision의 의미를 유지하도록 controlled pulse onset
근처의 고정 구간에서 연속 50 native samples를 취한다.

- 500 Hz: `[0.25, 0.35)`, 100 ms
- 1000 Hz: `[0.25, 0.30)`, 50 ms
- 기존 100 Hz baseline: historical canonical `medium_response [0.15, 0.65)`

Pulse start는 domain variation에 따라 0.245–0.255 s이다. 따라서 새 window는
transition excitation onset에 약 +/-5 ms로 정렬된다. 이 ablation은 sampling
rate만 독립적으로 바꾸는 실험이 아니라 50-sample fast observation engineering
ablation이다.

Protocol에는 physics timestep/rate, sensor rate, steps per sample, sample count,
observation duration, channel order, surface-family ownership, split, dataset seed와
measured generation time을 기록한다. 기존 100 Hz default invocation과 canonical
window 동작은 유지한다.

## Quick ablation 설계

Quick 규모는 full 4,480 candidates의 정확히 25%인 rate당 1,120 candidates로
정했다.

```text
4 terrains x 7 families x 4 surfaces/family x 10 runs/surface = 1,120
```

이는 모든 family/terrain pair에 40 candidates를 제공하고 양방향 pulse와
domain variation을 유지하면서 full 실행 비용을 제한한다. 500/1000 Hz의
candidate manifest는 byte-for-byte 동일하며 surface, physics, sensor-noise seed도
동일하다. Native sampling에 따른 validity 결과가 1건 달라 fair training 비교는
두 rate에서 모두 valid인 1,117-run 교집합을 별도 overwrite-safe dataset으로
만들어 사용했다. 원본 rate dataset은 변경하지 않았다.

| Rate | Candidates | Native valid | Common valid | Train / Val / Test | Generation time |
|---:|---:|---:|---:|---:|---:|
| 500 Hz | 1,120 | 1,117 | 1,117 | 479 / 320 / 318 | 622.98 s |
| 1000 Hz | 1,120 | 1,118 | 1,117 | 479 / 320 / 318 | 720.27 s |

CNN은 두 rate 모두 noisy Fusion10만 사용해 다음 동일 protocol로 학습했다.

```text
Input (50,10)
 -> Conv1D(12,k=5)
 -> Conv1D(16,k=3)
 -> GlobalAveragePooling1D
 -> Dense(4,softmax)

Adam 1e-3, maximum 120 epochs, patience 12, batch 64, seed 20260807
```

Normalization은 각 rate의 train families에서 각각 새로 계산했다. 두 quick
training 모두 120 epochs를 완료했고 measured model-fit time은 500 Hz 9.14 s,
1000 Hz 9.20 s였다. Keras artifact는 각각 45,928 bytes, parameter count는
1,272, float parameter payload는 5,088 bytes이다. Input은 모두 `(1,50,10)`이다.
Clean tensor도 함께 보존했지만 이번 fair model gate는 canonical selected
protocol과 같은 noisy Fusion으로 한정했으므로 별도 clean-vs-noisy CNN 재학습은
수행하지 않았다.

### Quick test result

| Rate | Accuracy | Macro F1 | C-M confusion | Mean confidence | Mean top-2 margin |
|---:|---:|---:|---:|---:|---:|
| 500 Hz | 97.170% | 97.207% | 3.125% (5/160) | 97.160% | 94.464% |
| 1000 Hz | 95.597% | 95.627% | 4.375% (7/160) | 94.285% | 89.097% |

| Rate | Class | Precision | Recall | F1 | Support |
|---:|---|---:|---:|---:|---:|
| 500 | Concrete | 90.80% | 98.75% | 94.61% | 80 |
| 500 | Marble | 98.70% | 95.00% | 96.82% | 80 |
| 500 | Ice | 100.00% | 100.00% | 100.00% | 79 |
| 500 | Sand | 100.00% | 94.94% | 97.40% | 79 |
| 1000 | Concrete | 90.80% | 98.75% | 94.61% | 80 |
| 1000 | Marble | 93.59% | 91.25% | 92.41% | 80 |
| 1000 | Ice | 98.68% | 94.94% | 96.77% | 79 |
| 1000 | Sand | 100.00% | 97.47% | 98.72% | 79 |

Family별 accuracy:

| Rate | smooth_random_patches | warped_multisine |
|---:|---:|---:|
| 500 Hz | 98.10% | 96.25% |
| 1000 Hz | 96.20% | 95.00% |

Confusion matrix의 class 순서는 Concrete, Marble, Ice, Sand이다.

```text
500 Hz quick                       1000 Hz quick
[[79, 1, 0, 0],                   [[79, 1, 0, 0],
 [ 4,76, 0, 0],                    [ 6,73, 1, 0],
 [ 0, 0,79, 0],                    [ 0, 4,75, 0],
 [ 4, 0, 0,75]]                    [ 2, 0, 0,77]]
```

1000 Hz는 500 Hz보다 Marble과 Ice의 separation이 약해졌지만 한 test family나
한 class에 붕괴하지 않았다. 50 ms latency가 100 ms의 절반인 점을 우선해 full
후보로 선택했다.

## 1000 Hz full result

Full 설계는 canonical expanded coverage인 4 terrains x 7 families x 8 surfaces x
20 runs = 4,480 candidates이다.

| Candidates / valid / invalid | Tensor | Train / Val / Test | Generation time |
|---:|---:|---:|---:|
| 4,480 / 4,466 / 14 | `(4466,50,10)` clean/noisy | 1,916 / 1,275 / 1,275 | 2,974.72 s (49.58 min) |

Family별 valid count는 `multisine=639`, `filtered_random=637`,
`sparse_aggregate=640`, `crosshatch=637`, `rounded_ridges=638`,
`warped_multisine=640`, `smooth_random_patches=635`이다. Family ownership과
surface/session/run-group leakage 검사를 통과했다.

### 3-seed noisy Fusion

| Seed | Test accuracy | Macro F1 | C-M confusion | Best val-loss epoch | Training time |
|---:|---:|---:|---:|---:|---:|
| 20260807 | 97.255% | 97.269% | 2.031% | 85 | 13.32 s (97 epochs) |
| 20260808 | 97.490% | 97.502% | 2.188% | 120 | 15.08 s |
| 20260809 | 97.961% | 97.964% | 2.500% | 120 | 15.34 s |
| Mean +/- sample SD | **97.569 +/- 0.359%** | **97.579 +/- 0.354%** | **2.240 +/- 0.239%** | — | — |

Per-class 3-seed mean +/- sample SD:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Concrete | 96.03 +/- 0.50% | 98.23 +/- 0.48% | 97.12 +/- 0.18% |
| Marble | 95.77 +/- 1.75% | 95.94 +/- 0.54% | 95.84 +/- 0.60% |
| Ice | 99.26 +/- 0.36% | 97.29 +/- 2.03% | 98.26 +/- 0.90% |
| Sand | 99.36 +/- 0.00% | 98.84 +/- 0.18% | 99.10 +/- 0.09% |

Test-family accuracy는 `smooth_random_patches=97.38 +/- 0.55%`,
`warped_multisine=97.76 +/- 0.18%`이다. 특정 held-out family 의존은 보이지
않는다.

Test score가 아니라 minimum validation loss로 seed `20260807`을 selected
candidate로 정했다. 이 모델의 mean confidence는 97.56%, mean top-2 margin은
95.30%이다. Keras artifact는 45,928 bytes이며 architecture resource는 quick과
동일하게 1,272 parameters / 5,088 float parameter bytes이다.

Selected float confusion matrix:

```text
[[316,  3,  0,  1],
 [ 10,308,  1,  1],
 [  0, 15,305,  0],
 [  4,  0,  0,311]]
```

주요 오류는 Marble→Concrete 10건, Ice→Marble 15건이다. Concrete↔Marble
mutual confusion은 13/640이다.

## Strict INT8 finalist

Calibration은 seed `20260807`의 noisy train partition 256개만 사용했다. 세
train families와 네 terrain이 포함되며 validation/test sample은 없다.

| Metric | Float32 | Full INT8 | Delta |
|---|---:|---:|---:|
| Accuracy | 97.255% | 97.098% | -0.157 pp |
| Macro F1 | 97.269% | 97.113% | -0.156 pp |
| Concrete-Marble confusion | 2.031% | 2.031% | 0.000 pp |
| Prediction agreement | — | 99.686% | — |

INT8 per-class result:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Concrete | 95.76% | 98.75% | 97.23% | 320 |
| Marble | 93.90% | 96.25% | 95.06% | 320 |
| Ice | 99.67% | 94.69% | 97.12% | 320 |
| Sand | 99.36% | 98.73% | 99.04% | 315 |

INT8 family accuracy는 `smooth_random_patches=96.54%`,
`warped_multisine=97.66%`이다. INT8 confusion은 float와 비교해 Ice→Marble이
15건에서 17건으로 늘어난 것만 다르다.

```text
[[316,  3,  0,  1],
 [ 10,308,  1,  1],
 [  0, 17,303,  0],
 [  4,  0,  0,311]]
```

- Interface: `int8 (1,50,10) -> int8 (1,4)`
- Model size: 7,048 bytes
- Input quantization: scale `0.0798353627`, zero point `3`
- Output quantization: scale `0.00390625`, zero point `-128`
- Serialized builtin operators: `EXPAND_DIMS`, `CONV_2D`, `RESHAPE`,
  `EXPAND_DIMS`, `CONV_2D`, `RESHAPE`, `MEAN`, `FULLY_CONNECTED`, `SOFTMAX`
- Flex/floating tensors: 없음
- TFLite SHA-256: `6f123c7727e3879e3a6c8ef41f55617a70d48a975fa44921042251b06049e74a`

Operator family, tensor shape와 model size는 기존 100 Hz deployed architecture와
같아 정적 compatibility risk는 낮다. 그러나 새 artifact에 대한 Vela import,
E84 execution, arena/latency 및 HIL은 이번 task에서 실행하지 않았으므로 embedded
PASS로 주장하지 않는다.

## 100 Hz canonical comparison 경계

100 Hz selected baseline은 4,453 valid samples, train/validation/test
1,909/1,271/1,273, 1,272 parameters, 45,928-byte Keras artifact, 7,048-byte
INT8 artifact이다. 기존 기록에는 dataset generation wall time과 selected CNN
model-fit wall time이 저장되지 않았으므로 이를 추정값으로 채우지 않았다.
Baseline mean confidence/top-2 margin은 보존된 selected model에서 재평가한
99.12%/98.26%이다.

100 Hz와 새 rate는 동일 50 samples, channels, architecture family, split-family
allocation과 domain design을 사용하지만 observation interval이 다르다. 따라서
accuracy 차이는 pure sampling-rate effect가 아니라 의도한 rate/window latency
trade-off이다.

## E84/U55 deployment와 HIL 경계

현재 TRN2 synchronous sample request/response RTT는 평균 약 1.5–1.7 ms이므로
1 ms마다 새 sample이 필요한 1 kHz sample-by-sample transport deadline을 만족할
수 없다. 따라서 기존 TRN1/TRN2 100 Hz 경로는 유지하고 async UART는 구현하지
않은 채, 별도 fast1000 model name과 fixed tensor로 배포 compatibility를 먼저
검증했다.

- CPU model: `TERRAIN_FAST1000_CPU`
- Vela/U55 model: `TERRAIN_FAST1000_U55`, 7,776 bytes, `ethos-u55-128`
- Fixed Host/board raw: `[114,-114,-128,-128]`, class 0 (`concrete`), exact PASS
- HIL sample 878 Host/board raw: `[114,-114,-128,-128]`, exact PASS
- Measured HIL profile: `cpu_cyc=24910`, `npu_cyc=9987`

Artifact 생성, build/flash 및 regression 절차는
`../Infineon_test/projects/terrain_e84_deploy/deployment/README.md`에 있다.

다음 단계는 다음 순서가 정확하다.

1. 1 kHz에 맞는 batched 또는 asynchronous binary transport와 E84 ring-buffer
   cadence를 설계한다. 기존 TRN2 compatibility는 유지한다.
2. Transport deadline을 충족한 뒤 native 1 kHz MuJoCo live HIL을 수행한다.
3. 그 다음 walking-based dataset과 terrain-transition dataset/latency metric을
   별도 milestone로 설계한다.
4. 장시간 soak/fault injection은 final-system validation으로 deferred한다.

1 kHz 결과는 MuJoCo raw high-frequency material vibration validation이 아니다.
현재 physics limitation에 맞춰 FSR/foot-IMU의 50 ms aggregate contact-response
classification evidence로만 해석한다.
