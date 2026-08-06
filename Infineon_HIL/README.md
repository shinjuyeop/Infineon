# Infineon HIL PoC — Scenario 1

Scenario 1 Terrain Classification의 Host reference pipeline과 embedded deployment 준비 pipeline이다. 기존 NumPy CNN/Mock HIL 흐름을 유지하면서 별도의 TensorFlow target CNN을 학습하고 strict INT8 TFLite로 변환해 Host에서 수치·정확도 동등성을 검증한다. 실제 KIT_PSE84_AI 펌웨어나 전기적 HIL은 아직 포함하지 않는다.

> **Synthetic-data warning**
>
> **All terrain parameters in v0.1 are engineering estimates for synthetic/HIL validation. They are not measured sensor data. The final model must be recalibrated and retrained/validated against Phase-1 bench recordings.**

> **Mock-HIL warning**
>
> **Mock HIL은 실제 Hardware-in-the-Loop 시험이 아니다.** 현재 구현은 software interface와 데이터 흐름을 검증할 뿐이며, 실제 timing, transport, DAC, analog signal integrity, E84 acquisition/deployment를 검증하지 않는다.

## 현재 단계

| Phase | 내용 | 상태 |
|---|---|---|
| Scenario 1A | Synthetic terrain signal generation | DONE |
| Scenario 1B | Host reference NumPy CNN | DONE |
| Scenario 1C | Mock HIL end-to-end software pipeline | DONE |
| Scenario 1D | Deployment-target TensorFlow CNN | DONE |
| Scenario 1E | INT8 TFLite export + Host validation | DONE |
| Scenario 2 | KIT_PSE84_AI fixed-window on-device inference | NEXT |
| Scenario 3 | Host ↔ KIT_PSE84_AI software injection | FUTURE |
| Scenario 4 | PSoC6 + DAC + J17 electrical HIL | FUTURE |
| Scenario 5 | Real sensor calibration / terrain validation | FUTURE |

현재와 다음 단계의 경계는 다음과 같다.

```text
CURRENT
Terrain YAML
    ↓
Synthetic Generator
    ↓
Dataset (2000, 50, 5)
    ├──────────────→ NumPy Reference CNN → MockE84
    │
    └──────────────→ TensorFlow Target CNN
                         ↓
                    Float Evaluation
                         ↓
                     INT8 TFLite
                         ↓
                  Host TFLite Validation

NEXT
Host Dataset → KIT_PSE84_AI → E84 / Ethos-U55 inference

FUTURE
Host → PSoC6 HIL → DAC → J17 / ADC → KIT_PSE84_AI
```

## Sensor schema

- Sampling: 1,000 Hz, 50 ms, 50 timesteps
- Canonical channel order: `FSR1`, `FSR2`, `FSR3`, `FSR4`, `vibration`
- Sample shape: `(50, 5)`
- Dataset shape: `(2000, 50, 5)`; 4 classes × 500 samples
- Units: FSR `[N]`, vibration `[g]`
- Classes: `0=concrete`, `1=marble`, `2=ice`, `3=sand`

채널 이름과 index는 `infineon_hil.schema` 한 곳에서 정의한다. YAML channel order도 이 schema와 일치하는지 로드 시 검증된다. Class ID나 terrain metadata는 sensor tensor에 포함되지 않는다.

## Directory tree

```text
.
├── configs/terrain_v0_1.yaml
├── src/infineon_hil/
│   ├── schema.py                   # canonical channel definition
│   ├── terrain/
│   │   ├── params.py               # YAML/provenance validation
│   │   ├── generator.py            # synthetic FSR/vibration generator
│   │   ├── dataset.py              # dataset, metadata, 70/15/15 split
│   │   ├── features.py             # four canonical sanity features
│   │   ├── validation.py           # nearest-centroid sanity baseline
│   │   └── mapping.py              # physical/normalized/code abstractions
│   ├── model/
│   │   ├── dataset.py              # load, split, train-only preprocessing
│   │   ├── network.py              # trainable NumPy 1D CNN and NPZ format
│   │   ├── train.py                # mini-batch Adam training
│   │   ├── inference.py            # shared preprocessing and inference
│   │   └── evaluation.py           # CNN-specific held-out metrics
│   ├── deployment/
│   │   ├── model.py                 # fixed TensorFlow target architecture
│   │   ├── dataset.py               # shared split/preprocessing adapter
│   │   ├── metadata.py              # portable preprocessing/TFLite metadata
│   │   ├── train.py                 # Keras training and history plot
│   │   ├── evaluate.py              # metrics and confusion-matrix plots
│   │   ├── quantization.py          # strict full-INT8 export/inspection
│   │   ├── inference.py             # Host INT8 TFLite inference
│   │   └── parity.py                # Float ↔ INT8 comparison
│   ├── hil/
│   │   ├── frame.py                # physical-unit five-channel frame
│   │   ├── interface.py            # HilOutput start/write/stop contract
│   │   ├── player.py               # sample selection and timestep playback
│   │   ├── mock.py                 # lossless in-memory output backend
│   │   └── simulation.py           # host-only mock composition
│   └── device/mock_e84.py           # frame collection and host inference
├── scripts/
│   ├── generate_dataset.py / train_model.py / run_mock_hil.py
│   ├── train_deployment_model.py / evaluate_deployment_model.py
│   ├── export_int8_tflite.py / infer_tflite_sample.py
│   └── compare_float_int8.py
├── tests/
├── data/synthetic/                 # generated; gitignored
├── models/deployment/              # .keras/.tflite/.json; generated, gitignored
└── outputs/plots/deployment/       # generated plots; gitignored
```

## Host reference NumPy model

```text
Input (50,5)
→ Conv1D(5→12, kernel=5, same)
→ ReLU
→ Conv1D(12→16, kernel=3, same)
→ ReLU
→ Global Average Pooling
→ Dense(16→4)
→ Softmax inference
```

총 trainable parameter는 972개다. 학습에는 mini-batch Adam과 cross-entropy를 사용한다. 전처리는 training partition에서 계산한 5채널별 mean/std만 사용하며, 값은 model artifact에 저장되어 Host inference와 MockE84가 완전히 동일하게 적용한다.

NumPy reference artifact는 pickle 없는 compressed NPZ이며 Host architecture와 Mock HIL 검증용이다. 이 모델과 `src/infineon_hil/model/` 코드는 deployment model로 대체되지 않는다.

## Deployment-target TensorFlow model

```text
Input (50,5)
→ Conv1D(16, kernel=5, same) → ReLU → MaxPooling1D(2)
→ Conv1D(32, kernel=3, same) → ReLU → GlobalAveragePooling1D
→ Dense(16) → ReLU → Dense(4) → Softmax
```

총 trainable parameter는 2,580개다. 동일한 dataset/split을 사용하고, training partition에서만 계산한 채널별 mean/std를 Keras/INT8 양쪽에 동일하게 적용한다. Metadata JSON에는 channel order, mean/std, class names, input/sampling 정보와 TFLite input/output quantization parameter를 저장한다. Representative dataset도 training partition에서만 선택한다.

TFLite converter는 `TFLITE_BUILTINS_INT8`만 허용하며 input/output을 `int8`로 강제한다. 지원되지 않는 operator가 있으면 Flex fallback 없이 변환이 실패한다.

Host inference와 향후 firmware가 재현해야 하는 계산 순서는 다음과 같다.

```text
standardized = (physical_input - channel_mean) / channel_std
int8_input   = clip(round(standardized / input_scale) + input_zero_point, -128, 127)
float_output = (int8_output - output_zero_point) * output_scale
prediction   = argmax(float_output)
```

`channel_mean/std`와 quantization scale/zero point는 반드시 해당 export에서 생성된 metadata JSON을 사용해야 한다. 재학습하거나 TensorFlow 버전이 바뀌면 값이 달라질 수 있다.

## 설치 및 end-to-end 실행

### Base Host reference 환경

TensorFlow 없이 synthetic generator, NumPy reference CNN, Mock HIL을 실행할 수 있다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Synthetic dataset
python scripts/generate_dataset.py \
  --config configs/terrain_v0_1.yaml \
  --samples-per-class 500 \
  --seed 42

# 2. CNN training and artifact save
python scripts/train_model.py \
  --dataset data/synthetic/terrain_v0_1.npz \
  --output models/terrain_cnn_v0_1.npz \
  --epochs 30 \
  --seed 42

# 3. Held-out CNN evaluation
python scripts/evaluate_model.py \
  --dataset data/synthetic/terrain_v0_1.npz \
  --model models/terrain_cnn_v0_1.npz \
  --seed 42

# 4. One-sample direct inference
python scripts/infer_sample.py \
  --dataset data/synthetic/terrain_v0_1.npz \
  --model models/terrain_cnn_v0_1.npz \
  --sample-index 123

# 5. One-sample end-to-end mock replay
python scripts/run_mock_hil.py \
  --dataset data/synthetic/terrain_v0_1.npz \
  --model models/terrain_cnn_v0_1.npz \
  --sample-index 123

# Optional consecutive-sample mock evaluation
python scripts/run_mock_hil.py \
  --dataset data/synthetic/terrain_v0_1.npz \
  --model models/terrain_cnn_v0_1.npz \
  --sample-index 450 \
  --count 100

pytest -q
```

`SignalPlayer.play(..., realtime=False)`가 기본값이므로 1 kHz timestamp는 유지하지만 wall-clock sleep은 하지 않는다. 향후 실제 backend가 buffer/timer timing을 책임지도록 player와 transport를 분리했다.

### Deployment 환경과 전체 pipeline

같은 가상환경에 선택적 TensorFlow dependency를 설치한다. Linux CPU 환경에서는 `tensorflow-cpu`를 사용하며 base requirements에는 TensorFlow가 포함되지 않는다.

```bash
pip install -r requirements-deploy.txt

# 1. Deployment-target float CNN training
python scripts/train_deployment_model.py \
  --dataset data/synthetic/terrain_v0_1.npz \
  --output models/deployment/terrain_cnn_v0_1.keras \
  --epochs 30 \
  --batch-size 64 \
  --seed 42

# 2. Held-out float evaluation and KPI report
python scripts/evaluate_deployment_model.py \
  --dataset data/synthetic/terrain_v0_1.npz \
  --model models/deployment/terrain_cnn_v0_1.keras \
  --metadata models/deployment/terrain_cnn_v0_1_metadata.json

# 3. Strict full-INT8 TFLite export (training samples only for calibration)
python scripts/export_int8_tflite.py \
  --dataset data/synthetic/terrain_v0_1.npz \
  --model models/deployment/terrain_cnn_v0_1.keras \
  --metadata models/deployment/terrain_cnn_v0_1_metadata.json \
  --output models/deployment/terrain_cnn_v0_1_int8.tflite \
  --representative-samples 200

# 4. One-sample Host INT8 inference
python scripts/infer_tflite_sample.py \
  --dataset data/synthetic/terrain_v0_1.npz \
  --model models/deployment/terrain_cnn_v0_1_int8.tflite \
  --metadata models/deployment/terrain_cnn_v0_1_metadata.json \
  --sample-index 123

# 5. Float/INT8 held-out parity and confusion matrices
python scripts/compare_float_int8.py \
  --dataset data/synthetic/terrain_v0_1.npz \
  --float-model models/deployment/terrain_cnn_v0_1.keras \
  --int8-model models/deployment/terrain_cnn_v0_1_int8.tflite \
  --metadata models/deployment/terrain_cnn_v0_1_metadata.json

# All tests; deployment tests skip cleanly if TensorFlow is absent
pytest -q

# Optional deployment-only checks after installing requirements-deploy.txt
pytest -q -m deployment
```

Generated artifacts:

```text
models/deployment/
├── terrain_cnn_v0_1.keras
├── terrain_cnn_v0_1_int8.tflite
├── terrain_cnn_v0_1_metadata.json
└── terrain_cnn_v0_1_training.json

outputs/plots/deployment/
├── training_history.png
├── float_confusion_matrix.png
└── int8_confusion_matrix.png
```

이 binary/JSON/plot artifact는 재생성 가능하므로 Git에 포함하지 않는다. Accuracy와 ice recall KPI는 synthetic held-out partition의 regression gate일 뿐 실제 terrain generalization 성능이 아니다.

> **TensorFlow/LiteRT compatibility note**
>
> 현재 검증 환경의 TensorFlow 2.21은 `tf.lite.Interpreter` deprecation warning을 출력하지만 변환과 Host inference는 정상 동작한다. 향후 TensorFlow에서 해당 API가 제거되기 전에 LiteRT interpreter로 migration해야 하며, KIT_PSE84_AI용 runtime 선택과 혼동해서는 안 된다.

## NumPy reference Seed 42 결과

CNN은 기존 v0.1 dataset을 그대로 사용했으며 synthetic parameter를 추가 튜닝하지 않았다.

- Split: train 1,400 / validation 300 / test 300, deterministic stratified 70/15/15
- Epochs: 30, batch size: 64, learning rate: 0.003
- Final train accuracy: `1.000`
- Final validation accuracy: `1.000`
- Held-out test accuracy: `1.000`
- Per-class test recall: concrete/marble/ice/sand 모두 `1.000`

이 높은 결과는 현재 engineering-estimate synthetic distribution 내부 성능일 뿐 실제 지면 일반화 성능의 증거가 아니다. 기존 four-feature nearest-centroid sanity baseline(`0.974`)과 CNN held-out evaluation은 별도 평가다.

Sample 123 mock replay 예:

```text
Sample ID: 123
Frames played: 50
Expected: concrete
Predicted: concrete
Confidence: 1.0000
Result: PASS
```

## Deployment Seed 42 검증 결과

Ubuntu/Python 3.10, `tensorflow-cpu 2.21.0`, 30 epochs에서 실제 전체 pipeline을 실행한 결과다.

- Split: train 1,400 / validation 300 / test 300, deterministic stratified 70/15/15
- Final float train accuracy: `1.0000`
- Final float validation accuracy: `1.0000`
- Held-out float test accuracy: `0.9933`
- Float per-class recall: concrete `1.0000`, marble `0.9867`, ice `1.0000`, sand `0.9867`
- Held-out INT8 test accuracy: `0.9933`
- INT8 ice recall: `1.0000`
- INT8 accuracy delta: `0.0000`
- Float/INT8 prediction agreement: `1.0000`
- TFLite input: `int8`, shape `(1,50,5)`, scale `0.0468267538`, zero point `-3`
- TFLite output: `int8`, shape `(1,4)`, scale `0.00390625`, zero point `-128`
- KPI overall accuracy ≥ 0.85: `PASS`
- KPI ice recall ≥ 0.90: `PASS`

이 수치는 v0.1 synthetic engineering-estimate distribution 내부 결과다. 실제 지면, KIT_PSE84_AI 성능, 전기적 interface 또는 on-device latency의 증거가 아니다.
위 quantization parameter도 이때 생성한 artifact의 관측값이며 firmware 상수로 복사하기 전 실제 배포 artifact의 metadata와 다시 대조해야 한다.

## HIL boundary와 mapping

`HilFrame`은 physical unit만 담고 `HilOutput`은 다음 contract만 정의한다.

```python
class HilOutput:
    def start(self) -> None: ...
    def write(self, frame: HilFrame) -> None: ...
    def stop(self) -> None: ...
```

`MockHilOutput`은 frame을 lossless하게 저장한다. `MockE84Inference`는 정확히 50개의 순차 frame을 다시 `(50,5)` tensor로 만들고 artifact의 동일 preprocessing과 CNN을 실행한다.

Mapping 계층은 physical ↔ normalized ↔ output code의 선형 interface를 제공하지만 hardware code range를 기본 가정하지 않는다. `max_code`, vibration `full_scale_g` 등은 향후 backend/calibration이 명시적으로 공급해야 한다. YAML의 `legacy_conceptual_hil_mapping`과 기존 0–200 N → 0–3.3 V/vibration mid-rail helper는 명시적으로 concept-only이며 KIT_PSE84_AI J17/ADC 또는 실제 DAC 사양으로 검증되지 않았다. Synthetic waveform 생성은 이 voltage mapping에 의존하지 않는다.

## RealHilOutput을 붙일 때 필요한 구현

실제 backend는 `HilOutput`을 구현하고 다음 책임을 가져야 한다.

- `start`: transport 연결, device readiness 확인, playback transaction/buffer 준비
- `write`: `HilFrame` channel order 검증, calibration 적용, physical→code 변환, packet/buffer 전달
- `stop`: flush/trigger 완료 확인, error/status 수집, 안전한 output state 복귀
- 명시적 calibration object: FSR N range, vibration g full-scale, voltage/reference, DAC resolution
- timing 정책: Host streaming인지 device-side buffered replay인지 결정
- frame sequence/CRC/timeout/backpressure/error recovery

Player, terrain generator, CNN에는 UART/USB/DAC 구현을 넣지 않는다. Real E84 integration도 MockE84를 수정하는 대신 별도 acquisition/result interface로 교체해야 한다.

## 아직 실제 hardware로 남은 범위

- KIT_PSE84_AI / PSoC Edge E84의 지원 deployment path와 operator set 확인
- 검증된 절차로 INT8 model을 E84/Ethos-U55에 배포하고 fixed-window inference 수행
- UART/USB packet protocol 및 Host transport
- DAC/ADC driver, reference voltage, channel routing
- actual voltage output과 1 kHz timing 검증
- FSR N↔V 및 vibration g↔V 실측 calibration
- analog signal integrity, saturation, noise, latency 측정
- E84 acquisition preprocessing의 Host 동등성 검증
- on-device prediction/result collection protocol
- real terrain recording 기반 재학습 및 end-to-end HIL validation

Next step: verify the supported deployment path/operator set for KIT_PSE84_AI / PSOC Edge E84 and deploy the INT8 model. DeepCraft import format, Vela command, firmware interface는 확인 전까지 추측하거나 자동화하지 않는다.
