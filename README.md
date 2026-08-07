# Infineon

KIT_PSE84_AI host/HIL 실험과 Unitree G1 MuJoCo simulation을 위한 통합
workspace이다.

## 폴더 구성

- `Infineon_HIL/`: host-side terrain 및 deployment 검증 pipeline
- `Infineon_test/`: KIT_PSE84_AI와 ModusToolbox project source
- `simulation/unitree_mujoco/`: G1 virtual sensor, terrain logging, controlled
  excitation, 분석 도구를 포함한 Unitree MuJoCo simulator
- `simulation/unitree_sdk2_python/`: Unitree SDK2 Python dependency

Local 문서, virtual environment, 내려받은 ModusToolbox dependency, build 결과,
생성 dataset, plot, model artifact는 의도적으로 version control에서 제외한다.
각 component README와 dependency 파일을 이용해 다시 생성할 수 있다.

Simulator의 terrain profile은 controlled signal-separation 실험을 위한 공학적
근사치이다. 실제 측정 material property나 real sensor data가 아니다.

Third-party license와 attribution 파일은 해당 source tree에 그대로 보존한다.
이 repository는 upstream project를 대체하거나 재라이선스하지 않는다.

## 주요 문서

- [`simulation/TERRAIN_DATASET_V1.md`](simulation/TERRAIN_DATASET_V1.md):
  leakage-safe MuJoCo `(N, 50, 10)` pilot와 legacy synthetic `(50, 5)` pipeline의
  경계
- [`simulation/TERRAIN_CLASSIFICATION_HANDOFF.md`](simulation/TERRAIN_CLASSIFICATION_HANDOFF.md):
  현재 설계, 실험 이력, repository map, reset handoff
- [`simulation/CLEANUP_AUDIT.md`](simulation/CLEANUP_AUDIT.md):
  source/generated/vendor 경계와 보수적 cleanup 판단
- [`simulation/NEXT_MILESTONE.md`](simulation/NEXT_MILESTONE.md):
  expanded dataset/1D-CNN milestone과 E84 진행 gate
- [`simulation/EXPANDED_DATASET_V1_RESULTS.md`](simulation/EXPANDED_DATASET_V1_RESULTS.md):
  expanded dataset, 3-seed CNN, strict INT8 host-parity 최종 결과

## 현재 terrain-classification 후보

```text
Terrain → MuJoCo full-body G1 → FSR4 + foot IMU6
        → domain variation → sensor noise
        → medium_response → (50, 10)
        → surface-disjoint train/validation/test → classifier → future E84
```

현재 결과는 simulation/host candidate이며 E84 deployment 완료를 의미하지
않는다. Pelvis IMU와 slip/contact trace는 diagnostic 전용이다. Legacy 실험과
synthetic `(50, 5)` HIL pipeline은 보존하지만 Dataset v1 입력에는 사용하지
않는다.

## G1 horizontal-pulse preview

Simulation virtual environment를 준비한 뒤 실행한다.

```bash
cd simulation/unitree_mujoco/simulate_python
python run_horizontal_pulse_dataset.py \
  --terrain ice \
  --runs 1 \
  --duration 1.0 \
  --sample-rate 50 \
  --seed 20260805 \
  --support-ratio 0.70 \
  --pulse-start 0.25 \
  --pulse-duration 0.20 \
  --pulse-magnitude 80 \
  --gui \
  --realtime-factor 0.25 \
  --gui-hold-seconds 5
```
