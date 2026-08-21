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

## Terrain 연구 문서

현재 frozen Terrain v4의 상태와 배포 준비도는 아래 순서로 확인한다.

- [`simulation/TERRAIN_RESEARCH_STATUS.md`](simulation/TERRAIN_RESEARCH_STATUS.md):
  현재 gate, frozen v4 모델/INT8/runtime HIL 결과와 전체 연구 이력의 정본
- [`simulation/TERRAIN_RESEARCH_GUIDE.md`](simulation/TERRAIN_RESEARCH_GUIDE.md):
  설계 결정, source/test map, reset 시 읽을 순서
- [`simulation/TERRAIN_RUNBOOK.md`](simulation/TERRAIN_RUNBOOK.md):
  재현·검증 명령과 안전한 output-directory 규칙
- [`Infineon_test/projects/terrain_e84_deploy/deployment/README.md`](Infineon_test/projects/terrain_e84_deploy/deployment/README.md):
  E84 Cortex-M55/Ethos-U55 배포 및 UART HIL 절차

다음은 현재 후보를 대체하지 않는 보존용 연구 기록이다.

- [`simulation/TERRAIN_DATASET_V1_PILOT.md`](simulation/TERRAIN_DATASET_V1_PILOT.md),
  [`simulation/TERRAIN_DATASET_V1_HISTORICAL_RESULTS.md`](simulation/TERRAIN_DATASET_V1_HISTORICAL_RESULTS.md):
  Dataset v1 pilot 및 expanded-dataset 결과
- [`simulation/TERRAIN_RATE_ABLATION.md`](simulation/TERRAIN_RATE_ABLATION.md):
  과거 sampling-rate/observation-window 근거
- [`simulation/TERRAIN_FAST_REFLEX_V1.md`](simulation/TERRAIN_FAST_REFLEX_V1.md),
  [`simulation/TERRAIN_FAST_REFLEX_V2.md`](simulation/TERRAIN_FAST_REFLEX_V2.md):
  terrain과 분리된 Fast Reflex 연구
- [`simulation/TERRAIN_WALKING_TOUCHDOWN_V1.md`](simulation/TERRAIN_WALKING_TOUCHDOWN_V1.md):
  보행 touchdown 후속 연구

## 현재 terrain-classification 후보

```text
Terrain → MuJoCo full-body G1 → FSR4 + foot IMU6
        → domain variation → sensor noise → 1 kHz causal 50 ms Fusion10 window
        → provenance/surface-realization-disjoint split → strict INT8 classifier
        → batched asynchronous T4B1 → physical PSoC Edge E84 / Ethos-U55
        → terrain result + target-runtime shadow logging
```

Frozen v4는 leakage-safe static/transition reservations와 fresh transition test를
통과했다. E84/U55 target runtime은 fixed-golden bounded raw parity 및 1 kHz
asynchronous T4B1 HIL gate를 통과했다. 원본 Float host와 board는 두 saturated
vector에서 비우승 logit 차이가 있으므로 raw-exact host parity를 주장하지 않으며,
target-runtime parity policy를 사용한다. Pelvis IMU와 slip/contact trace는
diagnostic 전용이고 legacy synthetic `(50, 5)` pipeline은 현재 입력에 사용하지
않는다.

## Walking v2 Fast Reflex host prototype

Walking v2의 현재 development 결론은
[`walking_v2_fast_reflex_host_v1/audit.md`](simulation/outputs/walking_v2_fast_reflex_host_v1/audit.md)에
기록되어 있다. Locked T2 Terrain과 S4-C Slip은 advisory telemetry만 제공하고,
deterministic authority firewall이 direct reflex/recovery actuation을 항상
비활성화한다. 이는 safety 또는 deployment lock이 아닌 development design lock이다.

재현과 regression test는 다음과 같이 실행한다. 기존 versioned output을 덮어쓰지
않도록 재현 시에는 새 `--output-dir`를 지정한다.

```bash
PYTHONPATH=simulation/unitree_mujoco/simulate_python \
  simulation/venv/bin/python \
  simulation/unitree_mujoco/simulate_python/run_walking_v2_fast_reflex_host_v1.py \
  --output-dir simulation/outputs/walking_v2_fast_reflex_host_v1_reproduction

PYTHONPATH=simulation/unitree_mujoco/simulate_python \
  simulation/venv/bin/python -m unittest -v \
  simulation/unitree_mujoco/simulate_python/test_walking_v2_fast_reflex_host_v1.py
```

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
