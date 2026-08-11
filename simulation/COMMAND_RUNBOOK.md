# Terrain Classification 명령어 Runbook

이 문서는 `/d/shin/Infineon`의 현재 terrain-classification 작업을 다시 확인하거나
재현할 때 사용하는 명령을 모은 실행 안내서이다. 현재 기준 pipeline은 다음과
같다.

```text
MuJoCo full-body G1 → FSR4 + left-foot IMU6 → 100 Hz
→ medium_response (50 samples) → surface-family-disjoint split
→ Fusion10 1D-CNN → strict full-INT8 host parity
```

현재까지 생성된 dataset, Keras model, TFLite model과 plot은
`simulation/outputs/` 아래의 local artifact이며 Git에서 제외된다. 명령을 다시
실행할 때는 완료된 output을 지우거나 덮어쓰지 말고 항상 새로운 directory를
사용한다.

## 1. 작업 위치와 Python 환경

Repository 상태 확인:

```bash
cd /d/shin/Infineon
git branch --show-current
git status --short
git log -5 --oneline
```

MuJoCo simulation용 Python:

```bash
cd /d/shin/Infineon/simulation/unitree_mujoco/simulate_python
../../venv/bin/python --version
../../venv/bin/python -c "import mujoco; print(mujoco.__version__)"
```

Dataset dependency를 새 환경에 설치해야 할 때:

```bash
cd /d/shin/Infineon/simulation/unitree_mujoco/simulate_python
../../venv/bin/python -m pip install -r requirements-dataset-v1.txt
```

CNN/INT8 작업은 TensorFlow가 설치된 HIL virtual environment를 사용한다.

```bash
/d/shin/Infineon/Infineon_HIL/.venv/bin/python --version
/d/shin/Infineon/Infineon_HIL/.venv/bin/python -c \
  "import tensorflow as tf; print(tf.__version__)"
```

새 환경을 구성해야 한다면 다음 dependency를 설치한다.

```bash
cd /d/shin/Infineon/Infineon_HIL
python -m pip install -r requirements-deploy.txt
```

## 2. 현재 결과 빠른 확인

Expanded Dataset v1 요약:

```bash
cd /d/shin/Infineon
cat simulation/outputs/terrain_dataset_v1_expanded/dataset_summary.txt
```

최종 seed의 CNN protocol과 metric 확인:

```bash
cd /d/shin/Infineon
cat simulation/outputs/terrain_dataset_v1_expanded_cnn_seed_20260809_e120/training_protocol.json
column -s, -t \
  simulation/outputs/terrain_dataset_v1_expanded_cnn_seed_20260809_e120/hard_surface_metrics.csv
```

INT8 parity 결과 확인:

```bash
cd /d/shin/Infineon
cat simulation/outputs/terrain_dataset_v1_expanded_int8_seed_20260809/parity_report.json
column -s, -t \
  simulation/outputs/terrain_dataset_v1_expanded_int8_seed_20260809/parity_metrics.csv
```

확정 artifact의 hash 확인:

```bash
cd /d/shin/Infineon
sha256sum \
  simulation/outputs/terrain_dataset_v1_expanded_cnn_seed_20260809_e120/noisy_fusion.keras \
  simulation/outputs/terrain_dataset_v1_expanded/dataset_noisy.npz \
  simulation/outputs/terrain_dataset_v1_expanded_int8_seed_20260809/noisy_fusion_int8.tflite
```

기록된 정상 hash는 `EXPANDED_DATASET_V1_RESULTS.md`에 있다.

## 3. MuJoCo G1 화면 확인

기존 dataset을 건드리지 않는 1-run GUI preview:

```bash
cd /d/shin/Infineon/simulation/unitree_mujoco/simulate_python
../../venv/bin/python run_horizontal_pulse_dataset.py \
  --terrain concrete \
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
  --gui-hold-seconds 5 \
  --gui-output-dir /d/shin/Infineon/simulation/outputs/gui_preview_manual
```

GUI가 필요 없는 작은 smoke output은 반드시 새 경로를 지정한다.

```bash
cd /d/shin/Infineon/simulation/unitree_mujoco/simulate_python
../../venv/bin/python run_horizontal_pulse_dataset.py \
  --terrain all \
  --runs 1 \
  --output-dir /d/shin/Infineon/simulation/outputs/horizontal_pulse_smoke_manual
```

## 4. Expanded Dataset v1

### 비용만 확인하는 dry-run

`--execute`를 붙이지 않으면 candidate manifest와 예상 시간/용량만 계산하고
MuJoCo dataset을 생성하지 않는다.

```bash
cd /d/shin/Infineon/simulation/unitree_mujoco/simulate_python
../../venv/bin/python run_expanded_terrain_dataset_v1.py
```

### 실제 생성

아래 명령은 약 4,480 candidate 규모의 장시간 작업이다. 이미 완료된 기본
output 대신 고유한 새 경로를 사용한다.

```bash
cd /d/shin/Infineon/simulation/unitree_mujoco/simulate_python
rerun_tag=manual_20260810
../../venv/bin/python run_expanded_terrain_dataset_v1.py \
  --output-dir "/d/shin/Infineon/simulation/outputs/terrain_dataset_v1_expanded_${rerun_tag}" \
  --execute
```

현재 milestone은 이미 완료되었으므로 데이터 무결성 확인이나 명시적인 재현
요구가 없다면 이 명령을 다시 실행할 필요가 없다.

## 5. Small 1D-CNN FSR/IMU/Fusion ablation

Artifact를 만들지 않는 CNN smoke test:

```bash
cd /d/shin/Infineon/simulation/unitree_mujoco/simulate_python
/d/shin/Infineon/Infineon_HIL/.venv/bin/python \
  train_terrain_1d_cnn.py --smoke
```

최종 protocol과 같은 noisy 120-epoch 단일 seed 학습:

```bash
cd /d/shin/Infineon/simulation/unitree_mujoco/simulate_python
cnn_seed=20260809
rerun_tag=manual_20260810
/d/shin/Infineon/Infineon_HIL/.venv/bin/python \
  train_terrain_1d_cnn.py \
  --dataset-dir /d/shin/Infineon/simulation/outputs/terrain_dataset_v1_expanded \
  --output-dir "/d/shin/Infineon/simulation/outputs/terrain_dataset_v1_expanded_cnn_seed_${cnn_seed}_${rerun_tag}" \
  --variants noisy \
  --epochs 120 \
  --batch-size 64 \
  --patience 12 \
  --seed "$cnn_seed"
```

3-seed gate를 완전히 재현해야 할 때만 다음 loop를 사용한다.

```bash
cd /d/shin/Infineon/simulation/unitree_mujoco/simulate_python
rerun_tag=manual_20260810
for cnn_seed in 20260807 20260808 20260809; do
  /d/shin/Infineon/Infineon_HIL/.venv/bin/python \
    train_terrain_1d_cnn.py \
    --dataset-dir /d/shin/Infineon/simulation/outputs/terrain_dataset_v1_expanded \
    --output-dir "/d/shin/Infineon/simulation/outputs/terrain_dataset_v1_expanded_cnn_seed_${cnn_seed}_${rerun_tag}" \
    --variants noisy \
    --epochs 120 \
    --batch-size 64 \
    --patience 12 \
    --seed "$cnn_seed"
done
```

## 6. Strict full-INT8 export와 Host parity

기본 인자는 최종 선택 seed `20260809`의 Fusion10 model을 가리킨다. 기본
output은 이미 존재하므로 재검증할 때는 새 output 경로가 필요하다.

```bash
cd /d/shin/Infineon/simulation/unitree_mujoco/simulate_python
rerun_tag=manual_20260810
/d/shin/Infineon/Infineon_HIL/.venv/bin/python \
  export_terrain_int8.py \
  --dataset /d/shin/Infineon/simulation/outputs/terrain_dataset_v1_expanded/dataset_noisy.npz \
  --model /d/shin/Infineon/simulation/outputs/terrain_dataset_v1_expanded_cnn_seed_20260809_e120/noisy_fusion.keras \
  --normalization /d/shin/Infineon/simulation/outputs/terrain_dataset_v1_expanded_cnn_seed_20260809_e120/normalization.json \
  --training-protocol /d/shin/Infineon/simulation/outputs/terrain_dataset_v1_expanded_cnn_seed_20260809_e120/training_protocol.json \
  --output-dir "/d/shin/Infineon/simulation/outputs/terrain_dataset_v1_expanded_int8_${rerun_tag}" \
  --representative-samples 256 \
  --calibration-seed 20260809
```

이 명령은 train-family calibration, strict INT8 tensor/operator 검사와 전체
held-out test parity 평가를 함께 수행한다. E84 importer나 firmware 배포를
수행하는 명령은 아직 구현되지 않았다.

## 7. 보고서용 그림 확인

현재 local report figure:

```bash
cd /d/shin/Infineon
ls -lh simulation/outputs/report_figures
xdg-open simulation/outputs/report_figures/figure1_mujoco_digital_twin.png
xdg-open simulation/outputs/report_figures/figure2_terrain_sensor_response.png
xdg-open simulation/outputs/report_figures/figure3_final_confusion_matrix.png
cat simulation/outputs/report_figures/figure_captions.txt
```

이 그림들은 generated output이라 Git에 포함되지 않는다. Figure 2와 Figure 3은
각각 기존 horizontal-pulse CSV와 최종 CNN confusion CSV를 재시각화한 결과다.

## 8. Test와 code health

Simulation의 현재 automated test:

```bash
cd /d/shin/Infineon/simulation/unitree_mujoco/simulate_python
../../venv/bin/python -m unittest discover -s test -p 'test_*.py'
```

TensorFlow가 필요한 CNN/INT8 test까지 포함하려면 HIL 환경에서 pytest를
사용한다.

```bash
cd /d/shin/Infineon/simulation/unitree_mujoco/simulate_python
/d/shin/Infineon/Infineon_HIL/.venv/bin/python -m pytest -q test
```

Infineon_HIL test:

```bash
cd /d/shin/Infineon/Infineon_HIL
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q -m deployment
```

Relevant Python compile과 Git whitespace 확인:

```bash
cd /d/shin/Infineon
simulation/venv/bin/python -m compileall -q \
  simulation/unitree_mujoco/simulate_python
Infineon_HIL/.venv/bin/python -m compileall -q Infineon_HIL
git diff --check
git status --short
```

## 9. 과거 physics/sensor validation entry point

다음 명령은 실험 이력 보존용이며 현재 production dataset을 만들기 위한 기본
명령이 아니다. 재실행 전 각 `--help`와 handoff의 결론을 먼저 확인한다.

```bash
cd /d/shin/Infineon/simulation/unitree_mujoco/simulate_python

../../venv/bin/python run_controlled_terrain_dataset.py --help
../../venv/bin/python run_bidirectional_pulse_validation.py --help
../../venv/bin/python run_imu_location_ab_validation.py --help
../../venv/bin/python run_surface_sampling_rate_study.py --help
../../venv/bin/python run_surface_factorization_study.py --help
../../venv/bin/python run_timestep_convergence_study.py --help
../../venv/bin/python run_final_timestep_convergence.py --help
../../venv/bin/python run_final_025ms_convergence.py --help
../../venv/bin/python run_terrain_dataset_v1_pilot.py --help
```

과거 output 경로를 `--output-dir`로 다시 지정하지 않는다. 실험이 꼭 필요하면
새 이름의 output을 사용하고 기존 480-run, lower-body, IMU A/B, Dataset v1
결과는 그대로 보존한다.

## 10. 다음 E84 단계

Host-side에서 사용할 최종 candidate는 다음 파일이다.

```text
simulation/outputs/terrain_dataset_v1_expanded_int8_seed_20260809/
└── noisy_fusion_int8.tflite
```

다음 단계는 명령이 준비된 simulation 재실행이 아니라 아래 embedded 작업이다.

1. DEEPCRAFT/TFLite importer compatibility 확인
2. E84 model integration과 embedded normalization/quantization 구현
3. tensor arena, scratch RAM, flash와 latency 실측
4. Host↔E84 UART/HIL protocol 구현
5. 실제 FSR/BMI270 calibration과 실제 지면 validation

완료되지 않은 E84 작업을 완료된 명령이나 결과로 기록하지 않는다.
