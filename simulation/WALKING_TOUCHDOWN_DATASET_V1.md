# Walking Touchdown Dataset v1

상태: **acquisition foundation, contact-model audit와 Ice/Sand stability sweep
완료; Sand walking support는 해결했지만 Ice policy gate가 실패하여 full
generation 및 classifier training 보류**.

## Locomotion path

현재 repository의 `unitree_sdk2_python/example/g1/high_level/` 예제는 실제
로봇의 내장 locomotion service에 요청을 보내는 client이며 `unitree_mujoco`
안에서 그 service를 구현하지 않는다. 저수준 simulator도 `LowCmd`를 받을 뿐
보행을 자체 생성하지 않는다.

따라서 임의 joint trajectory 대신 Unitree 공식
`unitreerobotics/unitree_rl_mjlab`의 pretrained G1-29DOF velocity ONNX policy를
사용한다. 이 policy는 MuJoCo backend에서 훈련되었고 현재 full-body XML의 29
actuator 순서와 일치한다. Policy artifact는 repository에 복사하지 않으며 실행
시 `--policy-path`로 명시한다.

## Canonical event

- Physics: 2 kHz (`dt=0.5 ms`)
- Sensor: 매 2 physics step의 native 1 kHz read; interpolation 없음
- Contact ground truth: `surface_floor`와 네 named left sole sphere 중 하나의
  MuJoCo contact
- Raw touchdown: sampled `previous_contact=False`, `current_contact=True`
- Chatter rejection: 선행 AIR 40 ms 이상, 이후 CONTACT 4 ms 지속을 확인한 뒤
  최초 CONTACT sample로 timestamp를 backdate
- Event: `[-10,+50) ms`, 60 samples, shape `(60,10)`
- Classifier candidate: `t >= 0`의 첫 1/2/5/10/15/20/30/50 samples
- FSR diagnostic: FSR sum이 5 N을 처음 넘은 timestamp와 MuJoCo touchdown 대비
  delta를 기록하되 canonical alignment에는 사용하지 않음

Channel order와 unit은 기존 deployment와 동일하다.

```text
foot_force_1..4 [N]
accel_x/y/z [m/s^2]
gyro_x/y/z [rad/s]
```

`events.npz`에는 `sensors`, `contact`, `fsr_sum`,
`sample_relative_time_ms`, `event_id`, `terrain_class`, `valid`가 저장된다.
`manifest.csv`에는 touchdown/run/surface/threshold metadata,
`runs.csv`에는 locomotion validity와 touchdown interval, `protocol.json`에는
전체 schema와 split policy가 저장된다.

Terrain label은 touchdown 순간 **왼발 sole sphere가 접촉한 terrain**이다.
몸 전체 위치나 오른발 contact는 label을 결정하지 않는다.

## Leakage policy

Expanded Dataset v1의 family ownership을 그대로 유지한다.

- train: `multisine`, `filtered_random`, `sparse_aggregate`
- validation: `crosshatch`, `rounded_ridges`
- test: `warped_multisine`, `smooth_random_patches`
- surface family, surface seed, session, run은 하나의 split에만 속함
- 같은 run의 touchdown을 event 단위로 random split하지 않음

## Setup and commands

한 번만 공식 policy repository와 Python dependency를 준비한다.

```bash
cd /d/shin/Infineon
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git simulation/unitree_rl_mjlab
git -C simulation/unitree_rl_mjlab checkout 1425b15f73bd4095f0df53709d7c389c3eb9e790
simulation/venv/bin/python -m pip install -r simulation/unitree_mujoco/simulate_python/requirements-walking-touchdown-v1.txt
```

Pilot은 4 terrain × multisine 1 surface × 2 runs = 8 runs, 약 16–32 event를
예상한다. 이 host의 concrete smoke 비율로 약 10–30초, plot 없이 1 MiB 미만을
예상한다.

```bash
cd /d/shin/Infineon/simulation/unitree_mujoco/simulate_python
../../venv/bin/python run_walking_touchdown_dataset_v1.py \
  --execute \
  --policy-path ../../unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx \
  --output-dir ../../outputs/terrain_walking_touchdown_v1_pilot \
  --terrains concrete marble ice sand \
  --families multisine \
  --surfaces-per-family 1 --runs-per-surface 2 \
  --duration-s 3.0 --settling-s 0.6 --walking-speed 0.2
```

## Locomotion stability gate

Pilot 결과는 Concrete/Marble 16개 event가 모두 유효했지만 Ice/Sand의 15개
event는 모두 run validity gate에서 탈락했다. 따라서 full generation 전에
Ice/Sand에서 0.10/0.15/0.20 m/s, 조건당 2 run의 짧은 sweep을 수행했다.

```bash
cd /d/shin/Infineon/simulation/unitree_mujoco/simulate_python
../../venv/bin/python run_walking_stability_sweep.py \
  --execute \
  --policy-path ../../unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx \
  --output-dir ../../outputs/terrain_walking_stability_sweep_v1_20260812
```

초기 full-body 결과에서 Sand의 최초 실패는 0.6005 s의 non-foot contact였고,
접촉 대상은 양쪽 knee/ankle-pitch collision mesh, 최대 normal force는
63.13 N이었다. 원본 G1 XML을 바꾸지 않는 `foot-spheres-only` runtime model로
분리하자 Concrete는 2/2 통과했지만, native Sand soft-contact에서는 로봇이
기울지 않은 채 지면으로 침하했다. 파라미터 factorization 결과 Sand의 native
`solimp`를 유지하면서 `solref`만 hard reference `(0.015, 1.0)`로 바꾸면
0.20 m/s 2/2, 8/8 touchdown과 0.517–0.522 m 전진을 얻었다. 이를
`walking-support-v1`로 고정했다. 이 변경은 walking runner에만 적용되며 기존
terrain profile과 과거 dataset을 변경하지 않는다.

최종 gate는 3초 완주, `run_valid=1`, 조기 종료 없음, 유효 left touchdown
3개 이상, 실제 전진량이 명령 전진량의 50% 이상이다. Foot-only +
walking-support-v1 sweep에서 Sand는 0.10/0.15/0.20 m/s 모두 2/2 통과했다.
Ice는 모두 0/2였다. 0.10 m/s 평균 x 이동은 -0.531 m, 0.15 m/s는 -0.333 m였고
0.20 m/s 두 run은 base-height fall이었다.

사용한 upstream velocity policy의 training configuration은 foot friction을
`[0.3, 1.6]`에서 randomize하지만 현재 Ice domain은 `[0.03, 0.12]`이다.
따라서 Ice 실패는 현재 pretrained policy의 명시적인 out-of-distribution
조건이며 terrain definition을 완화해서 통과 처리하지 않는다.

`--stop-on-fall`은 최초 실패 시점, 이유, base/left-foot planar speed, contact
상태, terminal pose와 offending geom/contact 위치·침투·normal force를
`runs.csv`에 남긴다. Sweep 결과는 `stability_runs.csv`,
`stability_matrix.csv`, `protocol.json`, `summary.txt`에 저장된다.

현재 재현 명령은 다음과 같다.

```bash
../../venv/bin/python run_walking_stability_sweep.py \
  --execute \
  --policy-path ../../unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx \
  --output-dir ../../outputs/terrain_walking_stability_sweep_foot_support_v1_20260812 \
  --terrains ice sand --speeds 0.10 0.15 0.20 --runs-per-condition 2 \
  --contact-model foot-spheres-only \
  --contact-parameters walking-support-v1 \
  --minimum-forward-progress-ratio 0.5
```

아래 Full foundation 명령은 stability gate를 통과한 뒤에만 실행한다. 현재는
실행하지 않는다. 권장안은 4 terrain × 7 family × 3 surfaces × 3 runs = 252
runs이다. Run당 2–4 touchdown 가정으로 약 504–1,008 event,
이 host에서 약 5–10분, 압축 output 약 3–10 MiB를 예상한다. 이는 학습을
실행하지 않는다.

```bash
cd /d/shin/Infineon/simulation/unitree_mujoco/simulate_python
../../venv/bin/python run_walking_touchdown_dataset_v1.py \
  --execute \
  --policy-path ../../unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx \
  --output-dir ../../outputs/terrain_walking_touchdown_v1_full \
  --terrains concrete marble ice sand \
  --families multisine filtered_random sparse_aggregate crosshatch rounded_ridges warped_multisine smooth_random_patches \
  --surfaces-per-family 3 --runs-per-surface 3 \
  --duration-s 3.0 --settling-s 0.6 --walking-speed 0.2 \
  --contact-model foot-spheres-only \
  --contact-parameters walking-support-v1 --stop-on-fall
```

정상 종료 마지막 log는 다음 형태다.

```text
terrain_walking_touchdown_v1 schema_version=1
runs=... events=... valid=...
sensors=(..., 60, 10) contact=(..., 60)
native_sampling=1000Hz physics=2000Hz steps_per_sample=2 timestamp_spacing=1ms
```

실행 뒤 `summary.txt`, `protocol.json`, `runs.csv`, `manifest.csv`와 마지막
console log를 다음 검토에 전달한다. Full generation 결과를 검토하기 전에는
CNN/INT8/E84 단계를 시작하지 않는다.
