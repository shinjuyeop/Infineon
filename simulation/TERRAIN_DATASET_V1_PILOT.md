# Terrain Dataset v1 Pilot (Historical)

상태: **historical simulation/host pilot**. 이 문서는 현재 frozen Terrain v4
candidate가 아니며, real sensor calibration 또는 E84 deployment evidence를
주장하지 않는다.

이 pipeline은 `Infineon_HIL` 아래의 legacy synthetic `(50, 5)` dataset과
분리되어 있다. Schema는 `(N, 50, 10)`이며 FSR 4채널 뒤에 left-foot
accelerometer XYZ와 gyroscope XYZ가 이어진다.

```text
Terrain parameters
      ↓
MuJoCo full-body G1
      ↓
FSR4 + left-foot IMU6
      ↓
Domain variation
      ↓
Sensor imperfections
      ↓
Canonical medium-response window
      ↓
Dataset v1 pilot
      ↓
RandomForest baseline
```

MuJoCo physics ground truth는 friction, slip, normal load, load distribution,
CoP-related response, surface geometry response, low-frequency foot dynamics로
구성된다. Virtual sensor 입력은 FSR4와 ankle-mounted foot IMU6이다. Pelvis
IMU와 raw contact diagnostic은 diagnostic 전용이며 AI tensor 밖에 별도로
저장한다. Sensor imperfection은 재현 가능한 pilot engineering assumption이며
KIT_PSE84_AI/BMI270의 측정 calibration 값이 아니다.

AI 입력에서는 timestep-sensitive raw high-frequency vibration PSD, 정확한
dominant vibration frequency, micro-contact spectral feature를 의도적으로
제외한다. Clean/noisy tensor를 별도로 저장하며 train/validation/test의 surface
realization과 session은 서로 겹치지 않는다.

`simulation/unitree_mujoco/simulate_python`에서 실행한다.

```bash
/d/shin/Infineon/simulation/venv/bin/python -m pip install -r requirements-dataset-v1.txt
/d/shin/Infineon/simulation/venv/bin/python run_terrain_dataset_v1_pilot.py
```

생성 데이터는 gitignored 경로인
`simulation/outputs/terrain_dataset_v1_pilot/`에 저장한다. Runner는 비어 있지
않은 output directory를 덮어쓰지 않는다.

## Pilot 결과

| 항목 | 결과 |
|---|---:|
| Candidate runs | 1,200 |
| Valid windows | 1,189 |
| Tensor | `(1189, 50, 10)` |
| FSR-only unseen-surface test accuracy | 84.81% |
| IMU-only unseen-surface test accuracy | 97.47% |
| Fusion unseen-surface test accuracy | 96.20% |
| Concrete recall, fusion | 96.7% |
| Marble recall, fusion | 90.0% |
| Concrete-Marble mutual confusion, fusion | 5.8% |

Fusion은 FSR-only보다 우수하지만, **이 pilot만으로 Fusion이 IMU-only보다
우수하다고 결론낼 수 없다**. 이 한계는 후속 expanded milestone에서 unseen
procedural surface family와 동일한 small 1D-CNN architecture를 사용해 다시
평가했다. 완료 결과는 `TERRAIN_DATASET_V1_HISTORICAL_RESULTS.md`를 참조한다.

## 상태 경계

- **CURRENT / LOCKED:** terrain label 4개, FSR4 + foot IMU6,
  `medium_response`, `(50, 10)`, surface-disjoint split, low-frequency MuJoCo
  response
- **DIAGNOSTIC-ONLY:** pelvis IMU, slip/contact trace, collision flag, raw clean
  simulation log
- **DEPRECATED / SUPERSEDED for Dataset v1:** pelvis IMU AI 입력, raw MuJoCo
  high-frequency vibration/PSD 직접 사용, synthetic `(50, 5)`와 MuJoCo
  `(50, 10)` data 혼합
- **OPTIONAL / DEFERRED:** virtual high-frequency vibration sensor model, exact
  PSD feature, gait-based collection, real sensor calibration
