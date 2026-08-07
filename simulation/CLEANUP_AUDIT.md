# Terrain-classification Repository Cleanup Audit

최초 audit 기준은 `main`의 `66b33ec`이며 이후 expanded dataset/CNN/INT8
milestone source가 추가됐다. 이 문서는 simulation behavior나 research history를
변경하지 않고 execution graph와 repository 경계를 설명한다.

## 상위 폴더 분류

| Path | 분류 | 현재 역할 |
|---|---|---|
| `README.md` | DOCS | Workspace entry point와 현재 pipeline 요약 |
| `simulation/` | ACTIVE / HISTORICAL / GENERATED / EXTERNAL | MuJoCo terrain 작업과 experiment history |
| `simulation/unitree_mujoco/` | EXTERNAL + CUSTOM overlay | Unitree MuJoCo fork와 repository-specific G1 sensor/terrain 도구 |
| `simulation/unitree_mujoco/simulate_python/` | ACTIVE / HISTORICAL | Current pipeline과 보존된 experiment runner/analyzer |
| `simulation/unitree_mujoco/simulate_python/test/` | TEST | Automated terrain regression과 manual HIL/gamepad check |
| `simulation/unitree_sdk2_python/` | EXTERNAL / VENDOR | Simulator bridge가 사용하는 Unitree SDK2 Python dependency |
| `simulation/outputs/` | GENERATED | Gitignored dataset, model, plot, experiment result |
| `simulation/models/g1_lower_body/` | HISTORICAL / CUSTOM | 보존된 lower-body comparison model |
| `simulation/*.md` | DOCS | Dataset contract, handoff, result, cleanup, next milestone |
| `Infineon_HIL/` | LEGACY for Dataset v1 / USEFUL HOST CODE | 별도의 synthetic `(50,5)` host/HIL pipeline |
| `Infineon_test/` | EXTERNAL / TARGET EXAMPLES | KIT_PSE84_AI/ModusToolbox source와 vendor example |
| `Infienon_documents/` | LOCAL / PRIVATE DOCS | 의도적으로 ignored된 local 자료 |

```text
Infineon/
  README.md                              DOCS
  simulation/
    *.md                                 DOCS
    outputs/                             GENERATED (ignored)
    models/g1_lower_body/                HISTORICAL / CUSTOM
    unitree_mujoco/                      EXTERNAL + CUSTOM overlay
      unitree_robots/g1/                 ACTIVE model + upstream assets
      simulate_python/                   ACTIVE + HISTORICAL
        test/                            TEST
    unitree_sdk2_python/                 EXTERNAL / VENDOR
  Infineon_HIL/                          LEGACY for Dataset v1
  Infineon_test/                         EXTERNAL / TARGET EXAMPLES
  Infienon_documents/                    LOCAL / PRIVATE DOCS
```

## Current Dataset v1 execution graph

### Simulation 및 dataset core

- `config.py`: shared simulator configuration
- `unitree_sdk2py_bridge.py`: Unitree SDK-to-MuJoCo bridge
- `controlled_excitation.py`: deterministic support/excitation control
- `hil_sensor.py`: FSR4 + left-foot IMU6 HIL vector; pelvis IMU diagnostic
- `terrain_profiles.py`: four-class terrain/contact profile
- `surface_profiles.py`: 기존 procedural surface representation
- `pulse_windows.py`: `medium_response` 포함 pulse-aligned window
- `slip_diagnostics.py`: slip/contact diagnostic
- `run_controlled_terrain_dataset.py`, `run_horizontal_pulse_dataset.py`,
  `run_surface_sampling_rate_study.py`: current runner가 계속 사용하는 historical
  utility와 scene constant

### Pilot 및 expanded milestone

- `terrain_dataset_v1.py`, `run_terrain_dataset_v1_pilot.py`: pilot schema와
  RandomForest baseline
- `expanded_terrain_dataset_v1.py`: family definition, manifest, cost model
- `run_expanded_terrain_dataset_v1.py`: 4,480-candidate generator
- `terrain_cnn.py`, `train_terrain_1d_cnn.py`: compact CNN ablation
- `terrain_int8.py`, `export_terrain_int8.py`: strict INT8 export와 host parity

Active non-Python input은 `unitree_robots/g1/scene_surface_study.xml`,
`unitree_robots/g1/g1_29dof.xml`, robot asset, Dataset/CNN requirement 파일이다.
Root `unitree_mujoco.py`는 upstream interactive simulator이며 Dataset v1 entry
point가 아니다.

현재 경로:

```text
Terrain -> MuJoCo full-body G1 -> FSR4 + Foot IMU6
        -> domain variation -> sensor noise -> medium_response
        -> (50, 10) -> surface-family-disjoint split
        -> compact CNN -> strict INT8 host candidate
```

## Historical experiment 분류

Historical은 삭제 가능하다는 뜻이 아니다. 일부 파일은 current utility도
제공한다.

| Experiment family | Primary source | 보존 이유 |
|---|---|---|
| Passive terrain | `run_hil_sensor_demo.py`, `analyze_terrain_data.py`, `analyze_controlled_terrain_data.py` | 초기 feasibility와 diagnostic 기록 |
| Horizontal pulse / controlled excitation | `run_controlled_terrain_dataset.py`, `run_horizontal_pulse_dataset.py`, `sweep_horizontal_pulse.py`, analyzer | 80 N, 70% support protocol 및 current dependency |
| Bidirectional validation | `run_bidirectional_pulse_validation.py`, analyzer, `pulse_windows.py` | Direction/window 비교와 `medium_response` 근거 |
| Foot IMU A/B | `run_imu_location_ab_validation.py`, `hil_sensor.py`, G1 XML | Foot IMU 선택 근거 |
| Surface/sampling | `run_surface_sampling_rate_study.py`, 관련 test | Timestep/sampling audit와 active helper |
| Friction x roughness | `run_surface_factorization_study.py`, analyzer | Material factor 분리 근거 |
| Timestep convergence | timestep runner/analyzer/test | Low-frequency validity 경계 |
| Final 0.25 ms convergence | final runner/wrapper/analyzer | Raw high-frequency vibration limitation |
| Lower-body comparison | build/validate/symmetry script와 `models/g1_lower_body/` | Full-body 선택 근거와 comparison asset |

Generated result는 `simulation/outputs/`의 대응 directory에 보존한다. Pilot,
expanded dataset, CNN, INT8 output을 historical result와 섞거나 덮어쓰지 않는다.

## Legacy 파일

- `Infineon_HIL/`은 usable하지만 별도의 synthetic `(50,5)` pipeline이다.
- `analyze_terrain_data.py`의 pelvis-labelled plot은 passive experiment era의
  재현 기록이다.
- Pelvis IMU sensor는 diagnostic이므로 삭제할 legacy file이 아니다.
- Historical run/analyze script는 current training entry point가 아니어도
  research provenance를 위해 유지한다.

## Generated 및 safe-to-delete

안전한 housekeeping 대상은 재생성 가능한 local by-product로 제한한다.

- Virtual environment 밖의 `__pycache__/`, `.pytest_cache/`, `*.pyc`, `*.pyo`
- Editor backup `*~`, `.DS_Store`
- Component `.gitignore`에 포함된 build/test product

Source, test, 문서, dataset result, model, robot asset, virtual environment는
cleanup 대상으로 보지 않는다. `simulation/outputs/` 전체는 root `.gitignore`로
제외한다. `Infineon_HIL`의 generated NPZ/model/report/plot도 component
`.gitignore`를 따른다.

## 중복 및 coupling audit

| 항목 | 현재 삭제하지 않는 이유 | 향후 권장 |
|---|---|---|
| Controlled/pulse/surface/bidirectional CSV helper 중복 | Current runner import와 historical format compatibility | Versioned schema adapter를 가진 공통 `io.py` |
| Analyzer와 dataset의 separation/stat helper 중복 | Historical metric 차이와 result reproducibility | Parity test 후 versioned metrics module |
| Dataset runner가 historical surface runner constant/helper import | 이동하면 current entry point가 깨짐 | Neutral core로 이동하고 compatibility re-export |
| Final 0.25 ms wrapper assessment coupling | Final study를 정확히 보존 | Convergence 재개 시 explicit policy argument |
| Passive analyzer의 pelvis label | 당시 experiment를 정확히 기록 | Historical package 이동 시 이름 명확화 |
| Dataset v1와 `Infineon_HIL` requirements 분리 | 서로 다른 schema/environment | Top-level environment matrix 문서화, 자동 병합 금지 |
| Manual test와 automated test 혼재 | Operator workflow 이름 보존 | `tools/manual/` 또는 test README |

확실하게 dead인 Python source는 없다. TODO나 duplication은 maintenance
candidate일 뿐 cleanup 승인으로 해석하지 않는다.

## Vendor/custom 경계

- `simulation/unitree_mujoco/` 대부분과 `simulation/unitree_sdk2_python/`
  전체를 upstream/vendor로 취급한다.
- G1 sensor XML 수정, `scene_surface_study.xml`, terrain runner/analyzer,
  Dataset/CNN/INT8 module과 terrain test는 Infineon custom layer이다.
- `Infineon_test/` BSP, generated documentation, example은 명시적인 project
  수정이 확인되기 전 target/vendor material로 취급한다.
- Upstream license, README, model-card, asset attribution은 그대로 보존한다.

## 향후 권장 구조 — 현재는 이동하지 않음

```text
simulation/
  docs/terrain/
  terrain_pipeline/
    core/                       # sensors, profiles, excitation, windows
    datasets/                   # schema, generation, split
    experiments/                # versioned historical runner
    analysis/                   # metrics and reports
    deployment/                 # CNN, INT8, E84 adapters
    io/                         # schema-aware artifact helpers
  tests/terrain/
  models/g1_lower_body/
  outputs/                      # ignored
  unitree_mujoco/               # upstream + minimal integration overlay
  unitree_sdk2_python/          # upstream dependency
```

Import/numerical parity test를 먼저 확보한 뒤 experiment family 하나씩
이동한다. Generated output을 이동하거나 legacy `(50,5)`를 Dataset v1과
결합하지 않는다.

## Git hygiene 상태

- `simulation/outputs/` 아래 파일은 tracked되지 않는다.
- Dataset array, CSV, plot, Keras/TFLite model은 새로 tracked되지 않는다.
- 가장 큰 tracked 파일은 예상된 Unitree robot mesh와 Infineon BSP 문서
  asset이며 generated terrain dataset이 아니다.
- Source cleanup은 active utility coupling을 제거하고 parity-test하기 전까지
  cache-only 원칙을 유지한다.
