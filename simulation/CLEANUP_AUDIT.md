# Terrain-classification repository cleanup audit

Audit baseline: `main` at `66b33ec` (`Document terrain classification handoff`).
This audit is intentionally non-destructive: it documents the current execution
graph and repository boundaries without changing simulation behavior or moving
research history.

## Top-level classification

| Path | Classification | Current role |
|---|---|---|
| `README.md` | DOCS | Workspace entry point and current pipeline summary. |
| `simulation/` | ACTIVE / HISTORICAL / GENERATED / EXTERNAL | MuJoCo terrain work, experiment history, ignored results, and vendored dependencies. |
| `simulation/unitree_mujoco/` | EXTERNAL with a CUSTOM overlay | Unitree MuJoCo fork. G1 sensors, terrain scenes, runners, analyses, and tests are repository-specific additions. Preserve upstream licenses. |
| `simulation/unitree_mujoco/simulate_python/` | ACTIVE / HISTORICAL | Current Dataset v1 modules and preserved experiment runners/analyses share this directory. |
| `simulation/unitree_mujoco/simulate_python/test/` | TEST | Automated MuJoCo terrain tests plus manual gamepad/HIL checks. |
| `simulation/unitree_sdk2_python/` | EXTERNAL / VENDOR | Unitree SDK2 Python dependency used by the simulator bridge. |
| `simulation/outputs/` | GENERATED | Ignored experiment and Dataset v1 outputs. Existing research results are retained locally and must not be overwritten. |
| `simulation/models/g1_lower_body/` | HISTORICAL / CUSTOM | Preserved lower-body comparison model and scene; not the production candidate. |
| `simulation/*.md` | DOCS | Dataset contract, experiment handoff, cleanup audit, and next milestone plan. |
| `Infineon_HIL/` | LEGACY relative to Dataset v1; still useful host code | Separate synthetic `(50, 5)` host/HIL pipeline. It must not be mixed with MuJoCo Dataset v1 `(50, 10)`. |
| `Infineon_test/` | EXTERNAL / TARGET EXAMPLES | KIT_PSE84_AI/ModusToolbox sources and vendor examples, not the current MuJoCo dataset generator. |
| `Infienon_documents/` | LOCAL / PRIVATE DOCS | Intentionally ignored local material (directory spelling retained for compatibility). |

```text
Infineon/
  README.md                              DOCS
  simulation/
    *.md                                 DOCS
    outputs/                             GENERATED (ignored)
    models/g1_lower_body/                HISTORICAL / CUSTOM
    unitree_mujoco/                      EXTERNAL + CUSTOM overlay
      unitree_robots/g1/                 ACTIVE model/assets + upstream assets
      simulate_python/                   ACTIVE + HISTORICAL
        test/                            TEST
    unitree_sdk2_python/                 EXTERNAL / VENDOR
  Infineon_HIL/                          LEGACY for Dataset v1; useful host code
  Infineon_test/                         EXTERNAL / TARGET EXAMPLES
  Infienon_documents/                    LOCAL / PRIVATE DOCS (ignored)
```

## Current Dataset v1 execution graph

The current production-candidate entry point is
`simulation/unitree_mujoco/simulate_python/run_terrain_dataset_v1_pilot.py`.
Its recursive local Python import graph contains 13 active modules:

| Active module | Role |
|---|---|
| `run_terrain_dataset_v1_pilot.py` | Pilot orchestration, matched candidate generation, output summaries. |
| `terrain_dataset_v1.py` | Dataset schema, windows, noise/domain variation, split and feature assembly. |
| `terrain_profiles.py` | Four-class low-frequency terrain/contact parameter profiles. |
| `surface_profiles.py` | Procedural surface-family representation used by Dataset v1. |
| `hil_sensor.py` | Stable FSR4 + left-foot IMU6 HIL vector; pelvis IMU remains diagnostic. |
| `pulse_windows.py` | Pulse/response window definitions, including `medium_response`. |
| `controlled_excitation.py` | Deterministic support and excitation control. |
| `slip_diagnostics.py` | Slip/contact diagnostic calculations. |
| `config.py` | Simulator configuration shared by the runners. |
| `unitree_sdk2py_bridge.py` | Unitree SDK-to-MuJoCo bridge used by controlled excitation. |
| `run_controlled_terrain_dataset.py` | Shared legacy runner utilities still imported by active runners. |
| `run_horizontal_pulse_dataset.py` | Horizontal-pulse mechanics and CSV helpers used transitively. |
| `run_surface_sampling_rate_study.py` | `SURFACE_SCENE_PATH`, spectral helpers, and CSV writing currently imported by Dataset v1. |

Active non-Python inputs are
`unitree_robots/g1/scene_surface_study.xml`,
`unitree_robots/g1/g1_29dof.xml`, and
`simulate_python/requirements-dataset-v1.txt` plus their referenced robot assets.
The root `unitree_mujoco.py` is the upstream interactive simulator entry point,
not the Dataset v1 entry point.

The current path is therefore:

```text
Terrain -> MuJoCo full-body G1 -> FSR4 + Foot IMU6
        -> domain variation -> sensor noise -> medium_response
        -> (50, 10) -> surface-family-disjoint split -> classifier
```

## Historical experiment mapping

These nine experiment families remain useful evidence. “Historical” does not
mean safe to delete: several files also supply utilities to the current path.

| Experiment family | Primary source files | Status / retained conclusion |
|---|---|---|
| Passive terrain | `run_hil_sensor_demo.py`, `analyze_terrain_data.py`, `analyze_controlled_terrain_data.py` | Early feasibility/diagnostic work. Plot labels in `analyze_terrain_data.py` describe the then-current pelvis IMU and are not Dataset v1 labels. |
| Horizontal pulse / controlled excitation | `run_controlled_terrain_dataset.py`, `run_horizontal_pulse_dataset.py`, `sweep_horizontal_pulse.py`, `analyze_horizontal_pulse_dataset.py` | Established the matched 80 N, 70% support protocol. Parts remain active dependencies. |
| Bidirectional validation | `run_bidirectional_pulse_validation.py`, `analyze_bidirectional_pulse_validation.py`, `pulse_windows.py` | Compared pulse direction and response windows; `medium_response` is retained. |
| Foot IMU location A/B | `run_imu_location_ab_validation.py`, `hil_sensor.py`, G1 XML sensor sites | Selected FSR4 + foot IMU6; pelvis IMU retained for diagnostics. |
| Surface/sampling study | `run_surface_sampling_rate_study.py`, `test/test_surface_sampling_study.py` | Audited timestep/sampling and surface-aware response. Module still provides active constants/helpers. |
| Friction x roughness factorization | `run_surface_factorization_study.py`, `analyze_surface_factorization.py` | Separated friction and roughness effects; high-frequency vibration is deferred. |
| Timestep convergence | `run_timestep_convergence_study.py`, `analyze_timestep_convergence.py`, `test/test_timestep_convergence.py` | Preliminary physics-step convergence evidence. |
| Final 0.25 ms convergence | `run_final_timestep_convergence.py`, `run_final_025ms_convergence.py`, `analyze_final_timestep_convergence.py` | Final limitation study; raw MuJoCo high-frequency vibration was not promoted to a production feature. |
| Lower-body comparison | `build_g1_lower_body_model.py`, `validate_g1_lower_body_model.py`, `run_g1_lower_body_symmetry_smoke.py`, `analyze_g1_lower_body_symmetry.py`, `simulation/models/g1_lower_body/` | Full-body G1 remained the production candidate; comparison assets are preserved. |

The generated result directories under `simulation/outputs/` are the matching
historical records. Dataset v1 pilot files live in their own ignored output
directory and are not intermingled with the earlier studies.

## Legacy files

- `Infineon_HIL/` remains a usable but separate synthetic `(50, 5)` pipeline;
  none of its arrays or models are valid Dataset v1 `(50, 10)` inputs.
- `analyze_terrain_data.py` and its pelvis-labelled plots belong to the passive
  experiment era and are retained for reproducibility.
- Historical run/analyze scripts and their ignored output directories are
  research records, not current training entry points.
- Pelvis IMU sensors in the G1 model are intentionally retained as diagnostics;
  they are not legacy files to delete.

## Generated and safely removable files

Safe housekeeping is limited to reproducible local by-products:

- `__pycache__/`, `.pytest_cache/`, `*.pyc`, and `*.pyo` outside virtual environments;
- editor backup files (`*~`) and `.DS_Store`;
- generated build/test products already covered by component `.gitignore` files.

During this audit, Python and pytest caches were removed. No source, test,
documentation, dataset, result, model, robot asset, or virtual environment was
deleted. Running tests recreates caches; they may be removed again after tests.

`simulation/outputs/` is covered by the repository root `.gitignore` and must
remain ignored. `Infineon_HIL/data/*.npz`, generated model formats, reports, and
plots are covered by `Infineon_HIL/.gitignore`; only intentional `.gitkeep`
placeholders are tracked.

## Duplication and coupling audit

| Finding | Why it is not changed now | Future consolidation |
|---|---|---|
| CSV readers/writers are repeated in controlled, pulse, surface, and bidirectional scripts. | Some are imported by the current Dataset v1 execution graph; changing them risks historical-format compatibility. | Introduce one tested `io.py` with explicit schema/version adapters. |
| Separation/statistical helpers are repeated across analyzers and `terrain_dataset_v1.py`. | Historical metrics may intentionally differ and published summaries must remain reproducible. | Add a versioned metrics module, migrate one experiment at a time, and parity-test old outputs. |
| Dataset v1 imports `SURFACE_SCENE_PATH` and `write_dict_rows` from a historical study runner. | Deleting or moving that runner would break the current entry point. | Move stable scene/config and I/O primitives into a neutral core module while keeping compatibility re-exports. |
| `run_final_025ms_convergence.py` wraps the generic final convergence runner and applies the terminal assessment. | The coupling is small and preserves the exact final study. | Make the assessment policy an explicit argument or data file if convergence work resumes. |
| `analyze_terrain_data.py` contains pelvis-IMU labels. | It correctly documents the old passive experiment; editing labels would rewrite history. | Move it under a clearly named historical experiment package when imports can be preserved. |
| Requirements are split between Dataset v1 and `Infineon_HIL`. | They describe different pipelines and environments. | Add a documented top-level environment matrix; do not silently merge incompatible stacks. |
| `gamepad_test.py` and `hil_sensor_test.py` are manual/hardware checks beside automated tests. | Renaming may break documented operator workflows. | Mark manual checks in a test README or move them to `tools/manual/`. |

No dead Python source can be removed with high confidence. TODO-like items and
duplication are maintenance candidates, not cleanup authorization.

## Vendor/custom boundary

- Treat the bulk of `simulation/unitree_mujoco/` and all of
  `simulation/unitree_sdk2_python/` as upstream/vendor code.
- Treat G1 sensor XML edits, `scene_surface_study.xml`, terrain runners,
  analyses, Dataset v1 modules, and terrain tests as the Infineon custom layer.
- Treat `Infineon_test/` BSP, generated documentation, and example application
  content as target/vendor material unless a later task identifies a deliberate
  project modification.
- Preserve upstream license, README, model-card, and asset attribution files.

## Recommended future structure (not implemented)

A later behavior-preserving refactor can make the boundary explicit while
keeping thin compatibility wrappers at today's entry points:

```text
simulation/
  docs/terrain/                 # handoff, history, cleanup, milestone docs
  terrain_pipeline/
    core/                       # sensors, profiles, excitation, windows, diagnostics
    datasets/                   # Dataset v1 schema/generation/splits
    experiments/                # versioned historical runners
    analysis/                   # versioned metrics and reports
    io/                         # schema-aware CSV/artifact helpers
  tests/terrain/                # automated MuJoCo terrain tests
  models/g1_lower_body/         # preserved comparison model
  outputs/                      # ignored generated results
  unitree_mujoco/               # upstream simulator + minimal integration overlay
  unitree_sdk2_python/          # upstream dependency
```

Refactor gate: capture import and numerical parity tests first, then migrate one
experiment family per commit. Do not move generated outputs or combine the old
`(50, 5)` schema with Dataset v1.

## Git hygiene result

- No file under `simulation/outputs/` is tracked.
- No newly generated Dataset v1 array, CSV, plot, report, or trained model is tracked.
- The largest tracked files are expected Unitree robot meshes and Infineon BSP
  documentation assets (the largest robot mesh is about 16.7 MB), not generated
  terrain datasets.
- Cleanup should remain cache-only until the active utility coupling above has
  been removed and parity-tested.
