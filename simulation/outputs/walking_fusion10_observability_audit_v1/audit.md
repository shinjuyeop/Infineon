# Walking Fusion10 observability / label-task compatibility audit v1

This is a diagnostic audit, not a production-candidate approval. Development
was limited to d2209cd v00/v01 training and v02 validation. Locked outer and
spatial arrays were not opened, selected against, or rerun.

## Terrain

- Conclusion: **OBSERVABLE_WITH_CURRENT_INPUT**; secondary **CONTROLLED_WALKING_DOMAIN_CONFLICT**.
- Frozen walking-v02 macro/worst/majority: 0.253 / 0.000 / 0.847.
- Float-candidate walking-v02 macro/worst/majority: 0.253 / 0.000 / 0.849.
- Best fixed simple probe (class_phase_balanced) macro/worst/majority: 0.779 / 0.640 / 0.435.
- Sand collapse cause: not a Fusion10 information absence: the balanced simple probe separates walking classes; the existing static-retention objective and Sand-heavy/midstance-heavy walking distribution admit collapse.
- Locked outer float result remains a failure: accuracy 0.280, macro 0.272, Concrete recall 0, Marble recall 0.017, Ice recall 0.108, Sand recall 0.963.

## Slip

- Conclusion: **LABEL_TASK_INCOMPATIBLE**; secondary **OBSERVABLE_WITH_LONGER_HISTORY**.
- Best fixed window: 100 ms; run recall 1.000; normal FP 0; anticipation 3; median latency 0.0 ms.
- Pre-oracle firing is classified as causal precursor evidence, not input leakage. Confirmed-event and risk-horizon objectives are simultaneously compatible: False.
- Locked outer candidate remains unapproved: 9/9 detected, 6/27 normal FP, 3 anticipation events.

## Sink

- Conclusion: **OBSERVABLE_WITH_LONGER_HISTORY**; secondary **LABEL_TASK_INCOMPATIBLE**.
- Best raw window: 20 ms; run recall 0.000; normal FP 0.
- Best stateful window: 200 ms; run recall 1.000; normal FP 9; median latency 5.0 ms.
- Touchdown-scoped stateful reference required: True; sufficient by itself: False.
- Zero normal FP and at least 50% recall coexist only with precursor/latency trade-offs: True; confirmed semantics without anticipation: False.
- Locked holdout remains unapproved: 2/18 detected, median physical-oracle-to-stable latency about 1.48 s.

## Readiness gates

- UPSTREAM_ARTIFACT_SHA_READY: PASS
- RUN_EPISODE_SPLIT_LEAKAGE_ZERO: PASS
- FUTURE_SAMPLE_LEAKAGE_ZERO: PASS
- AIR_TOUCHDOWN_POSTFALL_MASK_INVARIANT: PASS
- EXACT_1KHZ_TIMESTAMP_SCHEMA_READY: PASS
- DETERMINISTIC_RELOAD_PARITY_READY: PASS
- OUTER_NON_ACCESS_READY: PASS
- TERRAIN_CURRENT_WALKING_READY: FAIL
- TERRAIN_DIAGNOSTIC_OBSERVABLE: PASS
- SLIP_CONFIRMED_AND_PRECURSOR_COMPATIBLE: FAIL
- SINK_ZERO_FP_EFFECTIVE_RECALL_READY: PASS
- SINK_CONFIRMED_SEMANTICS_READY: FAIL
- WALKING_BOUNDED_RETRAINING_V2_AUTHORIZED: FAIL
- WALKING_INT8_PREPARATION_AUTHORIZED: FAIL
- PRODUCTION_CANDIDATE_READY: FAIL

Exactly one next step: **stateful detector prototype**.

Production/frozen models, production normalizations, runtime thresholds,
physical-oracle semantics, INT8/Vela, E84/HIL, System-v1, and final-test
artifacts were not changed.
