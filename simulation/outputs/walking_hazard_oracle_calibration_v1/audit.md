# Walking Slip/Sink physical oracle robustness + calibration v1

Fifty-four 3-second, 1-kHz train-only runs used three deterministic but
physically distinct initial policy conditions.  No test/final data, model
training, production threshold change, INT8/E84 export, or System-v1 change
was performed.

Replicate variation combined gait phase fractions 0, 1/3 and 2/3 with command
onset delays 0, 20 and 40 ms.  This adds no sensor or label noise.  Pairwise
full endpoint hashes found 0 duplicate traces.

The selected Slip audit candidate is {"persistence_ms": 20, "production_frozen": false, "threshold_m": 0.075}; the
selected Sink audit candidate is {"persistence_ms": 20, "production_frozen": false, "threshold_m": 0.0055}.  These are
**not frozen production thresholds**.

Train/validation envelopes:

```json
{
  "slip_calibration_train": {
    "normal_run_max_m": 0.016771160430382383,
    "positive_run_min_max_m": 0.1303936943976871
  },
  "slip_calibration_validation": {
    "normal_run_max_m": 0.013210608737598172,
    "positive_run_min_max_m": 0.07441127363861753
  },
  "sink_calibration_train": {
    "normal_run_max_m": 0.004294271603264094,
    "positive_run_min_max_m": 0.006403968922677472
  },
  "sink_calibration_validation": {
    "normal_run_max_m": 0.0036307742598676723,
    "positive_run_min_max_m": 0.006053937709824361
  }
}
```

Validation normal false-positive runs: 0.
Slip detected speeds: [0.1, 0.15].
Slip missing speeds: [0.2].
Sink all-speed profiles: ['sand_solref_interpolation_1of3', 'sand_solref_interpolation_2of3'].
AIR/post-fall/touchdown-transient violations: 0.
Split leakage: 0.
Failure reasons: ['held-out Slip validation had no detection at speeds 0.20 m/s'].

## Readiness gates

- WALKING_GT_REPLICATE_DIVERSITY_READY=true
- WALKING_SLIP_ORACLE_CALIBRATION_READY=false
- WALKING_SINK_ORACLE_CALIBRATION_READY=true
- WALKING_ORACLE_SPLIT_INTEGRITY_READY=true
- WALKING_ORACLE_ROBUSTNESS_READY=false
- WALKING_BOUNDED_RETRAINING_READY=false

Threshold and persistence values remain bounded calibration candidates only.
