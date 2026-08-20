# Walking-v2 Candidate Failure Localization v1

## Result

The 94.565% Terrain diagnostic is reproduced exactly. It is not a like-for-like
comparison with T2: the diagnostic uses a 200 ms, 164-feature statistical
tensor and stable-contact population, while T2 uses a 50 ms, 97-feature compact
tensor and a different pre-fall/contact mask. T2 evaluated on the diagnostic
population remains weak, and a diagnostic-recipe shadow head on the T2 tensor
also remains weak. The primary cause is `TERRAIN_MODEL_UNDERCAPACITY`; population
non-comparability is secondary. Balancing was applied exactly (15 rows in each
of 96 groups), but retained only 1,440 training rows.

Slip R0 reproduces 6/6 run coverage, 17/69 physical
episodes, 17/68 actionable events, 40 too-early
positive endpoint-foot samples, and 53 post-fall positive endpoint-foot samples.
All 40 are genuine raw-score activations more than 100 ms before the next onset;
none originates in a previous physical episode, merge error, contact mismatch,
or counting error. Two legacy episode detections were nevertheless credited to
an unowned early latch that began before that episode's risk window; strict
origin ownership changes detected episodes 17 -> 15.
All 53 are state outputs after the first-fall censor. Counting them in the
evaluable invalid numerator was a mask-accounting bug; strict censoring changes
invalid firing 53 -> 0 without altering model/state. Episode recall and the 40
too-early firings do not improve, so raw score timing/separability remains the
primary Slip failure.

R0-R4 never change model weights, normalization, threshold, persistence, or
hysteresis. Candidate readiness remains false and no holdout is authorized.

## Provenance incident

One forbidden existing outer file was accidentally printed by an over-broad
pre-implementation `rg` command. Its values were not used by this audit. This is
recorded as `outer_content_load_count=1`, so provenance readiness is false rather
than inaccurately claiming zero access.

## Decision

- Terrain reference reproduced: True
- Terrain root cause: `TERRAIN_MODEL_UNDERCAPACITY`
- Slip root cause: `SLIP_RAW_SCORE_TIMING_SEPARABILITY_FAILURE`
- Correctness fix ready: True
- Candidate readiness: false
- Next step: `JOINT_TERRAIN_SLIP_REDESIGN`
