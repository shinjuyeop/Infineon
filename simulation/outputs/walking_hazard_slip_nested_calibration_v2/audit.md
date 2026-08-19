# Walking Slip nested calibration + blind validation v2

The fixed grid was selected with three leave-one-variation-out folds over the
immutable d2209cd v00/v01/v02 development pool.  Whole runs and their contact
episodes stayed together.  No test/final data or v03/v04/v05 outer trace was
available to the selector.  Selection completed before outer acquisition.

The selected dataset-generation physical label-oracle is
`0.05 m / 3 ms`.
All-fold-pass candidates first met zero normal FP, all-speed/all-run physical
positive detection, and zero AIR/post-fall/touchdown violations.  A sufficient
margin band then admitted candidates within both 80% and 5 mm of the best
worst-fold margin; worst-fold p95 and mean latency decided next.  Longer
persistence received no preference.

The blind outer matrix contains 36 new 3-second,
1-kHz runs: 27 normal and 9 Ice.  v03/v04/v05 use new phase fractions and
command onsets, with observed first commands recorded in the manifest.

- Outer normal false-positive runs: 0
- Outer valid Ice physical-source runs: 9/9
- Outer Ice detections by speed: {'0.10': 3, '0.15': 3, '0.20': 3}
- AIR/post-fall/touchdown violations: 0
- Duplicate pairs: 0
- Leakage: 0
- Failure reasons: []

Latency is reported per contact episode and by speed/variation.  Physical
label onset is distinct from model inference: no model inference occurred.

## Readiness gates

- WALKING_SLIP_NESTED_CALIBRATION_READY=true
- WALKING_SLIP_OUTER_VARIATION_DIVERSITY_READY=true
- WALKING_SLIP_OUTER_VALIDATION_READY=true
- WALKING_SLIP_LABEL_ORACLE_FREEZE_READY=true
- WALKING_ORACLE_ROBUSTNESS_READY=true
- WALKING_BOUNDED_RETRAINING_READY=true

The Slip oracle is frozen only as a physical label-oracle for dataset
generation.  The d2209cd Sink candidate is carried forward without selection
or recollection.  No runtime threshold, trained model, INT8/E84 artifact,
frozen detector, or System-v1 behavior changed.
