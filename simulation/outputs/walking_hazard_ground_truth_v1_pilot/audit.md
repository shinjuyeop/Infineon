# Walking physical hazard ground-truth pilot v1

This checkpoint records physical observables only.  Terrain identity was not
used as a Slip or Sink label, AIR was excluded, all evidence at and after the
first sampled fall was censored, and no model or detector threshold was
trained or frozen.

## Acquisition

- Runs: 17 (9 normal hard-negative,
  3 Slip candidate, 5 Sink candidate)
- Candidate runs with a recorded fall: 3
- Samples censored from first fall onward: 2996

## Slip evidence

The maximum normal pre-fall, post-touchdown contact-anchor drift was
0.008933439521696398 m.  The minimum
Ice candidate run maximum was 0.1303936943976871 m.
Separation is therefore **observed**.
This is an envelope comparison, not a frozen detector threshold.

## Sink evidence

The normal maximum post-touchdown contact penetration was
0.0037352325661355558 m and the normal maximum
surface-relative sole depth was 0.0037352325661355592 m.
Profiles with stable, pre-fall measurements separated from either normal
envelope: sand_solref_interpolation_1of3, sand_solref_interpolation_2of3, sand_native, sand_slightly_compliant, sand_moderately_compliant.

## Readiness gates

- WALKING_HAZARD_GT_RECORDER_READY=true
- WALKING_FIRST_FALL_CENSOR_READY=true
- WALKING_CONTACT_ANCHOR_DRIFT_READY=true
- WALKING_TERRAIN_RELATIVE_PENETRATION_READY=true
- WALKING_SLIP_POSITIVE_ACQUISITION_READY=true
- WALKING_SINK_POSITIVE_ACQUISITION_READY=true
- WALKING_HARD_NEGATIVE_SOURCE_READY=true
- WALKING_BOUNDED_RETRAINING_AUTHORIZED=true

Even if the final authorization gate is true, retraining is intentionally not
part of this checkpoint.
