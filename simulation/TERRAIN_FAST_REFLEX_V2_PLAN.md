# Terrain Fast Reflex v2 experiment plan

Status: **planning only**. No v2 dataset was generated, no detector was trained,
and no new test evaluation was run. v1 artifacts remain immutable research
records.

## Preserved v1 result

Slip v1 remains the frozen 5 ms GAP+GlobalMax candidate, threshold
`0.7317748725`, persistence 3. Its one-shot final result is **Slip Host Final
Gate FAIL**: test run recall 64/65 (98.462%) and pre-onset run FPR 9/120
(7.50%). This is not reinterpreted as a pass.

Sand v1 remains a validation-only failure. Its best combined path obtained
75.56% Sand recall, 26.67% Tilt-only recall, and 5.00% run FPR at 50 ms. No
Sand test split was used.

## Slip v2: preserved pre-onset audit

The read-only audit uses the already recorded stable-firing timestamps in the
frozen v1 test artifact. It does not rerun the detector or select a new policy.
There are nine pre-onset firing runs:

| Classification | Runs | Meaning |
|---|---:|---|
| True false alarm | 2 | Full valid trace has no later canonical slip onset. |
| Incipient-slip candidate | 7 | A preserved stable firing is followed by the existing canonical slip onset. |
| Ambiguous | 0 | No invalid or chronologically inconsistent trace. |

All seven incipient candidates are `marble_to_sand`; their firing-to-confirmed
lead is median 21 ms and p95 39.2 ms. At firing, median horizontal foot speed
is 0.000972 m/s; at the already preserved canonical onset it is 0.001821 m/s.
The latter is near the v1 canonical speed threshold of 0.001771 m/s. Contact is
present in all seven firings. Their per-run audit records Fn, Ft, Ft/Fn, foot
velocity, FSR sum, acceleration magnitude, and gyro magnitude at firing and at
the later canonical onset. This supports *candidate precursor* status, not a
post-hoc change to the v1 label: two no-onset runs also show transient contact
loads/Ft/Fn, so a single contact transient is insufficient evidence.

Plots and raw audit values are in
`simulation/outputs/terrain_fast_reflex_v2_planning_diagnostic`.

### Recommended task definition

Use a binary reflex task: **Safe vs Slip Risk**. Preserve a three-state oracle
diagnostic (`safe`, `incipient/risk`, `confirmed`) for analysis and latency
reporting, but do not require a three-way fast classifier. Reflex needs an
early intervention decision; confirmation remains a separate outcome rather
than a class that competes with risk.

The v2 risk definition must be frozen before data generation and calibrated
only from fresh v2 train normal contact traces. A candidate risk episode is a
loaded contact state with a causal, sustained directional departure from normal
in oracle horizontal foot motion and contact tangential loading; confirmed slip
is the existing canonical loaded-contact speed event. Calibrate all magnitude,
trend, and persistence cutoffs from train normal robust quantiles/MAD, then
freeze them before validation. FSR/IMU are detector inputs and observability
diagnostics only; MuJoCo velocity/Fn/Ft remain oracle-only and are never inputs.

Safe endpoints must exclude a pre-registered guard interval immediately before
a risk/confirmed episode, so ambiguous transition-edge windows do not become
negative training labels. Report risk-to-confirmation lead separately from
false alarms; it must not be retroactively called v1 detection performance.

### Untouched v2 final-test policy

The v1 `warped_multisine` and `smooth_random_patches` test data have informed
this planning audit and are permanently ineligible as a v2 final score. Build
v2 train and validation from fresh, disjoint surface/session/run seeds. Reserve
a new final hold-out before fitting: new procedural surface realizations and
new excitation/run seeds, materialized only after the v2 detector, threshold,
and persistence policy are frozen. Prefer a newly named held-out morphology
family; if a known family must be reused, hold out all of its new surface seeds
and sessions at family-realization granularity. Do not read or score that hold-
out during development.

## Sand v2: meaningful physical hazards

The v1 Tilt-only oracle maximum is only 0.000381--0.000595 rad (median
0.000445 rad), 1.10--1.72 times its canonical 0.000347 rad threshold. This is
too close to the old instantaneous boundary to claim that every such event
merits fast reflex intervention. It is evidence for a new physically meaningful
definition, not authorization to modify the v1 threshold.

### Proposed label rules (thresholds intentionally unset)

**Sustained Tilt** requires loaded contact plus an oracle foot-orientation
departure calibrated from v2 train normal data, an angular-rate departure, and
a pre-registered duration/persistence requirement. Record FSR front/rear and
left/right imbalance as an independent support-mismatch corroboration tag. Keep
the physical oracle label and sensor corroboration separate so the detector is
not made tautological by its own input feature.

**Sustained Sink** requires loaded contact plus oracle vertical displacement,
downward velocity, and a duration/persistence requirement. Minimum load
prevents airborne motion from becoming sink. Calibrate each magnitude and
duration from v2 train normal traces using robust normal quantiles/MAD, freeze
before validation, and retain the raw physical signals for audit.

For both definitions, evaluate magnitude, temporal persistence, and direction
consistency rather than a single onset-aligned sample. Existing validation-only
Tilt-only diagnostics show front/rear direction consistency in 11/15 at 5 ms
and 10/15 at 10 ms; raw gyro-x and gyro-y direction remain consistent in 11/15
and 14/15 respectively over 5 ms. This is feasibility evidence only: the set
also contains different scenario mechanisms, and v1 causal replay showed that
instantaneous separation alone is not a deployable hazard rule.

## Sand v2 failure-mode dataset

Generate a balanced, explicitly labelled matrix after this plan is approved:

| Mode | Physical intent | Required outcome |
|---|---|---|
| A. Normal sand contact | Stable loaded compliant contact | No sustained hazard. |
| B. Sink-dominant | Symmetric soft/compliant vertical loading | Sustained Sink, no Sustained Tilt. |
| C. Tilt-dominant | Asymmetric support/contact with bounded vertical loss | Sustained Tilt, no Sustained Sink. |
| D. Boundary-step | Front/rear or left/right supports have different terrain response | Persistent spatial support mismatch; label outcome from physical rules. |
| E. Sink + Tilt | Asymmetric compliant support with vertical loss | Both labels. |

Balance each mode across fresh surface, session, pulse direction, and run seeds.
Keep mode allocation group-disjoint by surface realization/session/run, and
report per-mode run recall/FPR/latency rather than only pooled Sand metrics.
The current v1 validation cannot establish an independent Sink-only detector
claim because it has zero Sink-only runs.

### Boundary-step feasibility

It is implementable, but not with the current one-geom temporal switch alone.
`scene_walking_surface.xml` has one `study_surface` hfield and one
`surface_floor` geom; `run_terrain_fast_reflex_v1.py` changes that geom's
friction/solref/solimp globally at transition. A Marble/Sand support mismatch
needs two named adjacent ground geoms (initially simple tiled boxes or planes,
then optionally cropped hfields), each with its own terrain profile. Position
the x-aligned seam under the foot for front/rear mismatch or the y-aligned seam
for left/right mismatch, using the existing sole mapping below. Extend contact
audit to retain which sole sphere contacted which tile and verify no gap or
double-contact seam artifact. This is a future implementation and small smoke
test, not part of this planning milestone.

## v2 sensor features and detector order

Use only the existing E84-computable Fusion10 derivations:

- FSR: sum; front-rear and left-right difference; normalized imbalance; spatial
  variance; range; and causal imbalance derivatives.
- IMU: raw `gyro_x`, raw `gyro_y`, gyro-XY magnitude, and short-window gyro
  integral. Do not rename axes as roll/pitch before board orientation
  calibration.

FSR mapping is fixed as rear=(FSR1,FSR2), front=(FSR3,FSR4),
left=(FSR1,FSR3), right=(FSR2,FSR4). The strongest v1 feature-separation
candidate is normalized front/rear imbalance at 5 ms (ROC-AUC/PR-AUC 1.000,
Cohen's d 5.00); the strongest IMU candidate is 5 ms gyro-XY integral
(ROC-AUC 0.905, PR-AUC 0.756). They are candidates, not v2 thresholds.

Start v2 with **A. an explainable physical rule** that requires magnitude,
persistence, and compatible direction. It is the best fit for causal audit and
E84 simplicity. Escalate only if it fails frozen validation gates: **B. derived
features plus logistic regression**, then **C. a small MLP**. Do not start with
a raw-plus-derived CNN; reconsider it only after the failure-mode dataset is
balanced and the simple paths have failed for documented reasons.

## Evaluation protocol

Use train-only calibration, validation-only candidate selection, and a single
new hold-out final evaluation. For every candidate report endpoint metrics,
per-mode run recall, pre-onset run FPR, hazard-free run FPR, mode-specific
latency, persistence, and risk-to-confirmation lead. Apply a common gate of
run FPR <=5% before optimizing recall; retain the 20 ms observation objective
without lowering it post hoc. No INT8, Vela, E84, UART, firmware, or reflex rule
work begins until a separately approved host candidate passes its frozen gate.
