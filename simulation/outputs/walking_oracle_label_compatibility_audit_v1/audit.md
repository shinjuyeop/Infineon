# Walking Slip/Sink oracle-label compatibility audit v1

The frozen controlled-excitation oracles are **not compatible** with normal
walking labels.  Frozen Slip fired in 9/9
normal Marble/Concrete/hardened-Sand runs, and frozen Sink fired in
9/9 normal runs.

The three runs per homogeneous terrain are deterministic replicates, so these
counts cover three normal physical profiles rather than nine independent
random trials.

Slip now means contact-anchor-relative tangential stance motion, not absolute
world-frame foot speed.  The largest normal-control run drift was
0.008933 m while the smallest homogeneous-Ice run maximum
was 0.088157 m.  This supports the semantic distinction but
does not freeze a detector threshold.

Sink now means sole penetration relative to the local terrain surface.  The
current ankle-z proxy is rejected.  No valid positive walking Sink ground
truth exists because walking-support-v1 hardens Sand's solref.  A new physical
observable and positive acquisition are required before retraining.

All spatial A/B/C/D runs fell, and the stored traces do not contain a
first-fall sample suitable for censoring.  They remain diagnostic-only.

**Bounded retraining is not authorized by this checkpoint.**  Normal walking
hard negatives are ready, but Slip threshold calibration and real Sink
positive ground truth must be completed first.
