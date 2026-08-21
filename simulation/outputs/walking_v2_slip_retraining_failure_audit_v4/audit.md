# Walking v2 targeted Slip retraining failure audit v4

- Exact frozen R1/R2/R3 replay: `PASS`.
- Primary root cause: `MULTIPLE_INTERACTING_CAUSES`.
- First material divergence: R1 changes the prior S4-C feature/model/head/runtime recipe before normalization.
- Reconciled actionable denominator: `471` per-foot contact-owned physical-onset episodes.
- Invalid/post-fall exact overlap: `52`.
- V0 accumulates state after first fall; V4 masks before persistence and hard-resets at the fall boundary.
- The diagnostic variants do not create, select, or lock a candidate.
- No outer/holdout/final data, new simulation, blind evaluation, System, or INT8 work occurred.
- Terrain, M1, G0, and the physical oracle remained byte-identical.
- Sink remains `SINK_RUNTIME_DETECTION_DEFERRED`.

## Interpretation

R1 was named an S4-C reproduction but changed the 200ms/width40/LBFGS state+foot+proposal recipe to a 50ms/width12 diagonal-Gaussian state-only recipe; then the selected 0.99 threshold and 5ms persistence reject most remaining episode evidence. The denominator also expands from the legacy 170-event development definition to 471 per-foot contact-owned events.
