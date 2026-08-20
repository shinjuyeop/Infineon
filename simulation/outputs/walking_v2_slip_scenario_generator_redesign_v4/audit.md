# Walking v2 Slip Scenario Generator Redesign v4 audit

1. The exact identity mechanism was **POST_CONSTRAINT_MUTATION**: v3 changed `data.contact.friction` after `mj_step1` had already constructed the contact constraints.
2. Friction was previously applied between `mj_step1` and `mj_step2`, after collision and constraint construction.
3. Geom priority/max combination did not neutralize v3's direct contact mutation; it is a demonstrated secondary hazard for ground-only designs.
4. The correction is a height-preserving, non-overlapping `PRECOMPILED_TILED_PATCH_GEOM` with eight explicit sole/patch pairs, whose pair friction is set before `mj_step`.
5. Microbench monotonic friction-dependent dynamics: **True**.
6. Whole-surface native Ice reproduced physical Slip: **True**.
7. All local pairs had pre-contact parity and post-contact divergence: **True**.
8. Patch/base double-contact count: **0**.
9. Valid strong positives: **12/12**; valid moderate positives: **7/12**.
10. Both target feet have physically valid localized positives: **True**.
11. Silently discarded or relabeled runs: **0**. Every attempt and replay was retained.
12. All **216** v3 runs are quarantined as `INVALID_INTERVENTION_DO_NOT_TRAIN`; original artifacts were not modified.
13. Forbidden artifact accesses: **0**.
14. Terrain model, normalization, config and lock remained byte-identical: **True**.
15. Full 216-run reacquisition authorized: **True**. It was not performed here.
16. Retraining, blind holdout, System migration and INT8 preparation remain unauthorized: **True**.
17. Sink remains `SINK_RUNTIME_DETECTION_DEFERRED`: **True**.

Exactly one next step: **REACQUIRE_TARGETED_BILATERAL_SLIP_DATA**
