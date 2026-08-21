# Additional moderate-v2 Slip acquisition v8 audit

1. Calibration succeeded while v7 variations failed because: **PATCH_ENTRY_POSE_SENSITIVITY**; the isolated same-phase/same-delay comparison changed patch-entry state and produced 0.058234 m versus 0.034409 m maximum drift.
2. M1/G0/strong Ice/oracle unchanged: **True**.
3. Exactly 12 positive and 12 control unique frozen runs are retained: **True**. A disclosed deterministic serialization replay of the same 24 run IDs was required after a post-execution audit-field mismatch; no seed/configuration was replaced.
4. Supplemental source-valid positives: **6/12**.
5. Physically distinct valid variations: **6**.
6. All valid onsets remained in mid_late_stance: **True**.
7. All controls remained physically non-slip: **True**.
8. Parity/divergence/no-double-contact gates: **12/12, 12/12, 0**.
9. Final missing cell filled: **True**.
10. Base matrix remains explicitly **87/108** and **35/36**: **True**.
11. Augmented coverage reaches 36/36: **True**.
12. Calibration, failed-source, and quarantined data excluded correctly: **True**.
13. Future nested-fold manifest created: **True**.
14. Targeted Slip retraining authorized: **True**.
15. Forbidden artifact access count: **0**.
16. Terrain and M1 byte-identical: **True/True**.
17. Blind holdout/System/INT8 remain unauthorized: **True/True/True**.
18. Sink remains `SINK_RUNTIME_DETECTION_DEFERRED`: **True**.

Exactly one next step: **SLIP_TARGETED_RETRAINING_V3**
