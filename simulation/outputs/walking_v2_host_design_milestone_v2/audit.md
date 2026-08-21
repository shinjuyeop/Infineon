# Walking v2 host design milestone v2

1. **Aligned corpus:** generated 420 development-only bilateral 1 kHz runs (5 isolated surface/gait folds, two seeded replicates, three speeds, A/B/C/D, hard-to-hard and steady seams). Exact Fusion20, Terrain GT/scores/state, transition/touchdown/G0, physical Slip and evaluation-only first-fall timing are retained in five sub-45-MiB shards.
2. **Real bottleneck:** missing aligned transition supervision was repaired. The remaining direct-authority blocker is evidence independence, plus any false/latency failures shown in `direct_authority_decision.json`; a single simulator/controller family cannot establish false-actuation safety.
3. **Architectures evaluated:** locked T2, T1 200-ms linear, T2 dual-timescale 200-ms linear, 24 causal policies each, prior broad S4-C Slip, and the previously failed reactive confirmation role.
4. **Engineering changes:** bilateral seeded generation, physical per-foot terrain ownership, exact timestamps, grouped OOF isolation, dedicated transition retraining, retained-across-air state, queued unique G0 ownership and one-shot dedup.
5. **Selected design:** `CASE_A_TRANSITION_MONITOR_PLUS_SLIP_ADVISORY` using `T2_LINEAR_200` / `p0.50_d3_a10`.
6. **Authority:** Terrain transition and S4-C Slip are advisory only; G0 owns causal contact identity; deterministic state owns dwell/dedup/firewall; direct and recovery arrays are hard false; Sink is deferred.
7. **Results:** Case-A event recall `0.492537`, precision `0.138075`, F1 `0.215686`, within-20-ms recall `0.000000`, normal-run FP `118`, normal-contact FP `206`, median/p95 latency `169.0`/`461.8` ms. Per-speed and hard-source results are in `selected_transition_monitor_metrics.json`; Slip results are in `slip_advisory_metrics.json`.
8. **Direct reflex:** disabled. No runtime path consumes Terrain GT, Slip oracle, fall oracle, timestamps, identities or future samples.
9. **Stopping rationale:** the bounded 72-point transition comparison reached the one-source evidence ceiling; more same-source tuning cannot satisfy the frozen independence gate and risks development overfit.
10. **Lock:** one immutable development design lock is written; it is not a blind, safety, deployment or real-robot lock.
11. **INT8/Vela:** not performed; optional target preparation was intentionally left after independent evaluation.
12. **Exact next task:** `ACQUIRE_FRESH_BLIND_HOST_HOLDOUT`.
13. **Data boundary:** no outer, holdout, spatial-final or final-test content was accessed or generated; all new runs are permanently blind-ineligible.
14. **Hardware boundary:** no flashing, E84 execution, physical HIL or hardware action occurred.
15. **Sink:** remains `SINK_RUNTIME_DETECTION_DEFERRED`; Case B is evaluation context only.
