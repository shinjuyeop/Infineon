# Walking v2 Joint Terrain / Slip Redesign v1

This is a development-only redesign and validation-selection record. No blind evaluation artifact was opened,
created, or evaluated. The previously exposed artifact class remains `EXPOSED_NON_BLIND_DIAGNOSTIC_ONLY`.

## Answers required by the task

1. **Did 200 ms temporal representation recover Terrain performance?** True.
   Best validation accuracy/macro/worst/Sand recall: 0.9295 /
   0.8909 / 0.7695 /
   0.9634.
2. **Was destructive downsampling a material contributor?** True.
3. **Which Terrain architectures passed?** {'T1': True, 'T2': True, 'T3': False}.
4. **Did horizon-aware Slip remove all 40 genuine too-early activations?** False.
5. **Corrected Slip actionable recall overall/by speed:** 0.8382;
   {"0.10": 0.8333333333333334, "0.15": 0.7368421052631579, "0.20": 0.92}.
6. **Are normal FP, invalid firing, latch carryover, and cross-foot ownership zero?** False.
7. **Were both selection locks created?** False.
8. **Is acquisition of a completely new blind evaluation split authorized?** False.
   No such split was generated in this task.
9. **Was the exposed previous artifact accessed or used?** No. The allowlist contains no forbidden namespace,
   and the fail-closed read ledger records zero forbidden reads.
10. **Sink status:** `SINK_RUNTIME_DETECTION_DEFERRED`; no Sink runtime head, score, model, config, or output was created.

## Selection

- Terrain: selected_and_locked — T2_seed_202608211.
- Slip: diagnostic_fallback_only — S3_seed_202608212.
- Static/controlled detector: immutable compatibility hash only; not replaced.
- Sand: `SAND_TERRAIN_CAUTION`, never Sink detection.

## Next step

`SLIP_REDESIGN_ITERATION`
