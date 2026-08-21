# Walking v2 Fast Reflex host v1 audit

## Decision

Selected exactly one development architecture: `MONITORING_ONLY_TERRAIN_AND_SLIP_ADVISORY`.
The deterministic authority firewall sets direct reflex and recovery actuation to `false`.

```text
Fusion20 + causal G0 contact -> shared 200 ms history
  |-> locked T2 Terrain -> contact-local confidence/dwell -> Case-A advisory
  |-> finalized S4-C    -> V4 contact reset/persistence  -> Slip risk advisory
  `-> deterministic authority firewall -> direct_reflex = false
```

## Required answers

1. **Actual blocker:** v5 populated predicted Case A with an all-false placeholder. Its zero C1/C2 denominators were not model evidence. Real T2 replay exposed a second, material blocker: Case-A specificity is insufficient for direct authority.
2. **Engineering changes:** built the missing causal T2 timeline; added contact-local confidence/dwell, causal finite-data gating, G0 ownership and one-shot reset; removed fall/prefall oracle gates; separated direct and advisory metrics; fitted a reload-exact development advisory model with a raised optimizer ceiling.
3. **Alternatives:** Terrain-only direct, broad learned-Slip direct, Case-A-gated predictive, Case-A-gated reactive confirmation, and monitoring-only advisory were evaluated in `alternative_comparison.csv`.
4. **Selected architecture:** monitoring-only Terrain and Slip advisory, with deterministic direct-actuation firewall.
5. **Authority:** Terrain and Slip are advisory only; learned foot is diagnostic; G0 supplies the only permitted future owner tag; M1 is provenance only; deterministic logic owns masks/reset/dedup and blocks actuation; Sink is deferred.
6. **Metrics:** frozen OOF Slip advisory recall is `0.673036` episode and `0.936000` run; alert precision is `0.525705`; normal-run/contact FP are `0/256` and too-early outputs are `30`. Per-speed recall is `{"0.10":0.6727272727272727,"0.15":0.6987179487179487,"0.20":0.6466666666666666}`. Terrain advisory recall/precision are `0.091295` / `0.398148` with `57` normal-contact false advisories. Causal host-replay and latency details are in `advisory_metrics.json`.
7. **Limitations:** simulation-only development data, partial Terrain in-sample overlap, no aligned physical Terrain transition label, no causal fall estimator, no independent metric for the all-development final advisory fit, and no target latency/parity result.
8. **Invalid prior claims:** zero-denominator C1/C2 performance, broad direct learned-Slip readiness, Terrain-only validation from mapping alone, blind generalization, safety certification, target readiness and real-robot readiness remain invalid.
9. **Next action:** exactly `ACQUIRE_ALIGNED_CASE_A_TRANSITION_DEVELOPMENT_CORPUS`.
10. **Data boundary:** no outer, holdout, spatial-final or final-test content was accessed; the exact-path access log has zero forbidden and blocked reads.
11. **Hardware boundary:** no flashing, E84 execution, physical HIL, INT8 or Vela work occurred.
12. **Sink:** remains `SINK_RUNTIME_DETECTION_DEFERRED`.
