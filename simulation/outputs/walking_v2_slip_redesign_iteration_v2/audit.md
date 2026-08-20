# Walking v2 Slip Redesign Iteration v2

All 120 bilateral runs are treated as one development corpus. No old split is claimed blind, and no blind
evaluation artifact was accessed or generated. Terrain remained immutable.

1. Selected family: S4-C_seed_202608222 (diagnostic_fallback_only).
2. Pooled/per-speed actionable recall: 0.5588; {"0.10": 0.578125, "0.15": 0.5490196078431373, "0.20": 0.5454545454545454}.
3. All 14 previous too-early activations removed: False.
4. Normal run/contact FP both zero: True.
5. Affected-foot accuracy >=90%: False (0.8273).
6. Every fold passed every zero-safety gate: False.
7. Invalid/AIR/touchdown/post-fall attribution/latch/cross-foot violations all zero: True.
8. Slip selection lock created: False.
9. Terrain lock byte-identical: True.
10. New blind acquisition authorized: False; none generated here.
11. Forbidden artifact reads: 0. The exact allowlist and read ledger are stored beside this report.
12. Sink: `SINK_RUNTIME_DETECTION_DEFERRED`; no Sink runtime artifact exists.

Next step: `ADDITIONAL_BILATERAL_SLIP_ACQUISITION`
