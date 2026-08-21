# Walking v2 Targeted Slip Retraining v3 audit

- Starting checkpoint: `72df1e1fa692357a2cf8afedddc8006e669001a4`
- Frozen eligible population: 333 runs (93 targeted positives, 120 matched controls, 120 existing development).
- Exact frozen nested folds: 3; run/pair/variation/contact-episode leakage: 0.
- Candidate matrix: 3 families x 3 fixed seeds; operating grid frozen before training.
- Runtime inputs: causal virtual bilateral Fusion20, causal FSR-loaded contact/age, commanded speed.
- Privileged physical fields: offline label/evaluation only; future leakage count: 0.
- Selected candidate: `NONE`.
- Diagnostic fallback: `R1_seed_202608231`.
- Fresh blind holdout authorized: `false`; no blind data was generated or accessed.
- Terrain, M1, G0, and physical oracle remained byte-identical.
- No System, INT8, Vela, E84, or HIL work occurred.
- Sink remains `SINK_RUNTIME_DETECTION_DEFERRED`.
- Next step: `SLIP_MODEL_REDESIGN`.

## Pre-protocol disclosure

Two NPZ containers were opened before the formal run only to inspect key names, dtypes, and shapes while implementing the loader. No array values, labels, model scores, or metrics were inspected. The formal execution created and hashed its protocol, allowlist, forbidden policy, access log, immutable verification, candidate matrix, and selection policy before loading any trace values.
