# Walking Stateful Hazard Detector Prototype v1

This is a development-only v00/v01 training and v02 selection result. No nested outer,
new Sink holdout, spatial, final-test, production, INT8, or Vela content was used.

## Result

- Slip: RISK_ONLY_READY / LABEL_SEMANTICS_REDESIGN_REQUIRED
- Sink: STATEFUL_HISTORY_INSUFFICIENT / LABEL_SEMANTICS_REDESIGN_REQUIRED
- Slip selection: C_contact_gated_100ms (fallback=true)
- Sink selection: A_stateless_20ms (fallback=true)
- Full prototype ready: false

RISK is a bounded pre-onset early-warning state. CONFIRMED uses a separately fitted
score and persistence; it is never promoted solely by waiting after RISK. Physical
oracles and simulator gait phase are used only by offline labels/evaluation.

## Boundary audit

- Outer content loads: 0
- New Sink holdout/spatial/final-test accesses: 0
- Production/INT8/Vela/E84/System-v1 writes: 0
- Candidate count: 9 (four Slip, five Sink)
- Search/gate changes after v02 observation: 0
