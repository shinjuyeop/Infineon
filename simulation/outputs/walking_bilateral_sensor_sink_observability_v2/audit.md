# Walking Bilateral Sensor and Sink Observability v2

## Scope

This is a MuJoCo digital-twin development audit. All FSR, foot/pelvis IMU,
joint, command, and actuator-effort values are virtual runtime-observable
signals. No sensor purchase, PCB, wiring, E84 firmware interface, production,
System, INT8/Vela, outer holdout, or final-test work was performed. Privileged
penetration, world-frame positions, terrain/profile identity, fall state, and
physical oracles were confined to labels/evaluation.

## Results

- Bilateral runs: 120 (train 72, validation 48)
- Duplicate endpoint hashes: 0
- Diagnostic best C1--C4 candidate: C1
- Sink observability gate passed candidates: []
- Sink runtime readiness: False
- Selected next step: `SINK_RUNTIME_DETECTION_DEFERRED`

## Final judgments

1. **Bilateral change basis:** Yes for the versioned sensor architecture: both-foot schema, frame, contact, and dataset gates pass. Detector/holdout authorization remains separate.
2. **Exact shared encoder:** Yes: one deterministic encoder definition/fingerprint is applied to both canonical feet with exact identity.
3. **Minimum feature contributing to Sink:** C3 support kinematics was the first set to improve zero-FP recall numerically over C0, but the improvement was too small and failed the locked gate; no feature set is selected.
4. **Bilateral foot-only Sink:** No under the locked gate.
5. **Required kinematics/torque/pelvis:** No pelvis/kinematic/effort set proved sufficient under all gates.
6. **Robot/E84 reproducibility:** Yes as future robot direct sensors/controller telemetry; no physical interface was implemented here.
7. **Separate physical sensor:** Not added in this scope; the evidence supports defer rather than an unvalidated purchase/interface decision.
8. **Sink runtime continuation:** Defer runtime Sink detection.
9. **Legacy Slip blind authorization:** Keep the left-foot Slip blind authorization as v1-only; do not transfer it to bilateral v2.

The support-degradation target is diagnostic label evidence only. No recovery
intervention was executed, so no reduction in falls or recovery success is
claimed. Gates were not relaxed after observing validation.
