# G1 lower-body controlled HIL model

This directory contains a project-owned reduced-order model derived from the
Unitree G1 29-DOF MJCF. The upstream files under
`simulation/unitree_mujoco/unitree_robots/g1/` remain unchanged and are the
full-body baseline.

The reduced model is intended to remove upper-body articulated motion from
terrain-contact and lower-limb HIL experiments while retaining the reference
pose static mass and inertia loading. It is not a complete replacement for the
actual Unitree G1 dynamics.

## Structure

The model retains the floating `pelvis` and these 12 actuated joints:

- left/right hip pitch, hip roll, and hip yaw
- left/right knee
- left/right ankle pitch and ankle roll

It removes the articulated subtree rooted at `waist_yaw_link`: the three waist
joints, torso, both shoulders and arms, and both wrists. The pelvis and the six
bodies in each leg are copied from the upstream 29-DOF XML without changing
their inertial, joint, mesh, or collision definitions.

The removed 17-body subtree has a mass of `16.927142 kg`. At the upstream zero
joint reference pose its composite COM relative to the pelvis body origin is:

```text
[0.0259099280811, 0.00014971768087, 0.20065874815] m
```

Its composite inertia about that COM, expressed in the pelvis frame, is:

```text
[[ 0.353862719457, -0.000025579376,  0.043939998232],
 [-0.000025579376,  0.261701575616, -0.000009967367],
 [ 0.043939998232, -0.000009967367,  0.254865283184]] kg m^2
```

These properties are represented by the fixed `equivalent_upper_body`. The
conversion includes each removed body's oriented inertia and the parallel-axis
term. Consequently, mass, whole-model COM, and whole-model rotational inertia
match the full model at the reference pose. Articulated upper-body momentum and
configuration-dependent inertia are intentionally not reproduced.

Regenerate the model with:

```bash
cd /d/shin/Infineon/simulation/unitree_mujoco/simulate_python
/d/shin/Infineon/simulation/venv/bin/python build_g1_lower_body_model.py
```

## Sensors and contact

The upstream primary `imu` site is already attached to `pelvis`, not to the
torso. `imu_acc` and `imu_gyro` therefore remain pelvis-reference virtual
sensors in both models. The upstream `secondary_imu` on `waist_roll_link` is
removed with the upper-body subtree. The pelvis IMU is a simulation reference;
it is not a claim about the final physical KIT_PSE84_AI mounting location.

The four named left-foot collision spheres are copied unchanged:

```text
left_foot_contact_1
left_foot_contact_2
left_foot_contact_3
left_foot_contact_4
```

Their radius, local positions, contact-force mapping, and the resulting
Force4 + IMU6 interface are unchanged. The 10-channel order remains:

```text
[foot_force_1, foot_force_2, foot_force_3, foot_force_4,
 accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
```

## Support and excitation

`support_point` is located at the equivalent upper-body COM. The 70% vertical
ElasticBand support force is computed from the complete equivalent model mass.
Its horizontal anchor follows the pelvis, so it does not directly restrain
horizontal translation.

`pulse_point` preserves the original full-body `torso_link` COM force location,
which is `[-0.00064692, 0.000261533, 0.233856] m` relative to the pelvis in the
reference pose. The horizontal half-sine pulse is applied there with
`mujoco.mj_applyFT`. Positive and negative X pulses use this same world-space
point and opposite force vectors. A force at this elevated point naturally
generates opposite-sign pitch moments for the two directions.

## Reference-pose limitations

- All 12 leg joint angles and pelvis roll/pitch/yaw are zero initially.
- The contact sphere centers begin about `6.14 mm` above the plane; contact is
  established during the first settling steps rather than at time zero.
- G1 foot geometry is not fore-aft symmetric: heel contact X is about `-0.05 m`
  and toe contact X is about `+0.12 m` relative to the ankle.
- The robot's reference COM is also slightly forward of the pelvis. Removing
  arm and waist DOFs does not make positive/negative X dynamics identical.
- Terrain profiles remain engineering approximations and are not measured
  material properties.

## Validation and preview

Mass-property and short static validation:

```bash
/d/shin/Infineon/simulation/venv/bin/python \
  /d/shin/Infineon/simulation/unitree_mujoco/simulate_python/validate_g1_lower_body_model.py
```

Interactive concrete/+X preview:

```bash
cd /d/shin/Infineon/simulation/unitree_mujoco/simulate_python
/d/shin/Infineon/simulation/venv/bin/python run_g1_lower_body_symmetry_smoke.py \
  --terrain concrete --direction positive_x --runs 1 --gui \
  --realtime-factor 0.25 --gui-hold-seconds 5
```

The viewer is observational only; it does not change the physics timestep or
model parameters.
