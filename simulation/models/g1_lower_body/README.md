# G1 Lower-body Controlled HIL Model

이 directory에는 Unitree G1 29-DOF MJCF에서 파생한 project-owned reduced-order
model이 있다. `simulation/unitree_mujoco/unitree_robots/g1/`의 upstream 파일은
변경하지 않으며 full-body baseline으로 유지한다.

Reduced model의 목적은 reference pose의 static mass와 inertia loading을
보존하면서 terrain-contact/lower-limb HIL 실험에서 upper-body articulated
motion을 제거하는 것이다. 실제 Unitree G1 dynamics를 완전히 대체하지 않는다.

## 구조

Floating `pelvis`와 다음 12 actuated joint를 유지한다.

- 좌우 hip pitch, hip roll, hip yaw
- 좌우 knee
- 좌우 ankle pitch, ankle roll

`waist_yaw_link`에서 시작하는 articulated subtree, 즉 waist joint 3개, torso,
양쪽 shoulder/arm/wrist를 제거한다. Pelvis와 각 leg의 body 6개는 inertial,
joint, mesh, collision definition을 변경하지 않고 upstream 29-DOF XML에서
복사한다.

제거된 17-body subtree mass는 `16.927142 kg`이다. Upstream zero-joint reference
pose에서 pelvis body origin 기준 composite COM은 다음과 같다.

```text
[0.0259099280811, 0.00014971768087, 0.20065874815] m
```

Pelvis frame으로 표현한 composite COM 기준 inertia:

```text
[[ 0.353862719457, -0.000025579376,  0.043939998232],
 [-0.000025579376,  0.261701575616, -0.000009967367],
 [ 0.043939998232, -0.000009967367,  0.254865283184]] kg m^2
```

이 특성을 fixed `equivalent_upper_body`로 표현한다. 변환에는 제거된 각 body의
oriented inertia와 parallel-axis term이 포함된다. 따라서 reference pose에서
mass, whole-model COM, rotational inertia는 full model과 일치한다. Articulated
upper-body momentum과 configuration-dependent inertia는 의도적으로 재현하지
않는다.

Model 재생성:

```bash
cd /d/shin/Infineon/simulation/unitree_mujoco/simulate_python
/d/shin/Infineon/simulation/venv/bin/python build_g1_lower_body_model.py
```

## Sensor 및 contact

Upstream primary `imu` site는 torso가 아니라 `pelvis`에 attached되어 있다.
따라서 `imu_acc`, `imu_gyro`는 양쪽 model 모두 pelvis-reference virtual
sensor이다. `waist_roll_link`의 upstream `secondary_imu`는 upper-body subtree와
함께 제거된다. Pelvis IMU는 simulation reference이며 최종 KIT_PSE84_AI의
physical mounting location을 의미하지 않는다.

Named left-foot collision sphere 4개는 변경 없이 복사한다.

```text
left_foot_contact_1
left_foot_contact_2
left_foot_contact_3
left_foot_contact_4
```

Radius, local position, contact-force mapping, Force4 + IMU6 interface는
변경하지 않는다. 10-channel 순서:

```text
[foot_force_1, foot_force_2, foot_force_3, foot_force_4,
 accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
```

## Support 및 excitation

`support_point`는 equivalent upper-body COM에 위치한다. 70% vertical
ElasticBand support force는 complete equivalent model mass로 계산한다.
Horizontal anchor는 pelvis를 따라가므로 horizontal translation을 직접
구속하지 않는다.

`pulse_point`는 original full-body `torso_link` COM force location을 유지한다.
Reference pose의 pelvis 기준 좌표는
`[-0.00064692, 0.000261533, 0.233856] m`이다. Horizontal half-sine pulse는
`mujoco.mj_applyFT`로 이 위치에 적용한다. Positive/negative X pulse는 동일한
world-space point와 반대 force vector를 사용한다. Elevated point의 force는 두
방향에 반대 부호 pitch moment를 자연스럽게 만든다.

## Reference-pose limitation

- Leg joint 12개와 pelvis roll/pitch/yaw는 모두 zero로 시작
- Contact sphere center는 plane보다 약 `6.14 mm` 높게 시작하며 첫 settling
  step 동안 contact 형성
- G1 foot geometry는 fore-aft symmetric이 아님: ankle 기준 heel contact X는
  약 `-0.05 m`, toe contact X는 약 `+0.12 m`
- Reference COM도 pelvis보다 약간 앞쪽에 있으므로 arm/waist DOF를 제거해도
  positive/negative X dynamics가 같아지지 않음
- Terrain profile은 engineering approximation이며 measured material
  property가 아님

## Validation 및 preview

Mass-property와 short static validation:

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

Viewer는 observation 전용이며 physics timestep이나 model parameter를 변경하지
않는다.
