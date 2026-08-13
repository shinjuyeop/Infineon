# Terrain INT8 on-device validation

## 1000 Hz / 50-sample fast candidate (E84/U55 fixed and Host-golden HIL verified)

## Fast Reflex v2 fixed golden preparation

`FAST_REFLEX_SINK_V2_U55` and `FAST_REFLEX_SLIP_V2_U55` are generated from
frozen Vela artifacts and contain six validation-only INT8 vectors each. Build
them with `TERRAIN_MODE=fixed`; UART prints `FRV2 vec=... raw=... expected=...
decision=... raw_exact=PASS`. Sink's expected values use the recorded first-U55
baseline because a normal Host TFLite interpreter cannot execute Vela's
`ethos-u` custom op. Slip's mixed CPU/NPU runtime is not validated until board
execution. Neither profile uses final-test samples.

The fast candidate is intentionally separate from the deployed 100 Hz model:

| Path | Model name | Source |
|---|---|---|
| Cortex-M55 TFLM | `TERRAIN_FAST1000_CPU` | raw strict-INT8 fast model |
| Ethos-U55-128 | `TERRAIN_FAST1000_U55` | Vela-compiled fast model |

The existing `TERRAIN_CPU`, `TERRAIN_U55`, `deployment/fixed_test_metadata.json`,
TRN1, and TRN2 paths remain unchanged. Generated fast reports are written below
`deployment/fast1000/`. Artifact generation refuses to overwrite any selected
profile output unless `--force` is explicitly supplied; normal fast deployment
does not require `--force`.

The source identity expected by the generator is:

- raw TFLite: 7,048 bytes, SHA-256
  `6f123c7727e3879e3a6c8ef41f55617a70d48a975fa44921042251b06049e74a`
- input/output: strict INT8 `(1,50,10) -> (1,4)`
- operators: `EXPAND_DIMS`, `CONV_2D`, `RESHAPE`, `MEAN`,
  `FULLY_CONNECTED`, `SOFTMAX`
- fixed sample: full 1000 Hz test sample 878, `warped_multisine`, Concrete
- fixed input SHA-256:
  `77389783c1e3d558c2c616d6cca5279879a6b56f22be58628bf2f1314538be2d`
- Host raw golden: `[114,-114,-128,-128]`, class 0

No 1 kHz asynchronous UART transport is added here. `TERRAIN_MODE=hil` still
means the existing synchronous TRN1/TRN2 implementation and must not be used as
evidence of 1 kHz real-time streaming.

Probe `13070E98012D2400` verification produced the exact fixed/HIL raw output
`[114,-114,-128,-128]`, class 0. The measured HIL profile was
`cpu_cyc=24910`, `npu_cyc=9987`.

### Commands to run manually

Run all commands from this project directory:

```sh
cd /d/shin/Infineon/Infineon_test/projects/terrain_e84_deploy
```

#### A. Source verification and artifact generation

The first command performs Host-only identity/I/O/operator/golden checks and
writes nothing. The second invokes Vela and creates both separate model arrays
and their identical fixed regression arrays.

```sh
/d/shin/Infineon/Infineon_HIL/.venv/bin/python \
  tools/generate_terrain_artifacts.py --profile fast1000 --verify-only

/d/shin/Infineon/Infineon_HIL/.venv/bin/python \
  tools/generate_terrain_artifacts.py --profile fast1000 --backend all
```

Expected generated names are `TERRAIN_FAST1000_CPU_*` and
`TERRAIN_FAST1000_U55_*`; metadata is
`deployment/fast1000/fixed_test_metadata.json`. Review Vela's emitted summary
and confirm the `--show-cpu-operations` report contains no fallback operator.

#### B. Cortex-M55 compatibility build and U55 fixed build

The CPU build is the TFLM compatibility gate. The second command selects the
Vela model for the image that will be flashed.

```sh
make build NN_MODEL_NAME=TERRAIN_FAST1000_CPU TERRAIN_MODE=fixed
make build NN_MODEL_NAME=TERRAIN_FAST1000_U55 TERRAIN_MODE=fixed
```

If switching configurations in an already-built workspace does not trigger a
complete relink, run `make clean` once and repeat the desired build command.

#### C. Flash the U55 fixed image to probe 13070E98012D2400

```sh
make qprogram \
  NN_MODEL_NAME=TERRAIN_FAST1000_U55 \
  TERRAIN_MODE=fixed \
  MTB_PROBE_SERIAL=13070E98012D2400
```

#### D. Fixed tensor board test

Start the verifier, then press the physical RESET button on KIT_PSE84_AI while
it is waiting. It requires exact device/embedded/Host raw parity, class 0, and
the firmware `PASS` marker.

```sh
/d/shin/Infineon/Infineon_HIL/.venv/bin/python \
  tools/terrain_fixed_test_client.py --profile fast1000 --timeout 15
```

Expected raw output is `[114,-114,-128,-128]` and the final JSON must contain
`"passed": true`.

#### E. Host golden comparison or optional existing-TRN1 parity

Recompute the Host golden without writing generated files:

```sh
/d/shin/Infineon/Infineon_HIL/.venv/bin/python \
  tools/generate_terrain_artifacts.py --profile fast1000 --verify-only
```

If a full-window UART parity check is wanted after the fixed gate, explicitly
build/flash the existing TRN1 HIL mode and send the same fast sample. This does
not implement or validate asynchronous 1 kHz streaming.

```sh
make build NN_MODEL_NAME=TERRAIN_FAST1000_U55 TERRAIN_MODE=hil
make qprogram \
  NN_MODEL_NAME=TERRAIN_FAST1000_U55 \
  TERRAIN_MODE=hil \
  MTB_PROBE_SERIAL=13070E98012D2400
/d/shin/Infineon/Infineon_HIL/.venv/bin/python \
  tools/terrain_hil_client.py --profile fast1000 --sample-index 878 \
  --expect-fixed-golden
```

The CM55 project defaults to `TERRAIN_CPU`, which embeds the canonical raw
strict-INT8 TFLite model and runs one fixed test-partition tensor through TFLM
on Cortex-M55. `TERRAIN_U55` embeds the same model compiled by Vela 4.2.0 for
the KIT_PSE84_AI Ethos-U55-128.

This example does not load a `.tflite` file at runtime and does not provide a
`make ml-configurator` target. The selected model is a generated C byte array
under `proj_cm55/mtb_ml_gen/mtb_ml_models`; `proj_cm55/Makefile` selects it with
`NN_MODEL_NAME`, and `proj_cm55/main.c` passes the resulting
`mtb_ml_model_bin_t` to `shared_src/ml_validation.c`. Fixed input and golden
output arrays are selected by the same name under
`proj_cm55/mtb_ml_gen/mtb_ml_regression_data`.

The following historical command targets the existing 100 Hz names. Do not run
it for the fast candidate and do not overwrite the canonical generated arrays.
Source-only verification remains available with:

```sh
python3 tools/generate_terrain_artifacts.py --profile 100hz --verify-only
```

Build the CPU path (default):

```sh
make build
```

Build the U55 path after CPU parity succeeds:

```sh
make build NN_MODEL_NAME=TERRAIN_U55
```

Use `MTB_PROBE_SERIAL=13070E98012D2400` when programming this workspace so the
image is sent to the selected KIT_PSE84_AI rather than either of the other two
connected KitProg3 probes.

## Verified fixed-tensor result

The fixed tensor is dataset test sample 878 (`warped_multisine`, label
`concrete`). Its quantized input SHA-256 is
`6119c67c0cfd7465fc57befad8ff5d2da0b018ca3538c9cfe052948bae0cc9fc`.
The Host golden raw output is `[35, -35, -128, -128]`, class 0 (`concrete`).

The canonical model uses only `EXPAND_DIMS`, `CONV_2D`, `RESHAPE`, `MEAN`,
`FULLY_CONNECTED`, and `SOFTMAX`. Vela 4.2.0 maps the entire graph to U55 with
zero CPU fallback operators. The accelerator configuration is U55-128; this
was independently recovered from the original KIT_PSE84_AI generated model's
driver command header (128 MAC/cycle, 24 KiB SHRAM).

On the selected PSE846GPS2DBZC4A at 400 MHz:

| Path | Device raw output | Class | CPU cycles | NPU cycles | Invoke time |
| --- | --- | ---: | ---: | ---: | ---: |
| Cortex-M55 TFLM | `[35,-35,-128,-128]` | 0 | 225641 | 0 | 564.1 us |
| Ethos-U55 | `[35,-35,-128,-128]` | 0 | 26342 | 9758 | 90.3 us total |

These are one-shot measurements and vary slightly between resets. The CPU
middleware reported 3,236 bytes of arena use; the U55 path reported 2,180
bytes, including Vela's 1,600-byte scratch allocation. Configured arena sizes
keep additional margin: 8 KiB for the raw CPU model and 4 KiB for the U55
model.

## Full-window UART/USB HIL (`TRN1`)

After fixed-tensor validation, build and program the U55 HIL image:

```sh
make build NN_MODEL_NAME=TERRAIN_U55 TERRAIN_MODE=hil
make qprogram MTB_PROBE_SERIAL=13070E98012D2400
```

Send the same dataset sample through the selected board's KitProg3 USB-UART:

```sh
python3 tools/terrain_hil_client.py --sample-index 878
```

For a MuJoCo-produced physical-unit `float32` window shaped `(50,10)`, ordered
FSR4 then left-foot/ankle IMU6, use:

```sh
python3 tools/terrain_hil_client.py --npy path/to/window.npy
```

The PC applies the training normalization and strict-INT8 input quantization.
The 1 Mbaud binary protocol is `TRN1`, a little-endian uint16 payload length,
500 INT8 bytes, and a little-endian IEEE CRC-32. The device replies with the
four raw INT8 outputs, class index, CPU cycles, and NPU cycles. A physical HIL
run of sample 878 returned `[35,-35,-128,-128]`, class 0, with 23,890 CPU and
9,463 NPU cycles. No GPIO, FSR, or IMU peripheral is used.

## Continuous sample-stream HIL (`TRN2`)

The same HIL firmware accepts individual 10-channel INT8 samples without
breaking `TRN1`. The E84 stores them in a static 50x10 ring buffer. It reports
warm-up state for the first 49 responses, performs the first inference when
the 50th sample completes the window, then invokes at the requested sample
stride. Sequence zero explicitly begins a new stream session.

The little-endian request frame is:

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 4 | ASCII magic/version `TRN2` |
| 4 | 2 | Payload length, always 16 |
| 6 | 4 | Sequence number, uint32 |
| 10 | 2 | Inference stride, uint16 and at least 1 |
| 12 | 10 | One FSR4 + IMU6 INT8 sample |
| 22 | 4 | IEEE CRC-32 of the 16-byte payload |

The board emits one ASCII `STREAM_RESULT` per accepted sample with sequence,
ring fill, warm-up/inferred flags, class, four raw INT8 outputs, and CPU/NPU
cycles. CRC and length errors are discarded with `STREAM_ERROR`. A sequence
gap is reported and resets the ring before the current sample is accepted, so
an incomplete window is never silently inferred.

Replay dataset sample 878 at its native 100 Hz cadence and save all responses:

```sh
python3 tools/terrain_stream_client.py \
  --sample-index 878 \
  --samples 1000 \
  --rate-hz 100 \
  --stride 1 \
  --csv terrain_stream_1000.csv
```

`--npy path/to/window.npy` accepts a physical-unit float32 `(50,10)` MuJoCo
window, applying the same PC normalization/quantization as the full-window
client. The source window is repeated when `--samples` exceeds 50; this is a
transport/cadence stress replay, not a claim that separate dataset windows are
one physically continuous trajectory. `--rate-hz` controls host pacing and
`--stride` controls E84 inference cadence without changing the trained model.

### Verified continuous result

On probe `13070E98012D2400`, 1,000 samples at 100 Hz and stride 1 completed in
10 seconds with 951 inferences, zero device errors, sequence drops, timeouts,
or host deadline misses. The first completed ring window returned
`[35,-35,-128,-128]`, class 0, exactly matching Host and `TRN1`.

| Measurement | Mean | Std | Min | Max | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Host send period (ms) | 10.000 | 0.049 | 9.799 | 10.204 | 10.072 |
| Request/response RTT, all samples (ms) | 1.657 | 0.070 | 1.366 | 2.078 | 1.790 |
| Request/response RTT, inferred samples (ms) | 1.666 | 0.058 | 1.572 | 2.078 | 1.793 |
| U55-path CPU cycles | 5,838.365 | 50.302 | 5,831 | 7,364 | 5,847 |
| U55 NPU cycles | 7,143.938 | 9.973 | 7,115 | 7,178 | 7,162 |

A separate 100-sample stride-5 board run produced the expected 11 inferences
with no errors or deadline misses. Physical error injection verified recovery
after bad CRC and bad length frames; a sequence gap emitted the expected
sequence and reset the ring. The round-trip measurement includes UART request,
inference when scheduled, ASCII response serialization, USB bridge, and host
read latency.

The installed standalone Vela package does not contain Infineon's named
`PSE8x_U55_400MHz_SOCMEM_200MHz_QUAD_XIP` performance configuration, so it
uses Arm's default U55 memory-bandwidth model. This affects Vela's performance
estimate/scheduling assumptions, not operator support; successful execution on
the physical E84 verifies the generated command stream functionally.

## Live MuJoCo TRN2 HIL

Dataset replay와 별도로, `run_live_terrain_hil.py`는 현재 실행 중인 Dataset-v1
MuJoCo runner의 sensor callback을 사용한다. 0.5 ms physics step 20회마다
`G1HilSensorReader`에서 FSR4 force(N), left-foot accelerometer(m/s^2),
gyroscope(rad/s) 한 sample을 읽는다. Domain variation은 기존 surface/friction/
initial-condition configurator에 적용된다. Sensor imperfection은 Dataset 생성의
50-sample window 추출 뒤 적용되는 offline augmentation이므로 live stream은
`clean_live`로 명시해 기록한다.

MuJoCo virtual environment에 Host shadow runtime을 한 번 설치한다.

```sh
cd simulation/unitree_mujoco/simulate_python
../../venv/bin/python -m pip install -r requirements-live-hil.txt
```

지정된 physical board에서 deterministic concrete run을 먼저 실행한다.

```sh
../../venv/bin/python run_live_terrain_hil.py \
  --terrain concrete \
  --family multisine \
  --surface-index 0 \
  --run-index 0 \
  --runs 1 \
  --stride 1
```

그 후 동일 조건으로 네 terrain을 실행한다.

```sh
../../venv/bin/python run_live_terrain_hil.py \
  --terrain all \
  --family multisine \
  --surface-index 0 \
  --run-index 0 \
  --runs 1 \
  --stride 1
```

기본 port는 probe serial `13070E98012D2400`의 KitProg3 USB-UART이다. 각
run/terrain boundary에서 다음 sample은 sequence 0이 되어 E84 ring을 reset한다.
CSV에는 physical/quantized 10 channels, simulation/wall time, session/run/terrain,
E84와 Host raw/class, parity, cycles, RTT와 lateness를 모두 기록한다. Host
shadow는 canonical 7,048-byte model의 SHA-256을 검증하고 reference INT8
kernels를 사용한다. Host optimized XNNPACK은 일부 input에서 class가 같아도
E84와 raw가 몇 LSB 다를 수 있어 exact parity gate에는 사용하지 않는다.

실제 네 terrain `multisine` surface 0/run 0 결과는 400 samples, 204 inference,
raw/class exact parity 204/204, deadline/drop/timeout/device error 0이었다. Send
period는 mean 10.001 ms, p95 10.163 ms였고 RTT는 mean 1.574 ms, p95
1.724 ms였다. Medium-response aligned window는 4/4였고 모든 exploratory
continuous sliding window accuracy는 78.43%였다. Arbitrary window 결과는
controlled-response training distribution 밖이므로 99.29% Dataset-v1 test
accuracy와 직접 비교하지 않는다. Timing benchmark는 headless가 기본이며
`--gui`는 단일-terrain demonstration 전용이다.
