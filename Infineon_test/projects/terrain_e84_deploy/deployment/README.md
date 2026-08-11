# Terrain INT8 on-device validation

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

Regenerate both variants and the Host golden tensor from the immutable
simulation artifacts:

```sh
python3 tools/generate_terrain_artifacts.py
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
