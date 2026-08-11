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

## UART/USB HIL

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

The installed standalone Vela package does not contain Infineon's named
`PSE8x_U55_400MHz_SOCMEM_200MHz_QUAD_XIP` performance configuration, so it
uses Arm's default U55 memory-bandwidth model. This affects Vela's performance
estimate/scheduling assumptions, not operator support; successful execution on
the physical E84 verifies the generated command stream functionally.
