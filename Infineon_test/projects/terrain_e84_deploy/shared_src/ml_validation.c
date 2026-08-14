/******************************************************************************
* File Name:   ml_validation.c
*
* Description: This file contains the implementation of the validation of the
*              machine learning model.
*
* Related Document: See README.md
*
*
*******************************************************************************
* (c) 2023-2026, Infineon Technologies AG, or an affiliate of Infineon
* Technologies AG. All rights reserved.
* This software, associated documentation and materials ("Software") is
* owned by Infineon Technologies AG or one of its affiliates ("Infineon")
* and is protected by and subject to worldwide patent protection, worldwide
* copyright laws, and international treaty provisions. Therefore, you may use
* this Software only as provided in the license agreement accompanying the
* software package from which you obtained this Software. If no license
* agreement applies, then any use, reproduction, modification, translation, or
* compilation of this Software is prohibited without the express written
* permission of Infineon.
*
* Disclaimer: UNLESS OTHERWISE EXPRESSLY AGREED WITH INFINEON, THIS SOFTWARE
* IS PROVIDED AS-IS, WITH NO WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
* INCLUDING, BUT NOT LIMITED TO, ALL WARRANTIES OF NON-INFRINGEMENT OF
* THIRD-PARTY RIGHTS AND IMPLIED WARRANTIES SUCH AS WARRANTIES OF FITNESS FOR A
* SPECIFIC USE/PURPOSE OR MERCHANTABILITY.
* Infineon reserves the right to make changes to the Software without notice.
* You are responsible for properly designing, programming, and testing the
* functionality and safety of your intended application of the Software, as
* well as complying with any legal requirements related to its use. Infineon
* does not guarantee that the Software will be free from intrusion, data theft
* or loss, or other breaches ("Security Breaches"), and Infineon shall have
* no liability arising out of any Security Breaches. Unless otherwise
* explicitly approved by Infineon, the Software may not be used in any
* application where a failure of the Product or any consequences of the use
* thereof can reasonably be expected to result in personal injury.
*******************************************************************************/
#include "ml_validation.h"
#include "app_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>

#ifndef USE_STREAM_DATA
/* Include regression files */
#include MTB_ML_INCLUDE_MODEL_X_DATA_FILE(MODEL_NAME)
#include MTB_ML_INCLUDE_MODEL_Y_DATA_FILE(MODEL_NAME)
#endif

/*******************************************************************************
* Constants
*******************************************************************************/
#define SUCCESS_RATE       (98.0f)

/* Timeout value for streaming */
#define DEFAULT_TIMEOUT_MS (5000u)

/* Maximum number of output */
#define MAX_NUM_OUTPUT     (8u)

/*******************************************************************************
* Global Variables
*******************************************************************************/
/* NN Model Object */
static mtb_ml_model_t *model_obj;

/* Output/result buffers for the inference engine */
static MTB_ML_DATA_T *result_buffer[MAX_NUM_OUTPUT];

/* Model Output Size */
static int model_output_size[MAX_NUM_OUTPUT];

#ifdef USE_STREAM_DATA
/* Concatenated Output/result buffers for the inference engine */
static MTB_ML_DATA_T *concat_result_buffer;
#endif

/*******************************************************************************
* Function Name: ml_validation_init
********************************************************************************
* Summary:
*   Initialize the Neural Network based on the given model and setup to start
*   regression of the model and profiling configuration.
*
* Parameters:
*   profile_cfg: profiling configuration
*   model_bin: pointer to the model data
*
* Return:
*   cy_rslt_t: the status of the initialization.
*******************************************************************************/
cy_rslt_t ml_validation_init(mtb_ml_profile_config_t profile_cfg,
                             mtb_ml_model_bin_t *model_bin)
{
    cy_rslt_t result;

    /* Initialize the neural network */
    result = mtb_ml_model_init(model_bin,
                               NULL,
                               &model_obj);
    if (CY_RSLT_SUCCESS != result)
    {
        printf("MTB ML initialization failure: %lu\r\n", (unsigned long) result);
        return result;
    }

    mtb_ml_model_profile_config(model_obj, profile_cfg);

    /* Check the number of outputs */
    if (model_obj->output_count > MAX_NUM_OUTPUT)
    {
        printf("Increase the maximum number of output (MAX_NUM_OUTPUT) to %d\n\r", model_obj->output_count);
        return MTB_ML_RESULT_BAD_MODEL;
    }

    /* Obtain the output buffer pointers */
    for (int output_idx = 0; output_idx < model_obj->output_count; output_idx++)
    {
        mtb_ml_model_get_output_tensor(model_obj, &result_buffer[output_idx], &model_output_size[output_idx], output_idx);
    }

#ifdef USE_STREAM_DATA
    /* Allocate output buffer for inference results */
    concat_result_buffer = (MTB_ML_DATA_T *) malloc(model_obj->output_concat_bytes);
    if (concat_result_buffer == NULL)
    {
        printf("ERROR: Allocating memory for concat_result_buffer\r\n");
        mtb_ml_model_deinit(model_obj);
        return MTB_ML_RESULT_ALLOC_ERR;
    }
#endif

    /* Print information about the model */
    mtb_ml_utils_print_model_info(model_obj);
    printf("TERRAIN_MEMORY, model_bytes=%d, arena_configured_bytes=%d, "
           "arena_used_bytes=%d, input_bytes=%u, output_bytes=%u\r\n",
           model_obj->model_size, model_bin->arena_size, model_obj->buffer_size,
           (unsigned int) model_obj->input_concat_bytes,
           (unsigned int) model_obj->output_concat_bytes);
    printf("TERRAIN_QUANT, input_scale=%.9f, input_zero_point=%d, "
           "output_scale=%.9f, output_zero_point=%d\r\n",
           (double) model_obj->input_scale, model_obj->input_zero_point,
           (double) model_obj->output_scale, model_obj->output_zero_point);

    return CY_RSLT_SUCCESS;
}

#ifndef USE_STREAM_DATA
/*******************************************************************************
* UART HIL protocols:
* TRN1: uint16 LE length, 500 INT8 window bytes, uint32 LE payload CRC-32.
* TRN2: uint16 LE length, uint32 LE sequence, uint16 LE inference stride,
*       10 INT8 sample bytes, uint32 LE payload CRC-32.
*******************************************************************************/
#define TERRAIN_CHANNEL_COUNT          (10u)
#define TERRAIN_WINDOW_SAMPLES         (50u)
#define TERRAIN_WINDOW_BYTES           (TERRAIN_CHANNEL_COUNT * TERRAIN_WINDOW_SAMPLES)
#define TERRAIN_STREAM_HEADER_BYTES    (6u)
#define TERRAIN_STREAM_PAYLOAD_BYTES   (TERRAIN_STREAM_HEADER_BYTES + TERRAIN_CHANNEL_COUNT)

typedef struct
{
    int8_t samples[TERRAIN_WINDOW_SAMPLES][TERRAIN_CHANNEL_COUNT];
    uint16_t write_index;
    uint16_t fill;
    uint16_t stride;
    uint16_t inference_countdown;
    uint32_t last_sequence;
    bool have_sequence;
} terrain_stream_state_t;

static terrain_stream_state_t terrain_stream_state;

static uint32_t terrain_crc32(const uint8_t *data, size_t size)
{
    uint32_t crc = 0xffffffffu;
    for (size_t index = 0; index < size; index++)
    {
        crc ^= data[index];
        for (uint32_t bit = 0; bit < 8u; bit++)
        {
            crc = (crc >> 1) ^ ((crc & 1u) ? 0xedb88320u : 0u);
        }
    }
    return crc ^ 0xffffffffu;
}

static uint8_t terrain_uart_get_byte(void)
{
    uint8_t value = 0u;
    cy_rslt_t result = mtb_hal_uart_get(&mtb_ml_retarget_io_uart_obj, &value, 0u);
    if (result != CY_RSLT_SUCCESS)
    {
        printf("HIL_ERROR uart_rx=0x%08lx\r\n", (unsigned long) result);
    }
    return value;
}

static uint16_t terrain_uart_get_u16(void)
{
    uint16_t value = terrain_uart_get_byte();
    value |= (uint16_t) ((uint16_t) terrain_uart_get_byte() << 8);
    return value;
}

static uint32_t terrain_uart_get_u32(void)
{
    uint32_t value = 0u;
    for (uint32_t index = 0u; index < 4u; index++)
    {
        value |= (uint32_t) terrain_uart_get_byte() << (8u * index);
    }
    return value;
}

static uint8_t terrain_uart_find_protocol(void)
{
    static const uint8_t prefix[3] = {'T', 'R', 'N'};
    size_t matched = 0u;
    for (;;)
    {
        uint8_t value = terrain_uart_get_byte();
        if (matched < sizeof(prefix))
        {
            if (value == prefix[matched])
            {
                matched++;
            }
            else
            {
                matched = (value == prefix[0]) ? 1u : 0u;
            }
            continue;
        }
        if ((value == '1') || (value == '2'))
        {
            return value;
        }
        matched = (value == prefix[0]) ? 1u : 0u;
    }
}

static cy_rslt_t terrain_invoke(const int8_t *input)
{
    cy_rslt_t result = mtb_ml_model_inputs(model_obj, (MTB_ML_DATA_T *) input, 0);
    if (result == CY_RSLT_SUCCESS)
    {
        result = mtb_ml_model_invoke(model_obj);
    }
    return result;
}

static void terrain_print_hil_result(void)
{
    int predicted_class = mtb_ml_utils_find_max(result_buffer[0], model_output_size[0]);
    printf("HIL_RESULT raw=[");
    for (int index = 0; index < model_output_size[0]; index++)
    {
        printf("%d%s", (int) result_buffer[0][index],
               (index + 1 == model_output_size[0]) ? "" : ",");
    }
    printf("],class=%d,cpu_cyc=%lu", predicted_class,
           (unsigned long) model_obj->m_cpu_cycles);
#if defined(COMPONENT_U55) || defined(COMPONENT_NNLITE2)
    printf(",npu_cyc=%lu", (unsigned long) model_obj->m_npu_cycles);
#endif
    printf("\r\n");
}

static cy_rslt_t terrain_handle_window_frame(uint16_t length)
{
    int8_t input[TERRAIN_WINDOW_BYTES];

    if (length != sizeof(input))
    {
        printf("HIL_ERROR protocol=TRN1,bad_length=%u\r\n", (unsigned int) length);
        return CY_RSLT_SUCCESS;
    }
    for (size_t index = 0u; index < sizeof(input); index++)
    {
        input[index] = (int8_t) terrain_uart_get_byte();
    }
    uint32_t received_crc = terrain_uart_get_u32();
    uint32_t computed_crc = terrain_crc32((uint8_t *) input, sizeof(input));
    if (received_crc != computed_crc)
    {
        printf("HIL_ERROR protocol=TRN1,bad_crc=0x%08" PRIx32
               ",expected=0x%08" PRIx32 "\r\n", received_crc, computed_crc);
        return CY_RSLT_SUCCESS;
    }
    cy_rslt_t result = terrain_invoke(input);
    if (result != CY_RSLT_SUCCESS)
    {
        printf("HIL_ERROR protocol=TRN1,inference=0x%08lx\r\n",
               (unsigned long) result);
        return result;
    }
    terrain_print_hil_result();
    return CY_RSLT_SUCCESS;
}

static void terrain_stream_reset(void)
{
    terrain_stream_state.write_index = 0u;
    terrain_stream_state.fill = 0u;
    terrain_stream_state.inference_countdown = 0u;
}

static void terrain_stream_build_window(int8_t *window)
{
    for (uint16_t row = 0u; row < TERRAIN_WINDOW_SAMPLES; row++)
    {
        uint16_t source_row = (uint16_t)
            ((terrain_stream_state.write_index + row) % TERRAIN_WINDOW_SAMPLES);
        for (uint16_t channel = 0u; channel < TERRAIN_CHANNEL_COUNT; channel++)
        {
            window[(row * TERRAIN_CHANNEL_COUNT) + channel] =
                terrain_stream_state.samples[source_row][channel];
        }
    }
}

static void terrain_print_stream_result(uint32_t sequence, bool inferred)
{
    int predicted_class = -1;
    printf("STREAM_RESULT seq=%lu,fill=%u,warmup=%u,inferred=%u,class=",
           (unsigned long) sequence, (unsigned int) terrain_stream_state.fill,
           (terrain_stream_state.fill < TERRAIN_WINDOW_SAMPLES) ? 1u : 0u,
           inferred ? 1u : 0u);
    if (inferred)
    {
        predicted_class = mtb_ml_utils_find_max(result_buffer[0], model_output_size[0]);
    }
    printf("%d,raw=[", predicted_class);
    for (int index = 0; index < 4; index++)
    {
        int value = inferred ? (int) result_buffer[0][index] : 0;
        printf("%d%s", value, (index == 3) ? "" : ",");
    }
    printf("],cpu_cyc=%lu,npu_cyc=%lu\r\n",
           inferred ? (unsigned long) model_obj->m_cpu_cycles : 0ul,
#if defined(COMPONENT_U55) || defined(COMPONENT_NNLITE2)
           inferred ? (unsigned long) model_obj->m_npu_cycles : 0ul
#else
           0ul
#endif
    );
}

static cy_rslt_t terrain_handle_stream_frame(uint16_t length)
{
    uint8_t payload[TERRAIN_STREAM_PAYLOAD_BYTES];
    if (length != sizeof(payload))
    {
        printf("STREAM_ERROR code=bad_length,length=%u\r\n", (unsigned int) length);
        return CY_RSLT_SUCCESS;
    }
    for (size_t index = 0u; index < sizeof(payload); index++)
    {
        payload[index] = terrain_uart_get_byte();
    }
    uint32_t received_crc = terrain_uart_get_u32();
    uint32_t computed_crc = terrain_crc32(payload, sizeof(payload));
    uint32_t sequence = (uint32_t) payload[0]
        | ((uint32_t) payload[1] << 8)
        | ((uint32_t) payload[2] << 16)
        | ((uint32_t) payload[3] << 24);
    uint16_t stride = (uint16_t) payload[4] | ((uint16_t) payload[5] << 8);
    if (received_crc != computed_crc)
    {
        printf("STREAM_ERROR seq=%lu,code=bad_crc,received=0x%08" PRIx32
               ",expected=0x%08" PRIx32 "\r\n", (unsigned long) sequence,
               received_crc, computed_crc);
        return CY_RSLT_SUCCESS;
    }
    if (stride == 0u)
    {
        printf("STREAM_ERROR seq=%lu,code=bad_stride,stride=0\r\n",
               (unsigned long) sequence);
        return CY_RSLT_SUCCESS;
    }
    if (terrain_stream_state.have_sequence && (sequence == 0u))
    {
        /* Sequence zero starts a new client session without requiring a reset. */
        terrain_stream_reset();
    }
    else if (terrain_stream_state.have_sequence
             && (sequence != (terrain_stream_state.last_sequence + 1u)))
    {
        printf("STREAM_ERROR seq=%lu,code=sequence,expected=%lu,action=reset\r\n",
               (unsigned long) sequence,
               (unsigned long) (terrain_stream_state.last_sequence + 1u));
        terrain_stream_reset();
    }
    terrain_stream_state.have_sequence = true;
    terrain_stream_state.last_sequence = sequence;
    if (terrain_stream_state.stride != stride)
    {
        terrain_stream_state.stride = stride;
        terrain_stream_state.inference_countdown = 0u;
    }
    for (uint16_t channel = 0u; channel < TERRAIN_CHANNEL_COUNT; channel++)
    {
        terrain_stream_state.samples[terrain_stream_state.write_index][channel] =
            (int8_t) payload[TERRAIN_STREAM_HEADER_BYTES + channel];
    }
    terrain_stream_state.write_index =
        (uint16_t) ((terrain_stream_state.write_index + 1u) % TERRAIN_WINDOW_SAMPLES);
    if (terrain_stream_state.fill < TERRAIN_WINDOW_SAMPLES)
    {
        terrain_stream_state.fill++;
    }

    bool inferred = false;
    cy_rslt_t result = CY_RSLT_SUCCESS;
    if (terrain_stream_state.fill == TERRAIN_WINDOW_SAMPLES)
    {
        if (terrain_stream_state.inference_countdown == 0u)
        {
            int8_t input[TERRAIN_WINDOW_BYTES];
            terrain_stream_build_window(input);
            result = terrain_invoke(input);
            if (result != CY_RSLT_SUCCESS)
            {
                printf("STREAM_ERROR seq=%lu,code=inference,result=0x%08lx\r\n",
                       (unsigned long) sequence, (unsigned long) result);
                return result;
            }
            inferred = true;
            terrain_stream_state.inference_countdown = (uint16_t) (stride - 1u);
        }
        else
        {
            terrain_stream_state.inference_countdown--;
        }
    }
    terrain_print_stream_result(sequence, inferred);
    return CY_RSLT_SUCCESS;
}

cy_rslt_t ml_validation_hil_task(void)
{
    if ((model_obj->input_count != 1)
        || (model_obj->input_concat_bytes != TERRAIN_WINDOW_BYTES)
        || (model_obj->output_count != 1)
        || (model_output_size[0] != 4))
    {
        printf("HIL_ERROR unexpected_model_io,input_bytes=%u,output_values=%d\r\n",
               (unsigned int) model_obj->input_concat_bytes, model_output_size[0]);
        return MTB_ML_RESULT_INPUT_ERROR;
    }
    terrain_stream_reset();
    terrain_stream_state.have_sequence = false;
    terrain_stream_state.stride = 1u;
    printf("HIL_READY protocols=TRN1,TRN2,window_bytes=500,sample_bytes=10,"
           "window_samples=50,baud=1000000\r\n");
    for (;;)
    {
        uint8_t protocol = terrain_uart_find_protocol();
        uint16_t length = terrain_uart_get_u16();
        cy_rslt_t result = (protocol == '1')
            ? terrain_handle_window_frame(length)
            : terrain_handle_stream_frame(length);
        if (result != CY_RSLT_SUCCESS)
        {
            return result;
        }
    }
}

/* Fast Reflex v2 Sink continuous HIL.  FRV2 reuses the Terrain UART length
 * and CRC-32 framing but carries a batch of raw IEEE-754 Fusion10 samples:
 *   "FRV2" | u16 payload_len | u32 first_sequence | u16 count | count*10*f32 |
 *   u32 crc32(payload).  count is deliberately capped at 20: 800 byte payload
 * fits the existing blocking UART API while amortising response RTT. */
#if defined(FAST_REFLEX_SINK_HIL) || defined(FAST_REFLEX_SLIP_HIL)
#if defined(FAST_REFLEX_SLIP_HIL)
#define FRV2_WINDOW 5u
#define FRV2_THRESHOLD 121
#define FRV2_PERSISTENCE 3u
#define FRV2_INPUT_SCALE 0.0939839631319046f
static const float frv2_mean[10] = {31.594806671f,31.777748108f,13.546282768f,13.864206314f,-.086954400f,.008436359f,9.803998947f,.000235322f,-.003542150f,-.001164354f};
static const float frv2_std[10] = {8.742906570f,8.528360367f,3.996133566f,3.821703196f,.583785653f,.310120761f,.621062100f,.010392545f,.010778599f,.007863495f};
#else
#define FRV2_WINDOW 20u
#define FRV2_THRESHOLD 124
#define FRV2_PERSISTENCE 1u
#define FRV2_INPUT_SCALE 0.09580779820680618f
static const float frv2_mean[10] = {31.530698776f,31.693687439f,13.539832115f,13.843737602f,-.073472634f,.008472550f,9.797912598f,.000206427f,-.003185092f,-.001231437f};
static const float frv2_std[10] = {8.335794449f,8.145730972f,3.759648561f,3.584739923f,.541470349f,.275717556f,.565855443f,.009250504f,.010388087f,.007618863f};
#endif
#define FRV2_CHANNELS 10u
#define FRV2_MAX_BATCH 20u
#define FRV2_PAYLOAD_HEADER 6u
#define FRV2_SAMPLE_BYTES (FRV2_CHANNELS * 4u)
#define FRV2_DEADLINE_CYCLES 400000u /* 1 ms at the fixed 400 MHz CM55 clock */
static int8_t frv2_ring[FRV2_WINDOW][FRV2_CHANNELS];
static uint16_t frv2_write, frv2_fill;
static uint32_t frv2_last_sequence, frv2_drops, frv2_crc_errors, frv2_deadline_miss, frv2_consecutive;
static bool frv2_have_sequence;

static void frv2_reset(void) { frv2_write=0u; frv2_fill=0u; frv2_consecutive=0u; frv2_have_sequence=false; }
static int8_t frv2_quantize(float raw, uint16_t ch)
{
    long q = lrintf(((raw - frv2_mean[ch]) / frv2_std[ch]) / FRV2_INPUT_SCALE +
#if defined(FAST_REFLEX_SLIP_HIL)
                    -6.0f);
#else
                    -8.0f);
#endif
    return (int8_t)((q < -128l) ? -128l : ((q > 127l) ? 127l : q));
}
static void frv2_window(int8_t *out)
{
    for (uint16_t row=0u; row<FRV2_WINDOW; row++)
        for (uint16_t ch=0u; ch<FRV2_CHANNELS; ch++)
            out[row*FRV2_CHANNELS+ch]=frv2_ring[(frv2_write+row)%FRV2_WINDOW][ch];
}
static cy_rslt_t frv2_handle(uint16_t length)
{
    uint8_t payload[FRV2_PAYLOAD_HEADER + FRV2_MAX_BATCH * FRV2_SAMPLE_BYTES];
    if (length < FRV2_PAYLOAD_HEADER || length > sizeof(payload)) { printf("FRV2_ERROR code=bad_length,length=%u\r\n",(unsigned)length); return CY_RSLT_SUCCESS; }
    for (uint16_t i=0u;i<length;i++) payload[i]=terrain_uart_get_byte();
    uint32_t received=terrain_uart_get_u32(), computed=terrain_crc32(payload,length);
    if (received!=computed) { frv2_crc_errors++; printf("FRV2_ERROR code=bad_crc\r\n"); return CY_RSLT_SUCCESS; }
    uint32_t sequence=(uint32_t)payload[0]|((uint32_t)payload[1]<<8)|((uint32_t)payload[2]<<16)|((uint32_t)payload[3]<<24);
    uint16_t count=(uint16_t)payload[4]|((uint16_t)payload[5]<<8);
    if (count==0u || count>FRV2_MAX_BATCH || length != FRV2_PAYLOAD_HEADER+count*FRV2_SAMPLE_BYTES) { printf("FRV2_ERROR code=bad_count\r\n"); return CY_RSLT_SUCCESS; }
    if (sequence == 0u) { frv2_reset(); frv2_drops=0u; frv2_crc_errors=0u; frv2_deadline_miss=0u; }
    else if (frv2_have_sequence && sequence != frv2_last_sequence+1u) { frv2_drops += (sequence>frv2_last_sequence) ? sequence-frv2_last_sequence-1u : 1u; frv2_reset(); }
    int8_t window[FRV2_WINDOW*FRV2_CHANNELS]; uint32_t raw_crc=0xffffffffu, q_crc=0xffffffffu, inferred=0u, fires=0u, max_cpu=0u;
    for (uint16_t sample=0u;sample<count;sample++) {
        for (uint16_t ch=0u;ch<FRV2_CHANNELS;ch++) { union {uint32_t u; float f;} v; uint16_t off=FRV2_PAYLOAD_HEADER+sample*FRV2_SAMPLE_BYTES+ch*4u; v.u=(uint32_t)payload[off]|((uint32_t)payload[off+1]<<8)|((uint32_t)payload[off+2]<<16)|((uint32_t)payload[off+3]<<24); frv2_ring[frv2_write][ch]=frv2_quantize(v.f,ch); }
        frv2_write=(frv2_write+1u)%FRV2_WINDOW; if(frv2_fill<FRV2_WINDOW) frv2_fill++;
        if(frv2_fill==FRV2_WINDOW) { frv2_window(window); cy_rslt_t r=terrain_invoke(window); if(r!=CY_RSLT_SUCCESS)return r; int8_t raw=result_buffer[0][0]; raw_crc=terrain_crc32((uint8_t*)&raw,1u) ^ raw_crc; q_crc=terrain_crc32((uint8_t*)window,sizeof(window)) ^ q_crc; inferred++; frv2_consecutive=(raw>=FRV2_THRESHOLD)?frv2_consecutive+1u:0u; fires+=(frv2_consecutive>=FRV2_PERSISTENCE); if(model_obj->m_cpu_cycles>max_cpu)max_cpu=model_obj->m_cpu_cycles; if(model_obj->m_cpu_cycles>FRV2_DEADLINE_CYCLES)frv2_deadline_miss++; }
    }
    frv2_last_sequence=sequence+count-1u; frv2_have_sequence=true;
    printf("FRV2_RESULT seq=%lu,count=%u,inferred=%lu,fires=%lu,drops=%lu,crc_errors=%lu,deadline_miss=%lu,quant_digest=%08lx,raw_digest=%08lx,max_cpu_cyc=%lu\r\n",(unsigned long)sequence,(unsigned)count,(unsigned long)inferred,(unsigned long)fires,(unsigned long)frv2_drops,(unsigned long)frv2_crc_errors,(unsigned long)frv2_deadline_miss,(unsigned long)q_crc,(unsigned long)raw_crc,(unsigned long)max_cpu);
    return CY_RSLT_SUCCESS;
}
cy_rslt_t ml_validation_fast_reflex_sink_hil_task(void)
{
    if (model_obj->input_concat_bytes != FRV2_WINDOW*FRV2_CHANNELS || model_output_size[0] != 1) return MTB_ML_RESULT_INPUT_ERROR;
    frv2_reset(); printf("FRV2_READY protocol=FRV2,batch_max=20,sample_rate_hz=1000,window=%u,baud=1000000,threshold_raw=%u,persistence=%u\r\n",(unsigned)FRV2_WINDOW,(unsigned)FRV2_THRESHOLD,(unsigned)FRV2_PERSISTENCE);
    for (;;) { uint8_t p[4]; do {p[0]=terrain_uart_get_byte();} while(p[0]!='F'); p[1]=terrain_uart_get_byte();p[2]=terrain_uart_get_byte();p[3]=terrain_uart_get_byte(); if(p[1]=='R'&&p[2]=='V'&&p[3]=='2'){cy_rslt_t r=frv2_handle(terrain_uart_get_u16());if(r!=CY_RSLT_SUCCESS)return r;} }
}
#endif /* FAST_REFLEX_*_HIL */

/*******************************************************************************
* Function Name: ml_validation_local_task
********************************************************************************
* Summary:
*   Run the Neural Network Inference Engine based on the local data.
*
* Parameters:
*   void
*
* Return:
*   cy_rslt_t: the status of the task execution.
*******************************************************************************/
cy_rslt_t ml_validation_local_task(void)
{
    /* Regression pointers */
    MTB_ML_DATA_T  *input_reference;
    MTB_ML_DATA_T  *output_reference;
       
    uint32_t     num_loop;
    uint32_t     correct_result[MAX_NUM_OUTPUT] = {0};
    bool         test_result;
    uint32_t     total_count = 0;
    cy_rslt_t    result;
    int          file_input_size;
    int          model_input_size = (mtb_ml_model_get_input_size(model_obj) / sizeof(MTB_ML_DATA_T));

    /* Parse input data information: 
     * - Data type (TFLM only)
     * - Number of samples
     * - Frame size
     */
    mtb_ml_x_file_header_t *x_file_header = (mtb_ml_x_file_header_t *) MTB_ML_MODEL_X_DATA_BIN(MODEL_NAME);

    /* Point to regression data */
    input_reference  = (MTB_ML_DATA_T *) (((uint32_t) x_file_header) + sizeof(*x_file_header));
    output_reference = (MTB_ML_DATA_T *) MTB_ML_MODEL_Y_DATA_BIN(MODEL_NAME);

    /* Get the number of loops for this regression */
    num_loop = x_file_header->num_of_samples;

    /* Get the number of inputs of the NN */
    file_input_size = x_file_header->input_size;

#if defined(RNN_STREAMING)

    model_obj->recurrent_ts_size = x_file_header->recurrent_ts_size;

    /* Check if it is a non-RNN model */
    if (model_obj->recurrent_ts_size <= 0)
    {
        printf("This is not a RNN model (%d). Set the NN_RNN_MODEL variable to NO in the Makefile, aborting...\r\n", 
            model_obj->recurrent_ts_size);
        return MTB_ML_RESULT_MISMATCH_DATA_TYPE;
    }

    /* If using RRN Model, check if the data time steps matches */
    if ((file_input_size/model_obj->recurrent_ts_size) != model_input_size)
    {
        printf("Data size error, file input size=%d, model input size=%d recurrent time series size=%d, aborting...\r\n", 
            file_input_size, model_input_size, model_obj->recurrent_ts_size);
        return MTB_ML_RESULT_MISMATCH_DATA_TYPE;
    }

    /* Allocate memory for the RNN input slice */
    MTB_ML_DATA_T * input_slice = (MTB_ML_DATA_T *) malloc (model_input_size * model_obj->input_type_size);

    if (input_slice == NULL)
    {
        printf("ERROR: Allocating memory for input slice\r\n");
        return MTB_ML_RESULT_ALLOC_ERR;
    }

#else
    /* Check if the file input size matches the model input size */
    if (file_input_size != model_input_size)
    {
        printf("Input buffer size error, file input size=%d, model input size=%d, aborting...\r\n", 
                file_input_size, model_input_size);
        return MTB_ML_RESULT_MISMATCH_DATA_TYPE;
    }
#endif /* RNN_STREAMING */

    /* The following loop runs for number of examples used in regression */
    for (int j = 0; j < num_loop; j++)
    {

#if defined(RNN_STREAMING)
        result = mtb_ml_model_rnn_reset_all_parameters(model_obj);
        if (MTB_ML_RESULT_SUCCESS != result)
        {
            printf("ERROR: failed to reset model parameters\r\n");
            free(input_slice);
            return MTB_ML_RESULT_INFERENCE_ERROR;
        }

        for (int i = 0; i < model_obj->recurrent_ts_size; i++)
        {
            /* Input data is 2D array squashed to 1D array by Coretools */
            for (int z = 0; z < model_input_size; z++)
            {
                input_slice[z] = input_reference[i*model_input_size+z];
            }

            /* Load input data into each input tensor for multi-input models */
            for (int input_idx = 0; input_idx < model_obj->input_count; input_idx++)
            {
                MTB_ML_DATA_T *input_tensor_data = (MTB_ML_DATA_T *)((uint8_t *) input_slice +
                                            model_obj->inputs[input_idx].concat_offset_bytes);

                result = mtb_ml_model_inputs(model_obj, input_tensor_data, input_idx);
                if (MTB_ML_RESULT_SUCCESS != result)
                {
                    return result;
                }
            }

        /* Invoke the model */
        result = mtb_ml_model_invoke(model_obj);

            /* Check if the inferencing return any error */
            if (MTB_ML_RESULT_SUCCESS != result)
            {
                free(input_slice);
                return result;
            }
        }

#else
        /* Load input data into each input tensor for multi-input models */
        for (int input_idx = 0; input_idx < model_obj->input_count; input_idx++)
        {
            MTB_ML_DATA_T *input_tensor_data = (MTB_ML_DATA_T *)((uint8_t *) input_reference +
                                           model_obj->inputs[input_idx].concat_offset_bytes);

            result = mtb_ml_model_inputs(model_obj, input_tensor_data, input_idx);
            if (MTB_ML_RESULT_SUCCESS != result)
            {
                return result;
            }
        }

        /* Invoke the model */
        result = mtb_ml_model_invoke(model_obj);
        
        /* Check if the inferencing return any error */
        if (MTB_ML_RESULT_SUCCESS != result)
        {
            return result;
        }
#endif /* RNN_STREAMING */

        for (int output_idx = 0; output_idx < model_obj->output_count; output_idx++)
        {
            output_reference += model_obj->outputs[output_idx].concat_offset_bytes;

            printf("TERRAIN_RESULT, output[%d]_raw=[", output_idx);
            for (int value_idx = 0; value_idx < model_output_size[output_idx]; value_idx++)
            {
                printf("%d%s", (int) result_buffer[output_idx][value_idx],
                       (value_idx + 1 == model_output_size[output_idx]) ? "" : ",");
            }
            printf("], host_raw=[");
            for (int value_idx = 0; value_idx < model_output_size[output_idx]; value_idx++)
            {
                printf("%d%s", (int) output_reference[value_idx],
                       (value_idx + 1 == model_output_size[output_idx]) ? "" : ",");
            }
            printf("], device_class=%d, host_class=%d\r\n",
                   mtb_ml_utils_find_max(result_buffer[output_idx], model_output_size[output_idx]),
                   mtb_ml_utils_find_max(output_reference, model_output_size[output_idx]));

#if defined(FAST_REFLEX_GOLDEN)
            {
                bool raw_match = true;
                for (int value_idx = 0; value_idx < model_output_size[output_idx]; value_idx++)
                {
                    if (result_buffer[output_idx][value_idx] != output_reference[value_idx]) raw_match = false;
                }
                printf("FRV2 vec=%d raw=%d expected=%d decision=%d expected_decision=%d raw_exact=%s\r\n", j,
                       (int) result_buffer[output_idx][0], (int) output_reference[0],
                       ((int) result_buffer[output_idx][0] >= FAST_REFLEX_THRESHOLD_RAW),
                       ((int) output_reference[0] >= FAST_REFLEX_THRESHOLD_RAW), raw_match ? "PASS" : "FAIL");
                if (raw_match) correct_result[output_idx]++;
                continue;
            }
#endif

            /* Check if the results are accurate enough */
            if (mtb_ml_utils_find_max(result_buffer[output_idx], model_output_size[output_idx]) ==
                mtb_ml_utils_find_max(output_reference, model_output_size[output_idx]))
            {
                correct_result[output_idx]++;
            }
        }

        /* Increment buffers */
        input_reference  += file_input_size;
        output_reference += model_output_size[model_obj->output_count - 1];

        total_count++;
    }

#if defined(RNN_STREAMING)
    free(input_slice);
#endif /* RNN_STREAMING */

    /* Print the profiling information */
    mtb_ml_model_profile_log(model_obj);

    /* Print PASS or FAIL with Accuracy percentage 
     * Only for regression ... 
     */
    for (int output_idx = 0; output_idx < model_obj->output_count; output_idx++)
    {
        float success_rate;

        /* Check if total count is equal to ZERO */
        if (total_count == 0)
        {
            success_rate = 0;
        }
        else
        {
            success_rate = ((float) correct_result[output_idx]) * 100.0f / ((float) total_count);
        }
        
        test_result = (success_rate >= SUCCESS_RATE);
        
        printf("\r\n****************************************************************\r\n");
        if (test_result == true)
        {
            printf("Output[%d] : PASS with accuracy percentage =%3.2f, total_cnt=%d", output_idx, success_rate, (int) total_count);
        }
        else
        {
            printf("Output[%d] : FAIL with accuracy percentage =%3.2f, total_cnt=%d", output_idx, success_rate, (int) total_count);
        }
        printf("\r\n****************************************************************\r\n");
    }

    return CY_RSLT_SUCCESS;
}
#endif /* USE_STREAM_DATA */

#ifdef USE_STREAM_DATA
/*******************************************************************************
* Function Name: ml_validation_stream_task
********************************************************************************
* Summary:
*   Run the Neural Network Inference Engine based on the stream data.
*
* Parameters:
*   iface: pointer to the streaming interface
*
* Return:
*   cy_rslt_t: the status of the task execution.
*******************************************************************************/
cy_rslt_t ml_validation_stream_task(mtb_ml_stream_interface_t *iface)
{
    cy_rslt_t result = MTB_ML_RESULT_SUCCESS;
    int num_of_inferences;

    /* Initialize the streaming interface */
    result = mtb_ml_stream_init(iface, model_obj);
    if (CY_RSLT_SUCCESS != result)
    {
        printf("MTB ML streaming init failure: %lu\r\n", (unsigned long) result);
        return result;
    }

    /* Alloc RX buf */
    MTB_ML_DATA_T *rx_buf = (MTB_ML_DATA_T *) malloc(iface->input_size * model_obj->input_type_size);
    if(!rx_buf)
    {
        printf("ERROR: Allocating memory for rx_buf\r\n");
        return MTB_ML_RESULT_ALLOC_ERR;
    }

    num_of_inferences = iface->x_data_info.num_of_samples;

#if defined(RNN_STREAMING)
    /* Handle the RNN models differently */
    model_obj->recurrent_ts_size = iface->x_data_info.recurrent_ts_size;

    /* For non-streaming models data is sent by frames/samples, but for streaming frame slicing is applied.
    Number of slices can be calculated as: num_of_samples * recurrent_ts_size */
    num_of_inferences *= model_obj->recurrent_ts_size;

#endif /* RNN_STREAMING */

    /* Do frame-by-frame (sample == frame) inference */
    for (int i = 0; i < num_of_inferences; i++)
    {

#if defined(RNN_STREAMING)
        if ((i % model_obj->recurrent_ts_size) == 0)
        {
            result = mtb_ml_model_rnn_reset_all_parameters(model_obj);
        }
#else
        result = mtb_ml_model_rnn_reset_all_parameters(model_obj);
        if (MTB_ML_RESULT_SUCCESS != result)
        {
            printf("ERROR: failed to reset model parameters\r\n");
            free(rx_buf);
            return MTB_ML_RESULT_INFERENCE_ERROR;
        }
#endif /* RNN_STREAMING */

        /* Get input data */
        result = mtb_ml_stream_input_data(iface, rx_buf, DEFAULT_TIMEOUT_MS);
        if(MTB_ML_RESULT_SUCCESS != result)
        {
            printf("ERROR: Failed to receive input data from host.\r\n");
            break;
        }

        /* Load input data into each input tensor for multi-input models */
        for (int input_idx = 0; input_idx < model_obj->input_count; input_idx++)
        {
            MTB_ML_DATA_T *input_tensor_data = (MTB_ML_DATA_T *)((uint8_t *) rx_buf +
                                           model_obj->inputs[input_idx].concat_offset_bytes);

            result = mtb_ml_model_inputs(model_obj, input_tensor_data, input_idx);
            if (MTB_ML_RESULT_SUCCESS != result)
            {
                return result;
            }
        }

        /* Invoke the model */
        result = mtb_ml_model_invoke(model_obj);
        if (MTB_ML_RESULT_SUCCESS != result)
        {
            free(rx_buf);
            return result;
        }

        /* Concatenating outputs into result_buffer allocated in ml_validation_init() */
        result = mtb_ml_model_load_output(model_obj, &concat_result_buffer);
        if (MTB_ML_RESULT_SUCCESS != result)
        {
            free(rx_buf);
            return result;
        }

        /* Send output data.
           For RNN streaming models data needs to be sent only when whole frame was streamed */
#if defined(RNN_STREAMING)
        if (((i+1) % model_obj->recurrent_ts_size) == 0)
#endif
        {
            /* Send output data */
            result = mtb_ml_stream_output_data(iface, concat_result_buffer, DEFAULT_TIMEOUT_MS);
            if(MTB_ML_RESULT_SUCCESS != result)
            {
                printf("ERROR: Failed to send output data to host\r\n");
                free(rx_buf);
                return MTB_ML_RESULT_ALLOC_ERR;
            }
        }
    }

    /* Free allocated memory */
    free(rx_buf);

    /* Generate profiling log if it is enabled */
    result = mtb_ml_model_profile_log(model_obj);
    if(MTB_ML_RESULT_SUCCESS != result)
    {
        printf("ERROR: Failed to generate profile log.\r\n");
        return MTB_ML_RESULT_BAD_MODEL;
    }

    return mtb_ml_inform_host_done(iface, DEFAULT_TIMEOUT_MS);
}
#endif

/* [] END OF FILE */
