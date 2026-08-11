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
* UART HIL protocol: "TRN1", uint16 little-endian payload length, 500 INT8
* values, then IEEE CRC-32 of the payload as uint32 little-endian.
*******************************************************************************/
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

cy_rslt_t ml_validation_hil_task(void)
{
    static const uint8_t magic[4] = {'T', 'R', 'N', '1'};
    uint8_t input[500];
    size_t matched = 0;

    if ((model_obj->input_count != 1) || (model_obj->input_concat_bytes != sizeof(input)))
    {
        printf("HIL_ERROR unexpected_input_bytes=%u\r\n",
               (unsigned int) model_obj->input_concat_bytes);
        return MTB_ML_RESULT_INPUT_ERROR;
    }
    printf("HIL_READY protocol=TRN1,payload_bytes=500,baud=1000000\r\n");
    for (;;)
    {
        uint8_t value = terrain_uart_get_byte();
        if (value == magic[matched])
        {
            matched++;
        }
        else
        {
            matched = (value == magic[0]) ? 1u : 0u;
        }
        if (matched != sizeof(magic))
        {
            continue;
        }
        matched = 0;
        uint16_t length = (uint16_t) terrain_uart_get_byte();
        length |= (uint16_t) ((uint16_t) terrain_uart_get_byte() << 8);
        if (length != sizeof(input))
        {
            printf("HIL_ERROR bad_length=%u\r\n", (unsigned int) length);
            continue;
        }
        for (size_t index = 0; index < sizeof(input); index++)
        {
            input[index] = terrain_uart_get_byte();
        }
        uint32_t received_crc = 0u;
        for (uint32_t index = 0; index < 4u; index++)
        {
            received_crc |= (uint32_t) terrain_uart_get_byte() << (8u * index);
        }
        uint32_t computed_crc = terrain_crc32(input, sizeof(input));
        if (received_crc != computed_crc)
        {
            printf("HIL_ERROR bad_crc=0x%08" PRIx32 ",expected=0x%08" PRIx32 "\r\n",
                   received_crc, computed_crc);
            continue;
        }
        cy_rslt_t result = mtb_ml_model_inputs(model_obj, (MTB_ML_DATA_T *) input, 0);
        if (result == CY_RSLT_SUCCESS)
        {
            result = mtb_ml_model_invoke(model_obj);
        }
        if (result != CY_RSLT_SUCCESS)
        {
            printf("HIL_ERROR inference=0x%08lx\r\n", (unsigned long) result);
            return result;
        }
        terrain_print_hil_result();
    }
}

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
