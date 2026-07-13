/******************************************************************************
* File Name:   dev_pdm_pcm.c
*
* Description: This file implements the interface with the PDM/PCM.
*
* Related Document: See README.md
*
*******************************************************************************
* (c) 2025-2026, Infineon Technologies AG, or an affiliate of Infineon
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
#ifdef IM_ENABLE_PDM_PCM

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "cy_result.h"
#include "pdm_pcm.h"
#include "protocol/protocol.h"
#include "protocol/pb_encode.h"
#include "dev_pdm_pcm.h"

/******************************************************************************
 * Macros
 *****************************************************************************/
#define FREQUENCY_OPTION_COUNT      (3u)
#define MIC_OPTION_KEY_GAIN 10
#define MIC_OPTION_KEY_STEREO 20
#define MIC_OPTION_KEY_FREQUENCY 30


/* PDM-PCM Configuration data */
#define NUM_CHANNELS                            (2u)
#define LEFT_CH_INDEX                           (2u)
#define RIGHT_CH_INDEX                          (3u)
#define LEFT_CH_CONFIG                          channel_2_config
#define RIGHT_CH_CONFIG                         channel_3_config


/******************************************************************************
 * Type Definitions
 *****************************************************************************/

/******************************************************************************
 * Global Variables
*****************************************************************************/
PDM_PCM_CONFIG_t pdm_pcm_config;

pdm_pcm aligned_mic = NULL;

/*******************************************************************************
* Function Prototypes
*******************************************************************************/


/* Protocol callbacks */
static bool _write_payload(
    protocol_t* protocol,
    int device_id,
    int stream_id,
    int frame_count,
    int total_bytes,
    pb_ostream_t* ostream,
    void* arg);
static bool _configure_streams(protocol_t* protocol, int device, void* arg);
static void _start_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg);
static void _stop_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg);
static void _poll_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg);

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/*******************************************************************************
* Function Name: set_pdm_pcm_config
********************************************************************************
* Summary:
*   sets the PDM-PCM configuration based on user settings
*
* Parameters:
*   config: Pointer the config variable
*   stereo: boolean for enabling/disabling stereo mode
*
* Return:
*
*******************************************************************************/
void set_pdm_pcm_config(PDM_PCM_CONFIG_t *config, bool stereo) 
{
    if (true == stereo)
    {
        config->channel_config[0] = RIGHT_CH_CONFIG;
        config->channel_config[1] = LEFT_CH_CONFIG;
        config->channel_index_list[0] = RIGHT_CH_INDEX;
        config->channel_index_list[1] = LEFT_CH_INDEX;
        config->mode = MODE_STEREO;
        config->pdm_irq_cfg.intrSrc = CYBSP_PDM_CHANNEL_3_IRQ;
    }
    else 
    {
        config->channel_config[0] = RIGHT_CH_CONFIG;
        config->channel_index_list[0] = RIGHT_CH_INDEX;
        config->mode = MODE_MONO;
        config->pdm_irq_cfg.intrSrc = CYBSP_PDM_CHANNEL_3_IRQ;
    }
    
}

/*******************************************************************************
* Function Name: _configure_streams
********************************************************************************
* Summary:
*   Called when the device is configured or re-configured.
*
* Parameters:
*   protocol: Pointer the protocol handle
*   device: The device index
*   arg: Pointer the device struct (dev_pdm_pcm_t)
*
* Return:
*   True to keep the connection open, otherwise false.
*
*******************************************************************************/
static bool _configure_streams(protocol_t* protocol, int device, void* arg)
{
    pdm_pcm mic = (pdm_pcm)arg;

    int status;
    int frequency_index;

    printf("Configuring streams\r\n");

    status = protocol_get_option_oneof(
            protocol,
            device,
            MIC_OPTION_KEY_FREQUENCY,
            &frequency_index);
    if(PROTOCOL_STATUS_SUCCESS != status) {
        protocol_set_device_status(
                protocol,
                device,
                protocol_DeviceStatus_DEVICE_STATUS_ERROR,
                "Failed to get option frequency.");
        return true;
    }

    bool stereo;
    status = protocol_get_option_bool(protocol, device, MIC_OPTION_KEY_STEREO, &stereo);
    if(PROTOCOL_STATUS_SUCCESS != status) 
    {
        protocol_set_device_status(
                protocol,
                device,
                protocol_DeviceStatus_DEVICE_STATUS_ERROR,
                "Failed to get option stereo.");
        return true;
    }

    if(protocol_clear_streams(protocol, device) != PROTOCOL_STATUS_SUCCESS) 
    {
        protocol_set_device_status(
                protocol,
                device,
                protocol_DeviceStatus_DEVICE_STATUS_ERROR,
                "Failed to clear streams.");
        return true;
    }

    int max_numer_of_frames_in_chunk = pdm_pcm_get_frame_count(mic);
    int stream = protocol_add_stream(
        protocol,
        device,
        "Audio",
        protocol_StreamDirection_STREAM_DIRECTION_OUTPUT,
        protocol_DataType_DATA_TYPE_S16,
        pdm_pcm_get_frequency_from_frequency_index((SAMPLE_RATE)frequency_index),
        max_numer_of_frames_in_chunk,
        "dBFS");

    if(stream < 0) 
    {
        protocol_set_device_status(
                protocol,
                device,
                protocol_DeviceStatus_DEVICE_STATUS_ERROR,
                "Failed to add streams.");
        return true;
    }

    if (stereo) 
    {
        protocol_add_stream_rank(
                protocol,
                device,
                stream,
                "Channel",
                2,
                (const char* []) { "Left", "Right" });
    } 
    else 
    {
        protocol_add_stream_rank(
                protocol,
                device,
                stream,
                "Channel",
                1,
                (const char* []) { "Mono" });
    }

    protocol_set_device_status(
            protocol,
            device,
            protocol_DeviceStatus_DEVICE_STATUS_READY,
            "Device is ready.");

    return true;
}

/*******************************************************************************
* Function Name: _start_streams
********************************************************************************
* Summary:
*   Called when streaming is started. This may also initialize the device.
*
* Parameters:
*   protocol: Pointer the protocol handle
*   device: The device index
*   ostream: Pointer to the output stream to write to
*   arg: Pointer the device struct (dev_pdm_pcm_t)
*
*******************************************************************************/
static void _start_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg)
{
    cy_rslt_t result;
    int gain_db;
    int sample_rate;
    bool stereo;
    pdm_pcm mic = (pdm_pcm)arg;

    UNUSED(ostream);

    protocol_get_option_oneof(protocol, device, MIC_OPTION_KEY_FREQUENCY, &sample_rate);
    protocol_get_option_bool(protocol, device, MIC_OPTION_KEY_STEREO, &stereo);
    protocol_get_option_oneof(protocol, device, MIC_OPTION_KEY_GAIN, &gain_db);

    set_pdm_pcm_config(&pdm_pcm_config, stereo);
    pdm_pcm_update_config(mic, &pdm_pcm_config);

    result = pdm_pcm_init_hw((SAMPLE_RATE)sample_rate);
    if(CY_RSLT_SUCCESS != result) 
    {
        protocol_set_device_status(
                protocol,
                device,
                protocol_DeviceStatus_DEVICE_STATUS_ERROR,
                "Failed to initialize clocks.");
        return;
    }

    if (true == stereo)
    {
        pdm_pcm_set_gain(mic, pdm_pcm_config.channel_index_list[0], (cy_en_pdm_pcm_gain_sel_t)gain_db);
        pdm_pcm_set_gain(mic, pdm_pcm_config.channel_index_list[1], (cy_en_pdm_pcm_gain_sel_t)gain_db);
    }
    else 
    {
        pdm_pcm_set_gain(mic, pdm_pcm_config.channel_index_list[0], (cy_en_pdm_pcm_gain_sel_t)gain_db);
    }


    result = pdm_pcm_start(mic);
    if(CY_RSLT_SUCCESS != result) 
    {
        protocol_set_device_status(
                protocol,
                device,
                protocol_DeviceStatus_DEVICE_STATUS_ERROR,
                "Failed to initialize hardware.");
        return;
    }

    protocol_set_device_status(
            protocol,
            device,
            protocol_DeviceStatus_DEVICE_STATUS_ACTIVE,
            "Device is streaming");
}

/*******************************************************************************
* Function Name: _stop_streams
********************************************************************************
* Summary:
*  Called when streaming is stopped.
*
* Parameters:
*  protocol: Pointer the protocol handle
*  device: The device index
*  ostream: Pointer to the output stream to write to
*  arg: Pointer the device struct (dev_pdm_pcm_t)
*
*******************************************************************************/
static void _stop_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg)
{
    cy_rslt_t result;
    pdm_pcm mic = (pdm_pcm)arg;
    UNUSED(ostream);

    result = pdm_pcm_stop(mic);
    if(CY_RSLT_SUCCESS != result) 
    {
        protocol_set_device_status(
                protocol,
                device,
                protocol_DeviceStatus_DEVICE_STATUS_ERROR,
                "Failed to stop the PDM/PCM operation.");
        return;
    }

    protocol_set_device_status(
            protocol,
            device,
            protocol_DeviceStatus_DEVICE_STATUS_READY,
            "Device stopped");
}

/*******************************************************************************
* Function Name: _poll_streams
********************************************************************************
* Summary:
*  Called periodically to send data messages.
*
* Parameters:
*  protocol: Pointer the protocol handle
*  device: The device index
*  ostream: Pointer to the output stream to write to
*  arg: Pointer the device struct (dev_pdm_pcm_t)
*
*******************************************************************************/
static void _poll_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg)
{
    pdm_pcm mic = (pdm_pcm)arg;

    if(pdm_pcm_data_ready(mic)) 
    {

        uint32_t skipped_frames = 0;
        pdm_pcm_discard_samples(mic);

        int frames_in_chunk = pdm_pcm_get_frame_count(mic);

        protocol_send_data_chunk(
                protocol,
                device,
                0,
                frames_in_chunk,
                skipped_frames,
                ostream,
                _write_payload);

    }
}

/*******************************************************************************
* Function Name: _write_payload
********************************************************************************
* Summary:
*  Used by protocol_send_data_chunk to write the actual data.
*
* Parameters:
*  protocol: Pointer the protocol handle
*  device_id: The device index
*  stream_id: The stream index
*  frame_count: Number of frames to write
*  total_bytes: Total number of bytes to write (= frame_count * sizeof(type) * frame_shape.flat)
*  ostream: Pointer to the output stream to write to
*  arg: Pointer the device struct (dev_pdm_pcm_t)
*
* Return:
*   True if data writing is successful, otherwise false.
*
*******************************************************************************/
static bool _write_payload(
    protocol_t* protocol,
    int device_id,
    int stream_id,
    int frame_count,
    int total_bytes,
    pb_ostream_t* ostream,
    void* arg)
{
    UNUSED(protocol);
    UNUSED(stream_id);
    UNUSED(frame_count);
    UNUSED(protocol);

    pdm_pcm mic = (pdm_pcm)arg;

    if (!pb_write(ostream, (const pb_byte_t *)pdm_pcm_get_full_buffer(mic), total_bytes))
    {
        return false;
    }

    pdm_pcm_clear_data_ready_flag(mic);

    return true;
}


/*******************************************************************************
* Function Name: dev_pdm_pcm_register
********************************************************************************
* Summary:
*   Registers this device. This is the only exported symbol from this object.
*
* Parameters:
*   protocol: Pointer the protocol handle
*
* Returns:
*   True if registration is successful, otherwise false.
*
*******************************************************************************/
bool dev_pdm_pcm_register(protocol_t* protocol)
{
    int status;

    set_pdm_pcm_config(&pdm_pcm_config, false);
    printf("pdm: Initialized device.\r\n");
    
    aligned_mic = pdm_pcm_create(&pdm_pcm_config);
    if (NULL == aligned_mic)
    {
        return false;
    }

    device_manager_t manager = {
        .arg = aligned_mic,
        .configure_streams = _configure_streams,
        .start = _start_streams,
        .stop = _stop_streams,
        .poll = _poll_streams,
        .data_received = NULL /* has no input streams */
    };

    int device = protocol_add_device(
        protocol,
        protocol_DeviceType_DEVICE_TYPE_SENSOR,
        "Microphone",
        "PCM Microphone",
        manager);

    if(device < 0)
    {
        return false;
    }

    status = protocol_add_option_oneof(
        protocol,
        device,
        MIC_OPTION_KEY_GAIN,
        "Gain",
        "Microphone gain",
        CY_PDM_PCM_SEL_GAIN_5DB,
        pdm_pcm_get_string_list_of_gain_options(),
        pdm_pcm_get_gain_option_count());


    if (PROTOCOL_STATUS_SUCCESS != status)
    {
        return false;
    }

    status = protocol_add_option_bool(
        protocol,
        device,
        MIC_OPTION_KEY_STEREO,
        "Stereo",
        "Stereo or Mono",
        false);

    if(PROTOCOL_STATUS_SUCCESS != status)
    {
        return false;
    }

    status = protocol_add_option_oneof(
        protocol,
        device,
        MIC_OPTION_KEY_FREQUENCY,
        "Frequency",
        "Sample frequency (Hz)",
        SAMPLE_RATE_16000,
        pdm_pcm_get_string_list_of_sample_rates(),
        pdm_pcm_get_sample_rate_option_count());

    if(PROTOCOL_STATUS_SUCCESS != status)
    {
        return false;
    }

    if(!_configure_streams(protocol, device, manager.arg))
    {
        return false;
    }

    return true;
}

#endif /* IM_ENABLE_PDM_PCM */

/* [] END OF FILE */
