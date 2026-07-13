/******************************************************************************
* File Name:   dev_amic.c
*
* Description: This file implements the interface with the Analog microphone.
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

#ifdef IM_ENABLE_AMIC

#include <stdio.h>
#include <stdlib.h>
#include <cybsp.h>
#include "cy_pdl.h"
#include "mtb_hal.h"
#include "protocol/protocol.h"
#include "protocol/pb_encode.h"
#include "dev_amic.h"
#include "cy_tcpwm_counter.h"
#include "common.h"

/******************************************************************************
 * Macros
 *****************************************************************************/

#define ANALOG_FIFO_IDX                     (0U)
#define ANALOG_FIFO_BUF0_IDX                (0U)
#define ANALOG_FIFO_BUF1_IDX                (1U)

/* Define how many samples in a frame */
#define MONO_FRAME_SIZE                     (160)
#define STEREO_FRAME_SIZE                   (MONO_FRAME_SIZE * 2)

#define AMIC_Timer                          CYBSP_AMIC_TIMER_16KHZ_HW, CYBSP_AMIC_TIMER_16KHZ_NUM
#define AMIC_Timer_cfg                      &CYBSP_AMIC_TIMER_16KHZ_config

#define MIC_OPTION_KEY_STEREO               (20)

#define AMIC_STABILIZE_DELAY_MS             (100u)

#define AMIC_16KHZ_SAMPLE_RATE              (16000u)

typedef struct
{
    bool amic_initialized;
    bool stereo;
    int skipped_frames;
    int init_discard_counter;
} dev_amic_t;


dev_amic_t aligned_amic __attribute__ ((aligned (32)));
/******************************************************************************
 * Global Variables
 *****************************************************************************/
int32_t audio_buffer_left[MONO_FRAME_SIZE];
int32_t audio_buffer_right[MONO_FRAME_SIZE];

int16_t mic_audio_app_buffer_ping[STEREO_FRAME_SIZE] = {0};
int16_t mic_audio_app_buffer_pong[STEREO_FRAME_SIZE] = {0};
int16_t* ping_pong_local_pointer = NULL;

bool mic_have_data = false;
bool stereo_mode = false;
/*******************************************************************************
* Function Prototypes
*******************************************************************************/
void analog_interrupt_init(void);
static bool _init_hw(dev_amic_t *dev);

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

/******************************************************************************
* Function Name: _init_hw
********************************************************************************
* Summary:
*   Initializes the AMIC interface.
*
* Parameters:
*   none
*
* Return:
*   True if operation is successful, otherwise false.
*
*******************************************************************************/
static bool _init_hw(dev_amic_t *dev)
{
    UNUSED(dev);
    uint32_t status;

    /* Configure the 16kHz timer */
    if (CY_TCPWM_SUCCESS != Cy_TCPWM_Counter_Init(AMIC_Timer, AMIC_Timer_cfg))
    {
        return false;   
    }

    /* Initialize Autonomous Analog  */
    status = Cy_AutAnalog_Init(&autonomous_analog_init);

    if (CY_AUTANALOG_SUCCESS == status)
    {
        /* Advanced power controls:
         * Autonomous Controller power cycling: disabled (default)
         * LP Oscillator power cycling        : disabled (default)
         * SAR Core and Buffer power cycling  : enabled */
        Cy_AutAnalog_AdvPowerControl(false,false,true);

        /* Enable the required interrupts */
        analog_interrupt_init();

        /* Start the AC */
        Cy_AutAnalog_StartAutonomousControl();
    }
    /* Wait for the PRBs and the Preamps/Filters to stabilize */
    Cy_SysLib_Delay(AMIC_STABILIZE_DELAY_MS);
    
    /* Send a firmware trigger to the Autonomous Analog to start the state machine */
    Cy_AutAnalog_FwTrigger(CY_AUTANALOG_FW_TRIGGER0);

    /* Enable the 16kHz timer */
    Cy_TCPWM_Counter_Enable(AMIC_Timer);
    /* Start the 16kHz timer */
    Cy_TCPWM_TriggerStart_Single(AMIC_Timer);

    if (CY_AUTANALOG_SUCCESS == status)
    {
        dev->amic_initialized = true;
        printf("AMIC: Initialized device.\r\n");
        return true;
    }
    else
    {
        return false;
    }
}

/******************************************************************************
* Function Name: analog_fifo_isr
********************************************************************************
* Summary:
*   ISR for the Autonomous Analog FIFO. Reads audio samples into the
*   ping-pong buffer and sets the data-ready flag.
*
* Parameters:
*   none
*
* Return:
*   none
*
*******************************************************************************/
void analog_fifo_isr(void)
{
    uint32_t intr_status;
    static bool ping_pong = true;

    /* Read interrupt status register */
    intr_status = Cy_AutAnalog_FIFO_GetInterruptStatus(ANALOG_FIFO_IDX);

    /* Disable the FIFO level interrupt; re-enabled after reading the FIFO */
    Cy_AutAnalog_FIFO_ClearInterruptMask(ANALOG_FIFO_IDX, CY_AUTANALOG_INT_FIFO_LEVEL0);

    /* Clear handled interrupt */
    Cy_AutAnalog_FIFO_ClearInterrupt(ANALOG_FIFO_IDX, intr_status);

    /* Check FIFO0 interrupt */
    if ((intr_status & (uint32_t)(CY_AUTANALOG_INT_FIFO_LEVEL0)) != 0UL)
    {
        ping_pong_local_pointer = ping_pong ?
                                  mic_audio_app_buffer_ping :
                                  mic_audio_app_buffer_pong;

        if (stereo_mode)
        {
            /* Read Left & Right */
            Cy_AutAnalog_FIFO_ReadData(ANALOG_FIFO_IDX,
                                       ANALOG_FIFO_BUF0_IDX,
                                       MONO_FRAME_SIZE,
                                       audio_buffer_left);

            Cy_AutAnalog_FIFO_ReadData(ANALOG_FIFO_IDX,
                                       ANALOG_FIFO_BUF1_IDX,
                                       MONO_FRAME_SIZE,
                                       audio_buffer_right);

            /* Interleave L/R */
            for (uint16_t z = 0; z < MONO_FRAME_SIZE; z++)
            {
                *(ping_pong_local_pointer + (z * 2))     = (int16_t)(*(audio_buffer_left  + z));
                *(ping_pong_local_pointer + (z * 2) + 1) = (int16_t)(*(audio_buffer_right + z));
            }
        }
        else
        {
            /* Mono */
            Cy_AutAnalog_FIFO_ReadData(ANALOG_FIFO_IDX,
                                       ANALOG_FIFO_BUF0_IDX,
                                       MONO_FRAME_SIZE,
                                       audio_buffer_left);

            for (uint16_t z = 0; z < MONO_FRAME_SIZE; z++)
            {
                *(ping_pong_local_pointer + z) = (int16_t)(*(audio_buffer_left + z));
            }
        }

        ping_pong = !ping_pong;
        mic_have_data = true;

        /* Re-enable the FIFO0 level interrupt */
        Cy_AutAnalog_FIFO_SetInterruptMask(ANALOG_FIFO_IDX, CY_AUTANALOG_INT_FIFO_LEVEL0);
    }
}

/******************************************************************************
* Function Name: analog_interrupt_init
********************************************************************************
* Summary:
*   Configures and enables the Autonomous Analog FIFO interrupt.
*
* Parameters:
*   none
*
* Return:
*   none
*
******************************************************************************/
void analog_interrupt_init(void)
{
    /* Variable to capture return value of functions */
    cy_en_sysint_status_t sysIntStatus;

    /* Configuration of Autonomous Analog FIFO Interrupt */
    const cy_stc_sysint_t AUT_ANALOG_FIFO_IRQ_cfg =
    {
        .intrSrc      = pass_interrupt_fifo_IRQn,
        .intrPriority = 2U,
    };
    /* Register FIFO interrupt service routine */
    sysIntStatus = Cy_SysInt_Init(&AUT_ANALOG_FIFO_IRQ_cfg, &analog_fifo_isr);
    if (sysIntStatus == CY_SYSINT_SUCCESS)
    {
        /* Enable interrupt in NVIC */
        NVIC_ClearPendingIRQ(pass_interrupt_fifo_IRQn);
        NVIC_EnableIRQ(pass_interrupt_fifo_IRQn);
    }
    else
    {
        CY_ASSERT(0);
    }

    /* Enable the FIFO0 level interrupt */
    Cy_AutAnalog_FIFO_SetInterruptMask(ANALOG_FIFO_IDX,CY_AUTANALOG_INT_FIFO_LEVEL0);

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
*   arg: Pointer the device struct (dev_amic_t)
*
* Return:
*   True to keep the connection open, otherwise false.
*
*******************************************************************************/
static bool _configure_streams(protocol_t* protocol, int device, void* arg)
{
    UNUSED(arg);

    int status;
    int frequency = AMIC_16KHZ_SAMPLE_RATE;
    bool stereo;

    status = protocol_get_option_bool(protocol, device, MIC_OPTION_KEY_STEREO, &stereo);

    if(status != PROTOCOL_STATUS_SUCCESS) 
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

    int stream = protocol_add_stream(
        protocol,
        device,
        "Audio",
        protocol_StreamDirection_STREAM_DIRECTION_OUTPUT,
        protocol_DataType_DATA_TYPE_S16,
        frequency,
        MONO_FRAME_SIZE,
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
*   arg: Pointer the device struct (dev_amic_t)
*
*******************************************************************************/
static void _start_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg)
{
    dev_amic_t* mic = (dev_amic_t*)arg;

    UNUSED(ostream);

    protocol_get_option_bool(protocol, device, MIC_OPTION_KEY_STEREO, &mic->stereo);
    stereo_mode = mic->stereo;
    mic->init_discard_counter = 4;

    /* Enable and start the 16kHz timer */
    Cy_TCPWM_Counter_Enable(AMIC_Timer);
    Cy_TCPWM_TriggerStart_Single(AMIC_Timer);

    /* Enable the FIFO interrupt */
    Cy_AutAnalog_FIFO_SetInterruptMask(ANALOG_FIFO_IDX, CY_AUTANALOG_INT_FIFO_LEVEL0);

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
*  arg: Pointer the device struct (dev_amic_t)
*
*******************************************************************************/
static void _stop_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg)
{
    dev_amic_t* mic = (dev_amic_t*)arg;
    UNUSED(mic);
    UNUSED(ostream);

    /* Disable the FIFO interrupt and clear any pending */
    Cy_AutAnalog_FIFO_ClearInterruptMask(ANALOG_FIFO_IDX, CY_AUTANALOG_INT_FIFO_LEVEL0);
    Cy_AutAnalog_FIFO_ClearInterrupt(ANALOG_FIFO_IDX, CY_AUTANALOG_INT_FIFO_LEVEL0);

    /* Stop and disable the 16kHz timer */
    Cy_TCPWM_Counter_Disable(AMIC_Timer);

    /* Clear the data flag */
    mic_have_data = false;

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
*  arg: Pointer the device struct (dev_amic_t)
*
*******************************************************************************/
static void _poll_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg)
{
    dev_amic_t* mic = (dev_amic_t*)arg;

    if (mic_have_data)
    {
        if (mic->init_discard_counter > 0)
        {
            mic->init_discard_counter--;
            memset(ping_pong_local_pointer, 0, MONO_FRAME_SIZE * sizeof(int16_t));
        }

        protocol_send_data_chunk(
                protocol,
                device,
                0,
                MONO_FRAME_SIZE,
                mic->skipped_frames,
                ostream,
                _write_payload);

        mic->skipped_frames = 0;
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
*  arg: Pointer the device struct (dev_amic_t)
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
    UNUSED(device_id);
    UNUSED(stream_id);
    UNUSED(frame_count);
    UNUSED(arg);

    if (!pb_write(ostream, (const pb_byte_t *)ping_pong_local_pointer, total_bytes))
    {
        return false;
    }

    mic_have_data = false;

    return true;
}

/*******************************************************************************
* Function Name: dev_amic_register
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
bool dev_amic_register(protocol_t* protocol)
{
    int status;
    dev_amic_t* mic = &aligned_amic;

    memset(mic, 0, sizeof(dev_amic_t));

    if (!_init_hw(mic))
    {
        return false;
    }

    device_manager_t manager = {
        .arg = mic,
        .configure_streams = _configure_streams,
        .start = _start_streams,
        .stop = _stop_streams,
        .poll = _poll_streams,
        .data_received = NULL /* has no input streams */
    };

    int device = protocol_add_device(
        protocol,
        protocol_DeviceType_DEVICE_TYPE_SENSOR,
        "Analog Microphone",
        "Analog Microphone",
        manager);

    if(device < 0)
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

    if(status != PROTOCOL_STATUS_SUCCESS)
    {
        return false;
    }

    if(!_configure_streams(protocol, device, manager.arg))
    {
        return false;
    }

    return true;
}

#endif /* IM_ENABLE_AMIC */

/* [] END OF FILE */
