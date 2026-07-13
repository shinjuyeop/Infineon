/******************************************************************************
* File Name: dev_bmm350.c
*
* Description: This file implements the interface with the magnetometer sensor.
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

#ifdef IM_ENABLE_BMM350

#include <stdio.h>
#include <cybsp.h>
#include "cy_pdl.h"
#include "protocol/protocol.h"
#include "protocol/pb_encode.h"
#include "clock.h"
#include "dev_bmm350.h"
#include "mtb_bmm350.h"
#include "common.h"

/*******************************************************************************
* Macros
*******************************************************************************/

#define MAG_OPTION_KEY_FREQUENCY 1

/* At 400Hz/8 = 50 Hz chunk frequency */
/* Increase this to 16 if 800Hz mode is enabled */
#define MAX_FRAMES_IN_CHUNK   (8)

/* X Y Z */
#define AXIS_COUNT            (3)


/*******************************************************************************
* Types
*******************************************************************************/
typedef struct
{

    /* Tick of last sample */
    clock_tick_t sample_time_tick;

    /* Flag indicating if it the first sample */
    bool first_sample;

    /* The sample period in ticks */
    uint32_t period_tick;

    /* Converted data */
    float data[AXIS_COUNT * MAX_FRAMES_IN_CHUNK];

    /* Number of frames collected in accel_data and gyro_data. */
    /* Cleared after each sent data-chunk. Equal or less than frames_in_chunk */
    int frames_sampled;

    /* Max number of frames in each chunk. Is less or equal to MAX_FRAMES_IN_CHUNK*/
    int frames_target;

    /* Number of frames dropped. This is reset each data-chunk. */
    int frames_dropped;
} dev_bmm350_t;

/*******************************************************************************
 * Global Variables
 ******************************************************************************/

cy_stc_i3c_device_t i3c_device_address;

/* Hardware device */
mtb_bmm350_t magnetometer;

/*******************************************************************************
* Function Prototypes
*******************************************************************************/

static bool _init_hw(dev_bmm350_t *dev, I3C_CORE_Type* i3c_hw, cy_stc_i3c_context_t* i3c);

static bool _config_hw(dev_bmm350_t *dev, int rate);

static bool _read_hw(dev_bmm350_t* dev);

static bool _configure_streams(protocol_t* protocol, int device, void* arg);

static void _start_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg);

static void _stop_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg);

static bool _write_payload(protocol_t* protocol, int device_id, int stream_id, int frame_count, int total_bytes, pb_ostream_t* ostream, void* arg);

static void _poll_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg);

void mag_remap_sensor_orientation(struct bmm350_mag_temp_data *data);
/*******************************************************************************
* Function Definitions
*******************************************************************************/


/******************************************************************************
* Function Name: _init_hw
********************************************************************************
* Summary:
*   Initializes the bmm350.
*
* Parameters:
*   dev: Pointer to the dev_bmm350_t device handle.
*   i2c: Pointer to the I2C interface resource.
*
* Return:
*   True if operation is successful, otherwise false.
*
*******************************************************************************/
static bool _init_hw(dev_bmm350_t *dev, I3C_CORE_Type* i3c_hw, cy_stc_i3c_context_t* i3c_context)
{
    cy_rslt_t result;
    i3c_device_address.staticAddress = MTB_BMM350_ADDRESS_SEC;

    /* Initializes the sensor. */
    result = mtb_bmm350_init_i3c(&magnetometer, i3c_hw,
                                 i3c_context, &i3c_device_address);
    if(CY_RSLT_SUCCESS == result)
    {
        printf("bmm350: Initialized device.\r\n");
        return true;
    }
    else
    {
        return false;
    }
}

/******************************************************************************
* Function Name: _config_hw
********************************************************************************
* Summary:
*   Configures the bmm350 output data rate and measurement range.
*
* Parameters:
*   dev: Pointer to the dev_bmm350_t device handle.
*   rate: Sample frequency (Hz).
*
* Return:
*   True if operation is successful, otherwise false.
*
*******************************************************************************/
static bool _config_hw(dev_bmm350_t *dev, int rate)
{
    cy_rslt_t result;

    dev->sample_time_tick = 0;
    dev->period_tick = CLOCK_TICK_PER_SECOND / rate;
    dev->frames_dropped = 0;
    dev->frames_sampled = 0;

    switch(rate)
    {
        case 50:
            /* Set ODR and performance */
            result = mtb_bmm350_set_odr_performance(BMM350_DATA_RATE_50HZ, BMM350_NO_AVERAGING, &(magnetometer));
            dev->frames_target = 1;
            break;
        case 100:
            /* Set ODR and performance */
            result = mtb_bmm350_set_odr_performance(BMM350_DATA_RATE_100HZ, BMM350_NO_AVERAGING, &(magnetometer));
            dev->frames_target = 2;
            break;
        case 200:
            result = mtb_bmm350_set_odr_performance(BMM350_DATA_RATE_200HZ, BMM350_NO_AVERAGING, &(magnetometer));
            dev->frames_target = 4;
            break;
        case 400:
            result = mtb_bmm350_set_odr_performance(BMM350_DATA_RATE_400HZ, BMM350_NO_AVERAGING, &(magnetometer));
            dev->frames_target = 8;
            break;
        default:
        return false;
    }

    if (CY_RSLT_SUCCESS != result)
    {
        return false;
    }

    printf("bmm 350: Configured device. rate=%d Hz, frames/chunk=%d\r\n", rate, dev->frames_target);

    return true;
}

/******************************************************************************
* Function Name: _read_hw
********************************************************************************
* Summary:
*   Reads the current bmm350 data and convert it.
*
* Parameters:
*   dev: Pointer to the dev_bmm350_t device handle.
*
* Return:
*   True if data retrieval is successful, otherwise false.
*
*******************************************************************************/
static bool _read_hw(dev_bmm350_t* dev)
{
    cy_rslt_t result;
    mtb_bmm350_data_t bmm350_data;
  
    result = mtb_bmm350_read(&magnetometer, &bmm350_data);
    if(CY_RSLT_SUCCESS != result)
    {
       return false;
    }

    float *dest = dev->data + dev->frames_sampled * AXIS_COUNT;
    
#ifdef USE_SENSOR_REMAPPING    
     /* Remapping the data based on orientation of sensor */ 
     mag_remap_sensor_orientation(&(bmm350_data.sensor_data));
#endif
             
    *dest++ = bmm350_data.sensor_data.y;
    *dest++ = bmm350_data.sensor_data.x;
    *dest++ = bmm350_data.sensor_data.z;

    dev->frames_sampled++;
    return true;
}

/*******************************************************************************
* Function Name: mag_remap_sensor_orientation
********************************************************************************
* Summary:
* Remapping the sensor data to match with CY8CKIT-062S2-AI kit orientation.
* 
* Note: Ensure that the 'USE_KIT_PSE84_AI', 'USE_KIT_PSE84_EVAL_EPC2' and 
* 'USE_KIT_PSE84_EVAL_EPC4' are defined correctly based on the selected target.
* See common.mk file for details.
*
* Parameters:
* data: Pointer to BMM350 Sensors data
*
* Return:
* None
*
*******************************************************************************/
void mag_remap_sensor_orientation(struct bmm350_mag_temp_data *data)
{
    /* Remapping the data based on orientation of sensor */   
    float temp_x = data->x;
    float temp_y = data->y;
    #ifdef USE_KIT_PSE84_AI
    data->x =  temp_y;
    data->y = -temp_x;
    #elif defined(USE_KIT_PSE84_EVAL_EPC2) || defined(USE_KIT_PSE84_EVAL_EPC4)
    data->x = -temp_y;
    data->y =  temp_x;
    #endif   
}

/*******************************************************************************
* Function Name: _configure_streams
********************************************************************************
* Summary:
*   Configures the device streams based on the selected options.
*
* Parameters:
*   protocol: Pointer to the protocol handle.
*   device: The device index.
*   arg: Pointer to the device structure (dev_bmm350_t).
*
* Return:
*   True to keep the connection open, false to close.
*
*******************************************************************************/
static bool _configure_streams(protocol_t* protocol, int device, void* arg)
{
    dev_bmm350_t* dev = (dev_bmm350_t*)arg;
    int status;

    int frequency_index;
    int rate;
    status = protocol_get_option_oneof(protocol, device, MAG_OPTION_KEY_FREQUENCY, &frequency_index);
    if(PROTOCOL_STATUS_SUCCESS != status)
    {
        protocol_set_device_status(
                protocol,
                device,
                protocol_DeviceStatus_DEVICE_STATUS_ERROR,
                "Failed to get option frequency.");
        return true;
    }

    switch(frequency_index)
    {
       case 0: rate = 50; break;
       case 1: rate = 100; break;
       case 2: rate = 200; break;
       case 3: rate = 400; break;
       default: return false;
    }

    /* Clear any existing streams */
    if(protocol_clear_streams(protocol, device) != PROTOCOL_STATUS_SUCCESS)
    {
        protocol_set_device_status(
                protocol,
                device,
                protocol_DeviceStatus_DEVICE_STATUS_ERROR,
                "Failed to clear streams.");
        return true;
    }

    /* Add a stream based on current configuration */
    int stream = protocol_add_stream(
        protocol,
        device,
        "Mag",
        protocol_StreamDirection_STREAM_DIRECTION_OUTPUT,
        protocol_DataType_DATA_TYPE_F32,
        rate,
        1,
        "µT");

    if(stream < 0)
    {
        protocol_set_device_status(
                protocol,
                device,
                protocol_DeviceStatus_DEVICE_STATUS_ERROR,
                "Failed to add streams.");
        return true;
    }

    protocol_add_stream_rank(
            protocol,
            device,
            stream,
            "Axis",
            3,
            (const char* []) { "X", "Y", "Z" });

   if(!_config_hw(dev, rate))
   {
       protocol_set_device_status(
                 protocol,
                 device,
                 protocol_DeviceStatus_DEVICE_STATUS_ERROR,
                 "Failed to configure device.");
       return true;
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
*  Called when streaming is started. This may also initialize the device.
*
* Parameters:
*  protocol: Pointer to the protocol handle.
*  device: The device index.
*  ostream: Pointer to the output stream to write to
*  arg: Pointer to the device structure (dev_bmm350_t).
*
*******************************************************************************/
static void _start_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg)
{
    UNUSED(ostream);
    dev_bmm350_t* dev = (dev_bmm350_t*)arg;
    protocol_set_device_status(
            protocol,
            device,
            protocol_DeviceStatus_DEVICE_STATUS_ACTIVE,
            "Device is streaming");

    dev->first_sample = true;
    dev->sample_time_tick = clock_get_tick();
}

/*******************************************************************************
* Function Name: _stop_streams
********************************************************************************
* Summary:
*  Called when streaming is stopped.
*
* Parameters:
*  protocol: Pointer to the protocol handle.
*  device: The device index.
*  ostream: Pointer to the output stream to write to
*  arg: Pointer to the device structure (dev_bmm350_t).
*
*******************************************************************************/
static void _stop_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg)
{
    dev_bmm350_t* dev = (dev_bmm350_t*)arg;

    UNUSED(dev);
    UNUSED(ostream);

    protocol_set_device_status(
            protocol,
            device,
            protocol_DeviceStatus_DEVICE_STATUS_READY,
            "Device stopped");
}

/*******************************************************************************
* Function Name: _write_payload
********************************************************************************
* Summary:
*  Used by protocol_send_data_chunk to write the actual data.
*
* Parameters:
*  protocol: Pointer to the protocol handle.
*  device_id: The device index.
*  stream_id: The stream index.
*  frame_count: Number of frames to write.
*  total_bytes: Total number of bytes to write (= frame_count * sizeof(type) * frame_shape.flat).
*  ostream: Pointer to the output stream to write to.
*  arg: Pointer to the device structure (dev_bmm350_t).
*
* Return:
*  True if the payload was written successfully, else false.
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
    UNUSED(protocol);

    dev_bmm350_t* dev = (dev_bmm350_t*)arg;

    if (!pb_write(ostream, (const pb_byte_t *)dev->data, total_bytes))
    {
        return false;
    }

    return true;
}


/*******************************************************************************
* Function Name: _poll_streams
********************************************************************************
* Summary:
*  Called periodically to send data messages.
*
* Parameters:
*  protocol: Pointer to the protocol handle.
*  device: The device index.
*  ostream: Pointer to the output stream to write to.
*  arg: Pointer to the device structure (dev_bmm350_t).
*
*******************************************************************************/
static void _poll_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg)
{
    dev_bmm350_t* dev = (dev_bmm350_t*)arg;
    clock_tick_t current_time = clock_get_tick();

    /* Reinterpret this timing as the time we wish the sample to happen at. */
    /*  It is initialized to the current clock when starting the stream. */
    clock_tick_t current_treshold = dev->sample_time_tick;

    /* Calculate how late we are.*/
    clock_tick_t total_drift = current_time - current_treshold;

    /* If we are to late we skip this frame and save time.*/
    /* Previous data package will be resent.*/
    bool late = false;
    uint32_t drift_ms = (uint32_t)total_drift;
    if (drift_ms > 200)
    {
        /* Tens of microseconds. 200 is equal to 2 ms */
        late=1;
    }

    if(current_time >= current_treshold )
    {

        /* If we are late.. Skip the reading of the sensor.. */
        if ( late && !dev->first_sample)
        {
            /* Check the total drift here..*/
            late = false;
        }
        else
        {
            /* Ok, we are on time so try reading the sensor. */
            if(!_read_hw(dev))
            {
                return; /* If it fails it will retry until we will be late and skipping this frame.*/
            }
            dev->first_sample = false;
        }

        /* This is updated whenever there is an _read_hw call! However since the _read_hw might fail */
        /* and when we are resending the previous data we have to force this to 1 anyway. */
        dev->frames_sampled = 1;

        /* Since we in reality dont drop any frames anymore. */
        dev->frames_dropped = 0;

        /* When we should do the next frame. */
        dev->sample_time_tick += dev->period_tick;

        {
            protocol_send_data_chunk(protocol, device, 0, dev->frames_sampled, dev->frames_dropped, ostream, _write_payload);
            dev->frames_dropped = 0;
            dev->frames_sampled = 0;
        }
    }
}

/*******************************************************************************
* Function Name: dev_bmm350_register
********************************************************************************
* Summary:
*  Registers this device. This is the only exported symbol from this file.
*
* Parameters:
*  protocol: Pointer to the protocol handle.
*  i3c: Pointer to the I3C context.
*
*
* Returns:
*  True on success, else false.
*
*******************************************************************************/
bool dev_bmm350_register(protocol_t* protocol, I3C_CORE_Type* i3c_hw, cy_stc_i3c_context_t* i3c_context)
{
    int status;

    dev_bmm350_t* dev = (dev_bmm350_t*)malloc(sizeof(dev_bmm350_t));
    if(dev == NULL)
    {
        return false;
    }

    memset(dev, 0, sizeof(dev_bmm350_t));

    if(!_init_hw(dev,i3c_hw ,i3c_context))
    {
        free(dev);
        return false;
    }

    device_manager_t manager = {
        .arg = dev,
        .configure_streams = _configure_streams,
        .start = _start_streams,
        .stop = _stop_streams,
        .poll = _poll_streams,
        .data_received = NULL // has no input streams
    };

    int device = protocol_add_device(
        protocol,
        protocol_DeviceType_DEVICE_TYPE_SENSOR,
        "Magnetometer",
        "Magnetometer (BMM350)",
        manager);

    if(device < 0)
    {
        return false;
    }

    status = protocol_add_option_oneof(
            protocol,
            device,
            MAG_OPTION_KEY_FREQUENCY,
            "Frequency",
            "Sample frequency (Hz)",
            0,
            (const char* []) { "50 Hz", "100 Hz", "200 Hz", "400 Hz" },
            4);

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

#endif /* IM_ENABLE_BMM350 */

/* [] END OF FILE */
