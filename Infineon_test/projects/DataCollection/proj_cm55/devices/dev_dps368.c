/******************************************************************************
* File Name:   dev_dps368.c
*
* Description:
*   This file implements the interface with the temperature/pressure sensor.
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

#ifdef IM_ENABLE_DPS368

#include <cybsp.h>
#include <stdio.h>
#include "protocol/protocol.h"
#include "protocol/pb_encode.h"
#include "clock.h"
#include "dev_dps368.h"
#include <mtb_xensiv_dps3xx.h>
#include "common.h"

/*******************************************************************************
* Macros
*******************************************************************************/
#define DPS_OPTION_KEY_FREQUENCY    (1)
#define DPS_OPTION_KEY_STREAM_MODE  (2)

#define MAX_FRAMES_IN_CHUNK   (4)

/* Pressure, Temerature */
#define SENSOR_COUNT          (2)

#ifdef IM_XSS_DPS368
#define DPS368_ADDRESS (XENSIV_DPS3XX_I2C_ADDR_ALT)
#else
#define DPS368_ADDRESS (XENSIV_DPS3XX_I2C_ADDR_DEFAULT)
#endif

#define XENSIV_DPS3XX_PSR_TMP_READ_REG_ADDR         (0x00)
#define XENSIV_DPS3XX_PSR_TMP_READ_LEN              (0x06)
#define XENSIV_DPS3XX_CFG_REG_ADDR                  (0x09)
#define XENSIV_DPS3XX_CFG_TMP_SHIFT_EN_SET_VAL      (0x08)
#define XENSIV_DPS3XX_CFG_PRS_SHIFT_EN_SET_VAL      (0x04)
/*******************************************************************************
* Types
*******************************************************************************/

typedef enum {
    /* Both pressure and temprature are in the same stream.
     * The shape of each frame will be [2,1] as
     * Stream 0: {{Pressure}, {TEMPERATURE}} */
    DPS_MODE_STREAM_COMBINED = 0,

    /* The pressure and temprature are split in two separate streams
     * each with the shape [1] as
     * Stream 0: {Pressure}
     * Stream 1: {TEMPERATURE}
     */
    DPS_MODE_STREAM_SPLIT = 3,

    /* Only the pressure is enabled.
     * Stream 0: {Pressure}
     */
    DPS_MODE_STREAM_ONLY_PRESSURE = 1,

    /* Only the temprature is enabled.
     * Stream 0: {TEMPERATURE}
     */
    DPS_MODE_STREAM_ONLY_TEMP = 2,
} stream_mode_t;


typedef struct {

    xensiv_dps3xx_t dps3xx;
    mtb_hal_i2c_t* i2c;
    /* Tick of last sample */
    clock_tick_t sample_time_tick;

    /* The sample period in ticks */
    uint32_t period_tick;

    union {
        struct {
            /*When mode is DPS_MODE_STREAM_SPLIT or DPS_MODE_STREAM_ONLY_XXX*/
            /* Converted data as hPa */
            float pressure_data[MAX_FRAMES_IN_CHUNK];

            /* Converted data as degrees */
            float temperature_data[MAX_FRAMES_IN_CHUNK];
        };

        /*When mode is DPS_MODE_STREAM_COMBINED*/
        float data_combined[MAX_FRAMES_IN_CHUNK * SENSOR_COUNT];
    };

    /* Number of frames collected in Pressure_data and temperature_data. */
    /* Cleared after each sent data-chunk. Equal or less than frames_in_chunk */
    int frames_sampled;

    /* Max number of frames in each chunk. Is less or equal to MAX_FRAMES_IN_CHUNK*/
    int frames_target;

    /* Number of frames dropped. This is cleared each data-chunk. */
    int frames_dropped;

    /* How streams are presented. */
    stream_mode_t stream_mode;

} dev_dps368_t;

/*******************************************************************************
 * Global Variables
 ******************************************************************************/


/*******************************************************************************
* Function Prototypes
*******************************************************************************/
static bool _init_hw(dev_dps368_t* dps, mtb_hal_i2c_t* i2c);

static bool _config_hw(dev_dps368_t* dps, int rate, stream_mode_t mode, mtb_hal_i2c_t* i2c);

static bool _read_hw(dev_dps368_t* dps, mtb_hal_i2c_t* i2c);

static bool _configure_streams(protocol_t* protocol, int device, void* arg);

static void _start_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg);

static void _stop_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg);

static void _poll_streams(
        protocol_t* protocol,
        int device,
        pb_ostream_t* ostream,
        void* arg);

static bool _write_payload(
    protocol_t* protocol,
    int device_id,
    int stream_id,
    int frame_count,
    int total_bytes,
    pb_ostream_t* ostream,
    void* arg);

static bool disable_shift( dev_dps368_t* dps );
static bool enable_shift( dev_dps368_t* dps );

/*******************************************************************************
* Function Definitions
*******************************************************************************/
/*******************************************************************************
* Function Name: enable_shift
********************************************************************************
* Summary:
* enables the shift to avoid register overflow when using high oversampling.
*
* Parameters:

* dps: Pointer to the device struct (dev_dps368_t)
* 
* Return: True if shift enable is successful, otherwise false.
*
*******************************************************************************/
static bool enable_shift(dev_dps368_t* dps)
{
    cy_rslt_t result;
    uint8_t reg_val;
    result = mtb_xensiv_dps3xx_reg_read(&dps->dps3xx, XENSIV_DPS3XX_CFG_REG_ADDR, &reg_val, 1);

    if (CY_RSLT_SUCCESS == result)
    {
        reg_val |= XENSIV_DPS3XX_CFG_TMP_SHIFT_EN_SET_VAL | XENSIV_DPS3XX_CFG_PRS_SHIFT_EN_SET_VAL;
        result = mtb_xensiv_dps3xx_reg_write(&dps->dps3xx, XENSIV_DPS3XX_CFG_REG_ADDR, &reg_val, 1);
        return true;
    }
    else
    {
        return false;
    }

}

/*******************************************************************************
* Function Name: disable_shift
********************************************************************************
* Summary:
* disable shift for highest resoloultion
*
* Parameters:
* dps: Pointer to the device struct (dev_dps368_t)
* 
* Return: True if shift disable is successful, otherwise false.
*
*******************************************************************************/
static bool disable_shift(dev_dps368_t* dps)
{
    cy_rslt_t result;
    uint8_t reg_val;
    result = mtb_xensiv_dps3xx_reg_read(&dps->dps3xx, XENSIV_DPS3XX_CFG_REG_ADDR, &reg_val, 1);

    if (CY_RSLT_SUCCESS == result)
    {
        reg_val &= ~(XENSIV_DPS3XX_CFG_TMP_SHIFT_EN_SET_VAL | XENSIV_DPS3XX_CFG_PRS_SHIFT_EN_SET_VAL);
        result = mtb_xensiv_dps3xx_reg_write(&dps->dps3xx, XENSIV_DPS3XX_CFG_REG_ADDR, &reg_val, 1);
        return true;
    }
    else
    {
        return false;
    }

}

/******************************************************************************
* Function Name: _init_hw
********************************************************************************
* Summary:
*    A function used to initialize the DPS368 Pressure sensor.
*
* Parameters:
*  dps: Pointer to the device struct (dev_dps368_t)
*  i2c: Pointer to the I2C instance
*
* Return:
*   True if initialization is successful, otherwise false.
*
*******************************************************************************/
static bool _init_hw(dev_dps368_t* dps, mtb_hal_i2c_t* i2c)
{
    cy_rslt_t result;
    dps->i2c=i2c;
    dps->stream_mode = DPS_MODE_STREAM_COMBINED;
    
    /* Initialize pressure sensor */
    result = mtb_xensiv_dps3xx_init_i2c(&dps->dps3xx, i2c, DPS368_ADDRESS);
    if(CY_RSLT_SUCCESS == result)
    {
        printf("dps368: Initialized device.\r\n");
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
*    A function used to configure the DPS368 Pressure sensor.
*
* Parameters:
*   dps: Pointer to the device struct (dev_dps368_t)
*   rate: Sampling rate
*
* Return:
*   True if configuration is successful, otherwise false.
*
*******************************************************************************/
static bool _config_hw(dev_dps368_t* dps, int rate, stream_mode_t mode, mtb_hal_i2c_t* i2c)
{
    dps->stream_mode = mode;
    cy_rslt_t result;
    xensiv_dps3xx_config_t config;

    float pressure = 0;
    float temperature = 0;

    /* Gets the current configuration parameters. */
    result = xensiv_dps3xx_get_config(&dps->dps3xx, &config);
    if (result != CY_RSLT_SUCCESS)
        return false;

    config.fifo_enable = false;
    config.dev_mode = XENSIV_DPS3XX_MODE_BACKGROUND_ALL;

    switch(rate)
    {
        case 8: /*This will take approx 2*220 = 440 ms*/
            enable_shift(dps);
            config.pressure_oversample = XENSIV_DPS3XX_OVERSAMPLE_16;
            config.temperature_oversample = XENSIV_DPS3XX_OVERSAMPLE_16;
            config.pressure_rate = XENSIV_DPS3XX_RATE_8;
            config.temperature_rate = XENSIV_DPS3XX_RATE_8;
            break;

        case 16: /*This will take approx 2*236 = 472 ms*/
            disable_shift(dps);
            config.pressure_oversample = XENSIV_DPS3XX_OVERSAMPLE_8;
            config.temperature_oversample = XENSIV_DPS3XX_OVERSAMPLE_8;
            config.pressure_rate = XENSIV_DPS3XX_RATE_16;
            config.temperature_rate = XENSIV_DPS3XX_RATE_16;
            break;

        case 32: /*This will take approx 2*268 = 536 ms*/
            disable_shift(dps);
            config.pressure_oversample = XENSIV_DPS3XX_OVERSAMPLE_4;
            config.temperature_oversample = XENSIV_DPS3XX_OVERSAMPLE_4;
            config.pressure_rate = XENSIV_DPS3XX_RATE_32;
            config.temperature_rate = XENSIV_DPS3XX_RATE_32;
            break;

        case 64: /*This will take approx 2*332 = 664 ms*/
            disable_shift(dps);
            config.pressure_oversample = XENSIV_DPS3XX_OVERSAMPLE_2;
            config.temperature_oversample = XENSIV_DPS3XX_OVERSAMPLE_2;
            config.pressure_rate = XENSIV_DPS3XX_RATE_64;
            config.temperature_rate = XENSIV_DPS3XX_RATE_64;
            break;

        case 128: /*This will take approx 2*460 = 920 ms*/
            disable_shift(dps);
            config.pressure_oversample = XENSIV_DPS3XX_OVERSAMPLE_1;
            config.temperature_oversample = XENSIV_DPS3XX_OVERSAMPLE_1;
            config.pressure_rate = XENSIV_DPS3XX_RATE_128;
            config.temperature_rate = XENSIV_DPS3XX_RATE_128;
            break;

        default:
            return false;
    }

    /* Update sample period */
    dps->sample_time_tick = 0;
    dps->period_tick = CLOCK_TICK_PER_SECOND / rate;

    result = xensiv_dps3xx_set_config(&dps->dps3xx, &config);
    if (result != CY_RSLT_SUCCESS)
    {
        printf("xensiv_dps3xx_set_config FAILED\r\n");
        return false;
    }

    /* Poll and read out data to clear it from some junk.
    * Perhaps the sensor isn't ready?
    */
    xensiv_dps3xx_read(&dps->dps3xx, &pressure, &temperature);
    xensiv_dps3xx_read(&dps->dps3xx, &pressure, &temperature);
    xensiv_dps3xx_read(&dps->dps3xx, &pressure, &temperature);

    printf("dps368: Configured device. rate=%d Hz, mode=%i\r\n", rate, dps->stream_mode);

    return true;
}

/******************************************************************************
* Function Name: _read_hw
********************************************************************************
* Summary:
*   Reads the current DPS368 data and convert it.
*
* Parameters:
*   dps: Pointer to the dps368 device handle.
*
* Return:
*   True if data retrieval is successful, otherwise false.
*
*******************************************************************************/
static bool _read_hw(dev_dps368_t* dps, mtb_hal_i2c_t* i2c)
{
    float pressure = 0;
    float temperature = 0;
    
    uint8_t read_reg_addr = XENSIV_DPS3XX_PSR_TMP_READ_REG_ADDR;
    uint8_t read_reg_len = XENSIV_DPS3XX_PSR_TMP_READ_LEN;
    uint8_t raw_value[XENSIV_DPS3XX_PSR_TMP_READ_LEN];

    mtb_xensiv_dps3xx_reg_read(&dps->dps3xx, read_reg_addr, raw_value, read_reg_len);

    int32_t press_raw = (int32_t)(raw_value[2]) + (raw_value[1] << 8) + (raw_value[0] << 16);
    int32_t temp_raw = (int32_t)(raw_value[5]) + (raw_value[4] << 8) + (raw_value[3] << 16);
    pressure = mtb_xensiv_dps3xx_calc_pressure(&dps->dps3xx, press_raw);
    temperature = mtb_xensiv_dps3xx_calc_temperature(&dps->dps3xx, temp_raw);

    if(dps->stream_mode == DPS_MODE_STREAM_COMBINED)
    {
        float *dest = dps->data_combined + dps->frames_sampled * SENSOR_COUNT;
        *dest++ = pressure;
        *dest++ = temperature;
    }

    if(dps->stream_mode == DPS_MODE_STREAM_SPLIT || dps->stream_mode == DPS_MODE_STREAM_ONLY_PRESSURE)
    {
        float *dest = dps->pressure_data + dps->frames_sampled ;
        *dest++ = pressure;
    }

    if(dps->stream_mode == DPS_MODE_STREAM_SPLIT || dps->stream_mode == DPS_MODE_STREAM_ONLY_TEMP)
    {
         float *dest = dps->temperature_data + dps->frames_sampled ;
        *dest++ = temperature;
    }

    dps->frames_sampled++;

    return true;
}

/******************************************************************************
* Function Name: _configure_streams
********************************************************************************
* Summary:
*  Called when the device stream is configured or re-configured.
*
* Parameters:
*  protocol: pointer the protocol handle
*  device: the device index
*  arg: pointer the device struct (dev_dps368_t)
*
*  Return:
*    True to keep the connection open, false to close.

*******************************************************************************/
static bool _configure_streams(protocol_t* protocol, int device, void* arg)
{
    int result;
    int rate_index;
    int rate;
    int status;
    int mode_index;
    stream_mode_t mode;
    const char *stream0_unit;
    const char *stream1_unit;
    const char *stream0_name;
    const char *stream1_name;
    int stream0;
    int stream1;

    status = protocol_get_option_oneof(protocol, device, DPS_OPTION_KEY_FREQUENCY, &rate_index);

    if(status != PROTOCOL_STATUS_SUCCESS)
    {
        protocol_set_device_status(
                protocol,
                device,
                protocol_DeviceStatus_DEVICE_STATUS_ERROR,
                "Failed to get option frequency.");
        return true;
    }

    switch(rate_index)
    {
        case 0: rate = 8; break;
        case 1: rate = 16; break;
        case 2: rate = 32; break;
        case 3: rate = 64; break;
        case 4: rate = 128; break;
        default: return false;
    }

    if(protocol_get_option_oneof(protocol, device, DPS_OPTION_KEY_STREAM_MODE, &mode_index) != PROTOCOL_STATUS_SUCCESS)
    {
        protocol_set_device_status(
            protocol,
            device,
            protocol_DeviceStatus_DEVICE_STATUS_ERROR,
            "Failed to get option stream mode.");
        return true;
    }

    /* Clear any existing streams */
    if(protocol_clear_streams(protocol, device) != PROTOCOL_STATUS_SUCCESS) {
        protocol_set_device_status(
            protocol,
            device,
            protocol_DeviceStatus_DEVICE_STATUS_ERROR,
            "Failed to clear streams.");
        return true;
    }

    switch(mode_index)
    {
       case 0:
           mode = DPS_MODE_STREAM_COMBINED;
           stream0_unit = "hPa \xc2\xb2, \xc2\xb0 degC"; // hPa , Degrees C
           stream1_unit = NULL;
           stream0_name = "Combined";
           stream1_name = NULL;
           break;
       case 1:
           mode = DPS_MODE_STREAM_SPLIT;
           stream0_unit = "hPa \xc2\xb2";         // hPa
           stream1_unit = "\xc2\xb0 degC";         // Degrees C
           stream0_name = "Pressure";
           stream1_name = "Temperature";
           break;
       case 2:
           mode = DPS_MODE_STREAM_ONLY_PRESSURE;
           stream0_unit = "hPa \xc2\xb2";         // hPa
           stream0_name = "Pressure";
           stream1_name = NULL;
           break;
       case 3:
           mode = DPS_MODE_STREAM_ONLY_TEMP;
           stream0_unit = "\xc2\xb0 degC";         // Degrees C
           stream1_unit = NULL;
           stream0_name = "Temperature";
           stream1_name = NULL;
           break;
       default: return false;
    }

    /* Add stream #0 */
    stream0 = protocol_add_stream(
           protocol,
           device,
           stream0_name,
           protocol_StreamDirection_STREAM_DIRECTION_OUTPUT,
           protocol_DataType_DATA_TYPE_F32,
           rate,
           1,
           stream0_unit);
    if(stream0 < 0) {
        protocol_set_device_status(
            protocol,
            device,
            protocol_DeviceStatus_DEVICE_STATUS_ERROR,
            "Failed to add streams.");
        return true;
    }

    if(mode == DPS_MODE_STREAM_COMBINED)
    {
        result = protocol_add_stream_rank(
            protocol,
            device,
            stream0,
            "Sensor",
            2,
            (const char* []) { "Pressure", "Temperature" });

        if(result != PROTOCOL_STATUS_SUCCESS)
        {
            protocol_set_device_status(
                protocol,
                device,
                protocol_DeviceStatus_DEVICE_STATUS_ERROR,
                "Failed to add streams dimension.");
            return true;
        }
    }

    /* Add optional stream #1 */
    if(stream1_name != NULL)
    {
        stream1 = protocol_add_stream(
           protocol,
           device,
           stream1_name,
           protocol_StreamDirection_STREAM_DIRECTION_OUTPUT,
           protocol_DataType_DATA_TYPE_F32,
           rate,
           1,
           stream1_unit);
        if(stream1 < 0)
         {
            protocol_set_device_status(
                protocol,
                device,
                protocol_DeviceStatus_DEVICE_STATUS_ERROR,
                "Failed to add streams.");
            return true;
        }

        result = protocol_add_stream_rank(
            protocol,
            device,
            stream1,
            "Channel",
            1,
            (const char* []) { "Pressure" });
        if(result != PROTOCOL_STATUS_SUCCESS)
        {
            protocol_set_device_status(
                protocol,
                device,
                protocol_DeviceStatus_DEVICE_STATUS_ERROR,
                "Failed to add streams dimension.");
            return true;
        }
    }

    result = protocol_add_stream_rank(
        protocol,
            device,
            stream0,
            "Channel",
            1,
            (const char* []) { "Temperature"});
    if(result != PROTOCOL_STATUS_SUCCESS) {
        protocol_set_device_status(
            protocol,
            device,
            protocol_DeviceStatus_DEVICE_STATUS_ERROR,
            "Failed to add streams dimension.");
        return true;
    }

    protocol_set_device_status(
        protocol,
        device,
        protocol_DeviceStatus_DEVICE_STATUS_READY,
        "Device is ready.");

    return true;
}

/******************************************************************************
* Function Name: _start_streams
********************************************************************************
* Summary:
*  Called when streaming is started. This may also initialize the device.
*
* Parameters:
*  protocol: pointer the protocol handle
*  device: the device index
*  ostream: Pointer to the output stream to write to
*  arg: pointer the device struct (dev_dps368_t)
*
*******************************************************************************/
static void _start_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg)
{
    int rate_index;
    int rate = 8;
    int mode_index;
    stream_mode_t mode;
    dev_dps368_t* dps = (dev_dps368_t*)arg;
    UNUSED(ostream);

    protocol_get_option_oneof(protocol, device, DPS_OPTION_KEY_FREQUENCY, &rate_index);
    switch(rate_index)
    {
        case 0: rate = 8; break;
        case 1: rate = 16; break;
        case 2: rate = 32; break;
        case 3: rate = 64; break;
        case 4: rate = 128; break;
    }

    protocol_get_option_oneof(protocol, device, DPS_OPTION_KEY_STREAM_MODE, &mode_index);
    switch(mode_index)
    {
       case 0: mode = DPS_MODE_STREAM_COMBINED; break;
       case 1: mode = DPS_MODE_STREAM_SPLIT; break;
       case 2: mode = DPS_MODE_STREAM_ONLY_PRESSURE; break;
       case 3: mode = DPS_MODE_STREAM_ONLY_TEMP; break;
       default: return;
    }

    if(!_config_hw(dps, rate, mode, dps->i2c))
    {
        protocol_set_device_status(
            protocol,
            device,
            protocol_DeviceStatus_DEVICE_STATUS_ERROR,
            "Failed to configure hardware.");
    }
    else
    {
        protocol_set_device_status(
            protocol,
            device,
            protocol_DeviceStatus_DEVICE_STATUS_ACTIVE,
            "Device is streaming.");
    }
    dps->sample_time_tick = clock_get_tick();

}

/******************************************************************************
* Function Name: _stop_streams
********************************************************************************
* Summary:
*  Called when streaming is stopped.
*
* Parameters:
*  protocol: pointer the protocol handle
*  device: the device index
*  ostream: Pointer to the output stream to write to
*  arg: pointer the device struct (dev_dps368_t)
*
*******************************************************************************/
static void _stop_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg)
{

    UNUSED(ostream);

    protocol_set_device_status(
        protocol,
        device,
        protocol_DeviceStatus_DEVICE_STATUS_READY,
        "Device stopped");

}

/******************************************************************************
* Function Name: _write_payload
********************************************************************************
* Summary:
*  Used by protocol_send_data_chunk to write the actual data.
*
* Parameters:
*  protocol: pointer the protocol handle
*  stream_id: the device index
*  stream_id: the stream index
*  frame_count: number of frames to write
*  total_bytes: total number of bytes to write (= frame_count * sizeof(type) * frame_shape.flat)
*  ostream: pointer to the output stream to write to
*  arg: pointer the device struct (dev_dps368_t)
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
    UNUSED(frame_count);
    UNUSED(protocol);

    dev_dps368_t* dps = (dev_dps368_t*)arg;

    switch(dps->stream_mode)
    {
        case DPS_MODE_STREAM_COMBINED:
        {
            return pb_write(ostream, (const pb_byte_t *)dps->data_combined, total_bytes);
        }

        case DPS_MODE_STREAM_SPLIT:
        {
            float *frame = stream_id == 0 ? dps->pressure_data : dps->temperature_data;
            return pb_write(ostream, (const pb_byte_t *)frame, total_bytes);
        }

        case DPS_MODE_STREAM_ONLY_PRESSURE:
        {
            return pb_write(ostream, (const pb_byte_t *)dps->pressure_data, total_bytes);
        }

        case DPS_MODE_STREAM_ONLY_TEMP:
        {
            return pb_write(ostream, (const pb_byte_t *)dps->temperature_data, total_bytes);
        }

        default:
        {
            return false;
        }

    }
}

/******************************************************************************
* Function Name: _poll_streams
********************************************************************************
* Summary:
*  Called periodically to send data messages when the device is active.
*
* Parameters:
*  protocol: pointer the protocol handle
*  device: the device index
*  ostream: pointer to the output stream to write to
*  arg: pointer the device struct (dev_dps368_t)
*
*******************************************************************************/

static void _poll_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg)
{
    dev_dps368_t* dps = (dev_dps368_t*)arg;
    clock_tick_t current_time = clock_get_tick();
    /*Reinterpret this timing as the time we wish the sample to happen at*/
    clock_tick_t current_treshold = dps->sample_time_tick;
    clock_tick_t total_drift = current_time - current_treshold;
    
    /*If we are to late we skip this frame and save time.*/
    /*Previous data package will be resent. */
    bool late = false;
    uint32_t drift_ms = (uint32_t)total_drift;
    if (drift_ms > 200)
    {
        /*Tens of microseconds. 200 is equal to 2 ms.*/
        late=1;
    }    
    
    /*The first sampling is now done as soon as possible.*/
    if(current_time >= current_treshold )
    {
        /*If we are late.. Skip the reading of the sensor.*/
        if ( late )
        {
            late = false;

        }
        else
        {
            if(!_read_hw(dps,dps->i2c))
            {
                return;
            }
        }


        /* This is updated whenever there is an _read_hw call! However since the _read_hw might fail
        * and when we are resending the previous data we have to force this to 1 anyway.
        */
        dps->frames_sampled = 1;

        /*Since we in reality dont drop any frames anymore.*/
        dps->frames_dropped = 0;

        /*When we should do the next frame.*/
        dps->sample_time_tick += dps->period_tick;

        /*Always send something.*/
        {
            protocol_send_data_chunk(protocol, device, 0, dps->frames_sampled, dps->frames_dropped, ostream, _write_payload);

            if(dps->stream_mode == DPS_MODE_STREAM_SPLIT)
                protocol_send_data_chunk(protocol, device, 1, dps->frames_sampled, dps->frames_dropped, ostream, _write_payload);

            dps->frames_dropped = 0;
            dps->frames_sampled = 0;

        }
    }
}

/******************************************************************************
* Function Name: dev_dps368_register
********************************************************************************
* Summary:
*  Registers this device. This is the only exported symbol from this object.
*
* Parameters:
*  protocol: Pointer to the protocol handle.
*  i2c: Pointer to the I2C instance.
*
* Returns:
*  True on success, else false.
*
*******************************************************************************/
bool dev_dps368_register(protocol_t* protocol, mtb_hal_i2c_t* i2c)
{
    int status;

    dev_dps368_t* dps = (dev_dps368_t*)malloc(sizeof(dev_dps368_t));
    if(dps == NULL)
    {
        return false;
    }

    memset(dps, 0, sizeof(dev_dps368_t));
    if(!_init_hw(dps, i2c))
    {
        free(dps);
        return false;
    }

    device_manager_t manager = {
        .arg = dps,
        .configure_streams = _configure_streams,
        .start = _start_streams,
        .stop = _stop_streams,
        .poll = _poll_streams,
        .data_received = NULL // has no input streams
    };

    int device = protocol_add_device(
        protocol,
        protocol_DeviceType_DEVICE_TYPE_SENSOR,
        "DPS",
        "Pressure and Temperature (DPS368)",
        manager);

    if(device < 0)
    {
        return false;
    }

    status = protocol_add_option_oneof(
        protocol,
        device,
        DPS_OPTION_KEY_FREQUENCY,
        "Frequency",
        "Sample frequency (Hz). As frequency goes up, oversampling goes down.",
        3,
        (const char* []) { "8 Hz", "16 Hz", "32 Hz", "64 Hz", "128 Hz" },
        5);


    if(status != PROTOCOL_STATUS_SUCCESS)
    {
        return false;
    }


     status = protocol_add_option_oneof(
        protocol,
        device,
        DPS_OPTION_KEY_STREAM_MODE,
        "Mode",
        "Stream Configuration",
        0,
        (const char* []) { "Combined", "Split", "Only Pressure", "Only Temperature" },
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

#endif /* IM_ENABLE_DPS368 */

/* [] END OF FILE */
