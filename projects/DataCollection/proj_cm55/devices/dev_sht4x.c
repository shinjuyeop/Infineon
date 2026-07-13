/******************************************************************************
* File Name: dev_sht4x.c
*
* Description: This file implements the interface with the sht4x sensor.
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

#ifdef IM_ENABLE_SHT4X

#include <cybsp.h>
#include "protocol/protocol.h"
#include "protocol/pb_encode.h"
#include "clock.h"
#include "dev_sht4x.h"
#include "mtb_sht4x.h"
#include "common.h"

/*******************************************************************************
* Macros
*******************************************************************************/

#define DEV_SHT4X_OPTION_KEY_PRECISION    (1)
#define DEV_SHT4X_OPTION_KEY_STREAM_MODE  (2)

/* At 200Hz/4 = 50 Hz chunk frequency */
#define MAX_FRAMES_IN_CHUNK   (4)

/* Humidity, Temperature */
#define SENSOR_COUNT          (2)

#define HUMIDITY_TIMER_INT_PRIORITY       (3U)

typedef enum 
{
SHT40_IDLE,
SHT40_MEASURING,
SHT40_READY
} sht40_state_t;

/*******************************************************************************
* Types
*******************************************************************************/

typedef enum {
    /* Both humidity and temperature are in the same stream.
     * The shape of each frame will be [2,1] as
     * Stream 0: {{HUMIDITY}, {TEMPERATURE}} */
    SHT4X_MODE_STREAM_COMBINED = 0,

    /* The humidity and temperature are split in two separate streams
     * each with the shape [1] as
     * Stream 0: {HUMIDITY}
     * Stream 1: {TEMPERATURE}
     */
    SHT4X_MODE_STREAM_SPLIT = 3,

    /* Only the humidity is enabled.
     * Stream 0: {HUMIDITY}
     */
    SHT4X_MODE_STREAM_ONLY_HUMIDITY = 1,

    /* Only the temperature is enabled.
     * Stream 0: {TEMPERATURE}
     */
    SHT4X_MODE_STREAM_ONLY_TEMP = 2,
} stream_mode_t;

typedef enum {

    LOW_PRECISION = 0,
    MEDIUM_PRECISION = 1,
    HIGH_PRECISION = 2

} precision_t;

typedef struct {

    mtb_hal_i2c_t* i2c;
    /* Tick of last sample */
    clock_tick_t sample_time_tick;

    /* The sample period in ticks */
    uint32_t period_tick;

    union {
        struct {
            /*When mode is SHT4X_MODE_STREAM_SPLIT or SHT4X_MODE_STREAM_ONLY_XXX*/
            /* Converted data as RH */
            float humidity_data[MAX_FRAMES_IN_CHUNK];

            /* Converted data as degrees */
            float temperature_data[MAX_FRAMES_IN_CHUNK];
        };

        /*When mode is SHT4X_MODE_STREAM_COMBINED*/
        float data_combined[MAX_FRAMES_IN_CHUNK * SENSOR_COUNT];
    };

    /* Number of frames collected in humidity_data and temperature_data. */
    /* Cleared after each sent data-chunk. Equal or less than frames_in_chunk */
    int frames_sampled;

    /* Max number of frames in each chunk. Is less or equal to MAX_FRAMES_IN_CHUNK*/
    int frames_target;

    /* Number of frames dropped. This is cleared each data-chunk. */
    int frames_dropped;

    /* How streams are presented. */
    stream_mode_t stream_mode;

    /* data reading precision */
    precision_t precison;

} dev_sht4x_t;

/*******************************************************************************
 * Global Variables
 ******************************************************************************/
cy_stc_sysint_t humidity_timer_irq_cfg =
{
    .intrSrc = CYBSP_GENERAL_HUMIDITY_TIMER_IRQ,
    .intrPriority = HUMIDITY_TIMER_INT_PRIORITY
};

/*SHT40 sensor address*/
mtb_sht40_address_t sht40_address;

/*SHT40 sensor state*/
static sht40_state_t sht40_state = SHT40_IDLE;

/*******************************************************************************
* Function Prototypes
*******************************************************************************/
void humidity_timer_isr(void);

void humidity_timer_init(void);

void sht4x_start_measurement(uint8_t cmd, uint32_t delay_us);

static float _convert_milli_to_unit(int32_t val);

static bool _init_hw(dev_sht4x_t *dev, mtb_hal_i2c_t* i2c);

static bool _config_hw(dev_sht4x_t* dev, int rate, stream_mode_t mode, mtb_hal_i2c_t* i2c);

static bool _read_hw(dev_sht4x_t* dev, mtb_hal_i2c_t* i2c);

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

/*******************************************************************************
* Function Definitions
*******************************************************************************/
/******************************************************************************
* Function Name: humidity_timer_isr
********************************************************************************
* Summary:
*   Initializes timer isr
*
* Parameters:
*   
*
* Return:
*   None
*
*******************************************************************************/
void humidity_timer_isr(void)
{
     uint32_t intrMask = Cy_TCPWM_GetInterruptStatus(CYBSP_GENERAL_HUMIDITY_TIMER_HW, CYBSP_GENERAL_HUMIDITY_TIMER_NUM);
     Cy_TCPWM_ClearInterrupt(CYBSP_GENERAL_HUMIDITY_TIMER_HW, CYBSP_GENERAL_HUMIDITY_TIMER_NUM, intrMask);

     if (intrMask & CY_TCPWM_INT_ON_TC)
     {
        /*stop one-shot*/ 
        Cy_TCPWM_PWM_Disable(CYBSP_GENERAL_HUMIDITY_TIMER_HW, CYBSP_GENERAL_HUMIDITY_TIMER_NUM);
        /*Updating SHT40 sensor state to SHT40_READY */
        sht40_state = SHT40_READY;
     } 
}

/******************************************************************************
* Function Name: sht4x_start_measurement
********************************************************************************
* Summary:
*  start reading the humidity sensor
*
* Parameters:
*  cmd        command based on precision
*  delay_us   delay based on read command selected
*
* Return:
*   None
*
*******************************************************************************/
void sht4x_start_measurement(uint8_t cmd, uint32_t delay_us)
{
    /*Send the measurement command*/
    mtb_sht4x_measurement_init(cmd);
    
    /*Configure one-shot timer for conversion time*/ 
    Cy_TCPWM_PWM_Disable(CYBSP_GENERAL_HUMIDITY_TIMER_HW, CYBSP_GENERAL_HUMIDITY_TIMER_NUM);
    Cy_TCPWM_Counter_SetPeriod(CYBSP_GENERAL_HUMIDITY_TIMER_HW, CYBSP_GENERAL_HUMIDITY_TIMER_NUM, (delay_us - 1)); // at 1 MHz = 1 tick = 1us
    Cy_TCPWM_PWM_Enable(CYBSP_GENERAL_HUMIDITY_TIMER_HW, CYBSP_GENERAL_HUMIDITY_TIMER_NUM);
    Cy_TCPWM_TriggerStart_Single(CYBSP_GENERAL_HUMIDITY_TIMER_HW, CYBSP_GENERAL_HUMIDITY_TIMER_NUM);

    /*Updating SHT40 sensor state to SHT40_MEASURING */    
    sht40_state = SHT40_MEASURING; 
}

/******************************************************************************
* Function Name: humidity_timer_init
********************************************************************************
* Summary:
*   Initializes timer for humidity sensor reading
*
* Parameters:
*   
*
* Return:
*   None
*
*******************************************************************************/
void humidity_timer_init(void)
{
    Cy_TCPWM_Counter_Init(CYBSP_GENERAL_HUMIDITY_TIMER_HW, CYBSP_GENERAL_HUMIDITY_TIMER_NUM, &CYBSP_GENERAL_HUMIDITY_TIMER_config);
    /*hook ISR*/
    Cy_SysInt_Init(&humidity_timer_irq_cfg, &humidity_timer_isr);

    NVIC_EnableIRQ(CYBSP_GENERAL_HUMIDITY_TIMER_IRQ); 
    Cy_TCPWM_Counter_Enable(CYBSP_GENERAL_HUMIDITY_TIMER_HW, CYBSP_GENERAL_HUMIDITY_TIMER_NUM);
    
}

/******************************************************************************
* Function Name: _convert_milli_to_unit
********************************************************************************
* Summary:
*   This function converts milli to standard unit
*   eg. milli RH to RH and milli Degrees C to Degrees C
*
* Parameters:
*   val        value in milli unit
*
* Return:
*   unit value
*
*******************************************************************************/
static float _convert_milli_to_unit(int32_t val)
{
    return ((float)val/1000.0 );
}

/******************************************************************************
* Function Name: _init_hw
********************************************************************************
* Summary:
*   Initializes the SHT4X Sesnsor.
*
* Parameters:
*   dev: Pointer to the dev_sht4x_t device handle.
*   i2c: Pointer to the I2C interface resource.
*
* Return:
*   True if operation is successful, otherwise false.
*
*******************************************************************************/
static bool _init_hw(dev_sht4x_t *dev, mtb_hal_i2c_t* i2c)
{
    cy_rslt_t result;
    dev->i2c=i2c;
    dev->stream_mode = SHT4X_MODE_STREAM_COMBINED;
    sht40_address = MTB_SHT40_ADDRESS_DEFAULT;

    result = mtb_sht4x_init(i2c,sht40_address);
    humidity_timer_init();
    if(CY_RSLT_SUCCESS == result)
    {
        printf("SHT4X: Initialized device.\r\n");
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
*   Configures the SHT4X output data rate.
*
* Parameters:
*   dev: Pointer to the dev_sht4x_t device handle.
*   rate: Sample frequency (Hz).
*   mode: Stream mode.
*
* Return:
*   True if configuration is successful, otherwise false.
*
*******************************************************************************/
static bool _config_hw(dev_sht4x_t* dev, int rate, stream_mode_t mode, mtb_hal_i2c_t* i2c)
{
    dev->sample_time_tick = 0;
    dev->period_tick = CLOCK_TICK_PER_SECOND / rate;
    dev->frames_dropped = 0;
    dev->frames_sampled = 0;
    dev->stream_mode = mode;

    /* It takes a while for the sensor to deliver data.
     * Wait for 5 successful reads. Give up after 2 seconds.
     */
    int success = 0;
    for(int i = 0; i < 10; i++)
    {
        if(_read_hw(dev,i2c))
        {
            success++;
        }

        if(success >= 5)
        {
            dev->frames_sampled = 0;
               printf("SHT4X: Configured device. mode=%i, rate=%d Hz\r\n",
                        dev->stream_mode, rate);
            return true;
        }

        Cy_SysLib_Delay(10);
    }

   return false;
}

/******************************************************************************
* Function Name: _read_hw
********************************************************************************
* Summary:
*   Reads the current sht4x data and convert it.
*
* Parameters:
*   dev: Pointer to the sht4x device handle.
*
* Return:
*   True if data retrieval is successful, otherwise false.
*
*******************************************************************************/
static bool _read_hw(dev_sht4x_t* dev, mtb_hal_i2c_t* i2c)
{

    int32_t temperature = 0;
    int32_t humidity = 0;
    
    if(sht40_state == SHT40_IDLE)
    {
        if(dev->precison == HIGH_PRECISION)
        {
            /* Start high-precision measurement, delay 10,000 usec. */
            sht4x_start_measurement(CMD_HIGH_PRECISION, 10000);
        }

        if(dev->precison == MEDIUM_PRECISION)
        {
            /* Start medium-precision measurement, delay 5,000 usec. */         
           sht4x_start_measurement(CMD_MED_PRECISION, 5000);
        }

        if(dev->precison == LOW_PRECISION)
        {
            /* Start low-precision measurement, delay 2,000 usec. */                
           sht4x_start_measurement(CMD_LOW_PRECISION, 2000);
        }
    
    }
    
    if (sht40_state == SHT40_READY)
    {
        /* read the sensor */   
        mtb_sht4x_measure_nonblocking(&temperature, &humidity);
    
        if(dev->stream_mode == SHT4X_MODE_STREAM_COMBINED)
        {
           float *dest = dev->data_combined + dev->frames_sampled * SENSOR_COUNT;
           *dest++ = _convert_milli_to_unit(humidity);
           *dest++ = _convert_milli_to_unit(temperature);
        }

        if(dev->stream_mode == SHT4X_MODE_STREAM_SPLIT || dev->stream_mode == SHT4X_MODE_STREAM_ONLY_HUMIDITY)
        {
           float *dest = dev->humidity_data + dev->frames_sampled ;
           *dest++ = _convert_milli_to_unit(humidity);
        }

        if(dev->stream_mode == SHT4X_MODE_STREAM_SPLIT || dev->stream_mode == SHT4X_MODE_STREAM_ONLY_TEMP)
        {
           float *dest = dev->temperature_data + dev->frames_sampled ;
           *dest++ = _convert_milli_to_unit(temperature);
        }

        dev->frames_sampled++;
        /*Updating SHT40 sensor state to SHT40_IDLE */
        sht40_state = SHT40_IDLE;
    }
    return true;
}

/******************************************************************************
* Function Name: _configure_streams
********************************************************************************
* Summary:
*   Configures the data streams based on the user current settings.
*
* Parameters:
*   protocol: Pointer to the protocol handle.
*   device: The device index.
*   dev: Pointer to the sht4x  device handle.
*
* Return:
*   True to keep the connection open, otherwise false.
*
*******************************************************************************/
static bool _configure_streams(protocol_t* protocol, int device, void* arg)
{
    int result;
    int precision_index;
    int rate = 50;
    int mode_index;
    stream_mode_t mode;
    const char *stream0_unit;
    const char *stream1_unit;
    const char *stream0_name;
    const char *stream1_name;
    int stream0;
    int stream1;
    dev_sht4x_t* dev = (dev_sht4x_t*)arg;

    if(protocol_get_option_oneof(protocol, device, DEV_SHT4X_OPTION_KEY_PRECISION, &precision_index) != PROTOCOL_STATUS_SUCCESS)
    {
        protocol_set_device_status(
            protocol,
            device,
            protocol_DeviceStatus_DEVICE_STATUS_ERROR,
            "Failed to get option frequency.");
        return true;
    }

    switch(precision_index)
    {
       case LOW_PRECISION:
            dev->precison = LOW_PRECISION;
            rate = 50;
            break;

       case MEDIUM_PRECISION:
            dev->precison = HIGH_PRECISION;
            rate = 20;
            break;

       case HIGH_PRECISION:
            dev->precison = HIGH_PRECISION;
            rate = 10;
            break;
    }

    if(protocol_get_option_oneof(protocol, device, DEV_SHT4X_OPTION_KEY_STREAM_MODE, &mode_index) != PROTOCOL_STATUS_SUCCESS)
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
           mode = SHT4X_MODE_STREAM_COMBINED;
           stream0_unit = "RH \xc2\xb2, \xc2\xb0 degC"; // RH , Degrees C
           stream1_unit = NULL;
           stream0_name = "Combined";
           stream1_name = NULL;
           break;
       case 1:
           mode = SHT4X_MODE_STREAM_SPLIT;
           stream0_unit = "RH \xc2\xb2";         // RH
           stream1_unit = "\xc2\xb0 degC";         // Degrees C
           stream0_name = "Humidity";
           stream1_name = "Temperature";
           break;
       case 2:
           mode = SHT4X_MODE_STREAM_ONLY_HUMIDITY;
           stream0_unit = "RH \xc2\xb2";         // RH
           stream0_name = "Humidity";
           stream1_name = NULL;
           break;
       case 3:
           mode = SHT4X_MODE_STREAM_ONLY_TEMP;
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

    if(mode == SHT4X_MODE_STREAM_COMBINED)
    {
        result = protocol_add_stream_rank(
            protocol,
            device,
            stream0,
            "Sensor",
            2,
            (const char* []) { "Humidity", "Temperature" });

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
            (const char* []) { "Humidity" });
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
*   Starts the data streaming process.
*
* Parameters:
*   protocol: Pointer to the protocol handle.
*   device: The device index.
*   ostream: Pointer to the output stream to write to
*   arg: Pointer to the sht4x  device handle.
*
*******************************************************************************/
static void _start_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg)
{
    int precision_index;
    int rate = 50;
    int mode_index;
    stream_mode_t mode;
    dev_sht4x_t* dev = (dev_sht4x_t*)arg;
    UNUSED(ostream);
    protocol_get_option_oneof(protocol, device, DEV_SHT4X_OPTION_KEY_PRECISION, &precision_index);
    switch(precision_index)
    {
       case LOW_PRECISION:
            dev->precison = LOW_PRECISION;
            rate = 50;
            break;

       case MEDIUM_PRECISION:
            dev->precison = HIGH_PRECISION;
            rate = 20;
            break;

       case HIGH_PRECISION:
            dev->precison = HIGH_PRECISION;
            rate = 10;
            break;
    }


    protocol_get_option_oneof(protocol, device, DEV_SHT4X_OPTION_KEY_STREAM_MODE, &mode_index);
    switch(mode_index)
    {
       case 0: mode = SHT4X_MODE_STREAM_COMBINED; break;
       case 1: mode = SHT4X_MODE_STREAM_SPLIT; break;
       case 2: mode = SHT4X_MODE_STREAM_ONLY_HUMIDITY; break;
       case 3:mode = SHT4X_MODE_STREAM_ONLY_TEMP; break;
       default: return;
    }

    if(!_config_hw(dev, rate, mode, dev->i2c))
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
    dev->sample_time_tick = clock_get_tick();

}

/******************************************************************************
* Function Name: _stop_streams
********************************************************************************
* Summary:
*   Stops the streaming process.
*
* Parameters:
*   protocol: Pointer to the protocol handle.
*   device: The device index.
*   ostream: Pointer to the output stream to write to
*   arg: Pointer to the sht4x  device handle.
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
*   Writes the data into the output stream.
*
* Parameters:
*   protocol: Pointer to the protocol handle.
*   device_id: The device index.
*   stream_id: The stream index.
*   frame_count: Number of frames to write.
*   total_bytes: Total number of bytes to write (frame_count * sizeof(type) * frame_shape.flat).
*   ostream: Pointer to the output stream to write to.
*   arg: Pointer to the sht4x  device handle.
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
    UNUSED(frame_count);
    UNUSED(protocol);

    dev_sht4x_t* dev = (dev_sht4x_t*)arg;

    switch(dev->stream_mode)
    {
        case SHT4X_MODE_STREAM_COMBINED:
        {
            return pb_write(ostream, (const pb_byte_t *)dev->data_combined, total_bytes);
        }

        case SHT4X_MODE_STREAM_SPLIT:
        {
            float *frame = stream_id == 0 ? dev->humidity_data : dev->temperature_data;
            return pb_write(ostream, (const pb_byte_t *)frame, total_bytes);
        }

        case SHT4X_MODE_STREAM_ONLY_HUMIDITY:
        {
            return pb_write(ostream, (const pb_byte_t *)dev->humidity_data, total_bytes);
        }

        case SHT4X_MODE_STREAM_ONLY_TEMP:
        {
            return pb_write(ostream, (const pb_byte_t *)dev->temperature_data, total_bytes);
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
*   Periodically polls for data and sends data-chunk messages.
*
* Parameters:
*   protocol: Pointer to the protocol handle.
*   device: The device index.
*   ostream: Pointer to the output stream to write to.
*   dev: Pointer to the sht4x  device handle.
*
*******************************************************************************/
static void _poll_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg)
{
    dev_sht4x_t* dev = (dev_sht4x_t*)arg;
    clock_tick_t current_time = clock_get_tick();

    /*Reinterpret this timing as the time we wish the sample to happen at*/
    clock_tick_t current_treshold = dev->sample_time_tick;

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
            if(!_read_hw(dev,dev->i2c))
            {
                return;
            }
        }


        /* This is updated whenever there is an _read_hw call! However since the _read_hw might fail
        * and when we are resending the previous data we have to force this to 1 anyway.
        */
        dev->frames_sampled = 1;

        /*Since we in reality dont drop any frames anymore.*/
        dev->frames_dropped = 0;

        /*When we should do the next frame.*/
        dev->sample_time_tick += dev->period_tick;

        /*Always send something.*/
        {
            protocol_send_data_chunk(protocol, device, 0, dev->frames_sampled, dev->frames_dropped, ostream, _write_payload);

            if(dev->stream_mode == SHT4X_MODE_STREAM_SPLIT)
                protocol_send_data_chunk(protocol, device, 1, dev->frames_sampled, dev->frames_dropped, ostream, _write_payload);

            dev->frames_dropped = 0;
            dev->frames_sampled = 0;

        }
    }
}

/******************************************************************************
* Function Name: dev_sht4x_register
********************************************************************************
* Summary:
*   Registers the sht4x device with the protocol, configuring it for use.
*
* Parameters:
*   protocol: Pointer to the protocol handle.
*   i2c: Pointer to the I2C interface resource.
*
* Return:
*   True if registration is successful, otherwise false.
*
*******************************************************************************/
bool dev_sht4x_register(protocol_t* protocol, mtb_hal_i2c_t* i2c)
{
    int status;

    dev_sht4x_t *dev = (dev_sht4x_t*)malloc(sizeof(dev_sht4x_t));
    if(dev == NULL)
    {
        return false;
    }

    memset(dev, 0, sizeof(dev_sht4x_t));
    if(!_init_hw(dev, i2c))
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
        "SHT4X",
        "Humidity and temperature",
        manager);

    if(device < 0)
    {
        return false;
    }

    status = protocol_add_option_oneof(
        protocol,
        device,
        DEV_SHT4X_OPTION_KEY_PRECISION,
        "Precision",
        "Data reading precision",
        1,
        (const char* []) { "Low","Medium","High" },
        3);

    if(status != PROTOCOL_STATUS_SUCCESS)
    {
        return false;
    }


     status = protocol_add_option_oneof(
        protocol,
        device,
        DEV_SHT4X_OPTION_KEY_STREAM_MODE,
        "Mode",
        "Stream Configuration",
        0,
        (const char* []) { "Combined", "Split", "Only Humidity", "Only Temperature" },
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

#endif /* IM_ENABLE_SHT4X */

/* [] END OF FILE */

