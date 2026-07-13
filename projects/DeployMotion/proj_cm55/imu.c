/******************************************************************************
* File Name:   imu.c
*
* Description: This file implements the interface with the BMI270 motion sensor.
*              The configurations of the motion sensor are specified in the
*              config.h file.
*
* Related Document: See README.md
*
*
********************************************************************************
* (c) 2024-2026, Infineon Technologies AG, or an affiliate of Infineon
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
#include "cybsp.h"
#include "mtb_bmi270.h"
#include "bmi2.h"

#include "config.h"
#include "imu.h"

/*******************************************************************************
* Macros
*******************************************************************************/
#if IMU_SAMPLE_RANGE == BMI2_ACC_RANGE_2G
#define IMU_ACC_DIVIDE     0x4000
#elif IMU_SAMPLE_RANGE == BMI2_ACC_RANGE_4G
#define IMU_ACC_DIVIDE     0x2000
#elif IMU_SAMPLE_RANGE == BMI2_ACC_RANGE_8G
#define IMU_ACC_DIVIDE     0x1000
#elif IMU_SAMPLE_RANGE == BMI2_ACC_RANGE_16G
#define IMU_ACC_DIVIDE     0x800
#endif /* IMU_SAMPLE_RANGE */

#if IMU_GYRO_SAMPLE_RANGE == BMI2_GYR_RANGE_125
#define IMU_GYRO_DIVIDE    262.0f
#elif IMU_GYRO_SAMPLE_RANGE == BMI2_GYR_RANGE_250
#define IMU_GYRO_DIVIDE    131.0f
#elif IMU_GYRO_SAMPLE_RANGE == BMI2_GYR_RANGE_500
#define IMU_GYRO_DIVIDE    65.5f
#elif IMU_GYRO_SAMPLE_RANGE == BMI2_GYR_RANGE_1000
#define IMU_GYRO_DIVIDE    32.8f
#elif IMU_GYRO_SAMPLE_RANGE == BMI2_GYR_RANGE_2000
#define IMU_GYRO_DIVIDE    16.4f
#endif /* IMU_GYRO_SAMPLE_RANGE */

#if defined(IMU_ACC) && defined(IMU_GYR)
#define IMU_NUM_AXIS                (6U)
#else
#define IMU_NUM_AXIS                (3U)
#endif /* IMU_ACC && IMU_GYR */
#define IMU_INTERRUPT_PRIORITY      (2U)
#define IMU_INTR_MASK               (0x00000001UL << CYBSP_IMU_INT1_PORT_NUM)
#define MASKED_TRUE                 (1U)

/* Number of frames to skip before sending data for processing. */
#define NUM_IMU_FRAMES_TO_SKIP      (1U)

/*******************************************************************************
* Global Variables
*******************************************************************************/
static mtb_hal_i2c_t CYBSP_I2C_CONTROLLER_0_hal_obj;
static cy_stc_scb_i2c_context_t CYBSP_I2C_CONTROLLER_0_context;

/* Instance of BMI270 sensor structure */
mtb_bmi270_t bmi270;
/* Number of bytes of FIFO data */
static uint8_t fifo_data[BMI2_FIFO_RAW_DATA_BUFFER_SIZE] = { 0 };

#ifdef IMU_ACC
/* Array of accelerometer frames */
static struct bmi2_sens_axes_data fifo_accel_data[BMI2_FIFO_ACCEL_FRAME_COUNT] = { { 0 } };
#endif /* IMU_ACC */

#ifdef IMU_GYR
/* Array of gyroscope frames */
static struct bmi2_sens_axes_data fifo_gyro_data[BMI2_FIFO_GYRO_FRAME_COUNT] = { { 0 } };
#endif /* IMU_GYR */

/* Initialize FIFO frame structure. */
struct bmi2_fifo_frame fifoframe = { 0 };

/* Flag to check if the data from IMU is ready for processing. */
static volatile bool imu_flag;

/* Number of frames to skip before sending data for processing. */
static uint8_t imu_skip_frames = NUM_IMU_FRAMES_TO_SKIP;

/*******************************************************************************
* Function Prototypes
*******************************************************************************/
static void imu_interrupt_handler(void);
static cy_rslt_t imu_fifo_init(void);
void imu_remap_sensor_orientation(struct bmi2_sens_axes_data *data);

/*******************************************************************************
* Function Definitions
*******************************************************************************/
/*******************************************************************************
* Function Name: imu_init
********************************************************************************
* Summary:
*  A function used to initialize the IMU (BMI270). Starts a timer that
*  triggers an interrupt at at the sample rate specified in config.h.
*
* Parameters:
*  None
*
* Return:
*  The status of the initialization.
*
*******************************************************************************/
cy_rslt_t imu_init(void)
{
    cy_rslt_t result;

    /* Initialize the I2C master interface for BMI270 motion sensor */
    result = Cy_SCB_I2C_Init(CYBSP_I2C_CONTROLLER_HW,
                             &CYBSP_I2C_CONTROLLER_config,
                             &CYBSP_I2C_CONTROLLER_0_context);

    if(CY_RSLT_SUCCESS != result)
    {
        printf(" Error : I2C initialization failed !!\r\n");
        CY_ASSERT(0);
    }

    Cy_SCB_I2C_Enable(CYBSP_I2C_CONTROLLER_HW);

    /* Configure the I2C master interface with the desired clock frequency */
    result = mtb_hal_i2c_setup(&CYBSP_I2C_CONTROLLER_0_hal_obj,
                               &CYBSP_I2C_CONTROLLER_hal_config,
                               &CYBSP_I2C_CONTROLLER_0_context,
                              NULL);

    if(CY_RSLT_SUCCESS != result)
    {
        printf(" Error : I2C setup failed !!\r\n");
        CY_ASSERT(0);
    }

    /* Initialize the BMI270 motion sensor */
    result = mtb_bmi270_init_i2c(&bmi270, &CYBSP_I2C_CONTROLLER_0_hal_obj, MTB_BMI270_ADDRESS_DEFAULT);
    if(CY_RSLT_SUCCESS != result)
    {
        printf(" Error : IMU sensor init failed !!\r\n");
        CY_ASSERT(0);
    }

    result = mtb_bmi270_config_default(&bmi270);
    if(CY_RSLT_SUCCESS != result)
    {
        printf(" Error : IMU sensor config failed !!\r\n");
        CY_ASSERT(0);
    }

#ifdef IMU_ACC
    struct bmi2_sens_config acc_cfg = {0};
    acc_cfg.type = BMI2_ACCEL;
    bmi2_get_sensor_config(&acc_cfg, 1, &bmi270.sensor);
    acc_cfg.cfg.acc.odr = IMU_SAMPLE_RATE;
    acc_cfg.cfg.acc.range = IMU_SAMPLE_RANGE;
    acc_cfg.cfg.acc.bwp = BMI2_ACC_OSR4_AVG1;
    acc_cfg.cfg.acc.filter_perf = BMI2_PERF_OPT_MODE;
    result = bmi2_set_sensor_config(&acc_cfg, 1, &bmi270.sensor);
    if (BMI2_OK != result)
    {
        printf(" Error: Failed to set sensor config\n");
        CY_ASSERT(0);
    }
#endif /* IMU_ACC */

#ifdef IMU_GYR
    /* Configure gyroscope if enabled */
    struct bmi2_sens_config gyr_cfg = {0};
    gyr_cfg.type = BMI2_GYRO;
    bmi2_get_sensor_config(&gyr_cfg, 1, &bmi270.sensor);
    gyr_cfg.cfg.gyr.odr = IMU_GYRO_SAMPLE_RATE;
    gyr_cfg.cfg.gyr.range = IMU_GYRO_SAMPLE_RANGE;
    gyr_cfg.cfg.gyr.bwp = BMI2_GYR_NORMAL_MODE;
    gyr_cfg.cfg.gyr.noise_perf = BMI2_POWER_OPT_MODE;
    gyr_cfg.cfg.gyr.filter_perf = BMI2_PERF_OPT_MODE;
    result = bmi2_set_sensor_config(&gyr_cfg, 1, &bmi270.sensor);
    if (BMI2_OK != result)
    {
        printf(" Error: Failed to set sensor config\n");
        CY_ASSERT(0);
    }
#endif /* IMU_GYR */

    result = imu_fifo_init();
    if (BMI2_OK != result)
    {
        CY_ASSERT(0);
    }

    imu_flag = false;

    return result;
}

/*******************************************************************************
* Function Name: imu_fifo_init
********************************************************************************
* Summary:
*  A function used to initialize the FIFO related configurations for the
*  BMI270 IMU.
*
* Parameters:
*  None
*
* Return:
*  The status of the initialization.
*
*******************************************************************************/
static cy_rslt_t imu_fifo_init(void)
{
    cy_rslt_t result;

    /* Accel and gyro sensor are listed in array. */
    uint8_t sensor_sel[] =
    {
#ifdef IMU_ACC
        BMI2_ACCEL,
#endif /* IMU_ACC */
#ifdef IMU_GYR
        BMI2_GYRO,
#endif /* IMU_GYR */
    };


    /* Interrupt config structure for IMU interrupt */
    static cy_stc_sysint_t intrCfg =
    {
        CYBSP_IMU_INT1_IRQ,       /* Interrupt source */
        IMU_INTERRUPT_PRIORITY    /* Interrupt priority */
    };

    /* Interrupt pin configuration */
    static struct bmi2_int_pin_config pin_config = { 0 };


    /* Sensors must be enabled after setting configurations */
    result = bmi270_sensor_enable(sensor_sel, (sizeof(sensor_sel) /
                                  sizeof(sensor_sel[0])), &bmi270.sensor);
    if (BMI2_OK != result)
    {
        printf(" Error: Failed to enable sensor\n");
        return result;
    }

    /* Before setting FIFO, disable the advance power save mode. */
    result = bmi2_set_adv_power_save(BMI2_DISABLE, &bmi270.sensor);
    if (BMI2_OK != result)
    {
        printf(" Error: Failed to disable advance power save mode\n");
        return result;
    }

    /* Initially disable all configurations in fifo. */
    result = bmi2_set_fifo_config(BMI2_FIFO_ALL_EN, BMI2_DISABLE, &bmi270.sensor);
    if (BMI2_OK != result)
    {
        printf(" Error: Failed to disable all FIFO configurations\n");
        return result;
    }

    /* Mapping the buffer to store the fifo data. */
    fifoframe.data = fifo_data;

    /* Length of FIFO frame. */
    fifoframe.length = BMI2_FIFO_RAW_DATA_BUFFER_SIZE;

#ifdef IMU_ACC
    /* Set FIFO configuration by enabling accelerometer */
    result = bmi2_set_fifo_config(BMI2_FIFO_ACC_EN, BMI2_ENABLE, &bmi270.sensor);
    if (BMI2_OK != result)
    {
        printf(" Error: Failed to enable FIFO configuration\n");
        return result;
    }
#endif /* IMU_ACC */

#ifdef IMU_GYR
    /* Set FIFO configuration by enabling gyroscope */
    result = bmi2_set_fifo_config(BMI2_FIFO_GYR_EN, BMI2_ENABLE, &bmi270.sensor);
    if (BMI2_OK != result)
    {
        printf(" Error: Failed to enable FIFO configuration\n");
        return result;
    }
#endif /* IMU_GYR */

    /* To enable headerless mode, disable the header. */
    result = bmi2_set_fifo_config(BMI2_FIFO_HEADER_EN, BMI2_DISABLE, &bmi270.sensor);
    if (BMI2_OK != result)
    {
        printf(" Error: Failed to disable FIFO header\n");
        return result;
    }

    /* Interrupt pin configuration */
    pin_config.pin_type = BMI2_INT1;
    pin_config.pin_cfg[0].input_en = BMI2_INT_INPUT_DISABLE;
    pin_config.pin_cfg[0].output_en = BMI2_INT_OUTPUT_ENABLE;
    pin_config.pin_cfg[0].lvl = BMI2_INT_ACTIVE_LOW;
    pin_config.pin_cfg[0].od = BMI2_INT_PUSH_PULL;
    pin_config.int_latch = BMI2_INT_NON_LATCH;

    /* Set Hardware interrupt pin configuration */
    result = bmi2_set_int_pin_config(&pin_config, &bmi270.sensor);
    if (BMI2_OK != result)
    {
        printf(" Error: Failed to set interrupt pin configuration\n");
        return result;
    }

    /* FIFO water-mark interrupt is enabled. */
    fifoframe.data_int_map = BMI2_FWM_INT;

    /* Map water-mark interrupt to the required interrupt pin. */
    result = bmi2_map_data_int(fifoframe.data_int_map, BMI2_INT1, &bmi270.sensor);
    if (BMI2_OK != result)
    {
        printf(" Error: Failed to map data interrupt\n");
        return result;
    }

    /* Flush the FIFO before setting the watermark level. */
    result = bmi2_set_command_register(BMI2_FIFO_FLUSH_CMD, &bmi270.sensor);
    if (BMI2_OK != result)
    {
        printf(" Error: Failed to flush FIFO\n");
        return result;
    }

    /* Set the watermark level. */
    fifoframe.wm_lvl = BMI2_FIFO_WATERMARK_LEVEL;
    fifoframe.length = BMI2_FIFO_RAW_DATA_BUFFER_SIZE;

    /* Set the water-mark level if water-mark interrupt is mapped. */
    result = bmi2_set_fifo_wm(fifoframe.wm_lvl, &bmi270.sensor);
    if (BMI2_OK != result)
    {
        printf(" Error: Failed to set FIFO watermark level\n");
        return result;
    }

    /* Clear GPIO and NVIC interrupt before initializing to avoid false
     * triggering.
     */
    Cy_GPIO_ClearInterrupt(CYBSP_IMU_INT1_PORT, CYBSP_IMU_INT1_PIN);
    NVIC_ClearPendingIRQ(CYBSP_IMU_INT1_IRQ);

    /* Initialize the interrupt and register interrupt callback */
    Cy_SysInt_Init(&intrCfg, &imu_interrupt_handler);

    /* Enable the interrupt in the NVIC */
    NVIC_EnableIRQ(intrCfg.intrSrc);

    return result;
}


/*******************************************************************************
* Function Name: imu_interrupt_handler
********************************************************************************
* Summary:
*  Handler for the IMU interrupt. Sets the flag for the processing in the main
*  loop.
*
* Parameters:
*  None
*
* Return:
*  None
*
*******************************************************************************/
static void imu_interrupt_handler(void)
{
    /* Get interrupt cause */
    uint32_t intrSrc = Cy_GPIO_GetInterruptCause0();

    /* Check if the interrupt was from the user button's port and pin */
    if((IMU_INTR_MASK == (intrSrc & IMU_INTR_MASK)) &&
       (MASKED_TRUE == Cy_GPIO_GetInterruptStatusMasked(CYBSP_IMU_INT1_PORT,
               CYBSP_IMU_INT1_PIN)))
    {
        /* Clear the interrupt */
        Cy_GPIO_ClearInterrupt(CYBSP_IMU_INT1_PORT, CYBSP_IMU_INT1_PIN);
        NVIC_ClearPendingIRQ(CYBSP_IMU_INT1_IRQ);

        /* The flag is set to true, meaning data is ready to be processed */
        imu_flag = true;
    }
}


/*******************************************************************************
* Function Name: imu_data_process
********************************************************************************
* Summary:
*  This function checks for the IMU interrupt status, reads the FIFO data from
*  the IMU, feeds the data to the DEEPCRAFT pre-processor, and prints the
*  processed results.
*
* Parameters:
*  None
*
* Return:
*  CY_RSLT_SUCCESS if successful, otherwise an error code indicating failure.
*******************************************************************************/
cy_rslt_t imu_data_process(void)
{
    /* Variables required to get the IMU data and process the results. */
    cy_rslt_t result = CY_RSLT_SUCCESS;
    float label_scores[IMAI_DATA_OUT_COUNT];
    char *label_text[] = IMAI_DATA_OUT_SYMBOLS;
    int16_t best_label;
    float max_score;
    uint16_t fifo_length = 0;
#ifdef IMU_ACC
    uint16_t accel_frame_length;
#endif /* IMU_ACC */
#ifdef IMU_GYR
    uint16_t gyro_frame_length;
#endif /* IMU_GYR */
    uint16_t int_status = 0;
    uint16_t watermark = 0;
    uint16_t index = 0;

    /* Check if IMU Data is ready to be processed */
    if (!imu_flag)
    {
        result = IMU_DATA_NOT_READY;
        return result;
    }

    /* Reset the flag to false, indicating that the data is being processed */
    imu_flag = false;

    /* Read FIFO data on interrupt. */
    result = bmi2_get_int_status(&int_status, &bmi270.sensor);
    if (BMI2_OK != result)
    {
        printf(" Error: Failed to get interrupt status\n");
        return result;
    }

    /* To check the status of FIFO watermark interrupt. */
    if (int_status & BMI2_FWM_INT_STATUS_MASK)
    {
        /* Redraw the fixed-width inference result in place without clearing
         * the screen first, which avoids visible flicker.
         */
        printf("\x1b[H");

#ifdef COMPONENT_CM33
        printf("DEEPCRAFT Studio Deploy Motion Example - CM33\r\n\n");
#else
        printf("DEEPCRAFT Studio Deploy Motion Example - CM55\r\n\n");
#endif /* COMPONENT_CM33 */

        /* Turn user led on.*/
        Cy_GPIO_Write(CYBSP_USER_LED_PORT, CYBSP_USER_LED_PIN, CYBSP_LED_STATE_ON);

        result = bmi2_get_fifo_wm(&watermark, &bmi270.sensor);
        if (BMI2_OK != result)
        {
            printf(" Error: Failed to get FIFO watermark level\n");
            return result;
        }

        result = bmi2_get_fifo_length(&fifo_length, &bmi270.sensor);
        if (BMI2_OK != result)
        {
            printf(" Error: Failed to get FIFO length\n");
            return result;
        }

        /* Updating FIFO length to be read based on available length and dummy byte updation */
        fifoframe.length = fifo_length + bmi270.sensor.dummy_byte;

        /* Read FIFO data. */
        result = bmi2_read_fifo_data(&fifoframe, &bmi270.sensor);
        if (BMI2_OK != result)
        {
            printf(" Error: Failed to read FIFO data\n");
            return result;
        }

        /* Read FIFO data on interrupt. */
        result = bmi2_get_int_status(&int_status, &bmi270.sensor);
        if (BMI2_OK != result)
        {
            printf(" Error: Failed to get interrupt status\n");
            return result;
        }

        /* Skip frames if needed */
        if (imu_skip_frames > 0)
        {
            imu_skip_frames--;
            Cy_GPIO_Write(CYBSP_USER_LED_PORT, CYBSP_USER_LED_PIN, CYBSP_LED_STATE_OFF);
            return result;
        }

#ifdef IMU_ACC
        /* Parse the FIFO data to extract accelerometer data from the FIFO buffer. */
        accel_frame_length = BMI2_FIFO_ACCEL_FRAME_COUNT;
        (void)bmi2_extract_accel(fifo_accel_data, &accel_frame_length, &fifoframe, &bmi270.sensor);
        if (BMI2_FIFO_ACCEL_FRAME_COUNT != accel_frame_length)
        {
            printf("Error: Expected number of Accelerometer frames not received\n");
            result = IMU_FIFO_INSUFFICIENT_DATA;
            return result;
        }
#endif /* IMU_ACC */
#ifdef IMU_GYR
        /* Parse FIFO data to extract gyroscope data from the FIFO buffer. */
        gyro_frame_length = BMI2_FIFO_GYRO_FRAME_COUNT;
        (void)bmi2_extract_gyro(fifo_gyro_data, &gyro_frame_length, &fifoframe, &bmi270.sensor);
        if (BMI2_FIFO_GYRO_FRAME_COUNT != gyro_frame_length)
        {
            printf("Error: Expected number of Gyroscope frames not received\r\n");
            result = IMU_FIFO_INSUFFICIENT_DATA;
            return result;
        }
#endif /* IMU_GYR */

#ifdef IMU_ACC
        for (index = 0; index < accel_frame_length; index++)
#else
        for (index = 0; index < gyro_frame_length; index++)
#endif /* IMU_ACC */
        {

#ifdef USE_SENSOR_REMAPPING
#ifdef IMU_ACC
            /* Remapping the accelerometer data based on orientation of sensor */
            imu_remap_sensor_orientation(&(fifo_accel_data[index]));
#endif
#ifdef IMU_GYR
            /* Remapping the gyroscope data based on orientation of sensor */
            imu_remap_sensor_orientation(&(fifo_gyro_data[index]));
#endif
#endif
            /* Store the IMU values in a buffer. Change the orientation to
             * match training data.
             */
            float imu_buffer[IMU_NUM_AXIS] =
            {
#ifdef IMU_ACC
                fifo_accel_data[index].x / (float)IMU_ACC_DIVIDE,
                fifo_accel_data[index].y / (float)IMU_ACC_DIVIDE,
                fifo_accel_data[index].z / (float)IMU_ACC_DIVIDE,
#endif /* IMU_ACC */
#ifdef IMU_GYR
                fifo_gyro_data[index].x / (float)IMU_GYRO_DIVIDE,
                fifo_gyro_data[index].y / (float)IMU_GYRO_DIVIDE,
                fifo_gyro_data[index].z / (float)IMU_GYRO_DIVIDE,
#endif /* IMU_GYR */
            };

            /* Feed the DEEPCRAFT pre-processor */
            result = IMAI_enqueue(imu_buffer);
            CY_ASSERT(IMAI_RET_SUCCESS == result);

            /* Reset the scores and best label values. */
            best_label = 0;
            max_score = -1000.0f;

            /* Check if there is any model output */
            switch(IMAI_dequeue(label_scores))
            {
                /* On success, print the labels with their scores */
                case IMAI_RET_SUCCESS:
                {
                    for(int i = 0; i < IMAI_DATA_OUT_COUNT; i++)
                    {
                        printf("label: %-10s: score: %f\r\n", label_text[i],
                               label_scores[i]);
                        if (label_scores[i] > max_score)
                        {
                            max_score = label_scores[i];
                            best_label = i;
                        }
                    }
                    printf("\r\n");
                    printf("Output: %-30s\r\n", label_text[best_label]);
                    break;
                }

                /* No new output, continue with sampling */
                case IMAI_RET_NODATA:
                {
                    break;
                }

                /* Abort on error */
                case IMAI_RET_ERROR:
                {
                    printf("Error: IMAI_dequeue failed !\r\n");
                    CY_ASSERT(0);
                    break;
                }
            }
        }

        /* Turn user led off.*/
        Cy_GPIO_Write(CYBSP_USER_LED_PORT, CYBSP_USER_LED_PIN, CYBSP_LED_STATE_OFF);
    }

    return result;
}

/*******************************************************************************
* Function Name: imu_remap_sensor_orientation
********************************************************************************
* Summary:
* Remapping the sensor data to match with CY8CKIT-062S2-AI orientation.
*
* Parameters:
* data: pointer IMU Sensors data
*
* Return:
* None
*
*******************************************************************************/
void imu_remap_sensor_orientation(struct bmi2_sens_axes_data *data)
{
    /* remapping the data */
    data->x = -(data->x);
    data->y = -(data->y);
}

/* [] END OF FILE */
