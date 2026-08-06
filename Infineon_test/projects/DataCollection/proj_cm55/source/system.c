/*******************************************************************************
* File Name        : system.c
*
* Description      : This source file contains the system configurations and initializations
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

#include <stdio.h>
#include <cybsp.h>

#include "common.h"
#include "system.h"
#include "mtb_hal_spi.h"
#include "protocol/protocol.h"

#include "devices/dev_bmi270.h"
#include "devices/dev_pdm_pcm.h"
#include "devices/dev_bgt60trxx.h"
#include "devices/dev_dps368.h"
#include "devices/dev_sht4x.h"
#include "devices/dev_bmm350.h"
#include "devices/dev_amic.h"


/******************************************************************************
 * Macros
 ******************************************************************************/
#ifdef IM_ENABLE_BMM350
/*I3C*/
#define CHECK_ZERO                         (0UL)
#define I3C_CONTROLLER_ERROR_EVENT         (CY_I3C_CRC_ERROR_EVENT |  CY_I3C_PARITY_ERROR_EVENT | \
        CY_I3C_FRAME_ERROR_EVENT | CY_I3C_BROADCAST_ADDR_NACK_ERROR_EVENT | \
        CY_I3C_ADDR_NACK_ERROR_EVENT | CY_I3C_BUFFER_OVERFLOW_ERROR_EVENT | \
        CY_I3C_I2C_TGT_WDATA_NACK_ERROR_EVENT | CY_I3C_CONTROLLER_ERROR_CE0_EVENT)
#endif

/*SPI*/
#define SPI_INTR_NUM            ((IRQn_Type) RADAR_SPI_CONTROLLER_IRQ)
#define SPI_INTR_PRIORITY       (2U)

/*******************************************************************************
* Global Variables
*******************************************************************************/
/* I2C object */
static mtb_hal_i2c_t CYBSP_I2C_CONTROLLER_hal_obj;

/* I2C context */
cy_stc_scb_i2c_context_t CYBSP_I2C_CONTROLLER_context;

#ifdef USE_KIT_PSE84_HMI
/* I2C object for SHT40 sensor on KIT_PSE84_HMI*/
static mtb_hal_i2c_t CYBSP_I2C_SHT40_CONTROLLER_hal_obj;

/* I2C context */
cy_stc_scb_i2c_context_t CYBSP_I2C_SHT40_CONTROLLER_context;
#endif

/* SPI context */
cy_stc_scb_spi_context_t SPI_context;

#ifdef IM_ENABLE_BMM350
/* I3C context */
cy_stc_i3c_context_t CYBSP_I3C_CONTROLLER_context;
#endif

/*******************************************************************************
* Function Prototypes
*******************************************************************************/
static bool _init_i2c(void);

static bool _init_spi(void);

void SPI_Interrupt(void);

#ifdef IM_ENABLE_BMM350
static bool _init_i3c(void);

/*******************************************************************************
* Function name: I3C_handle_error
*******************************************************************************
* Summary:
*   I3C_handle_error
* Parameters:
*
*
* Return:
*
*******************************************************************************/
static void I3C_handle_error(void)
{
    /* Disable all interrupts. */
    __disable_irq();

    /* Infinite loop. */
    while (1u)
    {
    }
}

/*******************************************************************************
* Function Name: HandleEvents
********************************************************************************
* Summary:
*   I3C events
* Parameters:
*   event: I3C events
*
* Return:
*
*******************************************************************************/
static void HandleEvents(uint32_t event)
{
    if (CHECK_ZERO != (CY_I3C_CONTROLLER_WR_CMPLT_EVENT & event))
    {
        //Write complete event handling
    }
    else if (CHECK_ZERO != (CY_I3C_CONTROLLER_RD_CMPLT_EVENT & event))
    {
        //Read complete event handling
    }
    else if (CHECK_ZERO != (CY_I3C_XFER_ABORTED_ERROR_EVENT & event))
    {
        printf("Transfer aborted\n\r");
    }
    else if (CHECK_ZERO != (I3C_CONTROLLER_ERROR_EVENT & event))
    {
        printf("Error detected in the response packet\n\r");
        I3C_handle_error();
    }
}

/*******************************************************************************
* Function Name: I3C_Interrupt
********************************************************************************
* Summary:
*   I3C interrupts
* Parameters:
*
* Return:
*
*******************************************************************************/
void I3C_Interrupt(void)
{
    Cy_I3C_Interrupt(CYBSP_I3C_CONTROLLER_HW, &CYBSP_I3C_CONTROLLER_context);
}
#endif

/*******************************************************************************
* Function Name: SPI_Interrupt
********************************************************************************
* Summary:
*   SPI_Interrupt
* Parameters:
*
* Return:
*
*******************************************************************************/
void SPI_Interrupt(void)
{
    Cy_SCB_SPI_Interrupt(RADAR_SPI_CONTROLLER_HW, &SPI_context);
}

#ifdef IM_ENABLE_BMM350
/*******************************************************************************
* Function Name: _init_i3c
********************************************************************************
* Summary:
*   Initializes the I3C interface.
*
* Parameters:
*   i3c: Pointer to the I3C object.
*
* Return:
*   True if initialization is successful, otherwise false.
*******************************************************************************/
static bool _init_i3c(void)
{
    cy_en_i3c_status_t initStatus;
    cy_en_sysint_status_t sysStatus;
    cy_stc_sysint_t CYBSP_I3C_CONTROLLER_IRQ_cfg =
    {
        .intrSrc = CYBSP_I3C_CONTROLLER_IRQ,
        .intrPriority = 2UL
    };

    initStatus = Cy_I3C_Init(CYBSP_I3C_CONTROLLER_HW, &CYBSP_I3C_CONTROLLER_config, &CYBSP_I3C_CONTROLLER_context);
    if(CY_I3C_SUCCESS != initStatus)
    {
        return false;
    }
    /* Hook interrupt service routine. */
    sysStatus = Cy_SysInt_Init(&CYBSP_I3C_CONTROLLER_IRQ_cfg, &I3C_Interrupt);
    if(CY_SYSINT_SUCCESS != sysStatus)
    {
        return false;
    }
    /* Enable interrupt in NVIC. */
    NVIC_EnableIRQ((IRQn_Type) CYBSP_I3C_CONTROLLER_IRQ_cfg.intrSrc);
    Cy_I3C_RegisterEventCallback(CYBSP_I3C_CONTROLLER_HW, (cy_cb_i3c_handle_events_t) HandleEvents, &CYBSP_I3C_CONTROLLER_context);
    Cy_I3C_Enable(CYBSP_I3C_CONTROLLER_HW, &CYBSP_I3C_CONTROLLER_context);

    return true;
}
#endif

/*******************************************************************************
* Function Name: _init_i2c
********************************************************************************
* Summary:
*   Initializes the I2C interface.
*
* Parameters:
*   i2c: Pointer to the I2C object.
*
* Return:
*   True if initialization is successful, otherwise false.
*******************************************************************************/
static bool _init_i2c(void)
{
    cy_rslt_t result;

    /* Initialize the I2C master interface for BMI270 motion sensor */
    result = Cy_SCB_I2C_Init(CYBSP_I2C_CONTROLLER_HW,
            &CYBSP_I2C_CONTROLLER_config,
            &CYBSP_I2C_CONTROLLER_context);

    if(CY_RSLT_SUCCESS != result)
    {
        return false;
    }
    Cy_SCB_I2C_Enable(CYBSP_I2C_CONTROLLER_HW);

    /* Configure the I2C master interface with the desired clock frequency */
    result = mtb_hal_i2c_setup(&CYBSP_I2C_CONTROLLER_hal_obj,
            &CYBSP_I2C_CONTROLLER_hal_config,
            &CYBSP_I2C_CONTROLLER_context,
            NULL);

    if(CY_RSLT_SUCCESS != result)
    {
        return false;
    }

#ifdef USE_KIT_PSE84_HMI
    /* Initialize the I2C master interface for SHT40 sensor on KIT_PSE84_HMI */
    result = Cy_SCB_I2C_Init(CYBSP_I2C_SHT40_CONTROLLER_HW,
            &CYBSP_I2C_SHT40_CONTROLLER_config,
            &CYBSP_I2C_SHT40_CONTROLLER_context);

    if(CY_RSLT_SUCCESS != result)
    {
        return false;
    }
    Cy_SCB_I2C_Enable(CYBSP_I2C_SHT40_CONTROLLER_HW);

    /* Configure the I2C master interface with the desired clock frequency */
    result = mtb_hal_i2c_setup(&CYBSP_I2C_SHT40_CONTROLLER_hal_obj,
            &CYBSP_I2C_SHT40_CONTROLLER_hal_config,
            &CYBSP_I2C_SHT40_CONTROLLER_context,
            NULL);

    if(CY_RSLT_SUCCESS != result)
    {
        return false;
    }
#endif

    return true;
}

/*******************************************************************************
* Function Name: _init_spi
********************************************************************************
* Summary:
*   Initializes the SPI interface.
*
* Parameters:
*   spi: Pointer to the SPI object.
*
* Return:
*   True if initialization is successful, otherwise false.
*******************************************************************************/
static bool _init_spi(void)
{
    cy_en_scb_spi_status_t init_status;

    init_status = Cy_SCB_SPI_Init(RADAR_SPI_CONTROLLER_HW, &RADAR_SPI_CONTROLLER_config, &SPI_context);
    /* If the initialization fails, update status */
    if ( CY_SCB_SPI_SUCCESS != init_status )
    {
        return false;
    }
    else
    {
        cy_stc_sysint_t spiIntrConfig =
        {
            .intrSrc      = SPI_INTR_NUM,
            .intrPriority = SPI_INTR_PRIORITY,
        };

        Cy_SysInt_Init(&spiIntrConfig, &SPI_Interrupt);
        NVIC_EnableIRQ(SPI_INTR_NUM);

        /* Set active target select to line 0 */
        Cy_SCB_SPI_SetActiveSlaveSelect(RADAR_SPI_CONTROLLER_HW, RADAR_SPI_SLAVE_SELECT);
        /* Enable SPI Controller block. */
        Cy_SCB_SPI_Enable(RADAR_SPI_CONTROLLER_HW);
    }

    return true;
}

/*******************************************************************************
* Function Name: system_load_device_drivers
********************************************************************************
* Summary:
*   Load all device drivers
*
* Parameters:
*   protocol: Pointer to the protocol_t instance.
*
*******************************************************************************/
void system_load_device_drivers(protocol_t* protocol)
{

    /* Enable I2C */
    if(!_init_i2c())
    {
        halt_error(LED_CODE_I2C_ERROR);
    }

#ifdef IM_ENABLE_BMM350
    /* Enable i3c */
    if(!_init_i3c())
    {
        halt_error(LED_CODE_I3C_ERROR);
    }
#endif

    /* Enable SPI (if available)*/
    if(!_init_spi())
    {
        halt_error(LED_CODE_SPI_ERROR);
    }


#ifdef IM_ENABLE_PDM_PCM
    if(!dev_pdm_pcm_register(protocol)) {
        printf("PDM PCM registration failed.\n");
        halt_error(LED_CODE_SENSOR_PDM_PCM_ERROR);
    }
#endif

#ifdef IM_ENABLE_AMIC
    if(!dev_amic_register(protocol)) {
        printf("AMIC registration failed.\n");
        halt_error(LED_CODE_SENSOR_AMIC_ERROR);
    }
#endif

#ifdef IM_ENABLE_BMI270
    if(!dev_bmi270_register(protocol, &CYBSP_I2C_CONTROLLER_hal_obj)) {
        printf("BMI270 registration failed.\n");
        halt_error(LED_CODE_SENSOR_BMI270_ERROR);
    }
#endif

#ifdef IM_ENABLE_DPS368
    if(!dev_dps368_register(protocol, &CYBSP_I2C_CONTROLLER_hal_obj)) {
        printf("DPS368 registration failed.\n");
          halt_error(LED_CODE_SENSOR_DPS368_ERROR);
      }
#endif

#ifdef IM_ENABLE_SHT4X
#ifdef USE_KIT_PSE84_HMI
    if(!dev_sht4x_register(protocol, &CYBSP_I2C_SHT40_CONTROLLER_hal_obj)) {
        printf("SHT4X registration failed.\n");
          halt_error(LED_CODE_SENSOR_SHT4X_ERROR);
      }
#else
    if(!dev_sht4x_register(protocol, &CYBSP_I2C_CONTROLLER_hal_obj)) {
        printf("SHT4X registration failed.\n");
          halt_error(LED_CODE_SENSOR_SHT4X_ERROR);
      }
#endif
#endif

#ifdef IM_ENABLE_BGT60TRXX
      if(!dev_bgt60trxx_register(protocol)) {
          printf("BGT60TRXX registration failed.\n");
          halt_error(LED_CODE_SENSOR_BGT60TRXX_ERROR);
      }
#endif

#ifdef IM_ENABLE_BMM350
    if(!dev_bmm350_register(protocol,CYBSP_I3C_CONTROLLER_HW,&CYBSP_I3C_CONTROLLER_context)) {
        printf("BMM350 registration failed.\n");
        halt_error(LED_CODE_SENSOR_BMM350_ERROR);
    }
#endif

}

/* [] END OF FILE */
