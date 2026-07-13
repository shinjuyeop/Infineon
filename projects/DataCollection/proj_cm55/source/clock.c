/******************************************************************************
* File Name:   clock.c
*
* Description: This file provides a clock.
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

#include "cybsp.h"
#include "clock.h"

/*******************************************************************************
* Static Variables
*******************************************************************************/
static uint32_t uptime_seconds = 0;

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/*******************************************************************************
* Function Name: clock_interrupt_handler
********************************************************************************
* Summary:
*   Keeps track of total elapsed seconds.
*
* Parameters:
*   None
*******************************************************************************/
void clock_interrupt_handler(void)
{
    uptime_seconds++;
    uint32_t interrupts = Cy_TCPWM_GetInterruptStatusMasked(CYBSP_GENERAL_PURPOSE_TIMER_HW,
                                                            CYBSP_GENERAL_PURPOSE_TIMER_NUM);
    /*Clear the interrupt*/ 
    Cy_TCPWM_ClearInterrupt(CYBSP_GENERAL_PURPOSE_TIMER_HW, CYBSP_GENERAL_PURPOSE_TIMER_NUM, interrupts);
}

/*******************************************************************************
* Function Name: clock_get_tick
********************************************************************************
* Summary:
*   Returns the number of clock ticks elapsed since initialization.
*
* Return:
*   The number of clock ticks elapsed.
*
*******************************************************************************/
clock_tick_t clock_get_tick(void)
{
    /*Ensure it can be reread in case of a corner case.*/ 
    volatile uint32_t* p_sec = (uint32_t*)&uptime_seconds;

    uint32_t seconds_0 = *p_sec;
    uint32_t time = Cy_TCPWM_Counter_GetCounter(CYBSP_GENERAL_PURPOSE_TIMER_HW, CYBSP_GENERAL_PURPOSE_TIMER_NUM);
    uint32_t seconds_1 = *p_sec;

    if (seconds_0 == seconds_1) {
        return ((clock_tick_t)CLOCK_TICK_PER_SECOND * seconds_1) + (clock_tick_t)time;
    }

    /*If the latter differ from the first, then assume time is zero.*/ 
    return ((clock_tick_t)CLOCK_TICK_PER_SECOND * seconds_1);
}

/*******************************************************************************
* Function Name: clock_init
********************************************************************************
* Summary:
*   Initializes the clock.
*
* Return:
*   True if initialization is successful, otherwise false.
*
*******************************************************************************/
bool clock_init(void)
{

    cy_en_tcpwm_status_t status;
    Cy_TCPWM_GetInterruptStatusMasked(CYBSP_GENERAL_PURPOSE_TIMER_HW,
                                                            CYBSP_GENERAL_PURPOSE_TIMER_NUM);
    /*Initialize and enable TCPWM counter*/ 
    status = Cy_TCPWM_Counter_Init(CYBSP_GENERAL_PURPOSE_TIMER_HW,CYBSP_GENERAL_PURPOSE_TIMER_NUM, &CYBSP_GENERAL_PURPOSE_TIMER_config);
    if (status != CY_TCPWM_SUCCESS)
    {
        return false;
    }

    /*Assign the interrupt handler*/ 
    cy_stc_sysint_t irq_cfg = {
        .intrSrc = CYBSP_GENERAL_PURPOSE_TIMER_IRQ,
        .intrPriority = CLOCK_INTERRUPT_PRIORITY,
    };

    if (Cy_SysInt_Init(&irq_cfg, clock_interrupt_handler) != CY_SYSINT_SUCCESS)
    {
        return false;
    }

    NVIC_EnableIRQ(CYBSP_GENERAL_PURPOSE_TIMER_IRQ);

    /* Enable the initialized counter */
    Cy_TCPWM_Counter_Enable(CYBSP_GENERAL_PURPOSE_TIMER_HW,
            CYBSP_GENERAL_PURPOSE_TIMER_NUM);

    /* Start the counter */
    Cy_TCPWM_TriggerStart_Single(CYBSP_GENERAL_PURPOSE_TIMER_HW,
            CYBSP_GENERAL_PURPOSE_TIMER_NUM);
            
    return true;
}
/* [] END OF FILE */
