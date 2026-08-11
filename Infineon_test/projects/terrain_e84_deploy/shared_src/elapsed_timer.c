/******************************************************************************
* File Name:   elapsed_timer.c
*
* Description: This file contains the implementation of a elapsed timer.
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
#include <stdio.h>

#include "cybsp.h"
#include "cy_pdl.h"

#include "elapsed_timer.h"

/*******************************************************************************
* Constants
*******************************************************************************/
#define SYSTICK_MAX_CNT (0xFFFFFF)
#define RESET_VAL       (0u)

/*******************************************************************************
* Global Variables
*******************************************************************************/
/* System Tick overflow counter */
static volatile uint64_t elapsed_timer_ov = RESET_VAL;

/*******************************************************************************
* Function Name: elapsed_timer_callback
********************************************************************************
* Summary:
* This is the callback implementation for the elapsed timer. It increments an
* internal counter;
*
* Paramters:
*   void
*
* Return:
*   void
*
*******************************************************************************/
static void elapsed_timer_callback(void)
{
    elapsed_timer_ov++;  
}

/*******************************************************************************
* Function Name: elapsed_timer_init
********************************************************************************
* Summary:
*   Initialize the elapsed system tick timer.
*
* Parameters:
*   void
*
* Return:
*   cy_rslt_t: the status of the initialization.
*
*******************************************************************************/
cy_rslt_t elapsed_timer_init(void)
{
    /* Initialize the System Tick */
    Cy_SysTick_Init(CY_SYSTICK_CLOCK_SOURCE_CLK_CPU, SYSTICK_MAX_CNT);
    Cy_SysTick_SetCallback(0, elapsed_timer_callback);

    elapsed_timer_ov = RESET_VAL;

    return CY_RSLT_SUCCESS;
}

/*******************************************************************************
* Function Name: elapsed_timer_get_tick
********************************************************************************
* Summary:
*   Return the current tick (number of CPU cycles) since the timer was started.
*
* Parameters:
*   tick: current number of ticks.
*
* Return:
*   int: the status of the operation.
*
*******************************************************************************/
int elapsed_timer_get_tick(uint64_t *tick)
{
    uint32_t int_status;
    int_status = Cy_SysLib_EnterCriticalSection();

    uint64_t systick_value = (uint64_t)Cy_SysTick_GetValue();
    if (systick_value > (3*SYSTICK_MAX_CNT/4))
    {
        if (Cy_SysTick_GetCountFlag())
        {
            elapsed_timer_ov++;
        }
    }

    *tick = (SYSTICK_MAX_CNT - systick_value) + (elapsed_timer_ov * (SYSTICK_MAX_CNT + 1));

    Cy_SysLib_ExitCriticalSection(int_status);
    return 0;
}

