/*******************************************************************************
* File Name        : main.c
*
* Description      : This source file contains the main routine for CM55 CPU
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

/*******************************************************************************
* Header Files
*******************************************************************************/
#include "cybsp.h"
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdio.h>
/* RTOS header file */
#include <cyabs_rtos.h>
#include <FreeRTOS.h>
#include <task.h>
#include "protocol/protocol.h"
#include "build.h"
#include "common.h"
#include "clock.h"
#include "board.h"
#include "system.h"
#include "wifi_configs/im_config.h"
#include "initd.h"
#include "usbd.h"
#include "retarget_io_init.h"

#ifdef IM_ENABLE_SHELL
#include "shell/debug_uart.h"
#endif

/*******************************************************************************
* Function Name: main
********************************************************************************
* Summary:
* This is the main function for CM55 application.It also initializes the 
* the device and board peripherals, retarget-io middleware, clock and streaming interface.
* 
* CM33 application enables the CM55 CPU and then the CM33 CPU enters 
* deep sleep.
* 
* Parameters:
*  void
*
* Return:
*  int
*
*******************************************************************************/
int main(void)
{

    /* Initialize the device and board peripherals */
    board_init_system();

    /* Initialize retarget-io middleware */
    init_retarget_io();

    /* Start clock */
    if(!clock_init())
    {
        halt_error(LED_CODE_CLOCK_ERROR);
    }

   #ifdef IM_ENABLE_SHELL
    /* Init debug UART, continue if failed. */ 
      debug_uart_init();   
   #else
   #endif
   
    /* Start task with pid 1, this is the parent for all tasks */
    xTaskCreate(
        initd_task,
        "initd",
        INITD_STACK_SIZE,
        NULL,
        INITD_PRIO,
        NULL);

    /* Start the FreeRTOS scheduler. */
    vTaskStartScheduler();

    /* Should never get here. */
    return 0;
    
}

/* [] END OF FILE */
