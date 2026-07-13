/******************************************************************************
* File Name: common.c
*
* Description:
*   This file contains common utility functions used across the project.
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

#include <cy_result.h>
#include <cy_utils.h>
#include <cyabs_rtos.h>
#include <cybsp.h>
#include <stdarg.h>
#include <stdio.h>
#include "common.h"

#ifdef IM_ENABLE_SHELL
#include "shell/debug_uart.h"
#else
#define printk printf
#define vprintk vprintf
#endif /* IM_ENABLE_SHELL */

/*******************************************************************************
 * Global Variables
 ******************************************************************************/
/* The current log level */
static log_level_t current_log_level = LOG_LEVEL_INFO;
static bool logger_initialized = false;
static cy_mutex_t mutex;

/*******************************************************************************
* Function Definitions
*******************************************************************************/
/*******************************************************************************
* Function Name: log_set_level
********************************************************************************
* Summary:
*   Sets the current log level.
*
* Parameters:
*   level: The log level to set.
*
*******************************************************************************/
void log_set_level(log_level_t level) 
{
    current_log_level = level;
}

/*******************************************************************************
* Function Name: log_message
********************************************************************************
* Summary:
*   Logs a message with the specified log level and component.
*
* Parameters:
*   level: The log level of the message.
*   component: The component generating the log message.
*   format: The format string for the log message.
*   ...: Additional arguments for the format string.
*
*******************************************************************************/
void log_message(log_level_t level, const char* component, const char* format, ...) 
{
    const char* level_str;

    if (level > current_log_level) 
    {
        return;
    }

    if (!logger_initialized) 
    {
        if (__get_IPSR() != 0)
            return;
        logger_initialized = true;
        printk("\033[97m%s %-9s: %s\033[0m", "[INFO] ", "syslog", "System log started\n");
        cy_rtos_init_mutex(&mutex);
    }

    switch (level) 
    {
        case LOG_LEVEL_INFO:
            level_str = "\033[97m[INFO] ";
            break;
        case LOG_LEVEL_ERROR:
            level_str = "\033[91m[ERROR]";
            break;
        case LOG_LEVEL_DEBUG:
            level_str = "\033[96m[DEBUG]";
            break;
        default:
            level_str = "\033[91m[UNKNOWN]";
            break;
    }

    cy_rtos_get_mutex(&mutex, 1000);
    va_list args;
    va_start(args, format);

    printk("%s %-9s: ", level_str, component);
    vprintk(format, args);
    printk("\033[0m\n");

    va_end(args);

    cy_rtos_set_mutex(&mutex);
}

/*******************************************************************************
* Function Name: set_led0
********************************************************************************
* Summary:
*   Sets the status LED.
*
* Parameters:
*   state: The desired state of the LED. 'true' for ON, 'false' for OFF.
*
*******************************************************************************/
void set_led0(bool state)
{
   Cy_GPIO_Write(CYBSP_USER_LED_PORT, CYBSP_USER_LED_PIN, state ? CYBSP_LED_STATE_ON : CYBSP_LED_STATE_OFF );
}

/*******************************************************************************
* Function Name: set_led1
********************************************************************************
* Summary:
*   Sets the status LED.
*
* Parameters:
*   state: The desired state of the LED. 'true' for ON, 'false' for OFF.
*
*******************************************************************************/
void set_led1(bool state)
{
    Cy_GPIO_Write(CYBSP_USER_LED2_PORT, CYBSP_USER_LED2_PIN, state ? CYBSP_LED_STATE_ON : CYBSP_LED_STATE_OFF);
}

/*******************************************************************************
* Function Name: halt_error
********************************************************************************
* Summary:
*  Flash the given error code indefinitely. This function never returns.
*
*  Blink pattern:
*  0:  .        10: ..-      20: .--.
*  1:  -        11: -.-      21: ---.
*  2:  ..       12: .--      22: ...-
*  3:  -.       13: ---      23: -..-
*  4:  .-       14: ....     24: .-.-
*  5:  --       15: -...     25: --.-
*  6:  ...      16: .-..     26: ..--
*  7:  -..      17: --..     27: -.--
*  8:  .-.      18: ..-.     28: .---
*  9:  --.      19: -.-.     29: ----
*******************************************************************************/
void halt_error(int code)
{
    __disable_irq();

    if(code < 0)
    {
        code = -code;
    }

    do {
            int i = code + 2;
            do {
                set_led0(true);
                set_led1(true);
                Cy_SysLib_Delay((i & 1) ? LED_LONG_ON_TIME : LED_SHORT_ON_TIME);
                set_led0(false);
                set_led1(false);
                Cy_SysLib_Delay(LED_OFF_TIME);
                i >>= 1;
            } while(i > 1);

            Cy_SysLib_Delay(LED_CODE_SEPARATOR_TIME);

    } while(true);
}

/* [] END OF FILE */
