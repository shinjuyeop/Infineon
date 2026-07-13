/******************************************************************************
* File Name: debug_uart.c
*
* Description: This file provides the implementation for the debug UART interface.
*
* Related Document: See README.md
*
*******************************************************************************
* (c) 2026, Infineon Technologies AG, or an affiliate of Infineon
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

#ifdef IM_ENABLE_SHELL

#include <stdbool.h>
#include <cybsp.h>
#include <stdio.h>
#include <cyabs_rtos.h>
#include <stdarg.h>
#include <stream_buffer.h>
#include <cy_sysint.h>
#include "debug_uart.h"
#include "im_config.h"
#include "terminal.h"

/*******************************************************************************
 * Global Variables
 ******************************************************************************/
static cy_stc_scb_uart_context_t uart_context;
static mtb_hal_uart_t uart;
static terminal_t* input_terminal;
static volatile bool is_uart_initialized = false;

/******************************************************************************
* Function Name: uart_rx_isr
********************************************************************************
* Summary:
*   UART RX interrupt service routine.
*
* Parameters:
*   None
*
* Return:
*   None
*
*******************************************************************************/
static void uart_rx_isr(void)
{
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    
    /* Check if RX FIFO is not empty */
    if (Cy_SCB_UART_GetNumInRxFifo(CYBSP_DEBUG_UART_HW) > 0) 
    {
        /* Read one character */
        uint8_t ch = (uint8_t)Cy_SCB_UART_Get(CYBSP_DEBUG_UART_HW);
        
        /* Append to terminal input buffer */
        if (input_terminal != NULL) 
        {
            terminal_append_input_isr(input_terminal, &ch, 1, &xHigherPriorityTaskWoken);
        }
    }
    
    /* Clear the interrupt */
    Cy_SCB_ClearRxInterrupt(CYBSP_DEBUG_UART_HW, CY_SCB_RX_INTR_NOT_EMPTY);
    
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

/******************************************************************************
* Function Name: debug_start_input_stream
********************************************************************************
* Summary:
*   Starts the debug UART input stream.
*
* Parameters:
*   input: Pointer to the terminal structure.
*
* Return:
*   None
*
*******************************************************************************/
void debug_start_input_stream(terminal_t* input) 
{
    if (!is_uart_initialized) 
    {
        printk("ERROR: UART not initialized!\n");
        return;
    }
    
    input_terminal = input;
    /* Clear any pending RX data */
    Cy_SCB_UART_ClearRxFifo(CYBSP_DEBUG_UART_HW);

    /* Enable RX NOT EMPTY interrupt in the SCB */
    Cy_SCB_SetRxInterruptMask(CYBSP_DEBUG_UART_HW, CY_SCB_RX_INTR_NOT_EMPTY);
    
    /* Set up the NVIC interrupt */
    cy_stc_sysint_t uart_intr_config = {
        .intrSrc = CYBSP_DEBUG_UART_IRQ,
        .intrPriority = 3
    };
     
    Cy_SysInt_Init(&uart_intr_config, uart_rx_isr);
    NVIC_EnableIRQ(CYBSP_DEBUG_UART_IRQ);
     
}

/******************************************************************************
* Function Name: debug_uart_init
********************************************************************************
* Summary:
*   Initializes the debug UART interface.
*
* Parameters:
*   None
*
* Return:
*   true if initialization was successful, false otherwise.
*
*******************************************************************************/
bool debug_uart_init(void) 
{
    cy_rslt_t result;

    /* Initialize the UART Block */
    result = (cy_rslt_t)Cy_SCB_UART_Init(CYBSP_DEBUG_UART_HW, &CYBSP_DEBUG_UART_config, &uart_context);

    /* Enables the SCB block for the UART operation*/
    Cy_SCB_UART_Enable(CYBSP_DEBUG_UART_HW);

    /* Set up the HAL UART */
    result = mtb_hal_uart_setup(&uart, &CYBSP_DEBUG_UART_hal_config, &uart_context, NULL);
 
    result = mtb_hal_uart_enable(&uart, true);

    if (result != CY_RSLT_SUCCESS)
    {
        return false;
    }

    result = mtb_hal_uart_set_baud(&uart, 115200, NULL);
    
    if (result != CY_RSLT_SUCCESS)
    {
        return false;
    }

    is_uart_initialized = true;

    printk("\x1b[2J\x1b[;H");

    return true;
}

/******************************************************************************
* Function Name: debug_uart_write
********************************************************************************
* Summary:
*   Writes data to the debug UART interface.
*
* Parameters:
*   ptr: Pointer to the data buffer.
*   size: Number of bytes to write.
*
* Return:
*   Number of bytes successfully written.
*
*******************************************************************************/
size_t debug_uart_write(const char* ptr, size_t size) 
{
    cy_rslt_t rslt;
    size_t count;

    if (ptr == NULL)
    {
        count = 0;
        return count;
    }

    if (!is_uart_initialized)
    {
        count = 0;
        return count;
    }

    for (count = 0; count < size; count++) 
    {
        if (*ptr == '\n') 
        {
            rslt = mtb_hal_uart_put (&uart, '\r');
            if (rslt != CY_RSLT_SUCCESS)
            {
                break;
            }

        }

        rslt = mtb_hal_uart_put (&uart, (uint32_t)*ptr);
        if (rslt != CY_RSLT_SUCCESS)
        {
            break;
        }

        ptr++;
    }

    return count;
}

/******************************************************************************
* Function Name: printk
********************************************************************************
* Summary:
*   Writes formatted data to the debug UART interface.
*
* Parameters:
*   format: Format string.
*   ...: Additional arguments.
*
* Return:
*   Number of characters written.
*
*******************************************************************************/
int printk(const char* restrict format, ...) 
{
    va_list args;
    char line_buffer[512];
    int count;

    va_start(args, format);
    count = vsnprintf(line_buffer, sizeof(line_buffer), format, args);
    va_end(args);

    debug_uart_write(line_buffer, count);

    return count;
}

/******************************************************************************
* Function Name: vprintk
********************************************************************************
* Summary:
*   Writes formatted data to the debug UART interface using a variable argument list.
*
* Parameters:
*   format: Format string.
*   args: Variable argument list.
*
* Return:
*   Number of characters written.
*
*******************************************************************************/
int vprintk(const char* restrict format, va_list args) 
{
    char line_buffer[512];
    int count;

    count = vsnprintf(line_buffer, sizeof(line_buffer), format, args);
    debug_uart_write(line_buffer, count);

    return count;
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */