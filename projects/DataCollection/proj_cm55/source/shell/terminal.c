/******************************************************************************
* File Name: terminal.c
*
* Description: Implementation of terminal management for the shell environment.
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

#include <FreeRTOS.h>
#include <stream_buffer.h>
#include <semphr.h>
#include <stdlib.h>
#include <string.h>

#include "terminal.h"
#include "../../wifi_configs/im_config.h"
#include "../../common.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: terminal_init
********************************************************************************
* Summary:
*   Initializes the given terminal handle and creates the input stream.
*
* Parameters:
*   terminal - Pointer to the terminal structure to initialize.
*   write - Function pointer for writing to the terminal.
*
* Return:
*   None
*
*******************************************************************************/
void terminal_init(terminal_t* terminal, terminal_write_fn write) 
{
    if (terminal == NULL || write == NULL) 
    {
        return;
    }

    terminal->tag = NULL;
    terminal->attr = TERMINAL_ATTR_COLOR;
    /* Default terminal width */
    terminal->width = 80;  
    /* Default terminal height */
    terminal->height = 24; 
    /* 1 byte trigger level */
    terminal->input = xStreamBufferCreate(MAX_RX_LEN, 1);   
    terminal->write = write;
    terminal->last_input = 0;
    terminal->read_mutex = xSemaphoreCreateMutex();
}

/******************************************************************************
* Function Name: terminal_read
********************************************************************************
* Summary:
*   Reads data from the terminal's input stream buffer with a timeout.
*
* Parameters:
*   terminal - Pointer to the terminal structure.
*   buf - Buffer to store the read data.
*   size - Number of bytes to read.
*   timeout_ms - Maximum time in milliseconds to wait for data.
*                Use 0 for no wait, and -1 for an indefinite wait.
* Return:
*   The number of bytes read.
*
*******************************************************************************/
size_t terminal_read(terminal_t* terminal, void* buf, size_t size, int timeout_ms) 
{
    if (terminal == NULL || buf == NULL || size == 0) 
    {
        return 0;
    }

    TickType_t timeout_ticks;
    if (timeout_ms == 0) 
    {
        timeout_ticks = 0;
    }
    else if (timeout_ms == -1) 
    {
        timeout_ticks = portMAX_DELAY;
    }
    else {
        timeout_ticks = pdMS_TO_TICKS(timeout_ms);
    }

    /* Lock the mutex */
    if (xSemaphoreTake(terminal->read_mutex, portMAX_DELAY) != pdTRUE) 
    {
        /* Failed to take the mutex */
        return 0; 
    }

    size_t bytes_read = 0;
    while (bytes_read < size) 
    {
        size_t chunk_size = xStreamBufferReceive(terminal->input, (uint8_t*)buf + bytes_read, size - bytes_read, timeout_ticks);
        if (chunk_size > 0) {
            bytes_read += chunk_size;
        }
        else 
        {
            /* Timeout or no data available */
            break;
        }
    }

    /* Release the mutex */
    xSemaphoreGive(terminal->read_mutex);

    return bytes_read;
}

/******************************************************************************
* Function Name: terminal_append_input
********************************************************************************
* Summary:
*   Appends data to the terminal's input stream buffer.
*
* Parameters:
*   terminal - Pointer to the terminal structure.
*   buf - Buffer containing the data to append.
*   size - Number of bytes to append.
* Return:
*   The number of bytes appended.
*
*******************************************************************************/
size_t terminal_append_input(terminal_t* terminal, const void* buf, size_t size) 
{
    if (terminal == NULL || buf == NULL || size == 0) 
    {
        return 0;
    }

    uint8_t* bptr = (uint8_t*)buf;
    terminal->last_input = bptr[size - 1];
    /* 0 timeout for non-blocking send */
    return xStreamBufferSend(terminal->input, bptr, size, 0); 
}

/******************************************************************************
* Function Name: terminal_append_input_isr
********************************************************************************
* Summary:
*   Appends data to the terminal's input stream buffer from an ISR context.
*
* Parameters:
*   terminal - Pointer to the terminal structure.
*   buf - Buffer containing the data to append.
*   size - Number of bytes to append.
*   pxHigherPriorityTaskWoken - Pointer to a variable that will be set to pdTRUE 
*   if a higher priority task was woken
* 
* Return:
*  The number of bytes appended.
*
*******************************************************************************/
size_t terminal_append_input_isr(terminal_t* terminal, const void* buf, size_t size, BaseType_t* pxHigherPriorityTaskWoken) 
{
    if (terminal == NULL || buf == NULL || size == 0) 
    {
        return 0;
    }

    uint8_t* bptr = (uint8_t*)buf;

    /* ignore \n if preceeded by \r */
    if (terminal->last_input == '\r' && bptr[0] == '\n') 
    {
        bptr++;
        size--;
        if (size == 0)
        {
            return 0;
        }

    }

    terminal->last_input = bptr[size - 1];

    return xStreamBufferSendFromISR(terminal->input, buf, size, pxHigherPriorityTaskWoken);
}

/******************************************************************************
* Function Name: terminal_destroy
********************************************************************************
* Summary:
*   Destroys the terminal and releases its resources.
*
* Parameters:
*   terminal - Pointer to the terminal structure to destroy.
* Return:
*   None
*******************************************************************************/
void terminal_destroy(terminal_t* terminal) 
{
    if (terminal == NULL) 
    {
        return;
    }

    if (terminal->input != NULL) 
    {
        vStreamBufferDelete(terminal->input);
        terminal->input = NULL;
    }

    if (terminal->read_mutex != NULL) 
    {
        vSemaphoreDelete(terminal->read_mutex);
        terminal->read_mutex = NULL;
    }

    terminal->tag = NULL;
    terminal->write = NULL;
}

/******************************************************************************
* Function Name: terminal_write
********************************************************************************
* Summary:
*   Writes data to the terminal.
*
* Parameters:
*   terminal - Pointer to the terminal structure.
*   data - Buffer containing the data to write.
*   size - Number of bytes to write.
* Return:
*   The number of bytes consumed, or a negative value on error.
*******************************************************************************/
size_t terminal_write(terminal_t* terminal, const void* data, size_t size) 
{
    if (terminal == NULL || terminal->write == NULL || data == NULL || size == 0) 
    {
        return 0;
    }

    if (!(terminal->attr & TERMINAL_ATTR_INJECT_CR_BEFORE_LF)) 
    {
        /* Write the original data to the terminal */
        return terminal->write(terminal, data, size);
    }

    /* Allocate a larger buffer to handle \r insertions */
    char mod_buffer[MAX_TX_LEN];
    const char* data_ptr = (const char*)data;
    size_t mod_size = 0;

    for (size_t i = 0; i < size && mod_size < MAX_TX_LEN - 1; i++) 
    {
        if (data_ptr[i] == '\n') 
        {
            mod_buffer[mod_size++] = '\r';
        }
        mod_buffer[mod_size++] = data_ptr[i];
    }
    
    /* Write the modified buffer to the terminal */
    terminal->write(terminal, mod_buffer, mod_size);
    return size;
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */