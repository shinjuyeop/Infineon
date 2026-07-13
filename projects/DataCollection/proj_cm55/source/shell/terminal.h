/******************************************************************************
* File Name: terminal.h
*
* Description: This file declares structures and functions for terminal operations.
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

#ifndef _TERMINAL_H_
#define _TERMINAL_H_

#ifdef IM_ENABLE_SHELL

#include <stddef.h>
#include <stdbool.h>
#include <stream_buffer.h>
#include <FreeRTOS.h>
#include <semphr.h>

/*******************************************************************************
* Types
*******************************************************************************/

 /* This enum defines flags for different terminal attributes, such as whether
  * the terminal size is known or if it supports color. */
typedef enum terminal_attrib_t {
    TERMINAL_ATTR_NONE = 0x00,                /**< No attributes set */
    TERMINAL_ATTR_SIZE = 0x01,                /**< Terminal size is known */
    TERMINAL_ATTR_COLOR = 0x02,               /**< Terminal supports color */
    TERMINAL_ATTR_INJECT_CR_BEFORE_LF = 0x04, /**< Inject \r before \n */
    TERMINAL_ATTR_REDUCED_ANSI = 0x08,        /**< Reduced ANSI rendering, 
                                               * some ANSI features disabled. 
                                               * Conserve bandwidth */
} terminal_attrib_t;

/* Forward declaration of the terminal structure. */
typedef struct terminal_s terminal_t;

/* Function pointer type for writing to the terminal. */
typedef size_t(*terminal_write_fn)(terminal_t* terminal, const void* buf, size_t size);

/* This structure holds information about the terminal, including its attributes,
 * size, and function pointers for reading from, writing to, and aborting read operations.
 */
struct terminal_s {
    void* tag;                         /**< User-defined tag, e.g., telnet_client_t or NULL */
    terminal_attrib_t attr;            /**< Bitmask of terminal attributes (TERMINAL_ATTR_*) */
    int width;                         /**< Terminal width in characters */
    int height;                        /**< Terminal height in characters */
    StreamBufferHandle_t input;        /**< Stream buffer for terminal input */
    terminal_write_fn write;           /**< Function pointer for writing to the terminal */
    uint8_t last_input;                /**< Last input byte */
    SemaphoreHandle_t read_mutex;      /**< Mutex for protecting terminal_read */
};

/*******************************************************************************
* Function Prototypes
*******************************************************************************/
void terminal_init(terminal_t* terminal, terminal_write_fn write);
size_t terminal_read(terminal_t* terminal, void* buf, size_t size, int timeout_ms);
size_t terminal_append_input(terminal_t* terminal, const void* buf, size_t size);
size_t terminal_append_input_isr(terminal_t* terminal, const void* buf, size_t size, BaseType_t* pxHigherPriorityTaskWoken);
void terminal_destroy(terminal_t* terminal);
size_t terminal_write(terminal_t* terminal, const void* data, size_t size);

#endif /* IM_ENABLE_SHELL */
#endif /* _TERMINAL_H_ */

/* [] END OF FILE */
