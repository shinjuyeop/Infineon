
/******************************************************************************
* File Name:   debug_uart.h
*
* Description: This file contains the function prototypes for the debug UART interface.
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

#ifndef _DEBUG_UART_H_
#define _DEBUG_UART_H_

#ifdef IM_ENABLE_SHELL

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdarg.h>

/*******************************************************************************
* type definitions
*******************************************************************************/
typedef struct terminal_s terminal_t;

/*******************************************************************************
* Function Prototypes
*******************************************************************************/

bool debug_uart_init(void);
void debug_start_input_stream(terminal_t* input);
size_t debug_uart_write(const char* ptr, size_t size);
int printk(const char* restrict format, ...);
int vprintk(const char* restrict format, va_list args);

#endif /* IM_ENABLE_SHELL */

#endif /* _DEBUG_UART_H_ */

/* [] END OF FILE */
