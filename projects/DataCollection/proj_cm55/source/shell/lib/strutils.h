
/******************************************************************************
* File Name:   strutils.h
*
* Description: Interface for string utility functions.
*              This file provides the declarations for utility functions 
*              used for common tasks such as checking if a string is a number
*              and formatting byte sizes.
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
#ifndef _STRUTILS_H_
#define _STRUTILS_H_

#ifdef IM_ENABLE_SHELL

#include <stdbool.h>

/*******************************************************************************
* Function Prototypes
*******************************************************************************/
bool is_int(const char *str);
bool is_float(const char *str);
void format_byte_size(char *buffer, size_t size, long unsigned int number);
void tokenize(const char* str, char* tokens[], int* token_count, char* buffer, int buffer_size);
size_t strlcpy(char* dst, const char* src, size_t size);

#endif /* IM_ENABLE_SHELL */
#endif /* _STRUTILS_H_ */

/* [] END OF FILE */
