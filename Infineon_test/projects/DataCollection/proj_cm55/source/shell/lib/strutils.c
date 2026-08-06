/******************************************************************************
* File Name:   strutils.c
*
* Description: Implementation of string utility functions.
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

#include <ctype.h> /*  isdigit */
#include <stdio.h>
#include <string.h>
#include "strutils.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: is_int
********************************************************************************
* Summary:
*    Checks if a string represents a valid integer.
*
* Parameters:
*   str - The string to check.
*
* Return:
*   true if the string represents a valid integer, false otherwise.
*
*******************************************************************************/
bool is_int(const char *str)  
{
    bool has_digits = false;

    if(str == NULL || *str == '\0')
    {
        return false;
    }

    if (*str == '-' || *str == '+')
    {
        str++;
    }

    while (*str) 
    {
        if (!isdigit((int)*str)) 
        {
            return false;
        }
        has_digits = true;
        str++;
    }
    return has_digits;
}

/******************************************************************************
* Function Name: is_float
********************************************************************************
* Summary:
*    Checks if a string represents a valid float.
*
* Parameters:
*   str - The string to check.
*
* Return:
*   true if the string represents a valid float, false otherwise.
*
*******************************************************************************/
bool is_float(const char *str) 
{
    bool has_dot = false;
    bool has_digits = false;

    if(str == NULL || *str == '\0')
    {
        return false;
    }

    if (*str == '-' || *str == '+')
    {
        str++;
    }

    while (*str) 
    {
        if(*str == '.') 
        {
            if(has_dot)
            {
                return false;
            }

            has_dot = true;
        } 
        else if (!isdigit((int)*str)) 
        {
            return false;
        } 
        else 
        {
            has_digits = true;
        }
        str++;
    }
    return has_digits;
}

/******************************************************************************
* Function Name: format_byte_size
********************************************************************************
* Summary:
*    Formats a byte size into a human-readable string with spaces as thousand 
*    separators.
*
* Parameters:
*   buffer - The buffer to store the formatted string.
*   size - The size of the buffer.
*   number - The byte size to format.
*
* Return:
*   None
*
*******************************************************************************/
void format_byte_size(char *buffer, size_t size, long unsigned int number) 
{
    char temp[32];
    snprintf(temp, sizeof(temp), "%ld", number);
    size_t len = strlen(temp);
    size_t n_spaces = (len - 1) / 3;
    size_t new_len = len + n_spaces;
    if (new_len >= size) 
    {
        snprintf(buffer, size, "%ld", number);
        return;
    }

    buffer[new_len] = '\0';
    for (size_t i = 0, j = 0; i < len; i++, j++) 
    {
        buffer[new_len - j - 1] = temp[len - i - 1];
        if ((i + 1) % 3 == 0 && i != len - 1) 
        {
            j++;
            buffer[new_len - j - 1] = ' ';
        }
    }
}

/******************************************************************************
* Function Name: tokenize
********************************************************************************
* Summary:
*    Tokenizes a string into an array of tokens, handling quotes and escape 
*    sequences.
*
* Parameters:
*   str - The string to tokenize.
*   tokens - The array to store the tokens.
*   token_count - The number of tokens found.
*   buffer - The buffer to store the token strings.
*   buffer_size - The size of the buffer.
*
* Return:
*   None
*
*******************************************************************************/
void tokenize(const char* str, char* tokens[], int* token_count, char* buffer, 
    int buffer_size) 
{
    char* buf_ptr = buffer;
    int in_quotes = 0;
    char quote_type = 0;
    int remaining_buffer = buffer_size;

    /* Ensure buffer size is sufficient */
    if (buffer_size <= 0) 
    {
        return; /* Insufficient buffer size, cannot proceed */
    }

    int max_token_count = *token_count;
    *token_count = 0;

    while (*str) 
    {
        /* Skip leading spaces */
        while (isspace((int)*str)) 
        { 
            str++; 
        }
        if (*str == '\0') 
        { 
            break; 
        }

        if (*token_count >= max_token_count) 
        {
            printf("Maximum token count exceeded. Tokenization stopped.\n");
            break;
        }

        tokens[*token_count] = buf_ptr; /* Store the start of the token */
        (*token_count)++;

        char* token_start = buf_ptr; /* remember where this token begins */

        while (*str && (!isspace((int)*str) || in_quotes)) 
        {
            if (*str == '"' || *str == '\'') 
            {
                if (!in_quotes) 
                {
                    /* Handle single quotes that are not at the start of a token */
                    if (*str == '\'' && buf_ptr != token_start)
                    {
                        if (remaining_buffer > 1)
                        {
                            *buf_ptr++ = *str;
                            remaining_buffer--;
                        }
                    }
                    else
                    {
                        in_quotes = 1;
                        quote_type = *str;
                    }
                }
                else if (quote_type == *str) 
                {
                    in_quotes = 0;
                }
                else if (remaining_buffer > 1) 
                {
                    *buf_ptr++ = *str;
                    remaining_buffer--;
                }
            }
            else if (*str == '\\' && in_quotes && (*(str + 1) == '"' || *(str + 1) == '\'')) 
            {
                str++;
                if (remaining_buffer > 1) 
                {
                    *buf_ptr++ = *str;
                    remaining_buffer--;
                }
            }
            else if (remaining_buffer > 1) 
            {
                *buf_ptr++ = *str;
                remaining_buffer--;
            }
            str++;
        }

        if (remaining_buffer > 0) 
        {
            *buf_ptr++ = '\0';
            remaining_buffer--;
        }

        // Skip trailing spaces
        while (isspace((int)*str)) str++;
    }
} 

/******************************************************************************
* Function Name: strlcpy
********************************************************************************
* Summary:
*    Copies a string from source to destination, ensuring null-termination and
*    preventing buffer overflow.
*
* Parameters:
*   dst - The destination buffer to copy the string to.
*   src - The source string to copy from.
*   size - The size of the destination buffer.
*
* Return:
*   The length of the source string.
*
*******************************************************************************/
size_t strlcpy(char* dst, const char* src, size_t size) 
{
    size_t src_len = 0;
    const char* s = src;

    /* Calculate the length of the source string */
    while (*s++) 
    {
        src_len++;
    }

    if (size == 0) 
    {
        /* If size is 0, we cannot perform any copying */
        return src_len;
    }

    size_t copy_len = (src_len >= size) ? size - 1 : src_len;

    /* Copy up to copy_len characters from src to dst */
    for (size_t i = 0; i < copy_len; i++) 
    {
        dst[i] = src[i];
    }

    /* Null-terminate the destination string */
    dst[copy_len] = '\0';

    return src_len;
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */
