/******************************************************************************
* File Name:   base64.c
*
* Description: Base64 decoding implementation.
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
#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>

/******************************************************************************
 * Type
 *****************************************************************************/
/* Base64 decoding table */ 
static const uint8_t base64_table[256] = {
    [0 ... 255] = 0xff, // default value indicating invalid input
    ['A'] = 0,['B'] = 1,['C'] = 2,['D'] = 3,['E'] = 4,['F'] = 5,
    ['G'] = 6,['H'] = 7,['I'] = 8,['J'] = 9,['K'] = 10,['L'] = 11,
    ['M'] = 12,['N'] = 13,['O'] = 14,['P'] = 15,['Q'] = 16,['R'] = 17,
    ['S'] = 18,['T'] = 19,['U'] = 20,['V'] = 21,['W'] = 22,['X'] = 23,
    ['Y'] = 24,['Z'] = 25,['a'] = 26,['b'] = 27,['c'] = 28,['d'] = 29,
    ['e'] = 30,['f'] = 31,['g'] = 32,['h'] = 33,['i'] = 34,['j'] = 35,
    ['k'] = 36,['l'] = 37,['m'] = 38,['n'] = 39,['o'] = 40,['p'] = 41,
    ['q'] = 42,['r'] = 43,['s'] = 44,['t'] = 45,['u'] = 46,['v'] = 47,
    ['w'] = 48,['x'] = 49,['y'] = 50,['z'] = 51,['0'] = 52,['1'] = 53,
    ['2'] = 54,['3'] = 55,['4'] = 56,['5'] = 57,['6'] = 58,['7'] = 59,
    ['8'] = 60,['9'] = 61,['+'] = 62,['/'] = 63,['='] = 0,
};

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: base64_decode
********************************************************************************
* Summary:
*    Decodes a Base64 encoded string.
*
* Parameters:
*   input  - Pointer to the Base64 encoded string.
*   output - Pointer to the buffer where the decoded string will be stored.
*
* Return:
*   None
*
*******************************************************************************/
void base64_decode(const char* input, char* output) 
{
    size_t input_len = strlen(input);
    size_t output_len = 0;

    for (size_t i = 0; i < input_len; i += 4) 
    {

        if (base64_table[(int)(input[i])] == 0xff)
        {
            /* Illegal character i.e 0x0a or 0x0d. */
            break;
        }

        uint32_t val = base64_table[(unsigned char)input[i]] << 18;
        val += base64_table[(unsigned char)input[i + 1]] << 12;
        val += base64_table[(unsigned char)input[i + 2]] << 6;
        val += base64_table[(unsigned char)input[i + 3]];

        output[output_len++] = (val >> 16) & 0xFF;
        if (input[i + 2] != '=')
        {
            output[output_len++] = (val >> 8) & 0xFF;
        }
        if (input[i + 3] != '=')
        {
            output[output_len++] = val & 0xFF;
        }

    }

    output[output_len] = '\0';
}

/* [] END OF FILE */