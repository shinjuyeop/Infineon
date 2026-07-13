
/******************************************************************************
* File Name:   md5.c
*
* Description: This file implements the MD5 hashing algorithm.
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

#include "md5.h"

/******************************************************************************
 * Macros
 *****************************************************************************/
/* Constants as defined by the MD5 algorithm. */
#define A 0x67452301
#define B 0xefcdab89
#define C 0x98badcfe
#define D 0x10325476

/******************************************************************************
 * Global Variables
 *****************************************************************************/

 /* S specifies the per-round shift amounts. */
static uint32_t S[] = { 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
                       5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
                       4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                       6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21 };

/*  K represents the sine function constants used in each transformation. */
static uint32_t K[] = { 0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
                       0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
                       0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
                       0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
                       0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
                       0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
                       0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
                       0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
                       0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
                       0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
                       0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
                       0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
                       0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
                       0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
                       0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
                       0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391 };

/* Padding used to make the size (in bits) of the input congruent to 448 mod 512. */
static uint8_t PADDING[] = { 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

/* MD5 auxiliary functions as defined by the algorithm. */
#define F(X, Y, Z) (((X) & (Y)) | (~(X) & (Z)))
#define G(X, Y, Z) (((X) & (Z)) | ((Y) & ~(Z)))
#define H(X, Y, Z) ((X) ^ (Y) ^ (Z))
#define I(X, Y, Z) ((Y) ^ ((X) | ~(Z)))

/******************************************************************************
* Function Name: rotate_left
********************************************************************************
* Summary:
*    Rotates a 32-bit word left by the specified number of bits.
*
* Parameters:
*   x - The 32-bit word to rotate.
*   n - The number of bits to rotate.
*
* Return:
*   Rotated 32-bit word.
*
*******************************************************************************/
uint32_t rotate_left(uint32_t x, uint32_t n) 
{
    return (x << n) | (x >> (32 - n));
}

/******************************************************************************
* Function Name: md5_init
********************************************************************************
* Summary:
*    Initializes an MD5 context.
*
* Parameters:
*   ctx - Pointer to the MD5 context to initialize.
*
* Return:
*   None
*
*******************************************************************************/
void md5_init(md5_t* ctx) 
{
    ctx->size = (uint64_t)0;

    ctx->buffer[0] = (uint32_t)A;
    ctx->buffer[1] = (uint32_t)B;
    ctx->buffer[2] = (uint32_t)C;
    ctx->buffer[3] = (uint32_t)D;
}

/******************************************************************************
* Function Name: md5_update
********************************************************************************
* Summary:
*    Update the MD5 context with a chunk of input data.
*
* Parameters:
*   ctx - Pointer to the MD5 context to update.
*   input_buffer - Pointer to the input data buffer.
*   input_len - Length of the input data.
*
* Return:
*   None
*
*******************************************************************************/
void md5_update(md5_t* ctx, uint8_t* input_buffer, size_t input_len) 
{
    uint32_t input[16];
    unsigned int offset = ctx->size % 64;
    ctx->size += (uint64_t)input_len;

    /* Copy each byte in input_buffer into the next space in our context input */
    for (unsigned int i = 0; i < input_len; ++i) 
    {
        ctx->input[offset++] = (uint8_t) * (input_buffer + i);

        /* If we've filled our context input, copy it into our local array input 
        *  then reset the offset to 0 and fill in a new buffer.
        *  Every time we fill out a chunk, we run it through the algorithm
        *  to enable some back and forth between cpu and i/o */
        if (offset % 64 == 0) 
        {
            for (unsigned int j = 0; j < 16; ++j) 
            {
                /* Convert to little-endian
                * The local variable `input` our 512-bit chunk separated into 32-bit words
                * we can use in calculations */
                input[j] = (uint32_t)(ctx->input[(j * 4) + 3]) << 24 |
                    (uint32_t)(ctx->input[(j * 4) + 2]) << 16 |
                    (uint32_t)(ctx->input[(j * 4) + 1]) << 8 |
                    (uint32_t)(ctx->input[(j * 4)]);
            }
            md5_step(ctx->buffer, input);
            offset = 0;
        }
    }
}

/******************************************************************************
* Function Name: md5_finalize
********************************************************************************
* Summary:
*    Finalize the MD5 hash calculation, padding the last blocks, and producing 
*    the digest.
*
* Parameters:
*   ctx - Pointer to the MD5 context.
*
* Return:
*   None
*
*******************************************************************************/
void md5_finalize(md5_t* ctx) 
{
    uint32_t input[16];
    unsigned int offset = ctx->size % 64;
    unsigned int padding_length = offset < 56 ? 56 - offset : (56 + 64) - offset;

    /* Fill in the padding and undo the changes to size that resulted from the update */
    md5_update(ctx, PADDING, padding_length);
    ctx->size -= (uint64_t)padding_length;

    /* Do a final update (internal to this function)
     * Last two 32-bit words are the two halves of the size (converted from bytes to bits) */
    for (unsigned int j = 0; j < 14; ++j) 
    {
        input[j] = (uint32_t)(ctx->input[(j * 4) + 3]) << 24 |
            (uint32_t)(ctx->input[(j * 4) + 2]) << 16 |
            (uint32_t)(ctx->input[(j * 4) + 1]) << 8 |
            (uint32_t)(ctx->input[(j * 4)]);
    }
    input[14] = (uint32_t)(ctx->size * 8);
    input[15] = (uint32_t)((ctx->size * 8) >> 32);

    md5_step(ctx->buffer, input);

    /* Move the result into digest (convert from little-endian) */
    for (unsigned int i = 0; i < 4; ++i) 
    {
        ctx->digest[(i * 4) + 0] = (uint8_t)((ctx->buffer[i] & 0x000000FF));
        ctx->digest[(i * 4) + 1] = (uint8_t)((ctx->buffer[i] & 0x0000FF00) >> 8);
        ctx->digest[(i * 4) + 2] = (uint8_t)((ctx->buffer[i] & 0x00FF0000) >> 16);
        ctx->digest[(i * 4) + 3] = (uint8_t)((ctx->buffer[i] & 0xFF000000) >> 24);
    }
}

/******************************************************************************
* Function Name: md5_step
********************************************************************************
* Summary:
*    Perform the MD5 transformation on a 512-bit block of input.
*
* Parameters:
*   buffer - Current state of the MD5 hash.
*   input  - 512-bit block of input data.
*
* Return:
*   None
*
*******************************************************************************/
void md5_step(uint32_t* buffer, uint32_t* input) 
{
    uint32_t AA = buffer[0];
    uint32_t BB = buffer[1];
    uint32_t CC = buffer[2];
    uint32_t DD = buffer[3];

    uint32_t E;

    unsigned int j;

    for (unsigned int i = 0; i < 64; ++i) 
    {
        switch (i / 16) 
        {
            case 0:
            {
                E = F(BB, CC, DD);
                j = i;
                break;
            }

            case 1:
            {
                E = G(BB, CC, DD);
                j = ((i * 5) + 1) % 16;
                break;
            }
            case 2:
            {
                E = H(BB, CC, DD);
                j = ((i * 3) + 5) % 16;
                break;
            }
            default:
            {
                E = I(BB, CC, DD);
                j = (i * 7) % 16;
                break;
            }
        }

        j %= 16;

        uint32_t temp = DD;
        DD = CC;
        CC = BB;
        BB = BB + rotate_left(AA + E + K[i] + input[j], S[i]);
        AA = temp;
    }

    buffer[0] += AA;
    buffer[1] += BB;
    buffer[2] += CC;
    buffer[3] += DD;
}

/******************************************************************************
* Function Name: md5_string
********************************************************************************
* Summary:
*    Compute the MD5 hash of a string.
*
* Parameters:
*   input  - Input string to hash.
*   output - Buffer to store the resulting 32-character hexadecimal hash.

*
* Return:
*   None
*
*******************************************************************************/
void md5_string(char* input, char* output) 
{
    md5_t ctx;
    md5_init(&ctx);
    md5_update(&ctx, (uint8_t*)input, strlen(input));
    md5_finalize(&ctx);

    for (int i = 0; i < 16; i++) 
    {
        sprintf(output + (i * 2), "%02x", ctx.digest[i]);
    }
    output[32] = '\0'; // Null-terminate the output string
}

/* [] END OF FILE */