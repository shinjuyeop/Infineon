/******************************************************************************
* File Name:   md5.h
*
* Description: Header file for MD5 hash algorithm implementation.
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
#ifndef MD5_H
#define MD5_H

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/******************************************************************************
 * Type
 *****************************************************************************/

/* MD5 context structure to keep track of the state of the hash. */
typedef struct {
    uint64_t size;        /**< Size of the input in bytes. */
    uint32_t buffer[4];   /**< Current accumulation of the hash. */
    uint8_t input[64];    /**< Input to be used in the next step. */
    uint8_t digest[16];   /**< Result of the algorithm. */
} md5_t;

/*******************************************************************************
* Function Prototypes
*******************************************************************************/
void md5_init(md5_t* ctx);
void md5_update(md5_t* ctx, uint8_t* input, size_t input_len);
void md5_finalize(md5_t* ctx);
void md5_step(uint32_t* buffer, uint32_t* input);
void md5_string(char* input, char* output);

#endif /* MD5_H */

/* [] END OF FILE */
