/******************************************************************************
* File Name: mem.c
*
* Description: Implementation of the 'mem' command for the shell.
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

#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <malloc.h> /* mallinfo */
#include "../process.h"
#include "../lib/getopt.h"
#include "../lib/strutils.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: main_mem
********************************************************************************
* Summary:
*   Implementation of the 'mem' command for the shell.
*
* Parameters:
*   argc: Argument count.
*   argv: Argument vector.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
int main_mem(int argc, char** argv) 
{
    int opt;
    char buffer[32];

    const char* usage_text = "Usage: %s\n"
                             "List heap memory.\n"
                             "  -h   Display this help and exit\n";

    getopt_t go;
    getopt_init(&go);


    while ((opt = getopte(&go, argc, argv, "h")) != -1) 
    {
        switch (opt) {
            case 'h':
                printf(usage_text, argv[0]);
                return 0;
            default:
                printf(usage_text, argv[0]);
                return -1;
        }
    }

    if (go.optind < argc) 
    {
        printf("Error: Too many arguments\n");
        printf(usage_text, argv[0]);
        return -1;
    }

    size_t freeHeapSize = xPortGetFreeHeapSize();
    size_t minEverFreeHeapSize = xPortGetMinimumEverFreeHeapSize();
    size_t totalHeapSize = configTOTAL_HEAP_SIZE;
    size_t usedHeapSize = totalHeapSize - freeHeapSize;

    printf("--- Heap memory RTOS ---\n");

    format_byte_size(buffer, sizeof(buffer), totalHeapSize);
    printf("Total        %24s bytes\n", buffer);

    format_byte_size(buffer, sizeof(buffer), usedHeapSize);
    printf("Used         %24s bytes\n", buffer);

    format_byte_size(buffer, sizeof(buffer), freeHeapSize);
    printf("Free         %24s bytes\n", buffer);

    format_byte_size(buffer, sizeof(buffer), minEverFreeHeapSize);
    printf("Minimum free %24s bytes\n", buffer);

    struct mallinfo mall_info = mallinfo();

    extern uint8_t __HeapBase;  /* Symbol exported by the linker. */
    extern uint8_t __HeapLimit; /* Symbol exported by the linker. */

    uint8_t* heap_base = (uint8_t *)&__HeapBase;
    uint8_t* heap_limit = (uint8_t *)&__HeapLimit;
    uint32_t heap_size = (uint32_t)(heap_limit - heap_base);

    printf("--- Heap memory glibc ---\n");

    format_byte_size(buffer, sizeof(buffer), heap_size);
    printf("Total        %24s bytes\n", buffer);

    format_byte_size(buffer, sizeof(buffer), mall_info.uordblks);
    printf("Used         %24s bytes, (%.2f%%)\n", buffer,
            ((float)mall_info.uordblks * 100u)/heap_size);

    format_byte_size(buffer, sizeof(buffer), mall_info.arena);
    printf("Maximum      %24s bytes, (%.2f%%)\n",
            buffer,
            ((float)mall_info.arena * 100u)/heap_size);

    return 0;
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */