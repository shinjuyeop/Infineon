/******************************************************************************
* File Name: hex.c
*
* Description: Implementation of the 'hex' command for the shell.
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

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include "../lib/getopt.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: main_hex
********************************************************************************
* Summary:
*   Implementation of the 'hex' command for the shell.
*
* Parameters:
*   argc: Argument count.
*   argv: Argument vector.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
int main_hex(int argc, char** argv) 
{
    int opt;
    unsigned char buffer[16];
    size_t bytes_read;
    size_t offset = 0;
    const char* usage_text = "Usage: %s <file>\n"
                             "Display file contents in hex.\n"
                             "  -h   Display this help and exit\n";

    getopt_t go;
    getopt_init(&go);


    while ((opt = getopte(&go, argc, argv, "h")) != -1) 
    {
        switch (opt) 
        {
            case 'h':
                printf(usage_text, argv[0]);
                return 0;
            default:
                printf(usage_text, argv[0]);
                return -1;
        }
    }

    if (go.optind >= argc) 
    {
        printf("Error: No file specified\n");
        printf(usage_text, argv[0]);
        return -1;
    }

    if (go.optind < argc - 1) 
    {
        printf("Error: Too many arguments\n");
        printf(usage_text, argv[0]);
        return -1;
    }

    FILE* file = fopen(argv[go.optind], "rb");
    if (!file) 
    {
        printf("Failed to open file. %s\n", strerror(errno));
        return -1;
    }

    while ((bytes_read = fread(buffer, 1, sizeof(buffer), file)) > 0) 
    {
        printf("%08x: ", offset);
        for (int i = 0; i < bytes_read; i++) 
        {
            printf("%02x ", buffer[i]);
        }

        for (int i = bytes_read; i < 16; i++) 
        {
            printf("   ");
        }
        printf(" ");

        for (int i = 0; i < bytes_read; i++) 
        {
            printf("%c", (buffer[i] >= 32 && buffer[i] <= 126) ? buffer[i] : '.');
        }
        printf("\n");
        offset += bytes_read;
    }

    if (ferror(file)) 
    {
        printf("Failed to read file. %s\n", strerror(errno));
    }

    fclose(file);
    return 0;
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */