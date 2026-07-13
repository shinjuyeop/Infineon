/******************************************************************************
* File Name: disk.c
*
* Description: Implementation of the 'disk' command for the shell.
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
#include <stdio.h>
#include <errno.h>
#include "../lib/getopt.h"
#include "../lib/strutils.h"
#include "../fs/statvfs.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/
/******************************************************************************
* Function Name: main_disk
********************************************************************************
* Summary:
*   Implementation of the 'disk' command for the shell.
*
* Parameters:
*   argc: Argument count.
*   argv: Argument vector.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
int main_disk(int argc, char** argv) 
{
    
    const char* usage_text = "Usage: %s [path]\n"
                             "Print disk usage.\n"
                             "  -h   Display this help and exit\n";

    getopt_t go;
    getopt_init(&go);
    const char* path = ".";

    int opt;
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

    if (go.optind < argc - 1) 
    {
        printf("Error: Too many arguments\n");
        printf(usage_text, argv[0]);
        return -1;
    }

    if (go.optind < argc) 
    {
        path = argv[go.optind];
    }

    struct statvfs buf;

    if (statvfs(path, &buf)) 
    {
        printf("Failed to fetch mount stats. %s\n", strerror(errno));
        return -1;
    }
    
    int total_disk_size = buf.f_bsize * buf.f_blocks;
    int total_used_bytes = buf.f_bsize * buf.f_ublocks; 
    int free_bytes = total_disk_size - total_used_bytes;

    char buffer[32];
    printf("Total disk size   ");
    format_byte_size(buffer, sizeof(buffer), total_disk_size);
    printf("%12s bytes\n", buffer);

    printf("Total used bytes  ");
    format_byte_size(buffer, sizeof(buffer), total_used_bytes);
    printf("%12s bytes\n", buffer);

    printf("Free bytes        ");
    format_byte_size(buffer, sizeof(buffer), free_bytes);
    printf("%12s bytes\n", buffer);

    printf("\n");
    printf("LFS block size    ");
    format_byte_size(buffer, sizeof(buffer), buf.f_bsize);
    printf("%12s bytes\n", buffer);

    printf("LFS block cycles  ");
    format_byte_size(buffer, sizeof(buffer), buf.block_cycles);
    printf("%12s bytes\n", buffer);

    printf("LFS cache size    ");
    format_byte_size(buffer, sizeof(buffer), buf.cache_size);
    printf("%12s bytes\n", buffer);

    printf("LFS lookahead     ");
    format_byte_size(buffer, sizeof(buffer), buf.lookahead_size);
    printf("%12s bytes\n", buffer);
    
    return 0;
}


#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */