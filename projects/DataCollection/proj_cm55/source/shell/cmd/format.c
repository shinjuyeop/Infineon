/******************************************************************************
* File Name: format.c
*
* Description: Implementation of the 'format' command for the shell.
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
#include <stdbool.h>
#include <errno.h>
#include "../lib/getopt.h"
#include "../fs/fs.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: main_format
********************************************************************************
* Summary:
*   Implementation of the 'format' command for the shell.
*
* Parameters:
*   argc: Argument count.
*   argv: Argument vector.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
int main_format(int argc, char** argv) 
{

    const char* usage_text = "Usage: %s [options]\n"
                             "Format the filesystem.\n"
                             "  -t fstype   Format using given filesystem type\n"
                             "  -y          Confirm format. Required\n"
                             "  -h          Display this help and exit\n";

    getopt_t go;
    getopt_init(&go);
    char* fstype = "lfs";
    bool confirm = false;
    int argc_ext = 0;
    const char* argv_ext[32] = { 0 };
    int opt;

    while ((opt = getopte(&go, argc, argv, "yht:o:b:d:f:r:e:")) != -1) 
    {
        switch (opt) 
        {
            case 'y':
                confirm = true;
                break;
            case 't':
                fstype = go.optarg;
                break;
            case 'h':
                printf(usage_text, argv[0]);
                fs_print_usage();
                return 0;
            case 'o':
            case 'b':
            case 'd':
            case 'f':
            case 'r':
            case 'e':
                argv_ext[argc_ext++] = argv[go.optind - 2] + 1;
                argv_ext[argc_ext++] = argv[go.optind - 1];
                break;
        }
    }

    if (!confirm) 
    {
        printf("Error: -y option is required\n");
        printf(usage_text, argv[0]);
        fs_print_usage();
        return -1;
    }

    if (go.optind < argc) 
    {
        printf("Error: Too many arguments\n");
        printf(usage_text, argv[0]);
        return -1;
    }

    printf("Formatting. Please wait...\n");

 
    if (fs_format(fstype, argc_ext, argv_ext)) 
    {
        printf("Failed to format. %s\n", strerror(errno));
        return -1;
    }

    printf("Filesystem formatted successfully.\n");

    return 0;
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */