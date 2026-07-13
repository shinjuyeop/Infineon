/******************************************************************************
* File Name: dapply.c
*
* Description: Implementation of the 'dapply' command for the shell.
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
#include "../lib/getopt.h"
#include "../process.h"
#include "../../protocol/protocol.h"
#include "../lib/tsp_utils.h"
#include "../../services.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/
/******************************************************************************
* Function Name: main_dapply
********************************************************************************
* Summary:
*   Implementation of the 'dapply' command for the shell.
*
* Parameters:
*   argc: Argument count.
*   argv: Argument vector.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
int main_dapply(int argc, char** argv) 
{
    int opt;
    const char* usage_text = "Usage: %s [-h|-s] <device>\n"
                             "This command will apply the current setting to the device.\n"
                             "This will validate all the options and reconfigure the device streams.\n"
                             "\n"
                             "  -h             Show this help message and exit.\n"
                             "  -s             Silent, less chatty output.\n";
    getopt_t go;
    getopt_init(&go);
    bool silent = false;

    while ((opt = getopte(&go, argc, argv, "hs")) != -1) 
    {
        switch (opt) 
        {
            case 'h':
                printf(usage_text, argv[0]);
                return 0;
            case 's':
                silent = true;
                break;
            default:
                printf(usage_text, argv[0]);
                return -1;
        }
    }

    if (go.optind >= argc)
    {
        printf("Error: Missing <device> argument\n");
        printf(usage_text, argv[0]);
        return -1;
    }

    if (go.optind < argc - 1) 
    {
        printf("Error: Too many arguments\n");
        printf(usage_text, argv[0]);
        return -1;
    }

    char* device_str = argv[go.optind];

    protocol_t* protocol = SYS_SERVICE_PROTOCOL();

    protocol_Device* device = tsp_find(protocol, device_str);
    if (!device) 
    {
        printf("Error: No device named or index '%s' found.\n", device_str);
        return -1;
    }

    device_manager_t* manager = &protocol->device_managers[device->device_id];
    protocol_configure_streams_fn configure_streams = manager->configure_streams;
    if (configure_streams != NULL) 
    {
        configure_streams(protocol, device->device_id, manager->arg);
    }

    if (!silent) 
    {
        printf("Options have been applied.\n");
        printf("Check updated streams with command: dlist \"%s\"\n", device->name);
    }
    return 0;
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */