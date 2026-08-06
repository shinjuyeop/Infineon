/******************************************************************************
* File Name: version.c
*
* Description: Implementation of the 'version' command for the shell.
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
#include "../../protocol/protocol.h"
#include "../../services.h"
#include "../process.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: main_version
********************************************************************************
* Summary:
*   Implementation of the 'version' command for the shell.
*
* Parameters:
*   argc: Argument count.
*   argv: Argument vector.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
int main_version(int argc, char** argv) 
{
    const char* usage_text = "Usage: %s\n"
                             "Display build and version information.\n"
                             "  -h   Display this help and exit\n";

    getopt_t go;
    getopt_init(&go);

    int opt;
    while ((opt = getopte(&go, argc, argv, "h")) != -1) 
    {
        switch (opt) 
        {
            case 'h':
            {
                printf(usage_text, argv[0]);
                return 0;
            }

            default:
            {
                printf(usage_text, argv[0]);
                return -1;
            }

        }
    }

    if (go.optind < argc) 
    {
        printf("Error: Too many arguments\n");
        printf(usage_text, argv[0]);
        return -1;
    }

    protocol_t *protocol = SYS_SERVICE_PROTOCOL();
    if (protocol != NULL) 
    {
        uint8_t* serial = protocol->board.serial.uuid;

        printf("Board Name             : %s\n", protocol->board.name);
        printf("Board Serial           : %02X%02X%02X%02X-%02X%02X-%02X%02X-%02X%02X-%02X%02X%02X%02X%02X%02X\n",
            serial[0], serial[1], serial[2], serial[3],
            serial[4], serial[5],
            serial[6], serial[7],
            serial[8], serial[9],
            serial[10], serial[11], serial[12], serial[13], serial[14], serial[15]);
        printf("Firmware Version       : %ld.%ld.%ld.%ld\n",
            protocol->board.firmware_version.major,
            protocol->board.firmware_version.minor,
            protocol->board.firmware_version.build,
            protocol->board.firmware_version.revision);
        printf("Protocol Version       : %ld.%ld.%ld.%ld\n",
            protocol->board.protocol_version.major,
            protocol->board.protocol_version.minor,
            protocol->board.protocol_version.build,
            protocol->board.protocol_version.revision);
    }
    printf("GCC %d.%d.%d / STDC %ld\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__, __STDC_VERSION__);

    return 0;
}

#endif /* IM_ENABLE_SHELL */