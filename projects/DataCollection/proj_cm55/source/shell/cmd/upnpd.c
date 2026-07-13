/******************************************************************************
* File Name: upnpd.c
*
* Description: Implementation of the 'upnpd' command for the shell.
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
#include <stdlib.h>
#include <string.h>
#include <FreeRTOS.h>
#include "../../services.h"
#include "../net/upnp.h"
#include "../net/wifi.h"
#include "../process.h"

/*******************************************************************************
* Function Prototypes
*******************************************************************************/
static int start();
static int stop();

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: main_upnpd
********************************************************************************
* Summary:
*   Implementation of the 'upnpd' command for the shell.
*
* Parameters:
*   argc: Argument count.
*   argv: Argument vector.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
int main_upnpd(int argc, char** argv) 
{
    const char* usage_text = "Usage: %s <command>\n"
        "Commands:\n"
        "  start - Start UPnP service\n"
        "  stop  - Stop UPnP service\n";

    if (argc < 2) 
    {
        printf(usage_text, argv[0]);
        return -1;
    }

    const char* command = argv[1];

    if (strcmp(command, "start") == 0) 
    {
        if (argc != 2) 
        {
            printf("Usage: %s stop\n", argv[0]);
            return -1;
        }
        return start();
    } 
    else if (strcmp(command, "stop") == 0) 
    {
        if (argc != 2) 
        {
            printf("Usage: %s stop\n", argv[0]);
            return -1;
        }
        return stop();
    } 
    else 
    {
        printf("Unknown command: %s\n", argv[1]);
        printf(usage_text, argv[0]);
        return -1;
    }

    return 0;
}

/******************************************************************************
* Function Name: start
********************************************************************************
* Summary:
*   Starts the UPnP service.
*
* Parameters:
*  None
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int start() 
{
#ifdef IM_ENABLE_UPNP
    upnp_t* upnp = SYS_SERVICE_UPNP();
    if (upnp == NULL) 
    {
        printf("No UPnP service exist.\n");
        return -1;
    }
    
    wifi_t* wifi = SYS_SERVICE_WIFI();
    if (wifi == NULL) 
    {
        printf("No Wi-Fi service exist.\n");
        return -1;
    }

    if(!upnp_start(upnp, wifi->ip_addr)) 
    {
        printf("UPnP service start failed.\n");
        return -1;
    }

    return 0;
#else
    printf("UPnP not enabled in build. IM_ENABLE_UPNP must be defined.\n");
    return -1;
#endif
}

/******************************************************************************
* Function Name: stop
********************************************************************************
* Summary:
*   Stops the UPnP service.
*
* Parameters:
*  None
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int stop() 
{
#ifdef IM_ENABLE_UPNP
    upnp_t* upnp = SYS_SERVICE_UPNP();
    if (upnp == NULL) 
    {
        printf("No UPnP service exist.\n");
        return -1;
    }

    if(!upnp_stop(upnp)) 
    {
        printf("UPnP service stop failed.\n");
        return -1;
    }

    return 0;
#else
    printf("UPnP not enabled in build. IM_ENABLE_UPNP must be defined.\n");
    return -1;
#endif
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */