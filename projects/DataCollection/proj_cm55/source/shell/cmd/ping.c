/******************************************************************************
* File Name: ping.c
*
* Description: Implementation of the 'ping' command for the shell.
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
#include <cy_wcm.h>
#include <cy_nw_helper.h>
#include <FreeRTOS.h>
#include <task.h>
#include "../lib/getopt.h"

/*******************************************************************************
* Function Prototypes
*******************************************************************************/

static int send_ping(const char* ip_str, int count, int timeout);

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: main_ping
********************************************************************************
* Summary:
*   Implementation of the 'ping' command for the shell.
*
* Parameters:
*   argc: Argument count.
*   argv: Argument vector.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
int main_ping(int argc, char** argv) 
{
    const char* usage_text = "Usage: %s <IP address> [-n count] [-w timeout]\n"
        "Options:\n"
        "  -n count    Number of ping requests to send (default is 4)\n"
        "  -w timeout  Timeout in milliseconds to wait for each reply (default is 1000 ms)\n";

    int count = 3;
    int timeout = 1000;
    int opt;
    const char* ip_str = NULL;

    getopt_t go;
    getopt_init(&go);

    /* First pass: parse options */
    while ((opt = getopte(&go, argc, argv, "n:w:h")) != -1) 
    {
        switch (opt) 
        {
            case 'n':
                count = atoi(go.optarg);
                if (count <= 0) 
                {
                    printf("Invalid count value: %d\n", count);
                    return -1;
                }
                break;
            case 'w':
                timeout = atoi(go.optarg);
                if (timeout <= 0) 
                {
                    printf("Invalid timeout value: %d\n", timeout);
                    return -1;
                }
                break;
            case 'h':
            printf(usage_text, argv[0]);
                return 0;
            default:
                printf(usage_text, argv[0]);
                return -1;
        }
    }

    /* Identify the positional argument (IP address) */
    if (go.optind < argc) 
    {
        ip_str = argv[go.optind];
    }

    /* If IP address is not found, display usage and return error */
    if (!ip_str) 
    {
        printf("IP address is required.\n");
        printf(usage_text, argv[0]);
        return -1;
    }

    printf("Pinging address %s\n", ip_str);
    return send_ping(ip_str, count, timeout);
}

/******************************************************************************
* Function Name: send_ping
********************************************************************************
* Summary:
*   Sends ICMP echo requests (ping) to the specified IP address.
*
* Parameters:
*   ip_str: IP address as a string.
*   count: Number of ping requests to send.
*   timeout: Timeout in milliseconds to wait for each reply.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int send_ping(const char* ip_str, int count, int timeout) 
{

#ifdef IM_ENABLE_NET

    cy_wcm_ip_address_t ip_addr;
    cy_nw_ip_address_t nw_ip_addr;
    uint32_t elapsed_ms;
    cy_rslt_t result;

    /* Convert string IP address to cy_nw_ip_address_t format */
    if (cy_nw_str_to_ipv4(ip_str, &nw_ip_addr) != CY_RSLT_SUCCESS) 
    {
        printf("Invalid IP address format: %s\n", ip_str);
        return -1;
    }

    /* Copy the converted IP address to cy_wcm_ip_address_t structure */
    ip_addr.version = CY_WCM_IP_VER_V4;
    ip_addr.ip.v4 = nw_ip_addr.ip.v4;

    for (int i = 0; i < count; i++) 
    {
        result = cy_wcm_ping(CY_WCM_INTERFACE_TYPE_STA, &ip_addr, timeout, &elapsed_ms);
        if (result == CY_RSLT_SUCCESS) 
        {
            printf("Reply from %s: time=%lu ms\n", ip_str, elapsed_ms);
        }
        else 
        {
            printf("Request timed out.\n");
        }
        /* Wait 1 second between pings */
        vTaskDelay(pdMS_TO_TICKS(1000)); 
    }

    return 0;

#else
    printf("ping not enabled in build. IM_ENABLE_NET must be defined.\n");
    return -1;
#endif

}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */