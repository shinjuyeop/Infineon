/******************************************************************************
* File Name: gpio.c
*
* Description: Implementation of the 'gpio' command for the shell.
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
#include <cy_gpio.h>
#include <cybsp.h>
#include "../lib/getopt.h"
#include "../process.h"
#include "../common.h"

/*******************************************************************************
 * Global Variables
 ******************************************************************************/
extern volatile bool button_pressed;
/*******************************************************************************
* Function Prototypes
*******************************************************************************/
static int wait_button_press(int timeout_ms);

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: main_gpio
********************************************************************************
* Summary:
*   Implementation of the 'gpio' command for the shell.
*
* Parameters:
*   argc: Argument count.
*   argv: Argument vector.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
int main_gpio(int argc, char** argv) 
{
    const char* usage_text = "Usage: %s <command> [arguments]\n"
        "Commands:\n"
        "  wait_button [-t timeout_ms]  - Wait for a user button press\n"
        "  led0 <on/off>                - Set LED 0\n"
        "  led1 <on/off>                - Set LED 1\n";
    int opt;
    const char* command = argv[1];

    getopt_t go;
    getopt_init(&go);

    if (argc < 2) 
    {
        printf(usage_text, argv[0]);
        return -1;
    }

    if (strcmp(command, "wait_button") == 0) 
    {
       int timeout = 0;
       while ((opt = getopte(&go, argc - 1, argv + 1, "t:h")) != -1) 
       {
            switch (opt) 
            {
                case 'h':
                    printf(usage_text, argv[0]);
                    return 0;

                case 't':
                    if (go.optarg) 
                    {
                        timeout = atoi(go.optarg);
                        if (timeout <= 0) 
                        {
                            printf("Invalid timeout: %s\n", go.optarg);
                            return -1;
                        }
                    }
                    else 
                    {
                        printf("Option -p requires an argument.\n");
                        return -1;
                    }
                break;

                default:
                    printf(usage_text, argv[0]);
                    return -1;
            }
        }
        return wait_button_press(timeout);
    }
    else if (strcmp(command, "led0") == 0) 
    {
        if (argc != 3) 
        {
            printf("Usage: %s led0 <on|off>", argv[0]);
            return -1;
        }

        if (strcmp(argv[2], "on") == 0) 
        {
            set_led0(true);
        }
        else if (strcmp(argv[2], "off") == 0) 
        {
            set_led0(false);
        } 
        else 
        {
            printf("Usage: %s led0 <on|off>", argv[0]);
            return -1;
        }

        return 0;
    }
    else if (strcmp(command, "led1") == 0) 
    {
        if (argc != 3) 
        {
            printf("Usage: %s led1 <on|off>", argv[0]);
            return -1;
        }

        if (strcmp(argv[2], "on") == 0) 
        {
            set_led1(true);
        }
        else if (strcmp(argv[2], "off") == 0) 
        {
            set_led1(false);
        }
        else 
        {
            printf("Usage: %s led1 <on|off>", argv[0]);
            return -1;
        }

        return 0;
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
* Function Name: wait_button_press
********************************************************************************
* Summary:
*   Waits for a user button press with an optional timeout.
*
* Parameters:
*   timeout_ms: Timeout in milliseconds. If 0, wait indefinitely.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int wait_button_press(int timeout_ms) 
{
   
    printf("Waiting for user button press. Ctrl-C to cancel.\n");
    clock_tick_t start = clock_get_tick();
    while (true) 
    {

        if (timeout_ms > 0 && (clock_get_tick() - start) > (CLOCK_TICK_PER_SECOND / 1000 * timeout_ms)) 
        {
            printf("Timeout.\n");
            return -1;
        }

        if (button_pressed == true) 
        {
            printf("Button pressed!\n");
            button_pressed = false;
            return 0;
        }
    }

    return 0;
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */