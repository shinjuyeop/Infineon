/******************************************************************************
* File Name: wifi.c
*
* Description: Implementation of the 'wifi' command for the shell.
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
#include <errno.h>
#include <cy_wcm.h>
#include <FreeRTOS.h>
#include <task.h>
#include "../lib/readline.h"
#include "../lib/getopt.h"
#include "../../services.h"
#include "../process.h"
#include "../net/wifi.h"

#ifdef IM_ENABLE_NET
/******************************************************************************
 * Macros
 *****************************************************************************/
 /* Constants for security modes */ 
#define SECURITY_OPEN            "OPEN"
#define SECURITY_WPA2_AES_PSK    "WPA2_AES_PSK"
#define SECURITY_WPA2_MIXED_PSK  "WPA2_MIXED_PSK"
#define SECURITY_WPA3_SAE        "WPA3_SAE"
#define SECURITY_WPA3_WPA2_PSK   "WPA3_WPA2_PSK"

/* Constants for command handling */
#define COMMAND_ENTER 0
#define COMMAND_EXIT  1

#define MAX_SCAN_RESULTS 100

/******************************************************************************
 * Type
 *****************************************************************************/
/* Private structure to hold scan state */
typedef struct {
    TaskHandle_t scan_task_handle;
    uint32_t num_scan_result;
    QueueHandle_t scan_results_queue;
    cy_wcm_scan_result_t scan_results[MAX_SCAN_RESULTS]; /* Array to store scan results */
} wifi_scan_context_t;

/*******************************************************************************
* Function Prototypes
*******************************************************************************/
static int list_available_networks(wifi_scan_context_t* context);
static int get_user_input(char* prompt, char* buffer, size_t buffer_size, bool mask);
static int join_argv(int argc, char** argv, int start, char* buf, int max_len);
static int save_config(const char* ssid, const char* password, const char* security_mode);
static int wifi_setup();
static int key_press(readline_t* rl, char key);
static void scan_callback(cy_wcm_scan_result_t* result_ptr, void* user_data, cy_wcm_scan_status_t status);
static void print_scan_result(cy_wcm_scan_result_t* result, uint32_t num_scan_result);
static int set_ssid(char* ssid);
static int set_password(char* password);
static int set_security(cy_wcm_security_t security);
static int wifi_start_cmd();
static int wifi_stop_cmd();
static int save();

#endif /* IM_ENABLE_NET */

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: main_wifi
********************************************************************************
* Summary:
*   Implementation of the 'wifi' command for the shell.
*
* Parameters:
*   argc: Argument count.
*   argv: Argument vector.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
int main_wifi(int argc, char** argv) {
    const char* usage_text = "Usage: %s <command> [arguments]\n"
                             "Commands:\n"
                             "  setup                - Run the Wi-Fi setup wizard to configure a new connection\n"
                             "  set_ssid <ssid>      - Set the Wi-Fi network SSID\n"
                             "  set_passwd <pwd>     - Set the Wi-Fi network password\n"
                             "  set_security <mode>  - Set the Wi-Fi security mode\n"
                             "  start                - Bring the Wi-Fi connection up\n"
                             "  stop                 - Disconnect from the Wi-Fi network\n"
                             "  save                 - Save the configuration\n";

    getopt_t go;
    getopt_init(&go);

    if (argc < 2) 
    {
        printf(usage_text, argv[0]);
        return -1;
    }

#ifndef IM_ENABLE_NET
    printf("Wi-Fi not enabled in build. IM_ENABLE_NET must be defined.\n");
    return -1;
#else
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

    if (strcmp(argv[1], "setup") == 0) 
    {
        if (argc != 2) 
        {
            printf("Usage: %s setup\n", argv[0]);
            return -1;
        }
        return wifi_setup();
    } 
    else if (strcmp(argv[1], "set_ssid") == 0) 
    {
        if (argc < 3) 
        {
            printf("Usage: %s set_ssid <ssid>\n", argv[0]);
            return -1;
        }

        /* Buffer to hold SSID from command line arguments */
        char ssid_buf[WIFI_MAX_SSID_LEN + 1];
        join_argv(argc, argv, 2, ssid_buf, WIFI_MAX_SSID_LEN);
        return set_ssid(ssid_buf);
    } 
    else if (strcmp(argv[1], "set_passwd") == 0) 
    {
        if (argc < 3) 
        {
            printf("Usage: %s set_passwd <pwd>\n", argv[0]);
            return -1;
        }
        char pwd_buf[WIFI_MAX_PASSWORD_LEN + 1];
        join_argv(argc, argv, 2, pwd_buf, WIFI_MAX_PASSWORD_LEN);
        return set_password(pwd_buf);
    } 
    else if (strcmp(argv[1], "set_security") == 0) 
    {
        if (argc != 3) 
        {
            printf("Usage: %s set_security <mode>\n", argv[0]);
            return -1;
        }
        cy_wcm_security_t security;
        if (strcmp(argv[2], "OPEN") == 0) 
        {
            security = CY_WCM_SECURITY_OPEN;
        } 
        else if (strcmp(argv[2], "WEP_PSK") == 0) 
        {
            security = CY_WCM_SECURITY_WEP_PSK;
        } 
        else if (strcmp(argv[2], "WPA_TKIP_PSK") == 0) 
        {
            security = CY_WCM_SECURITY_WPA_TKIP_PSK;
        } 
        else if (strcmp(argv[2], "WPA_AES_PSK") == 0) 
        {
            security = CY_WCM_SECURITY_WPA_AES_PSK;
        } 
        else if (strcmp(argv[2], "WPA2_AES_PSK") == 0) 
        {
            security = CY_WCM_SECURITY_WPA2_AES_PSK;
        } 
        else if (strcmp(argv[2], "WPA2_TKIP_PSK") == 0) 
        {
            security = CY_WCM_SECURITY_WPA2_TKIP_PSK;
        } 
        else if (strcmp(argv[2], "WPA2_MIXED_PSK") == 0) 
        {
            security = CY_WCM_SECURITY_WPA2_MIXED_PSK;
        } 
        else if (strcmp(argv[2], "WPA3_SAE") == 0) 
        {
            security = CY_WCM_SECURITY_WPA3_SAE;
        } 
        else if (strcmp(argv[2], "WPA3_WPA2_PSK") == 0) 
        {
            security = CY_WCM_SECURITY_WPA3_WPA2_PSK;
        } 
        else 
        {
            printf("Invalid security mode: %s\n", argv[2]);
            printf("Valid modes are: OPEN, WEP_PSK, WPA_TKIP_PSK, WPA_AES_PSK, WPA2_AES_PSK, WPA2_TKIP_PSK, WPA2_MIXED_PSK, WPA3_SAE, WPA3_WPA2_PSK\n");
            return -1;
        }
        return set_security(security);
    } 
    else if (strcmp(argv[1], "start") == 0) 
    {
        if (argc != 2) 
        {
            printf("Usage: %s start\n", argv[0]);
            return -1;
        }
        return wifi_start_cmd();
    } 
    else if (strcmp(argv[1], "stop") == 0) 
    {
        if (argc != 2) 
        {
            printf("Usage: %s stop\n", argv[0]);
            return -1;
        }
        return wifi_stop_cmd();
    } 
    else if (strcmp(argv[1], "save") == 0) 
    {
        if (argc != 2) 
        {
            printf("Usage: %s save\n", argv[0]);
            return -1;
        }
        return save();
    } 
    else 
    {
        printf("Unknown command: %s\n", argv[1]);
        printf(usage_text, argv[0]);
        return -1;
    }

    return 0;
#endif /* IM_ENABLE_NET */
}

#ifdef IM_ENABLE_NET

/******************************************************************************
* Function Name: get_security_mode_string
********************************************************************************
* Summary:
*   Converts a cy_wcm_security_t enum value to a human-readable string.
*
* Parameters:
*   security_mode: Pointer to a buffer where the string representation will be stored.
*   security: The cy_wcm_security_t enum value to convert.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int get_security_mode_string( char* security_mode, cy_wcm_security_t security )
{

    switch (security) 
    {
        case CY_WCM_SECURITY_OPEN:
        {
            strcpy(security_mode, SECURITY_OPEN);
            break;
        }

        case CY_WCM_SECURITY_WEP_PSK:
        {
            strcpy(security_mode, "WEP_PSK");
            break;
        }

        case CY_WCM_SECURITY_WPA_TKIP_PSK:
        {
            strcpy(security_mode, "WPA_TKIP_PSK");
            break;
        }

        case CY_WCM_SECURITY_WPA_AES_PSK:
        {
            strcpy(security_mode, "WPA_AES_PSK");
            break;
        }

        case CY_WCM_SECURITY_WPA2_AES_PSK:
        {
            strcpy(security_mode, SECURITY_WPA2_AES_PSK);
            break;
        }

        case CY_WCM_SECURITY_WPA2_TKIP_PSK:
        {
            strcpy(security_mode, "WPA2_TKIP_PSK");
            break;
        }

        case CY_WCM_SECURITY_WPA2_MIXED_PSK:
        {
            strcpy(security_mode, SECURITY_WPA2_MIXED_PSK);
            break;
        }

        case CY_WCM_SECURITY_WPA3_SAE:
        {
            strcpy(security_mode, SECURITY_WPA3_SAE);
            break;
        }

        case CY_WCM_SECURITY_WPA3_WPA2_PSK:
        {
            strcpy(security_mode, SECURITY_WPA3_WPA2_PSK);
            break;
        }

        default:
        {
            strcpy(security_mode, "UNKNOWN");
            break;
        }

    }

    return 0;

}

/******************************************************************************
* Function Name: wifi_setup
********************************************************************************
* Summary:
*   Guides the user through the process of setting up a Wi-Fi connection.
*
* Parameters:
* None
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int wifi_setup() 
{
    char ssid[64];
    char password[64];
    char security_mode[32];
    int selected_network;
    cy_wcm_scan_result_t selected_result;
    cy_wcm_security_t security;
    wifi_t* wifi;

    /* Print welcome/instructions message */
    printf("Welcome to the Wi-Fi setup wizard!\n");
    printf("This wizard will guide you through the process of configuring a new Wi-Fi connection.\n");
    printf("You will be prompted to select a network, enter a password, and configure the connection.\n");
    printf("You can abort the setup at any time by pressing Ctrl-C.\n\n");

    /* Don't get killed by the shell when Ctrl-C is pressed. */
    process_set_attrib(NULL, PROCESS_NOEXIT);

    /* Create a context for the scan */
    wifi_scan_context_t context = {
        .scan_task_handle = xTaskGetCurrentTaskHandle(),
        .num_scan_result = 0,
        .scan_results_queue = xQueueCreate(MAX_SCAN_RESULTS, sizeof(cy_wcm_scan_result_t))
    };

    if (context.scan_results_queue == NULL) 
    {
        printf("Failed to create the scan results queue.\n");
        return -1;
    }

    /* Register the queue for debugging purposes */
    vQueueAddToRegistry(context.scan_results_queue, "ScanResultsQueue");

    /* Initialize Wi-Fi and perform scan */
    if (list_available_networks(&context) == COMMAND_EXIT) 
    {
        vQueueUnregisterQueue(context.scan_results_queue); // Unregister queue
        vQueueDelete(context.scan_results_queue); // Delete queue
        return -1;
    }

    /* Select network from the list */
    while (1) 
    {
        if (get_user_input("Enter the number of the network to use: ", ssid, sizeof(ssid), false) == COMMAND_EXIT) 
        {
            vQueueUnregisterQueue(context.scan_results_queue); /* Unregister queue */
            vQueueDelete(context.scan_results_queue); /* Delete queue */
            return -1;
        }
        selected_network = atoi(ssid);
        if (selected_network > 0 && selected_network <= context.num_scan_result) 
        {
            /* Use the selected network's SSID and security mode */
            selected_result = context.scan_results[selected_network - 1]; /* Access from array */

            strcpy(ssid, (const char*)selected_result.SSID);
            security = selected_result.security;
            
            get_security_mode_string( security_mode, security );

            /* Get user input for password */
            while (1) 
            {
                if (get_user_input("Enter Wi-Fi password: ", password, sizeof(password), true) == COMMAND_EXIT) 
                {
                    vQueueUnregisterQueue(context.scan_results_queue); /* Unregister queue */
                    vQueueDelete(context.scan_results_queue); /* Delete queue */
                    return -1;
                }
                if (strlen(password) > 0) 
                {
                    break;
                }
                printf("Password cannot be empty. Please try again.\n");
            }

            wifi = SYS_SERVICE_WIFI();

            strncpy(wifi->ssid, ssid, WIFI_MAX_SSID_LEN);
            wifi->ssid[WIFI_MAX_SSID_LEN] = '\0';
            strncpy(wifi->password, password, WIFI_MAX_PASSWORD_LEN);
            wifi->password[WIFI_MAX_PASSWORD_LEN] = '\0';
            wifi->security = security;

            /* Save the configuration */
            if (save_config(ssid, password, security_mode) != 0) 
            {
                vQueueUnregisterQueue(context.scan_results_queue); /* Unregister queue */
                vQueueDelete(context.scan_results_queue); /* Delete queue */
                return -1;
            }
            vQueueUnregisterQueue(context.scan_results_queue); /* Unregister queue */
            vQueueDelete(context.scan_results_queue); /* Delete queue */
            return 0;
        }
        printf("Invalid selection. Please try again.\n");
    }
}

/******************************************************************************
* Function Name: list_available_networks
********************************************************************************
* Summary:
*   Lists the available Wi-Fi networks.
*
* Parameters:
*   context - Pointer to the Wi-Fi scan context.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int list_available_networks(wifi_scan_context_t* context) 
{
    cy_wcm_config_t wcm_config = { .interface = CY_WCM_INTERFACE_TYPE_STA };
    cy_rslt_t result = cy_wcm_init(&wcm_config);
    if (result != CY_RSLT_SUCCESS)
    {
        printf("Failed to initialize Wi-Fi Connection Manager. Error code: %ld\n", result);
        return -1;
    }

    printf("Scanning for available networks...\n");

    /* Print header row */
    printf("No  SSID                              Signal Strength  Security\n");
    printf("--------------------------------------------------------------\n");

    /* Start the Wi-Fi scan */
    result = cy_wcm_start_scan(scan_callback, context, NULL);
    if (result != CY_RSLT_SUCCESS) 
    {
        printf("Wi-Fi scan failed. Error code: %ld\n", result);
        return -1;
    }

    /* Wait for scan to complete */
    if (ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(10000)) == 0) 
    { 
        /* Timeout after 10 seconds */
        printf("Wi-Fi scan timed out.\n");
        return -1;
    }

    for (uint32_t i = 0; i < context->num_scan_result; i++) 
    {
        print_scan_result(&context->scan_results[i], i + 1);
    }

    if (context->num_scan_result == 0) 
    {
        printf("No Wi-Fi networks found. Please try again.\n");
        return -1;
    }

    return 0;
}

/******************************************************************************
* Function Name: get_user_input
********************************************************************************
* Summary:
*   Gets input from the user.
*
* Parameters:
*   prompt - The prompt message to display to the user.
*   buffer - The buffer to store the user input.
*   buffer_size - The size of the buffer.
*   mask - Whether to mask the input (e.g., for passwords).
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int get_user_input(char* prompt, char* buffer, size_t buffer_size, bool mask) 
{
    readline_t rl;
    buffer[0] = '\0'; /* clear current input */
    readline_init_ex(&rl, strlen(prompt), 0, buffer, buffer_size, 0, key_press, NULL, false);
    rl.mask = mask;
    while (1) 
    {
        printf("%s", prompt);
        int result = readline(&rl);
        printf("\n"); /* readline doesn't emit a newline on enter or abort */
        switch (result) 
        {
            case COMMAND_ENTER:
            {
                buffer[strcspn(buffer, "\n")] = 0;  /* Remove newline character */
                if (strlen(buffer) == 0) 
                {
                    printf("Input cannot be empty. Please try again.\n");
                    continue;
                }
                return COMMAND_ENTER;
            }

            case COMMAND_EXIT:
            {
                printf("\nOperation canceled by user.\n");
                return COMMAND_EXIT;
            }

            default:
            {
                /* Handle other cases if necessary */
                break;
            }

        }
    }
}

/******************************************************************************
* Function Name: write_escaped
********************************************************************************
* Summary:
*   Writes a string to a file, escaping special characters as needed.
*
* Parameters:
*   f   - The file to write to.
*   str - The string to write.
*
* Return:
*   None.
*
*******************************************************************************/
static void write_escaped(FILE* f, const char* str)
{

    while (*str != '\0')
    {
        if (*str == '"')
        {
            fputc('\\', f);
        }
        else if (*str == '\\' && (*(str + 1) == '"' || *(str + 1) == '\''))
        {
            fputc('\\', f);  /* add extra backslash so tokenizer sees literal \ */
        }
        fputc((unsigned char)*str, f);
        str++;
    }
}

/******************************************************************************
* Function Name: join_argv
********************************************************************************
* Summary:
*   Joins command line arguments into a single string, starting from a specified index.
*
* Parameters:
*   argc       - Argument count.
*   argv       - Argument vector.
*   start      - First index to include (inclusive).
*   buf        - Destination buffer.
*   max_len    - Maximum bytes to write, not counting the null terminator.
*
* Return:
*   Number of bytes written (not including null terminator).
*
*******************************************************************************/
static int join_argv(int argc, char** argv, int start, char* buf, int max_len)
{
    int len = 0;
    for (int i = start; i < argc && len < max_len; i++)
    {
        if (i > start)
        {
            buf[len++] = ' ';
        }
        int n = (int)strlen(argv[i]);
        if (n > max_len - len)
        {
            n = max_len - len;
        }
        memcpy(buf + len, argv[i], n);
        len += n;
    }
    buf[len] = '\0';
    return len;
}

/******************************************************************************
* Function Name: save_config
********************************************************************************
* Summary:
*   Saves the Wi-Fi configuration to a file.
*
* Parameters:
*   ssid - The SSID of the Wi-Fi network.
*   password - The password of the Wi-Fi network.
*   security_mode - The security mode of the Wi-Fi network.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int save_config(const char* ssid, const char* password, const char* security_mode) 
{
    const char* config_file_path = "/system/wifi";
    FILE* config_file;

    config_file = fopen(config_file_path, "w");
    if (!config_file) 
    {
        printf("Failed to open configuration file. %s\n", strerror(errno));
        return -1;
    }

    /* Write Wi-Fi configuration commands */
    fputs("# Wi-Fi configuration script\n", config_file);
    fputs("wifi set_ssid \"", config_file);
    write_escaped(config_file, ssid);
    fputs("\"\n", config_file);
    fputs("wifi set_passwd \"", config_file);
    write_escaped(config_file, password);
    fputs("\"\n", config_file);
    fprintf(config_file, "wifi set_security %s\n", security_mode);
    fputs("wifi start\n", config_file);

    if (ferror(config_file))
    {
        printf("Failed to write configuration file. %s\n", strerror(errno));
        fclose(config_file);
        return -1;
    }

    fclose(config_file);
    printf("Configuration saved successfully. You may now reboot the board, or run command 'wifi start'\n");
    return 0;
}

/******************************************************************************
* Function Name: scan_callback
********************************************************************************
* Summary:
*   Callback function for handling WiFi scan results.
*
* Parameters:
*   result_ptr - Pointer to the scan result.
*   user_data - Pointer to user data (wifi_scan_context_t).
*   status - Status of the scan.
*
* Return:
*   None.
*
*******************************************************************************/
static void scan_callback(cy_wcm_scan_result_t* result_ptr, void* user_data, cy_wcm_scan_status_t status) 
{
    /* Cast user_data to wifi_scan_context_t */
    wifi_scan_context_t* context = (wifi_scan_context_t*)user_data; 

    if ((strlen((const char*)result_ptr->SSID) != 0) && (status == CY_WCM_SCAN_INCOMPLETE)) 
    {
        if (context->num_scan_result < MAX_SCAN_RESULTS) 
        {
            /* Store the result in array */
            context->scan_results[context->num_scan_result] = *result_ptr; 
            context->num_scan_result++;
        }
    }

    if (CY_WCM_SCAN_COMPLETE == status) 
    {
        /* Notify that scan has completed */
        xTaskNotifyGive(context->scan_task_handle);
    }
}

/******************************************************************************
* Function Name: print_scan_result
********************************************************************************
* Summary:
*   Prints the Wi-Fi scan results.
*
* Parameters:
*   result - Pointer to the scan result.
*   num_scan_result - The number of scan results.
*
* Return:
*   None.
*
*******************************************************************************/
static void print_scan_result(cy_wcm_scan_result_t* result, uint32_t num_scan_result) 
{
    char* security_type_string;

    switch (result->security) 
    {
        case CY_WCM_SECURITY_OPEN:
        {
            security_type_string = SECURITY_OPEN;
            break;  
        }

        case CY_WCM_SECURITY_WEP_PSK:
        {
            security_type_string = "WEP_PSK";
            break;
        }

        case CY_WCM_SECURITY_WPA_TKIP_PSK:
        {
            security_type_string = "WPA_TKIP_PSK";
            break;
        }
        case CY_WCM_SECURITY_WPA_AES_PSK:
        {
            security_type_string = "WPA_AES_PSK";
            break;
        }
        case CY_WCM_SECURITY_WPA2_AES_PSK:
        {
            security_type_string = SECURITY_WPA2_AES_PSK;
            break;
        }
        case CY_WCM_SECURITY_WPA2_TKIP_PSK:
        {
            security_type_string = "WPA2_TKIP_PSK";
            break;
        }
        case CY_WCM_SECURITY_WPA2_MIXED_PSK:
        {
            security_type_string = SECURITY_WPA2_MIXED_PSK;
            break;
        }
        case CY_WCM_SECURITY_WPA3_SAE:
        {
            security_type_string = SECURITY_WPA3_SAE;
            break;
        }
        case CY_WCM_SECURITY_WPA3_WPA2_PSK:
        {
            security_type_string = SECURITY_WPA3_WPA2_PSK;
            break;
        }
        default:
        {
            security_type_string = "UNKNOWN";
            break;
        }
    }

    printf(" %2lu   %-32s     %4d          %-15s\n",
           num_scan_result, result->SSID,
           result->signal_strength, security_type_string);
}

/******************************************************************************
* Function Name: key_press
********************************************************************************
* Summary:
*   Handles key presses.
*
* Parameters:
*   rl - Pointer to the readline context.
*   key - The key that was pressed.
*
* Return:
*   COMMAND_EXIT if Ctrl-C is pressed, otherwise 0.
*
*******************************************************************************/
static int key_press(readline_t* rl, char key) 
{
    switch (key) 
    {
        case 3: /* Ctrl-C */
        {
            return COMMAND_EXIT;
        }

        default:
        {
            return 0;
        }

    }
}

/******************************************************************************
* Function Name: set_ssid
********************************************************************************
* Summary:
*   Sets the Wi-Fi SSID.
*
* Parameters:
*   ssid - The SSID to set.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int set_ssid(char* ssid) 
{
    wifi_t* wifi = SYS_SERVICE_WIFI();

    strncpy(wifi->ssid, ssid, WIFI_MAX_SSID_LEN);
    wifi->ssid[WIFI_MAX_SSID_LEN] = '\0';  /* Ensure null-termination for max-length SSIDs */
    printf("Wi-Fi SSID updated. Change will take effect on next connect.\n");
    return 0;
}

/******************************************************************************
* Function Name: set_password
********************************************************************************
* Summary:
*   Sets the Wi-Fi password.
*
* Parameters:
*   password - The password to set.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int set_password(char* password) 
{
    wifi_t* wifi = SYS_SERVICE_WIFI();

    strncpy(wifi->password, password, WIFI_MAX_PASSWORD_LEN);
    wifi->password[WIFI_MAX_PASSWORD_LEN] = '\0';  /* Ensure null-termination */
    printf("Wi-Fi password updated. Change will take effect on next connect.\n");
    return 0;
}

/******************************************************************************
* Function Name: set_security
********************************************************************************
* Summary:
*   Sets the Wi-Fi security mode.
*
* Parameters:
*   security - The security mode to set.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int set_security(cy_wcm_security_t security) 
{
    wifi_t* wifi = SYS_SERVICE_WIFI();

    wifi->security = security;
    printf("Wi-Fi security updated. Change will take effect on next connect.\n");
    return 0;
}

/******************************************************************************
* Function Name: save
********************************************************************************
* Summary:
*   Saves the current Wi-Fi configuration.
*
* Parameters:
*   None
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int save() 
{
    wifi_t* wifi = SYS_SERVICE_WIFI();
    
    char security_mode[32];
    get_security_mode_string( security_mode, wifi->security );
    
    if (save_config(wifi->ssid, wifi->password, security_mode) != 0)
    {
       printf("Failed to save Wi-Fi configuration\n"); 
       return -1;
    }

    printf("Wi-Fi settings saved.\n");
    return 0;
}

/******************************************************************************
* Function Name: wifi_start_cmd
********************************************************************************
* Summary:
*   Starts the Wi-Fi service.
*
* Parameters:
*   None
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int wifi_start_cmd() 
{
    wifi_t* wifi = SYS_SERVICE_WIFI();
    
    int failcount=0;
    while(!wifi_start(wifi)) 
    {
        printf("Wi-Fi start will retry in 2 seconds.\n");
        vTaskDelay(pdMS_TO_TICKS(2000));
        failcount++;
        if (failcount == 3 )
        {
            printf("Wi-Fi service start failed.\n");
            return -1;
        }
    }

    return 0;
}

/******************************************************************************
* Function Name: wifi_stop_cmd
********************************************************************************
* Summary:
*   Stops the Wi-Fi service.
*
* Parameters:
*   None
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int wifi_stop_cmd() 
{
    wifi_t* wifi = SYS_SERVICE_WIFI();
    if(!wifi_stop(wifi)) 
    {
        printf("Wi-Fi service stop failed.\n");
        return -1;
    }

    return 0;
}

#endif /* IM_ENABLE_NET */

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */