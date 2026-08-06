/******************************************************************************
* File Name: dsave.c
*
* Description: Implementation of the 'dsave' command for the shell.
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
#include <stdarg.h>
#include "../lib/getopt.h"
#include "../../services.h"
#include "../lib/tsp_utils.h"
#include "../../protocol/protocol.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: replace_spaces_with_underscores
********************************************************************************
* Summary:
*   Replaces all spaces in a string with underscores.
*
* Parameters:
*   str: The string to be modified.
*
* Return:
*   None.
*
*******************************************************************************/
void replace_spaces_with_underscores(char* str) 
{
    for (char* p = str; *p; p++) 
    {
        if (*p == ' ') 
        {
            *p = '_';
        }
    }
}

/******************************************************************************
* Function Name: write_text_line
********************************************************************************
* Summary:
*   Writes a formatted line to a file.
*
* Parameters:
*   file: The file to write to.
*   format: The format string.
*   ...: Additional arguments for the format string.
*
* Return:
*   true if the line was written successfully, false otherwise.
*
*******************************************************************************/
static bool write_text_line(FILE* file, const char *restrict format, ...) 
{
    va_list args;
    va_start(args, format);

    /* Use a fixed-size buffer */
    char line[255];
    vsnprintf(line, sizeof(line), format, args);
    va_end(args);

    /* Write the formatted string to the file */
    size_t res = fwrite(line, 1, strlen(line), file);
    return (res == strlen(line));
}

/******************************************************************************
* Function Name: main_dsave
********************************************************************************
* Summary:
*   Saves the configuration of a device to a file.
*
* Parameters:
*   argc: The number of command-line arguments.
*   argv: The array of command-line arguments.
*
* Return:
*   0 if the configuration was saved successfully, -1 otherwise.
*
*******************************************************************************/
int main_dsave(int argc, char** argv) 
{
    char default_path[255];
    char value_buf[255];

    if (argc != 2 && argc != 3) 
    {
        printf("Usage: dsave <device> [file]\n");
        printf("If file is omitted, the configuration will be saved\n");
        printf("to /system/devices/<device_name> and will be loaded on bootup.\n");
        return -1;
    }

    protocol_t* protocol = SYS_SERVICE_PROTOCOL();
    protocol_Device* device = tsp_find(protocol, argv[1]);

    if (!device) 
    {
        printf("Error: No device named or index '%s' found.\n", argv[1]);
        return -1;
    }

    snprintf(default_path, sizeof(default_path), "/system/devices/%s", device->name);
    replace_spaces_with_underscores(default_path);

    const char* filepath = (argc == 3) ? argv[2] : default_path;
    if (argc == 3) 
    {
        replace_spaces_with_underscores((char*)filepath);
    }

    FILE* file = fopen(filepath, "w");
    if (!file) 
    {
        printf("Error: Unable to open file '%s' for writing.\n", filepath);
        return -1;
    }

    /* Write the initial echo line */
    if (!write_text_line(file, "echo \"Configures device %s\"\n", device->name)) 
    {
        printf("Error writing echo line to file.\n");
        fclose(file);
        return -1;
    }

    for (int i = 0; i < device->options_count; i++) 
    {
        protocol_Option* option = &device->options[i];

        /* Skip blob and password types */
        if (option->which_value == protocol_Option_blob_type_tag || option->which_value == protocol_Option_password_type_tag) 
        {
            continue;
        }

        tsp_option_get_value_str(option, true, value_buf, sizeof(value_buf));

        /* Determine if the value needs quotes */
        if (option->which_value == protocol_Option_oneof_type_tag || option->which_value == protocol_Option_string_type_tag) 
        {
            if (!write_text_line(file, "dset -s \"%s\" \"%s\" \"%s\"\n", device->name, option->name, value_buf)) 
            {
                printf("Error writing option line to file.\n");
                fclose(file);
                return -1;
            }
        } 
        else 
        {
            if (!write_text_line(file, "dset -s \"%s\" \"%s\" %s\n", device->name, option->name, value_buf)) 
            {
                printf("Error writing option line to file.\n");
                fclose(file);
                return -1;
            }
        }
    }

    /* Add the final dapply line */
    if (!write_text_line(file, "dapply -s \"%s\"\n", device->name)) 
    {
        printf("Error writing dapply line to file.\n");
        fclose(file);
        return -1;
    }

    fclose(file);
    printf("Device configuration saved to %s\n", filepath);
    return 0;
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */