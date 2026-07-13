/******************************************************************************
* File Name: dset.c
*
* Description: Implementation of the 'dset' command for the shell.
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
#include "../../services.h"
#include "../../protocol/protocol.h"
#include "../lib/tsp_utils.h"
#include "../../heap.h"
#include "../lib/strutils.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/
/******************************************************************************
* Function Name: device_set_option
********************************************************************************
* Summary:
*   Sets a device option based on the provided string value. The function parses
*
* Parameters:
*   option: The device option to be set.
*   str: The string representation of the value to set.
*
* Return:
*   true if the operation is successful, otherwise false.
*
*******************************************************************************/
bool device_set_option(protocol_Option *option, char* str) 
{

    if(str == NULL)
    {
        return false;
    }

    switch(option->which_value) 
    {
        case protocol_Option_int_type_tag:
        {
            if(!is_int(str)) 
            {
                printf("Unable to parse '%s' as an integer.\n", str);
                return false;
            }

            int int_value = atoi(str);
            int min = option->value.int_type.min_value;
            int max = option->value.int_type.max_value;

            if(int_value < min || int_value > max) 
            {
                printf("Values must be between %d and %d.\n", min, max);
                return false;
            }
            option->value.int_type.current_value = int_value;
            return true;
        }

        case protocol_Option_float_type_tag:
        {
            if(!is_float(str)) 
            {
                printf("Unable to parse '%s' as a decimal.\n", str);
                return false;
            }

            float float_value = strtof(str, NULL);
            float min = option->value.float_type.min_value;
            float max = option->value.float_type.max_value;

            if(float_value < min || float_value > max) 
            {
                printf("Values must be between %f and %f.\n", min, max);
                return false;
            }
            option->value.float_type.current_value = float_value;
            return true;
        }

        case protocol_Option_bool_type_tag:
        {
            if(strcmp("1", str)  == 0
            || strcmp("yes", str)  == 0
            || strcmp("true", str) == 0
            || strcmp("on", str) == 0) 
            {
                 option->value.bool_type.current_value = true;
                 return true;
            } 
            else if(strcmp("0", str)  == 0
            || strcmp("no", str)  == 0
            || strcmp("false", str) == 0
            || strcmp("off", str) == 0) 
            {
                 option->value.bool_type.current_value  = false;
                 return true;
            } 
            else 
            {
                printf("Unable to parse '%s' as an boolean.\n", str);
                return false;
            }
        }

        case protocol_Option_oneof_type_tag:
        {
            int item_count =  option->value.oneof_type.items_count;
            /* Check for matching name */ 
            for(int i = 0; i < item_count; i++) 
            {
                char* candidate = option->value.oneof_type.items[i];
                if(strcasecmp(str, candidate) == 0) 
                {
                    option->value.oneof_type.current_index = i;
                    return true;
                }
            }

            /* Check if it is an index. */
            if(is_int(str)) 
            {
                int index = atoi(str);

                if(index < 0 || index >= item_count) 
                {
                    printf("No such option index.\n");
                    return false;
                }

                option->value.oneof_type.current_index = index;
                return true;
            }

            printf("Option must either be an index or the name of the option.\n");
            printf("Valid options are: ");

            for(int i = 0; i < item_count; i++) 
            {
                printf("\"%s\"", option->value.oneof_type.items[i]);
                if(i < (item_count -1))
                {
                    printf(", ");
                }

            }
            printf("\n");
            return false;
        }

        case protocol_Option_blob_type_tag:
        {
            // todo: assume str is a filepath
            printf("Blob values not implemented\n");

            return false;
        }

        case protocol_Option_string_type_tag:
        {
            int len = strlen(str) + 1;
            char* copy = (char *)pmem_malloc(len);
            strcpy(copy, str);

            if(option->value.string_type.current_value != NULL)
            {
                pmem_free(option->value.string_type.current_value);
            }

            option->value.string_type.current_value = copy;
            return true;
        }

        case protocol_Option_password_type_tag:
        {
            int len = strlen(str) + 1;
            char* copy = (char *)pmem_malloc(len);
            strcpy(copy, str);
            if(option->value.password_type.current_value != NULL)
            {
                pmem_free(option->value.password_type.current_value);
            }

            option->value.password_type.current_value = copy;
            return true;
        }
        default:
            return false;
    }

    return true;
}

/******************************************************************************
* Function Name: main_dset
********************************************************************************
* Summary:
*   Main function for the 'dset' command. This function parses command-line arguments
*   and sets the specified device option to the provided value.
*
* Parameters:
*   argc: The number of command-line arguments.
*   argv: The array of command-line argument strings.
*
* Return:
*   0 if the operation is successful, otherwise -1.
*
*******************************************************************************/
int main_dset(int argc, char** argv) 
{
    int opt;
    bool silent = false;
    const char* usage_text = "Usage: %s [-h|-s] <device> <option> <value>\n"
                             "Sets a device option.\n"
                             "\n"
                             "<device> and <option> can either be the name or number.\n"
                             "If <option> is a OneOf type, <value> can be either an index or value.\n"
                             "\n"
                             "  -h             Show this help message and exit.\n"
                             "  -s             Silent, less chatty output.\n";

    getopt_t go;
    getopt_init(&go);

    if (argc < 4) {
        printf(usage_text, argv[0]);
        return -1;
    }

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

    char* device_str = argv[go.optind];
    char* option_str = argv[go.optind+1];
    char* value_str = argv[go.optind+2];

    protocol_t* protocol = SYS_SERVICE_PROTOCOL();

    protocol_Device *device = tsp_find(protocol, device_str);
    if(!device) 
    {
        printf("Error: No device named or index '%s' found.\n", device_str);
        return -1;
    }

    protocol_Option *option = tsp_option_find(device, option_str);
    if(!option) 
    {
        printf("Error: No option named or id '%s' found.\n", option_str);
        return -1;
    }

    if(!device_set_option(option, value_str))
    {
        return -1;     
    }

    if(!silent) 
    {
        char buffer[255];
        printf("Updated option %s to %s\n", option->name,
            tsp_option_get_value_str(option, true, buffer, sizeof(buffer)));

        printf("Note! Don't forget to run the command 'dapply <device>' for the changes to take effect.\n");
    }
    return 0;
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */