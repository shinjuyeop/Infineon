/******************************************************************************
* File Name: dlist.c
*
* Description: Implementation of the 'dlist' command for the shell.
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
#include <math.h>

#include "../lib/getopt.h"
#include "../../protocol/protocol.h"
#include "../lib/tsp_utils.h"
#include "../../services.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/
/******************************************************************************
* Function Name: main_dlist
********************************************************************************
* Summary:
*   Implementation of the 'dlist' command for the shell.
*
* Parameters:
*   argc: Argument count.
*   argv: Argument vector.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
int main_dlist(int argc, char** argv) 
{
    int opt;
    const char* usage_text = "Usage: %s [-h] [device]\n"
                             "If no device is given all devices are listed.\n"
                             "If a device number is given its options and streams are shown.\n"
                             "  -h   Display this help and exit\n";
    getopt_t go;
    getopt_init(&go);


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

    protocol_t* protocol = SYS_SERVICE_PROTOCOL();

    if (argc == 1) 
    {
        printf("%-6s %-20s %-7s %-8s %s\n", "Device", "Name", "Status", "Type", "Description");

            for(int i = 0; i < protocol->board.devices_count; i++) 
            {
                struct _protocol_Device *device = &protocol->board.devices[i];
                printf("%-6ld %-20s %-7s %-8s %s\n",
                        device->device_id,
                        device->name,
                        tsp_get_status_str(device->status),
                        tsp_get_type_str(device->type),
                        device->description);
            }

            printf("\nUsage: %s [device] for detailed device information.\n", argv[0]);
            return 0;
    } 
    else if (argc == 2) 
    {
        protocol_Device *device = tsp_find(protocol, argv[go.optind]);
        if(!device)
            return -1;

        printf("Device             %s\n", device->name);
        printf("Type               %s\n", tsp_get_type_str(device->type));
        printf("Status             %s\n", tsp_get_status_str(device->status));
        if(device->status_message != NULL && strlen(device->status_message) > 0)
            printf("Message            %s\n", device->status_message);
        if(device->description != NULL && strlen(device->description) > 0)
            printf("Description        %s\n", device->description);

        char tmp_buf[255];
        for(int i = 0; i < device->options_count; i++) 
        {
            protocol_Option *option = &device->options[i];
            const char* default_str = tsp_option_is_default(option) ? "(Default)" : "";

            printf("[Option %ld]\n", option->option_id);
            printf("  Name             %s\n", option->name);
            printf("  Description      %s\n", option->description);
            printf("  Value            %s %s\n", tsp_option_get_value_str(option, true, tmp_buf, sizeof(tmp_buf)), default_str);
            
            switch(option->which_value) 
            {
                case protocol_Option_oneof_type_tag:
                {
                    printf("  One of           ");
                    int item_count = option->value.oneof_type.items_count;

                    for(int j = 0; j < item_count; j++) 
                    {
                        printf("'%s'", option->value.oneof_type.items[j]);
                        if(j < (item_count - 1))
                        {
                            printf(", ");
                        }

                    }
                    printf("\n");
                    break;
                }

                case protocol_Option_int_type_tag:
                {
                    if(option->value.int_type.min_value != INT32_MIN)
                    {
                        printf("  Min              %ld\n", option->value.int_type.min_value);
                    }

                    if(option->value.int_type.max_value != INT32_MAX)
                    {
                        printf("  Max              %ld\n", option->value.int_type.max_value);
                    }
                    break;
                }


                case protocol_Option_float_type_tag:
                {
                    if(!isnan(option->value.float_type.min_value))
                    {
                        printf("  Min              %f\n", option->value.float_type.min_value);
                    }

                    if(!isnan(option->value.float_type.max_value))
                    {
                        printf("  Max              %f\n", option->value.float_type.max_value);
                    }
                    break;
                }

                default:
                {
                    break;
                }

            }
        }

        for(int i = 0; i < device->streams_count; i++) 
        {
            protocol_StreamConfig *stream = &device->streams[i];
            printf("[Stream %d]\n", i);
            printf("  Current Frame    %ld\n", stream->current_frame);
            printf("  Frames Dropped   %ld\n", stream->frames_dropped);
            printf("  Name             %s\n", stream->name);
            printf("  Direction        %s\n", tsp_stream_get_direction_str(stream->direction));

            if(stream->frequency != 0)
            {
                printf("  Frequency        %f\n", stream->frequency);
            }

            printf("  Type             %s\n", tsp_stream_get_type_str(stream->datatype));

            if(stream->datatype == protocol_DataType_DATA_TYPE_D8
              || stream->datatype == protocol_DataType_DATA_TYPE_D16
              || stream->datatype == protocol_DataType_DATA_TYPE_D32) 
            {
                printf("  Scale            %f\n", stream->scale);
                printf("  Offset           %llu\n", stream->offset);
            }

            if(stream->datatype == protocol_DataType_DATA_TYPE_Q7
                          || stream->datatype == protocol_DataType_DATA_TYPE_Q15
                          || stream->datatype == protocol_DataType_DATA_TYPE_Q31) 
            {
                printf("  Shift            %ld\n", stream->shift);
            }

            if(stream->unit && strlen(stream->unit) > 0)
            {
                printf("  Unit             %s\n", stream->unit);
            }

            printf("  Shape            %s\n", tsp_stream_get_shape(stream, tmp_buf, sizeof(tmp_buf)));
            printf("  Dimension Names  %s\n", tsp_stream_get_shape_dim_names(stream, tmp_buf, sizeof(tmp_buf)));
            printf("  Shape Labels     %s\n", tsp_stream_get_shape_labels(stream, tmp_buf, sizeof(tmp_buf)));
            printf("  Max Frame Count  %ld\n", stream->max_frame_count);
        }

        return 0;

    } 
    else
    {
        printf(usage_text, argv[0]);
        return -1;
    }
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */