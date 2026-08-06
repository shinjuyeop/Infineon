/******************************************************************************
* File Name:   tsp_utils.c
*
* Description: Utility functions for TSP (Transport Stream Protocol).
*              This file provides the implementations for utility functions 
*              used for common tasks such as getting string representations
*              of device types, statuses, and stream directions.
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
#include <stdbool.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <stdarg.h>

#include "strutils.h"
#include "../../heap.h"
#include "../../protocol/protocol.h"
#include "../../protocol/model.pb.h"
#include "../../protocol/protocol.pb.h"
#include "../../protocol/pb_decode.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: tsp_get_type_str
********************************************************************************
* Summary:
*    Returns a string representation of the given device type.
*
* Parameters:
*   type - The device type.
*
* Return:
*   A string representation of the device type.
*
*******************************************************************************/
const char* tsp_get_type_str(protocol_DeviceType type) 
{
    switch(type) 
    {
        case protocol_DeviceType_DEVICE_TYPE_SENSOR:   
        {
            return "Sensor";
        }
        case protocol_DeviceType_DEVICE_TYPE_PLAYBACK: 
        {
            return "Playback";
        }
        case protocol_DeviceType_DEVICE_TYPE_MODEL:    
        {
            return "Model";
        }
        case protocol_DeviceType_DEVICE_TYPE_OTHER:    
        {
            return "Other";
        }
        case protocol_DeviceType_DEVICE_TYPE_LOOPBACK: 
        {
            return "Loopback";
        }
        case protocol_DeviceType_DEVICE_TYPE_BOARD:    
        {
            return "Board";
        }
        default: 
        {
            return "invalid";
        }
    }
}

/******************************************************************************
* Function Name: tsp_get_status_str
********************************************************************************
* Summary:
*    Returns a string representation of the given device status.
*
* Parameters:
*   status - The device status.
*
* Return:
*   A string representation of the device status.
*
*******************************************************************************/
const char* tsp_get_status_str(protocol_DeviceStatus status) 
{
    switch(status) 
    {
        case protocol_DeviceStatus_DEVICE_STATUS_READY:       
        {
            return "Ready";
        }
        case protocol_DeviceStatus_DEVICE_STATUS_ACTIVE:      
        {
            return "Active";
        }
        case protocol_DeviceStatus_DEVICE_STATUS_ACTIVE_WAIT: 
        {
            return "Wait";
        }
        case protocol_DeviceStatus_DEVICE_STATUS_ERROR:       
        {
            return "Error";
        }
        default: 
        {
            return "invalid";
        }
    }
}

/******************************************************************************
* Function Name: tsp_option_get_type_str
********************************************************************************
* Summary:
*    Returns a string representation of the given option type.
*
* Parameters:
*   type - The option type.
*
* Return:
*   A string representation of the option type.
*
*******************************************************************************/
const char* tsp_option_get_type_str(int type) 
{
    switch(type) 
    {
        case protocol_Option_int_type_tag:      
        {
            return "Integer";
        }
        case protocol_Option_float_type_tag:    
        {
            return "Decimal";
        }
        case protocol_Option_bool_type_tag:     
        {
            return "Boolean";
        }
        case protocol_Option_oneof_type_tag:    
        {
            return "OneOf";
        }
        case protocol_Option_blob_type_tag:     
        {
            return "Binary";
        }
        case protocol_Option_string_type_tag:   
        {
            return "String";
        }
        case protocol_Option_password_type_tag: 
        {
            return "Secret";
        }
        default:                                
        {
            return "Invalid";
        }
    }
}

/******************************************************************************
* Function Name: tsp_stream_get_direction_str
********************************************************************************
* Summary:
*    Returns a string representation of the given stream direction.
*
* Parameters:
*   direction - The stream direction.
*
* Return:
*   A string representation of the stream direction.
*
*******************************************************************************/
const char* tsp_stream_get_direction_str(protocol_StreamDirection direction) 
{
    switch(direction) 
    {
        case protocol_StreamDirection_STREAM_DIRECTION_INPUT:  
        {
            return "input";
        }
        case protocol_StreamDirection_STREAM_DIRECTION_OUTPUT: 
        {
            return "output";
        }
        default: 
        {
            return "(invalid)";
        }
    }
}

/******************************************************************************
* Function Name: tsp_stream_get_type_str
********************************************************************************
* Summary:
*    Returns a string representation of the given stream data type.
*
* Parameters:
*   type - The stream data type.
*
* Return:
*   A string representation of the stream data type.
*
*******************************************************************************/
const char* tsp_stream_get_type_str(protocol_DataType type) 
{
    switch(type) 
    {
        case protocol_DataType_DATA_TYPE_U8:  
        {
            return "U8";
        }
        case protocol_DataType_DATA_TYPE_S8:  
        {
            return "S8";
        }
        case protocol_DataType_DATA_TYPE_U16: 
        {
            return "U16";
        }
        case protocol_DataType_DATA_TYPE_S16: 
        {
            return "S16";
        }
        case protocol_DataType_DATA_TYPE_U32: 
        {
            return "U32";
        }
        case protocol_DataType_DATA_TYPE_S32: 
        {
            return "S32";
        }
        case protocol_DataType_DATA_TYPE_F32: 
        {
            return "F32";
        }
        case protocol_DataType_DATA_TYPE_F64: 
        {
            return "F64";
        }
        case protocol_DataType_DATA_TYPE_Q7:  
        {
            return "Q7";
        }
        case protocol_DataType_DATA_TYPE_Q15: 
        {
            return "Q15";
        }
        case protocol_DataType_DATA_TYPE_Q31: 
        {
            return "Q31";
        }
        case protocol_DataType_DATA_TYPE_D8:  
        {
            return "D8";
        }
        case protocol_DataType_DATA_TYPE_D16: 
        {
            return "D16";
        }
        case protocol_DataType_DATA_TYPE_D32: 
        {
            return "D32";
        }
        default: 
        {
            return "(invalid)";
        }
    }
}

/******************************************************************************
* Function Name: tsp_find
********************************************************************************
* Summary:
*    Finds a device by its name or ID.
*
* Parameters:
*   protocol - The protocol instance.
*   token - The device name or ID.
*
* Return:
*   A pointer to the device if found, NULL otherwise.
*
*******************************************************************************/
protocol_Device* tsp_find(protocol_t* protocol, char* token) 
{
    if(token == NULL || *token == '\0')
    {
        return NULL;
    }

    for(int i = 0; i < protocol->board.devices_count; i++) 
    {
        struct _protocol_Device *device = &protocol->board.devices[i];
        if(strcasecmp(device->name, token) == 0)
        {
            return device;
        }

    }

    if(is_int(token)) 
    {
        int device_num = atoi(token);

        if(device_num < 0 || device_num >= protocol->board.devices_count)
        {
            return NULL;
        }

        return &protocol->board.devices[device_num];
    }

    return NULL;
}

/******************************************************************************
* Function Name: tsp_option_find
********************************************************************************
* Summary:
*    Finds an option by its name or ID.
*
* Parameters:
*   device - The device instance.
*   token - The option name or ID.
*
* Return:
*   A pointer to the option if found, NULL otherwise.
*
*******************************************************************************/
protocol_Option* tsp_option_find(protocol_Device *device, char* token) 
{
    if(token == NULL || *token == '\0')
    {
        return NULL;
    }

    int id = -1;
    if(is_int(token))
    {
        id = atoi(token);
    }

    for(int i = 0; i < device->options_count; i++) 
    {
        protocol_Option *option = &device->options[i];
        if(strcasecmp(option->name, token) == 0 || option->option_id == id)
        {
            return option;
        }

    }
    return NULL;
}

/******************************************************************************
* Function Name: tsp_option_get_value_str
********************************************************************************
* Summary:
*    Gets the value of an option as a string.
*
* Parameters:
*   option - The option instance.
*   current - Whether to get the current value or the default value.
*   str - The buffer to store the string.
*   size - The size of the buffer.
*
* Return:
*   A pointer to the string buffer containing the option value.
*
*******************************************************************************/
char* tsp_option_get_value_str(protocol_Option* option, bool current, char* str, size_t size) 
{
    switch(option->which_value) 
    {
        case protocol_Option_int_type_tag:
        {
            snprintf(str, size, "%ld", current ? option->value.int_type.current_value : option->value.int_type.default_value);
            return str;
        }

        case protocol_Option_float_type_tag:
        {
            snprintf(str, size, "%f", current ? option->value.float_type.current_value : option->value.float_type.default_value);
            return str;
        }

        case protocol_Option_bool_type_tag:
        {
            bool bool_value = current ? option->value.bool_type.current_value : option->value.bool_type.default_value;
            snprintf(str, size, "%s", bool_value ? "yes" : "no");
            return str;
        }

        case protocol_Option_oneof_type_tag:
        {
            int32_t index = current ? option->value.oneof_type.current_index : option->value.oneof_type.default_index;
            if(index < 0 || index >= option->value.oneof_type.items_count) {
                snprintf(str, size, "Invalid");
                return str;
            }
            snprintf(str, size, "%s", option->value.oneof_type.items[index]);
            return str;
        }

        case protocol_Option_blob_type_tag:
        {
            pb_bytes_array_t* data = current ? option->value.blob_type.current_value : option->value.blob_type.default_value;
            snprintf(str, size, "(%d bytes)", data->size);
            return str;
        }

        case protocol_Option_string_type_tag:
        {
            char* value_str = current ? option->value.string_type.current_value : option->value.string_type.default_value;
            snprintf(str, size, "%s", value_str);
            return str;
        }

        case protocol_Option_password_type_tag:
        {
            snprintf(str, size, "(hidden)");
            return str;
        }

        default:
        {
            snprintf(str, size, "(invalid)");
            return str;
        }

    }
}

/******************************************************************************
* Function Name: _append_to_buffer
********************************************************************************
* Summary:
*    Appends formatted data to a buffer.
*
* Parameters:
*   str - The buffer to store the formatted data.
*   size - The size of the buffer.
*   format - The format string.
*   ... - Additional arguments for the format string.
*
* Return:
*   The number of characters written to the buffer.
*
*******************************************************************************/
static int _append_to_buffer(char* str, ssize_t* size, const char* format, ...) 
{
    if (*size <= 0)
    {
        return 0;
    }

    va_list args;
    va_start(args, format);
    int n = vsnprintf(str, *size, format, args);
    va_end(args);

    if (n >= *size) 
    {
        str[*size - 1] = '\0';
        *size = 0;
        return 0;
    }

    *size -= n;
    return n;
}

/******************************************************************************
* Function Name: tsp_stream_get_shape
********************************************************************************
* Summary:
*    Gets the shape of a stream as a string.
*
* Parameters:
*   stream - The stream instance.
*   str - The buffer to store the shape string.
*   size - The size of the buffer.
*
* Return:   A pointer to the string buffer containing the stream shape.
*   
*
*******************************************************************************/
char* tsp_stream_get_shape(protocol_StreamConfig* stream, char* str, ssize_t size) 
{
    char* ret = str;
    if (size <= 0)
    {
        return NULL;
    }

    int n = 0;
    if ((n = _append_to_buffer(str, &size, "[")) > 0)
    {
        str += n;
    }

    for (int i = 0; i < stream->shape_count && size > 0; i++) 
    {
        if ((n = _append_to_buffer(str, &size, "%d", stream->shape[i].size)) > 0)
        {
            str += n;
        }

        if (i < stream->shape_count - 1 && size > 0) 
        {
            if ((n = _append_to_buffer(str, &size, ",")) > 0)
            {
                str += n;
            }
        }
    }

    if (size > 0 && (n = _append_to_buffer(str, &size, "]")) > 0)
    {
        str += n;
    }

    *str = '\0';
    return ret;
}

/******************************************************************************
* Function Name: tsp_stream_get_shape_dim_names
********************************************************************************
* Summary:
*    Gets the dimension names of a stream as a string.
*
* Parameters:
*   stream - The stream instance.
*   str - The buffer to store the dimension names string.
*   size - The size of the buffer.
*
* Return:   A pointer to the string buffer containing the stream dimension names.
*   
*
*******************************************************************************/
char* tsp_stream_get_shape_dim_names(protocol_StreamConfig* stream, char* str, ssize_t size) 
{
    if (size <= 0)
    {
        return NULL;
    }

    char* ret = str;
    int n = 0;
    for (int i = 0; i < stream->shape_count && size > 0; i++) 
    {
        if ((n = _append_to_buffer(str, &size, "%s", stream->shape[i].name)) > 0)
        {
            str += n;
        }

        if (i < stream->shape_count - 1 && size > 0) 
        {
            if ((n = _append_to_buffer(str, &size, ", ")) > 0)
            {
                str += n;
            }
        }
    }
    *str = '\0';
    return ret;
}

/******************************************************************************
* Function Name: tsp_stream_get_shape_labels
********************************************************************************
* Summary:
*    Gets the labels of a stream's dimensions as a string.
*
* Parameters:
*   stream - The stream instance.
*   str - The buffer to store the dimension labels string.
*   size - The size of the buffer.
*
* Return:   A pointer to the string buffer containing the stream dimension labels.
*   
*
*******************************************************************************/
char* tsp_stream_get_shape_labels(protocol_StreamConfig* stream, char* str, ssize_t size) 
{
    if (size <= 0) 
    {
        return NULL;
    }

    int n = 0;
    char* ret = str;
    if ((n = _append_to_buffer(str, &size, "{")) > 0)
    {
        str += n;
    }

    for (int i = 0; i < stream->shape_count && size > 0; i++) 
    {
        if (stream->shape[i].labels_count == 0) 
        {
            if ((n = _append_to_buffer(str, &size, "{...}")) > 0)
            {
                str += n;
            }
        } 
        else 
        {
            if ((n = _append_to_buffer(str, &size, "{")) > 0)
            {
                str += n;
            }

            for (int j = 0; j < stream->shape[i].labels_count && size > 0; j++) 
            {
                if ((n = _append_to_buffer(str, &size, "\"%s\"", stream->shape[i].labels[j])) > 0)
                {
                    str += n;
                }

                if (j < stream->shape[i].labels_count - 1 && size > 0) 
                {
                    if ((n = _append_to_buffer(str, &size, ",")) > 0)
                    {
                        str += n;
                    }
                }
            }

            if (size > 0 && (n = _append_to_buffer(str, &size, "}")) > 0)
            {
                str += n;
            }

        }

        if (i < stream->shape_count - 1 && size > 0) 
        {
            if ((n = _append_to_buffer(str, &size, ",")) > 0)
            {
                str += n;
            }
        }
    }

    if (size > 0 && (n = _append_to_buffer(str, &size, "}")) > 0)
    {
        str += n;
    }

    *str = '\0';
    return ret;
}

/******************************************************************************
* Function Name: tsp_option_is_default
********************************************************************************
* Summary:
*    Checks if an option is set to its default value.
*
* Parameters:
*   option - The option instance.
*
* Return:   true if the option is set to its default value, false otherwise.
*   
*
*******************************************************************************/
bool tsp_option_is_default(protocol_Option* option) 
{
    switch(option->which_value) 
    {
        case protocol_Option_int_type_tag:
        {
            return option->value.int_type.current_value  == option->value.int_type.default_value;
        }

        case protocol_Option_float_type_tag:
        {
            return option->value.float_type.current_value  == option->value.float_type.default_value;
        }

        case protocol_Option_bool_type_tag:
        {
            return option->value.bool_type.current_value  == option->value.bool_type.default_value;
        }

        case protocol_Option_oneof_type_tag:
        {
            return option->value.oneof_type.current_index  == option->value.oneof_type.default_index;
        }

        case protocol_Option_blob_type_tag:
        {
            if(option->value.blob_type.current_value->size != option->value.blob_type.default_value->size)
            {
                return false;
            }

            return memcmp(
                option->value.blob_type.current_value->bytes,
                option->value.blob_type.default_value->bytes,
                option->value.blob_type.current_value->size) == 0;
        }

        case protocol_Option_string_type_tag:
        {
            if(option->value.string_type.current_value == option->value.string_type.default_value)
            {
                return true;
            }
            return strcmp(option->value.string_type.current_value, option->value.string_type.default_value) == 0;
        }

        case protocol_Option_password_type_tag:
        {
            return false;
        }

        default:
        {
            return false;
        }

    }
}

/******************************************************************************
* Function Name: tsp_file_read
********************************************************************************
* Summary:
*    Reads data from a file stream.
*
* Parameters:
*   stream - The input stream.
*   buf - The buffer to read data into.
*   count - The number of bytes to read.
*
* Return:   true if the data was read successfully, false otherwise.
*   
*
*******************************************************************************/
static bool tsp_file_read(pb_istream_t* stream, pb_byte_t* buf, size_t count) 
{
    if (count == 0)
    {
        return true;
    }

    FILE* source_file = (FILE*)(stream->state);
    size_t read = fread(buf, 1, count, source_file);
    if (read != count) 
    {

        /* Are we at the end of the stream? */ 
        if (read == 0 && feof(source_file) != 0) 
        {
            stream->errmsg = NULL;
            return false;
        }

        stream->errmsg = "Failed to read all data";
        printf("TSP File read error.\n");
        return false;
    }

    return true;
}

pb_istream_t* tsp_file_open(const char* path) 
{
    pb_istream_t* stream = pmem_malloc(sizeof(pb_istream_t));
    if (stream == NULL)
    {
        return NULL;
    }

    FILE* source_file = fopen(path, "rb");
    if (source_file == NULL) 
    {
        printf("Failed to open file. %s\n", strerror(errno));
        pmem_free(stream);
        return NULL;
    }

    stream->callback = tsp_file_read;
    stream->state = source_file;
    stream->bytes_left = SIZE_MAX;
    stream->errmsg = NULL;

    return stream;
}

/******************************************************************************
* Function Name: tsp_file_close
********************************************************************************
* Summary:
*    Closes a file stream.
*
* Parameters:
*   stream - The input stream.
*
* Return:   None
*   
*
*******************************************************************************/
void tsp_file_close(pb_istream_t* stream) 
{
    if (stream == NULL)
    {
        return;
    }
    
    FILE* source_file = (FILE*)(stream->state);
    if (source_file == NULL)
    {
        return;
    }

    fclose(source_file);

    pmem_free(stream);
}

/******************************************************************************
* Function Name: tsp_file_is_eof
********************************************************************************
* Summary:
*    Checks if the end of a file stream has been reached.
*
* Parameters:
*   stream - The input stream.
*
* Return:   true if the end of the file stream has been reached, false otherwise.
*   
*
*******************************************************************************/
bool tsp_file_is_eof(pb_istream_t* stream) 
{
    FILE* source_file = (FILE*)(stream->state);
    if (source_file == NULL)
    {
        return true;
    }

    return feof(source_file) != 0;
}

/******************************************************************************
* Function Name: tsp_file_rewind
********************************************************************************
* Summary:
*    Rewinds a file stream to the beginning.
*
* Parameters:
*   stream - The input stream.
*
* Return:   true if the file stream was successfully rewound, false otherwise.
*   
*
*******************************************************************************/
bool tsp_file_rewind(pb_istream_t* stream) 
{
    FILE* source_file = (FILE*)(stream->state);
    if (source_file == NULL)
    {
        return false;
    }
  
    if (fseek(source_file, 0, SEEK_SET) != 0)
    {
        return false;
    }
     
    stream->errmsg = NULL;
    return true;
}

/******************************************************************************
* Function Name: tsp_file_tell
********************************************************************************
* Summary:
*    Returns the current position of a file stream.
*
* Parameters:
*   stream - The input stream.
*
* Return:   The current position of the file stream, or -1 on error.
*   
*
*******************************************************************************/
long tsp_file_tell(pb_istream_t* stream) 
{
    if (stream == NULL)
    {
        return -1;
    }
       
    FILE* source_file = (FILE*)(stream->state);
    if (source_file == NULL)
    {
        return -1;
    }
    
    return ftell(source_file);
}

/******************************************************************************
* Function Name: tsp_count_frames
********************************************************************************
* Summary:
*    Counts the number of frames for a specific stream and device in a file stream.
*
* Parameters:
*   stream - The input stream.
*   target_stream - The target stream to count frames for.
*   target_device - The target device to count frames for.
*
* Return:   The total number of frames for the specified stream and device.
*
*******************************************************************************/
uint64_t tsp_count_frames(pb_istream_t* stream, int target_stream, int target_device) 
{
    if (stream == NULL)
    {
        return 0;
    }
   
    long start_pos = tsp_file_tell(stream);
    if (start_pos < 0)
    {
        return 0;
    }
 
    uint64_t total_frames = 0;
    
    while (!tsp_file_is_eof(stream)) 
    {
        protocol_Response pkg = protocol_Response_init_zero;
        
        /* Don't decode payload data - just get the metadata */
        if (!pb_decode_ex(stream, protocol_Response_fields, &pkg, PB_DECODE_DELIMITED | PB_DECODE_NOINIT)) 
        {
            if (tsp_file_is_eof(stream))
            {
                break;
            }
            /* Skip corrupted packets */
            continue;
        }
        
        if (pkg.which_response_type == protocol_Response_data_tag) 
        {
            protocol_DataChunk* data_chunk = &(pkg.response_type.data);
            if (data_chunk->stream == target_stream && data_chunk->device == target_device) 
            {
                /* Use the frame_count field directly - it tells us exactly how many frames */
                total_frames += data_chunk->frame_count;
            }
        }
        
        pb_release(protocol_Response_fields, &pkg);
    }
    
    /* Restore file position */
    FILE* source_file = (FILE*)(stream->state);
    if (source_file) 
    {
        if (fseek(source_file, start_pos, SEEK_SET) != 0) 
        {
            return 0;
        }
    }
    
    return total_frames;
}

/******************************************************************************
* Function Name: tsp_board_open
********************************************************************************
* Summary:
*    Opens a board from a capabilities package in a file stream.
*
* Parameters:
*   stream - The input stream.
*
* Return:   A pointer to the board, or NULL on error.
*
*******************************************************************************/
protocol_Board* tsp_board_open(pb_istream_t* stream) 
{
    static protocol_Response pkg;

    memset(&pkg, 0, sizeof(protocol_Response));

    if (!pb_decode_ex(stream, protocol_Response_fields, &pkg, PB_DECODE_DELIMITED)) 
    {
        printf("Failed to decode package. %s\n", stream->errmsg);
        return NULL;
    }

    if (pkg.which_response_type != protocol_Response_capabilities_tag) 
    {
        printf("Corrupt file. File must start with a capabilities package.\n");
        return NULL;
    }

    return pkg.response_type.capabilities.board;
}

/******************************************************************************
* Function Name: tsp_board_close
********************************************************************************
* Summary:
*    Closes a board by releasing the capabilities package.
*
* Parameters:
*   board - The board to close.
*
* Return:   None.
*
*******************************************************************************/
void tsp_board_close(protocol_Board* board) 
{
    protocol_Response pkg;
    pkg.which_response_type = protocol_Response_capabilities_tag;
    pkg.response_type.capabilities.board = board;

    pb_release(protocol_Response_fields, &pkg);
}

#endif /* IM_ENABLE_SHELL  */

/* [] END OF FILE */