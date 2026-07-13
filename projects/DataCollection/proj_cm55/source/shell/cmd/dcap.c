/******************************************************************************
* File Name: dcap.c
*
* Description: Implementation of the 'dcap' command for the shell.
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
#include <ctype.h>
#include <stdio.h> 
#include <stdlib.h>
#include <errno.h>
#include <sys/stat.h>

#include "../process.h"
#include "../lib/tsp_utils.h"
#include "../lib/getopt.h"
#include "../../protocol/protocol.h"
#include "../../protocol/pb_encode.h"
#include "../../protocol/pb_decode.h"
#include "../../heap.h"
#include "../../services.h"

/*******************************************************************************
* Macros
*******************************************************************************/
#define BUFFER_SIZE (1024*8)

/*******************************************************************************
* Types
*******************************************************************************/
typedef struct {
    protocol_t* protocol;
    FILE* file;
    uint8_t* buffer;
    size_t buffer_use;
} capture_state_t;

/*******************************************************************************
* Function Definitions
*******************************************************************************/
/******************************************************************************
* Function Name: write_file
********************************************************************************
* Summary:
*   Writes data to a file.
*
* Parameters:
*   file: Pointer to the file to write to.
*   buf: Pointer to the buffer containing data to write.
*   count: Number of bytes to write.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int write_file(FILE* file, void* buf, size_t count) 
{
    if (count > 0) 
    {
        printf("Writes %d bytes\n", count);
        size_t written = fwrite(buf, 1, count, file);
        if (written < count) 
        {
            printf("Failed to write file. %s\n", strerror(errno));            
            fclose(file);
            return -1;
        }
    }
    return 0;
}

/******************************************************************************
* Function Name: device_write
********************************************************************************
* Summary:
*   Writes data to the device and buffers it for writing to the file.
*
* Parameters:
*   stream: Pointer to the output stream.
*   buf: Pointer to the buffer containing data to write.
*   count: Number of bytes to write.
*
* Return:
*   true if operation is successful, otherwise false.
*
*******************************************************************************/
static bool device_write(pb_ostream_t* stream, const pb_byte_t* buf, size_t count) 
{
    capture_state_t* state = (capture_state_t*)stream->state;

    /* Does the data fit in the buffer, if not flush it. */
    if (state->buffer_use + count > BUFFER_SIZE) 
    {
        if (write_file(state->file, state->buffer, state->buffer_use) != 0) 
        {
            stream->errmsg = "File write error";
            return false;
        }
        state->buffer_use = 0;
    }

    if (count > BUFFER_SIZE) 
    {
        printf("Buffer too small. Direct write.\n");
        if (write_file(state->file, (void *)buf, count) != 0) 
        {
            stream->errmsg = "File write error";
            return false;
        }

    } 
    else 
    {
        memcpy(state->buffer + state->buffer_use, (void*)buf, count);
        state->buffer_use += count;
    }
    
    return true;
}

/******************************************************************************
* Function Name: write_header
********************************************************************************
* Summary:
*   Writes the header information to the output stream.
*
* Parameters:
*   protocol: Pointer to the protocol structure.
*   ostream: Pointer to the output stream.
*   device: Index of the device.
*
* Return:
*   true if operation is successful, otherwise false.
*
*******************************************************************************/
static bool write_header(protocol_t* protocol, pb_ostream_t* ostream, int device) 
{
    protocol_Board* board = &protocol->board;

    protocol_Response response;
    response.which_response_type = protocol_Response_capabilities_tag;
    response.response_type.capabilities.tag = 0;

    protocol_Board response_board = *board;

    response_board.devices_count = 1;
    response_board.devices = &board->devices[device];
    response.response_type.capabilities.board = &response_board;

    return pb_encode_ex(ostream, protocol_Response_fields, &response, PB_ENCODE_DELIMITED);
}

/******************************************************************************
* Function Name: get_file_name
********************************************************************************
* Summary:
*   Generates a unique file name for the given device in the /data folder.
*
* Parameters:
*   device: Pointer to the device structure.
*
* Return:
*   Pointer to the generated file name, or NULL if a unique name could not be 
*   generated.
*
*******************************************************************************/
static char* get_file_name(protocol_Device* device) 
{
    static char path[128];
    char dname[64];
    struct stat statbuf;

    strncpy(dname, device->name, sizeof(dname) - 1);
    dname[sizeof(dname) - 1] = '\0';

    for (char* p = dname; *p; p++) 
    {
        *p = (char)tolower((char)*p);
        if (*p == ' ')
        {
            *p = '_';
        }

    }

    for (int i = 0; i < 255; i++) 
    {
        snprintf(path, sizeof(path), "/data/%s_%03d.tsp", dname, i);

        if (stat(path, &statbuf) != 0) 
        {
            return path;
        }
    }

    return NULL;
}

/******************************************************************************
* Function Name: main_dcap
********************************************************************************
* Summary:
*   Main function for the dcap command.
*
* Parameters:
*   argc: Argument count.
*   argv: Argument vector.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
int main_dcap(int argc, char** argv) 
{
    int opt;
    const char* usage_text = "Usage: %s <device> [file]\n"
                                 "Capture data from given device.\n"
                                 "If file is not supplied, a new file name will be choosen in the /data folder\n"
                                 "  -t sec    Number of seconds to collect data\n"
                                 "  -h        Display this help and exit\n";
    getopt_t go;
    getopt_init(&go);
    int seconds = INT_MAX;

    while ((opt = getopte(&go, argc, argv, "ht:")) != -1) 
    {
        switch (opt) 
        {
            case 'h':
                printf(usage_text, argv[0]);
                return 0;
            case 't':
                seconds = atoi(go.optarg);
                break;
            default:
                printf(usage_text, argv[0]);
                return -1;
        }
    }

    if (go.optind >= argc) 
    {
        printf("Error: No device specified\n");
        printf(usage_text, argv[0]);
        return -1;
    }

    char* target_file = NULL;
    if (go.optind < argc - 1)
    {
        target_file = argv[go.optind + 1];
    }

    if (go.optind < argc - 2) 
    {
        printf("Error: Too many arguments\n");
        printf(usage_text, argv[0]);
        return -1;
    }

    char* device_str = argv[go.optind];

    /* Don't get killed by the shell when Ctrl-C is pressed. */
    process_set_attrib(NULL, PROCESS_NOEXIT);
    process_t* process = process_get(NULL);

    /* Fetch source device */
    protocol_t* protocol = SYS_SERVICE_PROTOCOL();
    protocol_Device* device = tsp_find(protocol, device_str);
    if (!device) 
    {
        printf("Error: No device named or index '%s' found.\n", device_str);
        return -1;
    }

    if (target_file == NULL) 
    {
        target_file = get_file_name(device);
    }

    /* Print some info */
    printf("Device             %s\n", device->name);
    printf("Type               %s\n", tsp_get_type_str(device->type));
    printf("Status             %s\n", tsp_get_status_str(device->status));

    if (device->status_message != NULL && strlen(device->status_message) > 0)
    {
        printf("Message            %s\n", device->status_message);
    }

    if (device->description != NULL && strlen(device->description) > 0)
    {
        printf("Description        %s\n", device->description);
    }
   
    /* Check device is free to use */
    device_manager_t* manager = &(protocol->device_managers[device->device_id]);
    if (manager->busy != NULL || device->status != protocol_DeviceStatus_DEVICE_STATUS_READY) 
    {
        printf("Error: Device %s busy.\n", device->name);
        return -1;
    }
   
    char tmp_buf[255];
    for (int i = 0; i < device->streams_count; i++) 
    {
        protocol_StreamConfig* stream = &(device->streams[i]);
        stream->current_frame = 0;
        stream->frames_dropped = 0;

        printf("[Stream %d]\n", i);
        printf("  Name             %s\n", stream->name);
        printf("  Frequency        %f\n", stream->frequency);
        printf("  Type             %s\n", tsp_stream_get_type_str(stream->datatype));
        
        if (stream->unit && strlen(stream->unit) > 0)
        {
            printf("  Unit             %s\n", stream->unit);
        }

        printf("  Shape            %s\n", tsp_stream_get_shape(stream, tmp_buf, sizeof(tmp_buf)));
        printf("  Dimension Names  %s\n", tsp_stream_get_shape_dim_names(stream, tmp_buf, sizeof(tmp_buf)));
        printf("  Shape Labels     %s\n", tsp_stream_get_shape_labels(stream, tmp_buf, sizeof(tmp_buf)));
    }
 
    /* Create a application state */
    capture_state_t* state = (capture_state_t*)pmem_malloc(sizeof(capture_state_t));
    if (state == NULL) 
    {
        printf("Malloc failed\n");
        return -1;
    }
    state->protocol = protocol;
    state->buffer_use = 0;
    state->buffer = pmem_malloc(BUFFER_SIZE);

    /* Open target file */
    state->file = fopen(target_file, "w");
    if (!state->file) 
    {
        printf("Failed to open file for writing. %s\n", strerror(errno));
        pmem_free(state->buffer);
        free(state);
        return -1;
    }
    printf("Writes output to %s\n", target_file);

    /* Create output stream */
    pb_ostream_t ostream = (pb_ostream_t){ &device_write, (void*)state, SIZE_MAX, 0, NULL };

    /* Write header */
    write_header(protocol, &ostream, device->device_id);

    /* Start streaming */
    manager->busy = &ostream;       
    manager->start(protocol, device->device_id, &ostream, manager->arg);
   
    protocol_StreamConfig* first_stream = &(device->streams[0]);
    long max_frame_count = first_stream->frequency * seconds;
    while(device->status == protocol_DeviceStatus_DEVICE_STATUS_ACTIVE || device->status == protocol_DeviceStatus_DEVICE_STATUS_ACTIVE_WAIT)
    {      
        /* Do device processing */
        protocol_call_device_poll(protocol, &ostream);

        /* Check for error */
        if (ostream.errmsg != NULL) 
        {
             printf("Stream error. %s\n", ostream.errmsg);
             fclose(state->file);
             pmem_free(state->buffer);
             free(state);
             return -1;
        }

        /* Abort with Ctrl-C or timeout */
        if (first_stream->current_frame > max_frame_count || process->terminal->last_input == 3) 
        {           
            manager->stop(protocol, device->device_id, &ostream, manager->arg);
        }
    }
  
    /* Flush and close file */
    if (write_file(state->file, state->buffer, state->buffer_use) != 0) 
    {
        fclose(state->file);
        pmem_free(state->buffer);
        free(state);
        return -1;
    }
    state->buffer_use = 0;
    fclose(state->file);

    for (int i = 0; i < device->streams_count; i++) 
    {
        protocol_StreamConfig* stream = &(device->streams[i]);
        printf("[Stream %d]\n", i);
        printf("  Frame Collected  %ld\n", stream->current_frame);
        printf("  Frames Dropped   %ld\n", stream->frames_dropped);
    }

    pmem_free(state->buffer);
    return 0;
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */
