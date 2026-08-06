/******************************************************************************
* File Name: tsp2wav.c
*
* Description: Implementation of the 'tsp2wav' command for the shell.
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
#include <limits.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include "../lib/getopt.h"
#include "../lib/tsp_utils.h"
#include "../../protocol/protocol.h"
#include "../../protocol/pb_encode.h"
#include "../../protocol/pb_decode.h"
#include "../../heap.h"
#include "../../wifi_configs/im_config.h"
#include "../fs/fs.h"

/*******************************************************************************
* Macros
*******************************************************************************/
#define WAVE_FORMAT_PCM 1
#define WAVE_FORMAT_IEEE_FLOAT 3

/*******************************************************************************
* Types
*******************************************************************************/
typedef struct __attribute__((packed))
{
    uint32_t id;
    uint32_t size;
    uint32_t format;
} wave_riff_t;

typedef struct __attribute__((packed))
{
    uint32_t id;
    uint32_t size;
    uint16_t format;
    uint16_t channels;
    uint32_t frequency;
    uint32_t byterate;
    uint16_t align;
    uint16_t bps;
} wave_fmt_t;

typedef struct __attribute__((packed))
{
    uint32_t id;
    uint32_t size;
} wave_data_t;

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: print_device
********************************************************************************
* Summary:
*   Prints the details of a protocol device, including its options and streams.
*
* Parameters:
*   device: The protocol device to print.
*
* Return:
*   None.
*
*******************************************************************************/
static void print_device(protocol_Device* device) 
{
    char tmp_buf[255];

    printf("Device               %s (%s)\n", device->name, tsp_get_type_str(device->type));
    if (device->description != NULL && strlen(device->description) > 0)
    {
        printf("Description          %s\n", device->description);
    }

    for (int i = 0; i < device->options_count; i++) 
    {
        protocol_Option* option = &device->options[i];
        const char* default_str = tsp_option_is_default(option) ? "(Default)" : "";

        printf("%-20s %s %s\n", option->name, tsp_option_get_value_str(option, true, tmp_buf, sizeof(tmp_buf)), default_str);
    }

    for (int i = 0; i < device->streams_count; i++) 
    {
        protocol_StreamConfig* stream = &device->streams[i];
        printf("[Stream %s]\n", stream->name);
        printf("  Frames Dropped     %ld\n", stream->frames_dropped);
        
        if (stream->frequency != 0)
        {
            printf("  Frequency          %f\n", stream->frequency);
        }

        printf("  Type               %s\n", tsp_stream_get_type_str(stream->datatype)); 

        if (stream->unit && strlen(stream->unit) > 0)
        {
            printf("  Unit               %s\n", stream->unit);
        }

        printf("  Shape              %s\n", tsp_stream_get_shape(stream, tmp_buf, sizeof(tmp_buf)));
        printf("  Dimension Names    %s\n", tsp_stream_get_shape_dim_names(stream, tmp_buf, sizeof(tmp_buf)));
        printf("  Shape Labels       %s\n", tsp_stream_get_shape_labels(stream, tmp_buf, sizeof(tmp_buf)));
    }
}

/******************************************************************************
* Function Name: wav_file_create
********************************************************************************
* Summary:
*   Creates a WAV file for the given stream configuration.
*
* Parameters:
*   stream: The stream configuration.
*   path: The path to the WAV file to create.
*   sample_size: Pointer to store the sample size in bytes.
*
* Return:
*   FILE*: The file pointer to the created WAV file, or NULL on error.
*
*******************************************************************************/
static FILE* wav_file_create(protocol_StreamConfig* stream, const char* path, uint32_t* sample_size) 
{
    if (stream->shape_count != 1) 
    {
        printf("Error:: Shape error\n");       
        return NULL;
    }

    wave_riff_t riff = {
        .id = 0x46464952, /* "RIFF" */
        .size = 0,        /* to be updated later */
        .format = 0x45564157 /* "WAVE" */
    };
     
    int channels = stream->shape[0].size;
    int sample_rate = stream->frequency;
    int bits_per_sample = 0;
    uint16_t format = 0;
 
    switch (stream->datatype)
    {
        case protocol_DataType_DATA_TYPE_U8:
        {
            bits_per_sample = 8;
            format = WAVE_FORMAT_PCM;
            break;
        }  

        case protocol_DataType_DATA_TYPE_S16:
        {
            bits_per_sample = 16;
            format = WAVE_FORMAT_PCM;
            break;
        }

        case protocol_DataType_DATA_TYPE_S32:
        {
            bits_per_sample = 32;
            format = WAVE_FORMAT_PCM;
            break;
        } 

        case protocol_DataType_DATA_TYPE_F32:
        {
            bits_per_sample = 32;
            format = WAVE_FORMAT_IEEE_FLOAT;
            break;
        }

        default:
        {
            printf("ERROR: Data type error. Stream data type must be one of U8, S16, S32 or F32.\n");
            return NULL;
        }

    }

    wave_fmt_t fmt = {
       .id = 0x20746D66, /* "fmt " */
       .size = 16,
       .format = format,
       .channels = channels,
       .frequency = sample_rate,
       .byterate = sample_rate * channels * bits_per_sample / 8,
       .align = channels * bits_per_sample / 8,
       .bps = bits_per_sample
    };

    wave_data_t data = {
        .id = 0x61746164, /* "data" */
        .size = 0        /* to be updated later */
    };


    /* Sample size in bytes */
    *sample_size = channels * bits_per_sample / 8;   

    FILE* target_file = fopen(path, "wb");
    if (target_file == NULL) 
    {
        printf("Error: Failed to create/open file %s for writing. %s\n", path, strerror(errno));
        return NULL;
    }

    if (fwrite(&riff, sizeof(wave_riff_t), 1, target_file) != 1) 
    {
        printf("Error: RIFF write error\n");
        fclose(target_file);
        return NULL;
    }

    if (fwrite(&fmt, sizeof(wave_fmt_t), 1, target_file) != 1) 
    {
        printf("Error: FMT write error\n");
        fclose(target_file);
        return NULL;
    }

    if (fwrite(&data, sizeof(wave_data_t), 1, target_file) != 1) 
    {
        printf("Error: Wave data header write error\n");
        fclose(target_file);
        return NULL;
    }

    return target_file;
}

/******************************************************************************
* Function Name: wav_file_close
********************************************************************************
* Summary:
*   Closes a WAV file and updates its header with the correct sizes.
*
* Parameters:
*   file: The file pointer to the WAV file to close.
*
* Return:
*   None
*
*******************************************************************************/
static void wav_file_close(FILE* file) 
{
    uint32_t riff_size;
    uint32_t data_size;
    long size;

    if (file == NULL)
    {
        return;
    }

    if (ferror(file) == 0) 
    {
        size = ftell(file);
        if (size > 8) 
        {
            riff_size = size - 8;
            if (fseek(file, 4, SEEK_SET) == 0)
            {
                fwrite(&riff_size, sizeof(uint32_t), 1, file);
            }

            data_size = riff_size - 36;
            if (fseek(file, 40, SEEK_SET) == 0)
            {
                fwrite(&data_size, sizeof(uint32_t), 1, file);
            }
        }
    }
    fclose(file);
}

/******************************************************************************
* Function Name: change_extension
********************************************************************************
* Summary:
*   Changes the file extension of a given file path.
*
* Parameters:
*   file_path: The original file path.
*   new_extension: The new file extension.
*   new_path: The buffer to store the new file path.
*   new_path_size: The size of the new_path buffer.
*
* Return:
*   None
*
*******************************************************************************/
static void change_extension(const char* file_path, const char* new_extension, char* new_path, size_t new_path_size) 
{
    const char* last_dot;
    size_t base_length;
    size_t required_size;

    /* Validate input parameters */
    if (!file_path || !new_extension || !new_path || new_path_size == 0) 
    {
        fprintf(stderr, "Invalid input parameters.\n");
        return;
    }

    /* Find the last occurrence of '.' in the file path */
    last_dot = strrchr(file_path, '.');
    base_length = last_dot ? (size_t)(last_dot - file_path) : strlen(file_path);

    /* Calculate the required buffer size */
    required_size = base_length + 1 + strlen(new_extension) + 1;

    /* Check if the new_path buffer is large enough */
    if (required_size > new_path_size) 
    {
        fprintf(stderr, "Error: new_path buffer is too small. Required size: %zu\n", required_size);
        return;
    }

    /* Copy the base file path and new extension to the new path buffer */
    memcpy(new_path, file_path, base_length);
    new_path[base_length] = '.';
    strcpy(new_path + base_length + 1, new_extension);
}

/******************************************************************************
* Function Name: change_extension
********************************************************************************
* Summary:
*   Changes the file extension of a given file path.
*
* Parameters:
*   file_path: The original file path.
*   new_extension: The new file extension.
*   new_path: The buffer to store the new file path.
*   new_path_size: The size of the new_path buffer.
*
* Return:
*   None
*
*******************************************************************************/
static bool process_decode(pb_istream_t* stream, const pb_field_t* field, void** arg) 
{
    FILE* wav_file = (FILE*)*arg;
    uint8_t buffer[1024];
    size_t bytes_left = stream->bytes_left;
    while (bytes_left > 0) 
    {
        size_t copy = bytes_left;
        if (copy > sizeof(buffer))
        {
            copy = sizeof(buffer);
        }

        if (!pb_read(stream, buffer, copy))
        {
            return false;
        }

        if (fwrite(buffer, copy, 1, wav_file) != 1) 
        {
            printf("Error: Failed to write wav file.\n");
            return false;
        }

        bytes_left -= copy;
    }

    return true;
}

/******************************************************************************
* Function Name: main_tsp2wav
********************************************************************************
* Summary:
*   Converts a TSP file to a WAV file.
*
* Parameters:
*   argc: The number of command-line arguments.
*   argv: The array of command-line arguments.
*
* Return:
*   Exit status code.
*
*******************************************************************************/
int main_tsp2wav(int argc, char** argv)
{
    const char* usage_text = "Usage: %s <file>\n"
        "Display file contents.\n"
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

    if (go.optind >= argc) 
    {
        printf("Error: No file specified\n");
        printf(usage_text, argv[0]);
        return -1;
    }

    if (go.optind < argc - 1) 
    {
        printf("Error: Too many arguments\n");
        printf(usage_text, argv[0]);
        return -1;
    }

    char tsp_path[IM_PATH_MAX];
    if (realpath(argv[go.optind], tsp_path) == NULL) 
    {
        printf("Error: Failed to resolve path. %s\n", strerror(errno));
        return -1;
    }
    pb_istream_t* tsp_file = tsp_file_open(tsp_path);
    if (tsp_file == NULL) {
        return -1;
    }

    char wav_path[IM_PATH_MAX];
    change_extension(tsp_path, "wav", wav_path, IM_PATH_MAX);

    protocol_Board* board = tsp_board_open(tsp_file);

    if (board == NULL) 
    {
        tsp_file_close(tsp_file);
        return -1;
    }

    protocol_Device* device = &(board->devices[0]);

    print_device(device);

    if (device->streams_count != 1) 
    {
        printf("Error: Expected only one stream. Found %d in file.\n", device->streams_count);
        tsp_file_close(tsp_file);
        tsp_board_close(board);
        return -1;
    }

    protocol_StreamConfig* stream_config = &(device->streams[0]);

    uint32_t sample_size;
    FILE* wav_file = wav_file_create(stream_config, wav_path, &sample_size);
    if (wav_file == NULL) 
    {
        tsp_file_close(tsp_file);
        tsp_board_close(board);
        return -1;
    }
   
    printf("Saving file %s Please wait...\n", wav_path);
    while (true) 
    {
        protocol_Response pkg = protocol_Response_init_zero;

        pkg.which_response_type = protocol_Response_data_tag;
        pkg.response_type.data.payload.funcs.decode = process_decode;
        pkg.response_type.data.payload.arg = wav_file;

        if (!pb_decode_ex(tsp_file, protocol_Response_fields, &pkg, PB_DECODE_DELIMITED | PB_DECODE_NOINIT)) 
        {
            /* End of stream? */
            if (tsp_file_is_eof(tsp_file))
            {
                break;
            }

            printf("Error: Corrupt TSP file. Failed to decode package. %s\n", tsp_file->errmsg);
            tsp_file_close(tsp_file);
            tsp_board_close(board);
            wav_file_close(wav_file);
            return -1;
        }

        if (pkg.which_response_type != protocol_Response_data_tag) 
        {
            printf("Error: Corrupt TSP file. Expected data chunk.\n");
            tsp_file_close(tsp_file);
            tsp_board_close(board);
            wav_file_close(wav_file);
            return -1;
        }

        protocol_DataChunk* data = &(pkg.response_type.data);

        if (data->stream != 0 || data->device != device->device_id) 
        {
            printf("Error: Corrupt TSP file. Invalid data chunk.\n");
            tsp_file_close(tsp_file);
            tsp_board_close(board);
            wav_file_close(wav_file);
            return -1;
        }
        pb_release(protocol_Response_fields, &pkg);
    }

    tsp_file_close(tsp_file);
    tsp_board_close(board);
    wav_file_close(wav_file);
    printf("Done!\n");
    return 0;
}

#endif /* IM_ENABLE_SHELL */