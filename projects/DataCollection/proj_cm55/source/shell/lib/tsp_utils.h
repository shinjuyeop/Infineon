/******************************************************************************
* File Name:   tsp_utils.h
*
* Description: This file provides utility functions for 
* TSP (Test Signal Processing) operations.
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

#ifndef _DEVICE_UTILS_H_
#define _DEVICE_UTILS_H_

#ifdef IM_ENABLE_SHELL

#include "../../protocol/protocol.h"
#include "../../protocol/model.pb.h"

/*******************************************************************************
* Function Prototypes
*******************************************************************************/
const char* tsp_get_type_str(protocol_DeviceType type);
const char* tsp_get_status_str(protocol_DeviceStatus status);
const char* tsp_option_get_type_str(int type);
const char* tsp_stream_get_direction_str(protocol_StreamDirection direction);
const char* tsp_stream_get_type_str(protocol_DataType type);
char* tsp_option_get_value_str(protocol_Option* option, bool current, char* str, size_t size);
protocol_Device* tsp_find(protocol_t* protocol, char* token);
protocol_Option* tsp_option_find(protocol_Device *device, char* token);
char* tsp_stream_get_shape(protocol_StreamConfig* stream, char* str, ssize_t size);
char* tsp_stream_get_shape_dim_names(protocol_StreamConfig* stream, char* str, ssize_t size);
char* tsp_stream_get_shape_labels(protocol_StreamConfig* stream, char* str, ssize_t size);
bool tsp_option_is_default(protocol_Option* option);

pb_istream_t* tsp_file_open(const char* path);
void tsp_file_close(pb_istream_t* stream);
bool tsp_file_is_eof(pb_istream_t* stream);
bool tsp_file_rewind(pb_istream_t* stream);
long tsp_file_tell(pb_istream_t* stream);
uint64_t tsp_count_frames(pb_istream_t* stream, int target_stream, int target_device);

protocol_Board* tsp_board_open(pb_istream_t* stream);
void tsp_board_close(protocol_Board* board);

#endif /* IM_ENABLE_SHELL  */

#endif /* _DEVICE_UTILS_H_ */

/* [] END OF FILE */
