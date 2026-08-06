/******************************************************************************
* File Name:   dev_board.c
*
* Description:
*   This file implements the interface with the board it self.
*   It does not provide any streams but merely a configuration.
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

#ifdef IM_ENABLE_BOARD_CONFIG

#include <stdio.h>

#include "protocol/protocol.h"
#include "protocol/pb_encode.h"
#include "clock.h"
#include "common.h"

/*******************************************************************************
* Macros
*******************************************************************************/
#define OPTION_BOARD_NAME 1

/*******************************************************************************
* Types
*******************************************************************************/
extern char board_name[];

/*******************************************************************************
* Function Prototypes
*******************************************************************************/


static bool _configure_streams(protocol_t* protocol, int device, void* arg);
static void _start_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg);
static void _stop_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg);
static void _poll_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg);

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/*******************************************************************************
* Function Name: _configure_streams
********************************************************************************
* Summary:
*  The board it self shoudn't deliver any data. Receive the board configuration.
*
* Parameters:
*
*  Return:
*    True to keep the connection open, false to close.
*
*******************************************************************************/
static bool _configure_streams(protocol_t* protocol, int device, void* arg){

    // Retrieve the string too, to test this.
    char* name;
    protocol_get_option_string( protocol, device, OPTION_BOARD_NAME, &name);
    printf("Have name %s\n", name);
    

    if ( strlen(name) < 128 ){
        strcpy(board_name, name);

// For the boardname to survive a reset it have to be saved if possible.
#ifdef IM_ENABLE_SHELL
        FILE* board = fopen("system/boardname","wb");
        if (board){
            int len = strlen(board_name)+1;
            fwrite(board_name,1,len,board);
            fclose(board);
            printf("Wrote boardname to file.\n");
        }
#endif
    }
    
    return true;
}

/*******************************************************************************
* Function Name: _start_streams
********************************************************************************
* Summary:
*  There is no much use for this.
*
* Parameters:
*
*
*******************************************************************************/
static void _start_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg){
    UNUSED(protocol);
    UNUSED(device);
    UNUSED(ostream);
    UNUSED(arg);
}

/*******************************************************************************
* Function Name: _stop_streams
********************************************************************************
* Summary:
*  No stream to stop.
*
* Parameters:
*
*
*******************************************************************************/
static void _stop_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg){
    UNUSED(protocol);
    UNUSED(device);
    UNUSED(ostream);
    UNUSED(arg);
}

/*******************************************************************************
* Function Name: _poll_streams
********************************************************************************
* Summary:
*  Nothing to poll. But perhaps this is needed anyway?
*
* Parameters:
*
*
*******************************************************************************/

static void _poll_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg){
    UNUSED(protocol);
    UNUSED(device);
    UNUSED(ostream);
    UNUSED(arg);
}

/*******************************************************************************
* Function Name: dev_dps368_register
********************************************************************************
* Summary:
*  Registers this device. This is the only exported symbol from this object.
*  This should register the parameters that we might need to be sendt to the
*  board itself.
*
* Parameters:
*  protocol: Pointer to the protocol handle.
*  
* Returns:
*  True on success, else false.
*
*******************************************************************************/
bool dev_board_register(protocol_t* protocol )
{
    int status;

    // Again is this data copied to another storage type?
    // I need to check this.
    device_manager_t manager = {
        .arg = NULL,
        .configure_streams = _configure_streams,
        .start = _start_streams,
        .stop = _stop_streams,
        .poll = _poll_streams,
        .data_received = NULL /* has no input streams */
    };

    /* Add the board to the protocol and get its device identifier */
    int device = protocol_add_device(
        protocol,
        protocol_DeviceType_DEVICE_TYPE_BOARD,
        "BOARD",
        "Board configurations.",
        manager);

    if(device < 0)
    {
        return false;
    }

    // The current name should ba available 
    // in the protocol..
    status = protocol_add_option_string(
                protocol,
                device,
                OPTION_BOARD_NAME,
                "Name of this board",
                "Change some name..",
                board_name);

    if(status != PROTOCOL_STATUS_SUCCESS)
    {
        return false;
    }

    return true;
}

#endif /* IM_ENABLE_DPS368 */

/* [] END OF FILE */
