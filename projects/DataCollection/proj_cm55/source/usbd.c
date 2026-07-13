/******************************************************************************
* File Name:   usbd.c
*
* Description: This file contains functions for streaming data over a serial
*              USB CDC interface.
*
* Related Document: See README.md
*
*******************************************************************************
* (c) 2025-2026, Infineon Technologies AG, or an affiliate of Infineon
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

#include <stddef.h>
#include <stdio.h>
#include <USB.h>
#include <USB_CDC.h>
#include <FreeRTOS.h>
#include <task.h>

#include "protocol/protocol.h"
#include "protocol/pb_encode.h"
#include "protocol/pb_decode.h"
#include "common.h"
#include "usbd.h"
#include "heap.h"
#include "cybsp.h"
/*******************************************************************************
* Local Function Prototypes
*******************************************************************************/
static USB_CDC_HANDLE _usbd_add_cdc(void);
static bool _usbd_read(pb_istream_t* stream, pb_byte_t* buf, size_t count);
static bool _usbd_write(pb_ostream_t* stream, const pb_byte_t* buf, size_t count);

/*******************************************************************************
* Functions
*******************************************************************************/

/*******************************************************************************
* Function Name: _usbd_add_cdc
********************************************************************************
* Summary:
*  Initializes USB CDC.
*
*******************************************************************************/
static USB_CDC_HANDLE _usbd_add_cdc(void)
{
    static U8             OutBuffer[USB_HS_BULK_MAX_PACKET_SIZE];
    USB_CDC_INIT_DATA     InitData;
    USB_ADD_EP_INFO       EPBulkIn;
    USB_ADD_EP_INFO       EPBulkOut;
    USB_ADD_EP_INFO       EPIntIn;

    memset(&InitData, 0, sizeof(InitData));
    EPBulkIn.Flags          = 0;                             /* Flags not used */
    EPBulkIn.InDir          = USB_DIR_IN;                    /* IN direction (Device to Host) */
    EPBulkIn.Interval       = 0;                             /* Interval not used for Bulk endpoints */
    EPBulkIn.MaxPacketSize  = USB_HS_BULK_MAX_PACKET_SIZE;   /* Maximum packet size (512B for Bulk in full-speed) */
    EPBulkIn.TransferType   = USB_TRANSFER_TYPE_BULK;        /* Endpoint type - Bulk */
    InitData.EPIn  = USBD_AddEPEx(&EPBulkIn, NULL, 0);

    EPBulkOut.Flags         = 0;                             /* Flags not used */
    EPBulkOut.InDir         = USB_DIR_OUT;                   /* OUT direction (Host to Device) */
    EPBulkOut.Interval      = 0;                             /* Interval not used for Bulk endpoints */
    EPBulkOut.MaxPacketSize = USB_HS_BULK_MAX_PACKET_SIZE;   /* Maximum packet size (512B for Bulk in full-speed) */
    EPBulkOut.TransferType  = USB_TRANSFER_TYPE_BULK;        /* Endpoint type - Bulk */
    InitData.EPOut = USBD_AddEPEx(&EPBulkOut, OutBuffer, sizeof(OutBuffer));

    EPIntIn.Flags           = 0;                             /* Flags not used */
    EPIntIn.InDir           = USB_DIR_IN;                    /* IN direction (Device to Host) */
    EPIntIn.Interval        = 64;                            /* Interval of 8 ms (64 * 125us) */
    EPIntIn.MaxPacketSize   = USB_HS_BULK_MAX_PACKET_SIZE ;   /* Maximum packet size (64 for Interrupt) */
    EPIntIn.TransferType    = USB_TRANSFER_TYPE_INT;         /* Endpoint type - Interrupt */
    InitData.EPInt = USBD_AddEPEx(&EPIntIn, NULL, 0);

    return USBD_CDC_Add(&InitData);
}


/*******************************************************************************
* Function Name: _usbd_read
********************************************************************************
* Summary:
*   Reads data from the serial interface into the provided buffer.
*
* Parameters:
*   stream: Pointer to the input stream.
*   buf: Pointer to the buffer where the read data will be stored.
*   count: Number of bytes to read.
*
* Return:
*   True if reading is successful, otherwise false.
*
*******************************************************************************/
static bool _usbd_read(pb_istream_t* stream, pb_byte_t* buf, size_t count)
{
    if (count == 0)
    {
        return true;
    }

    usbd_t *usb = (usbd_t*)stream->state;

    USBD_CDC_ReadOverlapped(usb->usb_cdcHandle, buf, count);

    /*Always do at least one poll if we have one byte.*/ 
    protocol_call_device_poll(usb->protocol, &usb->ostream);

     /* Do device processing while waiting for data.
     * If the requested byte wasn't in the incoming buffer we just
     * wait for it here and do the polling the same time.
     */

    while(USBD_CDC_GetNumBytesRemToRead(usb->usb_cdcHandle) > 0) 
    {
        if ((USBD_GetState() & USB_STAT_CONFIGURED) != USB_STAT_CONFIGURED)
            return false;

        if (protocol_call_device_poll(usb->protocol, &usb->ostream) == 0) 
        {
            // If no devices invoked - sleep for a while (100 ms)
            vTaskDelay(pdMS_TO_TICKS(100));
        }    
    }

    return true;

}

/*******************************************************************************
* Function Name: _usbd_write
********************************************************************************
* Summary:
*   Writes data from the provided buffer to the serial interface.
*
* Parameters:
*   stream: Pointer to the output stream.
*   buf: Pointer to the buffer containing the data to be written.
*   count: Number of bytes to write.
*
* Return:
*   True if writing is successful, otherwise false.
*
*******************************************************************************/
static bool _usbd_write(pb_ostream_t* stream, const pb_byte_t* buf, size_t count)
{
    usbd_t *usb = (usbd_t*)stream->state;

    set_led0(true);

    USBD_CDC_Write(usb->usb_cdcHandle, buf, count, 0);

    /* Block until write is complete */
    USBD_CDC_WaitForTX(usb->usb_cdcHandle, 0);

    set_led0(false);
    return true;
}

/*******************************************************************************
* Function Name: usbd_create
********************************************************************************
* Summary:
*   Creates and initializes a new streaming instance.
*
* Parameters:
*   protocol: Pointer to the protocol handle.
*
* Return:
*   Pointer to the newly created streaming instance.
*
*******************************************************************************/
void usbd_task(void* arg)
{
    protocol_t* protocol = (protocol_t*)arg;

    usbd_t* usb = sys_malloc(sizeof(usbd_t));

    if (usb == NULL)
    {
        log_message(LOG_LEVEL_INFO, "usb", "USB Task Failed - Unable to allocate memory.");
        return;
    }

    log_message(LOG_LEVEL_INFO, "usb", "USB Task started");

    usb->protocol = protocol;

    static char serial_str[38];
    pb_byte_t *serial = protocol->board.serial.uuid;
    sprintf(serial_str, "%02X%02X%02X%02X-%02X%02X-%02X%02X-%02X%02X-%02X%02X%02X%02X%02X%02X",
            serial[0], serial[1], serial[2], serial[3],
            serial[4], serial[5],
            serial[6], serial[7],
            serial[8], serial[9],
            serial[10], serial[11], serial[12], serial[13], serial[14], serial[15]);
    usb->usb_deviceInfo.sSerialNumber = (const char *)serial_str;
    usb->usb_deviceInfo.VendorId = 0x058B;
    usb->usb_deviceInfo.ProductId = 0x027D;
    usb->usb_deviceInfo.sVendorName = "Infineon Technologies";
    usb->usb_deviceInfo.sProductName = "DEEPCRAFT Streaming Device";

    /* Initializes the USB stack */
    USBD_Init();

    /* Endpoint Initialization for CDC class */
    usb->usb_cdcHandle = _usbd_add_cdc();

    /* Set device info used in enumeration */
    USBD_SetDeviceInfo(&usb->usb_deviceInfo);

    /* Start the USB stack */
    USBD_Start();

    usb->istream = (pb_istream_t){ &_usbd_read, (void*)usb, SIZE_MAX, 0 };
    usb->ostream = (pb_ostream_t){ &_usbd_write, (void*)usb, SIZE_MAX, 0, NULL };

    for(;;)
    {
       int count = 0;

       while ((USBD_GetState() & USB_STAT_CONFIGURED) != USB_STAT_CONFIGURED)
       {
           set_led0((count & 8) != 0);
           count++;
           
           vTaskDelay(pdMS_TO_TICKS(50));
       }

       set_led0(true);
       protocol_process_request(protocol, &usb->istream, &usb->ostream);
       set_led0(false);
    }

    sys_free(usb);   
}


/* [] END OF FILE */
