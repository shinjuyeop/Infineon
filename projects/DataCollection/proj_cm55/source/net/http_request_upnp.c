/******************************************************************************
* File Name:   http_request_upnp.c
*
* Description: This file provides implementation for HTTP application 
*              specific endpoints.
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

#if defined(IM_ENABLE_HTTP) && defined(IM_ENABLE_UPNP)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>

#include "../heap.h"
#include "../common.h"
#include "../protocol/protocol.h"
#include "../services.h"
#include "tcp.h"
#include "http.h"
#include "http_utils.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: send_upnp_response
********************************************************************************
* Summary:
*    Sends a UPnP response.
*
* Parameters:
*   session - Pointer to the TCP session.
*
* Return:
*   True if the response was sent successfully, false otherwise.
*
*******************************************************************************/
bool send_upnp_response(tcp_session_t* session) 
{
    static char tmp[1024];  /* Temporary chunk buffer */
    protocol_t* protocol = SYS_SERVICE_PROTOCOL();

    char* ptr;

    if (!http_start_chunked_response(session, STATUS_200, MIME_XML))
    {
        return false;
    }

    const char* upnp_response_hdr =
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        "<root xmlns=\"urn:schemas-upnp-org:device-1-0\"\n"
        "      xmlns:dc=\"http://purl.org/dc/elements/1.1/\"\n"
        "      xmlns:upnp=\"urn:schemas-upnp-org:upnp\"\n"
        "      xmlns:os=\"urn:schemas-upnp-org:os\"\n"
        "      xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n"
        "      xsi:schemaLocation=\"urn:schemas-upnp-org:device-1-0 "
        "http://schemas.upnp.org/upnp/1.0/urn_schemas-upnp-org_device-1-0.xsd\">\n"
        "  <specVersion>\n"
        "    <major>1</major>\n"
        "    <minor>0</minor>\n"
        "  </specVersion>\n";
       
    if (!http_send_chunk(session, upnp_response_hdr, strlen(upnp_response_hdr)))
    {
        return false;
    }

    ptr = tmp;

    ptr += sprintf(ptr, "  <device>\n");
    ptr += sprintf(ptr, "    <deviceType>urn:schemas-upnp-org:device:Basic:1</deviceType>\n");
    ptr += sprintf(ptr, "    <friendlyName>%s</friendlyName>\n", protocol->board.name);
    ptr += sprintf(ptr, "    <manufacturer>Infineon</manufacturer>\n");
    ptr += sprintf(ptr, "    <manufacturerURL>https://www.infineon.com</manufacturerURL>\n");
    ptr += sprintf(ptr, "    <presentationURL>%s</presentationURL>\n", http_get_location_uri());
    ptr += sprintf(ptr, "    <modelURL>%s/devices.json</modelURL>\n", http_get_location_uri());
    ptr += sprintf(ptr, "    <modelName>%s</modelName>\n", protocol->board.name);
    ptr += sprintf(ptr, "    <modelNumber>1.0</modelNumber>\n");
    ptr += sprintf(ptr, "    <serialNumber>%s</serialNumber>\n", http_get_serial_string(&protocol->board.serial));
    ptr += sprintf(ptr, "    <UDN>uuid:%s</UDN>\n", http_get_serial_string(&protocol->board.serial));
    ptr += sprintf(ptr, "  </device>\n");

    ptr += sprintf(ptr, "</root>\n");

    if (!http_send_chunk(session, tmp, ptr - tmp))
    {
        return false;
    }

    if (!http_end_chunked_response(session))
    {
        return false;
    }

    return true;
}

#endif /* IM_ENABLE_HTTP and IM_ENABLE_UPNP */

/* [] END OF FILE */
