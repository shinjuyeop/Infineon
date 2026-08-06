/******************************************************************************
* File Name:   upnp.c
*
* Description: Implements SSDP for UPnP device discovery and description.
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
#if defined(IM_ENABLE_NET) && defined(IM_ENABLE_UPNP)

#include <cybsp.h>
#include <string.h>
#include <inttypes.h>
#include <FreeRTOS.h>
#include <signal.h>
#include <task.h>
#include <cy_secure_sockets.h>
#include <stdio.h>

#include "upnp.h"
#include "../common.h"
#include "../wifi_configs/im_config.h"
#include "../services.h"
#include "http.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: upnp_send_notify
********************************************************************************
* Summary:
*    Sends a SSDP NOTIFY message for UPnP device discovery.
*
* Parameters:
*   upnp     - Pointer to the UPnP structure.
*   sendto   - Address to send the notification to.
*   location - URL of the device description.
*   nt       - Notification type.
*   usn0     - Unique service name part 0.
*   usn1     - Unique service name part 1.
*
* Return:
*   None
*
*******************************************************************************/
static void upnp_send_notify(
    upnp_t* upnp, 
    cy_socket_sockaddr_t sendto,
    const char* location, 
    const char* nt,
    const char* usn0,
    const char* usn1) {

    char notify_message[512];
    int message_length;

    const char* usn_delimiter = (usn0 != NULL) && (usn1 != NULL) ? "::" : "";

    if (usn0 == NULL)
    {
        usn0 = "";
    }

    if (usn1 == NULL)
    {
        usn1 = "";
    }

    /* Construct the SSDP NOTIFY message */
    message_length = snprintf(
        notify_message, 
        sizeof(notify_message),

        "HTTP/1.1 200 OK\r\n"
        "CACHE-CONTROL: max-age=1800\r\n"
        "EXT: DEEPCRAFT %s\r\n"
        "LOCATION: %s\r\n"
        "SERVER: UPnP/1.0 DEEPCRAFT-Board/1.0\r\n"
        "ST: %s\r\n"
        "USN: %s%s%s\r\n"        
        "\r\n",

        upnp->protocol->board.name,
        location,
        nt,
        usn0,
        usn_delimiter, 
        usn1
        );

    if (message_length < 0 || message_length >= sizeof(notify_message)) 
    {
        log_message(LOG_LEVEL_ERROR, "upnp", "Failed to construct notify message");
        return;
    }

    /* Send the notification message */
    uint32_t bytes_sent = 0;
    cy_rslt_t result = cy_socket_sendto(
        upnp->socket,
        notify_message,
        message_length,
        CY_SOCKET_FLAGS_NONE,
        &sendto,
        sizeof(sendto),
        &bytes_sent
    );

    if (result != CY_RSLT_SUCCESS || bytes_sent != message_length) 
    {
        log_message(LOG_LEVEL_ERROR, "upnp", "Failed to send notify message");
        return;
    }
}

/******************************************************************************
* Function Name: get_location_uri
********************************************************************************
* Summary:
*    Constructs the location URI for the UPnP device description.
*
* Parameters:
*   upnp     - Pointer to the UPnP structure.
*
* Return:
*   Pointer to the location URI string.
*
*******************************************************************************/
static char* get_location_uri(upnp_t* upnp) 
{
    static char location[64];
    char ip_str[16];
    cy_nw_ip_address_t tmp;
    http_t* http = SYS_SERVICE_HTTP();
    int http_port = 8080;

    if (http != NULL && http->tcp_server != NULL)
    {
        http_port = http->tcp_server->address.port;
    }

    tmp.ip.v4 = upnp->address.ip.v4;
    cy_nw_ntoa(&tmp, ip_str);
    sprintf(location, "http://%s:%d/UPnP.xml", ip_str, http_port);
    return location;
}

/******************************************************************************
* Function Name: get_uuid
********************************************************************************
* Summary:
*    Constructs the UUID for the UPnP device.
*
* Parameters:
*   upnp     - Pointer to the UPnP structure.
*
* Return:
*   Pointer to the UUID string.
*
*******************************************************************************/
static char* get_uuid(upnp_t* upnp) 
{
    static char uuid[42];
    memset(uuid, 0, sizeof(uuid));
    uint8_t* serial = upnp->protocol->board.serial.uuid;
    sprintf(uuid, "uuid:%02X%02X%02X%02X-%02X%02X-%02X%02X-%02X%02X-%02X%02X%02X%02X%02X%02X",
        serial[0], serial[1], serial[2], serial[3],
        serial[4], serial[5],
        serial[6], serial[7],
        serial[8], serial[9],
        serial[10], serial[11], serial[12], serial[13], serial[14], serial[15]);
    return uuid;
}

/******************************************************************************
* Function Name: upnp_task
********************************************************************************
* Summary:
*    Handles the UPnP task, including listening for requests and sending responses.
*
* Parameters:
*   arg - Pointer to the UPnP structure.
*
* Return:
*   None
*
*******************************************************************************/
static void upnp_task(void* arg) 
{
    upnp_t* upnp = (upnp_t*)arg;
    cy_rslt_t result;
    cy_socket_sockaddr_t client_addr;
    uint32_t addr_len = sizeof(client_addr);
    char recv_buffer[255];
    uint32_t bytes_received = 0;
    uint32_t tcp_recv_timeout = 10000;

    /* Fetch uuid */
    char* uuid = get_uuid(upnp);

    /* Create location URI */
    char* location = get_location_uri(upnp);

    /* Create a socket for listening to UPnP requests */
    cy_socket_t listen_socket;
    if (cy_socket_create(CY_SOCKET_DOMAIN_AF_INET, CY_SOCKET_TYPE_DGRAM, CY_SOCKET_IPPROTO_UDP, &listen_socket) != CY_RSLT_SUCCESS) 
    {
        log_message(LOG_LEVEL_ERROR, "upnp", "Failed to create listen socket");
        return;
    }
    upnp->socket = listen_socket;
     
    /* Set timeout to 10 seconds */
    result = cy_socket_setsockopt(
        listen_socket,
        CY_SOCKET_SOL_SOCKET,
        CY_SOCKET_SO_RCVTIMEO,
        &tcp_recv_timeout,
        sizeof(tcp_recv_timeout));
    if (result != CY_RSLT_SUCCESS) 
    {
        log_message(LOG_LEVEL_ERROR, "upnp", "Failed to set socket option CY_SOCKET_SO_RCVTIMEO");
    }

    /* Join multicast group */
    cy_socket_ip_mreq_t mreq;
    mreq.if_addr.version = CY_SOCKET_IP_VER_V4;
    mreq.if_addr.ip.v4 = upnp->address.ip.v4;
    mreq.multi_addr.version = CY_SOCKET_IP_VER_V4;
    mreq.multi_addr.ip.v4 = 0xfaffffef; /* = 239.255.255.250 */
    result = cy_socket_setsockopt(
        listen_socket,
        CY_SOCKET_SOL_IP,
        CY_SOCKET_SO_JOIN_MULTICAST_GROUP,
        &mreq,
        sizeof(cy_socket_ip_mreq_t));
    if (result != CY_RSLT_SUCCESS) 
    {
        log_message(LOG_LEVEL_ERROR, "upnp", "Failed to set socket option CY_SOCKET_SO_JOIN_MULTICAST_GROUP");
    }

    /* Bind to socket */
    cy_socket_sockaddr_t bind_addr;
    bind_addr.port = 1900;
    bind_addr.ip_address.version = CY_SOCKET_IP_VER_V4;
    bind_addr.ip_address.ip.v4 = 0xfaffffef; /* = 239.255.255.250 */
    if (cy_socket_bind(listen_socket, &bind_addr, sizeof(cy_socket_sockaddr_t)) != CY_RSLT_SUCCESS) 
    {
        log_message(LOG_LEVEL_ERROR, "upnp", "Failed to bind listen socket");
        cy_socket_delete(listen_socket);
        return;
    }

    while (1) 
    {
        /* Listen for incoming UPnP requests. This is a blocking function until there is data. */
        bytes_received = 0;
        result = cy_socket_recvfrom(
            listen_socket,
            recv_buffer,
            sizeof(recv_buffer),
            CY_SOCKET_FLAGS_NONE,
            &client_addr,
            &addr_len,
            &bytes_received
        );

        if (bytes_received > 0) 
        {
            /* Process the received UPnP request */
            recv_buffer[bytes_received] = '\0'; /* Null-terminate the received data */
            char ip_str[20];
            cy_nw_ip_address_t tmp;
            tmp.ip.v4 = client_addr.ip_address.ip.v4;
            cy_nw_ntoa(&tmp, ip_str);

            if (strncmp(recv_buffer, "M-SEARCH *", 10) == 0) 
            {
                log_message(LOG_LEVEL_DEBUG, "upnp", "Responds to UPnP request from %s", ip_str);

                const char* device_type_urn = "urn:schemas-upnp-org:device:Basic:1";

                upnp_send_notify(upnp, client_addr, location, "upnp:rootdevice",  uuid, "upnp:rootdevice");
                upnp_send_notify(upnp, client_addr, location, uuid,               uuid, NULL);
                upnp_send_notify(upnp, client_addr, location, device_type_urn,    uuid, device_type_urn);
            }
            else 
            {
                /* Quite a lot of this messages. */
                /* log_message(LOG_LEVEL_DEBUG, "upnp", "Ignored UPnP message from %s", ip_str); */
            }
        }
    }

    /* Cleanup */
    cy_socket_delete(listen_socket);
}

/******************************************************************************
* Function Name: upnp_init
********************************************************************************
* Summary:
*    Initializes the UPnP structure.
*
* Parameters:
*   upnp - Pointer to the UPnP structure.
*   protocol - Pointer to the protocol structure.
*
* Return:
*   true if the initialization was successful, otherwise false.
*
*******************************************************************************/
bool upnp_init(upnp_t* upnp, protocol_t* protocol) 
{
    memset(upnp, 0, sizeof(upnp_t));
    upnp->protocol = protocol;
    return true;
}

/******************************************************************************
* Function Name: upnp_start
********************************************************************************
* Summary:
*    Starts the UPnP service.
*
* Parameters:
*   upnp - Pointer to the UPnP structure.
*   address - IP address to bind the UPnP service.
*
* Return:
*   true if the service was started successfully, otherwise false.
*
*******************************************************************************/
bool upnp_start(upnp_t* upnp, cy_nw_ip_address_t address) 
{
    if (upnp->active) 
    {
        log_message(LOG_LEVEL_ERROR, "upnp", "UPnP service already active.");
        return false;
    }
   
    log_message(LOG_LEVEL_INFO, "upnpd", "Starting UPnP service.");
    upnp->address = address;
    upnp->active = true;
    xTaskCreate(
        upnp_task,
        "upnpd",
        UPNPD_STACK_SIZE,
        upnp,
        UPNPD_PRIO,
        &(upnp->task)
    );

    return true;
}

/******************************************************************************
* Function Name: upnp_stop
********************************************************************************
* Summary:
*    Stops the UPnP service.
*
* Parameters:
*   upnp - Pointer to the UPnP structure.
*
* Return:
*   true if the service was stopped successfully, otherwise false.
*
*******************************************************************************/
bool upnp_stop(upnp_t* upnp) 
{
    if (!upnp->active) 
    {
        log_message(LOG_LEVEL_ERROR, "upnp", "UPnP service not running.");
        return false;
    }

    log_message(LOG_LEVEL_INFO, "upnp", "Stopping UPnP service.");
    vTaskDelete(upnp->task);
    cy_socket_delete(upnp->socket);
    upnp->active = false;
    return true;
}

#endif /* IM_ENABLE_NET and IM_ENABLE_UPNP */

/* [] END OF FILE */

