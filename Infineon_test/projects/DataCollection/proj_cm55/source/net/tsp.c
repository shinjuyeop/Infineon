/******************************************************************************
* File Name:   tsp.c
*
* Description: Tensor Streaming Protocol network server implementation.
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
#ifdef IM_ENABLE_NET

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cy_nw_helper.h>
#include <FreeRTOS.h>
#include <task.h>
#include <cyabs_rtos.h>

#include "tsp.h"
#include "tcp.h"
#include "../common.h"
#include "../heap.h"
#include "../wifi_configs/im_config.h"

#include "../protocol/protocol.h"
#include "../protocol/pb_encode.h"
#include "../protocol/pb_decode.h"

/*******************************************************************************
* Types
*******************************************************************************/
typedef struct {
    tcp_session_t* session;   /**< Pointer to the TCP session. */
    TaskHandle_t task;        /**< Task handle for the session. */
    pb_istream_t istream;
    pb_ostream_t ostream;
} tsp_client_t;

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: _tsp_read
********************************************************************************
* Summary:
*    Reads data from the TCP session.
*
* Parameters:
*   stream - Pointer to the input stream.
*   buf    - Pointer to the buffer where the read data will be stored.
*   count  - Number of bytes to read.
*
* Return:
*   true if the read was successful, false otherwise.
*
*******************************************************************************/
static bool _tsp_read(pb_istream_t* stream, pb_byte_t* buf, size_t count)
{
    tsp_client_t* client = (tsp_client_t*)stream->state;
    tsp_t* tsp = client->session->server->server_context;
    protocol_t* protocol = tsp->protocol;

    uint32_t bytes_received = 0;
    uint32_t read_count;
    cy_rslt_t result;
    uint32_t option_len = sizeof(uint32_t);

    while (count > 0) 
    {
        result = cy_socket_getsockopt(
            client->session->client_socket,
            CY_SOCKET_SOL_SOCKET,
            CY_SOCKET_SO_BYTES_AVAILABLE,
            &read_count,
            &option_len);
    
        if (result != CY_RSLT_SUCCESS) 
        {
            log_message(LOG_LEVEL_ERROR, "tsp", "Failed to get bytes available for reading");
        }

        /* Do device processing while waiting for data. */
        if (read_count == 0) 
        {
            /* Polling should probably yield anyway based on the expected data rate. */
            if (protocol_call_device_poll(protocol, &(client->ostream)) == 0) 
            {
                /* If no devices invoked - sleep for a while (100 ms) 
                 * Yielding here will also yield the protocol request handling
                 * of incoming data. */
                vTaskDelay(pdMS_TO_TICKS(100));
            }
            vTaskDelay(pdMS_TO_TICKS(1));   /* Yield anyway regardless. */
            continue;
        }

        /* Only consume what was requested */
        if (read_count > count)
        {
            read_count = count;
        }

        result = cy_socket_recv(
            client->session->client_socket,
            buf,
            read_count,
            CY_SOCKET_FLAGS_NONE,
            &bytes_received);

        if (bytes_received != read_count) 
        {
            log_message(LOG_LEVEL_ERROR, "tsp", "Read error, expected %i, but got %i", read_count, bytes_received);
            return false;
        }

        if (result != CY_RSLT_SUCCESS) 
        {
            log_message(LOG_LEVEL_ERROR, "tsp", "Failed to read socket. Error %i", result);
            return false;
        }
    
        buf += bytes_received;
        count -= bytes_received;
    }

    return true;
}

/******************************************************************************
* Function Name: _tsp_write
********************************************************************************
* Summary:
*    Writes data to the TCP session.
*
* Parameters:
*   stream - Pointer to the output stream.
*   buf    - Pointer to the buffer containing the data to write.
*   count  - Number of bytes to write.
*
* Return:
*   true if the write was successful, false otherwise.
*
*******************************************************************************/
static bool _tsp_write(pb_ostream_t* stream, const pb_byte_t* buf, size_t count)
{
    tsp_client_t* client = (tsp_client_t*)stream->state;

    uint32_t bytes_sent;
    cy_rslt_t result = cy_socket_send(client->session->client_socket, buf, count, CY_SOCKET_FLAGS_NONE, &bytes_sent);
    
    if (result != CY_RSLT_SUCCESS) 
    {
        log_message(LOG_LEVEL_ERROR, "tsp socket_send()", "Write error:");
        switch ( result )
        {
            case  CY_RSLT_MODULE_SECURE_SOCKETS_INVALID_SOCKET:
            {
                log_message(LOG_LEVEL_ERROR, "tsp socket_send() ","CY_RSLT_MODULE_SECURE_SOCKETS_INVALID_SOCKET");
                break;               
            }

            case CY_RSLT_MODULE_SECURE_SOCKETS_TLS_ERROR:
            {
                log_message(LOG_LEVEL_ERROR, "tsp socket_send() ","CY_RSLT_MODULE_SECURE_SOCKETS_TLS_ERROR");
                break;
            }

            case CY_RSLT_MODULE_SECURE_SOCKETS_TCPIP_ERROR:
            {
                log_message(LOG_LEVEL_ERROR, "tsp socket_send() ","CY_RSLT_MODULE_SECURE_SOCKETS_TCPIP_ERROR");
                break;
            }

            case CY_RSLT_MODULE_SECURE_SOCKETS_TIMEOUT:
            {
                log_message(LOG_LEVEL_ERROR, "tsp socket_send() ","CY_RSLT_MODULE_SECURE_SOCKETS_TIMEOUT");
                break;
            }

            case CY_RSLT_MODULE_SECURE_SOCKETS_NOT_CONNECTED:
            {
                log_message(LOG_LEVEL_ERROR, "tsp socket_send() ","CY_RSLT_MODULE_SECURE_SOCKETS_NOT_CONNECTED");
                break;
            }

            case CY_RSLT_MODULE_SECURE_SOCKETS_CLOSED:
            {
                log_message(LOG_LEVEL_ERROR, "tsp socket_send() ","CY_RSLT_MODULE_SECURE_SOCKETS_CLOSED");
                break;
            }

            case CY_RSLT_MODULE_SECURE_SOCKETS_WOULDBLOCK:
            {
                log_message(LOG_LEVEL_ERROR, "tsp socket_send() ","CY_RSLT_MODULE_SECURE_SOCKETS_WOULDBLOCK");
                log_message(LOG_LEVEL_ERROR, "tsp socket_send() ","This usually means that the sendbuffer is full.");
                break;
            }

            default:
            {
                log_message(LOG_LEVEL_ERROR, "tsp socket_send() ","UNKNOWN ERROR %i", result);
                break;
            }

        }
        return false;
    }

    return true;
}

/******************************************************************************
* Function Name: _session_task
********************************************************************************
* Summary:
*    Handles a TCP session task.
*
* Parameters:
*   args - Pointer to the task arguments.
*
* Return:
*   None.
*
*******************************************************************************/
static void _session_task(void* args) 
{
    tsp_client_t* client = (tsp_client_t*)args;
    tsp_t* tsp = client->session->server->server_context;
    protocol_t* protocol = tsp->protocol;

    client->istream = (pb_istream_t){ &_tsp_read, (void*)client, SIZE_MAX, 0 };
    client->ostream = (pb_ostream_t){ &_tsp_write, (void*)client, SIZE_MAX, 0, NULL };

    log_message(LOG_LEVEL_INFO, "tsp", "Started new TSP task");

    for (;;) 
    {
        log_message(LOG_LEVEL_DEBUG, "tsp", "Waiting for next package");

        protocol_process_request(protocol, &(client->istream), &(client->ostream));
    }
}

/******************************************************************************
* Function Name: _session_opened
********************************************************************************
* Summary:
*    Handles a TCP session opened event.
*
* Parameters:
*   session - Pointer to the TCP session.
*
* Return:
*   None.
*
*******************************************************************************/
static void _session_opened(tcp_session_t* session) 
{
    tsp_client_t* client = net_malloc(sizeof(tsp_client_t));

    if (client == NULL) 
    {
        log_message(LOG_LEVEL_ERROR, "tsp", "Failed to allocate memory for client");
        return;
    }
    memset(client, 0, sizeof(tsp_client_t));
    client->session = session;
    session->client_context = client;

    if (xTaskCreate(_session_task, "tsp_session", TSP_SESSION_STACK_SIZE, client, TSP_SESSION_PRIO, &(client->task)) != pdPASS) 
    {
        log_message(LOG_LEVEL_ERROR, "tsp", "Failed to create tsp_session_task");
        net_free(client);
        return;
    }
}

/******************************************************************************
* Function Name: _session_closed
********************************************************************************
* Summary:
*    Handles a TCP session closed event.
*
* Parameters:
*   session - Pointer to the TCP session.
*
* Return:
*   None.
*
*******************************************************************************/
static void _session_closed(tcp_session_t* session) 
{
    tsp_client_t* client = (tsp_client_t*)session->client_context;

    char ip_addr_str[16];
    cy_nw_ip_address_t nw_peer_addr;
    nw_peer_addr.version = NW_IP_IPV4;
    nw_peer_addr.ip.v4 = session->peer_addr.ip_address.ip.v4;
    cy_nw_ntoa(&nw_peer_addr, ip_addr_str);
  
    /* Can we shut-down the sensor here too?
    * It appears that if it is not shut-down there will be no reinitialization
    * of it when started again.
    */ 
    tsp_t* tsp = client->session->server->server_context;
    protocol_t* protocol = tsp->protocol;
    protocol_call_device_stop( protocol, (pb_ostream_t*)NULL);
  
  
    session->client_context = NULL;

    log_message(LOG_LEVEL_DEBUG, "tsp", "TSP session %s closing", ip_addr_str);

    if (client->task != NULL)
    {
       vTaskDelete(client->task);
    }

    net_free(client);

    log_message(LOG_LEVEL_INFO, "tsp", "TSP session %s closed", ip_addr_str);
}

/******************************************************************************
* Function Name: tsp_init
********************************************************************************
* Summary:
*    Initializes the TSP context.
*
* Parameters:
*   tsp - Pointer to the TSP context.
*   protocol - Pointer to the protocol context.
*
* Return:
*   true if initialization was successful, false otherwise.
*
*******************************************************************************/
bool tsp_init(tsp_t* tsp, protocol_t* protocol) 
{
    memset(tsp, 0, sizeof(tsp_t));
    tsp->protocol = protocol;
    return true;
}

/******************************************************************************
* Function Name: tsp_start
********************************************************************************
* Summary:
*    Starts the TSP server.
*
* Parameters:
*   tsp - Pointer to the TSP context.
*   address - IP address to bind the server.
*   port - Port number to bind the server.
*
* Return:
*   true if the server was started successfully, false otherwise.
*******************************************************************************/
bool tsp_start(tsp_t* tsp, cy_nw_ip_address_t address, int port) 
{
    tsp->tcp_server = network_create_tcp_server(
        address,
        port,
        3,
        _session_opened,
        NULL,
        _session_closed,
        tsp);

    if (tsp->tcp_server == NULL) 
    {
        log_message(LOG_LEVEL_ERROR, "tsp", "Failed to create TSP server");
        return false;
    }

    char ip_addr_str[16];
    cy_nw_ntoa(&address, ip_addr_str);
    log_message(LOG_LEVEL_INFO, "tsp", "TSP network server started on %s:%d", ip_addr_str, port);

    return true;
}

/******************************************************************************
* Function Name: tsp_stop
********************************************************************************
* Summary:
*    Stops the TSP server.
*
* Parameters:
*   tsp - Pointer to the TSP context.
*
* Return:
*   true if the server was stopped successfully, false otherwise.
*
*******************************************************************************/
bool tsp_stop(tsp_t* tsp) 
{
    if (tsp->tcp_server == NULL) 
    {
        log_message(LOG_LEVEL_ERROR, "tsp", "TSP server stopped.");
        return false;
    }

    network_destroy_tcp_server(tsp->tcp_server);
    tsp->tcp_server = NULL;

    return true;
}

#endif /* IM_ENABLE_NET */

/* [] END OF FILE */
