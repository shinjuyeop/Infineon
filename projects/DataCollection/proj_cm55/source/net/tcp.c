/******************************************************************************
* File Name:   tcp.c
*
* Description: Implementation of TCP server and session management for 
* handling client connections.
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
#include <string.h>
#include <stdlib.h>

#include <cybsp.h>
#include <cyabs_rtos.h>
#include <cy_secure_sockets.h>
#include <cy_nw_helper.h>
#include <inttypes.h>

#include "../common.h"
#include "../heap.h"
#include "tcp.h"

#include <sockets.h>

/*******************************************************************************
* Macros
*******************************************************************************/

/* TCP server related macros. */
#define TCP_SERVER_MAX_PENDING_CONNECTIONS       (3)
#define TCP_SERVER_RECV_TIMEOUT_MS               (500)

/* TCP keep alive related macros. */
#define TCP_KEEP_ALIVE_IDLE_TIME_MS              (10000)
#define TCP_KEEP_ALIVE_INTERVAL_MS               (1000)
#define TCP_KEEP_ALIVE_RETRY_COUNT               (2)

#define TLS_TIMEOUT_MS 1000

/*******************************************************************************
* Function Prototypes
*******************************************************************************/
static cy_rslt_t _tcp_connection_handler(cy_socket_t server_socket, void *arg);
static cy_rslt_t _tcp_receive_handler(cy_socket_t client_socket, void *arg);
static cy_rslt_t _tcp_disconnection_handler(cy_socket_t client_socket, void *arg);

/******************************************************************************
* Function Name: _tcp_connection_handler
********************************************************************************
* Summary:
*    Handles new TCP connections.
*    This function is called when a new TCP connection is accepted by the server.
*
* Parameters:
*   server_socket - Server's socket.
*   arg           - Pointer to the server structure.
*
* Return:
*   cy_rslt_t Result of the connection handling.
*
*******************************************************************************/
static cy_rslt_t _tcp_connection_handler(cy_socket_t server_socket, void *arg)
{
    cy_rslt_t result;
    char ip_addr_str[16];

    tcp_server_t* server = (tcp_server_t *)arg;

    cy_socket_sockaddr_t peer_addr;
    cy_socket_t client_socket;
    uint32_t peer_addr_len;

    /* TCP keep alive parameters. */
    int keep_alive = 1;
#if defined (COMPONENT_LWIP)
    uint32_t keep_alive_interval = TCP_KEEP_ALIVE_INTERVAL_MS;
    uint32_t keep_alive_count    = TCP_KEEP_ALIVE_RETRY_COUNT;
    uint32_t keep_alive_idle_time = TCP_KEEP_ALIVE_IDLE_TIME_MS;
#endif

    if(server->session_count >= server->sessions_max)
    {
        log_message(LOG_LEVEL_INFO, "network", "Max number of sessions reached");
        return CY_RSLT_SUCCESS;
    }

    /* Accept new incoming connection from a TCP client.*/
    log_message(LOG_LEVEL_DEBUG, "network", "Will accept a connection.");
    result = cy_socket_accept(server_socket, &peer_addr, &peer_addr_len, &client_socket);
    if(result != CY_RSLT_SUCCESS)
    {
        
        char ip_str[20];
        cy_nw_ip_address_t tmp;
        tmp.ip.v4 = peer_addr.ip_address.ip.v4;
        cy_nw_ntoa(&tmp, ip_str);
        log_message(LOG_LEVEL_DEBUG, "network accept()", "Failed to accept incoming client connection from %s", ip_str);
        switch ( result )
        {
            case CY_RSLT_MODULE_SECURE_SOCKETS_INVALID_SOCKET:
            {
                log_message(LOG_LEVEL_DEBUG, "network accept()", "CY_RSLT_MODULE_SECURE_SOCKETS_INVALID_SOCKET" );
                break;
            }
            case CY_RSLT_MODULE_SECURE_SOCKETS_TLS_ERROR:
            {
                log_message(LOG_LEVEL_DEBUG, "network accept()", "CY_RSLT_MODULE_SECURE_SOCKETS_TLS_ERROR" );
                break;
            }
            case CY_RSLT_MODULE_SECURE_SOCKETS_TIMEOUT:
            {
                log_message(LOG_LEVEL_DEBUG, "network accept()", "CY_RSLT_MODULE_SECURE_SOCKETS_TIMEOUT" );
                break;
            }
            case CY_RSLT_MODULE_SECURE_SOCKETS_NOMEM:
            {
                log_message(LOG_LEVEL_DEBUG, "network accept()", "CY_RSLT_MODULE_SECURE_SOCKETS_NOMEM" );
                break;
            }
            case CY_RSLT_MODULE_SECURE_SOCKETS_BADARG:
            {
                log_message(LOG_LEVEL_DEBUG, "network accept()", "CY_RSLT_MODULE_SECURE_SOCKETS_BADARG" );
                break;
            }
            case CY_RSLT_MODULE_SECURE_SOCKETS_NOT_LISTENING:
            {
                log_message(LOG_LEVEL_DEBUG, "network accept()", "CY_RSLT_MODULE_SECURE_SOCKETS_NOT_LISTENING" );
                break;
            }
            default:
            {
                log_message(LOG_LEVEL_DEBUG, "network accept()", "UNKNOWN ERROR 0x%08lx", result );
                break;
            }
        }
        
        return result;
    }
    
#if defined (COMPONENT_LWIP)
    /* Set the TCP keep alive interval. */
    result = cy_socket_setsockopt(
            client_socket,
            CY_SOCKET_SOL_TCP,
            CY_SOCKET_SO_TCP_KEEPALIVE_INTERVAL,
            &keep_alive_interval,
            sizeof(keep_alive_interval));
    if(result != CY_RSLT_SUCCESS)
    {
        log_message(LOG_LEVEL_ERROR, "network", "Set socket option: CY_SOCKET_SO_TCP_KEEPALIVE_INTERVAL failed");
        return result;
    }

    /* Set the retry count for TCP keep alive packet. */
    result = cy_socket_setsockopt(
            client_socket,
            CY_SOCKET_SOL_TCP,
            CY_SOCKET_SO_TCP_KEEPALIVE_COUNT,
            &keep_alive_count,
            sizeof(keep_alive_count));
    if(result != CY_RSLT_SUCCESS)
    {
        log_message(LOG_LEVEL_ERROR, "network", "Set socket option: CY_SOCKET_SO_TCP_KEEPALIVE_COUNT failed");
        return result;
    }

    /* Set the network idle time before sending the TCP keep alive packet. */
    result = cy_socket_setsockopt(
            client_socket,
            CY_SOCKET_SOL_TCP,
            CY_SOCKET_SO_TCP_KEEPALIVE_IDLE_TIME,
            &keep_alive_idle_time,
            sizeof(keep_alive_idle_time));
    if(result != CY_RSLT_SUCCESS)
    {
        log_message(LOG_LEVEL_ERROR, "network", "Set socket option: CY_SOCKET_SO_TCP_KEEPALIVE_IDLE_TIME failed");
        return result;
    }
#endif

    /* Enable TCP keep alive. */
    result = cy_socket_setsockopt(
        client_socket,
        CY_SOCKET_SOL_SOCKET,
        CY_SOCKET_SO_TCP_KEEPALIVE_ENABLE,
        &keep_alive,
        sizeof(keep_alive));

    if(result != CY_RSLT_SUCCESS)
    {
        log_message(LOG_LEVEL_ERROR, "network", "Set socket option: CY_SOCKET_SO_TCP_KEEPALIVE_ENABLE failed");
        return result;
    }

    /* IP variable for network utility functions */
    cy_nw_ip_address_t nw_peer_addr;
    nw_peer_addr.version = NW_IP_IPV4;
    nw_peer_addr.ip.v4 = peer_addr.ip_address.ip.v4;
    cy_nw_ntoa(&nw_peer_addr, ip_addr_str);
    log_message(LOG_LEVEL_DEBUG, "network", "Incoming TCP connection from : %s", ip_addr_str);

    tcp_session_t* session = net_malloc(sizeof(tcp_session_t));
    if (session == NULL) {
        log_message(LOG_LEVEL_ERROR, "network", "Failed to allocate memory for new session");
        cy_socket_disconnect(client_socket, 0);
        cy_socket_delete(client_socket);
        return CY_RSLT_SUCCESS;
    }

    server->sessions[server->session_count++] = session;
    session->server = server;
    session->client_context = NULL;
    session->client_socket = client_socket;
    session->peer_addr = peer_addr;

    // This creates s session thread for the tsp-server.
    if( server->session_opened != NULL)
        server->session_opened(session);

    return CY_RSLT_SUCCESS;
}

/******************************************************************************
* Function Name: _tcp_receive_handler
********************************************************************************
* Summary:
*    Handles incoming data on a TCP connection.
*    This function is called when data is received on a TCP connection.
*
* Parameters:
*   client_socket - Client's socket.
*   arg           - Pointer to the server structure.
*
* Return:
*   cy_rslt_t Result of the receive handling.
*
*******************************************************************************/
static cy_rslt_t _tcp_receive_handler(cy_socket_t client_socket, void *arg)
{
    tcp_server_t* server = (tcp_server_t *)arg;

    tcp_session_t* session = NULL;
    for(int i = 0; i < server->session_count; i++) 
    {
        if( server->sessions[i]->client_socket == client_socket) 
        {
            session = server->sessions[i];
            break;
        }
    }

    if(session == NULL) 
    {
         log_message(LOG_LEVEL_ERROR, "network", "TCP Server error, unknown session");
        /* Disconnect the socket. */
        cy_socket_disconnect(client_socket, 0);
        /* Delete the socket. */
        cy_socket_delete(client_socket);

        return CY_RSLT_SUCCESS;
    }

    if(server->session_receive != NULL)
    {
        server->session_receive(session);
    }
    return CY_RSLT_SUCCESS;
}

/******************************************************************************
* Function Name: _tcp_disconnection_handler
********************************************************************************
* Summary:
*    Handles TCP disconnections.
*    This function is called when a TCP connection is disconnected.
*
* Parameters:
*   client_socket - Client's socket.
*   arg           - Pointer to the server structure.
*
* Return:
*   cy_rslt_t Result of the disconnection handling.
*
*******************************************************************************/
static cy_rslt_t _tcp_disconnection_handler(cy_socket_t client_socket, void *arg)
{  
    tcp_server_t* server = (tcp_server_t *)arg;

    tcp_session_t* session = NULL;
    int session_index = -1;
    for(int i = 0; i < server->session_count; i++) 
    {
        if(server->sessions[i]->client_socket == client_socket) 
        {
            session_index = i;
            session = server->sessions[i];
            break;
        }
    }

    if(session != NULL) 
    {
        if (server->session_closed != NULL) 
        {
            server->session_closed(session);
        }

        /* Remove session from list */
        if (session_index >= 0 && session_index < server->session_count) 
        {
            if (session_index < server->session_count - 1) 
            {
                memmove(&server->sessions[session_index], &server->sessions[session_index + 1],
                        (server->session_count - session_index - 1) * sizeof(tcp_session_t*));
            }
            server->session_count--;
        }

        /* Clean up the session */
        net_free(session);
    }

    cy_socket_disconnect(client_socket, 0);
    cy_socket_delete(client_socket);

    log_message(LOG_LEVEL_DEBUG, "network", "TCP Client disconnected");
    return CY_RSLT_SUCCESS;
}

/******************************************************************************
* Function Name: network_create_tcp_server
********************************************************************************
* Summary:
*    Creates a TCP server.
*    This function initializes a TCP server with the specified parameters.
*
* Parameters:
*   address        - IP address for the server.
*   port           - Port number for the server.
*   sessions_max   - Maximum number of concurrent sessions.
*   session_opened - Callback function for session opened.
*   session_receive - Callback function for session received.
*   session_closed - Callback function for session closed.
*   server_context - Pointer to server context.
*
* Return:
*   tcp_server_t* Pointer to the created TCP server, or NULL on failure.
*
*******************************************************************************/
tcp_server_t* network_create_tcp_server(
        cy_nw_ip_address_t address,
        int port,
        int sessions_max,
        tcp_session_fn session_opened,
        tcp_session_fn session_receive,
        tcp_session_fn session_closed,
        void *server_context)
{
    cy_rslt_t result;
    uint32_t tcp_recv_timeout = TCP_SERVER_RECV_TIMEOUT_MS;
    cy_socket_opt_callback_t tcp_receive_option;
    cy_socket_opt_callback_t tcp_connection_option;
    cy_socket_opt_callback_t tcp_disconnection_option;

    /* Allocate server */
    tcp_server_t* server = (tcp_server_t*)net_malloc(sizeof(tcp_server_t));
    if(server == NULL) 
    {
        log_message(LOG_LEVEL_ERROR, "network", "Failed to allocate memory for server");
        return NULL;
    }
    server->address.ip_address.ip.v4 = address.ip.v4;
    server->address.ip_address.version = CY_SOCKET_IP_VER_V4;
    server->session_opened = session_opened;
    server->session_receive = session_receive;
    server->session_closed = session_closed;
    server->server_context = server_context;
    server->address.port = port;
    server->session_count = 0;
    server->sessions_max = sessions_max;
    server->sessions = (tcp_session_t**)net_malloc(sizeof(tcp_session_t *) * sessions_max);
    if(server->sessions == NULL) 
    {
        log_message(LOG_LEVEL_ERROR, "network", "Failed to allocate memory for session array");
        net_free(server);
        return NULL;
    }

    /* Create a TCP socket */
    result = cy_socket_create(
        CY_SOCKET_DOMAIN_AF_INET,
        CY_SOCKET_TYPE_STREAM,
        CY_SOCKET_IPPROTO_TCP,
        &server->server_socket);

    if(result != CY_RSLT_SUCCESS) 
    {
        log_message(LOG_LEVEL_ERROR, "network", "Failed to create socket");
        net_free(server->sessions);
        net_free(server);
        return NULL;
    }

    /* Set the TCP socket receive timeout period. */
    result = cy_socket_setsockopt(
        server->server_socket,
        CY_SOCKET_SOL_SOCKET,
        CY_SOCKET_SO_RCVTIMEO,
        &tcp_recv_timeout,
        sizeof(tcp_recv_timeout));

    if(result != CY_RSLT_SUCCESS) 
    {
        log_message(LOG_LEVEL_ERROR, "network", "Set socket option: CY_SOCKET_SO_RCVTIMEO failed");
        cy_socket_delete(server->server_socket);
        net_free(server->sessions);
        net_free(server);
        return NULL;
    }

    /* Register the callback function to handle connection request from a TCP client. */
    tcp_connection_option.callback = _tcp_connection_handler;
    tcp_connection_option.arg = server;

    result = cy_socket_setsockopt(
        server->server_socket,
        CY_SOCKET_SOL_SOCKET,
        CY_SOCKET_SO_CONNECT_REQUEST_CALLBACK,
        &tcp_connection_option,
        sizeof(cy_socket_opt_callback_t));

    if(result != CY_RSLT_SUCCESS) 
    {
        log_message(LOG_LEVEL_ERROR, "network", "Set socket option: CY_SOCKET_SO_CONNECT_REQUEST_CALLBACK failed");
        cy_socket_delete(server->server_socket);
        net_free(server->sessions);
        net_free(server);
        return NULL;
    }

    /* Register the callback function to handle messages received from a TCP client. */
    if (session_receive != NULL) 
    {
        tcp_receive_option.callback = _tcp_receive_handler;
        tcp_receive_option.arg = server;

        result = cy_socket_setsockopt(
            server->server_socket,
            CY_SOCKET_SOL_SOCKET,
            CY_SOCKET_SO_RECEIVE_CALLBACK,
            &tcp_receive_option,
            sizeof(cy_socket_opt_callback_t));

        if (result != CY_RSLT_SUCCESS) {
            log_message(LOG_LEVEL_ERROR, "network", "Set socket option: CY_SOCKET_SO_RECEIVE_CALLBACK failed");
            cy_socket_delete(server->server_socket);
            net_free(server->sessions);
            net_free(server);
            return NULL;
        }
    }

    /* Register the callback function to handle disconnection. */
    tcp_disconnection_option.callback = _tcp_disconnection_handler;
    tcp_disconnection_option.arg = server;

    result = cy_socket_setsockopt(
        server->server_socket,
        CY_SOCKET_SOL_SOCKET,
        CY_SOCKET_SO_DISCONNECT_CALLBACK,
        &tcp_disconnection_option,
        sizeof(cy_socket_opt_callback_t));

    if(result != CY_RSLT_SUCCESS) 
    {
        log_message(LOG_LEVEL_ERROR, "network", "Set socket option: CY_SOCKET_SO_DISCONNECT_CALLBACK failed");
        cy_socket_delete(server->server_socket);
        net_free(server->sessions);
        net_free(server);
        return NULL;
    }

    result = cy_socket_bind(server->server_socket, &server->address, sizeof(cy_socket_sockaddr_t));
    if(result != CY_RSLT_SUCCESS) 
    {
        log_message(LOG_LEVEL_ERROR, "network", "Failed to bind to socket.");
        cy_socket_delete(server->server_socket);
        net_free(server->sessions);
        net_free(server);
        return NULL;
    }

    /* Start listening on the TCP server socket. */
    result = cy_socket_listen(server->server_socket, TCP_SERVER_MAX_PENDING_CONNECTIONS);
    if (result != CY_RSLT_SUCCESS) 
    {
        log_message(LOG_LEVEL_ERROR, "network", "cy_socket_listen failed");
        cy_socket_delete(server->server_socket);
        net_free(server->sessions);
        net_free(server);
        return NULL;
    }

    log_message(LOG_LEVEL_INFO, "network", "Listening for incoming TCP client connection on Port: %d", port);
    return server;
}

/******************************************************************************
* Function Name: network_destroy_tcp_server
********************************************************************************
* Summary:
*    Destroys a TCP server.
*    This function cleans up and releases all resources associated with a TCP server.
*
* Parameters:
*   server - Pointer to the TCP server to be destroyed.

*
* Return:
*   None
*
*******************************************************************************/
void network_destroy_tcp_server(tcp_server_t* server)
{
     if (server == NULL) 
     {
        return;
     }

     log_message(LOG_LEVEL_INFO, "network", "Closing TCP server on port: %d", server->address.port);

     for (int i = 0; i < server->session_count; i++) 
     {
          /* Disconnect the socket. */
         cy_socket_disconnect(server->sessions[i]->client_socket, 0);
         cy_socket_delete(server->sessions[i]->client_socket);

         net_free(server->sessions[i]);
     }

     cy_socket_delete(server->server_socket);
     net_free(server->sessions);
     net_free(server);
}

#endif /* IM_ENABLE_NET */

/* [] END OF FILE */
