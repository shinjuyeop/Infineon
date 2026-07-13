/******************************************************************************
* File Name:   tcp.h
*
* Description:  Header file for TCP server and session management.
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

#ifndef _NETWORK_H_
#define _NETWORK_H_

#ifdef IM_ENABLE_NET

/* IP address related header files. */
#include <cy_nw_helper.h>
#include <cy_wcm.h>
#include <cy_secure_sockets.h>

/*******************************************************************************
* Types
*******************************************************************************/
typedef struct tcp_session_s tcp_session_t;

/* Function pointer type for TCP session events. */
typedef void (*tcp_session_fn)(tcp_session_t* session);

/* Structure for managing a TCP server. */
typedef struct {
    void* server_context;          /* Context associated with the server. */
    tcp_session_fn session_opened; /* Callback for session opened event. */
    tcp_session_fn session_receive;/* Callback for session receive event. */
    tcp_session_fn session_closed; /* Callback for session closed event. */
    tcp_session_t** sessions;      /* Array of active sessions. */
    int sessions_max;              /* Maximum number of sessions. */
    int session_count;             /* Current number of active sessions. */
    cy_socket_sockaddr_t address;  /* Server's socket address. */
    cy_socket_t server_socket;     /* Server's socket. */
} tcp_server_t;

/* Structure for managing a TCP session. */
typedef struct tcp_session_s {
    void* client_context;          /* Context associated with the client. */
    cy_socket_t client_socket;     /* Client's socket. */
    cy_socket_sockaddr_t peer_addr;/* Client's peer address. */
    tcp_server_t* server;          /* Pointer to the server structure. */
} tcp_session_t;

/*******************************************************************************
* Function Prototypes
*******************************************************************************/
tcp_server_t* network_create_tcp_server(
    cy_nw_ip_address_t address,
    int port,
    int sessions_max,
    tcp_session_fn session_opened,
    tcp_session_fn session_receive,
    tcp_session_fn session_closed,
    void *server_context);

void network_destroy_tcp_server(tcp_server_t* server);

#endif /* IM_ENABLE_NET */
#endif /* _NETWORK_H_ */

/* [] END OF FILE */
