/******************************************************************************
* File Name:   http.h
*
* Description: Header file for the HTTP server implementation, 
* defining the interface for handling HTTP requests and responses.
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
#ifndef _HTTP_H_
#define _HTTP_H_

#ifdef IM_ENABLE_HTTP

#include <stdbool.h>
#include "tcp.h"

/*******************************************************************************
* Macros
*******************************************************************************/
 /* Mime types */ 
#define MIME_HTML "text/html"
#define MIME_XML "application/xml"
#define MIME_PLAIN "text/plain"
#define MIME_CSS "text/css"
#define MIME_JS "application/javascript"
#define MIME_JSON "application/json"
#define MIME_PNG "image/png"
#define MIME_GIF "image/gif"
#define MIME_JPEG "image/jpeg"
#define MIME_ICO "image/x-icon"
#define MIME_PDF "application/pdf"
#define MIME_ZIP "application/zip"
#define MIME_WAV "audio/wav"
#define MIME_CSV "text/csv"
#define MIME_BINARY "application/octet-stream"

#define STATUS_200 "200 OK"

/* HTTP Protocol limits */
#define HTTP_MAX_HEADER_SIZE     (1024*4)
#define HTTP_MAX_REQUEST_SIZE    (1024*8)
#define HTTP_MAX_PARAMETER_COUNT (15)
#define HTTP_SESSION_TIMEOUT_MS  (30000)  /* 30 seconds */ 

/* Security constants */
#define HTTP_MAX_AUTH_ATTEMPTS   (5)
#define HTTP_AUTH_LOCKOUT_MS     (300000)  /* 5 minutes */
#define HTTP_MAX_FILENAME_LENGTH (255)
#define HTTP_MAX_PATH_LENGTH     (512)

/*******************************************************************************
* Types
*******************************************************************************/
/* Structure for managing the HTTP daemon. */
typedef struct {
    tcp_server_t* tcp_server;       /* Pointer to the TCP server instance. */
    void* request_buffer;           /* Buffer for incoming HTTP requests. */
} http_t;

/* Structure for HTTP parameters. */
typedef struct {
    const char* name;      /* Parameter name. */
    const char* filename;  /* Filename for file parameters. */
    char* data;            /* Parameter data. */
    int data_count;        /* Length of the data. */
} http_parameter_t;

/*******************************************************************************
* Function Prototypes
*******************************************************************************/
bool http_init(http_t* http);
bool http_start(http_t* http, cy_nw_ip_address_t address, int port);
bool http_stop(http_t* http);
void http_handle_request(tcp_session_t* session, const char* method, const char* path, http_parameter_t* params, int params_count);
bool http_send_response(tcp_session_t* session, const char* status, const char* content_type, const char* body);
bool http_send_file(tcp_session_t* session, const char* status, const char* path);
void http_send_status(tcp_session_t* session, int code);
bool http_start_chunked_response(tcp_session_t* session, const char* status, const char* content_type);
bool http_send_chunk(tcp_session_t* session, const char* data, size_t size);
bool http_end_chunked_response(tcp_session_t* session);

#endif /* IM_ENABLE_HTTP */
#endif /* _HTTP_H_ */

/* [] END OF FILE */
