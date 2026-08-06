/******************************************************************************
* File Name:   http.c
*
* Description: Implementation of a simple HTTP server for handling requests 
*              and responses.
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

#ifdef IM_ENABLE_HTTP

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tcp.h"
#include "http.h"
#include "http_utils.h"
#include "../shell/process.h"
#include "../common.h"
#include "../shell/lib/base64.h"

/*******************************************************************************
* Macros
*******************************************************************************/
#define HTTP_MAX_REQUEST_SIZE    (1024*8)
#define HTTP_MAX_PARAMETER_COUNT (15)

/* Security: Rate limiting for authentication */ 
#define MAX_AUTH_ATTEMPTS 5
#define AUTH_LOCKOUT_SECONDS 300  /* 5 minutes */

/*******************************************************************************
* Types
*******************************************************************************/
typedef struct {
    tcp_session_t* session;   /**< Pointer to the TCP session. */
    uint32_t auth_attempts;   /**< Number of failed auth attempts. */
    time_t last_auth_attempt; /**< Time of last auth attempt. */
} http_client_t;

/*******************************************************************************
* Function Prototypes
*******************************************************************************/
static void session_opened(tcp_session_t* session);
static void session_receive(tcp_session_t* session);
static bool session_send(tcp_session_t* session, const char* data, size_t count);
static void session_closed(tcp_session_t* session);

static void decode_url(char* str);
static void process_request(tcp_session_t* session, char* buffer, uint32_t bytes_received);
static bool extract_auth_header(const char* buffer, char* auth_header, size_t auth_header_size);
static bool parse_multipart_form_data(char* buffer, const char* boundary, http_parameter_t* parameters, int* parameter_count);
static bool send_response_header(tcp_session_t* session, const char* status, const char* content_type, long content_length);
static const char* get_mime_type(const char* path);
static bool check_authorization(tcp_session_t* session, const char* auth_header);
static bool is_rate_limited(http_client_t* client);

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/*****************************************************************************
* Function name: session_opened
*****************************************************************************
* Summary:
*  Handles the opening of a new HTTP session. Allocates and initializes a
*  new http_client_t structure for the session.
*
* Parameters:
*  session      Pointer to the TCP session.
*
* Return:
*  None
*
*****************************************************************************/
static void session_opened(tcp_session_t* session) 
{
    http_client_t* client = net_malloc(sizeof(http_client_t));
    if (client == NULL) 
    {
        log_message(LOG_LEVEL_ERROR, "httpd", "Failed to allocate memory for client");
        return;
    }
    memset(client, 0, sizeof(http_client_t));
    client->session = session;
    client->auth_attempts = 0;
    client->last_auth_attempt = 0;
    session->client_context = client;
}

/*****************************************************************************
* Function name: session_receive
*****************************************************************************
* Summary:
*  Handles the reception of data in an HTTP session.
*  
*
* Parameters:
*  session      Pointer to the TCP session.
*
* Return:
*  None
*
*****************************************************************************/
static void session_receive(tcp_session_t* session) 
{
    cy_rslt_t result;
    uint32_t bytes_available;
    uint32_t option_len = sizeof(uint32_t);

    result = cy_socket_getsockopt(
        session->client_socket,
        CY_SOCKET_SOL_SOCKET,
        CY_SOCKET_SO_BYTES_AVAILABLE,
        &bytes_available,
        &option_len);

    if (result != CY_RSLT_SUCCESS) 
    {
        log_message(LOG_LEVEL_ERROR, "httpd", "Failed to get bytes available for reading");
        return;
    }

    if (bytes_available > HTTP_MAX_REQUEST_SIZE) 
    {
        log_message(LOG_LEVEL_ERROR, "httpd", "Request too large: %u bytes", bytes_available);
        http_send_status(session, 413);  /* Send "413 Payload Too Large" response */
        return;
    }

    http_t* server = (http_t*)session->server->server_context;
    char* buffer = (char*)server->request_buffer;

    uint32_t bytes_received = 0;
    result = cy_socket_recv(session->client_socket, buffer, bytes_available, CY_SOCKET_FLAGS_NONE, &bytes_received);

    if (result != CY_RSLT_SUCCESS || bytes_received == 0) 
    {
        return;
    }

    /* Ensure null termination */
    if (bytes_received < HTTP_MAX_REQUEST_SIZE) 
    {
        buffer[bytes_received] = '\0';
    } 
    else 
    {
        buffer[HTTP_MAX_REQUEST_SIZE - 1] = '\0';
    }

    process_request(session, buffer, bytes_received);
}

/*****************************************************************************
* Function name: session_send
*****************************************************************************
* Summary:
*  Sends data over a TCP session.
*  
*
* Parameters:
*  session      Pointer to the TCP session.
*  data         Pointer to the data to be sent.
*  count        Number of bytes to send from the data buffer.
*
* Return:
*  true if all data is sent successfully, otherwise false.
*
*****************************************************************************/
static bool session_send(tcp_session_t* session, const char* data, size_t count) 
{
    uint32_t bytes_sent = 0;
    while (count > 0) 
    {
        cy_rslt_t result = cy_socket_send(session->client_socket, data, count, CY_SOCKET_FLAGS_NONE, &bytes_sent);
        if (result != CY_RSLT_SUCCESS || bytes_sent <= 0)
        {
            return false;
        }
        count -= bytes_sent;
        data += bytes_sent;
    }
    return true;
}

/*****************************************************************************
* Function name: session_closed
*****************************************************************************
* Summary:
*  Handles the closure of an HTTP session.
*  
*
* Parameters:
*  session      Pointer to the TCP session.
*  count        Number of bytes to send from the data buffer.
*
* Return:
*  None
*
*****************************************************************************/
static void session_closed(tcp_session_t* session) 
{
    http_client_t* client = (http_client_t*)session->client_context;
    net_free(client);
    session->client_context = NULL;
    log_message(LOG_LEVEL_DEBUG, "httpd", "HTTP session closed");
}

/*****************************************************************************
* Function name: process_request
*****************************************************************************
* Summary:
*  Processes an HTTP request.
*  
*
* Parameters:
*  session      Pointer to the TCP session.
*  buffer       Pointer to the request buffer.
*  bytes_received Number of bytes received.
*
* Return:
*  None
*
*****************************************************************************/
static void process_request(tcp_session_t* session, char* buffer, uint32_t bytes_received) 
{
    char method[8];
    char path[256];
    char auth_header[256] = { 0 };
    http_parameter_t http_parameters[HTTP_MAX_PARAMETER_COUNT];
    int parameter_count = 0;

    if (sscanf(buffer, "%s %s", method, path) != 2) 
    {
        log_message(LOG_LEVEL_ERROR, "httpd", "Bad request. Method and path.");
        http_send_status(session, 400);
        return;
    }

    /* Decode URL-encoded characters */
    decode_url(path);

    /* Extract authorization header for protected endpoints */
    bool has_auth = extract_auth_header(buffer, auth_header, sizeof(auth_header));

    /* Check Content-Type for multipart/form-data */
    char boundary[256];
    const char* boundary_ptr = strstr(buffer, "boundary=");
    if (strstr(buffer, "Content-Type: multipart/form-data;") != NULL &&
        boundary_ptr != NULL &&
        sscanf(boundary_ptr, "boundary=%255s", boundary) == 1) 
    {
            if (!parse_multipart_form_data(buffer, boundary, http_parameters, &parameter_count)) 
            {
                log_message(LOG_LEVEL_ERROR, "httpd", "Bad multipart request. parse_multipart_form_data.");
                http_send_status(session, 400);
                return;
            }
    }

    /* Check authorization for protected endpoints (skip for login, UPnP discovery, and public files) */
    if (strcmp(path, "/login") != 0 && 
        strcmp(path, "/UPnP.xml") != 0 &&
        strncmp(path, "/system/web/", 12) != 0 && 
        strcmp(path, "/") != 0) {
        if (!has_auth || !check_authorization(session, auth_header)) 
        {
            http_send_status(session, 401);
            return;
        }
    }

    http_handle_request(session, method, path, http_parameters, parameter_count);
}

/*****************************************************************************
* Function name: hex_to_int
*****************************************************************************
* Summary:
*  Converts a hexadecimal character to its integer value.
*  
*
* Parameters:
*  c      Hexadecimal character.
*
* Return:
*  Integer value of the hexadecimal character.
*
*****************************************************************************/
static int hex_to_int(char c) 
{
    return (c >= '0' && c <= '9') ? c - '0'
        : (c >= 'a' && c <= 'f') ? c - 'a' + 10
        : (c >= 'A' && c <= 'F') ? c - 'A' + 10
        : 0;
}

/*****************************************************************************
* Function name: decode_url
*****************************************************************************
* Summary:
*  Decodes a URL-encoded string in-place.
*  
*
* Parameters:
*  str      Pointer to the URL-encoded string.
*
* Return:
*  None
*
*****************************************************************************/
static void decode_url(char* str) 
{
    char* dest = str;
    char* data = str;
    while (*data) 
    {
        if (*data == '%') 
        {
            if (data[1] && data[2]) 
            {
                *dest = (char)((hex_to_int(data[1]) << 4) | hex_to_int(data[2]));
                data += 2;
            }
        }
        else if (*data == '+') 
        {
            *dest = ' ';
        }
        else 
        {
            *dest = *data;
        }
        data++;
        dest++;
    }
    *dest = '\0';
}

/*****************************************************************************
* Function name: extract_auth_header
*****************************************************************************
* Summary:
*  Extracts the Authorization header from an HTTP request buffer.
*  
*
* Parameters:
*  buffer          Pointer to the HTTP request buffer.
*  auth_header     Pointer to the buffer where the extracted authorization 
*                  header value will be stored.
*  auth_header_size Size of the `auth_header` buffer.
*
* Return:
*  true if the Authorization header is found and extracted successfully,
*  otherwise false.
*
*****************************************************************************/
static bool extract_auth_header(const char* buffer, char* auth_header, size_t auth_header_size) 
{
    const char* auth_prefix = "Authorization: Basic ";
    char* auth_line = strstr(buffer, auth_prefix);
    if (auth_line != NULL) 
    {
        /* Move pointer to the start of the encoded value */
        auth_line += strlen(auth_prefix); 

        /* Find the end of the line */
        const char* end_of_line = strchr(auth_line, '\n');
        if (end_of_line == NULL) 
        {
            return false; /* No newline character found, cannot extract */
        }

        size_t length_to_copy = end_of_line - auth_line;

        /* Ensure we don't exceed auth_header_size */
        if (length_to_copy >= auth_header_size) 
        {
            length_to_copy = auth_header_size - 1;
        }

        strncpy(auth_header, auth_line, length_to_copy);
        auth_header[length_to_copy] = '\0'; /* Null-terminate the output string */

        return true;
    }
    return false;
}

/*****************************************************************************
* Function name: parse_multipart_form_data
*****************************************************************************
* Summary:
*  Parses multipart form-data from an HTTP request buffer.
*  
* Parameters:
*  buffer          Pointer to the HTTP request buffer containing the multipart 
*                  form-data.
*  boundary        Pointer to the boundary string that separates parts in the 
*                  form-data.
*  parameters      Pointer to an array of `http_parameter_t` structures where 
*                  parsed parameters will be stored.
*  parameter_count Pointer to an integer that will be updated with the number 
*                  of parsed parameters.
*
* Return:
*  true if the multipart form-data is parsed successfully, otherwise false.
*
*****************************************************************************/
static bool parse_multipart_form_data(char* buffer, const char* boundary, 
    http_parameter_t* parameters, int* parameter_count)
{
    char* part = strstr(buffer, boundary);
    if (part == NULL) 
    {
        return false;
    }
    part += strlen(boundary);

    while (part != NULL && *parameter_count < HTTP_MAX_PARAMETER_COUNT) 
    {
        char* name_start = strstr(part, "name=\"");
        if (name_start == NULL) 
        {
            break;
        }
        name_start += 6;
        char* name_end = strstr(name_start, "\"");
        if (name_end == NULL) 
        {
            break;
        }
        *name_end = '\0'; // Null-terminate the name
        char* next = name_end + 1;

        char* filename_start = strstr(next, "filename=\"");
        char* filename = NULL;
        if (filename_start != NULL) 
        {
            filename_start += 10;
            char* filename_end = strstr(filename_start, "\"");
            if (filename_end != NULL) 
            {
                *filename_end = '\0'; // Null-terminate the filename
                filename = filename_start;
                next = filename_end + 1;
            }
        }

        part = strstr(next, "\r\n\r\n");
        if (part == NULL) 
        {
            break;
        }
        part += 4;

        char* data_end = strstr(part, boundary);
        if (data_end == NULL) 
        {
            break;
        }
        /* Calculate data length excluding the trailing CRLF */
        size_t data_length = data_end - part - 4;

        parameters[*parameter_count].name = name_start;
        parameters[*parameter_count].filename = filename;
        parameters[*parameter_count].data = (void*)part;
        parameters[*parameter_count].data_count = data_length;
        (*parameter_count)++;

        part = strstr(part + data_length, boundary); /* Move to the next part */
    }

    return true;
}


/*****************************************************************************
* Function name: send_response_header
*****************************************************************************
* Summary:
* This function constructs and sends an HTTP response header with the provided 
* status, content type, and content length over the given TCP session. 
* The header is formatted according to the HTTP/1.1 specification.
*
* Parameters:
*  session        Pointer to the TCP session.
*  status         HTTP status string (e.g., "200 OK").
*  content_type   Content type string (e.g., "text/html").
*  content_length Length of the content in bytes.
*
* Return:
*  true if the header is sent successfully, otherwise false.
*
*****************************************************************************/
static bool send_response_header(tcp_session_t* session, const char* status, 
    const char* content_type, long content_length) 
{
    char header[512];
    int header_length = snprintf(header, sizeof(header),
        "HTTP/1.1 %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %ld\r\n"
        "\r\n",
        status, content_type, content_length);

    if (header_length < 0 || header_length >= sizeof(header)) 
    {
        return false; // snprintf error or header truncated
    }

    return session_send(session, header, header_length);
}

/*****************************************************************************
* Function name: get_mime_type
*****************************************************************************
* Summary:
* This function determines the MIME type based on the file extension in the 
* provided path.
* If the file extension is not recognized, it returns a default MIME type 
* indicating binary data.
*
* Parameters:
*  path Pointer to the file path string.
* Return:
*  A string representing the MIME type.
*
*****************************************************************************/
static const char* get_mime_type(const char* path) 
{
    const char* ext = strrchr(path, '.');
    if (ext != NULL) 
    {
        if (strcmp(ext, ".html") == 0 || strcmp(ext, ".htm") == 0) 
        {
            return MIME_HTML;
        }
        else if (strcmp(ext, ".txt") == 0) 
        {
            return MIME_PLAIN;
        }
        else if (strcmp(ext, ".xml") == 0) 
        {
            return MIME_XML;
        }
        else if (strcmp(ext, ".css") == 0) 
        {
            return MIME_CSS;
        }
        else if (strcmp(ext, ".json") == 0) 
        {
            return MIME_JSON;
        }
        else if (strcmp(ext, ".js") == 0) 
        {
            return MIME_JS;
        }
        else if (strcmp(ext, ".png") == 0) 
        {
            return MIME_PNG;
        }
        else if (strcmp(ext, ".gif") == 0) 
        {
            return MIME_GIF;
        }
        else if (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".jpeg") == 0) 
        {
            return MIME_JPEG;
        }
        else if (strcmp(ext, ".ico") == 0) 
        {
            return MIME_ICO;
        }
        else if (strcmp(ext, ".pdf") == 0) 
        {
            return MIME_PDF;
        }
        else if (strcmp(ext, ".zip") == 0) 
        {
            return MIME_ZIP;
        }
        else if (strcmp(ext, ".wav") == 0) 
        {
            return MIME_WAV;
        }
        else if (strcmp(ext, ".csv") == 0) 
        {
            return MIME_CSV;
        }
        else if (strcmp(ext, ".npy") == 0) 
        {
            return MIME_BINARY;
        }
        else if (strcmp(ext, ".tsp") == 0) 
        {
            return MIME_BINARY;
        }
    }
    return MIME_BINARY;
}

/*****************************************************************************
* Function name: check_authorization
*****************************************************************************
* Summary:
* This function checks the HTTP Basic Authorization header for valid credentials.
* It decodes the provided HTTP Basic Authorization header, extracts the username
* and password, and validates the password using a provided validation function.
*
* Parameters:
*  session Pointer to the TCP session.
*  auth_header Pointer to the HTTP Basic Authorization header.
* Return:
*  true if the authorization is valid, otherwise false.
*
*****************************************************************************/
static bool check_authorization(tcp_session_t* session, const char* auth_header) 
{
    if (auth_header == NULL || strlen(auth_header) == 0) 
    {
        return false;
    }

    http_client_t* client = (http_client_t*)session->client_context;
    if (!client) 
    {
        return false;
    }

    /* Check rate limiting before processing */
    if (is_rate_limited(client)) 
    {
        log_message(LOG_LEVEL_ERROR, "httpd", "Client rate limited due to too many auth attempts");
        return false;
    }

    char decoded_auth[256];
    memset(decoded_auth, 0, sizeof(decoded_auth));
    
    /* Decode the base64 authorization header */
    base64_decode(auth_header, decoded_auth);
    
    /* Validate that we got some decoded data */
    if (strlen(decoded_auth) == 0) 
    {
        log_message(LOG_LEVEL_ERROR, "httpd", "Failed to decode authorization header or empty result");
        client->auth_attempts++;
        client->last_auth_attempt = time(NULL);
        return false;
    }

    char username[128];
    char password[128];
    memset(username, 0, sizeof(username));
    memset(password, 0, sizeof(password));
    
    /* 1. If the password field is empty it won't be scanned.
       2. If the username is empty, password should too be empty.
       We can't thus test for the scanning success here. */
    sscanf(decoded_auth, "%127[^:]:%127s", username, password);
    bool valid = http_is_password_valid(password);
    
    if (valid) 
    {
        /* Reset failed attempts on successful authentication */
        client->auth_attempts = 0;
        log_message(LOG_LEVEL_DEBUG, "httpd", "Authentication successful for user: %s", username);
    } 
    else 
    {
        client->auth_attempts++;
        client->last_auth_attempt = time(NULL);
        log_message(LOG_LEVEL_ERROR, "httpd", "Authentication failed for user: %s (attempt %u)", 
                   username, client->auth_attempts);
    }

    /* Clear sensitive data */
    memset(decoded_auth, 0, sizeof(decoded_auth));
    memset(password, 0, sizeof(password));
    
    return valid;
}

/*****************************************************************************
* Function name: is_rate_limited
*****************************************************************************
* Summary:
*  Checks if the client is rate-limited based on authentication attempts.
*  This function checks if the client has exceeded the maximum allowed 
*  authentication attempts within the lockout period.
*
* Parameters:
*  client Pointer to the HTTP client context.
*
* Return:
*  true if the client is rate-limited, otherwise false.
*
*****************************************************************************/
static bool is_rate_limited(http_client_t* client) 
{
    if (!client) 
    {
        return true;
    }
    
    time_t current_time = time(NULL);
    
    /* Reset attempts if lockout period has passed */
    if (client->last_auth_attempt > 0 && 
        (current_time - client->last_auth_attempt) > AUTH_LOCKOUT_SECONDS) 
    {
        client->auth_attempts = 0;
    }
    
    return client->auth_attempts >= MAX_AUTH_ATTEMPTS;
}

/*****************************************************************************
* Function name: http_init
*****************************************************************************
* Summary:
*  Initializes the HTTP context.
*  This function sets up the HTTP context by clearing its memory and preparing 
*  it for use.
*
* Parameters:
*  http Pointer to the HTTP context.
*
* Return:
*  true if the initialization was successful, otherwise false.
*
*****************************************************************************/
bool http_init(http_t* http) 
{
    memset(http, 0, sizeof(http_t));    
    return true;
}

/*****************************************************************************
* Function name: http_start
*****************************************************************************
* Summary:
*  Starts the HTTP server.
*  This function sets up the HTTP server by allocating necessary resources 
*  and preparing it for use.
*
* Parameters:
*  http Pointer to the HTTP context.
*  address IP address to bind the server to.
*  port Port number to bind the server to.
*
* Return:
*  true if the server was started successfully, otherwise false.
*
*****************************************************************************/
bool http_start(http_t* http, cy_nw_ip_address_t address, int port) 
{


    if (http->tcp_server != NULL) 
    {
        log_message(LOG_LEVEL_ERROR, "httpd", "HTTP server already started.");
        return false;
    }

    if (http->request_buffer != NULL) 
    {
        log_message(LOG_LEVEL_ERROR, "httpd", "HTTP server start failed. Expected request_buffer to be NULL.");
        return false;
    }

    http->request_buffer = net_malloc(HTTP_MAX_REQUEST_SIZE);
    if (http->request_buffer == NULL) 
    {
        log_message(LOG_LEVEL_ERROR, "httpd", "HTTP server start failed. Allocation of request_buffer failed.");
        return false;
    }

    http->tcp_server = network_create_tcp_server(address, port, 10, session_opened, session_receive, session_closed, http);

    if (http->tcp_server == NULL) 
    {
        log_message(LOG_LEVEL_ERROR, "httpd", "Failed to create TCP server");
        return false;
    }

    char ip_addr_str[16];
    cy_nw_ntoa(&address, ip_addr_str);
    log_message(LOG_LEVEL_INFO, "httpd", "HTTP service started on %s:%d", ip_addr_str, port);
    return true;
}

/*****************************************************************************
* Function name: http_stop
*****************************************************************************
* Summary:
*  Stops the HTTP server.
*  This function releases the resources allocated for the HTTP server and stops it.
*
* Parameters:
*  http Pointer to the HTTP context.
*
* Return:
*  true if the server was stopped successfully, otherwise false.
*
*****************************************************************************/
bool http_stop(http_t* http) 
{
    if (http->tcp_server == NULL) 
    {
        log_message(LOG_LEVEL_ERROR, "httpd", "HTTP server not started.");
        return false;
    }

    network_destroy_tcp_server(http->tcp_server);
    http->tcp_server = NULL;

    if (http->request_buffer != NULL)
    {
        net_free(http->request_buffer);
    }
    return true;
}

/*****************************************************************************
* Function name: http_send_response
*****************************************************************************
* Summary:
*  Sends an HTTP response with the specified status, content type, and body.
*
* Parameters:
*  session Pointer to the TCP session.
*  status HTTP status code.
*  content_type MIME type of the response.
*  body Response body.
*
* Return:
*  true if the response was sent successfully, otherwise false.
*
*****************************************************************************/
bool http_send_response(tcp_session_t* session, const char* status, const char* content_type, const char* body) 
{
    size_t len = strlen(body);

    if (!send_response_header(session, status, content_type, len)) 
    {
        return false;
    }

    if (!session_send(session, body, len)) 
    {
        return false;
    }

    return true;
}

/*****************************************************************************
* Function name: http_send_file
*****************************************************************************
* Summary:
*  Sends a file as an HTTP response.
*
* Parameters:
*  session Pointer to the TCP session.
*  status HTTP status code.
*  path Path to the file to be sent.
*
* Return:
*  true if the file was sent successfully, otherwise false.
*
*****************************************************************************/
bool http_send_file(tcp_session_t* session, const char* status, const char* path) 
{
    if (!session || !status || !path) 
    {
        log_message(LOG_LEVEL_ERROR, "httpd", "Invalid parameters for http_send_file");
        return false;
    }

    /* Basic path validation to prevent directory traversal */
    if (strstr(path, "..") != NULL) 
    {
        log_message(LOG_LEVEL_ERROR, "httpd", "Directory traversal attempt blocked: %s", path);
        http_send_status(session, 403);
        return false;
    }

    FILE* file = fopen(path, "rb");
    if (!file) 
    {
        log_message(LOG_LEVEL_DEBUG, "httpd", "File not found: %s", path);
        http_send_status(session, 404);
        return false;
    }

    /* Get file size */
    if (fseek(file, 0, SEEK_END) != 0) 
    {
        log_message(LOG_LEVEL_ERROR, "httpd", "Failed to seek to end of file: %s", path);
        fclose(file);
        http_send_status(session, 500);
        return false;
    }

    long file_size = ftell(file);
    if (file_size < 0) 
    {
        log_message(LOG_LEVEL_ERROR, "httpd", "Failed to get file size: %s", path);
        fclose(file);
        http_send_status(session, 500);
        return false;
    }

    if (fseek(file, 0, SEEK_SET) != 0) 
    {
        log_message(LOG_LEVEL_ERROR, "httpd", "Failed to seek to beginning of file: %s", path);
        fclose(file);
        http_send_status(session, 500);
        return false;
    }

    const char* mime_type = get_mime_type(path);

    if (!send_response_header(session, status, mime_type, file_size)) 
    {
        log_message(LOG_LEVEL_ERROR, "httpd", "Failed to send response header for file: %s", path);
        fclose(file);
        return false;
    }

    /* Send file in chunks with proper error handling */
    char buffer[1024];
    size_t bytes_read;
    uint32_t bytes_sent;
    size_t total_sent = 0;

    while ((bytes_read = fread(buffer, 1, sizeof(buffer), file)) > 0) 
    {
        cy_rslt_t result = cy_socket_send(session->client_socket, buffer, bytes_read, CY_SOCKET_FLAGS_NONE, &bytes_sent);
        if (result != CY_RSLT_SUCCESS) 
        {
            log_message(LOG_LEVEL_ERROR, "httpd", "Failed to send file data: %s", path);
            fclose(file);
            return false;
        }
        
        if (bytes_sent != bytes_read) 
        {
            log_message(LOG_LEVEL_ERROR, "httpd", "Partial send for file %s: sent %u of %zu bytes", 
                       path, bytes_sent, bytes_read);
        }
        
        total_sent += bytes_sent;
    }

    if (ferror(file)) 
    {
        log_message(LOG_LEVEL_ERROR, "httpd", "Error reading file: %s", path);
        fclose(file);
        return false;
    }

    fclose(file);

    log_message(LOG_LEVEL_DEBUG, "httpd", "Successfully sent file %s (%zu bytes)", path, total_sent);
    return true;
}

/*****************************************************************************
* Function name: http_send_status
*****************************************************************************
* Summary:
*  Sends an HTTP status response.
*
* Parameters:
*  session Pointer to the TCP session.
*  code HTTP status code.
*
* Return:
*  None
*
*****************************************************************************/
void http_send_status(tcp_session_t* session, int code) 
{
    switch (code) 
    {
        case 200:
        {
            http_send_response(session, "200 OK", MIME_PLAIN, "OK");
            break;
        }

        case 201:
        {
            http_send_response(session, "201 Created", MIME_PLAIN, "Created");
            break;
        }

        case 202:
        {
            http_send_response(session, "202 Accepted", MIME_PLAIN, "Accepted");
            break;
        }

        case 204:
        {
            http_send_response(session, "204 No Content", MIME_PLAIN, "No Content");
            break;
        }

        case 301:
        {
            http_send_response(session, "301 Moved Permanently", MIME_PLAIN, "Moved Permanently");
            break;
        }

        case 302:
        {
            http_send_response(session, "302 Found", MIME_PLAIN, "Found");
            break;
        }

        case 304:
        {
            http_send_response(session, "304 Not Modified", MIME_PLAIN, "Not Modified");
            break;
        }

        case 400:
        {
            http_send_response(session, "400 Bad Request", MIME_PLAIN, "Bad Request");
            break;
        }

        case 401:
        {
            const char* response = "HTTP/1.1 401 Unauthorized\r\n"
                "WWW-Authenticate: Basic realm=\"User Visible Realm\"\r\n"
                "Content-Type: text/plain\r\n"
                "Content-Length: 12\r\n"
                "\r\n"
                "Unauthorized";
            session_send(session, response, strlen(response));
            break;
        }

        case 403:
        {
            http_send_response(session, "403 Forbidden", MIME_PLAIN, "Forbidden");
            break;
        }

        case 404:
        {
            /* http_send_file(session, "404 Not Found", "/system/web/404.html"); */
            http_send_response(session, "404 Not Found", MIME_PLAIN, "Not Found");
            break;
        }

        case 405:
        {
            http_send_response(session, "405 Method Not Allowed", MIME_PLAIN, "Method Not Allowed");
            break;
        }

        case 500:
        {
            /* http_send_file(session, "500 Internal Server Error", "/system/web/500.html"); */
            http_send_response(session, "500 Internal Server Error", MIME_PLAIN, "Internal Server Error");
            break;
        }

        case 501:
        {
            http_send_response(session, "501 Not Implemented", MIME_PLAIN, "Not Implemented");
            break;
        }

        case 502:
        {
            http_send_response(session, "502 Bad Gateway", MIME_PLAIN, "Bad Gateway");
            break;
        }

        case 503:
        {
            http_send_response(session, "503 Service Unavailable", MIME_PLAIN, "Service Unavailable");
            break;
        }

        default:
        {
            /* http_send_file(session, "500 Internal Server Error", "/system/web/500.html"); */
            http_send_response(session, "500 Internal Server Error", MIME_PLAIN, "Internal Server Error");
            break;
        }

    }
}

/*****************************************************************************
* Function name: http_start_chunked_response
*****************************************************************************
* Summary:
*  Starts an HTTP chunked response.
*
* Parameters:
*  session Pointer to the TCP session.
*  status HTTP status code.
*  content_type Content type of the response.
*
* Return:
*  true if the chunked response header was sent successfully, otherwise false.
*
*****************************************************************************/
bool http_start_chunked_response(tcp_session_t* session, const char* status, const char* content_type) 
{
    char headers[512];
    snprintf(headers, sizeof(headers), "HTTP/1.1 %s\r\nTransfer-Encoding: chunked\r\nContent-Type: %s\r\n\r\n", status, content_type);
    return session_send(session, headers, strlen(headers));
}

/*****************************************************************************
* Function name: http_send_chunk
*****************************************************************************
* Summary:
*  Sends an HTTP chunk.
*
* Parameters:
*  session Pointer to the TCP session.
*  data Pointer to the chunk data.
*  size Size of the chunk data.
*
* Return:
*  true if the chunk was sent successfully, otherwise false.
*
*****************************************************************************/
bool http_send_chunk(tcp_session_t* session, const char* data, size_t size) 
{
    char chunk_header[16];
    snprintf(chunk_header, sizeof(chunk_header), "%x\r\n", size); /* Size in hexadecimal */ 
    if (!session_send(session, chunk_header, strlen(chunk_header))) 
    {
        return false;
    }
    if (!session_send(session, data, size)) 
    {
        return false;
    }
    /* End of chunk */
    return session_send(session, "\r\n", 2); 
}

/*****************************************************************************
* Function name: http_end_chunked_response
*****************************************************************************
* Summary:
*  Ends an HTTP chunked response.
*
* Parameters:
*  session Pointer to the TCP session.
*
* Return:
*  true if the final chunk was sent successfully, otherwise false.
*
*****************************************************************************/
bool http_end_chunked_response(tcp_session_t* session) 
{
    /* Final zero-length chunk */
    return session_send(session, "0\r\n\r\n", 5); 
}

#endif /* IM_ENABLE_HTTP */

/* [] END OF FILE */

