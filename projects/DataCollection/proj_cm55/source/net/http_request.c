/******************************************************************************
* File Name:   http_request.c
*
* Description: Implementation of the HTTP application specific endpoints.
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
#include <stdbool.h>
#include <math.h>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>

#include "../shell/fs/statvfs.h"
#include "../heap.h"
#include "../common.h"
#include "../shell/process.h"
#include "../protocol/protocol.h"
#include "../shell/lib/tsp_utils.h"
#include "../services.h"
#include "tcp.h"
#include "http.h"
#include "../shell/fs/fs.h"

/*******************************************************************************
* Function Prototypes
*******************************************************************************/
static bool path_equals(const char* path, const char* req);

#ifdef IM_ENABLE_UPNP
/* http_request_upnp.c */ 
bool send_upnp_response(tcp_session_t* session);
#endif

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: http_handle_request
********************************************************************************
* Summary:
*    Handles an HTTP request.
*    Invoked from http.c
* Parameters:
*   session - Pointer to the TCP session.
*   method  - HTTP method (e.g., "GET", "POST").
*   path    - Requested path.
*   params  - Array of HTTP parameters.
*   params_count - Number of HTTP parameters.
*
* Return:
*   None
*
*******************************************************************************/
void http_handle_request(tcp_session_t* session, const char* method, 
    const char* path, http_parameter_t* params, int params_count) 
{

    if (!session || !method || !path) 
    {
        log_message(LOG_LEVEL_ERROR, "httpd", "Invalid parameters in http_handle_request");
        if (session != NULL)
        {
            http_send_status(session, 400);
        }
        return;
    }

    log_message(LOG_LEVEL_DEBUG, "httpd", "Http request %s %s", method, path);

    if (strcmp(method, "GET") == 0) 
    {
        if (path_equals(path, "/login")) 
        {
            /* Send a 401 Unauthorized response to force re-authentication */
            http_send_status(session, 401);
            return;
        }
        else if (path_equals(path, "/example")) 
        {
            http_send_response(session, STATUS_200, MIME_HTML, "Example text");
        }
        
#ifdef IM_ENABLE_UPNP
        else if (path_equals(path, "/UPnP.xml")) 
        {
            if (!send_upnp_response(session)) 
            {
                log_message(LOG_LEVEL_ERROR, "http", "Failed to send UPnP response");
            }
        }
#endif
    }
    else if (strcmp(method, "POST") == 0) 
    {
        log_message(LOG_LEVEL_ERROR, "httpd", "Unknown POST endpoint: %s", path);
        http_send_status(session, 404);
    }
    else 
    {
        log_message(LOG_LEVEL_ERROR, "httpd", "Unsupported HTTP method: %s", method);
        http_send_status(session, 405);
    }
}

/******************************************************************************
* Function Name: path_equals
********************************************************************************
* Summary:
*    Compares two paths for equality.
*    
* Parameters:
*   path - First path to compare.
*   req  - Second path to compare.
*
* Return:
*   true if the paths are equal, false otherwise.
*
*******************************************************************************/
static bool path_equals(const char* path, const char* req) 
{
    return strcmp(path, req) == 0;
}

#endif /* IM_ENABLE_HTTP */

/* [] END OF FILE */