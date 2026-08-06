/******************************************************************************
* File Name:   http_utils.c
*
* Description: Implementation of common HTTP utility functions.
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
#include <string.h>
#include <stdbool.h>
#include <errno.h>

#include "http_utils.h"
#include "http.h"
#include "../services.h"
#include "../common.h"
#include "../shell/lib/md5.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: http_get_serial_string
********************************************************************************
* Summary:
*    Retrieves the serial string from the board serial structure.
*
* Parameters:
*   bserial - Pointer to the board serial structure.
*
* Return:
*   Pointer to the serial string.
*
*******************************************************************************/
char* http_get_serial_string(protocol_BoardSerial* bserial) 
{
    static char text[38];

    uint8_t* serial = (uint8_t*)(bserial->uuid);

    snprintf(text, sizeof(text), "%02X%02X%02X%02X-%02X%02X-%02X%02X-%02X%02X-%02X%02X%02X%02X%02X%02X",
        serial[0], serial[1], serial[2], serial[3],
        serial[4], serial[5],
        serial[6], serial[7],
        serial[8], serial[9],
        serial[10], serial[11], serial[12], serial[13], serial[14], serial[15]);

    return text;
}

/******************************************************************************
* Function Name: http_get_location_uri
********************************************************************************
* Summary:
*    Retrieves the location URI of the HTTP server.
*
* Parameters:
*   None
*
* Return:
*   Pointer to the location URI string.
*
*******************************************************************************/
char* http_get_location_uri(void) 
{
    static char location[64];
    http_t* http = SYS_SERVICE_HTTP();
    
    if (!http || !http->tcp_server) 
    {
        return "http://localhost";
    }
    
    int http_port = http->tcp_server->address.port;
    char ip_str[16];
    cy_nw_ip_address_t tmp;
    tmp.ip.v4 = http->tcp_server->address.ip_address.ip.v4;
    cy_nw_ntoa(&tmp, ip_str);
    snprintf(location, sizeof(location), "http://%s:%d", ip_str, http_port);
    return location;
}

/******************************************************************************
* Function Name: http_is_password_valid
********************************************************************************
* Summary:
*    Validates the provided password against the stored password hash.
*
* Parameters:
*   password - Pointer to the password string to validate.
*
* Return:
*   true if the password is valid, false otherwise.
*
*******************************************************************************/
bool http_is_password_valid(const char* password) 
{
    if (!password) 
    {
        return false;
    }
    
    /* Check if password file exists */
    FILE* password_file = fopen("/system/password", "r");
    if (!password_file) 
    {
        /* No password file means no password required (like passwd.c does) */
        return strlen(password) == 0;
    }
    
    /* Read stored hash */
    char stored_hash[33];
    memset(stored_hash, 0, sizeof(stored_hash));
    size_t read = fread(stored_hash, 1, 32, password_file);
    fclose(password_file);
    
    if (read != 32) 
    {
        log_message(LOG_LEVEL_ERROR, "http", "Invalid password file format");
        return false;
    }
    stored_hash[32] = '\0';
    
    /* Hash the provided password using MD5 (matching passwd.c implementation) */
    /* NOTE: MD5 is used for compatibility with existing passwd.c system */
    /* In production, consider upgrading to SHA-256 or bcrypt for better security */
    char provided_hash[33];
    md5_string((char*)password, provided_hash);
    
    /* Compare hashes */
    bool valid = (strcmp(stored_hash, provided_hash) == 0);
    
    if (!valid) 
    {
        log_message(LOG_LEVEL_ERROR, "http", "Invalid password attempt");
    }
    
    return valid;
} 

#endif /* IM_ENABLE_HTTP */

/* [] END OF FILE */