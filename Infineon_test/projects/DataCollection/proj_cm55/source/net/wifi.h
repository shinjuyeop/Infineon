/******************************************************************************
* File Name:   wifi.h
*
* Description: Interface for Wi-Fi management functions.
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

#ifndef _WIFI_H_
#define _WIFI_H_

#ifdef IM_ENABLE_NET

#include <cy_wcm.h>
#include <cy_nw_helper.h>

/*******************************************************************************
* Macros
*******************************************************************************/
#define WIFI_MAX_SSID_LEN           (32)
#define WIFI_MAX_PASSWORD_LEN       (63)

/*******************************************************************************
* Types
*******************************************************************************/

/* Structure representing a Wi-Fi instance. */
typedef struct wifi_s wifi_t;

/* Function pointer type for Wi-Fi events. */
typedef void (*wifi_fn)(wifi_t* wifi);

/* Structure for managing Wi-Fi connection and events. */
struct wifi_s {
    cy_nw_ip_address_t ip_addr; /* IP address assigned to the Wi-Fi interface. */
    wifi_fn connected;          /* Callback function for Wi-Fi connected event. */
    wifi_fn disconnected;       /* Callback function for Wi-Fi disconnected event. */

    cy_wcm_event_t status;
    cy_wcm_security_t security;
    char ssid[WIFI_MAX_SSID_LEN + 1];
    char password[WIFI_MAX_PASSWORD_LEN + 1];
};

/*******************************************************************************
* Function Prototypes
********************************************************************************/
bool wifi_init(wifi_t *wifi, wifi_fn connected, wifi_fn disconnected);
bool wifi_start(wifi_t* wifi);
bool wifi_stop(wifi_t* wifi);

#endif /* IM_ENABLE_NET */
#endif /* _WIFI_H_ */

/* [] END OF FILE */