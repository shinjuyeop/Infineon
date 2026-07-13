/*******************************************************************************
* File Name        : services.h
*
* Description      : This header file provides access to system services.
*
* Related Document : See README.md
*
********************************************************************************
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

#ifndef _SERVICES_H_
#define _SERVICES_H_

/*******************************************************************************
 * Macros
 ********************************************************************************/
#define ENV_SERVICE_PROTOCOL     (0)
#define ENV_SERVICE_TELNET       (1)
#define ENV_SERVICE_WIFI         (2)
#define ENV_SERVICE_USB          (3)
#define ENV_SERVICE_HTTP         (4)
#define ENV_SERVICE_TSP          (5)
#define ENV_SERVICE_UPNP         (6)
#define ENV_SERVICE_MDNS         (7)
#define ENV_SERVICE_COMMANDS     (8)
#define ENV_MAX_SERVICES         (9)

extern void* system_services[ENV_MAX_SERVICES];

/* Void pointers and macros to avoid this header file add dependencies to all modules */

/* include "protocol.h" */ 
#define SYS_SERVICE_PROTOCOL() ((protocol_t*)system_services[ENV_SERVICE_PROTOCOL])

/* include "telnet.h" */
#define SYS_SERVICE_TELNET() ((telnet_t*)system_services[ENV_SERVICE_TELNET])

/* include "wifi.h" */
#define SYS_SERVICE_WIFI() ((wifi_t*)system_services[ENV_SERVICE_WIFI])

/* include "usbd.h" */
#define SYS_SERVICE_TSP_USB() ((usbd_t*)system_services[ENV_SERVICE_TSP_USB])

/* include "http.h" */
#define SYS_SERVICE_HTTP() ((http_t*)system_services[ENV_SERVICE_HTTP])

/* include "tsp.h" */
#define SYS_SERVICE_TSP() ((tsp_t*)system_services[ENV_SERVICE_TSP])

/* include "upnp.h" */
#define SYS_SERVICE_UPNP() ((upnp_t*)system_services[ENV_SERVICE_UPNP])

/* include "mdns.h" */
#define SYS_SERVICE_MDNS() ((mdns_t*)system_services[ENV_SERVICE_MDNS])

/* include "commands.h" */
#define SYS_SERVICE_COMMANDS() ((commands_t*)system_services[ENV_SERVICE_COMMANDS])

#endif /* _SERVICES_H_ */

/* [] END OF FILE */
