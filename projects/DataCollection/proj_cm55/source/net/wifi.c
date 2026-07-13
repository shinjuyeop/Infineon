/******************************************************************************
* File Name:   wifi.c
*
* Description: Implementation of Wi-Fi management functions.
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
#include <inttypes.h>
#include <cybsp.h>
#include "retarget_io_init.h"
#include <cyabs_rtos.h>
#include <cy_secure_sockets.h>

/* Wi-Fi connection manager header files */
#include <cy_wcm.h>
#include <cy_wcm_error.h>

/* IP address related header files. */
#include <cy_nw_helper.h>

#include "../heap.h"
#include "../common.h"
#include "wifi.h"

/*******************************************************************************
* Macros
*******************************************************************************/
#define WIFI_INTERFACE_TYPE                          (CY_WCM_INTERFACE_TYPE_STA)
#define APP_SDIO_INTERRUPT_PRIORITY                  (7U)
#define APP_HOST_WAKE_INTERRUPT_PRIORITY             (2U)
#define APP_SDIO_FREQUENCY_HZ                        (25000000U)
#define SDHC_SDIO_64BYTES_BLOCK                      (64U)

/******************************************************************************
 * Global Variables
 *****************************************************************************/
static wifi_t* _wifi; /* Needed for wifi_event_callback */
static mtb_hal_sdio_t sdio_instance;
cy_stc_sd_host_context_t sdhc_host_context;
static cy_wcm_config_t wcm_config;

/******************************************************************************
 * Types
 *****************************************************************************/
#if (CY_CFG_PWR_SYS_IDLE_MODE == CY_CFG_PWR_MODE_DEEPSLEEP)

/* SysPm callback parameter structure for SDHC */
static cy_stc_syspm_callback_params_t sdcardDSParams =
{
    .context   = &sdhc_host_context,
    .base      = CYBSP_WIFI_SDIO_HW
};

/* SysPm callback structure for SDHC*/
static cy_stc_syspm_callback_t sdhcDeepSleepCallbackHandler =
{
    .callback           = Cy_SD_Host_DeepSleepCallback,
    .skipMode           = SYSPM_SKIP_MODE,
    .type               = CY_SYSPM_DEEPSLEEP,
    .callbackParams     = &sdcardDSParams,
    .prevItm            = NULL,
    .nextItm            = NULL,
    .order              = SYSPM_CALLBACK_ORDER
};
#endif

/*******************************************************************************
* Function Name: sdio_interrupt_handler
********************************************************************************
* Summary:
* Interrupt handler function for SDIO instance.
* Parameters:
*  None
*
* Return:
*   None
*
*******************************************************************************/
static void sdio_interrupt_handler(void)
{
    mtb_hal_sdio_process_interrupt(&sdio_instance);
}

/*******************************************************************************
* Function Name: host_wake_interrupt_handler
********************************************************************************
* Summary:
* Interrupt handler function for the host wake up input pin.
* Parameters:
*  None
*
* Return:
*   None
*
*******************************************************************************/
static void host_wake_interrupt_handler(void)
{
    mtb_hal_gpio_process_interrupt(&wcm_config.wifi_host_wake_pin);
}

/*******************************************************************************
* Function Name: app_sdio_init
********************************************************************************
* Summary:
* This function configures and initializes the SDIO instance used in 
* communication between the host MCU and the wireless device.
* Parameters:
*  None
*
* Return:
*   None
*
*******************************************************************************/
static void app_sdio_init(void)
{
    cy_rslt_t result;
    mtb_hal_sdio_cfg_t sdio_hal_cfg;
    
    cy_stc_sysint_t sdio_intr_cfg =
    {
        .intrSrc = CYBSP_WIFI_SDIO_IRQ,
        .intrPriority = APP_SDIO_INTERRUPT_PRIORITY
    };

    cy_stc_sysint_t host_wake_intr_cfg =
    {
            .intrSrc = CYBSP_WIFI_HOST_WAKE_IRQ,
            .intrPriority = APP_HOST_WAKE_INTERRUPT_PRIORITY
    };

    /* Initialize the SDIO interrupt and specify the interrupt handler. */
    cy_en_sysint_status_t interrupt_init_status = Cy_SysInt_Init(&sdio_intr_cfg, sdio_interrupt_handler);

    /* SDIO interrupt initialization failed. Stop program execution. */
    if(CY_SYSINT_SUCCESS != interrupt_init_status)
    {
        handle_app_error();
    }

    /* Enable NVIC interrupt. */
    NVIC_EnableIRQ(CYBSP_WIFI_SDIO_IRQ);

    /* Setup SDIO using the HAL object and desired configuration */
    result = mtb_hal_sdio_setup(&sdio_instance, &CYBSP_WIFI_SDIO_sdio_hal_config, NULL, &sdhc_host_context);

    /* SDIO setup failed. Stop program execution. */
    if(CY_RSLT_SUCCESS != result)
    {
        handle_app_error();
    }

    /* Initialize and Enable SD HOST */
    Cy_SD_Host_Enable(CYBSP_WIFI_SDIO_HW);
    Cy_SD_Host_Init(CYBSP_WIFI_SDIO_HW, CYBSP_WIFI_SDIO_sdio_hal_config.host_config, &sdhc_host_context);
    Cy_SD_Host_SetHostBusWidth(CYBSP_WIFI_SDIO_HW, CY_SD_HOST_BUS_WIDTH_4_BIT);

    sdio_hal_cfg.frequencyhal_hz = APP_SDIO_FREQUENCY_HZ;
    sdio_hal_cfg.block_size = SDHC_SDIO_64BYTES_BLOCK;

    /* Configure SDIO */
    mtb_hal_sdio_configure(&sdio_instance, &sdio_hal_cfg);

#if (CY_CFG_PWR_SYS_IDLE_MODE == CY_CFG_PWR_MODE_DEEPSLEEP)
    /* SDHC SysPm callback registration */
    Cy_SysPm_RegisterCallback(&sdhcDeepSleepCallbackHandler);
#endif /* (CY_CFG_PWR_SYS_IDLE_MODE == CY_CFG_PWR_MODE_DEEPSLEEP) */

    /* Setup GPIO using the HAL object for WIFI WL REG ON  */
    mtb_hal_gpio_setup(&wcm_config.wifi_wl_pin, CYBSP_WIFI_WL_REG_ON_PORT_NUM, CYBSP_WIFI_WL_REG_ON_PIN);

    /* Setup GPIO using the HAL object for WIFI HOST WAKE PIN  */
    mtb_hal_gpio_setup(&wcm_config.wifi_host_wake_pin, CYBSP_WIFI_HOST_WAKE_PORT_NUM, CYBSP_WIFI_HOST_WAKE_PIN);

    /* Initialize the Host wakeup interrupt and specify the interrupt handler. */
    cy_en_sysint_status_t interrupt_init_status_host_wake =  Cy_SysInt_Init(&host_wake_intr_cfg, host_wake_interrupt_handler);

    /* Host wake up interrupt initialization failed. Stop program execution. */
    if(CY_SYSINT_SUCCESS != interrupt_init_status_host_wake)
    {
        handle_app_error();
    }

    /* Enable NVIC interrupt. */
    NVIC_EnableIRQ(CYBSP_WIFI_HOST_WAKE_IRQ);
}

/*******************************************************************************
* Function Name: wifi_start
********************************************************************************
* Summary:
* This function connects to a Wi-Fi network using the provided Wi-Fi configuration.
* Parameters:
*  wifi - Pointer to the Wi-Fi configuration structure.
*
* Return:
*   true if the connection is successful, false otherwise.
*
*******************************************************************************/
bool wifi_start(wifi_t *wifi)
{
    cy_rslt_t result = CY_RSLT_SUCCESS;
    cy_wcm_connect_params_t wifi_conn_param;

    if(strlen(wifi->ssid) == 0) 
    {
         log_message(LOG_LEVEL_ERROR, "wifi", "Unable to connect. No SSID set.");
         return false;
    }

    if(strlen(wifi->password) == 0) 
    {
         log_message(LOG_LEVEL_ERROR, "wifi", "Unable to connect. No password set.");
         return false;
    }

    if (strlen(wifi->ssid) > CY_WCM_MAX_SSID_LEN) 
    {
        log_message(LOG_LEVEL_ERROR, "wifi", "SSID exceeds maximum length");
        return false;
    }

    if (strlen(wifi->password) > CY_WCM_MAX_PASSPHRASE_LEN) 
    {
        log_message(LOG_LEVEL_ERROR, "wifi", "Password exceeds maximum length");
        return false;
    }

    /* Set the Wi-Fi SSID, password and security type. */
    memset(&wifi_conn_param, 0, sizeof(cy_wcm_connect_params_t));
    strncpy((char *)wifi_conn_param.ap_credentials.SSID, wifi->ssid, WIFI_MAX_SSID_LEN);
    strncpy((char *)wifi_conn_param.ap_credentials.password, wifi->password, WIFI_MAX_PASSWORD_LEN);
    wifi_conn_param.ap_credentials.security = wifi->security;

    log_message(LOG_LEVEL_INFO, "wifi", "Connecting to Wi-Fi Network: '%s'", wifi_conn_param.ap_credentials.SSID);

    cy_wcm_ip_address_t wcm_ip_address;
    wcm_ip_address.version = CY_WCM_IP_VER_V4;

    result = cy_wcm_connect_ap(&wifi_conn_param, &wcm_ip_address);
    if(result != CY_RSLT_SUCCESS) 
    {
        log_message(LOG_LEVEL_ERROR, "wifi", "Wi-Fi Network connect failed");
        return false;
    }

    wifi->ip_addr.version = NW_IP_IPV4;
    wifi->ip_addr.ip.v4 = wcm_ip_address.ip.v4;

    char ip_addr_str[16];
    cy_nw_ntoa(&wifi->ip_addr, ip_addr_str);
    log_message(LOG_LEVEL_INFO, "wifi", "Successfully connected to network '%s'. IP Address Assigned: %s", wifi->ssid, ip_addr_str);

    if(wifi->connected != NULL)
    {
        wifi->connected(wifi);
    }

    return true;
}

/*******************************************************************************
* Function Name: wifi_stop
********************************************************************************
* Summary:
* This function disconnects from the currently connected Wi-Fi network.
* Parameters:
*  wifi - Pointer to the Wi-Fi configuration structure.
*
* Return:
*   true if the disconnection is successful, false otherwise.
*
*******************************************************************************/
bool wifi_stop(wifi_t *wifi)
{
    if(cy_wcm_is_connected_to_ap()) 
    {
        log_message(LOG_LEVEL_INFO, "wifi", "Disconnecting...");
        cy_wcm_disconnect_ap();

        if(_wifi->disconnected != NULL)
        {
            _wifi->disconnected(_wifi);
        }
        return true;
    } 
    else 
    {
        log_message(LOG_LEVEL_ERROR, "wifi", "Not connected");
        return false;
    }
}

/*******************************************************************************
* Function Name: wifi_event_callback
********************************************************************************
* Summary:
* This function handles Wi-Fi events such as disconnection.
* Parameters:
*  event - Wi-Fi event type.
*  event_data - Data associated with the event.
*
* Return:
*   None.
*
*******************************************************************************/
static void wifi_event_callback(cy_wcm_event_t event, cy_wcm_event_data_t *event_data)
{
    _wifi->status = event;

    switch(event) 
    {
        case CY_WCM_EVENT_CONNECTING:
        {
            /* STA connecting to an AP. */
            log_message(LOG_LEVEL_DEBUG, "wifi", "CY_WCM_EVENT_CONNECTING");
            break;
        }
        case CY_WCM_EVENT_CONNECTED:
        {
            /* STA connected to the AP. */
            log_message(LOG_LEVEL_DEBUG, "wifi", "CY_WCM_EVENT_CONNECTED");
            break;
        }
        case CY_WCM_EVENT_CONNECT_FAILED:
        {
            /* STA connection to the AP failed. */
            log_message(LOG_LEVEL_DEBUG, "wifi", "CY_WCM_EVENT_CONNECT_FAILED");
            break;
        }
        case CY_WCM_EVENT_RECONNECTED:
        {
            /* STA reconnected to the AP. */
            log_message(LOG_LEVEL_DEBUG, "wifi", "CY_WCM_EVENT_RECONNECTED");
            break;
        }
        case CY_WCM_EVENT_DISCONNECTED:
        {
            /* STA disconnected from the AP. */
            log_message(LOG_LEVEL_DEBUG, "wifi", "CY_WCM_EVENT_DISCONNECTED");
            break;
        }
        case CY_WCM_EVENT_IP_CHANGED:
        {
            /* IP address change event. Notified after connection, re-connection, and IP address change due to DHCP renewal. */
            log_message(LOG_LEVEL_DEBUG, "wifi", "CY_WCM_EVENT_IP_CHANGED");
            break;
        }
        case CY_WCM_EVENT_INITIATED_RETRY:
        {
            /* WCM will initiate a retry logic to re-connect to the AP. */
            log_message(LOG_LEVEL_DEBUG, "wifi", "CY_WCM_EVENT_INITIATED_RETRY");
            break;
        }
        case CY_WCM_EVENT_STA_JOINED_SOFTAP:
        {
            /* An STA device connected to SoftAP. */
            log_message(LOG_LEVEL_DEBUG, "wifi", "CY_WCM_EVENT_STA_JOINED_SOFTAP");
            break;
        }
        case CY_WCM_EVENT_STA_LEFT_SOFTAP:
        {
            /* An STA device disconnected from SoftAP. */
            log_message(LOG_LEVEL_DEBUG, "wifi", "CY_WCM_EVENT_STA_LEFT_SOFTAP");
            break;
        }
    }

    if(event == CY_WCM_EVENT_DISCONNECTED) 
    {
        log_message(LOG_LEVEL_INFO, "wifi", "Wi-Fi Disconnected. Begin network shutdown.");
        if(_wifi->disconnected != NULL)
        {
            _wifi->disconnected(_wifi);
        }

    }
}

/*******************************************************************************
* Function Name: wifi_init
********************************************************************************
* Summary:
* This function initializes the Wi-Fi module and registers event callbacks.
* Parameters:
*  wifi - Pointer to the Wi-Fi structure.
*  connected - Callback function for Wi-Fi connection event.
*  disconnected - Callback function for Wi-Fi disconnection event.
*
* Return:
*   true if initialization is successful, false otherwise.
*
*******************************************************************************/
bool wifi_init(wifi_t *wifi, wifi_fn connected, wifi_fn disconnected) 
{

    _wifi = wifi;

    /* Initialize wcm */
    wcm_config.interface = CY_WCM_INTERFACE_TYPE_STA;
    wcm_config.wifi_interface_instance = &sdio_instance;

    memset(wifi, 0, sizeof(wifi_t));
    wifi->connected = connected;
    wifi->disconnected = disconnected;
    
    app_sdio_init();
    /* Initialize Wi-Fi connection manager. */
    cy_rslt_t result = cy_wcm_init(&wcm_config);

    if (result != CY_RSLT_SUCCESS)
    {
        log_message(LOG_LEVEL_ERROR, "wifi", "Wi-Fi Connection Manager initialization failed");
        return false;
    }
    log_message(LOG_LEVEL_INFO, "wifi", "Wi-Fi Connection Manager initialized");

    /* Register Wi-Fi event callback. */
    cy_wcm_register_event_callback(wifi_event_callback);

    /* Initialize secure socket library. */
    result = cy_socket_init();
    if (result != CY_RSLT_SUCCESS)
    {
        log_message(LOG_LEVEL_ERROR, "wifi", "Secure Socket initialization failed");
        return false;
    }
    log_message(LOG_LEVEL_INFO, "wifi", "Secure Socket initialized");

    return true;
}

#endif /* IM_ENABLE_NET */

/* [] END OF FILE */
