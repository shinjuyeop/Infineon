/******************************************************************************
* File Name: common.h
*
* Description:
*   This file contains common utility functions used across the project.
*
* Related Document: See README.md
*
*******************************************************************************
* (c) 2025-2026, Infineon Technologies AG, or an affiliate of Infineon
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

#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdbool.h>

/*******************************************************************************
* Defines
*******************************************************************************/

#define LED_SHORT_ON_TIME 100         // Short blink duration in milliseconds
#define LED_LONG_ON_TIME 300          // Long blink duration in milliseconds
#define LED_OFF_TIME 300              // Pause duration between blinks
#define LED_CODE_SEPARATOR_TIME 3000  // Pause duration between repeating error codes in milliseconds

/*
*  Blink pattern:
*  0:  .        10: ..-      20: .--.
*  1:  -        11: -.-      21: ---.
*  2:  ..       12: .--      22: ...-
*  3:  -.       13: ---      23: -..-
*  4:  .-       14: ....     24: .-.-
*  5:  --       15: -...     25: --.-
*  6:  ...      16: .-..     26: ..--
*  7:  -..      17: --..     27: -.--
*  8:  .-.      18: ..-.     28: .---
*  9:  --.      19: -.-.     29: ----
*
*  Avoid patterns with only dots or dashes.
*  They are hard distinguish from each other.
*/
#define LED_CODE_GENERIC_ERROR          (3)
#define LED_CODE_UART_ERROR             (4)
#define LED_CODE_MEMORY_ERROR           (7)
#define LED_CODE_I2C_ERROR              (8)
#define LED_CODE_STDOUT_RETARGET_ERROR  (9)
#define LED_CODE_CLOCK_ERROR            (10)
#define LED_CODE_SPI_ERROR              (11)
#define LED_CODE_WATCHDOG_ERROR         (12)
#define LED_CODE_SENSOR_PDM_PCM_ERROR   (15) /* Microphone error */
#define LED_CODE_SENSOR_BMI270_ERROR    (16) /* Accelerometer/Gyro */
#define LED_CODE_SENSOR_BMM350_ERROR    (17) /* Magnetometer */
#define LED_CODE_SENSOR_DPS368_ERROR    (18) /* Digital pressure sensor */
#define LED_CODE_SENSOR_BGT60TRXX_ERROR (19) /* RADAR sensor */
#define LED_CODE_SENSOR_SHT4X_ERROR     (20) /* Humidity sensor*/
#define LED_CODE_SENSOR_AMIC_ERROR      (21) /* Analog microphone*/
#define LED_CODE_SENSOR_I3C_ERROR       (22) 
#define LED_CODE_I3C_ERROR              (23)
/*
 * Enable verbose printing to debug console.
 * This might have a performance impact and should normally be disabled.
 * Use this for high-frequency printing.
 */
// #define ENABLE_DEBUG_PRINT (1)

#if ENABLE_DEBUG_PRINT
    #define DEBUG_PRINT(...)  printf(__VA_ARGS__)
#else
    #define DEBUG_PRINT(...)  // Do nothing
#endif

/*******************************************************************************
* Types
*******************************************************************************/
typedef enum {
    LOG_LEVEL_NONE,  /**< No logging. */
    LOG_LEVEL_ERROR, /**< Error level for logging critical issues. */
    LOG_LEVEL_INFO,  /**< Info level for logging informational messages. */
    LOG_LEVEL_DEBUG  /**< Debug level for logging debug messages. */
} log_level_t;

/*******************************************************************************
* Function Prototypes
*******************************************************************************/
void log_message(log_level_t level, const char* component, const char* format, ...);
void log_set_level(log_level_t level);
void set_led0(bool state);
void set_led1(bool state);
void halt_error(int led_flash_code);

#endif /* __COMMON_H__ */

/* [] END OF FILE */
