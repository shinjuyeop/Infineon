/******************************************************************************
* File Name: im_config.h
*
* Description:
*   This file contains configuration settings for the embedded system, 
*   including stack sizes, process priorities, and other system parameters.
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
#ifndef _IM_CONFIG_H_
#define _IM_CONFIG_H_

#define STACK_KB                     (1024/4)   /* 1 Kb stack words */ 

/* Indicates that reentrant versions of system calls (_r variants) are provided
* This macro tells the Newlib library to use the reentrant system calls
* that take a reentrancy structure (_reent) as an argument. */
#define REENTRANT_SYSCALLS_PROVIDED

/* Indicates that the standard, non-reentrant syscall names (e.g., open, close) are missing
* This macro tells Newlib that it should not expect the regular syscall names to be available 
* and should rely entirely on the reentrant versions (_r variants) instead. */
#define MISSING_SYSCALL_NAMES

/*  Indicates that only reentrant versions of functions should be used
* This macro ensures that the Newlib library exclusively uses reentrant functions,
* which take a reentrancy structure (_reent) as an argument, for all internal operations. */
#define _REENT_ONLY

/* Define _REENT_SMALL to use the smaller reentrancy structure
* This macro reduces the size of the reentrancy structure (_reent) to minimize memory usage,
* making it more suitable for resource-constrained embedded systems. */
#define _REENT_SMALL


/* Max number of concurrent tasks */
#define MAX_TASKS                  (32)

/* Max length of a command line */
#define MAX_CMD_LEN                (128)

/* Max number of arguments for a command */
#define MAX_CMD_ARGC               (16)

/* Max path length */
#define IM_PATH_MAX                (128)

/* Max number of open files per process*/
//#define MAX_FDS                    (4)

#define MAX_TX_LEN                 (256)
#define MAX_RX_LEN                 (256)

/* Stack sizes */
#define INITD_STACK_SIZE           (8 * STACK_KB)
#define SHELL_STACK_SIZE           (8 * STACK_KB)
#define NETD_STACK_SIZE            (8 * STACK_KB)
#define USBD_STACK_SIZE            (8 * STACK_KB)
#define COMMAND_STACK_SIZE         (16 * STACK_KB)
#define TSP_SESSION_STACK_SIZE     (8 * STACK_KB)
#define UPNPD_STACK_SIZE           (4 * STACK_KB)

/* Process priorities, lower is less prioritized, current config have max 7 */
#define INITD_PRIO                 (4)
#define SHELL_PRIO                 (3)
#define USBD_PRIO                  (4)
#define NETD_PRIO                  (6)
#define TSP_SESSION_PRIO           (5)
#define UPNPD_PRIO                 (3)

/* Set to 1 to disable may ANSI features, 
e.g colors, cursor flickering reduction etc. */
#define LESS_ANSI                  (0)

/* Set to 0 to remove password requirement on USB serial terminal. */
#define REQUIRE_PASSWORD_USB_UART  (1)

#endif /* _IM_CONFIG_H_ */

/* [] END OF FILE */
