/******************************************************************************
* File Name: process.h
*
* Description: Interface for process command and environment management.
*              
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

#ifndef _PROCESS_H_
#define _PROCESS_H_

#ifdef IM_ENABLE_SHELL

#include <stdarg.h>
#include <FreeRTOS.h>

#include "terminal.h"
#include "../../wifi_configs/im_config.h"
#include "../../heap.h"

/*******************************************************************************
* Types
*******************************************************************************/
 /* Enum to specify the execution mode of a command. */
typedef enum {
    EXEC_MODE_BLOCK = 1,       /**< Execute in the current thread, blocking it until the command finishes. */
    EXEC_MODE_BACKGROUND = 2,  /**< Spawn a new thread to run the command in the background. Returns immediately. */
    EXEC_MODE_BLOCK_ABORT = 3  /**< Spawn a new thread to run the command. Waits for it to finish or aborts on Ctrl-C. */
} exec_mode_t;

typedef enum {
    PROCESS_NONE = 0,
    PROCESS_TERMINAL_OWNER = 1,     /**< The terminal object will be freed when the process terminates */
    PROCESS_NOEXIT = 2,             /**< Process handles its own shutdown, not killed by Ctrl-C */
    PROCESS_TRACK_RESOURCES = 4     /**< The process tracks resources (memory, child processes etc.) that are released when the process terminates. */
} process_attrib_t;

typedef struct process_s process_t;

typedef enum {
    PROCESS_RESOURCE_NONE = 0,
    PROCESS_RESOURCE_CHILD_PROCESS = 1,
    PROCESS_RESOURCE_ALLOCATED_MEMORY = 2,
} process_resouce_type_t;

typedef struct process_resouce_s process_resouce_t;

struct process_resouce_s {
    process_resouce_type_t type;
    void* resource;
    process_resouce_t* next;
};

/* Structure for managing the shell environment and commands. */
struct process_s {
    struct _reent reent;                /**< Reentrancy pointer*/
    char current_directory[IM_PATH_MAX];   /**< Current working directory */
    const char* name;                   /**< Name of the process */
    int pid;                            /**< Process ID */
    TaskHandle_t xHandle;               /**< Process handle */
    process_attrib_t attrib;            /**< Process attributes */
    terminal_t* terminal;               /**< Associated terminal */
    process_t* parent;                  /**< Parent process */
    process_resouce_t* resource_list;   /**< List of resources to be released when process terminates. See PROCESS_RESOURCES */
};

/*******************************************************************************
* Function Prototypes
*******************************************************************************/
void process_set_attrib(process_t* process, process_attrib_t flag);
int process_exec_command(char* line, exec_mode_t mode);
bool process_kill(process_t* process, int signal);
process_t* process_get(TaskHandle_t handle);
process_t* process_find(int pid);
terminal_t* process_new_terminal(process_t* process, terminal_write_fn write);
void process_sleep(int milliseconds);
void process_create_hook(TaskHandle_t handle);
void process_delete_hook(TaskHandle_t handle);
void process_switched_in_hook(void);

#endif /* IM_ENABLE_SHELL */
#endif /* _PROCESS_H_ */

/* [] END OF FILE */