/******************************************************************************
* File Name: process.c
*
* Description: Implementation of process command and environment management.
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

#ifdef IM_ENABLE_SHELL

#include <string.h>
#include <stdarg.h>
#include <ctype.h> 
#include <stdio.h>
#include <unistd.h>
#include <FreeRTOS.h>
#include <signal.h>
#include "task.h"

#include "process.h"
#include "../services.h"
#include "../common.h"
#include "commands.h"
#include "lib/strutils.h"
#include "../wifi_configs/im_config.h"

/*******************************************************************************
* Types
*******************************************************************************/

struct _reent system_reent = _REENT_INIT(reent);

typedef struct {
    char* command;
    exec_mode_t mode;
    TaskHandle_t parent;
    int status;
} background_job_t;

/******************************************************************************
* Function Name: terminal_write_log
 ********************************************************************************
* Summary:
*   Writes a log message to the terminal.
* Parameters:
*   terminal - The terminal to write the log message to.
*   buf - The buffer containing the log message.
*   size - The size of the buffer.
*
* Return:
*   The number of bytes written.
*******************************************************************************/
static size_t terminal_write_log(terminal_t* terminal, const void* buf, size_t size) 
{
    static char line[256] = { 0 };
    static int line_len = 0;

    process_t* process = process_get(NULL);

    const char* buf_ptr = (const char*)buf;

    for (int i = 0; i < size; i++) 
    {
        if (buf_ptr[i] == '\0' || buf_ptr[i] == '\n' || buf_ptr[i] == '\r') {
            line[line_len] = '\0';
            if (line_len > 0)
                log_message(LOG_LEVEL_INFO, process->name, "%s", line);
            line_len = 0;
        }
        else 
        {
            line[line_len++] = buf_ptr[i];
        }
    }
    return size;
}

/******************************************************************************
* Function Name: process_add_resource
 ********************************************************************************
* Summary:
*   Adds a resource to the process's resource list.
* Parameters:
*   process - The process to add the resource to.
*   type - The type of the resource.
*   ptr - A pointer to the resource.
*
* Return:
*   None.
*******************************************************************************/
static void process_add_resource(process_t* process, process_resouce_type_t type, void* ptr) 
{
    process_resouce_t* res = (process_resouce_t*)sys_malloc(sizeof(process_resouce_t));
    res->next = process->resource_list;
    res->type = type;
    res->resource = ptr;
}

/******************************************************************************
* Function Name: process_remove_resource
 ********************************************************************************
* Summary:
*   Removes a resource from the process's resource list.
* Parameters:
*   process - The process to remove the resource from.
*   ptr - A pointer to the resource.
*
* Return:
*   True if the resource was removed, false otherwise.
*******************************************************************************/
static bool process_remove_resource(process_t* process, void* ptr) 
{
    process_resouce_t* prev = NULL;
    process_resouce_t* current = process->resource_list;
    while (current != NULL) 
    {
        if (current->resource == ptr) 
        {
            if (prev == NULL)
            {
                process->resource_list = current->next;
            }
            else
            {
                prev->next = current->next;
            }
            sys_free(current);
            return true;
        }

        prev = current;
        current = current->next;
    }
    return false;
}

/******************************************************************************
* Function Name: command_background_task
 ********************************************************************************
* Summary:
*   Executes a background task.
* Parameters:
*   arg - A pointer to the background job structure.
*
* Return:
*   None.
*******************************************************************************/
static void command_background_task(void* arg) 
{
    background_job_t* job = (background_job_t*) arg;

    char* argv[MAX_CMD_ARGC];
    char buffer[MAX_CMD_LEN + MAX_CMD_ARGC];
    int argc = MAX_CMD_ARGC;

    tokenize(job->command, argv, &argc, buffer, sizeof(buffer));
    commands_t* commands = SYS_SERVICE_COMMANDS();
    command_t* cmd = command_find(commands, argv[0]);

    /* Update task name */ 
    if(argc > 0)
    {
        process_get(NULL)->name = argv[0];
    }

    /* A notification means that the parent can read the status code
    and free the job object. */
    if (cmd == NULL) 
    {
        printf("Command not found: %s\n", argv[0]);
        job->status = -1;
        xTaskNotifyGive(job->parent);
    } 
    else if (job->mode == EXEC_MODE_BACKGROUND) 
    {
        job->status = 0;
        xTaskNotifyGive(job->parent);
        int code = cmd->func(argc, argv);
        printf("Background task %s finished with code %d\n", argv[0], code);
    } 
    else if (job->mode == EXEC_MODE_BLOCK_ABORT) 
    {
        job->status = cmd->func(argc, argv);
        xTaskNotifyGive(job->parent);
    }

    /* argv[0] goes out of context, replace it */
    process_get(NULL)->name = "(zombie)";
    vTaskDelete(NULL);
}

/******************************************************************************
* Function Name: process_exec_command
 ********************************************************************************
* Summary:
*   Executes a command line in the specified execution mode.
* Parameters:
*   line - The command line to execute.
*   mode - The execution mode.
*
* Return:
*   The status code of the command.
*******************************************************************************/
int process_exec_command(char* line, exec_mode_t mode) 
{
    if (mode == EXEC_MODE_BACKGROUND || mode == EXEC_MODE_BLOCK_ABORT) 
    {
        /* Allocate memory for the job structure */
        background_job_t* job = sys_malloc(sizeof(background_job_t));
        if (job == NULL) 
        {
            printf("Failed to allocate memory for background job\n");
            return -1;
        }

        /* Initialize the job structure */
        job->command = line;
        job->mode = mode;
        job->parent = xTaskGetCurrentTaskHandle();
        
        /* Get the priority of the current task */
        TaskStatus_t status;
        vTaskGetInfo(NULL, &status, pdFALSE, eInvalid);
        UBaseType_t current_priority = status.uxCurrentPriority;

        /* Create the background task with the same priority as the current task */
        TaskHandle_t child;
        BaseType_t task_result = xTaskCreate(
            command_background_task,
            "(pending)",
            COMMAND_STACK_SIZE,
            (void*)job,
            current_priority,
            &child);

        process_t* child_process = process_get(child);

        /* Handle task creation failure */
        if (task_result != pdPASS) 
        {
            printf("Failed to create background task\n");
            sys_free(job);
            return -1;
        }

        if (mode == EXEC_MODE_BACKGROUND) 
        {
            ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
            sys_free(job);
            return 0;
        } 
        else if (mode == EXEC_MODE_BLOCK_ABORT) 
        {

            while (true) 
            {               
                /* Was Ctrl-C pressed? */
                if (child_process->terminal->last_input == 3 && !(child_process->attrib & PROCESS_NOEXIT)) 
                {
                    printf("Killed task\n");
                    process_kill(child_process, SIGKILL);
                    sys_free(job);
                    return -1;
                }               

                if (ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(250)) > 0) 
                {
                    sys_free(job);
                    return job->status;
                }
            }
        }
    } 
    else if(mode == EXEC_MODE_BLOCK) 
    { 
        char buffer[MAX_CMD_LEN + MAX_CMD_ARGC];
        char* tokens[MAX_CMD_ARGC];
        int token_count = MAX_CMD_ARGC;

        /* Tokenize the command line */
        tokenize(line, tokens, &token_count, buffer, sizeof(buffer));

        if (token_count == 0)
        {
            return 0;
        }


        commands_t* commands = SYS_SERVICE_COMMANDS();
        command_t* cmd = command_find(commands, tokens[0]);

        if (cmd == NULL) 
        {
            printf("Command not found: %s\n", tokens[0]);
            return -1;
        }

        return cmd->func(token_count, tokens);
    }
    return -1;
}

/******************************************************************************
* Function Name: process_kill
 ********************************************************************************
* Summary:
*   Kills a process with the specified signal.
* Parameters:
*   process - The process to kill.
*   signal - The signal to send to the process.
*
* Return:
*   true if the process was successfully killed, false otherwise.
*******************************************************************************/
bool process_kill(process_t* process, int signal) 
{

    if (process == NULL)
        process = process_get(NULL);

    vTaskDelete(process->xHandle);
    return true;
}

/******************************************************************************
* Function Name: process_set_attrib
 ********************************************************************************
* Summary:
*   Sets an attribute flag for a process.
* Parameters:
*   process - The process to set the attribute for.
*   flag - The attribute flag to set.
*
* Return:
*   None.
*******************************************************************************/
void process_set_attrib(process_t* process, process_attrib_t flag) 
{
    if (process == NULL)
        process = process_get(NULL);
    process->attrib |= flag;
}

/******************************************************************************
* Function Name: process_get
 ********************************************************************************
* Summary:
*   Retrieves the process structure for a given task handle.
* Parameters:
*   handle - The task handle to get the process for.
*
* Return:
*   The process structure for the given task handle, or NULL if not found.
*******************************************************************************/
process_t* process_get(TaskHandle_t handle) 
{
    if (handle == NULL)
    {
        handle = xTaskGetCurrentTaskHandle();
    }

    if (handle == NULL)
    {
        return NULL;
    }

    return (process_t*)xTaskGetApplicationTaskTag(handle);
}

/******************************************************************************
* Function Name: process_sleep
 ********************************************************************************
* Summary:
*   Suspends the execution of the current process for a specified duration.
* Parameters:
*   milliseconds - The duration to sleep in milliseconds.
*
* Return:
*   None.
*******************************************************************************/
void process_sleep(int milliseconds) 
{
    vTaskDelay(pdMS_TO_TICKS(milliseconds));
}

/******************************************************************************
* Function Name: process_find
 ********************************************************************************
* Summary:
*   Finds a process by its PID.
* Parameters:
*   pid - The PID of the process to find.
*
* Return:
*   The process structure for the given PID, or NULL if not found.
*******************************************************************************/
process_t* process_find(int pid) 
{

    TaskStatus_t taskStatusArray[MAX_TASKS];
    uint32_t totalRunTime;

    /* Get the snapshot of the current task state */
    UBaseType_t taskCount = uxTaskGetSystemState(taskStatusArray, MAX_TASKS, &totalRunTime);

    /* Iterate through each task to find the task with the given PID */
    for (UBaseType_t i = 0; i < taskCount; i++) 
    {
        if (taskStatusArray[i].xTaskNumber == pid) 
        {
            return process_get(taskStatusArray[i].xHandle);
        }
    }
    return NULL;
}

/******************************************************************************
* Function Name: process_new_terminal
 ********************************************************************************
* Summary:
*   Creates a new terminal for a process.
* Parameters:
*   process - The process to create the terminal for.
*   write - The function to use for writing to the terminal.
*
* Return:
*   The newly created terminal structure.

*******************************************************************************/
terminal_t* process_new_terminal(process_t* process, terminal_write_fn write) 
{
    if (process == NULL)
    {
        process = process_get(NULL);
    }

    if (process->attrib & PROCESS_TERMINAL_OWNER) 
    {
        terminal_destroy(process->terminal);
        shell_free(process->terminal);
    }

    terminal_t* terminal = (terminal_t*)shell_malloc(sizeof(terminal_t));
    terminal_init(terminal, write);

    process->terminal = terminal;
    process->attrib |= PROCESS_TERMINAL_OWNER;

    return terminal;
}

/******************************************************************************
* Function Name: process_create_hook
 ********************************************************************************
* Summary:
*   Hook function that is called when a new process is created. 
*   It initializes the process structure and assigns a terminal.
* Parameters:
*   handle - The handle of the newly created task.
*
* Return:
*   None.
*******************************************************************************/
void process_create_hook(TaskHandle_t handle) 
{
    _impure_ptr = &system_reent;

    process_t* process = process_get(handle);
    if (process != NULL) 
    {
        log_message(LOG_LEVEL_ERROR, "system", "Tag already assigned for '%s' (pid %d). Aborting.", process->name, process->pid);
        halt_error(LED_CODE_GENERIC_ERROR);
        return;
    }

    TaskStatus_t status;
    vTaskGetInfo(handle, &status, pdFALSE, eInvalid);

    process = (process_t*)shell_malloc(sizeof(process_t));
    memset(process, 0, sizeof(process_t));
    process->name = status.pcTaskName;
    process->pid = status.xTaskNumber;
    process->xHandle = handle;
    process->resource_list = NULL;

    _REENT_INIT_PTR(&process->reent);

    strlcpy(process->current_directory, "/", IM_PATH_MAX);
    vTaskSetApplicationTaskTag(handle, (TaskHookFunction_t)process);

    process_t* parent = process_get(NULL);
    if (parent == process) 
    {
        process_new_terminal(process, terminal_write_log);
        process->parent = NULL;

        log_message(LOG_LEVEL_DEBUG, "system", "Process '%s' (pid %d) initialized as a progenitor.",
            process->name, process->pid);
    } 
    else 
    {
        process->parent = parent;
        process->terminal = parent->terminal;
        strcpy(process->current_directory, parent->current_directory);

        if (parent->attrib & PROCESS_TRACK_RESOURCES) 
        {
            process->attrib |= PROCESS_TRACK_RESOURCES;
            process_add_resource(parent, PROCESS_RESOURCE_CHILD_PROCESS, process);
        }

        if (parent->pid == 1) 
        {
            log_message(LOG_LEVEL_DEBUG, "system", "Process '%s' (pid %d) spawned child '%s' (pid %d)",
                parent->name, parent->pid, process->name, process->pid);
        } 
    }
}

/******************************************************************************
* Function Name: process_switched_in_hook
 ********************************************************************************
* Summary:
*   Hook function that is called when a process is switched in.
*   It sets the global reentrancy pointer for the current process.
* Parameters:
*   None.
*
* Return:
*   None.
*******************************************************************************/
void process_switched_in_hook(void)
{
    TaskHandle_t xTask = xTaskGetCurrentTaskHandle();
    process_t* process = (process_t*)xTaskGetApplicationTaskTagFromISR(xTask);

    /* Set the global reentrancy pointer for Newlib */
    _impure_ptr = &(process->reent);   
}

/******************************************************************************
* Function Name: process_switched_out_hook
 ********************************************************************************
* Summary:
*   Hook function that is called when a process is switched out.
*   It sets the global reentrancy pointer for the system.
* Parameters:
*   None.
*
* Return:
*   None.
*******************************************************************************/
void process_switched_out_hook(void) 
{
    /* Set the global reentrancy pointer to the system reent structure */
    _impure_ptr = &system_reent;
}

/******************************************************************************
* Function Name: process_delete_hook
 ********************************************************************************
* Summary:
*   Hook function that is called when a process is deleted.
*   It performs cleanup of the process resources.
* Parameters:
*   handle - The handle of the task being deleted.
*
* Return:
*   None.
*******************************************************************************/
void process_delete_hook(TaskHandle_t handle) 
{
    process_t* process = process_get(handle);

    // log_message(LOG_LEVEL_DEBUG, "system", "Process '%s' (pid %d) terminated", process->name, process->pid);

    process->name = "(zombie)";

    if (process->attrib & PROCESS_TERMINAL_OWNER) 
    {
        terminal_destroy(process->terminal);
        shell_free(process->terminal);
    }

    if (process->attrib & PROCESS_TRACK_RESOURCES)
    {

        if (process->parent != NULL && process->parent->attrib & PROCESS_TRACK_RESOURCES)
        {
            process_remove_resource(process->parent, process);
        }


        for (process_resouce_t* r = process->resource_list; r != NULL;) 
        {
            process_t* child = NULL;
            switch (r->type) 
            {
                case PROCESS_RESOURCE_CHILD_PROCESS:
                {
                    child = (process_t *)r->resource;
                    process_kill(child, SIGKILL);
                    break;
                }

                case PROCESS_RESOURCE_ALLOCATED_MEMORY:
                {
                    sys_free(r->resource);
                    break;
                }

                default:
                {
                    break;                    
                }

            }
            process_resouce_t* next = r->next;
            sys_free(r);
            r = next;
        }
    }

    shell_free(process);
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */
