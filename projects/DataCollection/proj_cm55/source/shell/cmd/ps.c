/******************************************************************************
* File Name: ps.c
*
* Description: Implementation of the 'ps' command for the shell.
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

#ifdef IM_ENABLE_SHELL

#include <string.h>
#include <stdio.h>
#include <FreeRTOS.h>
#include <task.h>
#include "../process.h"
#include "../lib/getopt.h"
#include "../lib/strutils.h" 

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: main_ps
********************************************************************************
* Summary:
*   Implementation of the 'ps' command for the shell.
*
* Parameters:
*   argc: Argument count.
*   argv: Argument vector.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
int main_ps(int argc, char** argv) 
{
    const char* usage_text = "Usage: %s\n"
                             "List tasks.\n"
                             "  -a   Verbosed output\n"
                             "  -h   Display this help and exit\n";

    getopt_t go;
    getopt_init(&go);
    bool verbose = false;
    int opt;
    char stack_size_str[32];

    while ((opt = getopte(&go, argc, argv, "ha")) != -1) 
    {
        switch (opt) 
        {
            case 'a':
                verbose = true; 
                break;
            case 'h':
                printf(usage_text, argv[0]);
                return 0;
            default:
                printf(usage_text, argv[0]);
                return -1;
        }
    }

    if (go.optind < argc) 
    {
        printf("Error: Too many arguments\n");
        printf(usage_text, argv[0]);
        return -1;
    }

    TaskStatus_t taskStatusArray[MAX_TASKS];
    UBaseType_t taskCount;
    uint32_t totalRunTime;

    taskCount = uxTaskGetSystemState(taskStatusArray, MAX_TASKS, &totalRunTime);

    totalRunTime /= 100;

    if(verbose)
    {
       printf("%-15s  %-9s  %4s  %10s  %3s  %-15s  %4s  %s\n", "Process", "State", "Prio", "Stack Free", "Pid", "Parent", "CPU%", "Attrib");
    }
    else
    {
        printf("%-15s  %-9s  %4s  %3s  %4s\n", "Process", "State", "Prio", "Pid", "CPU%");
    }

    for (UBaseType_t i = 0; i < taskCount; i++) 
    {
        char* state;

        switch (taskStatusArray[i].eCurrentState) 
        {
            case eRunning:    
                state = "Running"; 
                break;

            case eReady:      
                state = "Ready"; 
                break;

            case eBlocked:    
                state = "Blocked"; 
                break;

            case eSuspended:  
                state = "Suspended"; 
                break;

            case eDeleted:    
                state = "Deleted"; 
                break;

            default:          
                state = "Unknown"; 
                break;
        }

        process_t* process = process_get(taskStatusArray[i].xHandle);
      
        if (verbose) 
        {
            format_byte_size(stack_size_str, sizeof(stack_size_str), taskStatusArray[i].usStackHighWaterMark * sizeof(StackType_t));

            char atrib_str[20] = "";

            if (process->attrib & PROCESS_TERMINAL_OWNER)
                strcat(atrib_str, "TERM ");

            if (process->attrib & PROCESS_NOEXIT)
                strcat(atrib_str, "NOEXIT ");

            if (process->attrib & PROCESS_TRACK_RESOURCES)
                strcat(atrib_str, "TRES ");

            printf("%-15s  %-9s  %4lu  %10s  %3u  %-15s  %4lu  %s\n",
                process->name,
                state,
                taskStatusArray[i].uxCurrentPriority,
                stack_size_str,
                process->pid,
                /* process->parent != NULL ? process->parent->name : */ "(none)",
                taskStatusArray[i].ulRunTimeCounter / totalRunTime,
                atrib_str);
        }
        else 
        {
            printf("%-15s  %-9s  %4lu  %3u  %4lu\n",
                process->name,
                state,
                taskStatusArray[i].uxCurrentPriority,
                process->pid,
                taskStatusArray[i].ulRunTimeCounter / totalRunTime
              );
        }
    }

    return 0;
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */