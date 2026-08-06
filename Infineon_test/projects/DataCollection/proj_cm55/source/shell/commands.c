/******************************************************************************
* File Name:   commands.c
*
* Description: This file provides basic shell command functionalities like
*         - command registration
*         - command execution
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

/*******************************************************************************
* Header Files
*******************************************************************************/
#ifdef IM_ENABLE_SHELL

#include <stdio.h>
#include <string.h>
#include "../../common.h"
#include "commands.h"

/*******************************************************************************
* Function Name: command_init_default
********************************************************************************
* Summary:
*  Initializes the default commands in the shell.
*
* Parameters:
*  store - Pointer to the commands_t structure to initialize
*
* Return:
*  void
*
*******************************************************************************/
void command_init_default(commands_t* store) 
{
    memset(store, 0, sizeof(commands_t));

    /*
    * NOTE! Don't forget to add the command to help.c as well!!
    */
    
    command_add(store, "cat",          main_cat);
    command_add(store, "cd",           main_cd);
    command_add(store, "clear",        main_clear);
    command_add(store, "cp",           main_cp);
    command_add(store, "dapply",       main_dapply);
    command_add(store, "dcap",         main_dcap);
    command_add(store, "dhalt",        main_dhalt);
    command_add(store, "disk",         main_disk);
    command_add(store, "dlist",        main_dlist);
    command_add(store, "dsave",        main_dsave);
    command_add(store, "dset",         main_dset);
    command_add(store, "echo",         main_echo);
    command_add(store, "edit",         main_edit);
    command_add(store, "format",       main_format);
    command_add(store, "gpio",         main_gpio);
    command_add(store, "help",         main_help);
    command_add(store, "hex",          main_hex);
    command_add(store, "httpd",        main_httpd);
    command_add(store, "kill",         main_kill);
    command_add(store, "login",        main_login);
    command_add(store, "loglevel",     main_loglevel);
    command_add(store, "ls",           main_ls);
    command_add(store, "mem",          main_mem);
    command_add(store, "mkdir",        main_mkdir);
    command_add(store, "mount",        main_mount);
    command_add(store, "mv",           main_mv);
    command_add(store, "passwd",       main_passwd);
    command_add(store, "ping",         main_ping);
    command_add(store, "ps",           main_ps);
    command_add(store, "reboot",       main_reboot);
    command_add(store, "rm",           main_rm);
    command_add(store, "sh",           main_sh);
    command_add(store, "upnpd",        main_upnpd);
    command_add(store, "tspd",         main_tspd);
    command_add(store, "tsp2wav",      main_tsp2wav);
    command_add(store, "umount",       main_umount);
    command_add(store, "version",      main_version);
    command_add(store, "wifi",         main_wifi);
    command_add(store, "set_boardname", main_set_boardname);

    //command_add(store, "mdnsd",        main_mdnsd); 
    //command_add(store, "dindex",       main_dindex);    
}

/*******************************************************************************
* Function Name: command_add
********************************************************************************
* Summary:
*  Adds a command to the shell.
*
* Parameters:
*  store - Pointer to the commands_t structure to add the command to
*  name - Name of the command
*  func - Function pointer for the command
*
* Return:
*  true if the command was added successfully, false otherwise
*
*******************************************************************************/
bool command_add(commands_t* store, const char* name, command_fn func) 
{
    if (store->count >= MAX_COMMANDS) 
    {
        halt_error(LED_CODE_GENERIC_ERROR);
        return false;
    }

    command_t* cmd = &(store->commands[store->count++]);
    cmd->name = name;
    cmd->func = func;
    return true;
}

/*******************************************************************************
* Function Name: command_find
********************************************************************************
* Summary:
*  Finds a command in the shell.
*
* Parameters:
*  store - Pointer to the commands_t structure to search
*  name - Name of the command to find
*
* Return:
*  Pointer to the command_t structure if found, NULL otherwise
*
*******************************************************************************/
command_t* command_find(commands_t* store, const char* name) 
{
    for (int i = 0; i < store->count; i++) 
    {
        if (strcmp(name, store->commands[i].name) == 0) 
        {
            return &(store->commands[i]);
        }
    }

    return NULL;
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */