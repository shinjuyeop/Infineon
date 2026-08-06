/******************************************************************************
* File Name:   commands.h
*
* Description: This file provides the interface for shell command registration.
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

#ifndef _COMMANDS_H_
#define _COMMANDS_H_

#ifdef IM_ENABLE_SHELL

/******************************************************************************
 * Macros
 *****************************************************************************/
#define MAX_COMMANDS              (48)

/*******************************************************************************
* Types
*******************************************************************************/

/* Function pointer type for shell commands. */
typedef int (*command_fn)(int argc, char** argv);

/* Structure for storing shell commands. */
typedef struct 
{
    const char* name;          /**< Command name */
    command_fn func;           /**< Function pointer for the command */
} command_t;

/* List of commands */
typedef struct 
{
    command_t commands[MAX_COMMANDS];        /**< Array of shell commands */
    int count;                               /**< Number of registered commands */
} commands_t;

/*******************************************************************************
* Function Prototypes
*******************************************************************************/

void command_init_default(commands_t* store);
bool command_add(commands_t* commands, const char* name, command_fn func);
command_t* command_find(commands_t* store, const char* name);

int main_cat(int argc, char** argv);
int main_cd(int argc, char** argv);
int main_clear(int argc, char** argv);
int main_cp(int argc, char** argv);
int main_disk(int argc, char** argv);
int main_echo(int argc, char** argv);
int main_edit(int argc, char** args);
int main_ls(int argc, char** argv);
int main_gpio(int argc, char** argv);
int main_help(int argc, char** argv);
int main_hex(int argc, char** argv);
int main_kill(int argc, char** argv);
int main_mem(int argc, char** argv);
int main_mkdir(int argc, char** argv);
int main_mv(int argc, char** argv);
int main_passwd(int argc, char** argv);
int main_ps(int argc, char** argv);
int main_reboot(int argc, char** argv);
int main_rm(int argc, char** argv);
int main_version(int argc, char** argv);
int main_dcap(int argc, char** argv);
int main_dset(int argc, char** argv);
int main_dapply(int argc, char** argv);
int main_dsave(int argc, char** argv);
int main_dlist(int argc, char** argv);
int main_wifi(int argc, char** argv);
int main_httpd(int argc, char** argv);
int main_upnpd(int argc, char** argv);
int main_tspd(int argc, char** argv);
int main_mdnsd(int argc, char** argv);
int main_ping(int argc, char** argv);
int main_login(int argc, char** argv);
int main_loglevel(int argc, char** argv);
int main_sh(int argc, char** argv);
int main_mount(int argc, char** argv);
int main_umount(int argc, char** argv);
int main_format(int argc, char** argv);
int main_tsp2wav(int argc, char** argv);
int main_dhalt(int argc, char** argv);

int main_dindex(int argc, char** argv);

int main_set_boardname(int argc, char** argv);

#endif /* IM_ENABLE_SHELL */
#endif /* _COMMANDS_H_ */

/* [] END OF FILE */
