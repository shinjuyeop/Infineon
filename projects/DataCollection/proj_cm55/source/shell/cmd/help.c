/******************************************************************************
* File Name: help.c
*
* Description: Implementation of the 'help' command for the shell.
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
#include <stdbool.h>
#include "../lib/getopt.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: main_help
********************************************************************************
* Summary:
*   Implementation of the 'help' command for the shell.
*
* Parameters:
*   argc: Argument count.
*   argv: Argument vector.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
int main_help(int argc, char** argv) 
{
    int opt;
    const char* usage_text = "Usage: %s [arguments...]\n"
        "Display a summary of all commands.\n";

    getopt_t go;
    getopt_init(&go);

    while ((opt = getopte(&go, argc, argv, "h")) != -1) {
        switch (opt) {      
        default:
            printf(usage_text, argv[0]);
            return 0;
        }
    }

    printf("User commands:\n");
    printf("  cat              - Display file contents\n");
    printf("  cd               - Change current directory\n");
    printf("  clear            - Clear the screen\n");
    printf("  cp               - Copy a file\n");
    printf("  dapply           - Apply device options\n");
    printf("  dcap             - Capture data to disk\n");
    printf("  dhalt            - Stop all captures\n");
    printf("  disk             - Print disk usage\n");
    printf("  dlist            - List devices\n");
    printf("  dsave            - Save device configuration boot script\n");
    printf("  dset             - Set device option\n");    
    printf("  echo             - Write arguments to the standard output\n");
    printf("  edit             - Simple text editor\n");
    printf("  format           - Format file system\n");
    printf("  gpio             - GPIO tool\n");
    printf("  help             - Display this help message\n");
    printf("  hex              - Display file contents in hex\n");
    printf("  httpd            - Manage HTTP service\n");
    printf("  kill             - Kill a task by PID\n");
    printf("  login            - Prompts for password and starts a shell\n");
    printf("  loglevel         - Sets the current log level\n");
    printf("  ls               - List directory contents\n");
    printf("  mem              - List heap memory\n");
    printf("  mkdir            - Create a new directory\n");
    printf("  mount            - Mount file system\n");
    printf("  mv               - Move or rename a file or directory\n");
    printf("  passwd           - Set or change the shell password\n");
    printf("  ping             - Ping the specified host\n");
    printf("  ps               - List tasks\n");
    printf("  reboot           - Reboot the board\n");
    printf("  rm               - Remove a file or directory\n");
    printf("  sh               - Start a shell, run scripts\n");
    printf("  tspd             - Manage Tensor Streaming Protocol service\n");
    printf("  tsp2wav          - Convert tsp files to wav\n");
    printf("  umount           - Unmount file system\n");
    printf("  upnpd            -  Manage UPnP service\n");
    printf("  version          - Display build and version information\n");
    printf("  wifi             - Manage Wi-Fi connections\n");
    printf("  set_boardname    - Rename the board\n");
    return 0;
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */