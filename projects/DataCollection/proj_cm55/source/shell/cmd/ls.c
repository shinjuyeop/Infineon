/******************************************************************************
* File Name: ls.c
*
* Description: Implementation of the 'ls' command for the shell.
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
#include <dirent.h>
#include <limits.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include "../process.h"
#include "../lib/getopt.h"
#include "../lib/strutils.h"  // format_byte_size
#include "../lib/cursor.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: is_not_dot_or_dotdot
********************************************************************************
* Summary:
*   Check if a directory entry is not "." or "..".
*
* Parameters:
*   name: The name of the directory entry.
*
* Return:
*   1 if the entry is not "." or "..", otherwise 0.
*
*******************************************************************************/
static int is_not_dot_or_dotdot(const char* name) 
{
    return strcmp(name, "..") != 0 && strcmp(name, ".") != 0;
}

/******************************************************************************
* Function Name: print_directory
********************************************************************************
* Summary:
*   Print the name of a directory entry.
*
* Parameters:
*   name: The name of the directory entry.
*
* Return:
*   None
*
*******************************************************************************/
static void print_directory(const char* name) 
{
    cursor_set_color(COLOR_BRIGHT_GREEN, COLOR_DEFAULT);
    printf("%s/ ", name);
    cursor_set_color(COLOR_DEFAULT, COLOR_DEFAULT);
    int spaces = 31 - (int)strlen(name);
    if(spaces < 0)
    {
        spaces = 0;
    }
    printf("%*s\n", spaces, "DIR");
}

/******************************************************************************
* Function Name: print_file
********************************************************************************
* Summary:
*   Print the name and size of a file.
*
* Parameters:
*   name: The name of the file.
*   size: The size of the file in bytes.
*
* Return:
*   None
*
*******************************************************************************/
static void print_file(const char* name, off_t size) 
{
    char size_buffer[32];
    format_byte_size(size_buffer, sizeof(size_buffer), size);
    cursor_set_color(COLOR_DEFAULT, COLOR_DEFAULT);
    printf("%-20s %10s B\n", name, size_buffer);
}

/******************************************************************************
* Function Name: main_ls
********************************************************************************
* Summary:
*   List the contents of a directory.
*
* Parameters:
*   argc: The number of command-line arguments.
*   argv: The array of command-line arguments.
*
* Return:
*   0 on success, -1 on error
*
*******************************************************************************/
int main_ls(int argc, char** argv) 
{
    int opt;
    const char* usage_text = "Usage: %s [directory]\n"
                             "List directory contents.\n"
                             "  -h   Display this help and exit\n";

    getopt_t go;
    getopt_init(&go);

    while ((opt = getopte(&go, argc, argv, "h")) != -1) 
    {
        switch (opt) 
        {
            case 'h':
            case '?':
                printf(usage_text, argv[0]);
                return 0;
            default:
                printf(usage_text, argv[0]);
                return -1;
        }
    }

    if (go.optind < argc - 1) 
    {
        printf("Error: Too many arguments\n");
        printf(usage_text, argv[0]);
        return -1;
    }

    const char* path = (go.optind < argc) ? argv[go.optind] : ".";
    char full_path[IM_PATH_MAX];

    if (realpath(path, full_path) == NULL) 
    {
        printf("Failed to resolve path. %s\n", strerror(errno));
        return -1;
    }

    DIR* dir = opendir(full_path);
    if (!dir) 
    {
        printf("Failed to list directory. %s\n", strerror(errno));
        return -1;
    }

    struct dirent* entry;

    /* First, print directories */
    while ((entry = readdir(dir)) != NULL) 
    {
        if (is_not_dot_or_dotdot(entry->d_name) && entry->d_type == DT_DIR) 
        {
            print_directory(entry->d_name);
        }
    }

    /* Rewind directory */
    rewinddir(dir);

    /* Then, print files */
    while ((entry = readdir(dir)) != NULL) 
    {
        if (is_not_dot_or_dotdot(entry->d_name) && entry->d_type == DT_REG) 
        {
            print_file(entry->d_name, entry->d_reclen);        
        }
    }

    closedir(dir);   
    return 0;
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */