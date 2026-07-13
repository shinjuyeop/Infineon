/******************************************************************************
* File Name: rm.c
*
* Description: Implementation of the 'rm' command for the shell.
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
#include <stdlib.h>
#include <unistd.h>
#include <dirent.h>
#include <errno.h>
#include "../process.h"
#include "../lib/getopt.h"
#include "../../services.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: is_directory
********************************************************************************
* Summary:
*   Check if a given path is a directory.
*
* Parameters:
*   path: The path to check.
*
* Return:
*   1 if the path is a directory, 0 otherwise.
*
*******************************************************************************/
int is_directory(const char* path) 
{
    struct stat info;
    if (stat(path, &info) != 0) 
    {
        return 0;
    }
    return S_ISDIR(info.st_mode);
}

/******************************************************************************
* Function Name: remove_directory_recursive
********************************************************************************
* Summary:
*   Remove a directory and its contents recursively.
*
* Parameters:
*   path: The path to the directory to remove.
*
* Return:
*   0 if the directory was successfully removed, -1 otherwise.
*
*******************************************************************************/
int remove_directory_recursive(const char* path) 
{
    DIR* d = opendir(path);
    size_t path_len = strlen(path);
    int r = -1;

    if (d) 
    {
        struct dirent* p;
        r = 0;
        while (!r && (p = readdir(d))) 
        {
            int r2 = -1;
            char* buf;
            size_t len;

            if (!strcmp(p->d_name, ".") || !strcmp(p->d_name, "..")) 
            {
                continue;
            }

            len = path_len + strlen(p->d_name) + 2;
            buf = malloc(len);

            if (buf) 
            {
                snprintf(buf, len, "%s/%s", path, p->d_name);

                /* Attempt unlink first; if it fails because buf is a directory,
                 * recurse into it. This avoids a stat+act TOCTOU race. */
                if (unlink(buf) == 0)
                {
                    r2 = 0;
                }
                else if (errno == EISDIR)
                {
                    r2 = remove_directory_recursive(buf);
                }
                free(buf);
            }
            r = r2;
        }
        closedir(d);
    }

    if (!r) 
    {
        r = rmdir(path);
    }

    return r;
}

/******************************************************************************
* Function Name: main_rm
********************************************************************************
* Summary:
*   Implementation of the 'rm' command for the shell.
*
* Parameters:
*   argc: The number of command-line arguments.
*   argv: The array of command-line arguments.
*
* Return:
*   0 if the command was successful, -1 otherwise.
*
*******************************************************************************/
int main_rm(int argc, char** argv) 
{
    const char* usage_text = "Usage: %s [-r] <file|directory>\n"
        "Remove a file or directory.\n"
        "  -h   Display this help and exit\n"
        "  -r   Remove directories and their contents recursively\n";

    getopt_t go;
    getopt_init(&go);

    int recursive = 0;

    int opt;
    while ((opt = getopte(&go, argc, argv, "hr")) != -1) 
    {
        switch (opt) 
        {
            case 'h':
                printf(usage_text, argv[0]);
                return 0;
            case 'r':
                recursive = 1;
                break;
            default:
                printf(usage_text, argv[0]);
                return -1;
        }
    }

    if (go.optind >= argc) 
    {
        printf("Error: No file or directory specified\n");
        printf(usage_text, argv[0]);
        return -1;
    }

    if (go.optind < argc - 1) 
    {
        printf("Error: Too many arguments\n");
        printf(usage_text, argv[0]);
        return -1;
    }

    char full_path[IM_PATH_MAX];
    realpath(argv[go.optind], full_path);

    if (is_directory(full_path)) 
    {
        if (recursive) 
        {
            if (remove_directory_recursive(full_path) != 0) 
            {
                printf("Error removing directory recursively. %s\n", strerror(errno));
                return -1;
            }
        }
        else 
        {
            if (rmdir(full_path) != 0) 
            {
                printf("Error removing directory. %s. Try rm -r\n", strerror(errno));
                return -1;
            }
        }
    }
    else 
    {
        if (unlink(full_path) != 0) 
        {
            printf("Error removing file. %s\n", strerror(errno));
            return -1;
        }
    }

    return 0;
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */