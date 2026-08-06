/******************************************************************************
* File Name:   fs_util.c
*
* Description: Utility functions for file system operations in the shell.
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
#include "process.h"
#include "../../wifi_configs/im_config.h"
#include "fs.h"
#include "../../heap.h"
#include "statvfs.h"
#include "dirent.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: dirname
********************************************************************************
* Summary:
*   Returns the directory portion of a file path.
*
* Parameters:
*   path: The file path.
*
* Return:
*   The directory portion of the path, or "." if no directory is found.
*
*******************************************************************************/
char* dirname(char* path) 
{
    /* If path is NULL or empty, return "." */
    if (path == NULL || *path == '\0') 
    {
        return ".";
    }

    size_t len = strlen(path);

    /* Remove trailing slashes */
    while (len > 1 && path[len - 1] == '/') 
    {
        path[--len] = '\0';
    }

    /* If the path becomes empty after removing slashes, it's the root */
    if (len == 0) 
    {
        return "/";
    }

    /* Find the last slash */
    char* last_slash = strrchr(path, '/');
    if (last_slash == NULL) 
    {
        /* No slash found, return "." */
        return ".";
    }

    /* If the last slash is the first character, return "/" */
    if (last_slash == path) 
    {
        *(last_slash + 1) = '\0';
        return path;
    }

    /* Null-terminate the path at the last slash */
    *last_slash = '\0';

    return path;
}

/******************************************************************************
* Function Name: realpath
********************************************************************************
* Summary:
*   Returns the absolute path of a file, resolving any symbolic links, "." 
*   or ".." components.
*
* Parameters:
*   path: The file path to resolve.
*   resolved_path: Buffer to store the resolved absolute path.
*
* Return:
*   The resolved absolute path, or NULL if an error occurs.
*
*******************************************************************************/
char* realpath(const char* path, char* resolved_path) 
{
    if (!path || !resolved_path) 
    {
        errno = EINVAL;
        return NULL;
    }

    char temp_path[IM_PATH_MAX] = { 0 };

    if (path[0] == '/') 
    {
        strncpy(temp_path, path, IM_PATH_MAX - 1);
    } 
    else 
    {
        if (!getcwd(temp_path, IM_PATH_MAX)) 
        {
            return NULL;
        }
        size_t cwd_len = strlen(temp_path);
        if (cwd_len + 1 + strlen(path) >= IM_PATH_MAX) 
        {
            errno = ENAMETOOLONG;
            return NULL;
        }
        temp_path[cwd_len] = '/';
        strncpy(temp_path + cwd_len + 1, path, IM_PATH_MAX - cwd_len - 2);
    }

    resolved_path[0] = '/';
    resolved_path[1] = '\0';
    char* out_ptr = resolved_path + 1;

    char* token_start = temp_path;
    while (*token_start) 
    {
        while (*token_start == '/') 
        {
            token_start++;
        }
        if (!*token_start) 
        {
            break;
        }

        char* token_end = token_start;
        while (*token_end && *token_end != '/') 
        {
            token_end++;
        }
        *token_end = '\0';

        if (strcmp(token_start, ".") == 0) 
        {
            token_start = token_end + 1;
            continue;
        }

        if (strcmp(token_start, "..") == 0) 
        {
            if (out_ptr > resolved_path + 1) 
            {
                out_ptr--;
                while (out_ptr > resolved_path && *out_ptr != '/') 
                {
                    out_ptr--;
                }
                *out_ptr = '\0';
            }
            token_start = token_end + 1;
            continue;
        }

        if (strchr(token_start, '"') || strchr(token_start, '\'')) 
        {
            errno = EINVAL;
            return NULL;
        }

        if (out_ptr > resolved_path + 1) 
        {
            *out_ptr++ = '/';
        }
        size_t token_len = strlen(token_start);
        if (out_ptr + token_len >= resolved_path + IM_PATH_MAX) 
        {
            errno = ENAMETOOLONG;
            return NULL;
        }
        memcpy(out_ptr, token_start, token_len);
        out_ptr += token_len;
        *out_ptr = '\0';

        token_start = token_end + 1;
    }

    if (resolved_path[0] == '\0') 
    {
        strcpy(resolved_path, "/");
    }

    return resolved_path;
}

/******************************************************************************
* Function Name: getcwd
********************************************************************************
* Summary:
*   Returns the current working directory of the calling process.
*
* Parameters:
*   buf: Buffer to store the current working directory.
*   size: Size of the buffer.
*
* Return:
*   The current working directory, or NULL if an error occurs.
*
*******************************************************************************/
char* getcwd(char* buf, size_t size) 
{
    /* Retrieve the current process information */
    process_t* process = process_get(NULL);
    if (process == NULL) 
    {
        errno = EINVAL; /* Invalid argument */
        return NULL;
    }

    /* Check if the buffer size is sufficient */
    if (size == 0 || size < strlen(process->current_directory) + 1) 
    {
        errno = ERANGE; /* Buffer size is insufficient */
        return NULL;
    }

    /* Copy the current directory path to the buffer */
    strncpy(buf, process->current_directory, size);
    buf[size - 1] = '\0'; /* Ensure null-termination */

    return buf;
}

/******************************************************************************
* Function Name: chdir
********************************************************************************
* Summary:
*   Changes the current working directory of the calling process.
*
* Parameters:
*   path: The path to the new working directory.
*
* Return:
*   0 if the directory was successfully changed, or -1 if an error occurs.
*
*******************************************************************************/
int chdir(const char* path) 
{
    if (path == NULL || *path == '\0') 
    {
        errno = EINVAL; /* Invalid argument */
        return -1;
    }

    char resolved_path[IM_PATH_MAX];
    if (realpath(path, resolved_path) == NULL) 
    {
        errno = ENOENT; /* No such file or directory */
        return -1;
    }

    struct stat statbuf;
    if (stat(resolved_path, &statbuf) != 0) 
    {
        /* stat() failed, errno is already set appropriately */
        return -1;
    }

    /* Check if the resolved path is a directory */
    if (!S_ISDIR(statbuf.st_mode)) 
    {
        errno = ENOTDIR; /* Not a directory */
        return -1;
    }

    /* Retrieve the current process information */
    process_t* process = process_get(NULL);
    if (process == NULL) 
    {
        errno = EINVAL; /* Invalid argument */
        return -1;
    }

    strncpy(process->current_directory, resolved_path, IM_PATH_MAX - 1);
    process->current_directory[IM_PATH_MAX - 1] = '\0';

    return 0;
}

/******************************************************************************
* Function Name: mkdir
********************************************************************************
* Summary:
*   Creates a new directory with the specified path and mode.
*
* Parameters:
*   dir_path: The path to the new directory.
*   mode: The permissions to set for the new directory.
*
* Return:
*   0 if the directory was successfully created, or -1 if an error occurs.
*
*******************************************************************************/
int mkdir(const char* dir_path, mode_t mode) 
{
    process_t* process = process_get(NULL);
    if (process == NULL) 
    {
        errno = EINVAL; /* Invalid argument */
        return -1;
    }

    char resolved_path[IM_PATH_MAX];
    if (realpath(dir_path, resolved_path) == NULL)
    {
        return -1;
    }

    return _mkdir_r(&process->reent, resolved_path, mode);
}

/******************************************************************************
* Function Name: rmdir
********************************************************************************
* Summary:
*   Removes a directory with the specified path.
*
* Parameters:
*   dir_path: The path to the directory to be removed.
*
* Return:
*   0 if the directory was successfully removed, or -1 if an error occurs.
*
*******************************************************************************/
int rmdir(const char* dir_path) 
{
    char resolved_path[IM_PATH_MAX];
    if (realpath(dir_path, resolved_path) == NULL)
    {
        return -1;
    }
  
    return fs_remove(resolved_path);
}

/******************************************************************************
* Function Name: statvfs
********************************************************************************
* Summary:
*   Retrieves file system statistics for the specified path.
*
* Parameters:
*   path: The path to the file system.
*   buf: A pointer to a statvfs structure to store the file system statistics.
*
* Return:
*   0 if the file system statistics were successfully retrieved, or -1 if an error occurs.
*
*******************************************************************************/
int statvfs(const char* restrict path, struct statvfs* buf) 
{
    char resolved_path[IM_PATH_MAX];
    if (realpath(path, resolved_path) == NULL)
    {
        return -1;
    }

    return fs_statvfs(resolved_path, buf);
}

/******************************************************************************
* Function Name: opendir
********************************************************************************
* Summary:
*   Opens a directory stream corresponding to the directory name.
*
* Parameters:
*   name: The name of the directory to be opened.
*
* Return:
*   A pointer to the directory stream, or NULL if an error occurs.
*
*******************************************************************************/
DIR* opendir(const char* name) 
{
    char resolved_path[IM_PATH_MAX];
    if (realpath(name, resolved_path) == NULL)
    {
        return NULL;
    }
  
    return fs_opendir(resolved_path);   
}

/******************************************************************************
* Function Name: readdir
********************************************************************************
* Summary:
*   Reads a directory entry from the directory stream.
*
* Parameters:
*   dirp: A pointer to the directory stream.
*
* Return:
*   A pointer to the directory entry, or NULL if an error occurs.
*
*******************************************************************************/
struct dirent* readdir(DIR* dirp) 
{
    return fs_readdir(dirp);
}

/******************************************************************************
* Function Name: rewinddir
********************************************************************************
* Summary:
*   Resets the position of the directory stream to the beginning.
*
* Parameters:
*   dirp: A pointer to the directory stream.
*
* Return:
*   None.
*
*******************************************************************************/
void rewinddir(DIR* dirp) 
{
    fs_rewinddir(dirp);
}

/******************************************************************************
* Function Name: closedir
********************************************************************************
* Summary:
*   Closes the directory stream.
*
* Parameters:
*   dirp: A pointer to the directory stream.
*
* Return:
*   0 if the directory stream was successfully closed, or -1 if an error occurs.
*
*******************************************************************************/
int closedir(DIR* dirp) 
{
    return fs_closedir(dirp);
}

/******************************************************************************
* Function Name: file_exist
********************************************************************************
* Summary:
*   Checks if a file exists.
*
* Parameters:
*   file: The path to the file.
*
* Return:
*   true if the file exists, false otherwise.
*
*******************************************************************************/
bool file_exist(const char* file) 
{
    struct stat path_stat;
    return stat(file, &path_stat) == 0 && S_ISREG(path_stat.st_mode);
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */
