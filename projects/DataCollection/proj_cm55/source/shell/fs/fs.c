/******************************************************************************
* File Name: fs.c
*
* Description: Implementation of the file system management for the shell.
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
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include "fs.h"
#include "heap.h"

/******************************************************************************
 * Macros
 *****************************************************************************/
#define FS_DRIVER_MAX   (2)
#define FS_MOUNT_MAX    (3)
#define MAX_FDS         (5)

/* stdin, stdout, stderr */
#define FDS_OFFSET      (3) 

/******************************************************************************
 * Types
 *****************************************************************************/
typedef struct {
    fs_file_p file_ptr;
    fs_mount_t* mnt;
} fs_fdmap_t;

typedef struct {
    fs_driver_t* drivers[FS_DRIVER_MAX];
    int drivers_count;

    fs_mount_t* mounts[FS_MOUNT_MAX];
    int mounts_count;

    fs_fdmap_t fd_table[MAX_FDS];
} fs_t;

/******************************************************************************
 * Global Variables
 *****************************************************************************/
static fs_t fs = { 0 };

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: fs_find_mount
********************************************************************************
* Summary:
*   Find the mount point for a given absolute path using longest prefix matching.
*
* Parameters:
*   abspath: Absolute path to search for.
*
* Return:
*   Pointer to the mount structure if found, otherwise NULL.
*
*******************************************************************************/
static fs_mount_t* fs_find_mount(const char* abspath) 
{
    int max_len = 0;
    fs_mount_t* result = NULL;
    
    for (int i = 0; i < fs.mounts_count; i++) 
    {
        const char* mount_path = fs.mounts[i]->location;
        int len = strlen(mount_path) - 1;  /* Exclude trailing '/' */
        
        /* Skip if this mount is shorter than our current best match */
        if (len < max_len)
        {
            continue; 
        }
           
        /* Check if mount path is a prefix of the input path */
        if (strncmp(mount_path, abspath, len) == 0) 
        {
            /* Ensure proper path boundary: exact match or followed by '/' */
            if (abspath[len] == '\0' || abspath[len] == '/') 
            {
                max_len = len;
                result = fs.mounts[i];
            }
        }
    }
    return result;
}

/******************************************************************************
* Function Name: fs_get_fd
********************************************************************************
* Summary:
*   Get the file descriptor mapping for a given file descriptor.
*
* Parameters:
*   fd: File descriptor to search for.
*
* Return:
*   Pointer to the file descriptor mapping if found, otherwise NULL.
*
*******************************************************************************/
static fs_fdmap_t* fs_get_fd(int fd) 
{
    fd -= FDS_OFFSET;
    if (fd >= 0 && fd < MAX_FDS) 
    {
        return &fs.fd_table[fd];
    }
    return NULL;
}

/******************************************************************************
* Function Name: fs_add_driver
********************************************************************************
* Summary:
*   Add a new filesystem driver to the system.
*
* Parameters:
*   driver: Pointer to the filesystem driver to add.
*
* Return:
*   0 on success, -1 if the maximum number of drivers is reached.
*
*******************************************************************************/
int fs_add_driver(fs_driver_t* driver) 
{
    if (fs.drivers_count >= FS_DRIVER_MAX)
    {
        return -1;
    }
    fs.drivers[fs.drivers_count++] = driver;
    return 0;
}

/******************************************************************************
* Function Name: fs_print_usage
********************************************************************************
* Summary:
*   Print usage information for all registered filesystem drivers.
*
* Parameters:
*   None
*
* Return:
*   None
*
*******************************************************************************/
void fs_print_usage(void) 
{
    for (int i = 0; i < fs.drivers_count; i++) 
    {
        fs.drivers[i]->fs_print_usage();
    }
}

/******************************************************************************
* Function Name: fs_format
********************************************************************************
* Summary:
*   Format a filesystem of the specified type.
*
* Parameters:
*   fs_type: Filesystem type to format.
*   argc: Number of arguments for the format operation.
*   argv: Arguments for the format operation.
*
* Return:
*   0 on success, -1 on failure.
*
*******************************************************************************/
int fs_format(const char* fs_type, int argc, const char** argv) 
{
    
    fs_driver_t* driver = NULL;
    for (int i = 0; i < fs.drivers_count; i++) 
    {
        if (strcmp(fs.drivers[i]->fs_type, fs_type) == 0) 
        {
            driver = fs.drivers[i];
        }
    }

    if (driver == NULL) 
    {
        printf("Unkown fs type '%s'. Valid fs types: ", fs_type);
        for (int i = 0; i < fs.drivers_count; i++) 
        {
            printf("%s ", fs.drivers[i]->fs_type);
        }
        printf("\n");
        return -1;
    }

    if (driver->format(argc, argv) != 0) 
    {
        return -1;
    }

    return 0;
}

/******************************************************************************
* Function Name: fs_mount
********************************************************************************
* Summary:
*   Mount a filesystem of the specified type.
*
* Parameters:
*   abspath: Absolute path where the filesystem should be mounted.
*   fs_type: Filesystem type to mount.
*   argc: Number of arguments for the mount operation.
*   argv: Arguments for the mount operation.
*
* Return:
*   0 on success, -1 on failure.
*
*******************************************************************************/
int fs_mount(const char* abspath, const char* fs_type, int argc, const char** argv) 
{
    if (fs.mounts_count >= FS_MOUNT_MAX || abspath == NULL) 
    {
        if (fs.mounts_count >= FS_MOUNT_MAX) 
        {
            fprintf(stderr, "fs_mount: Maximum number of mounts (%d) exceeded\n", FS_MOUNT_MAX);
        } 
        else 
        {
            fprintf(stderr, "fs_mount: Invalid mount path (NULL)\n");
        }
        return -1;
    }
    
    if (abspath[strlen(abspath) - 1] != '/' || *abspath != '/') 
    {
        fprintf(stderr, "fs_mount: Mount path '%s' must be absolute and end with '/'\n", abspath);
        return -1;
    }

    /* Check if mount point directory exists (except for root mount) */
    if (strcmp(abspath, "/") != 0) 
    {
        /* Create a temporary path without trailing '/' for stat check */
        char temp_path[256];
        strcpy(temp_path, abspath);
        temp_path[strlen(temp_path) - 1] = '\0';  /* Remove trailing '/' */
        
        struct stat st;
        if (fs_stat(temp_path, &st) != 0) 
        {
            fprintf(stderr, "fs_mount: Mount point directory '%s' does not exist\n", temp_path);
            errno = ENOENT;  /* Mount point directory does not exist */
            return -1;
        }
        
        if (!S_ISDIR(st.st_mode)) 
        {
            fprintf(stderr, "fs_mount: Mount point '%s' is not a directory\n", temp_path);
            errno = ENOTDIR;  /* Mount point is not a directory */
            return -1;
        }
    }

    fs_driver_t* driver = NULL;
    for (int i = 0; i < fs.drivers_count; i++) 
    {
        if (strcmp(fs.drivers[i]->fs_type, fs_type) == 0) 
        {
            driver = fs.drivers[i];
        }
    }

    if (driver == NULL) 
    {
        fprintf(stderr, "fs_mount: Unknown filesystem type '%s'. Available types: ", fs_type);
        for (int i = 0; i < fs.drivers_count; i++) 
        {
            fprintf(stderr, "%s ", fs.drivers[i]->fs_type);
        }
        fprintf(stderr, "\n");
        return -1;
    }

    fs_mount_t* mount = sys_malloc(sizeof(fs_mount_t));
    if (mount == NULL) 
    {
        fprintf(stderr, "fs_mount: Memory allocation failed for mount structure\n");
        return -1;
    }

    mount->driver = driver;
    strcpy(mount->location, abspath);
    mount->handle = NULL;
        
    if (driver->mount(mount, argc, argv) != 0) 
    {
        fprintf(stderr, "fs_mount: Driver mount operation failed for '%s' (type: %s)\n", abspath, fs_type);
        sys_free(mount);
        return -1;
    }

    fs.mounts[fs.mounts_count++] = mount;

    return 0;
}

/******************************************************************************
* Function Name: fs_umount
********************************************************************************
* Summary:
*   Unmount a filesystem from the specified mount point.
*
* Parameters:
*   abspath: Absolute path where the filesystem is mounted.
*
* Return:
*   0 on success, -1 on failure.
*
*******************************************************************************/
int fs_umount(const char* abspath) 
{
    if (abspath == NULL)
    {
        return -1;
    }

    if (abspath[strlen(abspath) - 1] != '/' || *abspath != '/')
    {
        return -1;
    }

    for (int i = 0; i < fs.mounts_count; i++) 
    {
        if (strcmp(fs.mounts[i]->location, abspath) == 0) 
        {

            fs_mount_t* mnt = fs.mounts[i];

            /* TODO: check if present in fs.fd_table */ 

            if (mnt->driver->umount(mnt))
            {
                return -1;
            }

            memmove(fs.mounts + i, fs.mounts + i + 1, (fs.mounts_count - i - 1) * sizeof(fs_mount_t*));
            fs.mounts_count--;
            sys_free(mnt);
            return 0;
        }
    }

    return -1;
}

/******************************************************************************
* Function Name: fs_open
********************************************************************************
* Summary:
*   Open a file in the specified filesystem.
*
* Parameters:
*   abspath: Absolute path to the file.
*   flags: File open flags.
*   mode: File mode.
*
* Return:
*   File descriptor on success, -1 on failure.
*
*******************************************************************************/
int fs_open(const char* abspath, int flags, int mode) 
{
    if (abspath == NULL || *abspath != '/') 
    {
        errno = EINVAL;
        return -1;
    }

    fs_mount_t* mnt = fs_find_mount(abspath);

    if (mnt == NULL) 
    {
        errno = ENOENT;
        return -1;
    }

    int baselen = strlen(mnt->location) - 1;

    fs_file_p file = mnt->driver->open(mnt, abspath + baselen, flags, mode);
    if (file == NULL)
    {
        return -1;
    }

    for (int index = 0; index < MAX_FDS; index++) 
    {
        if (fs.fd_table[index].file_ptr == NULL) 
        {
            fs.fd_table[index].file_ptr = file;
            fs.fd_table[index].mnt = mnt;
            return index + FDS_OFFSET;
        }
    }
    return -1;
}

/******************************************************************************
* Function Name: fs_close
********************************************************************************
* Summary:
*   Close a file in the specified filesystem.
*
* Parameters:
*   fd: File descriptor of the file to be closed.
*
* Return:
*   0 on success, -1 on failure.
*
*******************************************************************************/
int fs_close(int fd) 
{
    fs_fdmap_t* fsp = fs_get_fd(fd);
    if (fsp == NULL)
    {
        errno = EBADF;
        return -1;
    }

    fs_mount_t* mnt = fsp->mnt;

    if (mnt->driver->close(mnt, fsp->file_ptr) != 0)
    {
        return -1;
    }
   
    fs.fd_table[fd - FDS_OFFSET].file_ptr = NULL;
    fs.fd_table[fd - FDS_OFFSET].mnt = NULL;
    
    return 0;
}

/******************************************************************************
* Function Name: fs_lseek
********************************************************************************
* Summary:
*   Reposition the file offset of the open file associated with the file descriptor.
*
* Parameters:
*   fd: File descriptor of the file.
*   offset: Offset to set.
*   whence: Position from where offset is applied.
*
* Return:
*   New file offset on success, -1 on failure.
*
*******************************************************************************/
_off_t fs_lseek(int fd, off_t offset, int whence) 
{
    fs_fdmap_t* fsp = fs_get_fd(fd);
    if (fsp == NULL)
    {
        errno = EBADF;
        return -1;
    }

    fs_mount_t* mnt = fsp->mnt;

    return mnt->driver->lseek(mnt, fsp->file_ptr, offset, whence);
}

/******************************************************************************
* Function Name: fs_read
********************************************************************************
* Summary:
*   Read data from the open file associated with the file descriptor.
*
* Parameters:
*   fd: File descriptor of the file.
*   buf: Buffer to store the read data.
*   count: Number of bytes to read.
*
* Return:
*   Number of bytes read on success, -1 on failure.
*
*******************************************************************************/
_ssize_t fs_read(int fd, void* buf, size_t count) 
{
    fs_fdmap_t* fsp = fs_get_fd(fd);
    if (fsp == NULL)
    {
        errno = EBADF;
        return -1;
    }

    fs_mount_t* mnt = fsp->mnt;

    return mnt->driver->read(mnt, fsp->file_ptr, buf, count);
}

/******************************************************************************
* Function Name: fs_write
********************************************************************************
* Summary:
*   Write data to the open file associated with the file descriptor.
*
* Parameters:
*   fd: File descriptor of the file.
*   buf: Buffer containing the data to write.
*   count: Number of bytes to write.
*
* Return:
*   Number of bytes written on success, -1 on failure.
*
*******************************************************************************/
_ssize_t fs_write(int fd, const void* buf, size_t count) 
{
    fs_fdmap_t* fsp = fs_get_fd(fd);
    if (fsp == NULL)
    {
        errno = EBADF;
        return -1;
    }

    fs_mount_t* mnt = fsp->mnt;

    return mnt->driver->write(mnt, fsp->file_ptr, buf, count);
}

/******************************************************************************
* Function Name: fs_stat
********************************************************************************
* Summary:
*   Get file status information.
*
* Parameters:
*   abspath: Absolute path of the file.
*   buf: Buffer to store the file status information.
*
* Return:
*   0 on success, -1 on failure.
*
*******************************************************************************/
int fs_stat(const char* abspath, struct stat* buf) 
{
    if (abspath == NULL || *abspath != '/') 
    {
        errno = EINVAL;
        return -1;
    }

    fs_mount_t* mnt = fs_find_mount(abspath);

    if (mnt == NULL) 
    {
        errno = ENOENT;
        return -1;
    }

    int baselen = strlen(mnt->location) - 1;

    return mnt->driver->stat(mnt, abspath + baselen, buf);
}

/******************************************************************************
* Function Name: fs_remove
********************************************************************************
* Summary:
*   Remove a file.
*
* Parameters:
*   abspath: Absolute path of the file.
*
* Return:
*   0 on success, -1 on failure.
*
*******************************************************************************/
int fs_remove(const char* abspath) 
{
    if (abspath == NULL || *abspath != '/') 
    {
        errno = EINVAL;
        return -1;
    }

    fs_mount_t* mnt = fs_find_mount(abspath);

    if (mnt == NULL) 
    {
        errno = ENOENT;
        return -1;
    }

    int baselen = strlen(mnt->location) - 1;

    return mnt->driver->remove(mnt, abspath + baselen);
}

/******************************************************************************
* Function Name: fs_mkdir
********************************************************************************
* Summary:
*   Create a directory.
*
* Parameters:
*   abspath: Absolute path of the directory.
*   mode: Permissions for the new directory.
*
* Return:
*   0 on success, -1 on failure.
*
*******************************************************************************/
int fs_mkdir(const char* abspath, int mode) 
{
    if (abspath == NULL || *abspath != '/') 
    {
        errno = EINVAL;
        return -1;
    }

    fs_mount_t* mnt = fs_find_mount(abspath);

    if (mnt == NULL) 
    {
        errno = ENOENT;
        return -1;
    }

    int baselen = strlen(mnt->location) - 1;

    return mnt->driver->mkdir(mnt, abspath + baselen, mode);
}

/******************************************************************************
* Function Name: fs_rename
********************************************************************************
* Summary:
*   Rename a file or directory.
*
* Parameters:
*   oldabspath: Absolute path of the old file or directory.
*   newabspath: Absolute path of the new file or directory.
*
* Return:
*   0 on success, -1 on failure.
*
*******************************************************************************/
int fs_rename(const char* oldabspath, const char* newabspath) 
{
    if (oldabspath == NULL || *oldabspath != '/') 
    {
        errno = EINVAL;
        return -1;
    }

    if (newabspath == NULL || *newabspath != '/') 
    {
        errno = EINVAL;
        return -1;
    }

    fs_mount_t* oldmnt = fs_find_mount(oldabspath);

    if (oldmnt == NULL) 
    {
        errno = ENOENT;
        return -1;
    }

    fs_mount_t* newmnt = fs_find_mount(newabspath);

    if (newmnt == NULL) 
    {
        errno = ENOENT;
        return -1;
    }

    /* old path and new path are not on the same mount point */
    if (oldmnt != newmnt) 
    {
        errno = EXDEV;
        return -1;
    }

    int baselen = strlen(newmnt->location) - 1;

    return newmnt->driver->rename(newmnt, oldabspath + baselen, newabspath + baselen);
}

/******************************************************************************
* Function Name: fs_opendir
********************************************************************************
* Summary:
*   Open a directory.
*
* Parameters:
*   abspath: Absolute path of the directory.
*
* Return:
*   DIR* on success, NULL on failure.
*
*******************************************************************************/
DIR* fs_opendir(const char* abspath) 
{
    if (abspath == NULL || *abspath != '/') 
    {
        errno = EINVAL;
        return NULL;
    }

    fs_mount_t* mnt = fs_find_mount(abspath);

    if (mnt == NULL) 
    {      
        errno = ENOENT;
        return NULL;
    }

    int baselen = strlen(mnt->location) - 1;

    return mnt->driver->opendir(mnt, abspath + baselen);
}

/******************************************************************************
* Function Name: fs_readdir
********************************************************************************
* Summary:
*   Read a directory entry.
*
* Parameters:
*   dirp: Pointer to the directory stream.
*
* Return:
*   struct dirent* on success, NULL on failure.
*
*******************************************************************************/
struct dirent* fs_readdir(DIR* dirp) 
{
    if (dirp == NULL) {
        errno = EINVAL;
        return NULL;
    }

    fs_mount_t* mnt = dirp->mnt;

    return mnt->driver->readdir(mnt, dirp);
}

/******************************************************************************
* Function Name: fs_closedir
********************************************************************************
* Summary:
*   Close a directory stream.
*
* Parameters:
*   dirp: Pointer to the directory stream.
*
* Return:
*   0 on success, -1 on failure.
*
*******************************************************************************/
int fs_closedir(DIR* dirp) 
{
    if (dirp == NULL) 
    {
        errno = EINVAL;
        return -1;
    }

    fs_mount_t* mnt = dirp->mnt;

    return mnt->driver->closedir(mnt, dirp);
}

/******************************************************************************
* Function Name: fs_rewinddir
********************************************************************************
* Summary:
*   Rewind a directory stream.
*
* Parameters:
*   dirp: Pointer to the directory stream.
*
* Return:
*   
*
*******************************************************************************/
void fs_rewinddir(DIR* dirp) 
{
    if (dirp == NULL)
        return;

    fs_mount_t* mnt = dirp->mnt;

    mnt->driver->rewinddir(mnt, dirp);
}

/******************************************************************************
* Function Name: fs_statvfs
********************************************************************************
* Summary:
*   Get file system statistics.
*
* Parameters:
*   abspath: Absolute path of the file or directory.
*   buf: Pointer to the statvfs structure to fill.
*
* Return:
*   0 on success, -1 on failure.
*
*******************************************************************************/
int fs_statvfs(const char* abspath, struct statvfs* buf) 
{
    if (abspath == NULL || *abspath != '/' || buf == NULL) 
    {
        errno = EINVAL;
        return -1;
    }

    fs_mount_t* mnt = fs_find_mount(abspath);

    if (mnt == NULL) 
    {
        errno = ENOENT;
        return -1;
    }

    return mnt->driver->statvfs(mnt, buf);
}

#endif /* IM_ENABLE_SHELL */
/* [] END OF FILE */
