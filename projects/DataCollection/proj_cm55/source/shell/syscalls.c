/******************************************************************************
* File Name: syscalls.c
*
* Description: Implementation of system call stubs for the shell environment. 
* These functions provide basic file and process management capabilities, 
* allowing the shell to interact with the underlying operating system and file system.
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

#include <FreeRTOS.h>
#include <task.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/reent.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>

#include "../fs/fs.h"
#include "process.h"
#include "../../heap.h"
#include "../../common.h" 
#include "terminal.h"

/*******************************************************************************
* Macros
*******************************************************************************/
 /* Define a compile-time flag to enable or disable logging */ 
#define ENABLE_SYSCALL_LOGGING 0

#if ENABLE_SYSCALL_LOGGING
#define SYSCALL_LOG(fmt, ...) log_message(LOG_LEVEL_DEBUG, "syscalls", fmt, ##__VA_ARGS__)
#else
#define SYSCALL_LOG(fmt, ...) // Logging disabled
#endif

/*******************************************************************************
* Function Prototypes
*******************************************************************************/
/* Close a file descriptor */
int _close_r(struct _reent* reent, int fd);

/* Execute a program */
int _execve_r(struct _reent* reent, const char* pathname, char* const argv[], char* const envp[]);

/* Manipulate a file descriptor */
int _fcntl_r(struct _reent* reent, int fd, int cmd, int arg);

/* Create a new process */
int _fork_r(struct _reent* reent);

/* Get file status */
int _fstat_r(struct _reent* reent, int fd, struct stat* buf);

/* Get process ID */
int _getpid_r(struct _reent* reent);

/* Check if a file descriptor is a terminal */
int _isatty_r(struct _reent* reent, int fd);

/* Send a signal to a process */
int _kill_r(struct _reent* reent, int pid, int sig);

/* Create a new name for a file */
int _link_r(struct _reent* reent, const char* oldpath, const char* newpath);

/* Reposition read/write file offset */
_off_t _lseek_r(struct _reent* reent, int fd, _off_t offset, int whence);

/* Create a directory */
int _mkdir_r(struct _reent* reent, const char* pathname, int mode);

/* Open a file */
int _open_r(struct _reent* reent, const char* pathname, int flags, int mode);

/* Read from a file descriptor */
_ssize_t _read_r(struct _reent* reent, int fd, void* buf, size_t count);

/* Rename a file */
int _rename_r(struct _reent* reent, const char* oldpath, const char* newpath);

/* Change data segment size */
void* _sbrk_r(struct _reent* reent, ptrdiff_t increment);

/* Get file status */
int _stat_r(struct _reent* reent, const char* pathname, struct stat* buf);

/* Get process times */
_CLOCK_T_ _times_r(struct _reent* reent, struct tms* buf);

/* Remove a file */
int _unlink_r(struct _reent* reent, const char* pathname);

/* Wait for process to change state */
int _wait_r(struct _reent* reent, int* status);

/* Write to a file descriptor */
_ssize_t _write_r(struct _reent* reent, int fd, const void* buf, size_t count);

/* Get the current time of day */
int _gettimeofday_r(struct _reent* reent, struct timeval* tv, void* tz);

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: _close_r
********************************************************************************
* Summary:
*   Closes a file descriptor.
*
* Parameters:
*   reent - Pointer to the reentrancy structure.
*   fd - File descriptor to close.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
int _close_r(struct _reent* reent, int fd) 
{
    SYSCALL_LOG("_close(%d)", fd);
    if (fd == 0 || fd == 1 || fd == 2) 
    {
        errno = EINVAL;
        return -1;
    }

    return fs_close(fd);
}

/******************************************************************************
* Function Name: _execve_r
********************************************************************************
* Summary:
*   Executes a program.
*
* Parameters:
*   reent - Pointer to the reentrancy structure.
*   pathname - Path to the program to execute.
*   argv - Argument list for the program.
*   envp - Environment variables for the program.
*
* Return:
*  -1 and sets errno to ENOSYS, as this function is not implemented in this shell 
*   environment.
*
*******************************************************************************/
int _execve_r(struct _reent* reent, const char* pathname, char* const argv[], char* const envp[]) 
{
    SYSCALL_LOG("_execve('%s', %p, %p)", pathname, (void*)argv, (void*)envp);
    /* Function not implemented */
    errno = ENOSYS;  
    return -1;
}

/******************************************************************************
* Function Name: _fcntl_r
********************************************************************************
* Summary:
*   Performs file control operations.
*
* Parameters:
*   reent - Pointer to the reentrancy structure.
*   fd - File descriptor.
*   cmd - Command to perform.
*   arg - Argument for the command.
*
* Return:
*  -1 and sets errno to ENOSYS, as this function is not implemented in this shell 
*   environment.
*
*******************************************************************************/
int _fcntl_r(struct _reent* reent, int fd, int cmd, int arg) 
{
    SYSCALL_LOG("_fcntl(%d, %d, %d)", fd, cmd, arg);
    /* Function not implemented */
    errno = ENOSYS;  
    return -1;
}

/******************************************************************************
* Function Name: _fork_r
********************************************************************************
* Summary:
*   Creates a new process.
*
* Parameters:
*   reent - Pointer to the reentrancy structure.
*
* Return:
*  -1 and sets errno to ENOSYS, as this function is not implemented in this shell 
*   environment.
*
*******************************************************************************/
int _fork_r(struct _reent* reent) 
{
    SYSCALL_LOG("_fork()");
    /* Function not implemented */
    errno = ENOSYS; 
    return -1;
}

/******************************************************************************
* Function Name: _fstat_r
********************************************************************************
* Summary:
*   Gets file status.
*
* Parameters:
*   reent - Pointer to the reentrancy structure.
*   fd - File descriptor.
*   buf - Pointer to the stat structure to fill.
*
* Return:
*  -1 and sets errno to ENOSYS, as this function is not implemented in this shell 
*   environment.
*
*******************************************************************************/
int _fstat_r(struct _reent* reent, int fd, struct stat* buf) 
{
    SYSCALL_LOG("_fstat(%d, %p)", fd, buf);
    /* Function not implemented */
    errno = ENOSYS;  
    return -1;
}

/******************************************************************************
* Function Name: _getpid_r
********************************************************************************
* Summary:
*   Gets the process ID of the calling process.
*
* Parameters:
*   reent - Pointer to the reentrancy structure.
*
* Return:
*  The process ID of the calling process.
*
*******************************************************************************/
int _getpid_r(struct _reent* reent) 
{
    SYSCALL_LOG("_getpid()");
    process_t* process = process_get(NULL);
    return process->pid;
}

/******************************************************************************
* Function Name: _isatty_r
********************************************************************************
* Summary:
*   Checks if a file descriptor refers to a terminal.
*
* Parameters:
*   reent - Pointer to the reentrancy structure.
*   fd - File descriptor.
*
* Return:
*  1 if the file descriptor refers to a terminal, 0 otherwise.
*
*******************************************************************************/
int _isatty_r(struct _reent* reent, int fd) 
{
    SYSCALL_LOG("_isatty(%d)", fd);
    if (fd == 0 || fd == 1 || fd == 2) 
    {
        /* Standard input/output/error */
        return 1;  
    }
    errno = EBADF;
    return 0;
}

/******************************************************************************
* Function Name: _kill_r
********************************************************************************
* Summary:
*   Sends a signal to a process.
*
* Parameters:
*   reent - Pointer to the reentrancy structure.
*   pid - Process ID of the target process.
*   sig - Signal to send.
*
* Return:
*  0 on success, -1 on failure and sets errno.
*
*******************************************************************************/
int _kill_r(struct _reent* reent, int pid, int sig) 
{
    SYSCALL_LOG("_kill(%d, %d)", pid, sig);

    /* We only support signal 9 (SIGKILL) for task deletion */
    if (sig != SIGKILL) {
        errno = EINVAL;
        return -1;
    }

    process_t* to_kill = process_find(pid);

    if (to_kill == NULL) 
    {
        /* If task with the given pid is not found, set error */
        errno = ESRCH;
        return -1;
    }

    if (!process_kill(to_kill, sig)) 
    {
        errno = ESRCH;
        return -1;
    }

    return 0;
}

/******************************************************************************
* Function Name: _link_r
********************************************************************************
* Summary:
*   Creates a new link (also known as a hard link) to an existing file.
*
* Parameters:
*   reent - Pointer to the reentrancy structure.
*   oldpath - Path to the existing file.
*   newpath - Path to the new link.

*
* Return:
*  -1 and sets errno to ENOSYS, as this function is not implemented in this shell environment.
*
*******************************************************************************/
int _link_r(struct _reent* reent, const char* oldpath, const char* newpath) 
{
    SYSCALL_LOG("_link('%s', '%s')", oldpath, newpath);
    /* Function not implemented */
    errno = ENOSYS;  
    return -1;
}

/******************************************************************************
* Function Name: _lseek_r
********************************************************************************
* Summary:
*   Repositions the file offset of the open file associated with the file descriptor.
*
* Parameters:
*   reent - Pointer to the reentrancy structure.
*   fd - File descriptor.
*   offset - Offset to set.
*   whence - Position from which offset is applied.

*
* Return:
*  New file offset on success, -1 on failure and sets errno.
*
*******************************************************************************/
_off_t _lseek_r(struct _reent* reent, int fd, _off_t offset, int whence) 
{
    SYSCALL_LOG("_lseek(%d, %d, %d)", fd, offset, whence);

    if (fd == 0 || fd == 1 || fd == 2) 
    {
        errno = EINVAL;
        return -1;
    }

    return fs_lseek(fd, offset, whence);
}

/******************************************************************************
* Function Name: _mkdir_r
********************************************************************************
* Summary:
*   Creates a new directory.
*
* Parameters:
*   reent - Pointer to the reentrancy structure.
*   pathname - Path to the new directory.
*   mode - Permissions for the new directory.
*
* Return:
*  0 on success, -1 on failure and sets errno.
*
*******************************************************************************/
int _mkdir_r(struct _reent* reent, const char* pathname, int mode) 
{
    SYSCALL_LOG("_mkdir('%s', %d)", pathname, mode);

    char resolved_path[IM_PATH_MAX];
    if (realpath(pathname, resolved_path) == NULL) 
    {
        return -1;
    }

    return fs_mkdir(resolved_path, mode);
}

/******************************************************************************
* Function Name: _open_r
********************************************************************************
* Summary:
*   Opens a file.
*
* Parameters:
*   reent - Pointer to the reentrancy structure.
*   pathname - Path to the file.
*   flags - Flags for opening the file.
*   mode - Permissions for the file.
*
* Return:
*  File descriptor on success, -1 on failure and sets errno.
*
*******************************************************************************/
int _open_r(struct _reent* reent, const char* pathname, int flags, int mode) 
{
    SYSCALL_LOG("_open('%s', %d, %d)", pathname, flags, mode);

    char resolved_path[IM_PATH_MAX];
    if (realpath(pathname, resolved_path) == NULL) 
    {
        return -1;
    }

    return fs_open(resolved_path, flags, mode);
}

/******************************************************************************
* Function Name: _read_r
********************************************************************************
* Summary:
*   Reads data from a file descriptor.
*
* Parameters:
*   reent - Pointer to the reentrancy structure.
*   fd - File descriptor.
*   buf - Buffer to store read data.
*   count - Number of bytes to read.
*
* Return:
*  Number of bytes read on success, -1 on failure and sets errno.
*
*******************************************************************************/
_ssize_t _read_r(struct _reent* reent, int fd, void* buf, size_t count) 
{
    process_t* process = process_get(NULL);
    if (fd == 0 ) {
        return terminal_read(process->terminal , buf, count, -1);
    } 
    else if (fd == 1 || fd == 2) 
    {
        errno = EBADF;
        return -1;
    } 

    /* Don't log stdin. */
    SYSCALL_LOG("_read(%d, %p, %ld)", fd, buf, count);

    return fs_read(fd, buf, count);
}

/******************************************************************************
* Function Name: _rename_r
********************************************************************************
* Summary:
*   Renames a file or directory.
*
* Parameters:
*   reent - Pointer to the reentrancy structure.
*   oldpath - Path to the existing file or directory.
*   newpath - Path to the new file or directory.
*
* Return:
*  0 on success, -1 on failure and sets errno.
*
*******************************************************************************/
int _rename_r(struct _reent* reent, const char* oldpath, const char* newpath) 
{
    SYSCALL_LOG("_rename('%s', '%s')", oldpath, newpath);

    char resolved_oldpath[IM_PATH_MAX];
    char resolved_newpath[IM_PATH_MAX];

    if (realpath(oldpath, resolved_oldpath) == NULL) 
    {
        return -1;
    }

    if (realpath(newpath, resolved_newpath) == NULL) 
    {
        return -1;
    }

    return fs_rename(resolved_oldpath, resolved_newpath);
}

/******************************************************************************
* Function Name: _sbrk_r
********************************************************************************
* Summary:
*   Increases the program's data space.
*
* Parameters:
*   reent - Pointer to the reentrancy structure.
*   increment - Number of bytes to increase the data space by.
*
* Return:
*  Pointer to the previous end of the data space on success, 
* (caddr_t)-1 on failure and sets errno.
*
*******************************************************************************/
void* _sbrk_r(struct _reent* reent, ptrdiff_t increment) 
{
    SYSCALL_LOG("_sbrk(%ld)", increment);
    extern uint8_t  __HeapBase, __HeapLimit;
    static uint8_t* heapBrk = &__HeapBase;
    uint8_t* prevBrk = heapBrk;
    if ((increment > (int32_t)(&__HeapLimit - heapBrk)) ||
        (((int32_t)heapBrk + increment) < (int32_t)(&__HeapBase)))
    {
        errno = ENOMEM;
        return (caddr_t)-1;
    }
    heapBrk += increment;
    return (caddr_t)prevBrk;
}

/******************************************************************************
* Function Name: _stat_r
********************************************************************************
* Summary:
*   Retrieves information about a file.
*
* Parameters:
*   reent - Pointer to the reentrancy structure.
*   pathname - Path to the file.
*   buf - Pointer to the stat structure to store file information.
*
* Return:
*  0 on success, -1 on failure and sets errno.
*
*******************************************************************************/
int _stat_r(struct _reent* reent, const char* pathname, struct stat* buf) 
{
    SYSCALL_LOG("_stat('%s', %p)", pathname, buf);

    char resolved_path[IM_PATH_MAX];
    if (realpath(pathname, resolved_path) == NULL) 
    {
        return -1;
    }

    return fs_stat(resolved_path, buf);
}

/******************************************************************************
* Function Name: _times_r
********************************************************************************
* Summary:
*   Retrieves process times.
*
* Parameters:
*   reent - Pointer to the reentrancy structure.
*   buf - Pointer to the tms structure to store process times.
*
*
* Return:
*  -1 and sets errno to ENOSYS, as this function is not implemented in this shell 
*  environment.
*
*******************************************************************************/
_CLOCK_T_ _times_r(struct _reent* reent, struct tms* buf) 
{
    SYSCALL_LOG("_times(%p)", buf);
    /* Function not implemented */
    errno = ENOSYS;  
    return 0;
}

/******************************************************************************
* Function Name: _unlink_r
********************************************************************************
* Summary:
*   Deletes a file.
*
* Parameters:
*   reent - Pointer to the reentrancy structure.
*   pathname - Path to the file to be deleted.
*
*
* Return:
*  0 on success, -1 on failure and sets errno.
*
*******************************************************************************/
int _unlink_r(struct _reent* reent, const char* pathname) 
{
    SYSCALL_LOG("_unlink('%s')", pathname);

    char resolved_path[IM_PATH_MAX];
    if (realpath(pathname, resolved_path) == NULL) 
    {
        return -1;
    }

    return fs_remove(resolved_path);
}

/******************************************************************************
* Function Name: _wait_r
********************************************************************************
* Summary:
*   Waits for a child process to change state.
*
* Parameters:
*   reent - Pointer to the reentrancy structure.
*   status - Pointer to an integer to store the exit status of the child process.
*
*
* Return:
*  -1 and sets errno to ENOSYS, as this function is not implemented in this shell 
*  environment.
*
*******************************************************************************/
int _wait_r(struct _reent* reent, int* status) 
{
    SYSCALL_LOG("_wait(%p)", status);
    /* Function not implemented */
    errno = ENOSYS;  
    return -1;
}

/******************************************************************************
* Function Name: _write_r
********************************************************************************
* Summary:
*   Writes data to a file descriptor.
*
* Parameters:
*   reent - Pointer to the reentrancy structure.
*   fd - File descriptor to write to.
*   buf - Pointer to the buffer containing data to write.
*   count - Number of bytes to write.
*
* Return:
*  Number of bytes written on success, -1 on failure and sets errno.
*
*******************************************************************************/
_ssize_t _write_r(struct _reent* reent, int fd, const void* buf, size_t count) 
{
    process_t* process = process_get(NULL);
    if (fd == 1 || fd == 2) {

        if (process != NULL)
            return terminal_write(process->terminal, buf, count);
        return -1;
    }
    else if (fd == 0) {
        errno = EBADF;
        return -1;
    }

    /* Don't log stdout, stderr. */
    SYSCALL_LOG("_write(%d, %p, %ld)", fd, buf, count);

    return fs_write(fd, buf, count);
}

/******************************************************************************
* Function Name: _gettimeofday_r
********************************************************************************
* Summary:
*   Gets the current time of day.
*
* Parameters:
*   reent - Pointer to the reentrancy structure.
*   tv - Pointer to a timeval structure to store the current time.
*   tz - Pointer to a timezone structure (not used).
*
* Return:
*  0 on success, -1 on failure and sets errno.
*
*******************************************************************************/
int _gettimeofday_r(struct _reent* reent, struct timeval* tv, void* tz) 
{
    /* Function not implemented */
    errno = ENOSYS;  
    return -1;
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */