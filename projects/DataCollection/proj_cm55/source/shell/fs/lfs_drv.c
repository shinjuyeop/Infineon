/******************************************************************************
* File Name: lfs_drv.c
*
* Description: Implementation of the block device driver functions for SPI NOR
* flash for use with littlefs API. Uses cy_serial_flash_qspi for hardware access.
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
#include <cy_pdl.h>
#include <cybsp.h>
#include <cyabs_rtos.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cycfg_qspi_memslot.h>
#include <errno.h>
#include "../../common.h"
#include "lfs.h"
#include "fs.h"
#include "lfs_drv.h"
#include "lfs_spi_flash_bd.h"
#include "../../heap.h"
#include "../process.h"
#include "../../services.h"
#include "../../wifi_configs/im_config.h"
#include "../lib/getopt.h"
#include "mtb_serial_memory.h"

/******************************************************************************
 * Macros
 *****************************************************************************/
#ifdef USE_KIT_PSE84_HMI
#define FS_STORAGE_START_ADDRESS                    (0x2000000U)
#define FS_STORAGE_SIZE                             (0x2000000U)
#else
#define FS_STORAGE_START_ADDRESS                    (0xA00000U)
#define FS_STORAGE_SIZE                             (0x400000U)
#endif

#define QSPI_MIN_READ_SIZE                          (1UL)
#define LFS_CFG_DEFAULT_BLOCK_CYCLES                (512)

/*******************************************************************************
* Type
*******************************************************************************/
typedef struct {
    lfs_t lfs;
    struct lfs_config cfg;
} lfs_handle_t;

/*******************************************************************************
* Function Prototypes
*******************************************************************************/
static void set_errno_from_lfs(int lfs_err);
static bool _serial_memory_initialized = false;
static mtb_serial_memory_t serial_memory_obj;
static cy_stc_smif_mem_context_t smif_mem_context;
static cy_stc_smif_mem_info_t smif_mem_info;

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: lfsd_print_usage
********************************************************************************
* Summary:
*    Prints the usage information for the littlefs driver in the shell.
*
* Parameters:
*   None
*
* Return:
*   None
*
*******************************************************************************/
static void lfsd_print_usage(void)
{
    printf("Valid options when -t lfs\n");
    printf(" -o offset    Byte offset\n");
    printf(" -b size      Number of blocks for filesystem region.\n");
    printf(" -r size      Minimum size of a block read in bytes. All read operations will be a multiple of this value. Default is %lu.\n", QSPI_MIN_READ_SIZE);
    printf(" -e cycles    Number of erase cycles before littlefs evicts metadata logs and moves the metadata to another block. Default is %u.\n", LFS_CFG_DEFAULT_BLOCK_CYCLES);
}

/******************************************************************************
* Function Name: lfsd_load
********************************************************************************
* Summary:
*    Loads the littlefs driver and initializes the filesystem.
*
* Parameters:
*   mnt: Pointer to the mount structure.
*   argc: Argument count.
*   argv: Argument vector.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int lfsd_load(fs_mount_t* mnt, int argc, const char** argv)
{
    cy_rslt_t result;

    long offset = FS_STORAGE_START_ADDRESS;
    int block_count = -1;
    int read_size = QSPI_MIN_READ_SIZE;
    int block_cycles = LFS_CFG_DEFAULT_BLOCK_CYCLES;

    /* Parse arguments */
    for (int i = 0; i < argc; i++)
    {
        if (strcmp("o", argv[i]) == 0)
        {
            offset = atol(argv[++i]);
        }
        else if (strcmp("b", argv[i]) == 0)
        {
            block_count = atoi(argv[++i]);
        }
        else if (strcmp("r", argv[i]) == 0)
        {
            read_size = atoi(argv[++i]);
        }
        else if (strcmp("e", argv[i]) == 0)
        {
            block_cycles = atoi(argv[++i]);
        }
        else
        {
            lfsd_print_usage();
            errno = EINVAL;
            return -1;
        }
    }

    /* Set up serial memory using mtb_serial_memory (only if not already initialized) */
    if (!_serial_memory_initialized)
    {
#if defined(CYBSP_OSPI_FLASH_SS_ENABLED)
        result = mtb_serial_memory_setup(&serial_memory_obj,
                                          MTB_SERIAL_MEMORY_CHIP_SELECT_0,
                                          CYBSP_SMIF_CORE_0_XSPI_FLASH_hal_config.base,
                                          CYBSP_SMIF_CORE_0_XSPI_FLASH_hal_config.clock,
                                          &smif_mem_context,
                                          &smif_mem_info,
                                          &smif0BlockConfig);
#else
        result = mtb_serial_memory_setup(&serial_memory_obj,
                                          MTB_SERIAL_MEMORY_CHIP_SELECT_1,
                                          CYBSP_SMIF_CORE_0_XSPI_FLASH_hal_config.base,
                                          CYBSP_SMIF_CORE_0_XSPI_FLASH_hal_config.clock,
                                          &smif_mem_context,
                                          &smif_mem_info,
                                          &smif0BlockConfig);
#endif

        if (result != CY_RSLT_SUCCESS)
        {
            printf("Error: mtb_serial_memory_setup failed (0x%08X)\n", (unsigned int)result);
            errno = EIO;
            return -1;
        }

        _serial_memory_initialized = true;
    }

    /* Allocate filesystem handle */
    lfs_handle_t* fs = (lfs_handle_t*)sys_malloc(sizeof(lfs_handle_t));
    if (fs == NULL)
    {
        errno = ENOMEM;
        return -1;
    }
    memset(fs, 0, sizeof(lfs_handle_t));
    mnt->handle = fs;

    /* Get flash parameters */
    size_t flash_size = mtb_serial_memory_get_size(&serial_memory_obj);
    size_t erase_size = mtb_serial_memory_get_erase_size(&serial_memory_obj, offset);

    /* Calculate block count and region size with proper bounds checking */
    size_t region_size;
    if(block_count <= 0)
    {
        /* Use the smaller of remaining flash or designated storage size */
        size_t available_size = (flash_size > offset) ? (flash_size - offset) : 0;
        size_t max_size = (available_size < FS_STORAGE_SIZE) ? available_size : FS_STORAGE_SIZE;
        region_size = max_size;
        block_count = max_size / erase_size;
    }
    else
    {
        region_size = block_count * erase_size;
    }

    /* Configure memory region for littlefs */
    lfs_spi_flash_bd_configure_memory(&fs->cfg, offset, region_size);

    /* Create and initialize the block device using library function */
    result = lfs_spi_flash_bd_create(&fs->cfg, &serial_memory_obj);
    if (result != CY_RSLT_SUCCESS)
    {
        printf("Error: lfs_spi_flash_bd_create failed (0x%08X)\n", (unsigned int)result);
        sys_free(fs);
        errno = EIO;
        return -1;
    }

    /* Override configuration parameters if specified */
    if (read_size != QSPI_MIN_READ_SIZE)
    {
        fs->cfg.read_size = read_size;
    }
    if (block_cycles != LFS_CFG_DEFAULT_BLOCK_CYCLES)
    {
        fs->cfg.block_cycles = block_cycles;
    }

    if (fs->cfg.block_size >= (lfs_size_t)0x40000U)
    {
        fs->cfg.cache_size = 8192U;
    }

    log_message(LOG_LEVEL_INFO, "fs", "Flash total   : %lu bytes", flash_size);
    log_message(LOG_LEVEL_INFO, "fs", "offset        : 0x%lX (%ld bytes)", (unsigned long)offset, (long)offset);
    log_message(LOG_LEVEL_INFO, "fs", "region_size   : %lu bytes (%lu KB)",
                (unsigned long)(fs->cfg.block_count * fs->cfg.block_size),
                (unsigned long)(fs->cfg.block_count * fs->cfg.block_size / 1024));
    log_message(LOG_LEVEL_INFO, "fs", "read_size     : %ld", fs->cfg.read_size);
    log_message(LOG_LEVEL_INFO, "fs", "prog_size     : %ld", fs->cfg.prog_size);
    log_message(LOG_LEVEL_INFO, "fs", "block_size    : %ld", fs->cfg.block_size);
    log_message(LOG_LEVEL_INFO, "fs", "block_count   : %ld", fs->cfg.block_count);
    log_message(LOG_LEVEL_INFO, "fs", "cache_size    : %ld", fs->cfg.cache_size);
    log_message(LOG_LEVEL_INFO, "fs", "lookahead_size: %ld", fs->cfg.lookahead_size);

    return 0;
}

/******************************************************************************
* Function Name: lfsd_unload
********************************************************************************
* Summary:
*    Unloads the littlefs driver and cleans up the filesystem.
*
* Parameters:
*   mnt: Pointer to the mount structure.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int lfsd_unload(fs_mount_t* mnt)
{
    if (!_serial_memory_initialized)
    {
        printf("Error: serial memory not initialized.\n");
        errno = EIO;
        return -1;
    }

    lfs_handle_t* fs = (lfs_handle_t*)mnt->handle;

    /* Destroy the block device (cleans up resources) */
    lfs_spi_flash_bd_destroy(&fs->cfg);

    /* Note: Serial memory hardware remains initialized for reuse */
    /* Only free the filesystem handle */
    sys_free(fs);
    mnt->handle = NULL;

    return 0;
}

/******************************************************************************
* Function Name: lfsd_format
********************************************************************************
* Summary:
*    Formats the littlefs filesystem.
*
* Parameters:
*   argc: Argument count.
*   argv: Argument vector.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int lfsd_format(int argc, const char** argv)
{
    fs_mount_t mnt;

    log_message(LOG_LEVEL_INFO, "fs", "Formatting LFS drive");

    if (lfsd_load(&mnt, argc, argv) != 0)
    {
        return -1;
    }

    lfs_handle_t* fs = (lfs_handle_t*)mnt.handle;

    int status = lfs_format(&(fs->lfs), &fs->cfg);
    if (status < 0)
    {
        log_message(LOG_LEVEL_ERROR, "fs", "Failed to format file system. Error %d", status);
        return -1;
    }

    lfsd_unload(&mnt);

    return 0;
}

/******************************************************************************
* Function Name: lfsd_mount
********************************************************************************
* Summary:
*    Mounts the littlefs filesystem.
*
* Parameters:
*   mnt: Pointer to the mount structure.
*   argc: Argument count.
*   argv: Argument vector.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int lfsd_mount(fs_mount_t* mnt, int argc, const char** argv)
{
    log_message(LOG_LEVEL_INFO, "fs", "Mounting LFS drive");

    if (lfsd_load(mnt, argc, argv) != 0)
    {
        return -1;
    }

    lfs_handle_t* fs = (lfs_handle_t*)mnt->handle;

    int status = lfs_mount(&(fs->lfs), &(fs->cfg));
    if (status < 0)
    {
        log_message(LOG_LEVEL_ERROR, "fs", "Failed to mount file system. Error %d", status);

        lfsd_unload(mnt);
        return -1;
    }

    return 0;
}

/******************************************************************************
* Function Name: lfsd_umount
********************************************************************************
* Summary:
*    Unmounts the littlefs filesystem.
*
* Parameters:
*   mnt: Pointer to the mount structure.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int lfsd_umount(fs_mount_t* mnt)
{
    lfs_handle_t* fs = (lfs_handle_t*)mnt->handle;

    log_message(LOG_LEVEL_INFO, "fs", "Unmounting LFS drive");
    int status = lfs_unmount(&(fs->lfs));
    if (status < 0)
    {
        log_message(LOG_LEVEL_ERROR, "fs", "Failed to unmount file system. Error %d", status);
        return -1;
    }

    lfsd_unload(mnt);

    return 0;
}

/******************************************************************************
* Function Name: lfsd_open
********************************************************************************
* Summary:
*    Opens a file in the littlefs filesystem.
*
* Parameters:
*   mnt: Pointer to the mount structure.
*   pathname: Path to the file.
*   flags: File open flags.
*   mode: File mode.
*
* Return:
*   File pointer if operation is successful, otherwise NULL.
*
*******************************************************************************/
static fs_file_p lfsd_open(fs_mount_t* mnt, const char* pathname, int flags, int mode)
{
    lfs_handle_t* fs = (lfs_handle_t*)mnt->handle;

    lfs_file_t* file = sys_malloc(sizeof(lfs_file_t));
    if (file == NULL)
    {
        errno = ENOMEM;
        return NULL;
    }

    int lfs_flags = 0;

    /* Set LittleFS flags based on the provided flags */
    if ((flags & O_ACCMODE) == O_RDONLY) lfs_flags |= LFS_O_RDONLY;
    if ((flags & O_ACCMODE) == O_WRONLY) lfs_flags |= LFS_O_WRONLY;
    if ((flags & O_ACCMODE) == O_RDWR) lfs_flags |= LFS_O_RDWR;
    if (flags & O_CREAT) lfs_flags |= LFS_O_CREAT;
    if (flags & O_TRUNC) lfs_flags |= LFS_O_TRUNC;
    if (flags & O_APPEND) lfs_flags |= LFS_O_APPEND;

    /* Check if the file needs to be created with specific permissions */
   if ((flags & O_CREAT) && (flags & O_EXCL))
   {
        struct stat st;
        if (fs_stat(pathname, &st) == 0 && S_ISREG(st.st_mode))
        {
            /* File exists and O_EXCL is specified */
            errno = EEXIST;
            sys_free(file);
            return NULL;
        }

    }

    int res = lfs_file_open(&(fs->lfs), file, pathname, lfs_flags);
    if (res < 0)
    {
        sys_free(file);
        set_errno_from_lfs(res);
        return NULL;
    }

    return (fs_file_p)file;
}

/******************************************************************************
* Function Name: lfsd_close
********************************************************************************
* Summary:
*    Closes a file in the littlefs filesystem.
*
* Parameters:
*   mnt: Pointer to the mount structure.
*   file: File pointer.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int lfsd_close(fs_mount_t* mnt, fs_file_p file)
{
   lfs_handle_t* fs = (lfs_handle_t*)mnt->handle;

   int res = lfs_file_close(&(fs->lfs), (lfs_file_t*)file);
   if (res < 0)
   {
       set_errno_from_lfs(res);
       return -1;
   }

   sys_free(file);
   return 0;
}

/******************************************************************************
* Function Name: lfsd_lseek
********************************************************************************
* Summary:
*    Moves the file pointer to a specific location in the littlefs filesystem.
*
* Parameters:
*   mnt: Pointer to the mount structure.
*   file: File pointer.
*   offset: Offset to move the file pointer.
*   whence: Position from where the offset is applied.
*
* Return:
*   New file pointer position if operation is successful, otherwise -1.
*
*******************************************************************************/
static _off_t lfsd_lseek(fs_mount_t* mnt, fs_file_p file, off_t offset, int whence)
{
    lfs_handle_t* fs = (lfs_handle_t*)mnt->handle;

    lfs_soff_t res = lfs_file_seek(&(fs->lfs), (lfs_file_t*)file, offset, whence);
    if (res < 0)
    {
        set_errno_from_lfs(res);
        return -1;
    }
    return (_off_t)res;
}

/******************************************************************************
* Function Name: lfsd_read
********************************************************************************
* Summary:
*    Reads data from a file in the littlefs filesystem.
*
* Parameters:
*   mnt: Pointer to the mount structure.
*   file: File pointer.
*   buf: Buffer to store the read data.
*   count: Number of bytes to read.
*
* Return:
*   Number of bytes read if operation is successful, otherwise -1.
*
*******************************************************************************/
static _ssize_t lfsd_read(fs_mount_t* mnt, fs_file_p file, void* buf, size_t count)
{
    lfs_handle_t* fs = (lfs_handle_t*)mnt->handle;

    int res = lfs_file_read(&(fs->lfs), (lfs_file_t*)file, buf, count);
    if (res < 0)
    {
        set_errno_from_lfs(res);
        return -1;
    }
    return res;

}

/******************************************************************************
* Function Name: lfsd_write
********************************************************************************
* Summary:
*    Writes data to a file in the littlefs filesystem.
*
* Parameters:
*   mnt: Pointer to the mount structure.
*   file: File pointer.
*   buf: Buffer containing the data to write.
*   count: Number of bytes to write.
*
* Return:
*   Number of bytes written if operation is successful, otherwise -1.
*
*******************************************************************************/
static _ssize_t lfsd_write(fs_mount_t* mnt, fs_file_p file, const void* buf, size_t count)
{
    lfs_handle_t* fs = (lfs_handle_t*)mnt->handle;

    int res = lfs_file_write(&(fs->lfs), (lfs_file_t*)file, buf, count);
    if (res < 0)
    {
        set_errno_from_lfs(res);
        return -1;
    }
    return res;
}

/******************************************************************************
* Function Name: lfsd_stat
********************************************************************************
* Summary:
*    Retrieves information about a file in the littlefs filesystem.
*
* Parameters:
*   mnt: Pointer to the mount structure.
*   pathname: Path to the file.
*   buf: Buffer to store the file information.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int lfsd_stat(fs_mount_t* mnt, const char* pathname, struct stat* buf)
{
    lfs_handle_t* fs = (lfs_handle_t*)mnt->handle;
    struct lfs_info info;

    int res = lfs_stat(&(fs->lfs), pathname, &info);
    if (res < 0)
    {
        set_errno_from_lfs(res);
        return -1;
    }

    /* Initialize the stat structure to zero */
    memset(buf, 0, sizeof(struct stat));

    /* Populate the stat structure */
    buf->st_size = info.size;
    buf->st_mode = 0;
    if (info.type == LFS_TYPE_REG)
    {
        buf->st_mode |= S_IFREG;
    }
    else if (info.type == LFS_TYPE_DIR)
    {
        buf->st_mode |= S_IFDIR;
    }
    /* Add additional fields if necessary (e.g., st_mtime, st_ctime, etc.) */

    return 0;
}

/******************************************************************************
* Function Name: lfsd_remove
********************************************************************************
* Summary:
*    Removes a file in the littlefs filesystem.
*
* Parameters:
*   mnt: Pointer to the mount structure.
*   pathname: Path to the file.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int lfsd_remove(fs_mount_t* mnt, const char* pathname)
{

    lfs_handle_t* fs = (lfs_handle_t*)mnt->handle;
    int res = lfs_remove(&(fs->lfs), pathname);
    if (res < 0) {
        set_errno_from_lfs(res);
        return -1;
    }

    return 0;
}

/******************************************************************************
* Function Name: lfsd_mkdir
********************************************************************************
* Summary:
*    Creates a directory in the littlefs filesystem.
*
* Parameters:
*   mnt: Pointer to the mount structure.
*   pathname: Path to the directory.
*   mode: Mode for the new directory.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int lfsd_mkdir(fs_mount_t* mnt, const char* pathname, int mode)
{
    lfs_handle_t* fs = (lfs_handle_t*)mnt->handle;
    int res = lfs_mkdir(&(fs->lfs), pathname);
    if (res < 0)
    {
        set_errno_from_lfs(res);
        return -1;
    }

    return 0;
}

/******************************************************************************
* Function Name: lfsd_rename
********************************************************************************
* Summary:
*    Renames a file or directory in the littlefs filesystem.
*
* Parameters:
*   mnt: Pointer to the mount structure.
*   oldpath: Path to the existing file or directory.
*   newpath: Path to the new file or directory name.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int lfsd_rename(fs_mount_t* mnt, const char* oldpath, const char* newpath)
{
    lfs_handle_t* fs = (lfs_handle_t*)mnt->handle;

    int res = lfs_rename(&(fs->lfs), oldpath, newpath);
    if (res < 0)
    {
        set_errno_from_lfs(res);
        return -1;
    }

    return 0;
}

/******************************************************************************
* Function Name: lfsd_opendir
********************************************************************************
* Summary:
*    Opens a directory in the littlefs filesystem.
*
* Parameters:
*   mnt: Pointer to the mount structure.
*   name: Path to the directory.
*
* Return:
*   Pointer to the DIR structure if operation is successful, otherwise NULL.
*
*******************************************************************************/
static DIR* lfsd_opendir(fs_mount_t* mnt, const char* name)
{
    lfs_handle_t* fs = (lfs_handle_t*)mnt->handle;
    DIR* dirp = (DIR*)sys_malloc(sizeof(DIR));
    if (!dirp)
    {
        errno = ENOMEM;
        return NULL;
    }
    dirp->mnt = mnt;
    dirp->dir = (lfs_dir_t*)sys_malloc(sizeof(lfs_dir_t));

    int res = lfs_dir_open(&(fs->lfs), (lfs_dir_t *)(dirp->dir), name);
    if (res < 0)
    {
        sys_free(dirp->dir);
        dirp->dir = NULL;
        sys_free(dirp);
        errno = EIO;
        return NULL;
    }

    return dirp;
}

/******************************************************************************
* Function Name: lfsd_readdir
********************************************************************************
* Summary:
*    Reads the next entry in a directory in the littlefs filesystem.
*
* Parameters:
*   mnt: Pointer to the mount structure.
*   dirp: Pointer to the DIR structure.
*
* Return:
*   Pointer to the dirent structure containing the directory entry
*   information if operation is successful, otherwise NULL.
*
*******************************************************************************/
static struct dirent* lfsd_readdir(fs_mount_t* mnt, DIR* dirp)
{
    if (!dirp || !mnt)
    {
        errno = EBADF;
        return NULL;
    }

    struct lfs_info info;
    lfs_handle_t* fs = (lfs_handle_t*)mnt->handle;

    /* Read the next directory entry */
    int res = lfs_dir_read(&(fs->lfs), (lfs_dir_t*)(dirp->dir), &info);
    if (res < 0)
    {
        errno = EIO;
        return NULL;
    }

    /* Check if the end of the directory is reached */
    if (res == 0)
    {
        return NULL;
    }

    /* Copy the filename to the dirent structure */
    strncpy(dirp->entry.d_name, info.name, IM_PATH_MAX);
    dirp->entry.d_name[IM_PATH_MAX - 1] = '\0';  /* Ensure null termination */
    dirp->entry.d_reclen = info.size;
    switch (info.type)
    {
        case LFS_TYPE_REG:
        {
            dirp->entry.d_type = DT_REG;
            break;
        }

        case LFS_TYPE_DIR:
        {
            dirp->entry.d_type = DT_DIR;
            break;
        }

        default:
        {
            dirp->entry.d_type = DT_UNKNOWN;
            break;
        }

    }

    return &(dirp->entry);
}

/******************************************************************************
* Function Name: lfsd_rewinddir
********************************************************************************
* Summary:
*    Rewinds the directory stream to the beginning in the littlefs filesystem.
*
* Parameters:
*   mnt: Pointer to the mount structure.
*   dirp: Pointer to the DIR structure.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int lfsd_rewinddir(fs_mount_t* mnt, DIR* dirp)
{
    if (!mnt || !dirp)
    {
        errno = EBADF;
        return -1;
    }

    lfs_handle_t* fs = (lfs_handle_t*)mnt->handle;
    int res = lfs_dir_rewind(&(fs->lfs), (lfs_dir_t*)(dirp->dir));
    if (res < 0)
    {
        errno = EIO;
        return -1;
    }
    return 0;
}

/******************************************************************************
* Function Name: lfsd_closedir
********************************************************************************
* Summary:
*    Closes the directory stream in the littlefs filesystem.
*
* Parameters:
*   mnt: Pointer to the mount structure.
*   dirp: Pointer to the DIR structure.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int lfsd_closedir(fs_mount_t* mnt, DIR* dirp)
{
    if (!mnt || !dirp)
    {
        errno = EBADF;
        return -1;
    }

    lfs_handle_t* fs = (lfs_handle_t*)mnt->handle;

    int res = lfs_dir_close(&(fs->lfs), (lfs_dir_t*)(dirp->dir));

    sys_free(dirp->dir);
    dirp->dir = NULL;
    sys_free(dirp);

    if (res < 0)
    {
        errno = EIO;
        return -1;
    }

    return 0;
}

/******************************************************************************
* Function Name: lfsd_statvfs
********************************************************************************
* Summary:
*    Retrieves filesystem statistics in the littlefs filesystem.
*
* Parameters:
*   mnt: Pointer to the mount structure.
*   buf: Pointer to the statvfs structure.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
static int lfsd_statvfs(fs_mount_t* mnt, struct statvfs* buf)
{
    if (!mnt || !buf)
    {
        errno = EBADF;
        return -1;
    }

    lfs_handle_t* fs = (lfs_handle_t*)mnt->handle;

    buf->f_bsize = fs->cfg.block_size;
    buf->f_blocks = fs->cfg.block_count;
    buf->f_ublocks = lfs_fs_size(&fs->lfs);
    buf->block_cycles = fs->cfg.block_cycles;
    buf->cache_size = fs->cfg.cache_size;
    buf->lookahead_size = fs->cfg.lookahead_size;
    return 0;
}

/******************************************************************************
* Function Name: lfs_load_driver
********************************************************************************
* Summary:
*  Loads the littlefs driver and registers it with the filesystem.
*
* Parameters:
*   None
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
int lfs_load_driver(void)
{
    fs_driver_t* drv = sys_malloc(sizeof(fs_driver_t));
    memset(drv, 0, sizeof(fs_driver_t));
    memcpy(drv->fs_type, "lfs", 4);

    drv->format = lfsd_format;
    drv->mount = lfsd_mount;
    drv->fs_print_usage = lfsd_print_usage;
    drv->umount = lfsd_umount;
    drv->open = lfsd_open;
    drv->close = lfsd_close;
    drv->lseek = lfsd_lseek;
    drv->read = lfsd_read;
    drv->write = lfsd_write;
    drv->stat = lfsd_stat;
    drv->remove = lfsd_remove;
    drv->mkdir = lfsd_mkdir;
    drv->rename = lfsd_rename;
    drv->opendir = lfsd_opendir;
    drv->readdir = lfsd_readdir;
    drv->rewinddir = lfsd_rewinddir;
    drv->closedir = lfsd_closedir;
    drv->statvfs = lfsd_statvfs;

    return fs_add_driver(drv);
}

static void set_errno_from_lfs(int lfs_err)
{
    switch (lfs_err)
    {
        case LFS_ERR_OK: /* here was the bug */
        {
            /*  Do not set errno on success */
            break;
        }

        case LFS_ERR_IO:
        {
            errno = EIO;
            break;
        }
        case LFS_ERR_CORRUPT:
        {
            errno = EIO;  /* No direct match, EIO used as a general I/O error */
            break;
        }
        case LFS_ERR_NOENT:
        {
            errno = ENOENT;
            break;
        }
        case LFS_ERR_EXIST:
        {
            errno = EEXIST;
            break;
        }
        case LFS_ERR_NOTDIR:
        {
            errno = ENOTDIR;
            break;
        }
        case LFS_ERR_ISDIR:
        {
            errno = EISDIR;
            break;
        }
        case LFS_ERR_NOTEMPTY:
        {
            errno = ENOTEMPTY;
            break;
        }
        case LFS_ERR_BADF:
        {
            errno = EBADF;
            break;
        }
        case LFS_ERR_FBIG:
        {
            errno = EFBIG;
            break;
        }
        case LFS_ERR_INVAL:
        {
            errno = EINVAL;
            break;
        }
        case LFS_ERR_NOSPC:
        {
            errno = ENOSPC;
            break;
        }
        case LFS_ERR_NOMEM:
        {
            errno = ENOMEM;
            break;
        }
        case LFS_ERR_NOATTR:
        {
            errno = ENODATA;  /* Use ENODATA for missing attribute */
            break;
        }
        case LFS_ERR_NAMETOOLONG:
        {
            errno = ENAMETOOLONG;
            break;
        }
        default:
        {
            errno = EINVAL;  /* Default to invalid argument for unknown errors */
            break;
        }
    }
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */
