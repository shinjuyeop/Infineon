/******************************************************************************
* File Name:   fs.h
*
* Description: Header file for the filesystem abstraction layer used by the shell.
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
#ifndef _FS_H_
#define _FS_H_

#ifdef IM_ENABLE_SHELL

#include <stdint.h>
#include <sys/stat.h>
#include "../wifi_configs/im_config.h"

/******************************************************************************
 * Macros
 *****************************************************************************/
#define DT_DIR     2
#define DT_REG     1
#define DT_UNKNOWN 0

/******************************************************************************
 * Types
 *****************************************************************************/
typedef void* fs_file_p;

typedef struct fs_driver_s fs_driver_t;

typedef struct fs_mount_s {
    char location[64];
    void* handle;
    fs_driver_t* driver;
} fs_mount_t;

struct dirent {
    char d_name[IM_PATH_MAX];
    unsigned char d_type;
    unsigned int d_reclen;
};

typedef struct {
    void* dir;
    struct dirent entry;
    fs_mount_t* mnt;
} DIR;

struct statvfs {
    uint32_t  f_bsize;          /* Size of a logical block in bytes. */
    int64_t  f_blocks;          /* Number of logical blocks in filesystem. */
    int64_t  f_ublocks;         /* Best effort, number of allocated blocks. */
    int32_t cache_size;         /* Size of the cache in bytes. */
    int64_t block_cycles;       /* Number of block cycles. */
    int32_t lookahead_size;     /* Size of the lookahead buffer. */
};

struct fs_driver_s {
    char fs_type[16];
   
    int (*format)(int argc, const char** argv);
    int (*mount)(fs_mount_t* mnt, int argc, const char** argv);
    void (*fs_print_usage)(void);
    int (*umount)(fs_mount_t* mnt);
    fs_file_p (*open)(fs_mount_t* mnt, const char* pathname, int flags, int mode);
    int (*close)(fs_mount_t* mnt, fs_file_p file);
    _off_t (*lseek)(fs_mount_t* mnt, fs_file_p file, off_t offset, int whence);
    _ssize_t (*read)(fs_mount_t* mnt, fs_file_p file, void* buf, size_t count);
    _ssize_t (*write)(fs_mount_t* mnt, fs_file_p file, const void* buf, size_t count);
    int (*stat)(fs_mount_t* mnt, const char* pathname, struct stat* buf);
    int (*remove)(fs_mount_t* mnt, const char* pathname);
    int (*mkdir)(fs_mount_t* mnt, const char* pathname, int mode);
    int (*rename)(fs_mount_t* mnt, const char* oldpath, const char* newpath);
    DIR* (*opendir)(fs_mount_t* mnt, const char* name);
    struct dirent* (*readdir)(fs_mount_t* mnt, DIR* dirp);
    int (*rewinddir)(fs_mount_t* mnt, DIR* dirp);
    int (*closedir)(fs_mount_t* mnt, DIR* dirp);
    int (*statvfs)(fs_mount_t* mnt, struct statvfs* buf);
};

/*******************************************************************************
* Function Prototypes
*******************************************************************************/
int fs_add_driver(fs_driver_t* driver);

int fs_format(const char* fs_type, int argc, const char** argv);

int fs_mount(const char* abspath, const char* fs_type, int argc, const char** argv);

void fs_print_usage(void);

int fs_umount(const char* abspath);

int fs_open(const char* abspath, int flags, int mode);

int fs_close(int fd);

_off_t fs_lseek(int fd, off_t offset, int whence);

_ssize_t fs_read(int fd, void* buf, size_t count);

_ssize_t fs_write(int fd, const void* buf, size_t count);

int fs_stat(const char* abspath, struct stat* buf);

int fs_remove(const char* abspath);

int fs_mkdir(const char* abspath, int mode);

int fs_rename(const char* oldabspath, const char* newabspath);

DIR* fs_opendir(const char* abspath);

struct dirent* fs_readdir(DIR* dirp);

void fs_rewinddir(DIR* dirp);

int fs_closedir(DIR* dirp);

int fs_statvfs(const char* abspath, struct statvfs* buf);


/* Defined in romfs.c */
int fs_extract_romfs(const char* output_dir);

/* Missing in stdlib.h ? Defined in fs_until.c */
char* realpath(const char* path, char* resolved_path);

/* Defined in fs_until.c */
bool file_exist(const char* file);

#endif /* IM_ENABLE_SHELL */
#endif /* _FS_H_ */

/* [] END OF FILE */
