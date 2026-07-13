/******************************************************************************
* File Name: passwd.c
*
* Description: Implementation of the 'passwd' command for the shell.
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
#include <errno.h>
#include <unistd.h>
#include "../process.h"
#include "../lib/getopt.h"
#include "../fs/fs.h"
#include "../lib/readline.h"
#include "../lib/md5.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: main_passwd
********************************************************************************
* Summary:
*   Implementation of the 'passwd' command for the shell.
*
* Parameters:
*   argc: Argument count.
*   argv: Argument vector.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
int main_passwd(int argc, char** argv) 
{
    const char* usage_text = "Usage: %s\n"
                             "Set or change the shell password.\n"
                             "  -h   Display this help and exit\n";

    getopt_t go;
    getopt_init(&go);

    int opt;
    while ((opt = getopte(&go, argc, argv, "h")) != -1) 
    {
        switch (opt) 
        {
            case 'h':
                printf(usage_text, argv[0]);
                return 0;
            default:
                printf(usage_text, argv[0]);
                return -1;
        }
    }

    if (go.optind < argc) 
    {
        printf("Error: Too many arguments\n");
        printf(usage_text, argv[0]);
        return -1;
    }

    /* Prompt for the new password */ 
    char new_password[64];
    char confirm_password[64];

    new_password[0] = '\0';
    confirm_password[0] = '\0';

    readline_t rl;
    int col = printf("Enter new password: ");
    readline_init(&rl, col, 0, new_password, sizeof(new_password));
    rl.mask = true;
    readline(&rl);
    printf("\n");

    col = printf("Confirm new password: ");
    readline_init(&rl, col, 0, confirm_password, sizeof(confirm_password));
    rl.mask = true;
    readline(&rl);
    printf("\n");

    /* Check if passwords match */
    if (strcmp(new_password, confirm_password) != 0) 
    {
        printf("Passwords do not match. Password not changed.\n");
        return -1;
    }

    const char* password_file_path = "/system/password";

    /* Check if the new password is empty */ 
    if (strlen(new_password) == 0) 
    {
        /* Remove the password file if it exists */ 
        int res = unlink(password_file_path);
        if (res < 0 && errno != ENOENT) 
        {
            printf("Failed to delete password file. %s\n", strerror(errno));
            return -1;
        }

        printf("Password removed successfully.\n");
        return 0;
    }

    /* Hash the new password using MD5 */ 
    char hashed_password[33]; /* 32 hex characters + null terminator */ 
    md5_string(new_password, hashed_password);

    /* Write the new password to the file */ 
    FILE* password_file = fopen(password_file_path, "w");
    if (!password_file) 
    {
        printf("Failed to open password file. %s\n", strerror(errno));
        return -1;
    }

    size_t written = fwrite(hashed_password, sizeof(char), strlen(hashed_password), password_file);
    if (written < strlen(hashed_password)) 
    {
        printf("Failed to write password file. %s\n", strerror(errno));
        fclose(password_file);
        return -1;
    }

    fclose(password_file);

    printf("Password changed successfully.\n");
    return 0;
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */