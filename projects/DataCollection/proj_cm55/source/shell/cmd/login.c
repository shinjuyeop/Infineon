/******************************************************************************
* File Name: login.c
*
* Description: Implementation of the 'login' command for the shell.
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
#include "../process.h"
#include "../lib/getopt.h"
#include "../lib/readline.h"
#include "../lib/md5.h"
#include "../lib/cursor.h"

/*******************************************************************************
* Macros
*******************************************************************************/
#define MAX_ATTEMPTS     3

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: check_password
********************************************************************************
* Summary:
*   Checks if the entered password matches the stored password. 
*   If the password file does not exist, it allows access without a password.
*
* Parameters:
*   limit_attempts: If true, limits the number of failed attempts to MAX_ATTEMPTS.
*
* Return:
*   true if the password is correct or not required, otherwise false.
*
*******************************************************************************/
static bool check_password(bool limit_attempts) 
{
    const char password_file_path[] = "/system/password";

    /* Open the password file directly; absence is detected via errno */
    char stored_password[64];
    FILE* password_file = fopen(password_file_path, "r");
    if (password_file == NULL) 
    {
        if (errno == ENOENT)
        {
            /* Password file does not exist. No password check needed */
            return true;
        }
        printf("Error opening password file: %s\n", strerror(errno));
        return false;
    }

    size_t password_len = fread(stored_password, 1, sizeof(stored_password) - 1, password_file);
    if (ferror(password_file)) 
    {
        printf("Error reading password file: %s\n", strerror(errno));
        fclose(password_file);
        return false;
    }
    fclose(password_file);

    /* Null-terminate the password */
    stored_password[password_len] = '\0';  
    /* Give the user 3 attempts to enter the correct password */
    char entered_password[64];
    
    for (int attempt = 1; ; attempt++) 
    {
        entered_password[0] = '\0';
        readline_t rl;
        /* Start on a clean line to avoid log message interference */
        printf("\n"); 
        int column = printf("Enter password: ");
        readline_init(&rl, column, 0, entered_password, sizeof(entered_password));
        rl.mask = true;
        readline(&rl);
        printf("\n");

        /* Hash the entered password using MD5 */
        char hashed_entered_password[33];
        md5_string(entered_password, hashed_entered_password);

        if (strcmp(stored_password, hashed_entered_password) == 0) 
        {
            return true;
        }
        else 
        {
            if (limit_attempts)
            {
                printf("Incorrect password. %d attempts remaining.\n", MAX_ATTEMPTS - attempt);
            }
            else
            {
                printf("Incorrect password.\n");
            }

        }

        if (attempt == MAX_ATTEMPTS && limit_attempts) 
        {
            printf("Too many incorrect attempts. Access denied.\n");
            return false;
        }
    }
}

/******************************************************************************
* Function Name: main_login
********************************************************************************
* Summary:
*   Implementation of the 'login' command for the shell.
*
* Parameters:
*   argc: Argument count.
*   argv: Argument vector.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
int main_login(int argc, char** argv) 
{
    const char* usage_text = "Usage: %s [arguments...]\n"
        "Display a summary of all commands.\n"
        "  -u   No limit on failed attempts\n";

    getopt_t go;
    getopt_init(&go);

    int opt;
    bool limit_attempts = true;

    while ((opt = getopte(&go, argc, argv, "hu")) != -1) 
    {
        switch (opt) 
        {
            case 'u':
                limit_attempts = false;
                break;
            default:
                printf(usage_text, argv[0]);
                return 0;
        }
    }

    /* Check if password is required and validate it */
    if (!check_password(limit_attempts)) 
    {
        return -1;
    }

    /* Start the shell */
    return process_exec_command("sh -i", EXEC_MODE_BLOCK);
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */