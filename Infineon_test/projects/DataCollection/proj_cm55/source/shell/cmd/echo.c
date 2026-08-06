/******************************************************************************
* File Name: echo.c
*
* Description: Implementation of the 'echo' command for the shell.
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
#include "../lib/getopt.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/
/******************************************************************************
* Function Name: main_echo
********************************************************************************
* Summary:
*   Implementation of the 'echo' command for the shell.
*
* Parameters:
*   argc: Argument count.
*   argv: Argument vector.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
int main_echo(int argc, char** argv) 
{
    const char* usage_text = "Usage: %s [arguments...]\n"
                             "Write arguments to the standard output.\n"
                             "  -h   Display this help and exit\n"
                             "  -n   Do not append a newline\n"
                             "  -E   Explicitly suppress interpretation of backslash escapes\n";

    getopt_t go;
    getopt_init(&go);

    int opt;
    int append_newline = 1;
    int suppress_backslash = 0;

    while ((opt = getopte(&go, argc, argv, "hnE")) != -1) 
    {
        switch (opt) 
        {
            case 'h':
                printf(usage_text, argv[0]);
                return 0;
            case 'n':
                append_newline = 0;
                break;
            case 'E':
                suppress_backslash = 1;
                break;
            default:
                printf(usage_text, argv[0]);
                return -1;
        }
    }

    for (int i = go.optind; i < argc; i++) 
    {
        const char* str = argv[i];
        while (*str) 
        {
            if (*str == '\\' && !suppress_backslash) 
            {
                str++;
                switch (*str) 
                {
                    case 'n':
                        printf("\n");
                        break;
                    case 't':
                        printf("\t");
                        break;
                    case '\\':
                        printf("\\");
                        break;
                    case '\"':
                        printf("\"");
                        break;
                    case 'r':
                        printf("\r");
                        break;
                    default:
                        printf("\\%c", *str);
                        break;
                }
            } 
            else 
            {
                printf("%c", *str);
            }
            str++;
        }

        if (i < argc - 1) 
        {
            printf(" ");
        }
    }
    if (append_newline) 
    {
        printf("\n");
    }
    return 0;
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */