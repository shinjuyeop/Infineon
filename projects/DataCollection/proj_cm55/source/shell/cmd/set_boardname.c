/******************************************************************************
* File Name: set_boardname.c
*
* Description: Implementation of the 'set_boardname' command for the shell.
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
#include "../process.h"
#include "../lib/getopt.h"
#include "../../protocol/protocol.h"
#include "../../services.h"

/******************************************************************************
 * Global Variables
 *****************************************************************************/
extern char board_name[];

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: main_set_boardname
********************************************************************************
* Summary:
*   Implementation of the 'set_boardname' command for the shell.
*
* Parameters:
*   argc: Argument count.
*   argv: Argument vector.
*
* Return:
*   0 if operation is successful, otherwise -1.
*
*******************************************************************************/
int main_set_boardname(int argc, char** argv) 
{
    const char* usage_text = "Usage: %s boardname\n"
                             "Renames the board.\n";

    if ( argc == 2)
    {

        if (strcmp(argv[1], "-h") == 0)
        {
            printf(usage_text, argv[0] );
            return 0;    
        }

        if (argv[1][0] == '-')
        {
            printf("Unknown switch:\n");
            printf(usage_text, argv[0] );
            return 0;    
        }

        strncpy(board_name, argv[1], 128 - 1);
        board_name[128 - 1] = '\0';
        
        /* Create a file on the filesystem for saving this 
        * new name. Once the protocol or anywhere in the 
        * startup, check if the file exist and if so
        * overwrite any default value
        */ 

        FILE* board = fopen("system/boardname","wb");
        if (board)
        {
            int len = strlen(board_name)+1;
            fwrite(board_name,1,len,board);
            fclose(board);
            printf("Wrote boardname to file.\n");
        }
    }
    else
    {
        printf("Missing boardname:\n");
        printf(usage_text, argv[0] );
    }

    return 0;
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */