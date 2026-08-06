/******************************************************************************
* File Name:   getopt.c
*
* Description: Implementation of getopt utility functions.
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
#include <stdio.h>
#include <string.h>
#include "getopt.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: getopt_init
********************************************************************************
* Summary:
*    Initializes the getopt handle.
*
* Parameters:
*   handle - Pointer to the getopt handle to initialize.
*
* Return:
*   None
*
*******************************************************************************/
void getopt_init(getopt_t* handle) 
{
    handle->optarg = NULL;
    handle->optind = 1;
    handle->opterr = 1;
    handle->optopt = 0;

    handle->first_nonopt = 1;
    handle->last_nonopt = 1;
    handle->nextchar = NULL;
}

/******************************************************************************
* Function Name: getopte
********************************************************************************
* Summary:
*    Parses command-line options.
*
* Parameters:
*   handle - Pointer to the getopt handle.
*   argc - Argument count.
*   argv - Argument vector.
*   optstring - String of valid option characters.
*
* Return:
*   The next option character, or -1 if no more options are available.
*
*******************************************************************************/
int getopte(getopt_t* handle, int argc, char* const argv[], const char* optstring) 
{
    if (handle->nextchar == NULL || *handle->nextchar == '\0') 
    {
        if (handle->optind >= argc || argv[handle->optind][0] != '-' || 
            argv[handle->optind][1] == '\0') 
        {
            return -1; /* No more options */ 
        }
        if (strcmp(argv[handle->optind], "--") == 0) 
        {
            handle->optind++;
            return -1; /* No more options */
        }
        handle->nextchar = argv[handle->optind] + 1;
        handle->optind++;
    }

    char c = *handle->nextchar++;
    const char* opt = strchr(optstring, c);

    if (opt == NULL || c == ':') 
    {
        handle->optopt = c;
        if (handle->opterr) 
        {
            fprintf(stderr, "Unknown option `-%c'.\n", c);
        }
        return '?';
    }

    if (opt[1] == ':') 
    {
        if (*handle->nextchar != '\0') 
        {
            handle->optarg = handle->nextchar;
            handle->nextchar = NULL;
        }
        else if (handle->optind < argc) 
        {
            handle->optarg = argv[handle->optind];
            handle->optind++;
        }
        else 
        {
            handle->optopt = c;
            if (handle->opterr) 
            {
                fprintf(stderr, "Option `-%c' requires an argument.\n", c);
            }
            return '?';
        }
    }

    return c;
}

/* [] END OF FILE */