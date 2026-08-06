/******************************************************************************
* File Name: sh.c
*
* Description: Implementation of the shell.
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
#include <ctype.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <limits.h>
#include <stdlib.h>
#include <errno.h>
#include "../process.h"
#include "../lib/getopt.h"
#include "../fs/fs.h"
#include "../lib/readline.h"
#include "../lib/cursor.h"
#include "../../services.h"
#include "../../heap.h"

/*******************************************************************************
* Types
*******************************************************************************/

typedef enum {
    MODE_UNSPECIFIED     = 0,
    MODE_RUN_INTERACTIVE = 1,
    MODE_RUN_FILE        = 2,
    MODE_RUN_DIRECTORY   = 3
} sh_mode_t;
/*******************************************************************************
* Macros
*******************************************************************************/
#define MAX_HISTORY 10
#define MAX_LINE_LENGTH 255

#define COMMAND_ENTER 0
#define COMMAND_HISTORY_UP 1
#define COMMAND_HISTORY_DOWN 2

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: trim_last_amp
********************************************************************************
* Summary:
*   Trims the last ampersand '&' character from the input string if it exists.
*
* Parameters:
*   str: The input string to be trimmed.
*
* Return:
*   true if the last ampersand was found and removed, otherwise false.
*
*******************************************************************************/
static bool trim_last_amp(char* str) 
{
    int len = strlen(str);
    for (int i = len - 1; i >= 0; i--) 
    {
        if (str[i] == '&')
        {
            str[i] = '\0';
            return true;
        }

        if (isspace((int)str[i]))
        {
            continue;
        }
        else
        {
            break;
        }

    }
    return false;
}

/******************************************************************************
* Function Name: ansi_code
********************************************************************************
* Summary:
*   Handles ANSI escape codes for command history navigation.
* Parameters:
*   rl: The readline context.
*   ansi: The ANSI code structure.
*
* Return:
*   COMMAND_HISTORY_UP if the up arrow key is pressed,
*   COMMAND_HISTORY_DOWN if the down arrow key is pressed,
*   0 otherwise.
*
*******************************************************************************/
static int ansi_code(readline_t* rl, ansi_code_t* ansi) 
{
    if (ansi->final == FINAL_CURSOR_UP)
    {
        return COMMAND_HISTORY_UP;
    }
    else if (ansi->final == FINAL_CURSOR_DOWN)
    {
        return COMMAND_HISTORY_DOWN;
    }

    return 0;
}

/******************************************************************************
* Function Name: shell_strdup
********************************************************************************
* Summary:
*   Duplicates a string using shell_malloc.
* Parameters:
*   str: The input string to be duplicated.
*
* Return:
*   A pointer to the duplicated string, or NULL if memory allocation fails.
*
*******************************************************************************/
static char* shell_strdup(const char* str) 
{
    int len = strlen(str);
    char* copy = shell_malloc(len + 1);
    if (!copy) 
    {
        return NULL;
    }
    memcpy(copy, str, len);
    copy[len] = '\0';
    return copy;
}

/******************************************************************************
* Function Name: is_cmd
********************************************************************************
* Summary:
*   Checks if the input string matches the given command.
* Parameters:
*   cmd: The command to compare.
*   str: The input string to be checked.
*
* Return:
*   true if the input string matches the command, otherwise false.
*
*******************************************************************************/
static bool is_cmd(const char* cmd, const char* str) 
{
    while (isspace((int)*str)) str++;    
    while (*cmd != '\0') 
    {
        if (*cmd++ != *str++)
        {
            return false;
        }

    }
    return *str == '\0' || *str == ' ';
}

/******************************************************************************
* Function Name: is_valid_cmd_str
********************************************************************************
* Summary:
*   Validates that every character in the string is a printable ASCII character
*   or a tab, ensuring tainted file-read data is safe to pass as a command.
* Parameters:
*   str: The null-terminated string to validate.
* Return:
*   true if all characters are within a permissible range, otherwise false.
*
*******************************************************************************/
static bool is_valid_cmd_str(const char* str)
{
    while (*str != '\0')
    {
        unsigned char c = (unsigned char)*str;
        
        /* Check if character is not printable and not a tab */
        if (c < 0x80 && !isprint(c) && c != '\t')
        {
            return false;
        }
        str++;
    }
    return true;
}

/******************************************************************************
* Function Name: run_script
********************************************************************************
* Summary:
*   Executes a script file line by line.
* Parameters:
*   path: The path to the script file.
* Return:
*   0 if the script executed successfully, otherwise -1.
*
*******************************************************************************/
static int run_script(const char* path) 
{
    char buffer[128];
    char line[256];
    size_t line_index = 0;

    FILE* file = fopen(path, "r");
    if (file == NULL) 
    {
        printf("Failed to open file. %s\n", strerror(errno));
        return -1;
    }

    while (true) 
    {
        size_t read_size = fread(buffer, 1, sizeof(buffer), file);

        if (ferror(file)) 
        {
            printf("Failed to read file. %s\n", strerror(errno));
            break;
        }

        if (read_size == 0)
        {
            break;
        }
            
        for (int i = 0; i < read_size; i++) 
        {
            char ch = buffer[i];
            if (ch == '\n') 
            {
                line[line_index] = '\0';

                /* Trim leading whitespace */ 
                char* trimmed_line = line;
                while (*trimmed_line && isspace((unsigned char)*trimmed_line)) 
                {
                    trimmed_line++;
                }

                /* Ignore empty lines and comment lines */
                if (*trimmed_line != '\0' && *trimmed_line != '#' && *trimmed_line != '/') 
                {
                    /* Remove trailing whitespace */
                    char* end = trimmed_line + strlen(trimmed_line) - 1;
                    while (end > trimmed_line && isspace((unsigned char)*end)) 
                    {
                        *end = '\0';
                        end--;
                    }

                    /* Validate sanitized length before executing */
                    size_t cmd_len = strnlen(trimmed_line, sizeof(line));
                    if (cmd_len > 0 && cmd_len < sizeof(line))
                    {
                        /* Reject commands containing non-printable characters */
                        if (!is_valid_cmd_str(trimmed_line))
                        {
                            printf("Invalid characters in command, skipping\n");
                        }
                        else
                        {
                            /* Execute the command */
                            /* coverity[tainted_string_argument] - sanitized by is_valid_cmd_str */
                            int exec_res = process_exec_command(trimmed_line, EXEC_MODE_BLOCK);
                            if (exec_res < 0) 
                            {
                                printf("Error executing command: %s\n", trimmed_line);
                            }
                        }
                    }
                }

                line_index = 0;
            }
            else 
            {
                if (line_index < (sizeof(line) - 1)) 
                {
                    line[line_index++] = ch;
                }
                else 
                {
                    /* Line too long, reset buffer and continue */
                    printf("Line too long, skipping\n");
                    line_index = 0;
                }
            }
        }
    }

    /* Handle any remaining line after EOF */
    if (line_index > 0) 
    {
        line[line_index] = '\0';
        char* trimmed_line = line;
        while (*trimmed_line && isspace((unsigned char)*trimmed_line)) 
        {
            trimmed_line++;
        }
        if (*trimmed_line != '\0' && *trimmed_line != '#' && *trimmed_line != '/') 
        {
            /* Remove trailing whitespace */
            char* end = trimmed_line + strlen(trimmed_line) - 1;
            while (end > trimmed_line && isspace((unsigned char)*end)) 
            {
                *end = '\0';
                end--;
            }
            /* Reject commands containing non-printable characters */
            if (!is_valid_cmd_str(trimmed_line))
            {
                printf("Invalid characters in command, skipping\n");
            }
            else
            {
                /* coverity[tainted_string_argument] - sanitized by is_valid_cmd_str */
                int exec_res = process_exec_command(trimmed_line, EXEC_MODE_BLOCK);
                if (exec_res < 0) 
                {
                    printf("Error executing command: %s\n", trimmed_line);
                }
            }
        }
    }

    fclose(file);
    return 0; 
}

/******************************************************************************
* Function Name: run_directory
********************************************************************************
* Summary:
*   Executes all script files in a directory.
* Parameters:
*   path: The path to the directory.
* Return:
*   0 if the script executed successfully, otherwise -1.
*
*******************************************************************************/
static int run_directory(const char* path) 
{
    DIR* dir = opendir(path);
    if (dir == NULL) 
    {
        printf("Failed to read directory. %s\n", strerror(errno));
        return -1;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) 
    {
        if (strcmp(entry->d_name, "..") == 0 || strcmp(entry->d_name, ".") == 0)
        {
            continue;
        }

        if (entry->d_type == DT_REG) 
        {
            char full_file_path[IM_PATH_MAX + 1];
            int n = snprintf(full_file_path, sizeof(full_file_path), "%s/%s", path, entry->d_name);
            if (n < 0 || n >= (int)sizeof(full_file_path))
            {
                continue;
            }

            run_script(full_file_path);
        }
    }

    closedir(dir);
    return 0;
}

/******************************************************************************
* Function Name: run_interactive
********************************************************************************
* Summary:
*   Executes the interactive shell.
* Parameters:
*   None.
* Return:
*   0 if the shell executed successfully, otherwise -1.
*
*******************************************************************************/
static int run_interactive(void) 
{

    struct stat path_stat;
    if (stat("/system/login", &path_stat) == 0 && S_ISREG(path_stat.st_mode))
        process_exec_command("sh -e -f /system/login", EXEC_MODE_BLOCK);
   
    int history_count = 0;
    int history_cursor = 0;
    char* history[MAX_HISTORY];

    /* Initialize history */
    for (int i = 0; i < MAX_HISTORY; i++) 
    {
        history[i] = NULL;
    }

    char line[MAX_LINE_LENGTH + 1];
    line[0] = '\0';

    /* Main shell loop */
    while (1) 
    {
        cursor_set_color(COLOR_BRIGHT_GREEN, COLOR_DEFAULT);
        char cwd[IM_PATH_MAX];

        int prompt_width = printf("%s", getcwd(cwd, IM_PATH_MAX));
        cursor_set_color(COLOR_DEFAULT, COLOR_DEFAULT);
        prompt_width += printf("$ ");
        readline_t rl;
        readline_init_ex(&rl, prompt_width, 0, line, sizeof(line), strlen(line), NULL, ansi_code, false);

        switch (readline(&rl)) 
        {

            case COMMAND_ENTER:
            {
                printf("\n");

                if (strcmp(line, "exit") == 0) 
                {
                    printf("Bye!\n");

                    /* Free history items */
                    for (int i = 0; i < MAX_HISTORY; i++) 
                    {
                        if (history[i] != NULL) 
                        {
                            shell_free(history[i]);
                        }
                    }
                    return 0;
                }

                if (strlen(line) > 0) 
                {

                    /* Add command to history */
                    if (history_count < MAX_HISTORY) 
                    {
                        history[history_count] = shell_strdup(line);
                        history_count++;
                    }
                    else 
                    {
                        /* Shift history up and add the new command at the end */
                        /* Free the oldest entry */
                        shell_free(history[0]); 
                        for (int i = 1; i < MAX_HISTORY; i++) 
                        {
                            history[i - 1] = history[i];
                        }
                        history[MAX_HISTORY - 1] = shell_strdup(line);
                    }
                    history_cursor = history_count;

                    if (is_cmd("cd", line)) 
                    {
                        /* Commands that modifies the current process state, must run in the current process. */
                        process_exec_command(line, EXEC_MODE_BLOCK);
                    }
                    else 
                    {
                        bool run_in_background = trim_last_amp(line);
                        process_exec_command(line, run_in_background ? EXEC_MODE_BACKGROUND : EXEC_MODE_BLOCK_ABORT);
                    }

                    /* Clear line for the next command */
                    line[0] = '\0';
                }
                break;
            }

            case COMMAND_HISTORY_UP:
            {
                if (history_count > 0 && history_cursor > 0) 
                {
                    history_cursor--;
                    strncpy(line, history[history_cursor], sizeof(line) - 1);
                    line[sizeof(line) - 1] = '\0';
                }
                break;
            }

            case COMMAND_HISTORY_DOWN:
            {
                if (history_count > 0 && history_cursor < history_count - 1) 
                {
                    history_cursor++;
                    strncpy(line, history[history_cursor], sizeof(line) - 1);
                    line[sizeof(line) - 1] = '\0';
                }
                else if (history_cursor == history_count - 1) 
                {
                    history_cursor++;
                    line[0] = '\0';
                }
                break; 
            }
            
        }
    };
}

/******************************************************************************
* Function Name: print_error_help
********************************************************************************
* Summary:
*   Prints the error message and usage help.
* Parameters:
*   error - The error message to print.
* Return:
*   None.
*
*******************************************************************************/
static void print_error_help(char* error) 
{
    if(error != NULL)
    {
        printf("Error: %s\n", error);
    }

    const char* usage_text = "Usage: sh [options]\n"
        "Run commands from a file or directory.\n"
        "  -i              Interactive prompt\n"
        "  -f <file>       Run script file\n"
        "  -d <directory>  Run all files in directory\n"
        "  -e              Skip file or directory if it doesn't exists\n"
        "  -h              Display this help and exit\n";

    printf(usage_text);
}

/******************************************************************************
* Function Name: main_sh
********************************************************************************
* Summary:
*   The main function for the shell.
* Parameters:
*   argc - The number of command-line arguments.
*   argv - The array of command-line arguments.
* Return:
*   0 if the shell executed successfully, otherwise -1.
*
*******************************************************************************/
int main_sh(int argc, char** argv) 
{

    getopt_t go;
    getopt_init(&go);
    bool skip_missing = false;
    sh_mode_t mode = MODE_UNSPECIFIED;
    char* path = NULL;

    struct stat path_stat;
    char full_path[IM_PATH_MAX + 1];
    full_path[0] = '\0';

    int opt;
    while ((opt = getopte(&go, argc, argv, "if:d:eh")) != -1) 
    {
        switch (opt) 
        {
            case 'i':
            {
                if (mode != MODE_UNSPECIFIED) 
                {
                    print_error_help("Option -i can't be used with -f or -d");
                    return -1;
                }
                mode = MODE_RUN_INTERACTIVE;
                break;
            }

            case 'f':
            {
                if (mode != MODE_UNSPECIFIED) 
                {
                    print_error_help("Option -f can't be used with -i or -d");
                    return -1;
                }
                mode = MODE_RUN_FILE;
                path = go.optarg;
                break;                
            }
        
            case 'd':
            {
                if (mode != MODE_UNSPECIFIED) 
                {
                    print_error_help("Option -d can't be used with -f or -i");
                    return -1;
                }
                mode = MODE_RUN_DIRECTORY;
                path = go.optarg;
                break;
            }

            case 'e':
            {
                skip_missing = true;
                break;
            }

            case 'h':
            {
                print_error_help(NULL);
                return 0;
            }

            default:
            {
                print_error_help(NULL);
                return -1;
            }

        }
    }

    if (mode == MODE_UNSPECIFIED) 
    {
        print_error_help("One of options -i, -f or -d must be given");
        return -1;
    }

    if(mode == MODE_RUN_INTERACTIVE && skip_missing) 
    {
        print_error_help("Option -i can't be used with -e");
        return -1;
    }

    if (go.optind < argc) 
    {
        print_error_help("Too many arguments\n");
        return -1;
    }

    if (path != NULL) 
    {
        realpath(path, full_path);           
        if (stat(full_path, &path_stat) != 0) 
        {
            if (!skip_missing) 
            {
                printf("The path '%s' does not exist.\n", full_path);
                return -1;
            }
            return 0;
        }
    }
   
    switch (mode) 
    {
        case MODE_RUN_INTERACTIVE:
        {
            return run_interactive();
        }

        case MODE_RUN_FILE:
        {
            printf("Running script %s\n", full_path);
            return run_script(full_path);
            break;
        }

        case MODE_RUN_DIRECTORY:
        {
            printf("Running all scripts in folder %s\n", full_path);
            return run_directory(full_path);
        }

        default:
        {
            return -1;
        }

    }
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */