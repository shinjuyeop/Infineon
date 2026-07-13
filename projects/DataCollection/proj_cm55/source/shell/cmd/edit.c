/******************************************************************************
* File Name: edit.c
*
* Description: Implementation of the 'edit' command for the shell.
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
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#include <errno.h>
#include "../process.h"
#include "../lib/readline.h"
#include "../../heap.h"
#include "../lib/cursor.h"
#include "../lib/getopt.h"
#include "../../services.h"

/*******************************************************************************
* Macros
*******************************************************************************/
#define COMMAND_NEW_LINE              0
#define COMMAND_MOVE_UP               1
#define COMMAND_MOVE_DOWN             2
#define COMMAND_EXIT                  3
#define COMMAND_SAVE                  4
#define COMMAND_DELETE_LINE           5
#define COMMAND_HELP                  6
#define COMMAND_GOTO_END_PREV_LINE    7
#define COMMAND_GOTO_BEGIN_NEXT_LINE  8

#define STATUS_MESSAGE_MAX   80

/*******************************************************************************
* Types
*******************************************************************************/
typedef struct line {
    char *data;
    size_t size;
    struct line *next;
    struct line *prev;
} line_t;

/* Structure representing the editor state. */
typedef struct {
    readline_t readline;           /**< Readline structure for handling user input. */
    size_t line_count;             /**< Total number of lines in the editor. */
    line_t *current_line;          /**< Pointer to the current line being edited. */
    line_t *first;                 /**< Pointer to the first line in the editor. */
    line_t *last;                  /**< Pointer to the last line in the editor. */
    const char* filename;          /**< Name of the file being edited. */
    int screen_cols;               /**< Number of columns in the terminal screen. */
    int screen_rows;               /**< Number of rows in the terminal screen. */
    int cursor_line;               /**< Current line position of the cursor. */
    int display_line;              /**< First line displayed on the screen. */
    char status_msg[STATUS_MESSAGE_MAX]; /**< Status message to be displayed. */
    int desired_cursor_index;      /**< Desired cursor position within the line. */
    bool is_modified;              /**< Flag indicating if the file has been modified. */
} editor_t;

/*******************************************************************************
* Function Definitions
*******************************************************************************/
/******************************************************************************
* Function Name: shell_strdup
********************************************************************************
* Summary:
*   Duplicates a string using shell_malloc.
*
* Parameters:
*   str: The string to duplicate.
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
* Function Name: ansi_code
********************************************************************************
* Summary:
*   Determines the appropriate command based on the ANSI code received.
*
* Parameters:
*   rl: Pointer to the readline structure.
*   ansi: Pointer to the ANSI code structure.
*
* Return:
*   An integer command code.
*
*******************************************************************************/
static int ansi_code(readline_t* rl, ansi_code_t* ansi) 
{
    if(ansi->final == FINAL_CURSOR_UP)
    {
        return COMMAND_MOVE_UP;
    }
    else if(ansi->final == FINAL_CURSOR_DOWN)
    {
        return COMMAND_MOVE_DOWN;
    }
    else if(ansi->final == FINAL_CURSOR_BACKWARD && rl->buffer_cursor == 0)
    {
        return COMMAND_GOTO_END_PREV_LINE;
    }
    else if(ansi->final == FINAL_CURSOR_FORWARD && rl->buffer_cursor == strlen(rl->buffer))
    {
        return COMMAND_GOTO_BEGIN_NEXT_LINE;
    }

    return 0;
}

/******************************************************************************
* Function Name: key_press
********************************************************************************
* Summary:
*   Handles control key presses in the readline.
*
* Parameters:
*   rl: Pointer to the readline structure.
*   key: The key that was pressed.
*
* Return:
*   An integer command code.
*
*******************************************************************************/
static int key_press(readline_t* rl, char key) 
{
    switch (key) 
    {
        case 3: /* Ctrl-C */
            return COMMAND_EXIT;
        case 19: /* Ctrl-S */
            return COMMAND_SAVE;
        case 7: /* Ctrl-G */
            return COMMAND_HELP;
        case 127: /* Backspace */
            if (rl->buffer_cursor == 0) 
            {
                return COMMAND_DELETE_LINE;
            }
            return 0;
        default:
            return 0;
    }
}

/******************************************************************************
* Function Name: initialize_readline
********************************************************************************
* Summary:
*   Initializes the readline structure for a given line.
*
* Parameters:
*   rl: Pointer to the readline structure.
*   current_line: Pointer to the current line structure.
*   cursor_index: Initial cursor position within the line.
*
* Return:
*   None.
*
*******************************************************************************/
static void initialize_readline(readline_t* rl, line_t* current_line, int cursor_index) 
{
    readline_init_ex(
        rl,
        0,
        0,
        current_line->data,
        current_line->size,
        cursor_index,
        key_press,
        ansi_code,
        true
    );
}

/******************************************************************************
* Function Name: create_line_ex
********************************************************************************
* Summary:
*   Creates a new line structure with the given content.
*
* Parameters:
*   content: The content of the new line. Must be allocated with shell_malloc (or shell_realloc).
*   len: The length of the content.
*
* Return:
*   A pointer to the new line structure, or NULL if memory allocation fails.
*
*******************************************************************************/
static line_t* create_line_ex(char* content, int len) 
{
    line_t* line = (line_t*)shell_malloc(sizeof(line_t));
    if (!line) 
    {
        return NULL;
    }
    line->size = len + 1;
    line->data = content;
    line->next = NULL;
    line->prev = NULL;
    return line;
}

/******************************************************************************
* Function Name: create_line
********************************************************************************
* Summary:
*   Creates a new line structure with the given content.
*
* Parameters:
*   content: The content of the new line. Must be allocated with shell_malloc 
*            (or shell_realloc).
*
* Return:
*   A pointer to the new line structure, or NULL if memory allocation fails.
*
*******************************************************************************/
static line_t* create_line(const char* content) 
{
    char* copy = shell_strdup(content);
    return create_line_ex(copy, strlen(copy));
}

/******************************************************************************
* Function Name: insert_line
********************************************************************************
* Summary:
*   Inserts a new line into the editor buffer after the given line.
*
* Parameters:
*   editor    : Pointer to the editor structure.
*   prev_line : Pointer to the line after which the new line should be inserted.
*   new_line  : Pointer to the new line to be inserted.
*
* Return:
*   None.
*
*******************************************************************************/
static void insert_line(editor_t* editor, line_t* prev_line, line_t* new_line) 
{
    if (!prev_line) 
    {
        new_line->next = editor->first;
        if (editor->first) 
        {
            editor->first->prev = new_line;
        }
        editor->first = new_line;
        if (!editor->last) 
        {
            editor->last = new_line;
        }
    } 
    else 
    {
        new_line->next = prev_line->next;
        new_line->prev = prev_line;
        if (prev_line->next) 
        {
            prev_line->next->prev = new_line;
        } 
        else 
        {
            editor->last = new_line;
        }
        prev_line->next = new_line;
    }
    editor->line_count++;
}

/******************************************************************************
* Function Name: add_line
********************************************************************************
* Summary:
*   Adds a new line to the end of the editor buffer.
*
* Parameters:
*   editor    : Pointer to the editor structure.
*   new_line  : Pointer to the new line to be added.
*
* Return:
*   None.
*
*******************************************************************************/
static void add_line(editor_t* editor, line_t* new_line) 
{
    if(editor->first == NULL) 
    {
        editor->first = new_line;
        editor->last = new_line;
        editor->current_line = new_line;
    } 
    else 
    {
        new_line->prev = editor->last;
        editor->last->next = new_line;
        editor->last = new_line;
    }
    editor->line_count++;
}

/******************************************************************************
* Function Name: delete_line
********************************************************************************
* Summary:
*   Deletes a line from the editor buffer.
*
* Parameters:
*   editor    : Pointer to the editor structure.
*   line      : Pointer to the line to be deleted.
*
* Return:
*   None.
*
*******************************************************************************/
static void delete_line(editor_t* editor, line_t* line) 
{
    if (line->prev) 
    {
        line->prev->next = line->next;
    } 
    else 
    {
        editor->first = line->next;
    }

    if (line->next) 
    {
        line->next->prev = line->prev;
    } 
    else 
    {
        editor->last = line->prev;
    }
    shell_free(line->data);
    shell_free(line);

    /* Ensure the line counter does not underflow */
    if (editor->line_count > 0)
    {
        editor->line_count--;
    }
}

/******************************************************************************
* Function Name: merge_with_previous_line
********************************************************************************
* Summary:
*   Merges the current line with the previous line.
*
* Parameters:
*   editor    : Pointer to the editor structure.
*
* Return:
*   None.
*
*******************************************************************************/
static void merge_with_previous_line(editor_t* editor) 
{
    line_t* current = editor->current_line;
    line_t* prev = current->prev;

    if (!prev) 
    {
        return;
    }
    int prev_line_len = strlen(prev->data);
    size_t new_size = prev->size + current->size - 1;
    prev->data = shell_realloc(prev->data, new_size);
    strcat(prev->data, current->data);
    prev->size = new_size;

    delete_line(editor, current);
    editor->current_line = prev;
    editor->readline.buffer_cursor = prev_line_len;
    editor->desired_cursor_index = prev_line_len;
}

/******************************************************************************
* Function Name: insert_new_line
********************************************************************************
* Summary:
*   Inserts a new line at the cursor position.
*
* Parameters:
*   editor    : Pointer to the editor structure.
*
* Return:
*   None.
*
*******************************************************************************/
static void insert_new_line(editor_t* editor) 
{
    line_t* current = editor->current_line;
    size_t cursor_pos = editor->readline.buffer_cursor;

    char* new_line_data = shell_strdup(current->data + cursor_pos);
    if (!new_line_data) 
    {
        return;
    }

    current->data[cursor_pos] = '\0';
    current->size = cursor_pos + 1;

    line_t* new_line = create_line(new_line_data);
    shell_free(new_line_data);

    if (!new_line) 
    {
        return;
    }

    insert_line(editor, current, new_line);
    editor->current_line = new_line;

    initialize_readline(&editor->readline, editor->current_line, 0);
    editor->readline.buffer_cursor = 0;

    editor->cursor_line++;
    editor->is_modified = true;

    /* Scroll if necessary */
    if (editor->cursor_line >= editor->display_line + editor->screen_rows - 1) 
    {
        editor->display_line++;
    }
}

/******************************************************************************
* Function Name: print_status_line
********************************************************************************
* Summary:
*   Prints the status line at the bottom of the screen.
*
* Parameters:
*   editor    : Pointer to the editor structure.
*
* Return:
*   None.
*
*******************************************************************************/
static void print_status_line(editor_t* editor) 
{
    cursor_set_cursor_position(editor->screen_rows, 1);
    cursor_set_color(COLOR_WHITE, COLOR_BLUE);
    char status[editor->screen_cols + 1];
    snprintf(status, editor->screen_cols + 1, "File: %s | Line: %d/%d   %s", editor->filename, editor->cursor_line + 1, editor->line_count, editor->status_msg);
    printf("%s", status);
    for (int i = strlen(status); i < editor->screen_cols; i++) 
    {
        printf(" ");
    }
    cursor_reset_attributes();
}

/******************************************************************************
* Function Name: print_all_lines
********************************************************************************
* Summary:
*   Prints all lines in the editor buffer to the screen.
*
* Parameters:
*   editor    : Pointer to the editor structure.
*
* Return:
*   None.
*
*******************************************************************************/
static void print_all_lines(editor_t* editor) 
{
    cursor_clear_screen();
    cursor_set_cursor_column(0);

    int row = 0;
    int start_row = editor->display_line;

    line_t* line = editor->first;
    for (int i = 0; i < start_row; i++) 
    {
        if (line && line->next) 
        {
            line = line->next;
        }
    }

    int terminal_width = cursor_terminal_width();
    for (line_t* l = line; l; l = l->next) 
    {
        int line_len = strlen(l->data);
        if(line_len >= terminal_width) 
        {
            line_len = terminal_width;
            printf("%.*s\n", line_len, l->data);
        } 
        else 
        {
            printf("%s\n", l->data);
        }
        if (++row >= editor->screen_rows - 1) 
        {
            break;
        }
    }

    // Print status line
    print_status_line(editor);

    cursor_set_cursor_column(1);
}

/******************************************************************************
* Function Name: load_file
********************************************************************************
* Summary:
*   Loads a file into the editor buffer.
*
* Parameters:
*   editor    : Pointer to the editor structure.
*   filename  : The name of the file to load.
*
* Return:
*   None.
*
*******************************************************************************/
static void load_file(editor_t* editor, const char* filename) 
{
    char segment[512];
    int read_size;
    char* line_build = NULL;
    int line_build_len = 0;
    char full_path[IM_PATH_MAX];

    if (realpath(filename, full_path) == NULL) 
    {
        snprintf(editor->status_msg, STATUS_MESSAGE_MAX, "Error resolving full path");
        return;
    }

    FILE* file = fopen(full_path, "rb");
    if (!file) 
    {
        if (errno == ENOENT) 
        {
            snprintf(editor->status_msg, STATUS_MESSAGE_MAX, "New file buffer");
        }
        else 
        {
            snprintf(editor->status_msg, STATUS_MESSAGE_MAX, "Error loading file: %s", strerror(errno));
        }
        return;
    }

    while ((read_size = fread(segment, 1, sizeof(segment), file)) > 0) 
    {
        int segment_start = -1;
        int segment_len = 0;
        for (int i = 0; i < read_size; i++) 
        {

            if (segment[i] == '\r')
                segment[i] = '\0';

            /* Ignore \0 */
            if (segment[i] == '\0')
                continue;
            if (segment_start == -1)
                segment_start = i;

            if (segment[i] == '\n') {
                segment[i] = '\0';

                if (line_build_len == 0) 
                {
                    /* Create a new single line from the segment */
                    add_line(editor, create_line(segment + segment_start));
                }
                else 
                {
                    /* Append the segment to the existing line build */
                    line_build = shell_realloc(line_build, line_build_len + segment_len + 1);
                    if (!line_build) 
                    {
                        snprintf(editor->status_msg, STATUS_MESSAGE_MAX, "Memory allocation error");
                        fclose(file);
                        return;
                    }
                    memcpy(line_build + line_build_len, segment + segment_start, segment_len);
                    line_build_len += segment_len;
                    line_build[line_build_len] = '\0';

                    /* Add the completed line to the editor */
                    add_line(editor, create_line_ex(line_build, line_build_len));
                    line_build_len = 0;
                    line_build = NULL;
                }
                segment_start = -1;
                segment_len = 0;
            } 
            else 
            {
                segment_len++;
            }
        }

        if (segment_start != -1 && segment_len > 0) 
        {
            /* Append the remaining segment to the line build */
            line_build = shell_realloc(line_build, line_build_len + segment_len + 1);
            if (!line_build) 
            {
                snprintf(editor->status_msg, STATUS_MESSAGE_MAX, "Memory allocation error");
                fclose(file);
                return;
            }
            memcpy(line_build + line_build_len, segment + segment_start, segment_len);
            line_build_len += segment_len;
        }
    }

    if (line_build_len > 0) 
    {
        /* Handle the case where the file does not end with a newline */
        line_build[line_build_len] = '\0';
        add_line(editor, create_line_ex(line_build, line_build_len));
    }

    fclose(file);

    editor->is_modified = false;
}

/******************************************************************************
* Function Name: save_file
********************************************************************************
* Summary:
*   Saves the editor buffer to a file.
*
* Parameters:
*   editor: Pointer to the editor structure.
*   filename: The name of the file to save.
*
* Return:
*   None.
*
*******************************************************************************/
static void save_file(editor_t* editor, const char* filename) 
{  
    snprintf(editor->status_msg, STATUS_MESSAGE_MAX, "Saving file, please wait...");
    print_status_line(editor);

    FILE* file = fopen(filename, "wb");
    if (!file) {
        snprintf(editor->status_msg, STATUS_MESSAGE_MAX, "Error opening file: %s", strerror(errno));
        print_status_line(editor);
        return;
    }

    for (line_t* line = editor->first; line; line = line->next) 
    {
        int len = strlen(line->data);
        size_t written = fwrite(line->data, 1, len, file);
        if (written < len) 
        {
            snprintf(editor->status_msg, STATUS_MESSAGE_MAX, "Error writing to file: %s", strerror(errno));
            print_status_line(editor);
            fclose(file);
            return;
        }
        if (fwrite("\n", 1, 1, file) < 1) 
        {
            snprintf(editor->status_msg, STATUS_MESSAGE_MAX, "Error writing newline to file: %s", strerror(errno));
            print_status_line(editor);
            fclose(file);
            return;
        }
    }

    snprintf(editor->status_msg, STATUS_MESSAGE_MAX, "File saved successfully");
    print_status_line(editor);
    fclose(file);
    editor->is_modified = false;
}

/******************************************************************************
* Function Name: prompt_save_changes
********************************************************************************
* Summary:
*   Prompts the user to save changes if the document has been modified.
*
* Parameters:
*   editor: Pointer to the editor structure.
*
* Return:
*   1 if the editor should exit, 0 otherwise.
*
*******************************************************************************/
static int prompt_save_changes(editor_t* editor) 
{
    if (!editor->is_modified) 
    {
        /* No modifications, can exit */
        return 1; 
    }

    snprintf(editor->status_msg, STATUS_MESSAGE_MAX, "Save changes before exiting? (y/n)");
    print_status_line(editor);

    while(1) 
    {
        int c = getchar();
        if (c == EOF)
        {
            return 0;
        }
        if (c == 'y' || c == 'Y') 
        {
            save_file(editor, editor->filename);
            return 1;
        } 
        else if (c == 'n' || c == 'N') 
        {
            return 1;
        } 
        else 
        {
            snprintf(editor->status_msg, STATUS_MESSAGE_MAX, "Invalid input. Use 'y' or 'n'.");
            print_status_line(editor);
            /* Wait for another key press to continue */
            (void)getchar();
            return 0;
        }
    }
}

/******************************************************************************
* Function Name: show_help_screen
********************************************************************************
* Summary:
*   Displays the help screen.
*
* Parameters:
*   editor: Pointer to the editor structure.
*
* Return:
*   None.
*
*******************************************************************************/
static void show_help_screen(editor_t* editor) 
{
    cursor_clear_screen();

    const char* help_text =
        " HELP MENU\n"
        " ---------\n"
        " Ctrl-S   Save the file\n"
        " Ctrl-C   Exit the editor\n"
        " Ctrl-G   Show this help menu\n"
        " \n"
        " End      Move to end of line\n"
        " Home     Move to beginning of line\n"
        " \n"
        " Press any key to return...";

    printf("%s", help_text);

    /* Wait for any key press */
    (void)getchar();

    /* Refresh the editor screen */
    print_all_lines(editor);
    cursor_set_cursor_position(editor->cursor_line - editor->display_line + 1, editor->readline.buffer_cursor + 1);
}

/******************************************************************************
* Function Name: free_all_lines
********************************************************************************
* Summary:
*   Frees all lines in the editor buffer.
*
* Parameters:
*   editor: Pointer to the editor structure.
*
* Return:
*   None.
*
*******************************************************************************/
static void free_all_lines(editor_t* editor) 
{
    line_t* current = editor->first;
    while (current) 
    {
        line_t* next = current->next;
        shell_free(current->data);
        shell_free(current);
        current = next;
    }
    editor->first = NULL;
    editor->last = NULL;
    editor->line_count = 0;
}

/******************************************************************************
* Function Name: handle_cursor
********************************************************************************
* Summary:
*   Handles cursor movement within the editor.
*
* Parameters:
*   editor: Pointer to the editor structure.
*   move_up: Flag indicating whether to move the cursor up or down.
*
* Return:
*   None.
*
*******************************************************************************/
static void handle_cursor(editor_t* editor, bool move_up) 
{
    int len = strlen(editor->current_line->data);
    if(editor->desired_cursor_index > len)
        editor->desired_cursor_index = len;

    initialize_readline(&editor->readline, editor->current_line, editor->desired_cursor_index);

    if(move_up) 
    {
        editor->cursor_line--;
        if (editor->cursor_line < editor->display_line)
            editor->display_line--;
    } 
    else 
    {
        editor->cursor_line++;
        if (editor->cursor_line >= editor->display_line + editor->screen_rows - 1)
        {
                editor->display_line++;
        }

    }
}

/******************************************************************************
* Function Name: main_edit
********************************************************************************
* Summary:
*   Implementation of the 'edit' command for the shell.
*
* Parameters:
*   argc: Argument count.
*   argv: Argument vector.
*
* Return:
*   None.
*
*******************************************************************************/
int main_edit(int argc, char** argv) 
{
    int opt;
    const char* usage_text = "Usage: %s <file>\n"
                             "Simple text editor.\n"
                             "  -h   Display this help and exit\n";

    getopt_t go;
    getopt_init(&go);

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

    if (go.optind >= argc) 
    {
        printf("Error: No file specified\n");
        printf(usage_text, argv[0]);
        return -1;
    }

    if (go.optind < argc - 1) 
    {
        printf("Error: Too many arguments\n");
        printf(usage_text, argv[0]);
        return -1;
    }

    const char* filename = argv[go.optind];
    editor_t editor;

    memset(&editor, 0, sizeof(editor));
    editor.filename = filename;
    editor.screen_cols = cursor_terminal_width();
    editor.screen_rows = cursor_terminal_height();
    editor.cursor_line = 0;
    editor.display_line = 0;
    editor.desired_cursor_index = 0;
    editor.is_modified = true;
    strcpy(editor.status_msg, "Press Ctrl-G for help");

    load_file(&editor, filename);
    editor.current_line = editor.first;

    if (editor.line_count == 0) 
    {
        line_t* line = create_line("");
        editor.first = line;
        editor.last = line;
        editor.current_line = line;
        editor.line_count = 1;
        editor.is_modified = false;
    }

    print_all_lines(&editor);

    /* Don't get killed by the shell when Ctrl-C is pressed. */
    process_set_attrib(NULL, PROCESS_NOEXIT);

    readline_t* rl = &editor.readline;
    initialize_readline(rl, editor.current_line, editor.desired_cursor_index);

    cursor_set_cursor_position(editor.cursor_line - editor.display_line + 1, rl->buffer_cursor + 1);

    while (1) 
    {
        int result = readline(rl);
        strcpy(editor.status_msg, "Press Ctrl-G for help");

        /* If readline has reallocated this memory */
        editor.current_line->data = rl->buffer;
        editor.current_line->size = rl->buffer_size;
        editor.desired_cursor_index = rl->buffer_cursor;
        editor.is_modified |= rl->is_modified;
        rl->is_modified = false;

        switch (result) 
        {
            case COMMAND_NEW_LINE:
                insert_new_line(&editor);
                break;

            case COMMAND_GOTO_END_PREV_LINE:
                if (editor.current_line->prev) 
                {
                    editor.current_line = editor.current_line->prev;
                    editor.desired_cursor_index = strlen(editor.current_line->data);
                    handle_cursor(&editor, true);
                }
                break;

            case COMMAND_GOTO_BEGIN_NEXT_LINE:
                if (editor.current_line->next) 
                {
                    editor.current_line = editor.current_line->next;
                    editor.desired_cursor_index = 0;
                    handle_cursor(&editor, false);
                }
                break;

            case COMMAND_MOVE_UP:
                if (editor.current_line->prev) 
                {
                    editor.current_line = editor.current_line->prev;
                    handle_cursor(&editor, true);
                }
                break;

            case COMMAND_MOVE_DOWN:
                if (editor.current_line->next) 
                {
                    editor.current_line = editor.current_line->next;
                    handle_cursor(&editor, false);
                }
                break;

            case COMMAND_DELETE_LINE:
                if (editor.line_count > 1) 
                {
                    merge_with_previous_line(&editor);
                    handle_cursor(&editor, true);
                } 
                else if (editor.line_count == 1) 
                {
                    // Handle deleting the last line in the document
                    memset(editor.current_line->data, 0, editor.current_line->size);
                    editor.current_line->size = 1;
                    rl->buffer = editor.current_line->data;
                }
                break;

            case COMMAND_HELP:
                show_help_screen(&editor);
                break;

            case COMMAND_SAVE:
                save_file(&editor, editor.filename);
                break;

            case COMMAND_EXIT:
                if (prompt_save_changes(&editor)) 
                {
                    cursor_clear_screen();
                    free_all_lines(&editor);
                    return 0;
                }
                break;
                
            default:
                break;
        }

        print_all_lines(&editor);
        cursor_set_cursor_position(editor.cursor_line - editor.display_line + 1, rl->buffer_cursor + 1);
    }
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */
