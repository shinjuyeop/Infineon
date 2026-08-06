/******************************************************************************
* File Name:   readline.c
*
* Description: Implementation of readline functionality for a shell interface, 
*              including cursor movement, line editing, and buffer handling.
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
#include <stdbool.h>
#include <limits.h>
#include "readline.h"
#include "../../heap.h"
#include "cursor.h"

extern int printk(const char* restrict format, ...);

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: print
********************************************************************************
* Summary:
*    Prints a portion of the buffer to the shell, masking the characters 
*    if needed.
*
* Parameters:
*   rl     Pointer to the readline structure.
*   offset Offset within the buffer to start printing.
*   count  Number of characters to print.
*
* Return:
*   The number of characters printed.
*
*******************************************************************************/
static int print(readline_t* rl, int offset, int count) 
{
    if (!rl->mask) 
    {
        return printk("%.*s", count, rl->buffer + offset);
    } 
    else 
    {
        /* Create a large enough string of asterisks */
        static const char *asterisks =
              "********************************************"
              "********************************************"
              "********************************************"
              "********************************************";
        if(count > (int)strlen(asterisks))
        {
            count = (int)strlen(asterisks);
        }
        return printk("%.*s", count, asterisks);
   }
}

/******************************************************************************
* Function Name: get_edit_width
********************************************************************************
* Summary:
*    Calculates the width available for editing in the readline buffer.
*
* Parameters:
*   rl  Pointer to the readline structure.
*
* Return:
*   The number of characters available for editing.
*
*******************************************************************************/
static int get_edit_width(readline_t* rl) 
{
    int width = rl->width;
    if(width <= 0)
    {
        width = cursor_terminal_width();
    }

    return width - rl->column;
}

/******************************************************************************
* Function Name: redraw_full
********************************************************************************
* Summary:
*    Redraws the entire readline buffer, updating the display and cursor position.
*
* Parameters:
*   rl  Pointer to the readline structure.
*
* Return:
*   None
*
*******************************************************************************/
static void redraw_full(readline_t* rl) 
{
    int buffer_len = (int)strlen(rl->buffer);
    int edit_width = get_edit_width(rl);
    int count = buffer_len - rl->buffer_display;

    if(count > edit_width)
    {
        count = edit_width;
    }

    int column = rl->column + 1;
    cursor_set_cursor_column(column);
    column += print(rl, rl->buffer_display, count);
    cursor_erase_line();
    int cursor_column = rl->column + 1 + rl->buffer_cursor - rl->buffer_display;
    if(column != cursor_column)
    {
        cursor_set_cursor_column(cursor_column);
    }

}

/******************************************************************************
* Function Name: reallocate_buffer
********************************************************************************
* Summary:
*    Reallocates the buffer if needed, doubling its size.
*
* Parameters:
*   rl  Pointer to the readline structure.
*
* Return:
*   0 on success, -1 on failure.
*
*******************************************************************************/
static int reallocate_buffer(readline_t* rl) 
{
    if (!rl->remalloc) 
    {
        return -1;
    }

    if (rl->buffer_size > (INT_MAX / 2))
    {
        return -1; /* Overflow guard */
    }

    int new_size = rl->buffer_size * 2;
    char* new_buffer = shell_realloc(rl->buffer, new_size);
    if (!new_buffer) 
    {
        return -1; /* Allocation failed */
    }

    rl->buffer = new_buffer;
    rl->buffer_size = new_size;
    return 0;
}

/******************************************************************************
* Function Name: readline_init
********************************************************************************
* Summary:
*    Initializes the readline structure with the provided parameters.
*
* Parameters:
*   rl  Pointer to the readline structure.
*   column  Initial column position.
*   width  Width of the readline area.
*   buffer  Pointer to the buffer.
*   buffer_size  Size of the buffer.
*
* Return:
*   None
*
*******************************************************************************/
void readline_init(readline_t* rl, int column, int width, char* buffer, int buffer_size) 
{
    buffer[0] = '\0';
    rl->buffer = buffer;
    rl->buffer_size = buffer_size;
    rl->buffer_cursor = 0;
    rl->buffer_display = 0;
    rl->mask = false;
    rl->width = width;
    rl->column = column;
    rl->key_press = NULL;
    rl->ansi_code = NULL;
    rl->remalloc = false;
    rl->is_modified = false;
}

/******************************************************************************
* Function Name: readline_init_ex
********************************************************************************
* Summary:
*    Initializes the readline structure with extended parameters.
*
* Parameters:
*   rl  Pointer to the readline structure.
*   column  Initial column position.
*   width  Width of the readline area.
*   buffer  Pointer to the buffer.
*   buffer_size  Size of the buffer.
*   cursor_index  Initial cursor position.
*   key_press  Key hook function.
*   ansi_code  ANSI hook function.
*   remalloc  Flag indicating if buffer can be reallocated.
*
* Return:
*   None
*
*******************************************************************************/
void readline_init_ex(
        readline_t* rl,
        int column,
        int width,
        char* buffer,
        int buffer_size,
        int cursor_index,
        key_hook_fn key_press,
        ansi_hook_fn ansi_code,
        bool remalloc)
{
    rl->buffer = buffer;
    rl->buffer_size = buffer_size;
    rl->buffer_cursor = cursor_index;
    rl->buffer_display = 0;
    rl->mask = false;
    rl->width = width;
    rl->column = column;
    rl->key_press = key_press;
    rl->ansi_code = ansi_code;
    rl->remalloc = remalloc;
    rl->is_modified = false;
}

/******************************************************************************
* Function Name: parse_ansi_sequence
********************************************************************************
* Summary:
*    Parses the ANSI escape sequence and stores the result in the provided 
*    ansi_code_t structure.
*
* Parameters:
*   code  Pointer to the ansi_code_t structure to store the parsed ANSI code.
*
* Return:
*   None
*
*******************************************************************************/
static void parse_ansi_sequence(ansi_code_t* code) 
{
    int c;
    int current_param = 0;
    bool has_parameters = false;
    code->param_count = 0;
    code->final = FINAL_UNKNOWN;

    /* Read characters to parse parameters and the final character */
    while ((c = getchar()) != EOF) 
    {
        if (isdigit((unsigned char)c)) 
        {
            /* Accumulate numeric parameter */
            current_param = current_param * 10 + (c - '0');
            has_parameters = true;
        } 
        else if (c == ';') 
        {
            /* Store the current parameter and reset for the next one */
            if (code->param_count < 10)
            {
                code->parameters[code->param_count++] = current_param;
            }
            current_param = 0;
        } 
        else if (c >= '@' && c <= '~') 
        {
            /* Store the last parameter if any */
            if (has_parameters && code->param_count < 10) 
            {
                code->parameters[code->param_count++] = current_param;
            }
            code->final = c;
            if (code->final == FINAL_KEYCODE && code->param_count > 0) 
            {
                int keycode_param = code->parameters[0];
                code->keycode = (keycode_t)keycode_param;
            }
            return;
        } 
        else 
        {
            /* Invalid character in the sequence */
            return;
        }
    }
}

/******************************************************************************
* Function Name: readline
********************************************************************************
* Summary:
*    Manages the input line, handling cursor movement, character insertion, 
*    and special keys.
*
* Parameters:
*   rl  Pointer to the readline structure.
*
* Return:
*   0 on success, -1 on failure.
*
*******************************************************************************/
int readline(readline_t* rl) 
{
    /* Check for invalid inputs */
    if (rl == NULL || rl->buffer == NULL || rl->buffer_size < 1) 
    {
        return -1;
    }

    int edit_width = get_edit_width(rl);
    int buffer_len = (int)strlen(rl->buffer);


    if (rl->buffer_cursor > (buffer_len + 1))
    {
        rl->buffer_cursor = (buffer_len + 1);
    }

    if (rl->buffer_cursor < 0)
    {
        rl->buffer_cursor = 0;
    }

    if (rl->buffer_display > rl->buffer_cursor)
    {
        rl->buffer_display = rl->buffer_cursor;
    }

    if (rl->buffer_display < rl->buffer_cursor - edit_width)
    {
        rl->buffer_display = rl->buffer_cursor - edit_width;
    }


    if (rl->buffer_display < 0)
    {
        rl->buffer_display = 0;
    }

    redraw_full(rl);

    bool reduced_ansi = cursor_reduced_ansi();

    while (1) 
    {
       
        if(!reduced_ansi)
        {
            cursor_show();
        }

        int c = getchar();
        if (!reduced_ansi)
        {
            cursor_hide();
        }

        if (c == EOF)
        {
            rl->buffer[buffer_len] = '\0';
            return -1;
        }

        if (c == 27) 
        { 
        /* ESC key */ 
            c = getchar();
            if (c == EOF)
            {
                rl->buffer[buffer_len] = '\0';
                return -1;
            }
            if (c == '[') 
            {
                ansi_code_t ansi;
                parse_ansi_sequence(&ansi);

                if (rl->ansi_code != NULL) 
                {
                    int ret = rl->ansi_code(rl, &ansi);
                    if (ret != 0) 
                    {
                        rl->buffer[buffer_len] = '\0';
                        return ret;
                    }
                }

                if(ansi.final == FINAL_CURSOR_FORWARD) 
                {
                    if (rl->buffer_cursor < buffer_len) 
                    {
                        rl->buffer_cursor++;
                        if (rl->buffer_cursor - rl->buffer_display >= edit_width) 
                        {
                            rl->buffer_display++;
                            redraw_full(rl);
                        } 
                        else 
                        {
                            cursor_move_right(1);
                        }
                    }
                } 
                else  if(ansi.final == FINAL_CURSOR_BACKWARD) 
                {
                    if (rl->buffer_cursor > 0) 
                    {
                        rl->buffer_cursor--;
                        if (rl->buffer_cursor < rl->buffer_display) 
                        {
                            rl->buffer_display--;
                            redraw_full(rl);
                        } 
                        else 
                        {
                            cursor_move_left(1);
                        }
                    }
                } 
                else if(ansi.final == FINAL_KEYCODE && ansi.keycode == KC_HOME) 
                {
                    rl->buffer_cursor = 0;
                    rl->buffer_display = 0;
                    redraw_full(rl);
                } 
                else if(ansi.final == FINAL_KEYCODE && ansi.keycode == KC_END) 
                {
                    rl->buffer_cursor = buffer_len;
                    rl->buffer_display = buffer_len > edit_width ? buffer_len - edit_width + 1: 0;
                    redraw_full(rl);
                }
            }
        } 
        else 
        {
            if (rl->key_press != NULL) 
            {
                int ret = rl->key_press(rl, (char)c);
                if (ret != 0) 
                {
                    rl->buffer[buffer_len] = '\0';
                    return ret;
                }
             }

            if (c == 127 || c == '\b') 
            { 
                /* Handle backspace */
                if (buffer_len <= 0 || rl->buffer_cursor <= 0) 
                {
                    continue;  /* nothing to delete */
                }

                if (rl->buffer_cursor == buffer_len) 
                { 
                    /* Cursor at the end of the buffer */
                    buffer_len--;
                    rl->buffer_cursor--;
                    rl->buffer[buffer_len] = '\0';
                    rl->is_modified = true;
                    if (rl->buffer_cursor >= rl->buffer_display) 
                    {
                        printk("\b \b");
                    } 
                    else 
                    {
                        rl->buffer_display--;
                    }
                } 
                else 
                { 
                    /* Handle backspace in the middle of the buffer */
                    memmove(rl->buffer + rl->buffer_cursor - 1, rl->buffer + rl->buffer_cursor, buffer_len - rl->buffer_cursor);
                    buffer_len--;
                    rl->buffer_cursor--;
                    rl->buffer[buffer_len] = '\0';
                    rl->is_modified = true;

                    if (rl->buffer_cursor < rl->buffer_display)
                    {
                        rl->buffer_display--;
                    }
                    redraw_full(rl);
                }
            } 
            else if (c == '\n') 
            {
                continue;
            } 
            else if (c == '\r' || c == '\0') 
            { 
                /* Enter */ 
                rl->buffer[buffer_len] = '\0';
                return 0;
            } 
            else if (isprint((unsigned char)c)) 
            { 
                /* Printable characters */
                if (buffer_len >= (rl->buffer_size - 2)) 
                {
                    if (reallocate_buffer(rl) < 0) 
                    {
                        continue; /* Buffer is full and cannot be reallocated */
                    }
                }

                if(buffer_len == rl->buffer_cursor) 
                {
                    rl->buffer[buffer_len++] = c;
                    rl->buffer_cursor++;
                    rl->buffer[buffer_len] = '\0';
                    if (rl->buffer_cursor - rl->buffer_display >= edit_width) 
                    {
                        rl->buffer_display++;
                        redraw_full(rl);
                    } 
                    else 
                    {
                        if(rl->mask)
                        {
                            printk("*");
                        }
                        else
                        {
                            printk("%c", c);
                        }

                    }
                } 
                else 
                {
                    memmove(rl->buffer + rl->buffer_cursor + 1, rl->buffer + rl->buffer_cursor, buffer_len - rl->buffer_cursor);
                    rl->buffer[rl->buffer_cursor] = c;
                    buffer_len++;
                    rl->buffer_cursor++;
                    rl->buffer[buffer_len] = '\0';

                    if (rl->buffer_cursor - rl->buffer_display >= edit_width)
                    {
                        rl->buffer_display++;
                    }

                    redraw_full(rl);
                }

                rl->is_modified = true;
            }
        }
    }
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */
