/******************************************************************************
* File Name:   cursor.c
*
* Description: Implementation of cursor control functions.
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

#include <stdio.h>

#include "cursor.h"
#include "../process.h"
#include "../terminal.h"

/*******************************************************************************
* Function Definitions
*******************************************************************************/

/******************************************************************************
* Function Name: cursor_terminal_width
********************************************************************************
* Summary:
*    Retrieves the width of the terminal.
*
* Parameters:
*   None
*
* Return:
*   The width of the terminal in characters.
*
*******************************************************************************/
int cursor_terminal_width(void) 
{
    return process_get(NULL)->terminal->width;
}

/******************************************************************************
* Function Name: cursor_terminal_height
********************************************************************************
* Summary:
*    Retrieves the height of the terminal.
*
* Parameters:
*   None
*
* Return:
*   The height of the terminal in characters.
*
*******************************************************************************/
int cursor_terminal_height(void) 
{
    return process_get(NULL)->terminal->height;
}

/******************************************************************************
* Function Name: cursor_reduced_ansi
********************************************************************************
* Summary:
*    Checks if the terminal is using reduced ANSI mode.
*
* Parameters:
*   None
*
* Return:
*   true if the terminal is using reduced ANSI mode, false otherwise.
*
*******************************************************************************/
bool cursor_reduced_ansi(void) 
{
    return process_get(NULL)->terminal->attr & TERMINAL_ATTR_REDUCED_ANSI;
}

/******************************************************************************
* Function Name: cursor_clear_screen
********************************************************************************
* Summary:
*    Clears the terminal screen.
*
* Parameters:
*   None
*
* Return:
*   None
*
*******************************************************************************/
void cursor_clear_screen() 
{
    printf("\033[2J\033[H");
}

/******************************************************************************
* Function Name: cursor_set_color
********************************************************************************
* Summary:
*    Sets the foreground and background color of the terminal.
*
* Parameters:
*   fg - The foreground color.
*   bg - The background color.
*
* Return:
*   None
*
*******************************************************************************/
void cursor_set_color(cursor_color_t fg, cursor_color_t bg) 
{
    if(process_get(NULL)->terminal->attr & TERMINAL_ATTR_COLOR)
    {
        printf("\033[%d;%dm", fg, bg + 10);
    }

}

/******************************************************************************
* Function Name: cursor_reset_attributes
********************************************************************************
* Summary:
*    Resets the terminal attributes to their default values.
*
* Parameters:
*   None
*
* Return:
*   None
*
*******************************************************************************/
void cursor_reset_attributes() 
{
    printf("\033[0m");
}

/******************************************************************************
* Function Name: cursor_erase_line
********************************************************************************
* Summary:
*    Erases the current line in the terminal.
*
* Parameters:
*   None
*
* Return:
*   None
*
*******************************************************************************/
void cursor_erase_line() 
{
    printf("\033[K");
}

/******************************************************************************
* Function Name: cursor_hide
********************************************************************************
* Summary:
*    Hides the cursor in the terminal.
*
* Parameters:
*   None
*
* Return:
*   None
*
*******************************************************************************/
void cursor_hide() 
{
    printf("\033[?25l");
}

/******************************************************************************
* Function Name: cursor_show
********************************************************************************
* Summary:
*    Shows the cursor in the terminal.
*
* Parameters:
*   None
*
* Return:
*   None
*
*******************************************************************************/
void cursor_show() 
{
    printf("\033[?25h");
}

/******************************************************************************
* Function Name: cursor_move_right
********************************************************************************
* Summary:
*    Moves the cursor to the right by the specified number of positions.
*
* Parameters:
*   positions - The number of positions to move the cursor.
*
* Return:
*   None
*
*******************************************************************************/
void cursor_move_right(int positions) 
{
    printf("\033[%dC", positions);
}

/******************************************************************************
* Function Name: cursor_move_left
********************************************************************************
* Summary:
*    Moves the cursor to the left by the specified number of positions.
*
* Parameters:
*   positions - The number of positions to move the cursor.
*
* Return:
*   None
*
*******************************************************************************/
void cursor_move_left(int positions) 
{
    printf("\033[%dD", positions);
}

/******************************************************************************
* Function Name: cursor_set_cursor_column
********************************************************************************
* Summary:
*    Sets the cursor to the specified column in the current row.
*
* Parameters:
*   position - The column position to set the cursor.
*
* Return:
*   None
*
*******************************************************************************/
void cursor_set_cursor_column(int position) 
{
    printf("\033[%dG", position);
}

/******************************************************************************
* Function Name: cursor_set_cursor_position
********************************************************************************
* Summary:
*    Sets the cursor to the specified row and column position.
*
* Parameters:
*   row - The row position to set the cursor.
*   col - The column position to set the cursor.
*
* Return:
*   None
*
*******************************************************************************/
void cursor_set_cursor_position(int row, int col) 
{
    printf("\033[%d;%dH", row, col);
}

/******************************************************************************
* Function Name: cursor_scroll_up
********************************************************************************
* Summary:
*    Scrolls the terminal up by the specified number of lines.
*
* Parameters:
*   lines - The number of lines to scroll up.
*
* Return:
*   None
*
*******************************************************************************/
void cursor_scroll_up(int lines) 
{
    printf("\033[%dS", lines);
}

/******************************************************************************
* Function Name: cursor_scroll_down
********************************************************************************
* Summary:
*    Scrolls the terminal down by the specified number of lines.
*
* Parameters:
*   lines - The number of lines to scroll down.
*
* Return:
*   None
*
*******************************************************************************/
void cursor_scroll_down(int lines) 
{
    printf("\033[%dT", lines);
}

#endif /* IM_ENABLE_SHELL */

/* [] END OF FILE */
