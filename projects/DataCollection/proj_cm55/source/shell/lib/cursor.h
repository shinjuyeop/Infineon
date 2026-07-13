/******************************************************************************
* File Name:   cursor.h
*
* Description: Header file for cursor control functions.
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
#ifndef _CURSOR_H_
#define _CURSOR_H_

#ifdef IM_ENABLE_SHELL

#include <stdbool.h>

/******************************************************************************
 * Type 
 *****************************************************************************/

/* This enum defines the possible colors that can be used for the cursor's
 * foreground and background.
 */
typedef enum cursor_color_t {
    COLOR_DEFAULT = 39,
    COLOR_BLACK = 30,
    COLOR_RED = 31,
    COLOR_GREEN = 32,
    COLOR_YELLOW = 33,
    COLOR_BLUE = 34,
    COLOR_MAGENTA = 35,
    COLOR_CYAN = 36,
    COLOR_WHITE = 37,
    COLOR_BRIGHT_BLACK = 90,
    COLOR_BRIGHT_RED = 91,
    COLOR_BRIGHT_GREEN = 92,
    COLOR_BRIGHT_YELLOW = 93,
    COLOR_BRIGHT_BLUE = 94,
    COLOR_BRIGHT_MAGENTA = 95,
    COLOR_BRIGHT_CYAN = 96,
    COLOR_BRIGHT_WHITE = 97
} cursor_color_t;

/*******************************************************************************
* Function Prototypes
*******************************************************************************/
int cursor_terminal_width(void);
int cursor_terminal_height(void);
bool cursor_reduced_ansi(void);
void cursor_clear_screen();
void cursor_set_color(cursor_color_t fg, cursor_color_t bg);
void cursor_reset_attributes();
void cursor_erase_line();
void cursor_hide();
void cursor_show();
void cursor_scroll_up(int lines);
void cursor_scroll_down(int lines);
void cursor_move_right(int positions);
void cursor_move_left(int positions);
void cursor_set_cursor_column(int position);
void cursor_set_cursor_position(int row, int col);

#endif /* IM_ENABLE_SHELL */
#endif /* _CURSOR_H_ */

/* [] END OF FILE */
