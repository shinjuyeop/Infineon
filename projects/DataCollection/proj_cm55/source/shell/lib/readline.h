/******************************************************************************
* File Name:   readline.h
*
* Description: Header file for readline functionality, including cursor movement,
*              line editing, and buffer handling.
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

#ifndef _READLINE_H_
#define _READLINE_H_

#ifdef IM_ENABLE_SHELL

/******************************************************************************
 * Types
 *****************************************************************************/

/* Enum for ANSI final characters.
 * This enum defines the possible final characters for ANSI escape sequences.
 */
typedef enum {
    FINAL_UNKNOWN = 0,
    FINAL_CURSOR_UP = 'A',
    FINAL_CURSOR_DOWN = 'B',
    FINAL_CURSOR_FORWARD = 'C',
    FINAL_CURSOR_BACKWARD = 'D',
    FINAL_CURSOR_NEXT_LINE = 'E',
    FINAL_CURSOR_PREVIOUS_LINE = 'F',
    FINAL_CURSOR_HORIZONTAL = 'G',
    FINAL_CURSOR_POSITION = 'H',
    FINAL_ERASE_DISPLAY = 'J',
    FINAL_ERASE_LINE = 'K',
    FINAL_INSERT_LINE = 'L',
    FINAL_DELETE_LINE = 'M',
    FINAL_DELETE_CHARACTER = 'P',
    FINAL_SCROLL_UP = 'S',
    FINAL_SCROLL_DOWN = 'T',
    FINAL_GRAPHICS_MODE = 'm',
    FINAL_DEVICE_STATUS = 'c',
    FINAL_LINE_POSITION = 'd',
    FINAL_SET_MODE = 'h',
    FINAL_RESET_MODE = 'l',
    FINAL_DEVICE_REPORT = 'n',
    FINAL_SCROLLING_REGION = 'r',
    FINAL_SAVE_CURSOR = 's',
    FINAL_RESTORE_CURSOR = 'u',
    FINAL_KEYCODE = '~'
} ansi_final_t;

/* Enum for keycodes when final is FINAL_KEYCODE.
 * This enum defines the possible keycodes.
 */
typedef enum {
    KC_UNKNOWN = 0,
    KC_HOME = 1,        /* Home key */
    KC_INSERT = 2,      /* Insert key */
    KC_DELETE = 3,      /* Delete key */
    KC_END = 4,         /* End key */
    KC_PAGE_UP = 5,     /* Page Up key */
    KC_PAGE_DOWN = 6,   /* Page Down key */
    KC_F1 = 11,         /* F1 key */
    KC_F2 = 12,         /* F2 key */
    KC_F3 = 13,         /* F3 key */
    KC_F4 = 14,         /* F4 key */
    KC_F5 = 15,         /* F5 key */
    KC_F6 = 17,         /* F6 key */
    KC_F7 = 18,         /* F7 key */
    KC_F8 = 19,         /* F8 key */
    KC_F9 = 20,         /* F9 key */
    KC_F10 = 21,        /* F10 key */
    KC_F11 = 23,        /* F11 key */
    KC_F12 = 24         /* F12 key */
} keycode_t;

/* Structure to store ANSI escape sequence information.
 * This structure holds the information parsed from an ANSI escape sequence.
 */
typedef struct
{
    union {
        int parameters[10]; /* Parameters of the ANSI escape sequence */
        keycode_t keycode;  /* Valid if final == FINAL_KEYCODE */
    };
    int param_count;        /* Number of parameters */
    ansi_final_t final;     /* Final character of the ANSI escape sequence */
} ansi_code_t;

typedef struct readline_s readline_t;

/* Key hook function type.
 * This function type defines the callback function for handling key presses.
 * If the return value is anything other than 0, readline will return with this number.
 */
typedef int (*key_hook_fn)(readline_t* rl, char key);

/* ANSI hook function type.
 * This function type defines the callback function for handling ANSI escape sequences.
 * If the return value is anything other than 0, readline will return with this number.
 */
typedef int (*ansi_hook_fn)(readline_t* rl, ansi_code_t* ansi);

/* Struct for managing readline operations.
 * This structure holds the information and state for managing readline operations.
 */
struct readline_s {
    void* arg;                  /* User-defined pointer for any specific use */

    bool mask;                  /* If true, all the text will be shown as '*' */
    char *buffer;               /* Pointer to the text data. It is always null-terminated 
                                 * and assumed to never contain any \n or \r, only printable 
                                 * characters */
    int buffer_size;            /* Number of bytes allocated for the buffer */
    int buffer_cursor;          /* Cursor position within the buffer; always less than or equal to buffer size */
    int buffer_display;         /* Index of the first displayed character in the buffer */

    key_hook_fn key_press;      /* Hook for keys. */
    ansi_hook_fn ansi_code;     /* Hook for ansi escape.  */

    bool remalloc;              /* If true, the buffer is reallocated if needed */
    bool is_modified;           /* If true, the buffer has been modified */

    int column;                 /* Column (indexed at 0) */
    int width;                  /* Width (If 0 terminal width is used) */
};

/*******************************************************************************
* Function Prototypes
*******************************************************************************/
void readline_init(readline_t* rl, int column, int width, char* buffer, int buffer_size);
void readline_init_ex(
        readline_t* rl,
        int column,
        int width,
        char* buffer,
        int buffer_size,
        int cursor_index,
        key_hook_fn control_key,
        ansi_hook_fn ansi_code,
        bool remalloc);

int readline(readline_t* rl);

#endif /* IM_ENABLE_SHELL */
#endif /* _READLINE_H_ */

/* [] END OF FILE */
