/******************************************************************************
* File Name:   dev_pdm_pcm.c
*
* Description: This file implements the PDM/PCM peripheral for PSOC Edge using 
* Peripheral Driver Layer (PDL) APIs.
*
* Related Document: See README.md
*
*******************************************************************************
* (c) 2025-2026, Infineon Technologies AG, or an affiliate of Infineon
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
#ifdef IM_ENABLE_PDM_PCM
#include "cy_pdm_pcm_v2.h"
#include "pdm_pcm.h"

/******************************************************************************
 * Macros
 *****************************************************************************/
/* PDM PCM hardware FIFO size */
#define HW_FIFO_SIZE                            (64u)

/* Rx FIFO trigger level/threshold configured by user */
#define RX_FIFO_TRIG_LEVEL                      (HW_FIFO_SIZE/2)

/* Total number of interrupts to get the FRAME_SIZE number of samples*/
#define NUMBER_INTERRUPTS_FOR_FRAME             ((FRAME_SIZE)/RX_FIFO_TRIG_LEVEL)

/* PDM PCM interrupt priority */
#define PDM_PCM_ISR_PRIORITY                    (2u)

/* Decimation Rate of the PDM/PCM block. Typical value is 64 */
#define DECIMATION_RATE             (64u)

/* Define how many samples in a frame */
#define FRAME_SIZE                  (1024)

/* Define the DPLL Configurations */
#define DPLL_INPUT_FREQ_HZ          (17203200UL)
#define DPLL_OUTPUT1_FREQ_HZ        (73728000UL)
#define DPLL_OUTPUT2_FREQ_HZ        (169344000UL)
#define DPLL_ENABLE_TIMEOUT_MS      (10000U)

#ifdef INCLUDE_STRINGS
const char* sample_rate_string_list[] =  { "8 kHz", "16 kHz", "22.05 kHz", "44.1 kHz", "48 kHz" };

const char* gain_option_list[] = {"83 dB", "77 dB", "71 dB", "65 dB","59 dB", "53 dB", "47 dB",
                                  "41 dB", "35 dB","29 dB","23 dB","17 dB","11 dB",
                                 "5 dB","-1 dB","-7 dB","-13 dB","-19 dB","-25 dB",
                                 "-31 dB","-37 dB","-43 dB","-49 dB","-55 dB","-61 dB",
                                 "-67 dB","-73 dB","-79 dB","-85 dB","-91 dB",
                                 "-97 dB","-103 dB"};
#endif 
/******************************************************************************
 * Type Definitions
 *****************************************************************************/

typedef struct pdm_pcm_t
{
    uint8_t index;
    int16_t audio_buffer0[FRAME_SIZE];
    int16_t audio_buffer1[FRAME_SIZE];
    int16_t* active_rx_buffer;
    int16_t* full_rx_buffer;
    PDM_PCM_CONFIG_t * config;
    cy_en_pdm_pcm_gain_sel_t gain[MAX_CHANNEL_COUNT_PER_HANDLE];
    bool is_used;
    bool pdm_pcm_initialized;
    bool have_data;
    int skipped_frames;
    int init_discard_counter;
} pdm_pcm_t;


struct pdm_pcm_t pool_of_mics[MAX_HANDLE_COUNT] = {0};

SAMPLE_RATE pdm_pcm_sample_rate = SAMPLE_RATE_48000;

static void pdm_pcm_event_handler_0(void);
static void pdm_pcm_event_handler_1(void);

void (*event_handlers[MAX_HANDLE_COUNT])(void) = 
{   pdm_pcm_event_handler_0, 
    pdm_pcm_event_handler_1};


/*******************************************************************************
* Function Name: dpll_lp_set_freq
********************************************************************************
* Summary:
*    A function used to update the DPLL LP 1 frequency.
*
* Parameters:
*   sample_rate: Desired DPLL LP 1 frequency.
*
* Return:
*
*******************************************************************************/
static void dpll_lp_set_freq(uint32_t freq)
{
    cy_stc_pll_config_t dpll_lp1;

    /* Check if the current DPLL_LP_1 frequency is same as 'freq' */
    if (Cy_SysClk_PllGetFrequency(SRSS_DPLL_LP_1_PATH_NUM) == freq)
    {
        printf("DPLL_LP_1 is already set to the desired frequency: %lu Hz\n", (unsigned long int)freq);
        return;
    }

    /* Set the DPLL_LP_1 PLL configurations */
    dpll_lp1.inputFreq = DPLL_INPUT_FREQ_HZ;
    dpll_lp1.outputFreq = freq;
    dpll_lp1.outputMode = CY_SYSCLK_FLLPLL_OUTPUT_AUTO;
    dpll_lp1.lfMode = false;

    /* Disable the DPLL_LP_1 PLL path */
    Cy_SysClk_PllDisable(SRSS_DPLL_LP_1_PATH_NUM);

    /* Configure the PLL with the specified settings */
    if (CY_SYSCLK_SUCCESS !=
            Cy_SysClk_PllConfigure(SRSS_DPLL_LP_1_PATH_NUM, &dpll_lp1))
    {
        printf("Failed to configure DPLL LP1\r\n");
        CY_ASSERT(0);
    }

    /* Enable the DPLL_LP_1 PLL path with a timeout */
    if (CY_SYSCLK_SUCCESS !=
            Cy_SysClk_PllEnable(SRSS_DPLL_LP_1_PATH_NUM, DPLL_ENABLE_TIMEOUT_MS))
    {
        /* Handle error if PLL enable fails */
        printf("Failed to enable DPLL LP1\r\n");
        CY_ASSERT(0);
    }
}

/******************************************************************************
* Function Name: pdm_pcm_set_gain
*******************************************************************************/
/**
*
* Updates the gain value setting in the given PDM-PCM instance. The gain will be 
* applied when pdm_pcm_start is invoked. 
*
* \param handle
* The pointer to the PDM-PCM instance handle.
*
* \param channel_num
* The channel number for gain setting.
*
* \param gain
* Gain value defined as cy_en_pdm_pcm_gain_sel_t
*
******************************************************************************/
PDM_PCM_STATUS_t pdm_pcm_set_gain (pdm_pcm handle, uint8_t channel_num, cy_en_pdm_pcm_gain_sel_t gain)
{

    if (NULL == handle)
    {
        return PDM_PCM_STATUS_BAD_PARAM;
    }

    for (uint8_t ch_idx = 0; ch_idx < (handle->config->mode+1); ch_idx++)
    {
        if (handle->config->channel_index_list[ch_idx] == channel_num)
        {
            handle->gain[ch_idx] = gain;
            return PDM_PCM_STATUS_SUCCESS;
        }
    }
    
    return PDM_PCM_STATUS_BAD_PARAM;
}

/*******************************************************************************
* Function Name: pdm_pcm_init_hw
********************************************************************************
* Summary:
*    A function used to initialize and configure the PDM to PCM converter and 
*    the clocks that source it.
*
* Parameters:
*   sample_rate: Desired Sample Rate for the converter.
*
* Return:
*     The status of the initialization.
*
*******************************************************************************/
PDM_PCM_STATUS_t pdm_pcm_init_hw(SAMPLE_RATE sample_rate)
{
    pdm_pcm_sample_rate = sample_rate;

    if ( (SAMPLE_RATE_8000 == sample_rate) || (SAMPLE_RATE_16000 == sample_rate) ||
         (SAMPLE_RATE_48000 == sample_rate) )
    {
        dpll_lp_set_freq(DPLL_OUTPUT1_FREQ_HZ);
    }
    else if ( (SAMPLE_RATE_22050 == sample_rate) || (SAMPLE_RATE_44100 == sample_rate) )
    {
        dpll_lp_set_freq(DPLL_OUTPUT2_FREQ_HZ);
    }
    else
    {
        printf("Invalid Sample Rate\r\n");
        return PDM_PCM_STATUS_BAD_PARAM;
    }

    /* Set the clock dividers for the selected sample rate */
    Cy_SysClk_PeriPclkDisableDivider((en_clk_dst_t)CYBSP_PDM_CLK_DIV_GRP_NUM, CY_SYSCLK_DIV_16_5_BIT, 1U);
    switch(sample_rate)
    {
        case SAMPLE_RATE_8000:
        {
            Cy_SysClk_PeriPclkSetFracDivider((en_clk_dst_t)CYBSP_PDM_CLK_DIV_GRP_NUM, CY_SYSCLK_DIV_16_5_BIT, 1U, 11U, 0U);
            break;
        }
        case SAMPLE_RATE_16000:
        {
            Cy_SysClk_PeriPclkSetFracDivider((en_clk_dst_t)CYBSP_PDM_CLK_DIV_GRP_NUM, CY_SYSCLK_DIV_16_5_BIT, 1U, 5U, 0U);
            break;
        }
        case SAMPLE_RATE_48000:
        {
            Cy_SysClk_PeriPclkSetFracDivider((en_clk_dst_t)CYBSP_PDM_CLK_DIV_GRP_NUM, CY_SYSCLK_DIV_16_5_BIT, 1U, 1U, 0U);
            break;
        }
        case SAMPLE_RATE_22050:
        {
            Cy_SysClk_PeriPclkSetFracDivider((en_clk_dst_t)CYBSP_PDM_CLK_DIV_GRP_NUM, CY_SYSCLK_DIV_16_5_BIT, 1U, 9U, 0u);
            break;
        }
        case SAMPLE_RATE_44100:
        {
            Cy_SysClk_PeriPclkSetFracDivider((en_clk_dst_t)CYBSP_PDM_CLK_DIV_GRP_NUM, CY_SYSCLK_DIV_16_5_BIT, 1U, 4U, 0U);
            break;
        }
    }
    Cy_SysClk_PeriPclkEnableDivider((en_clk_dst_t)CYBSP_PDM_CLK_DIV_GRP_NUM, CY_SYSCLK_DIV_16_5_BIT, 1U);

    return PDM_PCM_STATUS_SUCCESS;
}

/*******************************************************************************
* Function Name: pdm_pcm_create
********************************************************************************
* Summary:
*    A function used to create an instance of the PDM-PCM interface.
*
* Parameters:
*   config: Channel configuration.
*
* Return:
*     A handle to the PDM-PCM interface.
*
*******************************************************************************/
pdm_pcm pdm_pcm_create(PDM_PCM_CONFIG_t * config)
{
    pdm_pcm mic_handle = NULL;
    for (uint8_t i = 0; i < 2; i++)
    {
        if (false == pool_of_mics[i].is_used)
        {
            mic_handle = &pool_of_mics[i];
            mic_handle->config = config;
            mic_handle->index = i;
            break;
        }
    }

    return mic_handle;
}

/*******************************************************************************
* Function Name: pdm_pcm_update_config
********************************************************************************
* Summary:
*    A function used to modify the PDM-PCM configuration of a given instance. 
*
* Parameters:
*   config: config to be used after function is invoked.
*
* Return:
*     The status of the operation.
*
*******************************************************************************/
PDM_PCM_STATUS_t pdm_pcm_update_config(pdm_pcm handle, PDM_PCM_CONFIG_t * config)
{
    if (NULL == handle)
    {
        return PDM_PCM_STATUS_BAD_PARAM;
    }

    handle->config = config;

    return PDM_PCM_STATUS_SUCCESS;
}
/*******************************************************************************
* Function Name: pdm_pcm_event_handler
********************************************************************************
* Summary:
*  PDM/PCM ISR handler. Check the interrupt status and clears it.
*  Fills a buffer and then swaps that buffer with an empty one.
*  Once a buffer is full, a flag is set which is used in main.
*
* Parameters:
*  handle: Handle for the PDM-PCM interface.
*
* Return:
*  None
*
*******************************************************************************/
static void pdm_pcm_event_handler(pdm_pcm handle)
{
    PDM_PCM_CONFIG_t*  config = handle->config;
    uint8_t num_channels = config->mode + 1;
    uint8_t int_channel_index = config->pdm_irq_cfg.intrSrc - pdm_0_interrupts_0_IRQn;

    /* Used to track how full the buffer is */
    static uint16_t frame_counter = 0;

    /* Check the interrupt status */
    uint32_t intr_status = Cy_PDM_PCM_Channel_GetInterruptStatusMasked(CYBSP_PDM_HW, int_channel_index);
    if(CY_PDM_PCM_INTR_RX_TRIGGER & intr_status)
    {
        /* Move data from the PDM fifo and place it in a buffer */
        uint32_t index = 0;
        for(uint32_t i=0; i < (RX_FIFO_TRIG_LEVEL/num_channels); i++)
        {
            for(uint8_t ch_idx = 0; ch_idx < num_channels; ch_idx++)
            {
                int32_t data = (int32_t)Cy_PDM_PCM_Channel_ReadFifo(CYBSP_PDM_HW, config->channel_index_list[ch_idx]);
                handle->active_rx_buffer[frame_counter * RX_FIFO_TRIG_LEVEL + index] = (int16_t)(data);
                index++;
            }
        }
        Cy_PDM_PCM_Channel_ClearInterrupt(CYBSP_PDM_HW, int_channel_index, CY_PDM_PCM_INTR_RX_TRIGGER);
        frame_counter++;
    }
    /* Check if the buffer is full */
    if((NUMBER_INTERRUPTS_FOR_FRAME) <= frame_counter)
    {
        /* Flip the active and the next rx buffers */
        int16_t* temp = handle->active_rx_buffer;
        handle->active_rx_buffer = handle->full_rx_buffer;
        handle->full_rx_buffer = temp;

        /* Set the have_data flag as true, signaling there is data ready for use */
        handle->have_data = true;
        frame_counter = 0;
    }

    if((CY_PDM_PCM_INTR_RX_FIR_OVERFLOW | CY_PDM_PCM_INTR_RX_OVERFLOW|
            CY_PDM_PCM_INTR_RX_IF_OVERFLOW | CY_PDM_PCM_INTR_RX_UNDERFLOW) & intr_status)
    {
        Cy_PDM_PCM_Channel_ClearInterrupt(CYBSP_PDM_HW, int_channel_index, CY_PDM_PCM_INTR_MASK);
    }
}

/*******************************************************************************
* Function Name: pdm_pcm_event_handler_0
********************************************************************************
* Summary:
*  PDM/PCM ISR handler for the first instance of the interface in the pool. 
*
* Parameters:
*
* Return:
*  None
*
*******************************************************************************/
static void pdm_pcm_event_handler_0(void)
{
    pdm_pcm_event_handler(&pool_of_mics[0]);
}

/*******************************************************************************
* Function Name: pdm_pcm_event_handler_1
********************************************************************************
* Summary:
*  PDM/PCM ISR handler for the second instance of the interface in the pool. 
*
* Parameters:
*
* Return:
*  None
*
*******************************************************************************/
static void pdm_pcm_event_handler_1(void)
{
    pdm_pcm_event_handler(&pool_of_mics[1]);
}
/*******************************************************************************
* Function Name: pdm_pcm_start
********************************************************************************
* Summary:
*    A function to start the PDM-PCM converter for a given handle. 
*
* Parameters:
*   handle: Handle for the PDM-PCM interface.
*
* Return:
*     The status of the operation.
*
*******************************************************************************/
PDM_PCM_STATUS_t pdm_pcm_start(pdm_pcm handle)
{
    cy_rslt_t result;

    if (NULL == handle)
    {
        return PDM_PCM_STATUS_BAD_PARAM;
    }

    PDM_PCM_CONFIG_t*  config = handle->config;
    uint8_t int_channel_index = config->pdm_irq_cfg.intrSrc - pdm_0_interrupts_0_IRQn;

    /* Stop and free the device if running */
    result = pdm_pcm_stop(handle);
    if (CY_RSLT_SUCCESS != result)
    {
        return PDM_PCM_STATUS_FAIL;
    }

    /* Set up pointers to two buffers to implement a ping-pong buffer system.
     * One gets filled by the PDM while the other can be processed. */
    memset(handle->audio_buffer0, 0, FRAME_SIZE*sizeof(int16_t));
    memset(handle->audio_buffer1, 0, FRAME_SIZE*sizeof(int16_t));
    handle->active_rx_buffer = handle->audio_buffer0;
    handle->full_rx_buffer = handle->audio_buffer1;
    handle->have_data = false;
    handle->init_discard_counter = 4; /* Skip the first 4 frames. */


    /* Initialize PDM/PCM block */
    cy_en_pdm_pcm_status_t status = Cy_PDM_PCM_Init(CYBSP_PDM_HW, &CYBSP_PDM_config);
    
    if(CY_PDM_PCM_SUCCESS != status)
    {
        return PDM_PCM_STATUS_FAIL;
    }

    /* Initialize PDM/PCM channels */
    for (uint8_t i = 0; i < (config->mode+1); i++)
    {
        Cy_PDM_PCM_Channel_Enable(CYBSP_PDM_HW, config->channel_index_list[i]);
        Cy_PDM_PCM_Channel_Init(CYBSP_PDM_HW, &config->channel_config[i], config->channel_index_list[i]);
        Cy_PDM_PCM_SetGain(CYBSP_PDM_HW, config->channel_index_list[i], handle->gain[i]);
    }

    /* An interrupt is registered for right channel, clear and set masks for it. */
    Cy_PDM_PCM_Channel_ClearInterrupt(CYBSP_PDM_HW, int_channel_index, CY_PDM_PCM_INTR_MASK);
    Cy_PDM_PCM_Channel_SetInterruptMask(CYBSP_PDM_HW, int_channel_index, CY_PDM_PCM_INTR_MASK);

    config->pdm_irq_cfg.intrPriority = PDM_PCM_ISR_PRIORITY;
    /* Register the IRQ handler */
    result = Cy_SysInt_Init(&config->pdm_irq_cfg, event_handlers[handle->index]);
    if(CY_SYSINT_SUCCESS != result)
    {
        return PDM_PCM_STATUS_FAIL;
    }
    NVIC_ClearPendingIRQ(config->pdm_irq_cfg.intrSrc);
    NVIC_EnableIRQ(config->pdm_irq_cfg.intrSrc);

    /* Global variable used to determine if PDM data is available */
    handle->have_data = false;

    /* Set up pointers to two buffers to implement a ping-pong buffer system.
     * One gets filled by the PDM while the other can be processed. */
    handle->active_rx_buffer = handle->audio_buffer0;
    handle->full_rx_buffer = handle->audio_buffer1;

    for (uint8_t i = 0; i < (config->mode+1); i++)
    {
        Cy_PDM_PCM_Activate_Channel(CYBSP_PDM_HW, config->channel_index_list[i]);
    }

    handle->pdm_pcm_initialized = true;

    char* mode_string[] = {"Mono", "Stereo"};

    cy_en_pdm_pcm_gain_sel_t fir1_ctl = _FLD2VAL(PDM_CH_FIR1_CTL_SCALE, PDM_PCM_CH_FIR1_CTL(CYBSP_PDM_HW, config->channel_index_list[0]));

    printf("pdm_pcm: Configured device mode=%s, sample rate=%s, gain=%s\r\n", 
                mode_string[config->mode], 
                sample_rate_string_list[pdm_pcm_sample_rate],
            gain_option_list[(fir1_ctl)]);


    return PDM_PCM_STATUS_SUCCESS;
}
/*******************************************************************************
* Function Name: pdm_pcm_stop
********************************************************************************
* Summary:
*    A function to stop the PDM-PCM converter for a given handle. 
*
* Parameters:
*   handle: Handle for the PDM-PCM interface.
*
* Return:
*     The status of the operation.
*
*******************************************************************************/
PDM_PCM_STATUS_t pdm_pcm_stop(pdm_pcm handle)
{
    if (NULL == handle)
    {
        return PDM_PCM_STATUS_BAD_PARAM;
    }

    PDM_PCM_CONFIG_t*  config = handle->config;
    if(handle->pdm_pcm_initialized) 
    {
        handle->pdm_pcm_initialized = false;
        for (uint8_t i = 0; i < (config->mode+1); i++)
        {
            Cy_PDM_PCM_DeActivate_Channel(CYBSP_PDM_HW, config->channel_index_list[i]);
        }
    }

    return PDM_PCM_STATUS_SUCCESS;
}

/*******************************************************************************
* Function Name: pdm_pcm_data_ready
********************************************************************************
* Summary:
*    A function that returns the status of the full buffer.  
*
* Parameters:
*   handle: Handle for the PDM-PCM interface.
*
* Return:
*     The status of the full buffer.
*
*******************************************************************************/
bool pdm_pcm_data_ready(pdm_pcm handle)
{
    return handle->have_data;
}

/*******************************************************************************
* Function Name: pdm_pcm_discard_samples
********************************************************************************
* Summary:
*    A function that discards the first 4 samples after sampling has started.  
*
* Parameters:
*   handle: Handle for the PDM-PCM interface.
*
* Return:
*
*******************************************************************************/
void pdm_pcm_discard_samples(pdm_pcm handle)
{
    if ( handle->init_discard_counter > 0)
    {
        handle->init_discard_counter--;
        memset(handle->full_rx_buffer,0,FRAME_SIZE*sizeof(int16_t));
    }
}

/*******************************************************************************
* Function Name: pdm_pcm_get_full_buffer
********************************************************************************
* Summary:
*    A function that returns a pointer to the full buffer.  
*
* Parameters:
*   handle: Handle for the PDM-PCM interface.
*
* Return:
*    A pointer to the full buffer. 
*
*******************************************************************************/
int16_t* pdm_pcm_get_full_buffer(pdm_pcm handle)
{
    return handle->full_rx_buffer;
}

#ifdef INCLUDE_STRINGS
/*******************************************************************************
* Function Name: pdm_pcm_get_string_list_of_sample_rates
********************************************************************************
* Summary:
*    A function that returns a list of strings the indicate the available 
*    sample rates when using this module to configure the PDM-PCM converter. 
*
* Parameters:
*
* Return:
*    List of strings to label available sample rates. 
*
*******************************************************************************/
const char ** pdm_pcm_get_string_list_of_sample_rates(void)
{
    return sample_rate_string_list;
}

/*******************************************************************************
* Function Name: pdm_pcm_get_sample_rate_option_count
********************************************************************************
* Summary:
*    A function that returns how many sample rate options are available when
*    configuring the PDM-PCM converter with this module. 
*
* Parameters:
*
* Return:
*    Number of sample rate options available. 
*
*******************************************************************************/
uint8_t pdm_pcm_get_sample_rate_option_count(void)
{
    return SAMPLE_RATE_48000 + 1;
}

/*******************************************************************************
* Function Name: pdm_pcm_get_string_list_of_sample_rates
********************************************************************************
* Summary:
*    A function that returns a list of strings the indicate the available 
*    gain values when using this module to configure the PDM-PCM converter. 
*
* Parameters:
*
* Return:
*    List of strings to label available gain values. 
*
*******************************************************************************/
const char ** pdm_pcm_get_string_list_of_gain_options(void)
{
    return gain_option_list;
}

/*******************************************************************************
* Function Name: pdm_pcm_get_gain_option_count
********************************************************************************
* Summary:
*    A function that returns how many gain options are available when
*    configuring the PDM-PCM converter with this module. 
*
* Parameters:
*
* Return:
*    Number of gain options available. 
*
*******************************************************************************/
uint8_t pdm_pcm_get_gain_option_count(void)
{
    return CY_PDM_PCM_SEL_GAIN_NEGATIVE_103DB + 1;
}
#endif

/*******************************************************************************
* Function Name: pdm_pcm_get_frequency_from_frequency_index
********************************************************************************
* Summary:
*    A function that returns the sample rate as an integer based on the sample
*    rate option index input. 
*
* Parameters:
*    frequency_index: sample rate index selected. 
*
* Return:
*    sample rate in integer form. 
*
*******************************************************************************/
int pdm_pcm_get_frequency_from_frequency_index(SAMPLE_RATE frequency_index)
{
    int frequency;
    uint32_t frequency_options[SAMPLE_RATE_48000+1] = {8000, 16000, 22050, 44100, 48000};
    if ((SAMPLE_RATE_48000+1) > frequency_index)
    {
        frequency = frequency_options[frequency_index];
    }
    else 
    {
        /* Use default */
        frequency = frequency_options[0];
    }
    return frequency;
}

/*******************************************************************************
* Function Name: pdm_pcm_clear_data_ready_flag
********************************************************************************
* Summary:
*    A function that clears the data ready flag for a given handle.   
*
* Parameters:
*   handle: Handle for the PDM-PCM interface.
*
* Return:
*
*******************************************************************************/
void pdm_pcm_clear_data_ready_flag(pdm_pcm handle)
{
    handle->have_data = false;
}

/*******************************************************************************
* Function Name: pdm_pcm_get_frame_count
********************************************************************************
* Summary:
*    A function that returns the frame count size for a given handle.    
*
* Parameters:
*   handle: Handle for the PDM-PCM interface.
*
* Return:
*   frame count
*
*******************************************************************************/
uint32_t pdm_pcm_get_frame_count(pdm_pcm handle)
{
    uint8_t num_channels = handle->config->mode +1;
    return FRAME_SIZE / num_channels;
}

#endif /* IM_ENABLE_PDM_PCM */

/* [] END OF FILE */