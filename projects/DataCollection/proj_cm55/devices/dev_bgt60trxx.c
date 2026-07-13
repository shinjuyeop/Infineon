/******************************************************************************
* File Name:   dev_bgt60trxx.c
*
* Description: This file implements the interface with the radar sensor.
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

#ifdef IM_ENABLE_BGT60TRXX

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <cybsp.h>
#include <cy_pdl.h>
#include "xensiv_bgt60trxx_mtb.h"
#include "protocol/protocol.h"
#include "protocol/pb_encode.h"
#include "clock.h"
#include "dev_bgt60trxx_settings.h"
#include "dev_bgt60trxx.h"
#include "xensiv_bgt60trxx_platform.h"

/*******************************************************************************
* Macros
*******************************************************************************/

#define DEV_BGT60TRXX_OPTION_PRESET     (1)
#define DEV_BGT60TRXX_OPTION_CUSTOM_SET (2)

#define XENSIV_BGT60TRXX_SPI_BURST_MODE_CMD             (0xFF000000UL)  /* Write addr 7f<<1 | 0x01; */
#define XENSIV_BGT60TRXX_SPI_BURST_MODE_SADR_POS        (17U)
#define XENSIV_BGT60TRXX_SPI_BURST_MODE_RWB_POS         (16U)
#define XENSIV_BGT60TRXX_SPI_BURST_MODE_LEN_POS         (9U)

#define DMA_CHANNEL_SPI_TX 14
#define DMA_CHANNEL_SPI_RX 15
#define SPI_FIFO_USE 32
/*******************************************************************************
* Preset configuration variables and structs.
*******************************************************************************/
/* This value could be modified later to increase response time if the data
* rate from the radar is low. Note that the minimum size is the size of one frame.
*/

#define DEFAULT_FIFO_SETTING 12288 // This is in number of samples.
struct radar_config radar_configs[5];
int current_config_g = 0;

/*******************************************************************************
* Types
*******************************************************************************/

typedef struct {
    xensiv_bgt60trxx_mtb_t bgt60_obj;
    int skipped_frames;
} dev_bgt60trxx_t;

struct xensiv_bgt60trxx_type
{
    uint32_t fifo_addr;
    uint16_t fifo_size;
    xensiv_bgt60trxx_device_t device;
};

/*******************************************************************************
 * Global Variables
 ******************************************************************************/
int radar_sent_frames=0;
extern cy_stc_scb_spi_context_t SPI_context;

CY_SECTION(".cy_socmem_data") uint16_t bgt60_send_buffer[16384]={0}; // Max number of samples that fit in the radar fifo.

/******************************************************************************
 * DMA-Descriptors for the radar-dma
 *****************************************************************************/
CY_SECTION(".cy_socmem_data") CY_ALIGN(32) cy_stc_dma_descriptor_t dma_desc_tx[16]={0};
CY_SECTION(".cy_socmem_data") CY_ALIGN(32) cy_stc_dma_descriptor_t dma_desc_rx[16]={0};

/* This is used for the tx data source when feeding the clock output.
** Some notes: The SPI wire require transport of outgoing data to
** get some data back. Data will be in a shiftregister that is driven
** by a clock. Some dummy data will thus be shifted in. The value
** of the data is "Dont Care" in this case.
*/
CY_SECTION(".cy_socmem_data") uint16_t dummy_tx_buffer = 0x1122;

/* The SCB used for SPI is SCB3.*/
volatile uint32_t* PERI_SCB3_TX_FIFO_WR         = (uint32_t*)0x429B0240;
volatile uint32_t* PERI_SCB3_RX_FIFO_RD         = (uint32_t*)0x429B0340;

/*******************************************************************************
* Function Prototypes
*******************************************************************************/
static bool _config_hw(dev_bgt60trxx_t *radar, int preset);
static bool _init_hw(dev_bgt60trxx_t *radar);
static bool _configure_streams(protocol_t* protocol, int device, void* arg);
static void _start_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg);
static void _stop_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg);
static void _poll_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg);

static void _load_presets();
static void _load_custom_values( uint8_t* data_blob );

static bool _write_payload(
    protocol_t* protocol,
    int device_id,
    int stream_id,
    int frame_count,
    int total_bytes,
    pb_ostream_t* ostream,
    void* arg);

/*******************************************************************************
* Function Name: _configure_dma_tx_descriptors
********************************************************************************
* Summary:
*    configuring the dma tx descriptors
*
* Parameters:
*   dma_desc[]   DMA descriptor structure
* Return:
*   None.
*
*******************************************************************************/
static void _configure_dma_tx_descriptors( cy_stc_dma_descriptor_t dma_desc[] ){

    int frames = radar_configs[current_config_g].fifo_int_level   /   radar_configs[current_config_g].num_samples_per_frame;
    int actual_samples = frames * radar_configs[current_config_g].num_samples_per_frame;

    /* We like to unload the number of frames from the radar.
    ** The total amount of samples is calculated upon this value.
    ** Here we only care about the number of samples we need to read out
    ** from the radar. The spi-fifo is ideally set to 32 samples per burst.
    */
    int descriptor = 0;
    int samples_to_do = actual_samples;
    while ( samples_to_do )
    {

        /* Common stuff for each new descriptor.
         * In general a review of any POSSIBLE uninitialized fields should be done!
         */
        Cy_DMA_Descriptor_SetRetrigger(&dma_desc[descriptor], CY_DMA_WAIT_FOR_REACT);

        /* Perhaps a 2D transfer is better. 384*32 words.*/
        Cy_DMA_Descriptor_SetDescriptorType(&dma_desc[descriptor], CY_DMA_2D_TRANSFER);

        /*Each trigger should only run an X-loop */
        Cy_DMA_Descriptor_SetTriggerInType(&dma_desc[descriptor], CY_DMA_X_LOOP);

        Cy_DMA_Descriptor_SetDataSize(&dma_desc[descriptor], CY_DMA_HALFWORD);
        Cy_DMA_Descriptor_SetSrcTransferSize(&dma_desc[descriptor], CY_DMA_TRANSFER_SIZE_DATA);
        Cy_DMA_Descriptor_SetDstTransferSize(&dma_desc[descriptor], CY_DMA_TRANSFER_SIZE_WORD ); /* Bus is 32 bit  wide! */

        Cy_DMA_Descriptor_SetXloopSrcIncrement(&dma_desc[descriptor], 0);
        Cy_DMA_Descriptor_SetXloopDstIncrement(&dma_desc[descriptor], 0);

        Cy_DMA_Descriptor_SetYloopSrcIncrement(&dma_desc[descriptor], 0);
        Cy_DMA_Descriptor_SetYloopDstIncrement(&dma_desc[descriptor], 0);

        Cy_DMA_Descriptor_SetInterruptType(&dma_desc[descriptor], CY_DMA_DESCR_CHAIN );

        Cy_DMA_Descriptor_SetSrcAddress(&dma_desc[descriptor], &dummy_tx_buffer );
        Cy_DMA_Descriptor_SetDstAddress(&dma_desc[descriptor], (void*)PERI_SCB3_TX_FIFO_WR);

        if ( samples_to_do < SPI_FIFO_USE ){
            /* Create a special handling for this. */

            /* Code in this case is not tested. Redundant code due to keep the similarity
            ** with main part below in the else{} case. */
            int y_count = 1;
            Cy_DMA_Descriptor_SetXloopDataCount(&dma_desc[descriptor], samples_to_do);
            Cy_DMA_Descriptor_SetYloopDataCount(&dma_desc[descriptor], y_count);
            samples_to_do -= samples_to_do*y_count;
        }
        else
        {
            /* We can have max 256 y-loop counter.*/
            int y_count = samples_to_do/SPI_FIFO_USE;
            if ( y_count > 256 ) y_count=256;

            Cy_DMA_Descriptor_SetXloopDataCount(&dma_desc[descriptor], SPI_FIFO_USE);
            Cy_DMA_Descriptor_SetYloopDataCount(&dma_desc[descriptor], y_count);
            samples_to_do -= SPI_FIFO_USE*y_count;
        }

        if ( samples_to_do ){
            Cy_DMA_Descriptor_SetNextDescriptor(&dma_desc[descriptor], &dma_desc[descriptor+1] );
            /* Increment to the next descriptor.*/
            descriptor++;
        }
        else
        {
            /* Finalize with this descriptor by setting round robin.*/
            Cy_DMA_Descriptor_SetNextDescriptor(&dma_desc[descriptor], &dma_desc[0] );
            /* Disable any  more transfers. This is needed for the sending at least. */
            Cy_DMA_Descriptor_SetChannelState(&dma_desc[descriptor], CY_DMA_CHANNEL_DISABLED );
        }

        /*make dummy_tx_buffer cacheable. Need to clean tx desc D-cache*/
        SCB_CleanDCache_by_Addr((uint32_t*)&dma_desc[descriptor], sizeof(dma_desc[descriptor]));

    }

}

/*******************************************************************************
* Function Name: _configure_dma_rx_descriptors
********************************************************************************
* Summary:
*    This is our destination buffer. Use the same concept as above for the sizes.
*
* Parameters:
*   dest_address address of bgt60_send_buffer
*   dma_desc[]   DMA descriptor structure
* Return:
*   None.
*
*******************************************************************************/

static void _configure_dma_rx_descriptors( void* dest_address, cy_stc_dma_descriptor_t dma_desc[] )
{

    int frames = radar_configs[current_config_g].fifo_int_level / radar_configs[current_config_g].num_samples_per_frame;
    int actual_samples = frames * radar_configs[current_config_g].num_samples_per_frame;

    int addr_offset = 0;

    /* We like to unload the number of frames from the radar.
    * The total amount of samples is calculated upon this value.
    * Here we only care about the number of samples we need to read out
    * from the radar. The spi-fifo is ideally set to 32 samples per burst.
    */
    int descriptor = 0;
    int samples_to_do = actual_samples;
    while ( samples_to_do )
    {

        /* Common stuff for each new descriptor.*/
        /* In general a review of any POSSIBLE uninitialized fields should be done!*/
        Cy_DMA_Descriptor_SetRetrigger(&dma_desc[descriptor], CY_DMA_WAIT_FOR_REACT);

        /* Perhaps a 2D transfer is better. 384*32 words.*/
        Cy_DMA_Descriptor_SetDescriptorType(&dma_desc[descriptor], CY_DMA_2D_TRANSFER);

        /*Each trigger should only run an X-loop*/
        Cy_DMA_Descriptor_SetTriggerInType(&dma_desc[descriptor], CY_DMA_X_LOOP);

        Cy_DMA_Descriptor_SetDataSize(&dma_desc[descriptor], CY_DMA_HALFWORD);
        Cy_DMA_Descriptor_SetSrcTransferSize(&dma_desc[descriptor], CY_DMA_TRANSFER_SIZE_WORD);
        Cy_DMA_Descriptor_SetDstTransferSize(&dma_desc[descriptor], CY_DMA_TRANSFER_SIZE_DATA );

        Cy_DMA_Descriptor_SetXloopSrcIncrement(&dma_desc[descriptor], 0);
        Cy_DMA_Descriptor_SetXloopDstIncrement(&dma_desc[descriptor], 1); // One 16-bit word

        Cy_DMA_Descriptor_SetYloopSrcIncrement(&dma_desc[descriptor], 0);
        Cy_DMA_Descriptor_SetYloopDstIncrement(&dma_desc[descriptor], SPI_FIFO_USE);

        Cy_DMA_Descriptor_SetInterruptType(&dma_desc[descriptor], CY_DMA_DESCR_CHAIN );

        Cy_DMA_Descriptor_SetSrcAddress(&dma_desc[descriptor], (void*)PERI_SCB3_RX_FIFO_RD);
        Cy_DMA_Descriptor_SetDstAddress(&dma_desc[descriptor], (void*)((int)dest_address + addr_offset));

        if ( samples_to_do < SPI_FIFO_USE )
        {
            /* Create a special handling for this.*/
            /* Code in this case is not tested. Redundant code due to keep the similarity
            ** with main part. */
            int y_count = 1;
            Cy_DMA_Descriptor_SetXloopDataCount(&dma_desc[descriptor], samples_to_do);
            Cy_DMA_Descriptor_SetYloopDataCount(&dma_desc[descriptor], y_count);
            addr_offset += samples_to_do*2*y_count;
            samples_to_do -= samples_to_do*y_count;
        }
        else
        {
            /* We can have max 256 y-loop counter.*/
            int y_count = samples_to_do/SPI_FIFO_USE;
            if ( y_count > 256 ) y_count=256;

            Cy_DMA_Descriptor_SetXloopDataCount(&dma_desc[descriptor], SPI_FIFO_USE);
            Cy_DMA_Descriptor_SetYloopDataCount(&dma_desc[descriptor], y_count);
            addr_offset += SPI_FIFO_USE*2*y_count;
            samples_to_do -= SPI_FIFO_USE*y_count;
        }

        if ( samples_to_do )
        {
            Cy_DMA_Descriptor_SetNextDescriptor(&dma_desc[descriptor], &dma_desc[descriptor+1] );
            /* Increment to the next descriptor.*/
            descriptor++;
        }
        else
        {
            /* Finalize with this descriptor by setting round robin.*/
            Cy_DMA_Descriptor_SetNextDescriptor(&dma_desc[descriptor], &dma_desc[0] );
            /* Disable any furter transfers.*/
            Cy_DMA_Descriptor_SetChannelState(&dma_desc[descriptor], CY_DMA_CHANNEL_DISABLED );
        }

        /* dest_address cacheable. Need to clean rx - desc d-cache*/
        SCB_CleanDCache_by_Addr((uint32_t*)&dma_desc[descriptor], sizeof(dma_desc[descriptor]));

    }
}

/*******************************************************************************
* Function Name: enable_dma_triggers
********************************************************************************
* Summary:
*    A function used to enable the dma
*
* Parameters:
*   dest_address address of bgt60_send_buffer
*
* Return:
*   None.
*
*******************************************************************************/
void enable_dma_triggers( void* dest_address )
{

    _configure_dma_tx_descriptors( dma_desc_tx );
    _configure_dma_rx_descriptors( dest_address, dma_desc_rx );

    Cy_DMA_Channel_SetDescriptor(DW0, DMA_CHANNEL_SPI_TX, &dma_desc_tx[0]);
    Cy_DMA_Channel_SetPriority(DW0, DMA_CHANNEL_SPI_TX, 0);

    Cy_DMA_Channel_SetDescriptor(DW0, DMA_CHANNEL_SPI_RX, &dma_desc_rx[0]);
    Cy_DMA_Channel_SetPriority(DW0, DMA_CHANNEL_SPI_RX, 0);

    Cy_DMA_Enable(DW0);

    /* Enable the triggering of DMA transfers trough the trigger matrix.
    * Note that there is a PERI_TR_CMD register that can be used to
    * do a manual triggering if desired for testing or other purposes.
    */
    Cy_TrigMux_Select( PERI_0_TRIG_OUT_1TO1_0_SCB3_TX_TO_PDMA0_TR_IN14, false, TRIGGER_TYPE_LEVEL );
    Cy_TrigMux_Select( PERI_0_TRIG_OUT_1TO1_0_SCB3_RX_TO_PDMA0_TR_IN15, false, TRIGGER_TYPE_LEVEL );

}

/*******************************************************************************
* Function Name: _init_hw
********************************************************************************
* Summary:
*    A function used to initialize the Radar sensor
*
* Parameters:
*   None
*
* Return:
*     The status of the initialization.
*
*******************************************************************************/
static bool _init_hw(dev_bgt60trxx_t *radar)
{
    cy_rslt_t result;
    radar->skipped_frames = 0;

   /* Enable the RADAR. */
    radar->bgt60_obj.iface.scb_inst = RADAR_SPI_CONTROLLER_HW;
    radar->bgt60_obj.iface.spi = &SPI_context;
    radar->bgt60_obj.iface.sel_port = CYBSP_RSPI_CS_PORT;
    radar->bgt60_obj.iface.sel_pin = CYBSP_RSPI_CS_PIN;
    radar->bgt60_obj.iface.rst_port = CYBSP_RADAR_RESET_PORT;
    radar->bgt60_obj.iface.rst_pin = CYBSP_RADAR_RESET_PIN;
    radar->bgt60_obj.iface.irq_port = CYBSP_RADAR_INT_PORT;
    radar->bgt60_obj.iface.irq_pin = CYBSP_RADAR_INT_PIN;
    radar->bgt60_obj.iface.irq_num = CYBSP_RADAR_INT_IRQ;

    /* Reduce drive strength to improve EMI */
    Cy_GPIO_SetSlewRate(CYBSP_RSPI_MOSI_PORT, CYBSP_RSPI_MOSI_PIN, CY_GPIO_SLEW_FAST);
    Cy_GPIO_SetDriveSel(CYBSP_RSPI_MOSI_PORT, CYBSP_RSPI_MOSI_PIN, CY_GPIO_DRIVE_1_8);
    Cy_GPIO_SetSlewRate(CYBSP_RSPI_CLK_PORT, CYBSP_RSPI_CLK_PIN, CY_GPIO_SLEW_FAST);
    Cy_GPIO_SetDriveSel(CYBSP_RSPI_CLK_PORT, CYBSP_RSPI_CLK_PIN, CY_GPIO_DRIVE_1_8);

    result = xensiv_bgt60trxx_mtb_init(&radar->bgt60_obj, radar_configs[0].register_list, radar_configs[0].number_of_regs);
    if (CY_RSLT_SUCCESS == result)
    {
        printf("RADAR: Initialized device.\r\n");
        xensiv_bgt60trxx_mtb_interrupt_init(&radar->bgt60_obj, radar_configs[0].fifo_int_level);
    }
    else
    {
        printf("ERROR: xensiv_bgt60trxx_mtb_init failed\n");
        return false;
    }

    Cy_SysLib_Delay(1000);

    /* Enable the radar triggers*/
    enable_dma_triggers( &bgt60_send_buffer );

    return true;
}



/*******************************************************************************
 * Platform functions implementation
 ********************************************************************************/
__STATIC_INLINE void spi_set_data_width(CySCB_Type* base, uint32_t data_width)
{
    CY_ASSERT(CY_SCB_SPI_IS_DATA_WIDTH_VALID(data_width));

    CY_REG32_CLR_SET(SCB_TX_CTRL(base),
                     SCB_TX_CTRL_DATA_WIDTH,
                     (uint32_t)data_width - 1U);
    CY_REG32_CLR_SET(SCB_RX_CTRL(base),
                     SCB_RX_CTRL_DATA_WIDTH,
                     (uint32_t)data_width - 1U);
}

/*******************************************************************************
* Function Name: _configure_streams
********************************************************************************
* Summary:
*   Called when the device is configured or re-configured.
*
* Parameters:
*   protocol: Pointer the protocol handle
*   device: The device index
*   arg: Pointer the device struct (dev_bgt60trxx_t)
*
* Return:
*   True to keep the connection open, otherwise false.
*
*******************************************************************************/
static bool _configure_streams(protocol_t* protocol, int device, void* arg)
{
    UNUSED(arg);
    int preset_index=0;

    if(protocol_clear_streams(protocol, device) != PROTOCOL_STATUS_SUCCESS)
    {
        protocol_set_device_status(
                protocol,
                device,
                protocol_DeviceStatus_DEVICE_STATUS_ERROR,
                "Failed to clear streams.");
        return true;
    }

    /* Check if custom option exist. */
    int status;
    pb_bytes_array_t* pb_bytes;

    status = protocol_get_option_blob(protocol, device,
                DEV_BGT60TRXX_OPTION_CUSTOM_SET,
                &pb_bytes);

    if (PROTOCOL_STATUS_SUCCESS != status)
    {
        printf("Getting blob option failed\n");
    }

    if ( pb_bytes == NULL )
    {
        /*printf("Don't have any blob option.\n"); */
    }
    else
    {
        _load_custom_values( (uint8_t*)pb_bytes);
    }

    protocol_get_option_oneof(protocol, device, DEV_BGT60TRXX_OPTION_PRESET, &preset_index);
    current_config_g = preset_index;

    int stream = protocol_add_stream(
        protocol,
        device,
        "Radar",
        protocol_StreamDirection_STREAM_DIRECTION_OUTPUT,
        protocol_DataType_DATA_TYPE_S16,
        radar_configs[preset_index].frame_rate, /* Frequency in number of frames per second. */
                                                /* This needs to be rounded to some integer value. */
        1,      /* Max frame_count? */
        "???");

    if(stream < 0)
    {
        protocol_set_device_status(
                protocol,
                device,
                protocol_DeviceStatus_DEVICE_STATUS_ERROR,
                "Failed to add streams.");
        return true;
    }

    protocol_add_stream_rank(
            protocol,
            device,
            stream,
            "Chirp",
            radar_configs[preset_index].chirps_per_frame,
            NULL);

    protocol_add_stream_rank(
            protocol,
            device,
            stream,
            "Sample",
            radar_configs[preset_index].samples_per_chirp,
            NULL);

    protocol_add_stream_rank(
            protocol,
            device,
            stream,
            "Antenna",
            radar_configs[preset_index].rx_antennas,
            NULL);

    return true;
}
/******************************************************************************
* Function Name: _config_hw
********************************************************************************
* Summary:
*   Configures radar using presets.
*
* Parameters:
*   dev: Pointer to the dev_bgt60trxx_t device handle.
*   preset: One of the presets available.
*
* Return:
*   True if configuration is successful, otherwise false.
*
*******************************************************************************/
static bool _config_hw(dev_bgt60trxx_t *radar, int preset)
{

    int status;

/* Reinitialize the radar using the selected radar_settings. */
    status = xensiv_bgt60trxx_config(&radar->bgt60_obj.dev,
                                    radar_configs[preset].register_list,
                                    radar_configs[preset].number_of_regs);
    if ( status != XENSIV_BGT60TRXX_STATUS_OK )
    {
        printf("Radar config HW failed\n");
        return false;
    }

/* Update the fifo limit for the interrupts. */
    status = xensiv_bgt60trxx_set_fifo_limit(&radar->bgt60_obj.dev, radar_configs[preset].fifo_int_level);
    if ( status != XENSIV_BGT60TRXX_STATUS_OK )
    {
        printf("Radar config fifo limit failed\n");
        return false;
    }

    enable_dma_triggers( &bgt60_send_buffer );
    return true;
}

/*******************************************************************************
* Function Name: _start_streams
********************************************************************************
* Summary:
*   Called when streaming is started. This may also initialize the device.
*
* Parameters:
*   protocol: Pointer the protocol handle
*   device: The device index
*   ostream: Pointer to the output stream to write to
*   arg: Pointer the device struct (dev_bgt60trxx_t)
*
*******************************************************************************/
static void _start_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg)
{
    dev_bgt60trxx_t* radar = (dev_bgt60trxx_t*)arg;
    int preset_index;


    UNUSED(radar);
    UNUSED(ostream);

    protocol_get_option_oneof(protocol, device, DEV_BGT60TRXX_OPTION_PRESET, &preset_index);
    current_config_g = preset_index;

    if(!_config_hw(radar, preset_index) )
    {
        protocol_set_device_status(
            protocol,
            device,
            protocol_DeviceStatus_DEVICE_STATUS_ERROR,
            "Failed to configure hardware.");
    }

    /* TODO: enable radar */
    if (xensiv_bgt60trxx_soft_reset(&radar->bgt60_obj.dev, XENSIV_BGT60TRXX_RESET_FIFO) != XENSIV_BGT60TRXX_STATUS_OK)
    {
        /* Error */
        printf("Fifo reset error error.\r\n");
    }

    if (xensiv_bgt60trxx_start_frame(&radar->bgt60_obj.dev, true) != XENSIV_BGT60TRXX_STATUS_OK)
    {
        /* Error */
        printf("Start frame error.\r\n");
    }
    else
    {
       printf("\nStreaming RADAR data.\n");
    }

    protocol_set_device_status(
            protocol,
            device,
            protocol_DeviceStatus_DEVICE_STATUS_ACTIVE,
            "Device is streaming");

    radar_sent_frames = 0;

}

/*******************************************************************************
* Function Name: _stop_streams
********************************************************************************
* Summary:
*  Called when streaming is stopped.
*
* Parameters:
*  protocol: Pointer the protocol handle
*  device: The device index
*  ostream: Pointer to the output stream to write to
*  arg: Pointer the device struct (dev_bgt60trxx_t)
*
*******************************************************************************/
static void _stop_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg)
{

    dev_bgt60trxx_t* radar = (dev_bgt60trxx_t*)arg;

    UNUSED(ostream);

    printf("Stopping RADAR data Streams!\n");

    /* TODO: disable radar */
    if (xensiv_bgt60trxx_start_frame(&radar->bgt60_obj.dev, false) != XENSIV_BGT60TRXX_STATUS_OK)
    {
        /* Error */
    }

    protocol_set_device_status(
            protocol,
            device,
            protocol_DeviceStatus_DEVICE_STATUS_READY,
            "Device stopped");

    Cy_DMA_Channel_Disable(DW0, DMA_CHANNEL_SPI_RX);
    Cy_DMA_Channel_Disable(DW0, DMA_CHANNEL_SPI_TX);

    radar_sent_frames = 0;

}


/*******************************************************************************
* Function Name: _poll_streams
********************************************************************************
* Summary:
*  Called periodically to send data messages.
*
* Parameters:
*  protocol: Pointer the protocol handle
*  device: The device index
*  ostream: Pointer to the output stream to write to
*  arg: Pointer the device struct (dev_bgt60trxx_t)
*
*******************************************************************************/

enum spi_state {
    NONE = 0,
    IDLE,
    SPI_MODE_CHANGE,
    BURST_PENDING,
    FIFO_READ_PENDING,
    FIFO_READ_DONE
};

enum spi_state sp_state = IDLE;

static void _poll_streams(protocol_t* protocol, int device, pb_ostream_t* ostream, void* arg)
{

    dev_bgt60trxx_t* radar = (dev_bgt60trxx_t*)arg;

    xensiv_bgt60trxx_t* p_device = &radar->bgt60_obj.dev;

    void* iface = p_device->iface;
    xensiv_bgt60trxx_mtb_iface_t* mtb_iface = iface;

    /* If any dma channel for reading or writing is enabled we can skip the rest of
    ** this polling. The transfer status only checks if the command is sent or not.
    ** The actual raw data transport also needs to be monitored and we must ensure
    ** we are NOT DISTURBING that by trying to send a new command! Raw data transport
    ** is detected by the activity of the DMA channels.
    */

    if (0UL != (CY_SCB_SPI_TRANSFER_ACTIVE &
                Cy_SCB_SPI_GetTransferStatus(mtb_iface->scb_inst, mtb_iface->spi))){
        return;
    }

    if ( Cy_DMA_Channel_IsEnabled(DW0, DMA_CHANNEL_SPI_RX) ||
         Cy_DMA_Channel_IsEnabled(DW0, DMA_CHANNEL_SPI_TX) ){
        return;
    }

    switch (sp_state)
    {
        case IDLE:
        {
            /* This line comes from the radar IC and indicates the fifo
            ** on the radar have passed it's triggering level.
            **/
            int a;
            a = Cy_GPIO_Read(CYBSP_RADAR_INT_PORT,CYBSP_RADAR_INT_NUM);

            if (a)
            {

                /* Make Chip select */
                xensiv_bgt60trxx_platform_spi_cs_set(iface, false);

                /* Set the burstmode on the radar. */
                static uint32_t command;
                uint32_t* head = (uint32_t*)&command;
                const xensiv_bgt60trxx_mtb_iface_t* mtb_iface = iface;

                *head = XENSIV_BGT60TRXX_SPI_BURST_MODE_CMD |
                        (p_device->type->fifo_addr << XENSIV_BGT60TRXX_SPI_BURST_MODE_SADR_POS) |   /* Addr 0x60.. */
                        (0 << XENSIV_BGT60TRXX_SPI_BURST_MODE_RWB_POS) |    /* Read mode    */
                        (0 << XENSIV_BGT60TRXX_SPI_BURST_MODE_LEN_POS);     /* Read until termination. */

                /* Ensure correct byte order for sending the command. */
                *head = xensiv_bgt60trxx_platform_word_reverse(*head);

                spi_set_data_width(mtb_iface->scb_inst, 8U);
                Cy_SCB_SetByteMode(mtb_iface->scb_inst, true);

                Cy_SCB_SPI_Transfer(mtb_iface->scb_inst, (uint8_t*)head,
                                                                NULL, /* Received data is discarded. */
                                                                4,
                                                                mtb_iface->spi);
                sp_state = SPI_MODE_CHANGE;
                return;
            }
            break;
        }

        case SPI_MODE_CHANGE:
        {
            /* Each data sample from the radar is 12 bit. This appears as 16 bit
            ** in the SPI-fifo. */
            spi_set_data_width(mtb_iface->scb_inst, 12U);
            Cy_SCB_SetByteMode(mtb_iface->scb_inst, false);

            /* Prepare the reception of data by starting the RX channel */
            Cy_SCB_SetRxFifoLevel(mtb_iface->scb_inst, 31);
            Cy_DMA_Channel_Enable(DW0, DMA_CHANNEL_SPI_RX);

            /* Start a DMA transfer for writing SPI data */
            Cy_SCB_SetTxFifoLevel(mtb_iface->scb_inst, 1);
            Cy_DMA_Channel_Enable(DW0, DMA_CHANNEL_SPI_TX);

            sp_state = BURST_PENDING;
            break;
        }

        case BURST_PENDING:
        {
            /* When the DMA transfer is ready, the channels will automatically get disabled. */
            sp_state = FIFO_READ_PENDING;
            break;
        }

        case FIFO_READ_PENDING:
        {
            /* Clear the Chip Select. */
            xensiv_bgt60trxx_platform_spi_cs_set(iface, true);
            sp_state = FIFO_READ_DONE;  /* This also indicates that buffer copying below can be done.*/
            break;
        }

        case FIFO_READ_DONE:
        {
            sp_state = IDLE;
            break;
        }

        default:
        {
            break;
        }
    }

    if(sp_state == FIFO_READ_DONE)
    {

        int frames=0;
        frames = radar_configs[current_config_g].fifo_int_level / radar_configs[current_config_g].num_samples_per_frame;

/*********************************************************************************/

        if ( frames )
        {
            protocol_send_data_chunk(
                protocol,
                device,
                0,
                frames,
                radar->skipped_frames,
                ostream,
                _write_payload);

        }
        radar_sent_frames+=frames;
        radar->skipped_frames = 0;

        static int sent_counter=0;
        if (sent_counter == 10)
        {
            sent_counter=0;
        }
        else{
            sent_counter++;
        }


    }
}

/*******************************************************************************
* Function Name: _write_payload
********************************************************************************
* Summary:
*  Used by protocol_send_data_chunk to write the actual data.
*
* Parameters:
*  protocol: Pointer the protocol handle
*  device_id: The device index
*  stream_id: The stream index
*  frame_count: Number of frames to write
*  total_bytes: Total number of bytes to write (= frame_count * sizeof(type) * frame_shape.flat)
*  ostream: Pointer to the output stream to write to
*  arg: Pointer the device struct (dev_bgt60trxx_t)
*
* Return:
*   True if data writing is successful, otherwise false.
*
*******************************************************************************/
static bool _write_payload(
    protocol_t* protocol,
    int device_id,
    int stream_id,
    int frame_count,
    int total_bytes,
    pb_ostream_t* ostream,
    void* arg)
{
    UNUSED(protocol);
    UNUSED(stream_id);
    UNUSED(frame_count);
    UNUSED(protocol);

   /*Invalidating data in DCache*/
    SCB_InvalidateDCache_by_Addr((uint32_t*)&bgt60_send_buffer, total_bytes);

    if (!pb_write(ostream, (const pb_byte_t *)bgt60_send_buffer, total_bytes))
    {
        return false;
    }

    return true;
}

/*******************************************************************************
* Summary:
*   This gets a binary data_blob via the protocol and transfer this values to
*   setting index 4. This 'slot' is initially identical to the first but is
*   overwritten by custom data if set.
*******************************************************************************/
static void _load_custom_values( uint8_t* data_blob )
{

    /* Binary file is prepended with a 2 byte size that should be skipped. */
    uint32_t* values = (uint32_t*)(data_blob+2);

/* Ensure this is updated if the binary file is generated diffrently. */
#define OFF_REGS_START                      16
#define OFF_CONF_NUM_REGS                   0
#define OFF_CONF_REGS_START                 1
#define OFF_CONF_START_FREQ_LO_HZ           2
#define OFF_CONF_START_FREQ_HI_HZ           3
#define OFF_CONF_END_FREQ_LO_HZ             4
#define OFF_CONF_END_FREQ_HI_HZ             5
#define OFF_CONF_NUM_SAMPLES_PER_CHIRP      6
#define OFF_CONF_NUM_CHIRPS_PER_FRAME       7
#define OFF_CONF_NUM_RX_ANTENNAS            8
#define OFF_CONF_NUM_TX_ANTENNAS            9
#define OFF_CONF_SAMPLE_RATE                10
#define OFF_CONF_CHIRP_REPETITION_TIME_S    11
#define OFF_CONF_FRAME_REPETITION_TIME_S    12

    /* Misaligned 64 bit data needs bytewise copying.  */
    memcpy( (char*)&radar_configs[4].start_freq, (char*)&values[OFF_CONF_START_FREQ_LO_HZ], 8 );
    memcpy( (char*)&radar_configs[4].end_freq, (char*)&values[OFF_CONF_END_FREQ_LO_HZ], 8 );

    radar_configs[4].samples_per_chirp =     values[OFF_CONF_NUM_SAMPLES_PER_CHIRP];
    radar_configs[4].chirps_per_frame =      values[OFF_CONF_NUM_CHIRPS_PER_FRAME];
    radar_configs[4].rx_antennas =           values[OFF_CONF_NUM_RX_ANTENNAS];
    radar_configs[4].tx_antennas =           values[OFF_CONF_NUM_TX_ANTENNAS];
    radar_configs[4].sample_rate =           values[OFF_CONF_SAMPLE_RATE];

    radar_configs[4].chirp_repetition_time = *(float*)&values[OFF_CONF_CHIRP_REPETITION_TIME_S];  /* 32bit float */
    radar_configs[4].frame_repetition_time = *(float*)&values[OFF_CONF_FRAME_REPETITION_TIME_S];  /* 32bit float */

    radar_configs[4].frame_rate = (int)(0.5 + 1.0/(double)radar_configs[4].frame_repetition_time);

    radar_configs[4].num_samples_per_frame =
            radar_configs[4].samples_per_chirp *
            radar_configs[4].chirps_per_frame *
            radar_configs[4].rx_antennas;

    radar_configs[4].fifo_int_level = DEFAULT_FIFO_SETTING; /* Since the set_fifo_limit function takes samples this is only half. */
    radar_configs[4].number_of_regs =           values[OFF_CONF_NUM_REGS];
    radar_configs[4].register_list = &values[values[OFF_CONF_REGS_START]];

    return;

}
/*******************************************************************************
* Summary:
*   This just load preset from header file constants to enable selection of
*   various radar presets.
* See dev_bgt60trxx_settings.h for information of the sources of radar settings.
*
*******************************************************************************/
static void _load_presets()
{
/* Config 1 Default testing configuration. */
    radar_configs[0].start_freq =            P0_XENSIV_BGT60TRXX_CONF_START_FREQ_HZ;
    radar_configs[0].end_freq =              P0_XENSIV_BGT60TRXX_CONF_END_FREQ_HZ;
    radar_configs[0].samples_per_chirp =     P0_XENSIV_BGT60TRXX_CONF_NUM_SAMPLES_PER_CHIRP;
    radar_configs[0].chirps_per_frame =      P0_XENSIV_BGT60TRXX_CONF_NUM_CHIRPS_PER_FRAME;
    radar_configs[0].rx_antennas =           P0_XENSIV_BGT60TRXX_CONF_NUM_RX_ANTENNAS;
    radar_configs[0].tx_antennas =           P0_XENSIV_BGT60TRXX_CONF_NUM_TX_ANTENNAS;
    radar_configs[0].sample_rate =           P0_XENSIV_BGT60TRXX_CONF_SAMPLE_RATE;
    radar_configs[0].chirp_repetition_time = P0_XENSIV_BGT60TRXX_CONF_CHIRP_REPETITION_TIME_S;
    radar_configs[0].frame_repetition_time = P0_XENSIV_BGT60TRXX_CONF_FRAME_REPETITION_TIME_S;

    radar_configs[0].frame_rate = (int)(0.5 + 1.0/(double)radar_configs[0].frame_repetition_time);
    radar_configs[0].num_samples_per_frame =
            radar_configs[0].samples_per_chirp *
            radar_configs[0].chirps_per_frame *
            radar_configs[0].rx_antennas;

    radar_configs[0].fifo_int_level = DEFAULT_FIFO_SETTING;


    radar_configs[0].number_of_regs =           P0_XENSIV_BGT60TRXX_CONF_NUM_REGS;
    radar_configs[0].register_list = register_list_p0;

/* Config 2 Gesture detection */
    radar_configs[1].start_freq =            P1_XENSIV_BGT60TRXX_CONF_START_FREQ_HZ;
    radar_configs[1].end_freq =              P1_XENSIV_BGT60TRXX_CONF_END_FREQ_HZ;
    radar_configs[1].samples_per_chirp =     P1_XENSIV_BGT60TRXX_CONF_NUM_SAMPLES_PER_CHIRP;
    radar_configs[1].chirps_per_frame =      P1_XENSIV_BGT60TRXX_CONF_NUM_CHIRPS_PER_FRAME;
    radar_configs[1].rx_antennas =           P1_XENSIV_BGT60TRXX_CONF_NUM_RX_ANTENNAS;
    radar_configs[1].tx_antennas =           P1_XENSIV_BGT60TRXX_CONF_NUM_TX_ANTENNAS;
    radar_configs[1].sample_rate =           P1_XENSIV_BGT60TRXX_CONF_SAMPLE_RATE;
    radar_configs[1].chirp_repetition_time = P1_XENSIV_BGT60TRXX_CONF_CHIRP_REPETITION_TIME_S;
    radar_configs[1].frame_repetition_time = P1_XENSIV_BGT60TRXX_CONF_FRAME_REPETITION_TIME_S;

    radar_configs[1].frame_rate = (int)(0.5 + 1.0/(double)radar_configs[1].frame_repetition_time);
    radar_configs[1].num_samples_per_frame =
            radar_configs[1].samples_per_chirp *
            radar_configs[1].chirps_per_frame *
            radar_configs[1].rx_antennas;

    radar_configs[1].fifo_int_level = DEFAULT_FIFO_SETTING;


    radar_configs[1].number_of_regs =           P1_XENSIV_BGT60TRXX_CONF_NUM_REGS;
    radar_configs[1].register_list = register_list_p1;

/* Config 3 Macro movements */
    radar_configs[2].start_freq =            P2_XENSIV_BGT60TRXX_CONF_START_FREQ_HZ;
    radar_configs[2].end_freq =              P2_XENSIV_BGT60TRXX_CONF_END_FREQ_HZ;
    radar_configs[2].samples_per_chirp =     P2_XENSIV_BGT60TRXX_CONF_NUM_SAMPLES_PER_CHIRP;
    radar_configs[2].chirps_per_frame =      P2_XENSIV_BGT60TRXX_CONF_NUM_CHIRPS_PER_FRAME;
    radar_configs[2].rx_antennas =           P2_XENSIV_BGT60TRXX_CONF_NUM_RX_ANTENNAS;
    radar_configs[2].tx_antennas =           P2_XENSIV_BGT60TRXX_CONF_NUM_TX_ANTENNAS;
    radar_configs[2].sample_rate =           P2_XENSIV_BGT60TRXX_CONF_SAMPLE_RATE;
    radar_configs[2].chirp_repetition_time = P2_XENSIV_BGT60TRXX_CONF_CHIRP_REPETITION_TIME_S;
    radar_configs[2].frame_repetition_time = P2_XENSIV_BGT60TRXX_CONF_FRAME_REPETITION_TIME_S;

    radar_configs[2].frame_rate = (int)(0.5 + 1.0/(double)radar_configs[2].frame_repetition_time);
    radar_configs[2].num_samples_per_frame =
            radar_configs[2].samples_per_chirp *
            radar_configs[2].chirps_per_frame *
            radar_configs[2].rx_antennas;

    radar_configs[2].fifo_int_level = DEFAULT_FIFO_SETTING;


    radar_configs[2].number_of_regs =           P2_XENSIV_BGT60TRXX_CONF_NUM_REGS;
    radar_configs[2].register_list = register_list_p2;

/* Config 4 Micro movements. */
    radar_configs[3].start_freq =            P3_XENSIV_BGT60TRXX_CONF_START_FREQ_HZ;
    radar_configs[3].end_freq =              P3_XENSIV_BGT60TRXX_CONF_END_FREQ_HZ;
    radar_configs[3].samples_per_chirp =     P3_XENSIV_BGT60TRXX_CONF_NUM_SAMPLES_PER_CHIRP;
    radar_configs[3].chirps_per_frame =      P3_XENSIV_BGT60TRXX_CONF_NUM_CHIRPS_PER_FRAME;
    radar_configs[3].rx_antennas =           P3_XENSIV_BGT60TRXX_CONF_NUM_RX_ANTENNAS;
    radar_configs[3].tx_antennas =           P3_XENSIV_BGT60TRXX_CONF_NUM_TX_ANTENNAS;
    radar_configs[3].sample_rate =           P3_XENSIV_BGT60TRXX_CONF_SAMPLE_RATE;
    radar_configs[3].chirp_repetition_time = P3_XENSIV_BGT60TRXX_CONF_CHIRP_REPETITION_TIME_S;
    radar_configs[3].frame_repetition_time = P3_XENSIV_BGT60TRXX_CONF_FRAME_REPETITION_TIME_S;

    radar_configs[3].frame_rate = (int)(0.5 + 1.0/(double)radar_configs[3].frame_repetition_time);
    radar_configs[3].num_samples_per_frame =
            radar_configs[3].samples_per_chirp *
            radar_configs[3].chirps_per_frame *
            radar_configs[3].rx_antennas;

    radar_configs[3].fifo_int_level = DEFAULT_FIFO_SETTING;

    radar_configs[3].number_of_regs =           P3_XENSIV_BGT60TRXX_CONF_NUM_REGS;
    radar_configs[3].register_list = register_list_p3;

/* The fifth configuration is the configuration used for customization and is loaded
 * with the same as the first as default. This will then be overwritten if a new
 * blob arrives. */

    memcpy( &radar_configs[4], &radar_configs[0], sizeof(radar_configs[0]) );

    return;
}

/*******************************************************************************
* Function Name: dev_bgt60trxx_register
********************************************************************************
* Summary:
*   Registers this device. This is the only exported symbol from this object.
*
* Parameters:
*   protocol: Pointer the protocol handle
*
* Returns:
*   True if registration is successful, otherwise false.
*
*******************************************************************************/
bool dev_bgt60trxx_register(protocol_t* protocol)
{
    int status;
    dev_bgt60trxx_t *radar = (dev_bgt60trxx_t*)malloc(sizeof(dev_bgt60trxx_t));

    if(radar == NULL)
    {
        printf("BGT60TRXX::malloc failed.\r\n");
        return false;
    }
    memset(radar, 0, sizeof(dev_bgt60trxx_t));

    _load_presets();

    if( !_init_hw(radar) )
    {
        printf("BGT60TRXX::hw_init failed.\r\n");
        return false;
    }

    device_manager_t manager = {
        .arg = radar,
        .configure_streams = _configure_streams,
        .start = _start_streams,
        .stop = _stop_streams,
        .poll = _poll_streams,
        .data_received = NULL /* has no input streams */
    };

    int device = protocol_add_device(
        protocol,
        protocol_DeviceType_DEVICE_TYPE_SENSOR,
        "Radar",
        "Radar device (bgt60trxx)",
        manager);

    if(device < 0)
    {
        printf("BGT60TRXX::adding device failed.\r\n");
        return false;
    }

    status = protocol_add_option_oneof(
        protocol,
        device,
        DEV_BGT60TRXX_OPTION_PRESET,
        "Use case",
        "Presets",
        0,
        (const char* []) { "Default", "Gesture", "Macro Presence", "Micro Presence", "Custom" },
        5);

    if(PROTOCOL_STATUS_SUCCESS != status)
    {
        printf("Adding option failed\n");
        return false;
    }

    status = protocol_add_option_blob(
        protocol,
        device,
        DEV_BGT60TRXX_OPTION_CUSTOM_SET,
        "Custom data",
        "Binary Custom data",
        NULL /*pb_bytes_array_t* default_value  */
        );

    if(PROTOCOL_STATUS_SUCCESS != status)
    {
        printf("Adding option custom set failed\n");
        return false;
    }

    if(!_configure_streams(protocol, device, manager.arg))
    {
        printf("BGT60TRXX::configure streams failed.\r\n");
        return false;
    }

    return true;
}

#endif /* IM_ENABLE_BGT60TRXX */
