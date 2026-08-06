[Click here](../README.md) to view the README.

## Design and implementation

This code example allows collecting data from an IMU (BMI270), magnetometer (BMM350), PDM/PCM microphone, analog microphone, barometric pressure sensor (DPS368), humidity and temperature sensor (SHT40), and 60 GHz radar (BGT60TR13C) using the [DEEPCRAFT&trade; streaming protocol v2](https://developer.imagimob.com/getting-started/tensor-streaming-protocol/registering-sensors-using-protocolv2). The application supports transmitting sensor data to DEEPCRAFT&trade; Studio either over USB or over Wi-Fi using the Tensor Streaming Protocol (TSP).

This application is designed with a minimalistic approach to help you get started with code examples on PSOC&trade; Edge MCU devices. All PSOC&trade; Edge E84 MCU applications have a dual-CPU three-project structure to develop code for the CM33 and CM55 cores. The CM33 core has two separate projects for the secure processing environment (SPE) and non-secure processing environment (NSPE). A project folder consists of various subfolders, each denoting a specific aspect of the project. The three project folders are as follows:

**Table 1. Application projects**

Project | Description
--------|------------------------
*proj_cm33_s* | Project for CM33 secure processing environment (SPE)
*proj_cm33_ns* | Project for CM33 non-secure processing environment (NSPE)
*proj_cm55* | CM55 project ? sensor drivers, streaming, shell, Wi-Fi, and filesystem

<br>

In this code example, at device reset, the secure boot process starts from the ROM boot with the secure enclave (SE) as the root of trust (RoT). From the secure enclave, the boot flow is passed on to the system CPU subsystem where the secure CM33 application starts. After all necessary secure configurations, the flow is passed on to the non-secure CM33 application. Resource initialization for this example is performed by this CM33 non-secure project. It configures the system clocks, pins, clock to peripheral connections, and other platform resources. It then enables the CM55 core using the `Cy_SysEnableCM55()` function and the CM33 core is subsequently put to DeepSleep mode.

In the CM33 non-secure application, the clocks and system resources are initialized by the BSP initialization function. The retarget-io middleware is configured to use the debug UART. The debug UART prints a message (as shown in **Figure 1** of [Operation](../README.md#operation) section) on the terminal emulator. The onboard KitProg3 acts as the USB-UART bridge to create the virtual COM port.

The CM55 application runs the main application logic via the `initd_task`. It initializes the filesystem, loads device drivers, sets up the USB data service, Wi-Fi, TSP, HTTP, and UPnP services, and finally runs the boot script from the filesystem.


### Data transport - USB

The board enumerates as a USB device over the dedicated USB connector (see [README](../README.md#usb-connection-to-deepcraft-studio-for-data-transfer) for connector details per kit). The `usbd` task (`source/usbd.c`) manages the USB interface. DEEPCRAFT&trade; Studio connects to this serial port to stream sensor data.


### Data transport - Wi-Fi

The board can connect to a Wi-Fi network and expose a Tensor Streaming Protocol (TSP) server on port 14325, allowing DEEPCRAFT&trade; Studio to connect wirelessly.

- Wi-Fi credentials are stored in the LittleFS filesystem at `/system/wifi` and loaded on every boot via the boot script
- Configure Wi-Fi interactively using the debug shell command `wifi setup`
- The `_wifi_connected` callback (in `source/initd.c`) runs the `/system/net-up` script, which starts the TSP (`tspd start -p 14325`), HTTP (`httpd start -p 80`), and UPnP (`upnpd start`) services
- The `_wifi_disconnected` callback runs `/system/net-down` to stop those services and turn off the Wi-Fi LED

<br>

**Table 1. Service-to-port mapping**

Service | Default port | Source file
--------|-------------|------------
TSP (Tensor Streaming Protocol) | 14325 | `source/net/tsp.c`
HTTP server | 80 | `source/net/http.c`
UPnP | - | `source/net/upnp.c`


### Debug shell

The board exposes an interactive shell over the KitProg3 UART (`source/shell/`). It is accessible via any serial terminal at 115200 baud 8N1. The shell supports a wide set of commands for filesystem access, device control, Wi-Fi configuration, and service management.

Key commands include:

**Table 2. System commands and descriptions**

Command | Description
--------|------------
`wifi setup` | Configure and save Wi-Fi credentials
`sh -e -f /system/boot` | Run the boot script
`httpd start/stop` | Start or stop the HTTP server
`tspd start/stop` | Start or stop the TSP server
`upnpd start/stop` | Start or stop the UPnP service
`mount / umount / format` | Manage the LittleFS filesystem
`dsave / dlist / dapply` | Save, list, and restore device configurations
`version` | Print firmware version


### Filesystem

The board uses LittleFS on QSPI/OSPI NOR flash. The filesystem is mounted at `/` on boot.

Default filesystem contents after factory reset:

**Table 3. System file paths and their descriptions**

Path | Description
-----|------------
`/system/boot` | Boot script, executed on every startup
`/system/wifi` | Wi-Fi credentials script
`/system/net-up` | Script executed when network IP is acquired
`/system/net-down` | Script executed when network goes down
`/system/login` | Login script for the debug shell
`/system/password` | MD5 hashed shell password
`/system/devices/` | Saved device configurations (`dsave`)
`/data/` | Capture data storage

A factory reset is triggered automatically if the filesystem cannot be mounted, or by holding the user button for 3 seconds during boot.


### IMU capture

- Code example is designed to collect data from a motion sensor (BMI270)

- Data consists of the 3-axis accelerometer and 3-axis gyroscope data obtained from the IMU. The data is read from the IMU over an I2C interface based on the configured data rate and then the data is transmitted over USB or Wi-Fi

> **Note:**
Use `SENSOR_REMAPPING` macro in the `common.mk` file to align sensor orientation with PSOC&trade; 6 AI Evaluation Kit (CY8CKIT-062S2-AI)

**Table 4. IMU Supported configurations**

 Supported configurations  |  Ranges/options
:-------- | :-------------
Frequency | 50 Hz, 100 Hz, 200 Hz, 400 Hz
Accelerometer | 2 G, 4 G, 8 G, 16 G
Gyroscope | 125 dps, 250 dps, 500 dps, 1000 dps, 2000 dps
Mode | Combined, split, only accelerometer, only gyroscope

### Microphone capture

- Code example can be configured to collect microphone data from a PDM microphone using the PDM to PCM IP

- Data is sampled at the rate configured in the DEEPCRAFT&trade; Studio and then the data is transmitted over USB or Wi-Fi

**Table 5. PDM microphone supported configurations**

 Supported configurations  |  Ranges/options
:-------- | :-------------
Gain |  83 dB ,  77 dB ,  71 dB ,  65 dB , 59 dB ,  53 dB ,  47 dB , 41 dB ,  35 dB , 29 dB , 23 dB , 17 dB , 11 dB , 5 dB , -1 dB , -7 dB , -13 dB , -19 dB , -25 dB , -31 dB , -37 dB , -43 dB , -49 dB , -55 dB , -61 dB , -67 dB , -73 dB , -79 dB , -85 dB , -91 dB , -97 dB , -103 dB 
Frequency | 8 kHz, 16 kHz, 22.05 kHz, 44.1 kHz, 48 kHz
Stereo (Mode) | Yes, No

### Analog microphone capture

- Code example can be configured to collect microphone data from a analog microphone using an ADC and other analog peripherals

- Data is sampled at the rate configured in the DEEPCRAFT&trade; Studio and then the data is transmitted over USB or Wi-Fi

**Table 6. Analog microphone supported configurations**

 Supported configurations  | Options
:-------- | :-------------
Frequency | 16 kHz
Stereo (Mode) | Yes, No

### Magnetometer capture

- Code example is designed to collect data from a magnetometer sensor (BMM350)

- Data consists of the 3-axis magnetic field data obtained from the sensor. The data is read from the magnetometer over an I3C interface based on the configured data rate and then the data is transmitted over USB or Wi-Fi

> **Note:**
Use `SENSOR_REMAPPING` macro in the `common.mk` file to align sensor orientation with PSOC&trade; 6 AI Evaluation Kit (CY8CKIT-062S2-AI).

**Table 7. Magnetometer supported configurations**

 Supported configurations  |  Ranges
:-------- | :-------------
Frequency | 50 Hz, 100 Hz, 200 Hz, 400 Hz

### Pressure capture

- Code example is designed to collect data from a XENSIV&trade; digital barometric air pressure (DPS368) sensor

- Pressure and temperature data is read from the sensor over an I2C interface based on the configured data rate and then the data is transmitted over USB or Wi-Fi

**Table 8. Pressure sensor supported configurations**

 Supported configurations  |  Ranges
:-------- | :-------------
Frequency | 8 Hz, 16 Hz, 32 Hz, 64 Hz, 128 Hz
Mode | Combined, Split, Only Pressure, Only Temperature

### Humidity capture

- Code example is designed to collect data from a digital temperature and humidity sensor (SHT40)

- Temperature and humidity data are read from the sensor over an I2C interface based on the configured data rate and then the data is transmitted over USB or Wi-Fi

**Table 9. Humidity and temperature sensor supported configurations**

 Supported configurations  | Options
:-------- | :-------------
Precision | Low, Medium, High
Mode | Combined, split, only humidity, only temperature

### Radar capture

- Code example is designed to collect data from a XENSIV&trade; 60 GHz radar (BGT60TR13C) sensor

- Radar data is read from the sensor over an SPI interface based on the use case and then the data is transmitted over USB or Wi-Fi

**Table 10. Radar supported configurations**

 Supported configurations  | Options
:-------- | :-------------
Use case | Default, Gesture, Macro Presence, Micro Presence, Custom

### Files and folders

```
|-- proj_cm55/devices/                  # Sensor implementation files
   |- dev_bmi270.c/h                    # IMU (BMI270) driver
   |- dev_bmm350.c/h                    # Magnetometer (BMM350) driver
   |- dev_pdm_pcm.c/h                   # PDM digital microphone driver
   |- dev_amic.c/h                      # Analog microphone driver
   |- dev_dps368.c/h                    # Barometric pressure sensor (DPS368) driver
   |- dev_sht4x.c/h                     # Humidity and temperature sensor (SHT40) driver
   |- dev_bgt60trxx.c/h                 # 60 GHz radar (BGT60TR13C) driver
   |- dev_bgt60trxx_settings.h          # Radar sensor settings
   |- dev_board.c/h                     # Board-level device registration
   |- pdm_pcm.c/h                       # PDM/PCM PDL API implementation
|-- proj_cm55/protocol/                 # Streaming protocol (DEEPCRAFT v2)
|-- proj_cm55/source/                   # Main application source
   |- initd.c                           # Init task: filesystem, services, boot script
   |- usbd.c/h                          # USB data service task
   |- services.c/h                      # System service registry
   |- board.c/h                         # Board hardware abstraction
   |- system.c/h                        # Device driver loading
   |- net/wifi.c/h                      # Wi-Fi connection management
   |- net/tsp.c/h                       # Tensor Streaming Protocol server
   |- net/http.c/h                      # HTTP server
   |- net/upnp.c/h                      # UPnP service
   |- shell/                            # Interactive debug shell and commands
   |- shell/fs/lfs_drv.c                # LittleFS SPI flash block device driver
```

<br>