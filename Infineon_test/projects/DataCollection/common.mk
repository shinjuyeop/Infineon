################################################################################
# \file common.mk
# \version 1.0
#
# \brief
# Settings shared across all projects.
#
################################################################################
# \copyright
# (c) 2025-2026, Infineon Technologies AG, or an affiliate of Infineon
# Technologies AG.  SPDX-License-Identifier: Apache-2.0
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

MTB_TYPE=PROJECT

# Target board/hardware (BSP).
# To change the target, it is recommended to use the Library manager
# ('make library-manager' from command line), which will also update
# Eclipse IDE launch configurations.
TARGET=APP_KIT_PSE84_AI

# Name of toolchain to use. Options include:
#
# GCC_ARM 	-- GCC is available as part of ModusToolbox Setup program
#
# See also: CY_COMPILER_PATH below
TOOLCHAIN=GCC_ARM

# Toolchains supported by this code example. See README.md file.
# This is used by automated build systems to identify the supported toolchains.
MTB_SUPPORTED_TOOLCHAINS?=GCC_ARM

# Default build configuration. Options include:
#
# Debug -- build with minimal optimizations, focus on debugging.
# Release -- build with full optimizations
# Custom -- build with custom configuration, set the optimization flag in CFLAGS
#
# If CONFIG is manually edited, ensure to update or regenerate
# launch configurations for your IDE.
CONFIG=Debug

# Option to remap the sensor orientation to align with PSOC 6 AI Evaluation Kit (CY8CKIT-062S2-AI).
#
# ENABLED   - Sensor data is remapped
#
# DISABLED  - Actual sensor data
SENSOR_REMAPPING=ENABLED

ifeq ($(SENSOR_REMAPPING),ENABLED)
DEFINES+=USE_SENSOR_REMAPPING
endif

# Uncomment the following line to enable Wi-Fi support.
DEFINES += IM_ENABLE_NET

# Uncomment the following line to enable a shell and file system.
# The shell will be accessible via the USB UART.
# This feature is recommended if you use WiFi, as it allows you to utilize the
# "wifi setup" command to store WiFi credentials in flash memory and reconnect
# automatically on each power-up.
DEFINES += IM_ENABLE_SHELL

# Uncomment the following line to enable HTTP server support.
# This feature is required by IM_ENABLE_UPNP.
# Note: Requires IM_ENABLE_NET.
DEFINES += IM_ENABLE_HTTP

# Uncomment the following line to enable UPnP support.
# This allows the device to appear in the Windows Explorer Network view.
# Note: Requires IM_ENABLE_HTTP and IM_ENABLE_NET.
DEFINES += IM_ENABLE_UPNP

ifeq ($(filter APP_KIT_PSE84_AI KIT_PSE84_AI, $(TARGET)), $(TARGET))
DEFINES+=USE_KIT_PSE84_AI
# Sensors on kit
DEFINES+=IM_ENABLE_BOARD_CONFIG
DEFINES+=IM_ENABLE_PDM_PCM
DEFINES+=IM_ENABLE_AMIC
DEFINES+=IM_ENABLE_BMI270
DEFINES+=IM_ENABLE_DPS368
DEFINES+=IM_ENABLE_BMM350
DEFINES+=IM_ENABLE_BGT60TRXX
DEFINES+=IM_ENABLE_SHT4X
endif

ifeq ($(filter APP_KIT_PSE84_EVAL_EPC2 KIT_PSE84_EVAL_EPC2, $(TARGET)), $(TARGET))
DEFINES+=USE_KIT_PSE84_EVAL_EPC2
# Sensors on kit
DEFINES+=IM_ENABLE_BOARD_CONFIG
DEFINES+=IM_ENABLE_PDM_PCM
DEFINES+=IM_ENABLE_AMIC
DEFINES+=IM_ENABLE_BMI270
DEFINES+=IM_ENABLE_BMM350
endif

ifeq ($(filter APP_KIT_PSE84_EVAL_EPC4 KIT_PSE84_EVAL_EPC4, $(TARGET)), $(TARGET))
DEFINES+=USE_KIT_PSE84_EVAL_EPC4
# Sensors on kit
DEFINES+=IM_ENABLE_BOARD_CONFIG
DEFINES+=IM_ENABLE_PDM_PCM
DEFINES+=IM_ENABLE_AMIC
DEFINES+=IM_ENABLE_BMI270
DEFINES+=IM_ENABLE_BMM350
endif

ifeq ($(filter APP_KIT_PSE84_HMI KIT_PSE84_HMI, $(TARGET)), $(TARGET))
DEFINES+=USE_KIT_PSE84_HMI
# Sensors on kit
DEFINES+=IM_ENABLE_BOARD_CONFIG
DEFINES+=IM_ENABLE_PDM_PCM
DEFINES+=IM_ENABLE_BMI270
DEFINES+=IM_ENABLE_BGT60TRXX
DEFINES+=IM_ENABLE_SHT4X
DEFINES+=IM_ENABLE_AMIC
# GFXSS is disabled
override DISABLE_COMPONENTS+=GFXSS
endif

# Config file for postbuild sign and merge operations.
# NOTE: Check the JSON file for the command parameters
COMBINE_SIGN_JSON?=configs/boot_with_extended_boot.json

include ../common_app.mk
