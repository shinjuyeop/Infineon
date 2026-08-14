################################################################################
# \file Makefile
# \version 1.0
#
# \brief
# Makefile for configuring the Cortex-M55 core project
#
################################################################################
# \copyright
# (c) 2023-2026, Infineon Technologies AG, or an affiliate of Infineon
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

include ../common.mk

################################################################################
# Basic Configuration
################################################################################

# Name of application (used to derive name of final linked file).
#
# If APPNAME is edited, ensure to update or regenerate launch
# configurations for your IDE.
APPNAME=proj_cm55

# If set to "true" or "1", display full command-lines when building.
VERBOSE=

# Neural Network Configuration. Options include
#
# float    -- floating point for the input data and weights
# int8x8   -- 8-bit fixed-point for the input data and weights
# int16x8  -- 16-bit fixed-point for the input data and 8-bit for weights
NN_TYPE=int8x8

# Model Name to be loaded to the firmware
# TERRAIN_CPU contains the original strict-INT8 TFLite model and exercises the
# Cortex-M55 TFLM kernels. Override with NN_MODEL_NAME=TERRAIN_U55 to exercise
# the Vela-compiled Ethos-U55 path with the same fixed regression tensor.
# The separate 1 kHz/50-sample candidate uses TERRAIN_FAST1000_CPU or
# TERRAIN_FAST1000_U55. These names never select or overwrite the 100 Hz arrays.
NN_MODEL_NAME=TERRAIN_CPU

# fixed: run the generated test-partition tensor once at boot
# hil:   receive TRN1 full windows or TRN2 INT8 samples over KitProg3 UART
# frv2_sink_hil: receive asynchronous/batched raw Fusion10 replay for Sink v2
TERRAIN_MODE?=fixed
ifneq (,$(findstring FAST_REFLEX_,$(NN_MODEL_NAME)))
DEFINES+=FAST_REFLEX_GOLDEN
endif
ifeq (FAST_REFLEX_SLIP_V2_U55,$(NN_MODEL_NAME))
DEFINES+=FAST_REFLEX_THRESHOLD_RAW=121
endif
ifeq (FAST_REFLEX_SINK_V2_U55,$(NN_MODEL_NAME))
DEFINES+=FAST_REFLEX_THRESHOLD_RAW=124
endif
ifeq (hil, $(TERRAIN_MODE))
DEFINES+=TERRAIN_HIL
endif
ifeq (frv2_sink_hil, $(TERRAIN_MODE))
DEFINES+=FAST_REFLEX_SINK_HIL
endif
ifeq (frv2_slip_hil, $(TERRAIN_MODE))
DEFINES+=FAST_REFLEX_SLIP_HIL
endif

# Folder name containing the model and regression data
NN_MODEL_FOLDER=mtb_ml_gen

# Choose the inference engine
# tflm      -- TensorFlow Lite for Microcontrollers inference engine with interpreter
# Note tflm_less option is not supported on CM55+U55
NN_INFERENCE_ENGINE=tflm

# Is a RNN model? yes or no
NN_RNN_MODEL=no

# Choose which cache management strategy for the Ethos-U55
# MTB_ML_ETHOSU_CACHE_MGMT_ALL_LAYERS   -- (default) clear/invalidate cache in all layers
# MTB_ML_ETHOSU_CACHE_MGMT_OUTER_LAYERS -- clear/invalidate cache in the input/output layers only
# MTB_ML_ETHOSU_CACHE_MGMT_CONDITIONAL  -- clear/invalidate cache based on internal state
DEFINES+=MTB_ML_ETHOSU_CACHE_MGMT_TYPE=MTB_ML_ETHOSU_CACHE_MGMT_ALL_LAYERS

################################################################################
# Advanced Configuration
################################################################################

# Enable optional code that is ordinarily disabled by default.
#
# Available components depend on the specific targeted hardware and firmware
# in use. In general, if you have
#
#    COMPONENTS=foo bar
#
# ... then code in directories named COMPONENT_foo and COMPONENT_bar will be
# added to the build
#
COMPONENTS+=

# Like COMPONENTS, but disable optional code that was enabled by default.
DISABLE_COMPONENTS=

CORE=CM55
CORE_NAME=CM55_0
NPU=U55

# By default the build system automatically looks in the Makefile's directory
# tree for source code and builds it. The SOURCES variable can be used to
# manually add source code to the build process from a location not searched
# by default, or otherwise not found by the build system.
SOURCES+=

# Select only the regression and model files that belong to the desired
# settings. 
MODEL_PREFIX=$(subst $\",,$(NN_MODEL_NAME))
CY_IGNORE+=$(NN_MODEL_FOLDER)

# Add additional defines to the build process (without a leading -D).
DEFINES+=CY_RETARGET_IO_CONVERT_LF_TO_CRLF 

# Add additional defines related to ARM Helium and DSP extensions
DEFINES+=ARM_MATH_HELIUM ARM_MATH_DSP ARM_MATH_AUTOVECTORIZE

# Add additional components related to MTB ML
COMPONENTS+=U55 ML_MW_STREAM

# Like SOURCES, but for include directories. Value should be paths to
# directories (without a leading -I).
INCLUDES+=$(NN_MODEL_FOLDER)/mtb_ml_regression_data $(NN_MODEL_FOLDER)/mtb_ml_models

# Add additional define/components to select the inference engine
ifeq (tflm, $(NN_INFERENCE_ENGINE))
COMPONENTS+=ML_TFLM
DEFINES+=TF_LITE_STATIC_MEMORY
endif

# Add additional define for RRN model
ifeq (yes, $(NN_RNN_MODEL))
DEFINES+=RNN_STREAMING
endif

# Add define to build code for CM55 and include additional files
ifeq (cm55, $(ML_PROFILER_CPU))
DEFINES+=ML_PROFILER_CM55 MODEL_NAME=$(NN_MODEL_NAME) CY_ML_MODEL_MEM=.cy_socmem_data
include ../ml_profiler.mk

# Add the model file based on the inference and data types
SOURCES+=$(wildcard $(NN_MODEL_FOLDER)/mtb_ml_models/$(MODEL_PREFIX)_$(NN_INFERENCE_ENGINE)_model_$(NN_TYPE).c*)

ifeq (local, $(ML_VALIDATION_SOURCE))
# Add the regression files
SOURCES+=$(wildcard $(NN_MODEL_FOLDER)/mtb_ml_regression_data/$(MODEL_PREFIX)_tflm_*_data_$(NN_TYPE).c)
endif
endif

# Depending which Neural Network Type, add a specific DEFINE and COMPONENT
ifeq (float, $(NN_TYPE))
COMPONENTS+=ML_FLOAT32
endif
ifeq (int8x8, $(NN_TYPE))
COMPONENTS+=ML_INT8x8
endif
ifeq (int16x8, $(NN_TYPE))
COMPONENTS+=ML_INT16x8
endif

# Select softfp or hardfp floating point. Default is softfp.
VFP_SELECT+=

# Additional / custom C compiler flags.
#
# NOTE: Includes and defines should use the INCLUDES and DEFINES variable
# above.
CFLAGS+=

# Additional / custom C++ compiler flags.
#
# NOTE: Includes and defines should use the INCLUDES and DEFINES variable
# above.
CXXFLAGS+=

# Additional / custom assembler flags.
#
# NOTE: Includes and defines should use the INCLUDES and DEFINES variable
# above.
ASFLAGS+=

# Additional / custom linker flags.
LDFLAGS+=

# Additional / custom libraries to link in to the application.
LDLIBS+=

# Path to the linker script to use (if empty, use the default linker script).
LINKER_SCRIPT=

# Custom pre-build commands to run.
PREBUILD=

# Custom post-build commands to run.
POSTBUILD=


################################################################################
# Paths
################################################################################

# Relative path to the project directory (default is the Makefile's directory).
#
# This controls where automatic source code discovery looks for code.
CY_APP_PATH=

# Relative path to the shared repo location.
#
# All .mtb files have the format, <URI>#<COMMIT>#<LOCATION>. If the <LOCATION> 
# field begins with $$ASSET_REPO$$, then the repo is deposited in the path 
# specified by the CY_GETLIBS_SHARED_PATH variable. The default location is one 
# directory level above the current app directory. This is used with 
# CY_GETLIBS_SHARED_NAME variable, which specifies the directory name.
CY_GETLIBS_SHARED_PATH=../..

# Directory name of the shared repo location.
#
CY_GETLIBS_SHARED_NAME=mtb_shared

include $(CY_TOOLS_DIR)/make/start.mk
