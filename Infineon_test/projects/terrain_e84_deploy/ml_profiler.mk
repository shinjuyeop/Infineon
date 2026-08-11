################################################################################
# \file ml_profiler.mk
# \version 1.0
#
# \brief
# Settings shared across CM33 and CM55 projects.
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

# Add shared SOURCES and INCLUDES
SOURCES+=$(wildcard ../shared_src/*.c)
INCLUDES+=../shared_src/

# Add where to source the regression data from
ifeq (stream, $(ML_VALIDATION_SOURCE))
	DEFINES+=USE_STREAM_DATA
endif
