#  Copyright (c) 2019 ETH Zurich
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.


#.rst:
# FindHIP
# -----------
#
# This module searches for the fftw3 library.
#
# The following variables are set
#
# ::
#
#   HIP_FOUND           - True if hip is found
#   HIP_LIBRARIES       - The required libraries
#   HIP_INCLUDE_DIRS    - The required include directory
#   HIP_DEFINITIONS     - The required definitions
#
# The following import target is created
#
# ::
#
#   HIP::hip



# set paths to look for library
set(_HIP_PATHS ${HIP_ROOT} $ENV{HIP_ROOT})
set(_HIP_INCLUDE_PATHS)

set(_HIP_DEFAULT_PATH_SWITCH)

if(_HIP_PATHS)
    # disable default paths if ROOT is set
    set(_HIP_DEFAULT_PATH_SWITCH NO_DEFAULT_PATH)
else()
    set(_HIP_PATHS /opt/rocm)
endif()


find_path(HIP_INCLUDE_DIRS
    NAMES "hip/hip_runtime_api.h"
    HINTS ${_HIP_PATHS} ${_HIP_INCLUDE_PATHS}
    PATH_SUFFIXES "include" "hip/include"
    ${_HIP_DEFAULT_PATH_SWITCH}
)
find_library(
    HIP_LIBRARIES
    NAMES "hip_hcc"
    HINTS ${_HIP_PATHS}
    PATH_SUFFIXES "hip" "hip/lib" "lib"
    ${_HIP_DEFAULT_PATH_SWITCH}
)

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HIP REQUIRED_VARS HIP_INCLUDE_DIRS HIP_LIBRARIES )

# add target to link against
if(HIP_FOUND)
    if(NOT TARGET HIP::hip)
        add_library(HIP::hip INTERFACE IMPORTED)
    endif()
    set_property(TARGET HIP::hip PROPERTY INTERFACE_LINK_LIBRARIES ${HIP_LIBRARIES})
    set_property(TARGET HIP::hip PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${HIP_INCLUDE_DIRS})
    set_property(TARGET HIP::hip PROPERTY INTERFACE_COMPILE_DEFINITIONS __HIP_PLATFORM_HCC__)
endif()

# prevent clutter in cache
MARK_AS_ADVANCED(HIP_FOUND HIP_LIBRARIES HIP_INCLUDE_DIRS)
