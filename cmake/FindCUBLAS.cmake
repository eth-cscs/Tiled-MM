# 
# To find CUDA:
#    a) set the env. / cmake variable CUDA_PATH
#    b) enable cuda as a language
#
# Imported Targets: 
#   CUDA::cublas
#
include(FindPackageHandleStandardArgs)

function(__cuda_find_library _name)
    find_library(${_name}
        NAMES ${ARGN}
        HINTS ${CUDA_PATH}
        PATH_SUFFIXES lib
                      lib64
        )
    mark_as_advanced(${_name})
endfunction()

# Only try to set if not already set.
#
if (NOT CUDA_PATH)
    if (DEFINED ENV{CUDA_PATH})
        set(CUDA_PATH $ENV{CUDA_PATH})
    elseif (DEFINED ENV{CUDATOOLKIT_HOME})
        set(CUDA_PATH $ENV{CUDATOOLKIT_HOME})
    elseif (DEFINED ENV{CUDA_TOOLKIT_INCLUDE})
        set(CUDA_PATH $ENV{CUDA_TOOLKIT_INCLUDE})
    elseif (CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
        # If CUDA was enabled as a language, we already have the variable.
        #
        get_filename_component(CUDA_PATH "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" DIRECTORY)
    else()
        message("CUDA not found. Please specify CUDA_PATH variable.")
    endif()
endif()

find_path(CUDA_TOOLKIT_INCLUDE 
          device_functions.h # Header included in toolkit
    PATHS ${CUDA_PATH}
          ${CUDA_TOOLKIT_TARGET_DIR}
          ${CUDATOOLKIT_HOME}
          ${CUDA_TOOLKIT_INCLUDE}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH)

__cuda_find_library(CUDA_CUDART_LIB cudart)
__cuda_find_library(CUDA_CUBLAS_LIB cublas)

find_package_handle_standard_args(CUBLAS 
  DEFAULT_MSG CUDA_TOOLKIT_INCLUDE
              CUDA_CUBLAS_LIB
              CUDA_CUDART_LIB
              )

if (CUBLAS_FOUND AND NOT TARGET CUDA::cublas)
    add_library(CUDA::cublas IMPORTED INTERFACE)
    set_target_properties(CUDA::cublas PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${CUDA_TOOLKIT_INCLUDE}"
      INTERFACE_LINK_LIBRARIES "${CUDA_CUDART_LIB};${CUDA_CUBLAS_LIB}")
endif()

