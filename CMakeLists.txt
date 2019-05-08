cmake_minimum_required(VERSION 3.5)
PROJECT(Tiled-MM)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release." FORCE)
endif()

#################################
#  Generic Compilation options  #
#################################
if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -lstdc++")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -ggdb -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
else ()
    message(FATAL_ERROR "This library supports only GNU compilers.")
endif ()

################
# CUDA support #
################
find_package(CUDA REQUIRED)
add_definitions(-DWITH_CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60 -gencode=arch=compute_60,code=sm_60")
set(cuda_deps ${CUDA_LIBRARIES} cublas)

set(DEPENDENCIES ${cuda_deps})

#############
#  Testing  #
#############
include(CTest)
enable_testing()

#####################################
#  Adding various code directories  #
#####################################
add_subdirectory(src)
add_subdirectory(libs)
add_subdirectory(examples)
add_subdirectory(tests)
