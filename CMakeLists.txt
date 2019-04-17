cmake_minimum_required(VERSION 3.5)
PROJECT(tiled-mm C CXX)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 14)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release." FORCE)
endif()

#################################
#  Generic Compilation options  #
#################################
# Compiler must support c++14
if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
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
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(src)
add_subdirectory(include)
add_subdirectory(examples)
add_subdirectory(tests)
