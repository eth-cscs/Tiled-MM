cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

# generate compile_commands.json with compile commands for each target
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
project(Tiled-MM VERSION 2.3.1 LANGUAGES CXX)

option(TILEDMM_WITH_TESTS "Enable tests." ${MASTER_PROJECT})
option(TILEDMM_WITH_EXAMPLES "Enable examples." ${MASTER_PROJECT})
set(TILEDMM_GPU_BACKEND "CUDA" CACHE STRING "GPU backend")
set_property(CACHE TILEDMM_GPU_BACKEND PROPERTY STRINGS "CUDA" "ROCM")

option(BUILD_SHARED_LIBS "Build shared libraries." OFF)

if(MASTER_PROJECT AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release." FORCE)
endif()

message(STATUS "Selected TILEDMM_GPU_BACKEND: ${TILEDMM_GPU_BACKEND}")

set(TILED_MM_CUDA OFF)
set(TILED_MM_ROCM OFF)
# Dependencies
#
if(TILEDMM_GPU_BACKEND STREQUAL "CUDA")
    find_package(CUDAToolkit REQUIRED)
    set(TILED_MM_CUDA ON)
elseif(TILEDMM_GPU_BACKEND STREQUAL "ROCM")
    find_package(rocblas REQUIRED)
    set(TILED_MM_ROCM ON)
else()
    message(FATAL_ERROR "TILED_MM: GPU Backend must be CUDA or ROCM!")
endif()

# Tiled-MM
#
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

add_subdirectory(src/Tiled-MM)

install(EXPORT Tiled-MM_targets
  FILE Tiled-MMTargets.cmake
  NAMESPACE Tiled-MM::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Tiled-MM")

install(DIRECTORY "${Tiled-MM_SOURCE_DIR}/src/Tiled-MM"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING PATTERN "*.hpp")

write_basic_package_version_file("${Tiled-MM_BINARY_DIR}/Tiled-MMConfigVersion.cmake"
  VERSION ${Tiled-MM_VERSION}
  COMPATIBILITY SameMajorVersion)

configure_file("${Tiled-MM_SOURCE_DIR}/cmake/Tiled-MMConfig.cmake.in"
  "${Tiled-MM_BINARY_DIR}/Tiled-MMConfig.cmake"
  @ONLY)

install(FILES "${Tiled-MM_BINARY_DIR}/Tiled-MMConfig.cmake"
  "${Tiled-MM_BINARY_DIR}/Tiled-MMConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Tiled-MM")

if(TILEDMM_WITH_TESTS OR TILEDMM_WITH_EXAMPLES)
  include(FetchContent)
  FetchContent_Declare(cxxopts
                       GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
                       GIT_TAG        4bf61f08697b110d9e3991864650a405b3dd515d # v3.2.1
  )
  FetchContent_MakeAvailable(cxxopts)
endif()

if(TILEDMM_WITH_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(TILEDMM_WITH_EXAMPLES)
  add_subdirectory(examples)
endif()


