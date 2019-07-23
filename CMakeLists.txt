cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

set(MASTER_PROJECT OFF)
if(NOT DEFINED PROJECT_NAME)
    set(MASTER_PROJECT ON)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
project(Tiled-MM VERSION 0.1 LANGUAGES CXX)

option(TILEDMM_WITH_TESTS "Enable tests." ${MASTER_PROJECT})
option(TILEDMM_WITH_EXAMPLES "Enable examples." ${MASTER_PROJECT})
option(TILEDMM_WITH_INSTALL "Enable installation." ${MASTER_PROJECT})

if(MASTER_PROJECT AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release." FORCE)
endif()

find_package(CUBLAS REQUIRED)
add_library(Tiled-MM STATIC "${PROJECT_SOURCE_DIR}/src/gpu_context.cpp"
                            "${PROJECT_SOURCE_DIR}/src/mm_handle.cpp"
                            "${PROJECT_SOURCE_DIR}/src/tile_dim.cpp"
                            "${PROJECT_SOURCE_DIR}/src/tiled_matrix.cpp"
                            "${PROJECT_SOURCE_DIR}/src/tiled_mm.cpp"
                            "${PROJECT_SOURCE_DIR}/src/tile_coord.cpp")
target_link_libraries(Tiled-MM PUBLIC CUDA::cublas)
target_include_directories(Tiled-MM PUBLIC 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    )
target_compile_features(Tiled-MM PUBLIC cxx_std_14)

if(TILEDMM_WITH_INSTALL)
    include(CMakePackageConfigHelpers)
    include(GNUInstallDirs)
    
    install(TARGETS Tiled-MM
            EXPORT Tiled-MM_targets
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
    
    install(EXPORT Tiled-MM_targets
            FILE Tiled-MMTargets.cmake
            NAMESPACE Tiled-MM::
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Tiled-MM")
    
    install(DIRECTORY "${Tiled-MM_SOURCE_DIR}/src" 
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
                  "${Tiled-MM_SOURCE_DIR}/cmake/FindCUBLAS.cmake"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Tiled-MM")
endif()

if(TILEDMM_WITH_TESTS OR TILEDMM_WITH_EXAMPLES)
    add_subdirectory("${PROJECT_SOURCE_DIR}/libs/options")
endif()

if(TILEDMM_WITH_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TILEDMM_WITH_EXAMPLES)
    add_subdirectory(examples)
endif()
