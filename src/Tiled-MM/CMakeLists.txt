add_library(Tiled-MM gpu_context.cpp
                     mm_handle.cpp
                     tile_dim.cpp
                     tiled_matrix.cpp
                     tiled_mm.cpp
                     tile_coord.cpp)

target_link_libraries(Tiled-MM PUBLIC
  $<$<BOOL:${TILED_MM_ROCM}>:roc::rocblas>
  $<$<BOOL:${TILED_MM_CUDA}>:CUDA::cublas CUDA::cudart>
)

target_include_directories(Tiled-MM PUBLIC
    $<BUILD_INTERFACE:${Tiled-MM_SOURCE_DIR}/src>
    INTERFACE
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

target_compile_features(Tiled-MM PUBLIC cxx_std_14)

target_compile_definitions(Tiled-MM PUBLIC
                           $<$<BOOL:${TILED_MM_CUDA}>:TILED_MM_CUDA>
                           $<$<BOOL:${TILED_MM_ROCM}>:TILED_MM_ROCM TILED_MM_ROCBLAS_HAS_SGEMM TILED_MM_ROCBLAS_HAS_DGEMM TILED_MM_ROCBLAS_HAS_ZGEMM>)


install(TARGETS Tiled-MM
  EXPORT Tiled-MM_targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/Tiled-MM")


# FetchContent mechanism. It should really use the config.cmake stuff instead of this
if (NOT TARGET Tiled-MM::Tiled-MM AND NOT TILED_MM_MASTER_PROJECT)
  add_library(Tiled-MM::Tiled-MM ALIAS Tiled-MM)
endif()

