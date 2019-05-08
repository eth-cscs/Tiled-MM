###############
#  Src files  #
###############
set(sources
    gpu_context.cpp
    mm_handle.cpp
    tile_dim.cpp
    tiled_matrix.cpp
    tiled_mm.cpp
    tile_coord.cpp)

###################
#     Library     #
###################
add_library(Tiled-MM ${sources})
target_link_libraries(Tiled-MM ${DEPENDENCIES})
target_include_directories(Tiled-MM PUBLIC ${PROJECT_SOURCE_DIR}/src)
