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
add_library(tiled-mm ${sources})
target_link_libraries(tiled-mm ${DEPENDENCIES})
