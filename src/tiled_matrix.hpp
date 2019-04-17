#pragma once
#include "tile_dim.hpp"
#include "tile_coord.hpp"
#include <algorithm>
#include <cmath>

namespace gpu {

class tiled_matrix {
public:
    tiled_matrix(double* host_ptr, int rows, int cols, tile_dim d);

    tile_dim tile_dimensions();

    tile_dim tile_dimensions(tile_coord t_coord);

    int rows();
    int cols();

    double* data();

    int tile_offset(tile_coord t_coord);

    double* tile_data(tile_coord tile);

    int num_tiles_row();
    int num_tiles_col();

private:
    double* ptr;
    int n_rows;
    int n_cols;
    tile_dim tile;

    int n_tiles_row;
    int n_tiles_col;

    tile_dim short_tile;

    int actual_size(int n_tiles, int tile_id, int tile_length, int tile_remainder);
};

}
