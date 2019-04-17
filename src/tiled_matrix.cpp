#include "tiled_matrix.hpp"

using namespace gpu;

tiled_matrix::tiled_matrix(double* host_ptr, int rows, int cols, tile_dim d):
        ptr(host_ptr), n_rows(rows), n_cols(cols), tile(d) {

    tile.set_rows(std::min(tile.rows(), rows));
    tile.set_cols(std::min(tile.cols(), cols));

    n_tiles_row = (int) std::ceil(1.0 * rows / tile.rows());
    n_tiles_col = (int) std::ceil(1.0 * cols / tile.cols());

    int short_tile_size_row = rows % tile.rows();
    int short_tile_size_col = cols % tile.cols();
    short_tile = tile_dim(short_tile_size_row, short_tile_size_col);
}

tile_dim tiled_matrix::tile_dimensions() {
    return tile;
}

tile_dim tiled_matrix::tile_dimensions(tile_coord t_coord) {
    int row_size = actual_size(n_tiles_row, t_coord.row_index(),
           tile.rows(), short_tile.rows());

    int col_size = actual_size(n_tiles_col, t_coord.col_index(),
           tile.cols(), short_tile.cols());

    return tile_dim(row_size, col_size);
}

int tiled_matrix::rows() {
    return n_rows;
}

int tiled_matrix::cols() {
    return n_cols;
}

double* tiled_matrix::data() {
    return ptr;
}

int tiled_matrix::tile_offset(tile_coord t_coord) {
    int tile_offset_global = (t_coord.col_index()*tile.cols()) * n_rows;
    int el_offset_global = t_coord.row_index() * tile.rows();
    int offset = tile_offset_global + el_offset_global;
    return offset;
}

double* tiled_matrix::tile_data(tile_coord tile) {
    return data() + tile_offset(tile);
}

int tiled_matrix::actual_size(int n_tiles, int tile_id, int tile_length, int tile_remainder) {
    bool last_tile = tile_id == n_tiles - 1;
    bool not_divisible = tile_remainder > 0;

    return last_tile && not_divisible ? tile_remainder : tile_length;
}

int tiled_matrix::num_tiles_row() {
    return n_tiles_row;
}

int tiled_matrix::num_tiles_col() {
    return n_tiles_col;
}
