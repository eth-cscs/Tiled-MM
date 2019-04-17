#include "mm_handle.hpp"
namespace gpu {
mm_handle::mm_handle(): ctx(n_streams) {
    cudaSetDevice(0);

    a_buff = device_buffer<double>(n_streams, {tile_size_m, tile_size_k});
    b_buff = device_buffer<double>(n_streams, {tile_size_k, tile_size_n});
    c_buff = device_buffer<double>(n_streams, {tile_size_m, tile_size_n});
}

mm_handle::mm_handle(int streams, int tile_m, int tile_n, int tile_k): n_streams(streams), 
        tile_size_m(tile_m), tile_size_n(tile_n), tile_size_k(tile_k), ctx(streams) {
    cudaSetDevice(0);

    a_buff = device_buffer<double>(n_streams, {tile_size_m, tile_size_k});
    b_buff = device_buffer<double>(n_streams, {tile_size_k, tile_size_n});
    c_buff = device_buffer<double>(n_streams, {tile_size_m, tile_size_n});
}

mm_handle::~mm_handle() {
    // std::cout << "freeing mm_handle" << std::endl;
}

void mm_handle::set_num_streams(int streams) {
    n_streams = streams;
    ctx.set_num_streams(streams);

    a_buff.set_num_streams(streams);
    b_buff.set_num_streams(streams);
    c_buff.set_num_streams(streams);
}

int mm_handle::get_num_streams() {
    return n_streams;
}

gpu_context& mm_handle::get_gpu_context() {
    return ctx;
}

void mm_handle::set_tile_sizes(int tile_m, int tile_n, int tile_k) {
    tile_size_m = tile_m;
    tile_size_n = tile_n;
    tile_size_k = tile_k;

    a_buff.set_tile_dimensions({tile_m, tile_k});
    b_buff.set_tile_dimensions({tile_k, tile_n});
    c_buff.set_tile_dimensions({tile_m, tile_n});
}

void mm_handle::set_tile_sizes(int tile_size) {
    set_tile_sizes(tile_size, tile_size, tile_size);
}

std::tuple<int, int, int> mm_handle::get_tile_sizes() {
    return {tile_size_m, tile_size_n, tile_size_k};
}

void mm_handle::set_streams_and_tiles(int streams, int tile_m, int tile_n, int tile_k) {
    n_streams = streams;
    tile_size_m = tile_m;
    tile_size_n = tile_n;
    tile_size_k = tile_k;

    ctx.set_num_streams(n_streams);
    a_buff.set_streams_and_tiles(streams, {tile_m, tile_k});
    b_buff.set_streams_and_tiles(streams, {tile_k, tile_n});
    c_buff.set_streams_and_tiles(streams, {tile_m, tile_n});
}

device_buffer<double>& mm_handle::get_device_buffer_a() {
    return a_buff;
}

device_buffer<double>& mm_handle::get_device_buffer_b() {
    return b_buff;
}

device_buffer<double>& mm_handle::get_device_buffer_c() {
    return c_buff;
}

context make_context() {
    return std::make_unique<mm_handle>();
}

context make_context(int streams, int tile_m, int tile_n, int tile_k) {
    return std::make_unique<mm_handle>(streams, tile_m, tile_n, tile_k);
}
}
