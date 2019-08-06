#include "mm_handle.hpp"

#include<complex>

namespace gpu {

template <typename Scalar>
mm_handle<Scalar>::mm_handle(): ctx(n_streams) {
    cudaSetDevice(0);

    a_buff = device_buffer<Scalar>(n_streams, {tile_size_m, tile_size_k});
    b_buff = device_buffer<Scalar>(n_streams, {tile_size_k, tile_size_n});
    c_buff = device_buffer<Scalar>(n_streams, {tile_size_m, tile_size_n});
}

template <typename Scalar>
mm_handle<Scalar>::mm_handle(int streams, int tile_m, int tile_n, int tile_k): n_streams(streams),
        tile_size_m(tile_m), tile_size_n(tile_n), tile_size_k(tile_k), ctx(streams) {
    cudaSetDevice(0);

    a_buff = device_buffer<Scalar>(n_streams, {tile_size_m, tile_size_k});
    b_buff = device_buffer<Scalar>(n_streams, {tile_size_k, tile_size_n});
    c_buff = device_buffer<Scalar>(n_streams, {tile_size_m, tile_size_n});
}

template <typename Scalar>
mm_handle<Scalar>::~mm_handle() {
    // std::cout << "freeing mm_handle" << std::endl;
}

template <typename Scalar>
void mm_handle<Scalar>::set_num_streams(int streams) {
    n_streams = streams;
    ctx.set_num_streams(streams);

    a_buff.set_num_streams(streams);
    b_buff.set_num_streams(streams);
    c_buff.set_num_streams(streams);
}

template <typename Scalar>
int mm_handle<Scalar>::get_num_streams() {
    return n_streams;
}

template <typename Scalar>
gpu_context& mm_handle<Scalar>::get_gpu_context() {
    return ctx;
}

template <typename Scalar>
void mm_handle<Scalar>::set_tile_sizes(int tile_m, int tile_n, int tile_k) {
    tile_size_m = tile_m;
    tile_size_n = tile_n;
    tile_size_k = tile_k;

    a_buff.set_tile_dimensions({tile_m, tile_k});
    b_buff.set_tile_dimensions({tile_k, tile_n});
    c_buff.set_tile_dimensions({tile_m, tile_n});
}

template <typename Scalar>
void mm_handle<Scalar>::set_tile_sizes(int tile_size) {
    set_tile_sizes(tile_size, tile_size, tile_size);
}

template <typename Scalar>
std::tuple<int, int, int> mm_handle<Scalar>::get_tile_sizes() {
    return {tile_size_m, tile_size_n, tile_size_k};
}

template <typename Scalar>
void mm_handle<Scalar>::set_streams_and_tiles(int streams, int tile_m, int tile_n, int tile_k) {
    n_streams = streams;
    tile_size_m = tile_m;
    tile_size_n = tile_n;
    tile_size_k = tile_k;

    ctx.set_num_streams(n_streams);
    a_buff.set_streams_and_tiles(streams, {tile_m, tile_k});
    b_buff.set_streams_and_tiles(streams, {tile_k, tile_n});
    c_buff.set_streams_and_tiles(streams, {tile_m, tile_n});
}

template <typename Scalar>
device_buffer<Scalar>& mm_handle<Scalar>::get_device_buffer_a() {
    return a_buff;
}

template <typename Scalar>
device_buffer<Scalar>& mm_handle<Scalar>::get_device_buffer_b() {
    return b_buff;
}

template <typename Scalar>
device_buffer<Scalar>& mm_handle<Scalar>::get_device_buffer_c() {
    return c_buff;
}

template class mm_handle<float>;
template class mm_handle<double>;
template class mm_handle<std::complex<float>>;
template class mm_handle<std::complex<double>>;

}
