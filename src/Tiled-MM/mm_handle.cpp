#include "mm_handle.hpp"
#include "gpu_runtime_api.hpp"

#include<complex>
#include <cmath>
#include <cassert>

namespace gpu {

template <typename Scalar>
int get_tile_size(int n_streams, int max_tile_size, int ranks_per_gpu, double allowance_ratio) {
    size_t free, total;
    // query available memory
    runtime_api::mem_get_info(&free, &total);

    // use up to allowance_ratio % of the available memory
    // divide by 3, because 3 matrices A, B and C
    double memory_available = allowance_ratio * free / (3 * n_streams * sizeof(Scalar));
    // set tiles to be square by default, so take the sqrt
    int tile_size = (int) std::sqrt(memory_available);

    // don't make tiles larger than 5000
    tile_size = std::min(tile_size, max_tile_size);

    assert(tile_size > 0 && tile_size <= max_tile_size);

    return tile_size;
}

template <typename Scalar>
mm_handle<Scalar>::mm_handle(int ranks_per_gpu, double allowance_ratio): ctx(n_streams) {
    runtime_api::set_device(0);

    int tile_size = get_tile_size<Scalar>(n_streams, tile_size_m, ranks_per_gpu, allowance_ratio);
    tile_size_m = tile_size;
    tile_size_n = tile_size;
    tile_size_k = tile_size;

    a_buff = device_buffer<Scalar>(n_streams, {tile_size_m, tile_size_k});
    b_buff = device_buffer<Scalar>(n_streams, {tile_size_k, tile_size_n});
    c_buff = device_buffer<Scalar>(n_streams, {tile_size_m, tile_size_n});

    a_pinned = pinned_buffer<Scalar>(n_streams, {tile_size_m, tile_size_k});
    b_pinned = pinned_buffer<Scalar>(n_streams, {tile_size_k, tile_size_n});
    c_pinned = pinned_buffer<Scalar>(n_streams, {tile_size_m, tile_size_n});
}

template <typename Scalar>
mm_handle<Scalar>::mm_handle(int streams, int tile_m, int tile_n, int tile_k): n_streams(streams),
        tile_size_m(tile_m), tile_size_n(tile_n), tile_size_k(tile_k), ctx(streams) {
    runtime_api::set_device(0);

    a_buff = device_buffer<Scalar>(n_streams, {tile_size_m, tile_size_k});
    b_buff = device_buffer<Scalar>(n_streams, {tile_size_k, tile_size_n});
    c_buff = device_buffer<Scalar>(n_streams, {tile_size_m, tile_size_n});

    a_pinned = pinned_buffer<Scalar>(n_streams, {tile_size_m, tile_size_k});
    b_pinned = pinned_buffer<Scalar>(n_streams, {tile_size_k, tile_size_n});
    c_pinned = pinned_buffer<Scalar>(n_streams, {tile_size_m, tile_size_n});
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

    a_pinned.set_num_streams(streams);
    b_pinned.set_num_streams(streams);
    c_pinned.set_num_streams(streams);
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

    a_pinned.set_tile_dimensions({tile_m, tile_k});
    b_pinned.set_tile_dimensions({tile_k, tile_n});
    c_pinned.set_tile_dimensions({tile_m, tile_n});
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

    a_pinned.set_streams_and_tiles(streams, {tile_m, tile_k});
    b_pinned.set_streams_and_tiles(streams, {tile_k, tile_n});
    c_pinned.set_streams_and_tiles(streams, {tile_m, tile_n});
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

template <typename Scalar>
pinned_buffer<Scalar>& mm_handle<Scalar>::get_pinned_buffer_a() {
    return a_pinned;
}
template <typename Scalar>
pinned_buffer<Scalar>& mm_handle<Scalar>::get_pinned_buffer_b() {
    return b_pinned;
}
template <typename Scalar>
pinned_buffer<Scalar>& mm_handle<Scalar>::get_pinned_buffer_c() {
    return c_pinned;
}

template class mm_handle<float>;
template class mm_handle<double>;
template class mm_handle<std::complex<float>>;
template class mm_handle<std::complex<double>>;

}
