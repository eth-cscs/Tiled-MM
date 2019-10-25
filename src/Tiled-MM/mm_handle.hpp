#pragma once
#include "gpu_context.hpp"
#include "device_buffer.hpp"
#include "cuda_stream.hpp"
#include "cublas_handle.hpp"
#include <tuple>
#include <memory>

namespace gpu{

template <typename Scalar>
class mm_handle {
public:
    mm_handle(int ranks_per_gpu);
    mm_handle(int streams, int tile_m, int tile_n, int tile_k);
    ~mm_handle();

    mm_handle(mm_handle&&) = delete;
    mm_handle(const mm_handle&) = delete;
    mm_handle& operator=(const mm_handle&& other) = delete;

    void set_num_streams(int streams);
    int get_num_streams();

    gpu_context& get_gpu_context();

    void set_tile_sizes(int tile_size_m, int tile_size_n, int tile_size_k);
    void set_tile_sizes(int tile_size);
    std::tuple<int, int, int> get_tile_sizes();

    void set_streams_and_tiles(int streams, int tile_size_m, int tile_size_n, int tile_size_k);

    device_buffer<Scalar>& get_device_buffer_a();
    device_buffer<Scalar>& get_device_buffer_b();
    device_buffer<Scalar>& get_device_buffer_c();

private:
    int n_streams = 2;
    int tile_size_m = 5000;
    int tile_size_n = 5000;
    int tile_size_k = 5000;

    gpu_context ctx;

    device_buffer<Scalar> a_buff;
    device_buffer<Scalar> b_buff;
    device_buffer<Scalar> c_buff;
};

template <typename Scalar>
std::unique_ptr<mm_handle<Scalar>> make_context(int ranks_per_gpu = 1) {
    return std::make_unique<mm_handle<Scalar>>(ranks_per_gpu);
}

template <typename Scalar>
std::unique_ptr<mm_handle<Scalar>> make_context(int streams, int tile_m, int tile_n, int tile_k) {
    return std::make_unique<mm_handle<Scalar>>(streams, tile_m, tile_n, tile_k);
}

}
