#pragma once
#include "gpu_context.hpp"
#include "device_buffer.hpp"
#include "cuda_stream.hpp"
#include "cublas_handle.hpp"
#include <tuple>
#include <memory>

namespace gpu{

class mm_handle {
public:
    mm_handle();
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

    device_buffer<double>& get_device_buffer_a();
    device_buffer<double>& get_device_buffer_b();
    device_buffer<double>& get_device_buffer_c();

private:
    int n_streams = 4;
    int tile_size_m = 4000;
    int tile_size_n = 4000;
    int tile_size_k = 4000;

    gpu_context ctx;

    device_buffer<double> a_buff;
    device_buffer<double> b_buff;
    device_buffer<double> c_buff;
};

typedef std::unique_ptr<mm_handle> context;
context make_context();
context make_context(int streams, int tile_m, int tile_n, int tile_k);

}
