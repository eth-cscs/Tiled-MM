#pragma once
#include "cublas_handle.hpp"
#include "cuda_stream.hpp"
#include "cuda_event.hpp"
#include <vector>
#include <stdexcept>
#include "util.hpp"

namespace gpu {

class gpu_context {
public:
    gpu_context(int streams);

    gpu_context(gpu_context&&) = delete;
    gpu_context(gpu_context&) = delete;
    // context& operator=(context&& other) = delete;

    cublasHandle_t get_cublas_handle(int stream_id) const;

    cudaStream_t get_cuda_stream(int stream_id) const;

    cuda_event enqueue_event(int stream_id) const;

    int get_num_streams() const;
    void set_num_streams(int streams);

    ~gpu_context();

private:
    int n_streams;
    std::vector<cublas_handle> handles;
    std::vector<cuda_stream> streams;
};

}
