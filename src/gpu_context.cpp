#include "gpu_context.hpp"

namespace gpu {
gpu_context::gpu_context(int _streams): n_streams(_streams) {
    handles.reserve(n_streams);
    streams.reserve(n_streams);
    for (int stream_id = 0; stream_id < n_streams; ++stream_id) {
        handles.emplace_back();
        streams.emplace_back();
        // let each stream use a separate cuBLAS handle
        // and let each handle be bound to a separate stream
        auto status=
        cublasSetStream(get_cublas_handle(stream_id), get_cuda_stream(stream_id));
        cublas_check_status(status);
    }
}

cublasHandle_t gpu_context::get_cublas_handle(int stream_id) const {
    if (stream_id < 0 || stream_id >= n_streams) {
        throw std::runtime_error("index of cublas handle has to be in the range [0, n_streams)");
    }
    return handles[stream_id].handle();
}

cudaStream_t gpu_context::get_cuda_stream(int stream_id) const {
    if (stream_id < 0 || stream_id >= n_streams) {
        throw std::runtime_error("index of cublas handle has to be in the range [0, n_streams)");
    }
    return streams[stream_id].stream();
}

cuda_event gpu_context::enqueue_event(int stream_id) const {
    return streams[stream_id].enqueue_event();
}

int gpu_context::get_num_streams() const {
    return n_streams;
}

void gpu_context::set_num_streams(int _streams) {
    n_streams = _streams;
    handles.resize(_streams);
    streams.resize(_streams);
}

gpu_context::~gpu_context() {
    // std::cout << "freeing gpu_context" << std::endl;
    // std::cout << "streams.size = " << streams.size() << std::endl;
    // std::cout << "handles.size = " << handles.size() << std::endl;
}
}
