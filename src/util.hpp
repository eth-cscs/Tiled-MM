#pragma once
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <string>
#include <cmath>
//#include <mutex>

namespace gpu {

using value_type = double;
using size_type  = std::size_t;

static const char* cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}
///////////////////////////////////////////////////////////////////////////////
// CUDA error checking
///////////////////////////////////////////////////////////////////////////////
static void cuda_check_status(cudaError_t status) {
    if(status != cudaSuccess) {
        std::cerr << "error: CUDA API call : "
        << cudaGetErrorString(status) << std::endl;
        throw(std::runtime_error("CUDA ERROR"));
    }
}

static void cublas_check_status(cublasStatus_t status) {
    if(status != CUBLAS_STATUS_SUCCESS) {
        auto error = cudaGetErrorEnum(status); 
        std::cerr << "error: CUDABLAS API call: " << error << std::endl;
        throw(std::runtime_error("CUDA ERROR"));
    }
}

static void cuda_check_last_kernel(std::string const& errstr) {
    auto status = cudaGetLastError();
    if(status != cudaSuccess) {
        std::cout << "error: CUDA kernel launch : " << errstr << " : "
        << cudaGetErrorString(status) << std::endl;
        throw(std::runtime_error("CUDA ERROR"));
    }
}

///////////////////////////////////////////////////////////////////////////////
// TOTAL AVAILABLE MEMORY ON GPU 
///////////////////////////////////////////////////////////////////////////////
inline
std::size_t gpu_allocated_memory() {
    cudaDeviceSynchronize();
    auto status = cudaGetLastError();
    cuda_check_status(status);
    std::size_t free;
    std::size_t total;
    status = cudaMemGetInfo(&free, &total);
    return status == cudaSuccess ? total-free : -1;
}

///////////////////////////////////////////////////////////////////////////////
// allocating memory
///////////////////////////////////////////////////////////////////////////////

// allocate space on GPU for n instances of type T
template <typename T>
T* malloc_device(size_t n) {
    void* p;
    auto status = cudaMalloc(&p, n*sizeof(T));
    cuda_check_status(status);
    return (T*)p;
}

// allocate managed memory
template <typename T>
T* malloc_managed(size_t n, T value=T()) {
    T* p;
    auto status = cudaMallocManaged(&p, n*sizeof(T));
    cuda_check_status(status);
    std::fill(p, p+n, value);
    return p;
}

template <typename T>
T* malloc_pinned(size_t N, T value=T()) {
    T* ptr;
    auto status = cudaHostAlloc((void**)&ptr, N*sizeof(T), 0);
    cuda_check_status(status);
    std::fill(ptr, ptr+N, value);
    return ptr;
}

///////////////////////////////////////////////////////////////////////////////
// copying memory
///////////////////////////////////////////////////////////////////////////////

// copy n*T from host to device
template <typename T>
void copy_to_device(T* from, T* to, size_t n) {
    cudaMemcpy(to, from, n*sizeof(T), cudaMemcpyHostToDevice);
}

// copy n*T from device to host
template <typename T>
void copy_to_host(T* from, T* to, size_t n) {
    cudaMemcpy(to, from, n*sizeof(T), cudaMemcpyDeviceToHost);
}

// copy n*T from host to device
// If a cuda stream is passed as the final argument the copy will be performed
// asynchronously in the specified stream, otherwise it will be serialized in
// the default (NULL) stream
template <typename T>
void copy_to_device_async(const T* from, T* to, size_t n, cudaStream_t stream=NULL) {
    //cudaDeviceSynchronize();
    // auto status = cudaGetLastError();
    // if(status != cudaSuccess) {
    //    std::cout << "error: CUDA kernel launch:"
    //    << cudaGetErrorString(status) << std::endl;
    //    throw(std::runtime_error("CUDA ERROR"));
    //}

    auto status =
    cudaMemcpyAsync(to, from, n*sizeof(T), cudaMemcpyHostToDevice, stream);
    cuda_check_status(status);
}

// copy n*T from device to host
// If a cuda stream is passed as the final argument the copy will be performed
// asynchronously in the specified stream, otherwise it will be serialized in
// the default (NULL) stream
template <typename T>
void copy_to_host_async(const T* from, T* to, size_t n, cudaStream_t stream=NULL) {
    auto status =
    cudaMemcpyAsync(to, from, n*sizeof(T), cudaMemcpyDeviceToHost, stream);
    cuda_check_status(status);
}
}
