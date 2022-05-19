#include <Tiled-MM/tiled_mm.hpp>
#include <Tiled-MM/device_vector.hpp>
#include <Tiled-MM/util.hpp>
#include <Tiled-MM/gpu_blas_handle.hpp>
#include <Tiled-MM/gpu_blas_api.hpp>

#include <options.hpp>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <chrono>

#ifdef TILED_MM_CUDA
#include <cublasXt.h>

void compute_reference(double* a, double* b, double* c,
        int m, int n, int k,
        double alpha, double beta) {
    cublasXtHandle_t handle;
    cublasXtCreate(&handle);
    int devices[1] = {0};
    cublasXtDeviceSelect(handle, 1, devices);

    // perform dgemm
    cublasXtDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k, &alpha, a, m, b, k, &beta, c, m);

    cudaDeviceSynchronize();

    if (handle)
        cublasXtDestroy(handle);
}
#endif


#ifdef TILED_MM_ROCM
#include "Tiled-MM/gpu_runtime_api.hpp"
void compute_reference(double* a, double* b, double* c,
        int m, int n, int k,
        double alpha, double beta) {
    gpu::device_vector<double> a_device(m * k);
    gpu::device_vector<double> b_device(k * n);
    gpu::device_vector<double> c_device(m * n);
    gpu::copy_to_device(a, a_device.data(), a_device.size());
    gpu::copy_to_device(b, b_device.data(), b_device.size());
    gpu::copy_to_device(c, c_device.data(), c_device.size());
    gpu::gpu_blas_handle handle;

    gpu::blas_api::dgemm(handle.handle(), gpu::blas_api::operation::None,
                         gpu::blas_api::operation::None, m, n, k, &alpha, a_device.data(), m,
                         b_device.data(), k, &beta, c_device.data(), m);

    gpu::copy_to_host(c_device.data(), c, c_device.size());
}

#endif

using value_type = double;
using size_type  = size_t;

template <typename T>
bool equal(T* v1, T* v2, size_t len, double eps=1e-6) {
    for (unsigned i = 0; i < len; ++i) {
        if (std::abs(*(v1 + i) - *(v2 + i)) > eps) {
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv){
    options::initialize(argc, argv);

    auto M = options::next_long_long("-m", "--m_dim", "Number of rows of A and C.", 10000);
    auto N = options::next_long_long("-n", "--n_dim", "Number of columns of B and C.", 10000);
    auto K = options::next_long_long("-k", "--k_dim", "Number of columns of A and rows of B.", 10000);

    std::cout << "Multiplying: (M, N, K) = (" << M << ", " << N << ", " << K << ")\n";

    // A dimensions: MxK
    auto a_host = gpu::malloc_pinned<value_type>(M*K, 1);
    // B dimensions: KxN
    auto b_host = gpu::malloc_pinned<value_type>(K*N, 1);
    // C dimensions: MxN
    auto c_host = gpu::malloc_pinned<value_type>(M*N, 0);
    auto c_host_reference = gpu::malloc_pinned<value_type>(M*N, 0);

    value_type alpha{1.};
    value_type beta{1.};

    compute_reference(a_host, b_host, c_host_reference, M, N, K, alpha, beta);

    auto ctx = gpu::make_context<double>();
    // compute c = alpha * a * b + beta * c
    gpu::gemm(*ctx, a_host, b_host, c_host, M, N, K, alpha, beta, false);

    bool correct = equal(c_host, c_host_reference, M*N);

    std::cout << "The result is " << (correct ? "CORRECT" : "NOT CORRECT") << std::endl;;

    return !correct;
}


