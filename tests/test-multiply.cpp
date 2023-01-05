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
#include <random>

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
void fill_matrix(T* ptr, size_t size) {
    static std::random_device dev;                        // seed
    static std::mt19937 rng(dev());                       // generator
    static std::uniform_real_distribution<T> dist(10.0); // distribution

    for (unsigned i = 0; i < size; ++i) {
        ptr[i] = T{dist(rng)};
    }
}

template <typename T>
void copy_matrix(T* from, T* to, std::size_t size) {
    for (unsigned i = 0; i < size; ++i) {
        to[i] = from[i];
    }
}

template <typename T>
bool equal(T* v1, T* v2, size_t len, double eps=1e-6) {
    for (unsigned i = 0; i < len; ++i) {
        auto value1 = *(v1 + i);
        auto value2 = *(v2 + i);
        if (std::abs(value1 - value2) > eps) {
            return false;
        }
    }
    return true;
}

template <typename T>
void print_matrix(T* mat, int m, int n) {
    for (unsigned i = 0; i < m; ++i) {
        for (unsigned j = 0; j < n; ++j) {
            auto el = j * m + i;
            std::cout << mat[el] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "\n";
}

int main(int argc, char** argv){
    options::initialize(argc, argv);

    auto M = options::next_long_long("-m", "--m_dim", "Number of rows of A and C.", 10000);
    auto N = options::next_long_long("-n", "--n_dim", "Number of columns of B and C.", 10000);
    auto K = options::next_long_long("-k", "--k_dim", "Number of columns of A and rows of B.", 10000);

    bool small_sizes = std::max(M, std::max(N, K)) < 20;

    std::cout << "Multiplying: (M, N, K) = (" << M << ", " << N << ", " << K << ")\n";

    // A dimensions: MxK
    auto a_host = gpu::malloc_pinned<value_type>(M*K, 1);
    // B dimensions: KxN
    auto b_host = gpu::malloc_pinned<value_type>(K*N, 1);
    // C dimensions: MxN
    auto c_host = gpu::malloc_pinned<value_type>(M*N, 0);
    auto c_host2 = gpu::malloc_pinned<value_type>(M*N, 0);
    auto c_host_reference = gpu::malloc_pinned<value_type>(M*N, 0);

    fill_matrix(a_host, M * K);
    fill_matrix(b_host, K * N);
    fill_matrix(c_host, M * N);

    copy_matrix(c_host, c_host_reference, M * N);
    copy_matrix(c_host, c_host2, M * N);

    if (small_sizes) {
        std::cout << "Initial values in matrix A: " << std::endl;
        print_matrix(a_host, M, K);
        std::cout << "Initial values in matrix B: " << std::endl;
        print_matrix(b_host, K, N);
        std::cout << "Initial values in matrix C: " << std::endl;
        print_matrix(c_host, M, N);
    }

    value_type alpha{1.};
    value_type beta{1.};

    compute_reference(a_host, b_host, c_host_reference, M, N, K, alpha, beta);

    if (small_sizes) {
        std::cout << "Correct result C = C + A*B: " << std::endl;
        print_matrix(c_host_reference, M, N);
    }

    auto ctx = gpu::make_context<double>();
    // VERSION WITH COPYING C BACK
    bool copy_c_back = true;
    // compute c = alpha * a * b + beta * c

    auto start = std::chrono::steady_clock::now();
    gpu::gemm(*ctx, a_host, b_host, c_host, 'N', 'N', M, N, K, alpha, beta, false, copy_c_back);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time [ms] with copying C back: " << duration << std::endl;

    if (small_sizes) {
        std::cout << "Computed result by Tiled-MM with copying C back : " << std::endl;
        print_matrix(c_host, M, N);
    }

    bool correct = equal(c_host, c_host_reference, M*N);

    // VERSION WITHOUT COPYING C BACK
    // compute the same but don't copy c back
    start = std::chrono::steady_clock::now();
    gpu::gemm(*ctx, a_host, b_host, c_host2, 'N', 'N', M, N, K, alpha, beta, false, !copy_c_back);
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time [ms] without copying C back: " << duration << std::endl;

    gpu::copy_to_host(ctx->get_full_device_buffer_c().data(), c_host2, M * N);

    if (small_sizes) {
        std::cout << "Computed result by Tiled-MM without copying C back : " << std::endl;
        print_matrix(c_host2, M, N);
    }

    correct = correct && equal(c_host2, c_host_reference, M*N);

    std::cout << "The result is " << (correct ? "CORRECT" : "NOT CORRECT") << std::endl;;

    return !correct;
}


