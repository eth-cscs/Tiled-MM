#include <iostream>
#include <cmath>
#include <cstdio>
#include <chrono>

#include <options.hpp>
#include <tiled_mm.hpp>

#include <cublasXt.h>

using value_type = double;
using size_type  = size_t;

int main(int argc, char** argv){
    options::initialize(argc, argv);

    auto M = options::next_long_long("-m", "--m_dim", "Number of rows of A and C.", 10000);
    auto N = options::next_long_long("-n", "--n_dim", "Number of columns of B and C.", 10000);
    auto K = options::next_long_long("-k", "--k_dim", "Number of columns of A and rows of B.", 10000);
    auto num_runs = options::next_int("-r", "--n_rep", "Number of repetitions", 1);

    std::cout << "Multiplying: (M, N, K) = (" << M << ", " << N << ", " << K << ")\n";

    size_t flops_per_mul = 2 * N * M * K;

    // A dimensions: MxK
    auto a_host = gpu::malloc_pinned<value_type>(M*K, 1);
    // B dimensions: KxN
    auto b_host = gpu::malloc_pinned<value_type>(K*N, 1);
    // C dimensions: MxN
    auto c_host = gpu::malloc_pinned<value_type>(M*N, 0);

    value_type alpha{1.};
    value_type beta{1.};

    cublasXtHandle_t handle;
    cublasXtCreate(&handle);
    int devices[1] = {0};
    cublasXtDeviceSelect(handle, 1, devices);
    cublasXtSetBlockDim(handle, 4000);

    auto start = std::chrono::steady_clock::now();
    for(int i=0; i<num_runs; ++i) {
        // perform dgemm
        cublasXtDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, &alpha, a_host, M, b_host, K, &beta, c_host, M);
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    auto time_per_mul =  time / num_runs;
    std::cout << "Avg Time [ms] = " << time_per_mul << std::endl;

    auto flops = flops_per_mul / (1e-3 * time_per_mul) / 1e9;
    std::cout << "Throughput [Gflops] = " << flops << std::endl;

    if (handle)
        cublasXtDestroy(handle);

    return 0;
}


