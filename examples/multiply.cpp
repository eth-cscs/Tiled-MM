#include <Tiled-MM/tiled_mm.hpp>

#include <options.hpp>

#include <iostream>
#include <cmath>
#include <cstdio>
#include <chrono>

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

    auto ctx = gpu::make_context<double>(2, 5000, 5000, 5000);

    std::cout << "==============================================" << std::endl;
    std::cout << "The version with copying C to back to host: " << std::endl;
    std::cout << "==============================================" << std::endl;

    auto start = std::chrono::steady_clock::now();
    for(int i=0; i<num_runs + 1; ++i) {
        // compute c = alpha * a * b + beta * c
        // no need to pin the memory, since matrices already pinned
        if (i == 1) {
            start = std::chrono::steady_clock::now();
        }
        bool pin_buffers = false;
        bool copy_c_back = true;
        gpu::gemm(*ctx, a_host, b_host, c_host, 'N', 'N', M, N, K, alpha, beta, pin_buffers, copy_c_back);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    auto time_per_mul =  time / num_runs;
    std::cout << "Avg Time [ms] = " << time_per_mul << std::endl;

    auto flops = flops_per_mul / (1e-3 * time_per_mul) / 1e9;
    std::cout << "Throughput [Gflops] = " << flops << std::endl;

    std::cout << "==============================================" << std::endl;
    std::cout << "The version without copying C to back to host: " << std::endl;
    std::cout << "==============================================" << std::endl;

    start = std::chrono::steady_clock::now();
    for(int i=0; i<num_runs+1; ++i) {
        // compute c = alpha * a * b + beta * c
        // no need to pin the memory, since matrices already pinned
        if (i == 1) {
            start = std::chrono::steady_clock::now();
        }
        bool pin_buffers = false;
        bool copy_c_back = false;
        gpu::gemm(*ctx, a_host, b_host, c_host, 'N', 'N', M, N, K, alpha, beta, pin_buffers, copy_c_back);
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    time_per_mul =  time / num_runs;
    std::cout << "Avg Time [ms] = " << time_per_mul << std::endl;

    flops = flops_per_mul / (1e-3 * time_per_mul) / 1e9;
    std::cout << "Throughput [Gflops] = " << flops << std::endl;

    return 0;
}


