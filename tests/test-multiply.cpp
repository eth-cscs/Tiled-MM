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

#include "Tiled-MM/gpu_runtime_api.hpp"
void compute_reference(double* a, double* b, double* c,
        int m, int n, int k,
	int ld_a, int ld_b, int ld_c,
        double alpha, double beta,
	char trans_a, char trans_b) {
    // sumatrix size to multiply
    int a_m = trans_a == 'N' ? m : k;
    int a_n = trans_a == 'N' ? k : m;

    int b_m = trans_b == 'N' ? k : n;
    int b_n = trans_b == 'N' ? n : k;

    int c_m = m;
    int c_n = n;

    gpu::device_vector<double> a_device(a_m * a_n);
    gpu::device_vector<double> b_device(b_m * b_n);
    gpu::device_vector<double> c_device(c_m * c_n);

    gpu::copy_to_device(a, a_device.data(), a_device.size());
    gpu::copy_to_device(b, b_device.data(), b_device.size());
    gpu::copy_to_device(c, c_device.data(), c_device.size());

    gpu::gpu_blas_handle handle;

    auto transa = gpu::get_blas_operation(trans_a);
    auto transb = gpu::get_blas_operation(trans_b);

    gpu::blas_api::dgemm(handle.handle(), transa, transb,
                         m, n, k, 
			 &alpha, 
			 a_device.data(), ld_a,
                         b_device.data(), ld_b, 
			 &beta, 
			 c_device.data(), ld_c);

    gpu::copy_to_host(c_device.data(), c, c_device.size());
}

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
void print_matrix(T* mat, int m, int n, char trans) {
    if (trans != 'N') {
        std::swap(m, n);
    }

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

    auto m = options::next_long_long("-m", "--m_dim", "Number of rows of A and C.", 10000);
    auto n = options::next_long_long("-n", "--n_dim", "Number of columns of B and C.", 10000);
    auto k = options::next_long_long("-k", "--k_dim", "Number of columns of A and rows of B.", 10000);

    char trans_a = 'N';
    char trans_b = 'N';

    // sumatrix size to multiply
    int a_m = trans_a == 'N' ? m : k;
    int a_n = trans_a == 'N' ? k : m;

    int b_m = trans_b == 'N' ? k : n;
    int b_n = trans_b == 'N' ? n : k;

    int c_m = m;
    int c_n = n;

    int ld_a = a_m;
    int ld_b = b_m;
    int ld_c = c_m;

    bool small_sizes = std::max(m, std::max(n, k)) < 20;

    std::cout << "Multiplying: (M, N, K) = (" << m << ", " << n << ", " << k << ")\n";

    // A dimensions: MxK
    auto a_host = gpu::malloc_pinned<value_type>(a_m * a_n, 1);
    // B dimensions: KxN
    auto b_host = gpu::malloc_pinned<value_type>(b_m * b_n, 1);
    // C dimensions: MxN
    auto c_host = gpu::malloc_pinned<value_type>(c_m * c_n, 0);
    auto c_host2 = gpu::malloc_pinned<value_type>(c_m * c_n, 0);
    auto c_host_reference = gpu::malloc_pinned<value_type>(c_m * c_n, 0);

    fill_matrix(a_host, a_m * a_n);
    fill_matrix(b_host, b_m * b_n);
    fill_matrix(c_host, c_m * c_n);

    copy_matrix(c_host, c_host_reference, c_m * c_n);
    copy_matrix(c_host, c_host2, c_m * c_n);

    if (small_sizes) {
        std::cout << "Initial values in matrix A: " << std::endl;
        print_matrix(a_host, a_m, a_n, trans_a);
        std::cout << "Initial values in matrix B: " << std::endl;
        print_matrix(b_host, b_m, b_n, trans_b);
        std::cout << "Initial values in matrix C: " << std::endl;
        print_matrix(c_host, c_m, c_n, 'N');
    }

    value_type alpha{1.};
    value_type beta{1.};

    compute_reference(a_host, b_host, c_host_reference, 
		      m, n, k, 
		      ld_a, ld_b, ld_c,
		      alpha, beta, 
		      trans_a, trans_b);

    if (small_sizes) {
        std::cout << "Correct result C = C + A*B: " << std::endl;
        print_matrix(c_host_reference, m, n, 'N');
    }

    auto ctx = gpu::make_context<double>();
    // VERSION WITH COPYING C BACK
    bool copy_c_back = true;
    // compute c = alpha * a * b + beta * c

    auto start = std::chrono::steady_clock::now();
    gpu::gemm(*ctx, a_host, b_host, c_host, 
              trans_a, trans_b, 
              m, n, k, 
	      ld_a, ld_b, ld_c, 
	      alpha, beta, 
	      false, copy_c_back);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time [ms] with copying C back: " << duration << std::endl;

    if (small_sizes) {
        std::cout << "Computed result by Tiled-MM with copying C back : " << std::endl;
        print_matrix(c_host, m, n, 'N');
    }

    bool correct = equal(c_host, c_host_reference, c_m*c_n);

    // VERSION WITHOUT COPYING C BACK
    // compute the same but don't copy c back
    start = std::chrono::steady_clock::now();
    gpu::gemm(*ctx, a_host, b_host, c_host2, 
              trans_a, trans_b, 
	      m, n, k, 
	      ld_a, ld_b, ld_c, 
	      alpha, beta, 
	      false, !copy_c_back);
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time [ms] without copying C back: " << duration << std::endl;

    gpu::copy_to_host(ctx->get_full_device_buffer_c().data(), c_host2, c_m * c_n);

    if (small_sizes) {
        std::cout << "Computed result by Tiled-MM without copying C back : " << std::endl;
        print_matrix(c_host2, c_m, c_n, 'N');
    }

    correct = correct && equal(c_host2, c_host_reference, c_m*c_n);

    std::cout << "The result is " << (correct ? "CORRECT" : "NOT CORRECT") << std::endl;;

    return !correct;
}


