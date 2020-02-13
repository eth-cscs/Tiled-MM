#include "tiled_mm.hpp"

#include "util.hpp"
#include "device_stream.hpp"
#include "device_event.hpp"
#include "gpu_blas_handle.hpp"
#include "tile_coord.hpp"
#include "gpu_context.hpp"
#include "device_2d_buffer.hpp"
#include "tiled_matrix.hpp"
#include "tile_coord.hpp"
#include "gpu_blas_api.hpp"
#include "gpu_runtime_api.hpp"

// #include <omp.h>
// #include <cublasXt.h>
// #include <libsci_acc.h>

#include <vector>
#include <cstring>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <complex>
#include <tuple>

namespace gpu {

using zfloat = std::complex<float>;
using zdouble = std::complex<double>;

template<typename Scalar>
void copy_tile_to_device_async(
        tiled_matrix<Scalar>& tiled_mat, 
        Scalar* d_buffer,
        std::size_t d_buffer_pitch,
        tile_coord tile, device_stream& stream) {
    Scalar* from = tiled_mat.tile_data(tile);
    Scalar* to = d_buffer;

    tile_dim tile_dims = tiled_mat.tile_dimensions(tile);
    // std::cout << "host->device" << std::endl;

    auto status=
    runtime_api::memcpy_2d_async(to, d_buffer_pitch,
            from, tiled_mat.rows() * sizeof(Scalar),
            tile_dims.rows() * sizeof(Scalar), tile_dims.cols(),
            runtime_api::flag::MemcpyHostToDevice, stream.stream());
    check_runtime_status(status);
}

template<typename Scalar>
void copy_tile_to_device_async(
        tiled_matrix<Scalar>& tiled_mat, 
        device_2d_buffer<Scalar>& d_buffer,
        std::size_t d_buffer_pitch,
        tile_coord tile, gpu_context& ctx, int stream_id) {
    copy_tile_to_device_async(
            tiled_mat, 
            d_buffer.stream_buffer(stream_id), 
            d_buffer_pitch, 
            tile, 
            ctx.get_device_stream(stream_id));
}

template<typename Scalar>
void copy_tile_to_host_async(tiled_matrix<Scalar>& tiled_mat,
                             Scalar* d_buffer,
                             std::size_t d_buffer_pitch,
                             tile_coord tile,
                             device_stream& stream) {
    Scalar* from  = d_buffer;
    Scalar* to = tiled_mat.tile_data(tile);

    tile_dim tile_dims = tiled_mat.tile_dimensions(tile);

    // std::cout << "device->host" << std::endl;
    auto status=
    runtime_api::memcpy_2d_async(to, tiled_mat.rows() * sizeof(Scalar),
            from, d_buffer_pitch,
            tile_dims.rows() * sizeof(Scalar), tile_dims.cols(),
            runtime_api::flag::MemcpyDeviceToHost, stream.stream());
    check_runtime_status(status);
}

template<typename Scalar>
void copy_tile_to_host_async(
        tiled_matrix<Scalar>& tiled_mat, 
        device_2d_buffer<Scalar>& d_buffer,
        std::size_t d_buffer_pitch,
        tile_coord tile, gpu_context& ctx, int stream_id) {
    copy_tile_to_host_async(tiled_mat, d_buffer.stream_buffer(stream_id), tile, ctx.get_device_stream(stream_id));
}


template<typename Scalar>
std::tuple<int, int, int> get_num_tiles(tiled_matrix<Scalar>& a, tiled_matrix<Scalar>& b, tiled_matrix<Scalar>& c) {
    if (a.num_tiles_row() != c.num_tiles_row() ||
            a.num_tiles_col() != b.num_tiles_row() ||
            b.num_tiles_col() != c.num_tiles_col()) {
        throw std::runtime_error("Number of tiles mismatch in tiled_matrix inside get_num_tiles.");
    }
    return {a.num_tiles_row(), c.num_tiles_col(), b.num_tiles_row()};
}


template<typename Scalar>
std::tuple<int, int, int> get_tile_sizes(tiled_matrix<Scalar>& a,
        tiled_matrix<Scalar>& b, tiled_matrix<Scalar>& c,
        int m_tile_id, int n_tile_id, int k_tile_id) {
    tile_dim a_dim = a.tile_dimensions({m_tile_id, k_tile_id});
    tile_dim b_dim = b.tile_dimensions({k_tile_id, n_tile_id});
    tile_dim c_dim = c.tile_dimensions({m_tile_id, n_tile_id});

    if (a_dim.cols() != b_dim.rows() ||
            a_dim.rows() != c_dim.rows() ||
            b_dim.cols() != c_dim.cols()) {
        throw std::runtime_error("Tile dimension mismatch in device_2d_buffers inside get_tile_sizes.");
    }
    return {a_dim.rows(), b_dim.cols(), a_dim.cols()};
}



blas_api::StatusType cublas_gemm_wrapper(blas_api::HandleType handle,
                                   int m, int n, int k,
                                   const float* alpha,
                                   const float* a,
                                   const float* b,
                                   const float* beta,
                                   float* c) {
  return blas_api::sgemm(handle, blas_api::operation::None, blas_api::operation::None, m, n, k,
                         alpha, a, m, b, k, beta, c, m);
}

blas_api::StatusType cublas_gemm_wrapper(blas_api::HandleType handle,
                                   int m, int n, int k,
                                   const double* alpha,
                                   const double* a,
                                   const double* b,
                                   const double* beta,
                                   double* c) {
  return blas_api::dgemm(handle, blas_api::operation::None, blas_api::operation::None, m, n, k,
                         alpha, a, m, b, k, beta, c, m);
}

// Note: Converting from std::complex to cuComplex and cuDoubleComple
//       works because they are binary compatible.
//
//       http://icl.cs.utk.edu/magma/forum/viewtopic.php?f=2&t=902
//
blas_api::StatusType cublas_gemm_wrapper(blas_api::HandleType handle,
                                   int m, int n, int k,
                                   const zfloat* alpha,
                                   const zfloat* a,
                                   const zfloat* b,
                                   const zfloat* beta,
                                   zfloat* c) {
  return blas_api::cgemm(handle, blas_api::operation::None, blas_api::operation::None, m, n, k,
                         reinterpret_cast<const blas_api::ComplexFloatType*>(alpha),
                         reinterpret_cast<const blas_api::ComplexFloatType*>(a), m,
                         reinterpret_cast<const blas_api::ComplexFloatType*>(b), k,
                         reinterpret_cast<const blas_api::ComplexFloatType*>(beta),
                         reinterpret_cast<blas_api::ComplexFloatType*>(c), m);
}

blas_api::StatusType cublas_gemm_wrapper(blas_api::HandleType handle,
                                   int m, int n, int k,
                                   const zdouble* alpha,
                                   const zdouble* a,
                                   const zdouble* b,
                                   const zdouble* beta,
                                   zdouble* c) {
  return blas_api::zgemm(handle, blas_api::operation::None, blas_api::operation::None, m, n, k,
                         reinterpret_cast<const blas_api::ComplexDoubleType*>(alpha),
                         reinterpret_cast<const blas_api::ComplexDoubleType*>(a), m,
                         reinterpret_cast<const blas_api::ComplexDoubleType*>(b), k,
                         reinterpret_cast<const blas_api::ComplexDoubleType*>(beta),
                         reinterpret_cast<blas_api::ComplexDoubleType*>(c), m);
}

blas_api::StatusType cublas_gemm_wrapper(blas_api::HandleType handle,
                                   int m, int n, int k,
                                   const float* alpha,
                                   const float* a,
                                   const int lld_a,
                                   const float* b,
                                   const int lld_b,
                                   const float* beta,
                                   float* c,
                                   const int lld_c) {
  return blas_api::sgemm(handle, blas_api::operation::None, blas_api::operation::None, m, n, k,
                         alpha, a, lld_a, b, lld_b, beta, c, lld_c);
}

blas_api::StatusType cublas_gemm_wrapper(blas_api::HandleType handle,
                                   int m, int n, int k,
                                   const double* alpha,
                                   const double* a,
                                   const int lld_a,
                                   const double* b,
                                   const int lld_b,
                                   const double* beta,
                                   double* c,
                                   const int lld_c) {
  return blas_api::dgemm(handle, blas_api::operation::None, blas_api::operation::None, m, n, k,
                         alpha, a, lld_a, b, lld_b, beta, c, lld_c);
}

// Note: Converting from std::complex to cuComplex and cuDoubleComple
//       works because they are binary compatible.
//
//       http://icl.cs.utk.edu/magma/forum/viewtopic.php?f=2&t=902
//
blas_api::StatusType cublas_gemm_wrapper(blas_api::HandleType handle,
                                   int m, int n, int k,
                                   const zfloat* alpha,
                                   const zfloat* a,
                                   const int lld_a,
                                   const zfloat* b,
                                   const int lld_b,
                                   const zfloat* beta,
                                   zfloat* c,
                                   const int lld_c) {
  return blas_api::cgemm(handle, blas_api::operation::None, blas_api::operation::None, m, n, k,
                         reinterpret_cast<const blas_api::ComplexFloatType*>(alpha),
                         reinterpret_cast<const blas_api::ComplexFloatType*>(a), lld_a,
                         reinterpret_cast<const blas_api::ComplexFloatType*>(b), lld_b,
                         reinterpret_cast<const blas_api::ComplexFloatType*>(beta),
                         reinterpret_cast<blas_api::ComplexFloatType*>(c), lld_c);
}

blas_api::StatusType cublas_gemm_wrapper(blas_api::HandleType handle,
                                   int m, int n, int k,
                                   const zdouble* alpha,
                                   const zdouble* a,
                                   const int lld_a,
                                   const zdouble* b,
                                   const int lld_b,
                                   const zdouble* beta,
                                   zdouble* c,
                                   const int lld_c) {
  return blas_api::zgemm(handle, blas_api::operation::None, blas_api::operation::None, m, n, k,
                         reinterpret_cast<const blas_api::ComplexDoubleType*>(alpha),
                         reinterpret_cast<const blas_api::ComplexDoubleType*>(a), lld_a,
                         reinterpret_cast<const blas_api::ComplexDoubleType*>(b), lld_b,
                         reinterpret_cast<const blas_api::ComplexDoubleType*>(beta),
                         reinterpret_cast<blas_api::ComplexDoubleType*>(c), lld_c);
}

template<typename Scalar>
void round_robin(tiled_matrix<Scalar>& a_host, tiled_matrix<Scalar>& b_host, tiled_matrix<Scalar>& c_host,
        device_2d_buffer<Scalar>& a_device,
        device_2d_buffer<Scalar>& b_device,
        device_2d_buffer<Scalar>& c_device,
        int m, int n, int k, Scalar alpha, Scalar beta, mm_handle<Scalar>& handle) {

    int n_tiles_m, n_tiles_n, n_tiles_k;
    std::tie(n_tiles_m, n_tiles_n, n_tiles_k) = get_num_tiles(a_host, b_host, c_host);

    int n_streams = std::min(handle.get_num_streams(), n_tiles_m * n_tiles_n);
    auto& gpu_ctx = handle.get_gpu_context();

    auto& result_stream = gpu_ctx.get_result_stream();

    std::vector<device_event> c_computed_on_device(n_streams);
    std::vector<device_event> c_copied_to_device(n_streams);

    for (int i = 0; i < n_tiles_m * n_tiles_n; i += n_streams) {
        for (int k_tile_id = 0; k_tile_id < n_tiles_k; ++k_tile_id) {
            for (int round = 0; round < 2; ++round) {
                int current_i = i;

                for (int stream_id = 0; stream_id < n_streams
                        && current_i < n_tiles_m * n_tiles_n; ++stream_id) {

                    int m_tile_id = current_i / n_tiles_n;
                    int n_tile_id = current_i % n_tiles_n;

                    int actual_size_m, actual_size_n, actual_size_k;
                    std::tie(actual_size_m, actual_size_n, actual_size_k) =
                        get_tile_sizes(a_host, b_host, c_host,
                                m_tile_id, n_tile_id, k_tile_id);

                    Scalar new_beta = k_tile_id == 0 ? beta : 1.0;

                    auto& current_stream = gpu_ctx.get_device_stream(stream_id);

                    if (round == 0) {
                        // copy A tile
                        copy_tile_to_device_async(a_host, a_device,
                                a_device.pitch(),
                                {m_tile_id, k_tile_id},
                                gpu_ctx, stream_id);

                        // copy B tile
                        copy_tile_to_device_async(b_host, b_device,
                                b_device.pitch(),
                                {k_tile_id, n_tile_id},
                                gpu_ctx, stream_id);

                        // copy C tile if this is the first partial result and beta > 0
                        if (k_tile_id == 0 && std::abs(beta) > 0) {
                            current_stream.wait_on_event(c_copied_to_device[stream_id]);
                            copy_tile_to_device_async(c_host, c_device,
                                    c_device.pitch(),
                                    {m_tile_id, n_tile_id},
                                    gpu_ctx, stream_id);
                        }
                    } else {
                        // perform dgemm
                        // cublasSetStream(get_blas_handle(stream_id), streams[stream_id].stream());
                        // std::cout << "performing dgemm" << std::endl;
                        auto& gemm_stream = gpu_ctx.get_device_stream(stream_id);
                        gemm_stream.wait_on_event(c_copied_to_device[stream_id]);
                        auto status = cublas_gemm_wrapper(
                                gpu_ctx.get_blas_handle(stream_id),
                                actual_size_m, actual_size_n, actual_size_k,
                                &alpha,
                                a_device.stream_buffer(stream_id), a_device.pitch()/sizeof(Scalar),
                                b_device.stream_buffer(stream_id), b_device.pitch()/sizeof(Scalar),
                                &new_beta,
                                c_device.stream_buffer(stream_id), c_device.pitch()/sizeof(Scalar));
                        check_blas_status(status);

                        c_computed_on_device[stream_id] = gemm_stream.enqueue_event();

                        if (k_tile_id == n_tiles_k - 1) {
                            // copy result back to host
                            result_stream.wait_on_event(c_computed_on_device[stream_id]);
                            copy_tile_to_host_async(c_host, c_device.stream_buffer(stream_id),
                                    c_device.pitch(),
                                    {m_tile_id, n_tile_id},
                                    result_stream);
                            c_copied_to_device[stream_id] = gemm_stream.enqueue_event();
                        }
                    }
                    current_i++;
                }
            }
        }
    }
}



/*
void gpu_dgemm_(mm_handle& m_handle, double* a, double* b, double* c,
        int m, int n, int k,
        double alpha, double beta) {

    int tile_size_m, tile_size_n, tile_size_k;
    std::tie(tile_size_m, tile_size_n, tile_size_k) = m_handle.get_tile_sizes();

    // perform dgemm
    dgemm('N', 'N', m, n, k, alpha, a, m, b, k, beta, c, m);
}

void gpu_dgemm_(mm_handle& m_handle, double* a, double* b, double* c,
        int m, int n, int k,
        double alpha, double beta) {
    cublasXtHandle_t handle;
    cublasXtCreate(&handle);
    int devices[1] = {0};
    cublasXtDeviceSelect(handle, 1, devices);

    int tile_size_m, tile_size_n, tile_size_k;
    std::tie(tile_size_m, tile_size_n, tile_size_k) = m_handle.get_tile_sizes();

    cublasXtSetBlockDim(handle, tile_size_m);

    // cublasXtSetCpuRoutine(handle, CUBLASXT_GEMM, CUBLASXT_DOUBLE, (void*)(&dgemm_));
    // perform dgemm
    cublasXtDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k, &alpha, a, m, b, k, &beta, c, m);

    cudaDeviceSynchronize();

    if (handle)
        cublasXtDestroy(handle);
}
*/
template<typename Scalar>
void gemm(mm_handle<Scalar>& handle, Scalar* a, Scalar* b, Scalar* c,
        int m, int n, int k,
        Scalar alpha, Scalar beta) {
    int tile_size_m, tile_size_n, tile_size_k;
    std::tie(tile_size_m, tile_size_n, tile_size_k) = handle.get_tile_sizes();

    tiled_matrix<Scalar> a_host(a, m, k, {tile_size_m, tile_size_k});
    tiled_matrix<Scalar> b_host(b, k, n, {tile_size_k, tile_size_n});
    tiled_matrix<Scalar> c_host(c, m, n, {tile_size_m, tile_size_n});

    device_2d_buffer<Scalar>& a_device = handle.get_device_buffer_a();
    device_2d_buffer<Scalar>& b_device = handle.get_device_buffer_b();
    device_2d_buffer<Scalar>& c_device = handle.get_device_buffer_c();

    round_robin(a_host, b_host, c_host,
                a_device, b_device, c_device,
                m, n, k, alpha, beta, handle);

    auto status =
    runtime_api::device_synchronize();
    check_runtime_status(status);
}



template void gemm<float>(mm_handle<float>& handle, float* a, float* b, float* c,
        int m, int n, int k,
        float alpha, float beta);

template void gemm<double>(mm_handle<double>& handle, double* a, double* b, double* c,
        int m, int n, int k,
        double alpha, double beta);

template void gemm<zfloat>(mm_handle<zfloat>& handle, zfloat* a, zfloat* b, zfloat* c,
        int m, int n, int k,
        zfloat alpha, zfloat beta);

template void gemm<zdouble>(mm_handle<zdouble>& handle, zdouble* a, zdouble* b, zdouble* c,
        int m, int n, int k,
        zdouble alpha, zdouble beta);

}
