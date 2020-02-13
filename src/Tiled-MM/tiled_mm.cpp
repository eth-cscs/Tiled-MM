#include "tiled_mm.hpp"

#include "util.hpp"
#include "device_stream.hpp"
#include "device_event.hpp"
#include "gpu_blas_handle.hpp"
#include "tile_coord.hpp"
#include "gpu_context.hpp"
#include "device_buffer.hpp"
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

// copies n entries of elem_type from src_ptr to desc_ptr
template <typename elem_type>
void copy(std::size_t n, const elem_type *src_ptr, elem_type *dest_ptr) {
    static_assert(std::is_trivially_copyable<elem_type>(),
                  "Element type must be trivially copyable!");
    std::memcpy(dest_ptr, src_ptr, sizeof(elem_type) * n);
}
// copies 2D block of given size from src_ptr with stride ld_src
// to dest_ptr with stride ld_dest
template <class elem_type>
void copy2D(const tile_dim &block_dim,
            const elem_type *src_ptr,
            int ld_src,
            elem_type *dest_ptr,
            int ld_dest) {
    static_assert(std::is_trivially_copyable<elem_type>(),
                  "Element type must be trivially copyable!");
    auto block_size = block_dim.rows() * block_dim.cols();
    // std::cout << "invoking copy2D." << std::endl;
    if (!block_size) {
        return;
    }

    // if not strided, copy in a single piece
    if (block_dim.rows() == (size_t)ld_src &&
        block_dim.rows() == (size_t)ld_dest) {
        copy(block_size, src_ptr, dest_ptr);
    } else {
        for (size_t col = 0; col < block_dim.cols(); ++col) {
            copy(block_dim.rows(),
                 src_ptr + ld_src * col,
                 dest_ptr + ld_dest * col);
        }
    }
}

template<typename Scalar>
void copy_tile_to_pinned_buffer(
        tiled_matrix<Scalar>& tiled_mat, 
        Scalar* pinned_buff,
        tile_coord tile) {
    Scalar* from = tiled_mat.tile_data(tile);
    Scalar* to = pinned_buff;

    tile_dim tile_dims = tiled_mat.tile_dimensions(tile);
    copy2D<Scalar>(tile_dims, from, tiled_mat.rows(), to, tile_dims.rows());

    std::cout << "pinned_buffer content = " << std::endl;
    for (int i = 0; i < tile_dims.rows() * tile_dims.cols(); ++i) {
        std::cout << to[i] << ", ";
    }
    std::cout << std::endl;
}

template<typename Scalar>
void copy_tile_from_pinned_buffer(
        Scalar* pinned_buff,
        tiled_matrix<Scalar>& tiled_mat, 
        tile_coord tile) {
    Scalar* from = pinned_buff;
    Scalar* to = tiled_mat.tile_data(tile);

    tile_dim tile_dims = tiled_mat.tile_dimensions(tile);
    copy2D<Scalar>(tile_dims, from, tile_dims.rows(), to, tiled_mat.rows());
}

template<typename Scalar>
void copy_tile_to_device_async(
        Scalar* pinned_buff,
        Scalar* d_buffer,
        tiled_matrix<Scalar>& tiled_mat,
        tile_coord tile,
        runtime_api::StreamType stream) {
    Scalar* from = pinned_buff;
    Scalar* to = d_buffer;
    tile_dim tile_dims = tiled_mat.tile_dimensions(tile);
    copy_to_device_async(from, to, tile_dims.size(), stream);
    std::cout << "copying to device buffer the following: " << std::endl;
    for (int i = 0; i < tile_dims.size(); ++i) {
        std::cout << from[i] << ", ";
    }
    std::cout << std::endl;
}

template<typename Scalar>
void copy_tile_to_host_async(
        Scalar* d_buffer,
        Scalar* pinned_buff,
        tiled_matrix<Scalar>& tiled_mat,
        tile_coord tile,
        runtime_api::StreamType stream) {
    Scalar* from = d_buffer;
    Scalar* to  = pinned_buff;
    tile_dim tile_dims = tiled_mat.tile_dimensions(tile);
    copy_to_host_async(from, to, tile_dims.size(), stream);
    runtime_api::device_synchronize();
    std::cout << "host buffer content: " << std::endl;
    for (int i = 0; i < tile_dims.size(); ++i) {
        std::cout << to[i] << ", ";
    }
    std::cout << std::endl;
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
        throw std::runtime_error("Tile dimension mismatch in device_buffers inside get_tile_sizes.");
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

template<typename Scalar>
void round_robin(
        // host buffers (not pinned)
        tiled_matrix<Scalar>& a_host,
        tiled_matrix<Scalar>& b_host,
        tiled_matrix<Scalar>& c_host,
        // pinned host buffers
        pinned_buffer<Scalar>& a_pinned,
        pinned_buffer<Scalar>& b_pinned,
        pinned_buffer<Scalar>& c_pinned,
        // device buffers
        device_buffer<Scalar>& a_device,
        device_buffer<Scalar>& b_device,
        device_buffer<Scalar>& c_device,
        // gemm parameters
        int m, int n, int k, Scalar alpha, Scalar beta, mm_handle<Scalar>& handle) {

    int n_tiles_m, n_tiles_n, n_tiles_k;
    std::tie(n_tiles_m, n_tiles_n, n_tiles_k) = get_num_tiles(a_host, b_host, c_host);

    int n_streams = std::min(handle.get_num_streams(), n_tiles_n * n_tiles_m);
    auto& gpu_ctx = handle.get_gpu_context();

    auto& result_stream = gpu_ctx.get_result_stream();
    std::vector<device_event> c_computed_on_device(n_streams);
    std::vector<device_event> c_copied_to_device(n_streams);

    // copy the first tile to pinned buffers immediately
    // copy tile {0, 0} from host->pinned buffers(0th stream, 0th tile within stream)
    for (int i = 0; i < n_streams; ++i) {
        int m_tile_id = i / n_tiles_n;
        int n_tile_id = i % n_tiles_n;

        std::cout << "initial copy A" << std::endl;
        copy_tile_to_pinned_buffer(a_host, a_pinned.buffer(i), {m_tile_id, 0});
        std::cout << "initial copy B" << std::endl;
        copy_tile_to_pinned_buffer(b_host, b_pinned.buffer(i), {0, n_tile_id});
        if (std::abs(beta) > 0) {
            std::cout << "initial copy C" << std::endl;
            copy_tile_to_pinned_buffer(c_host, c_pinned.buffer(i), {m_tile_id, n_tile_id});
        }
    }

    // start tiling
    for (int i = 0; i < n_tiles_m * n_tiles_n; i += n_streams) {
        for (int k_tile_id = 0; k_tile_id < n_tiles_k; ++k_tile_id) {
            int n_rounds = 3;
            // if this is the last tile, then there is no 
            // copying host->pinned, thus we have only 2 rounds
            if (i == n_tiles_m * n_tiles_n - 1 && k_tile_id == n_tiles_k - 1) {
                n_rounds = 2;
            }
            for (int round = 0; round < n_rounds; ++round) {
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
                        copy_tile_to_device_async(
                                a_pinned.buffer(stream_id),
                                a_device.stream_buffer(stream_id),
                                a_host,
                                {m_tile_id, k_tile_id},
                                current_stream.stream());
                        auto a_copied_to_device = current_stream.enqueue_event();
                        a_pinned.record_event(stream_id, std::move(a_copied_to_device));
                        // A tile in pinned memory is now completely finished
                        // so we can move to the next pinned buffer
                        a_pinned.advance_buffer(stream_id);

                        // copy B tile
                        copy_tile_to_device_async(
                                b_pinned.buffer(stream_id),
                                b_device.stream_buffer(stream_id),
                                b_host,
                                {k_tile_id, n_tile_id},
                                current_stream.stream());
                        auto b_copied_to_device = current_stream.enqueue_event();
                        b_pinned.record_event(stream_id, std::move(b_copied_to_device));
                        // B tile in pinned memory is now completely finished
                        // so we can move to the next pinned buffer
                        b_pinned.advance_buffer(stream_id);

                        // copy C tile if this is the first partial result and beta > 0
                        // for C, we still cannot move the pinned buffer, since
                        // we want to copy the result back into this pinned memory
                        // once the computation is finished
                        if (k_tile_id == 0 && std::abs(beta) > 0) {
                            current_stream.wait_on_event(c_copied_to_device[stream_id]);
                            copy_tile_to_device_async(
                                    c_pinned.buffer(stream_id),
                                    c_device.stream_buffer(stream_id),
                                    c_host,
                                    {m_tile_id, n_tile_id},
                                    current_stream.stream());
                        }
                    } else if (round == 1) {
                        // perform dgemm
                        // cublasSetStream(get_blas_handle(stream_id), streams[stream_id].stream());
                        // std::cout << "performing dgemm" << std::endl;
                        auto& gemm_stream = gpu_ctx.get_device_stream(stream_id);
                        gemm_stream.wait_on_event(c_copied_to_device[stream_id]);
                        auto status = cublas_gemm_wrapper(
                                gpu_ctx.get_blas_handle(stream_id),
                                actual_size_m, actual_size_n, actual_size_k,
                                &alpha,
                                a_device.stream_buffer(stream_id),
                                b_device.stream_buffer(stream_id),
                                &new_beta,
                                c_device.stream_buffer(stream_id));
                        check_blas_status(status);

                        c_computed_on_device[stream_id] = gemm_stream.enqueue_event();

                        if (k_tile_id == n_tiles_k - 1) {
                            // copy result back to host
                            result_stream.wait_on_event(c_computed_on_device[stream_id]);
                            // c_pinned.wait_buffer(stream_id);
                            copy_tile_to_host_async(
                                    c_device.stream_buffer(stream_id),
                                    c_pinned.buffer(stream_id),
                                    c_host,
                                    {m_tile_id, n_tile_id},
                                    result_stream.stream());
                            c_copied_to_device[stream_id] = gemm_stream.enqueue_event();
                            auto event = gemm_stream.enqueue_event();
                            c_pinned.record_event(stream_id, std::move(event));
                            c_pinned.advance_buffer(stream_id);
                        }
                    } else {
                        // copy the next tile to pinned buffers
                        // unless this is the last tile
                        int next_m_tile_id = m_tile_id;
                        int next_n_tile_id = n_tile_id;
                        int next_k_tile_id = k_tile_id;

                        if (k_tile_id + 1 < n_tiles_k) {
                            next_k_tile_id = k_tile_id + 1;
                        } else {
                            next_k_tile_id = 0;
                            next_m_tile_id = (current_i + n_streams) / n_tiles_n;
                            next_n_tile_id = (current_i + n_streams) % n_tiles_n;
                        }

                        Scalar new_beta = next_k_tile_id == 0 ? beta : 1.0;

                        a_pinned.wait_buffer(stream_id);
                        copy_tile_to_pinned_buffer(a_host, 
                                                   a_pinned.buffer(stream_id), 
                                                   {next_m_tile_id, next_k_tile_id});
                        b_pinned.wait_buffer(stream_id);
                        copy_tile_to_pinned_buffer(b_host, 
                                                   b_pinned.buffer(stream_id), 
                                                   {next_k_tile_id, next_n_tile_id});
                        if (std::abs(new_beta) > 0) {
                            c_pinned.wait_buffer(stream_id);
                            copy_tile_from_pinned_buffer(c_pinned.buffer(stream_id), c_host, {m_tile_id, n_tile_id});
                            copy_tile_to_pinned_buffer(c_host, 
                                                       c_pinned.buffer(stream_id), 
                                                       {next_m_tile_id, next_n_tile_id});
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

    device_buffer<Scalar>& a_device = handle.get_device_buffer_a();
    device_buffer<Scalar>& b_device = handle.get_device_buffer_b();
    device_buffer<Scalar>& c_device = handle.get_device_buffer_c();

    pinned_buffer<Scalar>& a_pinned = handle.get_pinned_buffer_a();
    pinned_buffer<Scalar>& b_pinned = handle.get_pinned_buffer_b();
    pinned_buffer<Scalar>& c_pinned = handle.get_pinned_buffer_c();

    round_robin(a_host, b_host, c_host,
                a_pinned, b_pinned, c_pinned,
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
