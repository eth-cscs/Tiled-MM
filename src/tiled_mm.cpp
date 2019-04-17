#include "tiled_mm.hpp"
namespace gpu {

void copy_tile_to_device_async(tiled_matrix& tiled_mat, device_buffer<double>& d_buffer,
        tile_coord tile, gpu_context& ctx, int stream_id) {
    double* from = tiled_mat.tile_data(tile);
    double* to = d_buffer.stream_buffer(stream_id);

    tile_dim tile_dims = tiled_mat.tile_dimensions(tile);
    // std::cout << "host->device" << std::endl;

    auto status=
    cudaMemcpy2DAsync(to, tile_dims.rows() * sizeof(double),
            from, tiled_mat.rows() * sizeof(double),
            tile_dims.rows() * sizeof(double), tile_dims.cols(),
            cudaMemcpyHostToDevice, ctx.get_cuda_stream(stream_id));
    cuda_check_status(status);
}

void copy_tile_to_host_async(tiled_matrix& tiled_mat, device_buffer<double>& d_buffer,
        tile_coord tile, gpu_context& ctx, int stream_id) {
    double* from  = d_buffer.stream_buffer(stream_id);
    double* to = tiled_mat.tile_data(tile);

    tile_dim tile_dims = tiled_mat.tile_dimensions(tile);

    // std::cout << "device->host" << std::endl;
    auto status=
    cudaMemcpy2DAsync(to, tiled_mat.rows() * sizeof(double),
            from, tile_dims.rows() * sizeof(double),
            tile_dims.rows() * sizeof(double), tile_dims.cols(),
            cudaMemcpyDeviceToHost, ctx.get_cuda_stream(stream_id));
    cuda_check_status(status);
}

std::tuple<int, int, int> get_num_tiles(tiled_matrix& a, tiled_matrix& b, tiled_matrix& c) {
    if (a.num_tiles_row() != c.num_tiles_row() || 
            a.num_tiles_col() != b.num_tiles_row() || 
            b.num_tiles_col() != c.num_tiles_col()) {
        throw std::runtime_error("Number of tiles mismatch in tiled_matrix inside get_num_tiles.");
    }
    return {a.num_tiles_row(), c.num_tiles_col(), b.num_tiles_row()};
}


std::tuple<int, int, int> get_tile_sizes(tiled_matrix& a, 
        tiled_matrix& b, tiled_matrix& c, 
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

void round_robin(tiled_matrix& a_host, tiled_matrix& b_host, tiled_matrix& c_host,
        device_buffer<double>& a_device, 
        device_buffer<double>& b_device, 
        device_buffer<double>& c_device,
        int m, int n, int k, double alpha, double beta, mm_handle& handle) {

    int n_tiles_m, n_tiles_n, n_tiles_k;
    std::tie(n_tiles_m, n_tiles_n, n_tiles_k) = get_num_tiles(a_host, b_host, c_host);

    int n_streams = handle.get_num_streams();
    auto& gpu_ctx = handle.get_gpu_context();

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

                    double new_beta = k_tile_id == 0 ? beta : 1.0;

                    if (round == 0) {
                        // copy A tile
                        copy_tile_to_device_async(a_host, a_device,
                                {m_tile_id, k_tile_id},
                                gpu_ctx, stream_id);

                        // copy B tile
                        copy_tile_to_device_async(b_host, b_device,
                                {k_tile_id, n_tile_id},
                                gpu_ctx, stream_id);

                        // copy C tile if this is the first partial result and beta > 0
                        if (k_tile_id == 0 && beta > 0) {
                            copy_tile_to_device_async(c_host, c_device,
                                    {m_tile_id, n_tile_id},
                                    gpu_ctx, stream_id);
                        }
                    } else {
                        // perform dgemm
                        // cublasSetStream(get_cublas_handle(stream_id), streams[stream_id].stream());
                        // std::cout << "performing dgemm" << std::endl;
                        auto status =
                        cublasDgemm(gpu_ctx.get_cublas_handle(stream_id), CUBLAS_OP_N, CUBLAS_OP_N,
                                actual_size_m, actual_size_n, actual_size_k, &alpha,
                                a_device.stream_buffer(stream_id), actual_size_m,
                                b_device.stream_buffer(stream_id), actual_size_k, &new_beta,
                                c_device.stream_buffer(stream_id), actual_size_m);
                        cublas_check_status(status);


                        if (k_tile_id == n_tiles_k - 1) {
                            // copy result back to host
                            copy_tile_to_host_async(c_host, c_device,
                                    {m_tile_id, n_tile_id},
                                    gpu_ctx, stream_id);
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
void dgemm(mm_handle& handle, double* a, double* b, double* c,
        int m, int n, int k,
        double alpha, double beta) {

    int tile_size_m, tile_size_n, tile_size_k;
    std::tie(tile_size_m, tile_size_n, tile_size_k) = handle.get_tile_sizes();

    tiled_matrix a_host(a, m, k, {tile_size_m, tile_size_k});
    tiled_matrix b_host(b, k, n, {tile_size_k, tile_size_n});
    tiled_matrix c_host(c, m, n, {tile_size_m, tile_size_n});

    device_buffer<double>& a_device = handle.get_device_buffer_a();
    device_buffer<double>& b_device = handle.get_device_buffer_b();
    device_buffer<double>& c_device = handle.get_device_buffer_c();

    round_robin(a_host, b_host, c_host, 
                a_device, b_device, c_device,
                m, n, k, alpha, beta, handle);

    auto status =
    cudaDeviceSynchronize();
    cuda_check_status(status);
}

void dgemm(context& ctx, double* a, double* b, double* c,
          int m, int n, int k,
          double alpha, double beta) {
    dgemm(*ctx, a, b, c, m, n, k, alpha, beta);
}
}
