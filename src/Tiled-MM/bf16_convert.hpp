/*
 * BFloat16 Conversion Utilities for GPU
 * 
 * Provides device-side conversion kernels between FP32 and BF16.
 * Supports both CUDA (NVIDIA) and ROCm (AMD) backends.
 */
#pragma once

#include <cstddef>

#if defined(TILED_MM_CUDA)
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#elif defined(TILED_MM_ROCM)
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#else
#error Either TILED_MM_CUDA or TILED_MM_ROCM must be defined!
#endif

namespace gpu {
namespace bf16_convert {

#if defined(TILED_MM_CUDA)
using StreamType = cudaStream_t;
using BF16Type = __nv_bfloat16;
#elif defined(TILED_MM_ROCM)
using StreamType = hipStream_t;
using BF16Type = hip_bfloat16;
#endif

/**
 * @brief Convert FP32 array to BF16 on device
 * 
 * Launches a GPU kernel to convert a device-allocated FP32 array to BF16.
 * Uses hardware intrinsics for efficient conversion.
 * 
 * @param d_input Device pointer to FP32 input array
 * @param d_output Device pointer to BF16 output array
 * @param n Number of elements to convert
 * @param stream CUDA/HIP stream for asynchronous execution
 * 
 * @note Both pointers must be device-allocated (cudaMalloc/hipMalloc)
 * @note Kernel is launched asynchronously; use stream synchronization if needed
 */
void convert_fp32_to_bf16(
    const float* d_input,
    BF16Type* d_output,
    size_t n,
    StreamType stream = 0);

/**
 * @brief Convert BF16 array to FP32 on device
 * 
 * Launches a GPU kernel to convert a device-allocated BF16 array to FP32.
 * Conversion is lossless (BF16 is truncated FP32).
 * 
 * @param d_input Device pointer to BF16 input array
 * @param d_output Device pointer to FP32 output array
 * @param n Number of elements to convert
 * @param stream CUDA/HIP stream for asynchronous execution
 * 
 * @note Both pointers must be device-allocated (cudaMalloc/hipMalloc)
 * @note Kernel is launched asynchronously; use stream synchronization if needed
 */
void convert_bf16_to_fp32(
    const BF16Type* d_input,
    float* d_output,
    size_t n,
    StreamType stream = 0);

} // namespace bf16_convert
} // namespace gpu
