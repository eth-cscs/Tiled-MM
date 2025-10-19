/*
 * BFloat16 Conversion Kernels for CUDA
 * 
 * Implements efficient FP32 ↔ BF16 conversion using CUDA intrinsics.
 */
#include "bf16_convert.hpp"

#if defined(TILED_MM_CUDA)

namespace gpu {
namespace bf16_convert {

// Kernel configuration
constexpr int THREADS_PER_BLOCK = 256;

/**
 * @brief CUDA kernel to convert FP32 → BF16
 * 
 * Uses __float2bfloat16 intrinsic for hardware-accelerated conversion.
 * Applies round-to-nearest-even (RNE) rounding.
 */
__global__ void fp32_to_bf16_kernel(
    const float* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    size_t n) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // CUDA intrinsic: converts FP32 to BF16 with RNE rounding
        // Available on all GPUs (software fallback on pre-Ampere)
        output[idx] = __float2bfloat16(input[idx]);
    }
}

/**
 * @brief CUDA kernel to convert BF16 → FP32
 * 
 * Uses __bfloat162float intrinsic for hardware-accelerated conversion.
 * Conversion is lossless (zero-extends mantissa).
 */
__global__ void bf16_to_fp32_kernel(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ output,
    size_t n) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // CUDA intrinsic: converts BF16 to FP32 (lossless)
        output[idx] = __bfloat162float(input[idx]);
    }
}

// Host-side wrapper functions

void convert_fp32_to_bf16(
    const float* d_input,
    BF16Type* d_output,
    size_t n,
    StreamType stream) {
    
    if (n == 0) return;
    
    // Calculate grid dimensions
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Launch kernel
    fp32_to_bf16_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        d_input,
        reinterpret_cast<__nv_bfloat16*>(d_output),
        n);
    
    // Note: Kernel is asynchronous. Caller should sync stream if needed.
}

void convert_bf16_to_fp32(
    const BF16Type* d_input,
    float* d_output,
    size_t n,
    StreamType stream) {
    
    if (n == 0) return;
    
    // Calculate grid dimensions
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Launch kernel
    bf16_to_fp32_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(d_input),
        d_output,
        n);
    
    // Note: Kernel is asynchronous. Caller should sync stream if needed.
}

} // namespace bf16_convert
} // namespace gpu

#endif // TILED_MM_CUDA
