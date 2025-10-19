/*
 * BFloat16 Conversion Kernels for ROCm/HIP
 * 
 * Implements efficient FP32 ↔ BF16 conversion using HIP intrinsics.
 */
#include "bf16_convert.hpp"

#if defined(TILED_MM_ROCM)

namespace gpu {
namespace bf16_convert {

// Kernel configuration
constexpr int THREADS_PER_BLOCK = 256;

/**
 * @brief HIP kernel to convert FP32 → BF16
 * 
 * Uses float_to_bfloat16 intrinsic for hardware-accelerated conversion.
 * Applies round-to-nearest-even (RNE) rounding.
 */
__global__ void fp32_to_bf16_kernel(
    const float* __restrict__ input,
    hip_bfloat16* __restrict__ output,
    size_t n) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // HIP intrinsic: converts FP32 to BF16 with RNE rounding
        // Hardware-accelerated on CDNA2+ (MI200 series)
        output[idx] = float_to_bfloat16(input[idx]);
    }
}

/**
 * @brief HIP kernel to convert BF16 → FP32
 * 
 * Uses bfloat16_to_float intrinsic for hardware-accelerated conversion.
 * Conversion is lossless (zero-extends mantissa).
 */
__global__ void bf16_to_fp32_kernel(
    const hip_bfloat16* __restrict__ input,
    float* __restrict__ output,
    size_t n) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // HIP intrinsic: converts BF16 to FP32 (lossless)
        output[idx] = bfloat16_to_float(input[idx]);
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
    hipLaunchKernelGGL(
        fp32_to_bf16_kernel,
        dim3(blocks),
        dim3(THREADS_PER_BLOCK),
        0,
        stream,
        d_input,
        reinterpret_cast<hip_bfloat16*>(d_output),
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
    hipLaunchKernelGGL(
        bf16_to_fp32_kernel,
        dim3(blocks),
        dim3(THREADS_PER_BLOCK),
        0,
        stream,
        reinterpret_cast<const hip_bfloat16*>(d_input),
        d_output,
        n);
    
    // Note: Kernel is asynchronous. Caller should sync stream if needed.
}

} // namespace bf16_convert
} // namespace gpu

#endif // TILED_MM_ROCM
