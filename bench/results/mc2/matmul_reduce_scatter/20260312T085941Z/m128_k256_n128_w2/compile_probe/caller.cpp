#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out_ptr, uint8_t *x1_ptr, uint8_t *x2_ptr)
{
    matmul_reduce_scatter_local_mm<<<blockDim, nullptr, stream>>>((__fp16 *)out_ptr, (__fp16 *)x1_ptr, (__fp16 *)x2_ptr);
}
