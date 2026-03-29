#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out_ptr, uint8_t *gathered_x1_ptr, uint8_t *x2_ptr)
{
    all_gather_matmul_dense<<<blockDim, nullptr, stream>>>((__fp16 *)out_ptr, (__fp16 *)gathered_x1_ptr, (__fp16 *)x2_ptr);
}
