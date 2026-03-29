#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out_ptr, uint8_t *x_ptr, uint8_t *weight_ptr)
{
    grouped_mat_mul_all_reduce_local<<<blockDim, nullptr, stream>>>((__fp16 *)out_ptr, (__fp16 *)x_ptr, (__fp16 *)weight_ptr);
}
