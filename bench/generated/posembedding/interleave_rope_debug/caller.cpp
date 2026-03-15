#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out_ptr, uint8_t *x_ptr, uint8_t *cos_ptr, uint8_t *sin_ptr, uint8_t *gather_even_ptr, uint8_t *gather_odd_ptr, int32_t rows_i32)
{
    interleave_rope_fp16_bnsd<<<blockDim, nullptr, stream>>>((__fp16 *)out_ptr, (__fp16 *)x_ptr, (__fp16 *)cos_ptr, (__fp16 *)sin_ptr, (int32_t *)gather_even_ptr, (int32_t *)gather_odd_ptr, rows_i32);
}
