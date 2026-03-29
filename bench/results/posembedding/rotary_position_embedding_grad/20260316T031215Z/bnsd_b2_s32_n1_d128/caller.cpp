#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *dx_ptr, uint8_t *dcos_ptr, uint8_t *dsin_ptr, uint8_t *dy_ptr, uint8_t *x_ptr, uint8_t *cos_ptr, uint8_t *sin_ptr, int32_t rows_i32)
{
    rotary_position_embedding_grad<<<blockDim, nullptr, stream>>>((__fp16 *)dx_ptr, (__fp16 *)dcos_ptr, (__fp16 *)dsin_ptr, (__fp16 *)dy_ptr, (__fp16 *)x_ptr, (__fp16 *)cos_ptr, (__fp16 *)sin_ptr, rows_i32);
}
