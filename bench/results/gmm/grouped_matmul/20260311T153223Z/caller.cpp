#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out_ptr, uint8_t *a_ptr, uint8_t *b_ptr, int32_t batch_i32)
{
    grouped_matmul_dense_bf16_f32<<<blockDim, nullptr, stream>>>((float *)out_ptr, (__fp16 *)a_ptr, (__fp16 *)b_ptr, batch_i32);
}
