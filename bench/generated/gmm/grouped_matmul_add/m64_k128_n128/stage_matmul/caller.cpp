#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out_ptr, uint8_t *a_ptr, uint8_t *b_ptr, int32_t batch_i32)
{
    grouped_matmul_add_matmul_stage<<<blockDim, nullptr, stream>>>((float *)out_ptr, (bfloat16_t *)a_ptr, (bfloat16_t *)b_ptr, batch_i32);
}
