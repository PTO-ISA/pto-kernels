#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out_ptr, uint8_t *mm_ptr, uint8_t *y_ptr)
{
    grouped_matmul_add_add_stage<<<blockDim, nullptr, stream>>>((float *)out_ptr, (float *)mm_ptr, (float *)y_ptr);
}
