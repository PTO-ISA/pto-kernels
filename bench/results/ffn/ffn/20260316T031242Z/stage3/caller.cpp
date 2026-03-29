#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out_ptr, uint8_t *a_ptr, uint8_t *b_ptr)
{
    _stage<<<blockDim, nullptr, stream>>>((__fp16 *)out_ptr, (__fp16 *)a_ptr, (__fp16 *)b_ptr);
}
