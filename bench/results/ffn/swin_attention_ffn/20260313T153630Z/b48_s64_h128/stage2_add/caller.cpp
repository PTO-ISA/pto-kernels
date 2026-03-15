#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out_ptr, uint8_t *tmp_ptr, uint8_t *bias_ptr, uint8_t *x3_ptr)
{
    _stage<<<blockDim, nullptr, stream>>>((__fp16 *)out_ptr, (__fp16 *)tmp_ptr, (__fp16 *)bias_ptr, (__fp16 *)x3_ptr);
}
