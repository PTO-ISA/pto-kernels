#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *dst_ptr, uint8_t *src_ptr, uint8_t *idx_ptr)
{
    k<<<blockDim, nullptr, stream>>>((__fp16 *)dst_ptr, (__fp16 *)src_ptr, (int16_t *)idx_ptr);
}
