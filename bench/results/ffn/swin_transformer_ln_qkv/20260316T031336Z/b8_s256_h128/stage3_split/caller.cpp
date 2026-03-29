#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *q_ptr, uint8_t *k_ptr, uint8_t *v_ptr, uint8_t *packed_ptr, uint8_t *bias_ptr)
{
    _stage<<<blockDim, nullptr, stream>>>((__fp16 *)q_ptr, (__fp16 *)k_ptr, (__fp16 *)v_ptr, (__fp16 *)packed_ptr, (__fp16 *)bias_ptr);
}
