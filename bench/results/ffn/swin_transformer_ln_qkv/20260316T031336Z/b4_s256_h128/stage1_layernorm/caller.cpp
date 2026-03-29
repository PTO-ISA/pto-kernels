#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out_ptr, uint8_t *x_ptr, uint8_t *gamma_ptr, uint8_t *beta_ptr)
{
    _stage<<<blockDim, nullptr, stream>>>((__fp16 *)out_ptr, (__fp16 *)x_ptr, (__fp16 *)gamma_ptr, (__fp16 *)beta_ptr);
}
