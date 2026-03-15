#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *hidden_ptr, uint8_t *x_ptr, uint8_t *w1_ptr)
{
    ffn_stage1<<<blockDim, nullptr, stream>>>((__fp16 *)hidden_ptr, (__fp16 *)x_ptr, (__fp16 *)w1_ptr);
}
