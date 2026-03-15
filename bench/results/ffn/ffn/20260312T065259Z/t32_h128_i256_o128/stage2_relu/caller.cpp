#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *hidden_ptr)
{
    dense_relu_stage<<<blockDim, nullptr, stream>>>((__fp16 *)hidden_ptr);
}
