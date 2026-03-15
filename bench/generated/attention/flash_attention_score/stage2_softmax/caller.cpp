#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *scores_ptr)
{
    dense_attention_row_softmax<<<blockDim, nullptr, stream>>>((__fp16 *)scores_ptr);
}
