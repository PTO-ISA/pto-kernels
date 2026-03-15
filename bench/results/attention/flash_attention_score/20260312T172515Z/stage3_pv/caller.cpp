#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out_ptr, uint8_t *scores_ptr, uint8_t *value_ptr)
{
    dense_attention_pv_stage<<<blockDim, nullptr, stream>>>((__fp16 *)out_ptr, (__fp16 *)scores_ptr, (__fp16 *)value_ptr);
}
