#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *scores_ptr, uint8_t *query_ptr, uint8_t *key_t_ptr)
{
    dense_attention_qk_stage<<<blockDim, nullptr, stream>>>((__fp16 *)scores_ptr, (__fp16 *)query_ptr, (__fp16 *)key_t_ptr);
}
