#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *attn_out_ptr, uint8_t *softmax_max_out_ptr, uint8_t *softmax_sum_out_ptr, uint8_t *prev_attn_out_ptr, uint8_t *prev_softmax_max_ptr, uint8_t *prev_softmax_sum_ptr, uint8_t *cur_attn_out_ptr, uint8_t *cur_softmax_max_ptr, uint8_t *cur_softmax_sum_ptr)
{
    ring_attention_update<<<blockDim, nullptr, stream>>>((__fp16 *)attn_out_ptr, (float *)softmax_max_out_ptr, (float *)softmax_sum_out_ptr, (__fp16 *)prev_attn_out_ptr, (float *)prev_softmax_max_ptr, (float *)prev_softmax_sum_ptr, (__fp16 *)cur_attn_out_ptr, (float *)cur_softmax_max_ptr, (float *)cur_softmax_sum_ptr);
}
