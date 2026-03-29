#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out_tokens_ptr, uint8_t *out_scales_ptr, uint8_t *out_idx_ptr, uint8_t *out_expert_token_num_ptr, uint8_t *tokens_ptr, uint8_t *counts_ptr, uint8_t *scales_ptr)
{
    moe_re_routing_seed<<<blockDim, nullptr, stream>>>((__fp16 *)out_tokens_ptr, (float *)out_scales_ptr, (int32_t *)out_idx_ptr, (int32_t *)out_expert_token_num_ptr, (__fp16 *)tokens_ptr, (int32_t *)counts_ptr, (float *)scales_ptr);
}
