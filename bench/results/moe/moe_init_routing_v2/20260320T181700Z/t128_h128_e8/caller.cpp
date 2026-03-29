#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *expanded_x_out_ptr, uint8_t *expanded_row_idx_out_ptr, uint8_t *expert_tokens_count_or_cumsum_out_ptr, uint8_t *expert_tokens_before_capacity_out_ptr, uint8_t *x_ptr, uint8_t *expert_idx_ptr)
{
    moe_init_routing_v2_seed<<<blockDim, nullptr, stream>>>((__fp16 *)expanded_x_out_ptr, (int32_t *)expanded_row_idx_out_ptr, (int32_t *)expert_tokens_count_or_cumsum_out_ptr, (int32_t *)expert_tokens_before_capacity_out_ptr, (__fp16 *)x_ptr, (int32_t *)expert_idx_ptr);
}
