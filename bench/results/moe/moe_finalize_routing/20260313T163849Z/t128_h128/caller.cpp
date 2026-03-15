#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out_ptr, uint8_t *expanded_ptr, uint8_t *x1_ptr, uint8_t *bias_ptr, uint8_t *scales_ptr, uint8_t *expanded_row_idx_ptr, uint8_t *expanded_expert_idx_ptr)
{
    moe_finalize_routing_seed<<<blockDim, nullptr, stream>>>((__fp16 *)out_ptr, (__fp16 *)expanded_ptr, (__fp16 *)x1_ptr, (__fp16 *)bias_ptr, (__fp16 *)scales_ptr, (int32_t *)expanded_row_idx_ptr, (int32_t *)expanded_expert_idx_ptr);
}
