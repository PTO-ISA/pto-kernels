#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *grad_expanded_x_out_ptr, uint8_t *grad_scales_out_ptr, uint8_t *grad_y_ptr, uint8_t *expanded_row_idx_ptr, uint8_t *expanded_x_ptr, uint8_t *scales_ptr, uint8_t *expert_idx_ptr, uint8_t *bias_ptr)
{
    moe_finalize_routing_v2_grad_seed<<<blockDim, nullptr, stream>>>((__fp16 *)grad_expanded_x_out_ptr, (__fp16 *)grad_scales_out_ptr, (__fp16 *)grad_y_ptr, (int32_t *)expanded_row_idx_ptr, (__fp16 *)expanded_x_ptr, (__fp16 *)scales_ptr, (int32_t *)expert_idx_ptr, (__fp16 *)bias_ptr);
}
