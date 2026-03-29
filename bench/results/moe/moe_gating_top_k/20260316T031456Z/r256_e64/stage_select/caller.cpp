#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *y_ptr, uint8_t *expert_idx_ptr, uint8_t *x_ptr)
{
    moe_gating_top_k_select_stage<<<blockDim, nullptr, stream>>>((__fp16 *)y_ptr, (int32_t *)expert_idx_ptr, (__fp16 *)x_ptr);
}
