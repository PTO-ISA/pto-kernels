#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out_ptr, uint8_t *sorted_experts_ptr)
{
    moe_compute_expert_tokens_seed<<<blockDim, nullptr, stream>>>((int32_t *)out_ptr, (int32_t *)sorted_experts_ptr);
}
