#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *probs_ptr, uint8_t *x_ptr)
{
    moe_gating_top_k_softmax_stage<<<blockDim, nullptr, stream>>>((__fp16 *)probs_ptr, (__fp16 *)x_ptr);
}
