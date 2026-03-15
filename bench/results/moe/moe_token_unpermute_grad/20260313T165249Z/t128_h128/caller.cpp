#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *permuted_grad_ptr, uint8_t *probs_grad_ptr, uint8_t *unpermuted_grad_ptr, uint8_t *sorted_indices_ptr)
{
    moe_token_unpermute_grad_seed<<<blockDim, nullptr, stream>>>((__fp16 *)permuted_grad_ptr, (__fp16 *)probs_grad_ptr, (__fp16 *)unpermuted_grad_ptr, (int32_t *)sorted_indices_ptr);
}
