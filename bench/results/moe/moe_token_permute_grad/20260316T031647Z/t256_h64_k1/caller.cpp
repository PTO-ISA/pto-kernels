#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out_ptr, uint8_t *grad_perm_ptr, uint8_t *sorted_indices_ptr)
{
    moe_token_permute_grad_seed<<<blockDim, nullptr, stream>>>((__fp16 *)out_ptr, (__fp16 *)grad_perm_ptr, (int32_t *)sorted_indices_ptr);
}
