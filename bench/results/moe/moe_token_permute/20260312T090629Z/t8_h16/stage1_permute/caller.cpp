#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out_ptr, uint8_t *tokens_ptr, uint8_t *gather_ptr)
{
    moe_token_permute_seed<<<blockDim, nullptr, stream>>>((__fp16 *)out_ptr, (__fp16 *)tokens_ptr, (int32_t *)gather_ptr);
}
