#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out_ptr, uint8_t *indices_ptr)
{
    moe_token_permute_with_routing_map_copy_indices<<<blockDim, nullptr, stream>>>((int32_t *)out_ptr, (int32_t *)indices_ptr);
}
