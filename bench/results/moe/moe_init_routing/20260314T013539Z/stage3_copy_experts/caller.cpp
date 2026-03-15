#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out_ptr, uint8_t *src_ptr)
{
    moe_init_routing_copy_indices<<<blockDim, nullptr, stream>>>((int32_t *)out_ptr, (int32_t *)src_ptr);
}
