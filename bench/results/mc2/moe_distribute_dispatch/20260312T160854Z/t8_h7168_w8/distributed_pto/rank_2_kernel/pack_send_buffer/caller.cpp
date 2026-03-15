#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *send_ptr, uint8_t *x_ptr, uint8_t *gather_ptr)
{
    moe_distribute_dispatch_pack<<<blockDim, nullptr, stream>>>((__fp16 *)send_ptr, (__fp16 *)x_ptr, (int32_t *)gather_ptr);
}
