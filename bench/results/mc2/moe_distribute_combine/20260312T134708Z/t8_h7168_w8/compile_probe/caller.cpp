#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *x_out_ptr, uint8_t *compact_expand_x_ptr)
{
    moe_distribute_combine_seed<<<blockDim, nullptr, stream>>>((__fp16 *)x_out_ptr, (__fp16 *)compact_expand_x_ptr);
}
