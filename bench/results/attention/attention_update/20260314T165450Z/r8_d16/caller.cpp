#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out_ptr, uint8_t *lse0_ptr, uint8_t *lse1_ptr, uint8_t *local_out0_ptr, uint8_t *local_out1_ptr)
{
    attention_update<<<blockDim, nullptr, stream>>>((__fp16 *)out_ptr, (float *)lse0_ptr, (float *)lse1_ptr, (__fp16 *)local_out0_ptr, (__fp16 *)local_out1_ptr);
}
