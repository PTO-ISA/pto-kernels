#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out_ptr, uint8_t *out_tmp_ptr)
{
    recurrent_scale_out<<<blockDim, nullptr, stream>>>((bfloat16_t *)out_ptr, (float *)out_tmp_ptr);
}
