#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *proj_ptr, uint8_t *state_ptr, uint8_t *key_ptr)
{
    recurrent_state_key_proj<<<blockDim, nullptr, stream>>>((float *)proj_ptr, (bfloat16_t *)state_ptr, (bfloat16_t *)key_ptr);
}
