#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *state_out_ptr, uint8_t *state_in_ptr, uint8_t *proj_ptr, uint8_t *value_ptr, uint8_t *key_ptr, uint8_t *beta_ptr, uint8_t *g_ptr)
{
    recurrent_state_update<<<blockDim, nullptr, stream>>>((bfloat16_t *)state_out_ptr, (bfloat16_t *)state_in_ptr, (float *)proj_ptr, (bfloat16_t *)value_ptr, (bfloat16_t *)key_ptr, (bfloat16_t *)beta_ptr, (float *)g_ptr);
}
