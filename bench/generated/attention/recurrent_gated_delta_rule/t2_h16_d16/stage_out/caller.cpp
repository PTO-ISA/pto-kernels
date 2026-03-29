#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *out_tmp_ptr, uint8_t *state_ptr, uint8_t *query_ptr)
{
    recurrent_state_query_proj<<<blockDim, nullptr, stream>>>((float *)out_tmp_ptr, (bfloat16_t *)state_ptr, (bfloat16_t *)query_ptr);
}
