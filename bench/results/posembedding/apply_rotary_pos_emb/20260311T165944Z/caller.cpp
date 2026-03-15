#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *query_ptr, uint8_t *key_ptr, uint8_t *cos_ptr, uint8_t *sin_ptr, int32_t tokens_i32)
{
    apply_rotary_pos_emb_tnd_half_fp16<<<blockDim, nullptr, stream>>>((__fp16 *)query_ptr, (__fp16 *)key_ptr, (__fp16 *)cos_ptr, (__fp16 *)sin_ptr, tokens_i32);
}
