#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *query_ptr, uint8_t *key_ptr, uint8_t *cos_ptr, uint8_t *sin_ptr, int32_t rows_i32, int32_t rotary_mode_i32, uint8_t *interleave_reorder_ptr)
{
    apply_rotary_pos_emb_half_fp16_rows<<<blockDim, nullptr, stream>>>((__fp16 *)query_ptr, (__fp16 *)key_ptr, (__fp16 *)cos_ptr, (__fp16 *)sin_ptr, rows_i32, rotary_mode_i32, (int32_t *)interleave_reorder_ptr);
}
