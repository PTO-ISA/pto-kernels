#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *q_ptr, uint8_t *k_ptr, uint8_t *v_ptr, uint8_t *qkv_ptr, uint8_t *q_gamma_ptr, uint8_t *k_gamma_ptr, uint8_t *cos_ptr, uint8_t *sin_ptr, int32_t rows_i32)
{
    qkv_rms_norm_rope_cache_rotary_stage<<<blockDim, nullptr, stream>>>((__fp16 *)q_ptr, (__fp16 *)k_ptr, (__fp16 *)v_ptr, (__fp16 *)qkv_ptr, (__fp16 *)q_gamma_ptr, (__fp16 *)k_gamma_ptr, (__fp16 *)cos_ptr, (__fp16 *)sin_ptr, rows_i32);
}
