#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *q_ptr, uint8_t *k_ptr, uint8_t *v_ptr, uint8_t *x_ptr, uint8_t *cos_ptr, uint8_t *sin_ptr, uint8_t *k_cache_ptr, uint8_t *v_cache_ptr, uint8_t *scale_k_ptr, uint8_t *scale_v_ptr, int32_t rows_i32)
{
    rope_quant_kvcache_full_stage<<<blockDim, nullptr, stream>>>((__fp16 *)q_ptr, (__fp16 *)k_ptr, (__fp16 *)v_ptr, (__fp16 *)x_ptr, (__fp16 *)cos_ptr, (__fp16 *)sin_ptr, (int8_t *)k_cache_ptr, (int8_t *)v_cache_ptr, (float *)scale_k_ptr, (float *)scale_v_ptr, rows_i32);
}
