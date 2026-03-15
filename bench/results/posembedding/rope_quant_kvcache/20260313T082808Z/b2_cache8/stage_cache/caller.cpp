#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *k_ptr, uint8_t *v_ptr, uint8_t *k_cache_ptr, uint8_t *v_cache_ptr, int32_t rows_i32)
{
    rope_quant_kvcache_cache_stage<<<blockDim, nullptr, stream>>>((__fp16 *)k_ptr, (__fp16 *)v_ptr, (int8_t *)k_cache_ptr, (int8_t *)v_cache_ptr, rows_i32);
}
