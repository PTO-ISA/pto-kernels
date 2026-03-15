#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *query_out_ptr, uint8_t *key_out_ptr, uint8_t *query_ptr, uint8_t *key_ptr, uint8_t *cache_ptr)
{
    rope_with_sin_cos_cache<<<blockDim, nullptr, stream>>>((__fp16 *)query_out_ptr, (__fp16 *)key_out_ptr, (__fp16 *)query_ptr, (__fp16 *)key_ptr, (__fp16 *)cache_ptr);
}
