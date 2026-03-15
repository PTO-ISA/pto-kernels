#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *y_ptr, uint8_t *norm_out_ptr, uint8_t *mm_ptr, uint8_t *residual_ptr, uint8_t *gamma_ptr, uint8_t *inv_n_ptr, uint8_t *eps_ptr)
{
    add_rms_norm_stage<<<blockDim, nullptr, stream>>>((__fp16 *)y_ptr, (__fp16 *)norm_out_ptr, (__fp16 *)mm_ptr, (__fp16 *)residual_ptr, (__fp16 *)gamma_ptr, (__fp16 *)inv_n_ptr, (__fp16 *)eps_ptr);
}
