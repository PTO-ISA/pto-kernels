#include "kernel.cpp"
#include <cstdint>

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *scores_ptr)
{
    flash_attention_score_stage2<<<blockDim, nullptr, stream>>>((__fp16 *)scores_ptr);
}
