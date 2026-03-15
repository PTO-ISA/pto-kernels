#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_compute_expert_tokens_seed(__gm__ int32_t* v1, __gm__ int32_t* v2) {
  int32_t v3 = 64;
  int32_t v4 = 4096;
  int32_t v5 = 1;
  int32_t v6 = 0;
  using T = float;
  size_t v7 = (size_t) v5;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v8 = get_block_idx();
  int64_t v9 = get_block_num();
  int32_t v10 = (int32_t) ((int64_t) v9);
  int32_t v11 = v3 / v10;
  int32_t v12 = v3 % v10 != v6 && v3 < v6 == v10 < v6 ? v11 + v5 : v11;
  int32_t v13 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v8) * (uint32_t) v12);
  int32_t v14 = (int32_t) ((uint32_t) v13 + (uint32_t) v12);
  for (size_t v15 = (size_t) v13; v15 < ((size_t) ((uint32_t) v14 < (uint32_t) v3 ? v14 : v3)); v15 += v7) {
    v1[v15] = v6;
    for (size_t v16 = (size_t) v6; v16 < ((size_t) v4); v16 += v7) {
      int32_t v17 = v2[v16];
      int32_t v18 = v1[v15];
      int32_t v19 = (int32_t) ((uint32_t) ((int32_t) v15) + (uint32_t) v5) > v17 ? (int32_t) ((uint32_t) ((int32_t) v16) + (uint32_t) v5) : v18;
      v1[v15] = v19;
    };
  }
  pipe_barrier(PIPE_ALL);
  #endif // __DAV_VEC__

  return;
}

