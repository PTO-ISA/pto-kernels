#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_token_permute_grad_seed(__gm__ half* v1, __gm__ half* v2, __gm__ int32_t* v3) {
  unsigned v4 = 16;
  unsigned v5 = 1;
  unsigned v6 = 0;
  int32_t v7 = 0;
  int32_t v8 = 16;
  int32_t v9 = 8;
  int32_t v10 = 1;
  int64_t v11 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v12 = get_block_idx();
  int64_t v13 = get_block_num();
  int32_t v14 = (int32_t) ((int64_t) v13);
  int32_t v15 = v9 / v14;
  int32_t v16 = v9 % v14 != v7 && v9 < v7 == v14 < v7 ? v15 + v10 : v15;
  int32_t v17 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v12) * (uint32_t) v16);
  int32_t v18 = (int32_t) ((uint32_t) v17 + (uint32_t) v16);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v19;
  TASSIGN(v19, v11);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  for (size_t v20 = (size_t) v17; v20 < ((size_t) ((uint32_t) v18 < (uint32_t) v9 ? v18 : v9)); v20 += (size_t) v10) {
    int32_t v21 = v3[v20];
    pto::Shape<1, 1, 1, 1, 16> v22 = pto::Shape<1, 1, 1, 1, 16>();
    pto::Stride<16, 16, 16, 16, 1> v23 = pto::Stride<16, 16, 16, 16, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v24 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v21 * (unsigned) v8 + v6 * (unsigned) v10), v22, v23);
    pto::Shape<1, 1, 1, 1, 16> v25 = pto::Shape<1, 1, 1, 1, 16>();
    pto::Stride<16, 16, 16, 16, 1> v26 = pto::Stride<16, 16, 16, 16, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v27 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) ((int32_t) v20) * (unsigned) v8 + v6 * (unsigned) v10), v25, v26);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v19, v24);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v27, v19);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  #endif // __DAV_VEC__

  return;
}

