#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_token_permute_grad_seed(__gm__ half* v1, __gm__ half* v2, __gm__ int32_t* v3) {
  unsigned v4 = 128;
  unsigned v5 = 1;
  unsigned v6 = 0;
  int32_t v7 = 0;
  int32_t v8 = 128;
  int32_t v9 = 1;
  int64_t v10 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v11 = get_block_idx();
  int64_t v12 = get_block_num();
  int32_t v13 = (int32_t) ((int64_t) v12);
  int32_t v14 = v8 / v13;
  int32_t v15 = v8 % v13 != v7 && v8 < v7 == v13 < v7 ? v14 + v9 : v14;
  int32_t v16 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v11) * (uint32_t) v15);
  int32_t v17 = (int32_t) ((uint32_t) v16 + (uint32_t) v15);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v18;
  TASSIGN(v18, v10);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  for (size_t v19 = (size_t) v16; v19 < ((size_t) ((uint32_t) v17 < (uint32_t) v8 ? v17 : v8)); v19 += (size_t) v9) {
    int32_t v20 = v3[v19];
    pto::Shape<1, 1, 1, 1, 128> v21 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v22 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v23 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v20 * (unsigned) v8 + v6 * (unsigned) v9), v21, v22);
    pto::Shape<1, 1, 1, 1, 128> v24 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v25 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v26 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) ((int32_t) v19) * (unsigned) v8 + v6 * (unsigned) v9), v24, v25);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v18, v23);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v26, v18);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  #endif // __DAV_VEC__

  return;
}

