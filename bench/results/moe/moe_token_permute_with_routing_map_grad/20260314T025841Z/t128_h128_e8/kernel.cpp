#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_token_permute_with_routing_map_grad_seed(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3, __gm__ int32_t* v4) {
  unsigned v5 = 128;
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 128;
  int32_t v9 = 1;
  int32_t v10 = 0;
  half v11 = 0.0f;
  int64_t v12 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v13 = get_block_idx();
  int32_t v14 = (int32_t) ((int64_t) v13);
  int64_t v15 = get_block_num();
  int32_t v16 = (int32_t) ((int64_t) v15);
  int32_t v17 = v8 / v16;
  int32_t v18 = v8 % v16 != v10 && v8 < v10 == v16 < v10 ? v17 + v9 : v17;
  int32_t v19 = (int32_t) ((uint32_t) v14 * (uint32_t) v18);
  int32_t v20 = (int32_t) ((uint32_t) v19 + (uint32_t) v18);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v21;
  TASSIGN(v21, v12);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  if (v14 == v10) {
    v2[v10] = v11;
  }
  for (size_t v22 = (size_t) v19; v22 < ((size_t) ((uint32_t) v20 < (uint32_t) v8 ? v20 : v8)); v22 += (size_t) v9) {
    int32_t v23 = v4[v22];
    pto::Shape<1, 1, 1, 1, 128> v24 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v25 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v26 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v3 + (v7 + (unsigned) ((int32_t) v22) * (unsigned) v8 + v7 * (unsigned) v9), v24, v25);
    pto::Shape<1, 1, 1, 1, 128> v27 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v28 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v29 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v23 * (unsigned) v8 + v7 * (unsigned) v9), v27, v28);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v21, v26);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v29, v21);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  #endif // __DAV_VEC__

  return;
}

