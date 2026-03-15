#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_distribute_combine_seed(__gm__ half* v1, __gm__ half* v2) {
  unsigned v3 = 7168;
  unsigned v4 = 1;
  unsigned v5 = 0;
  int32_t v6 = 0;
  int32_t v7 = 7168;
  int32_t v8 = 8;
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
  int32_t v15 = v8 % v13 != v6 && v8 < v6 == v13 < v6 ? v14 + v9 : v14;
  int32_t v16 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v11) * (uint32_t) v15);
  int32_t v17 = (int32_t) ((uint32_t) v16 + (uint32_t) v15);
  Tile<TileType::Vec, half, 1, 7168, BLayout::RowMajor, 1, 7168, SLayout::NoneBox, 512, PadValue::Null> v18;
  TASSIGN(v18, v10);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  for (size_t v19 = (size_t) v16; v19 < ((size_t) ((uint32_t) v17 < (uint32_t) v8 ? v17 : v8)); v19 += (size_t) v9) {
    int32_t v20 = (int32_t) v19;
    pto::Shape<1, 1, 1, 1, 7168> v21 = pto::Shape<1, 1, 1, 1, 7168>();
    pto::Stride<7168, 7168, 7168, 7168, 1> v22 = pto::Stride<7168, 7168, 7168, 7168, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 7168>, pto::Stride<7168, 7168, 7168, 7168, 1>, pto::Layout::ND> v23 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 7168>, pto::Stride<7168, 7168, 7168, 7168, 1>, pto::Layout::ND>(v2 + (v5 + (unsigned) v20 * (unsigned) v7 + v5 * (unsigned) v9), v21, v22);
    pto::Shape<1, 1, 1, 1, 7168> v24 = pto::Shape<1, 1, 1, 1, 7168>();
    pto::Stride<7168, 7168, 7168, 7168, 1> v25 = pto::Stride<7168, 7168, 7168, 7168, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 7168>, pto::Stride<7168, 7168, 7168, 7168, 1>, pto::Layout::ND> v26 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 7168>, pto::Stride<7168, 7168, 7168, 7168, 1>, pto::Layout::ND>(v1 + (v5 + (unsigned) v20 * (unsigned) v7 + v5 * (unsigned) v9), v24, v25);
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

