#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_distribute_combine_seed(__gm__ half* v1, __gm__ half* v2, __gm__ int16_t* v3) {
  unsigned v4 = 7168;
  unsigned v5 = 14336;
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 57344;
  int32_t v9 = 14336;
  int32_t v10 = 2;
  int32_t v11 = 7168;
  int32_t v12 = 1;
  int64_t v13 = 28672;
  int64_t v14 = 43008;
  int64_t v15 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v16 = get_block_idx();
  int32_t v17 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v16) * (uint32_t) v10);
  Tile<TileType::Vec, half, 1, 7168, BLayout::RowMajor, 1, 7168, SLayout::NoneBox, 512, PadValue::Null> v18;
  TASSIGN(v18, v13);
  Tile<TileType::Vec, int16_t, 1, 7168, BLayout::RowMajor, 1, 7168, SLayout::NoneBox, 512, PadValue::Null> v19;
  TASSIGN(v19, v14);
  Tile<TileType::Vec, half, 1, 14336, BLayout::RowMajor, 1, 14336, SLayout::NoneBox, 512, PadValue::Null> v20;
  TASSIGN(v20, v15);
  pto::Shape<1, 1, 1, 1, 14336> v21 = pto::Shape<1, 1, 1, 1, 14336>();
  pto::Stride<14336, 14336, 14336, 14336, 1> v22 = pto::Stride<14336, 14336, 14336, 14336, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 1, 14336>, pto::Stride<14336, 14336, 14336, 14336, 1>, pto::Layout::ND> v23 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 14336>, pto::Stride<14336, 14336, 14336, 14336, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) ((int32_t) (uint32_t) v17 * (uint32_t) v11) * (unsigned) v12), v21, v22);
  set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
  TLOAD(v20, v23);
  for (size_t v24 = (size_t) v17; v24 < ((size_t) ((int32_t) (uint32_t) v17 + (uint32_t) v10)); v24 += (size_t) v12) {
    int32_t v25 = (int32_t) ((uint32_t) ((int32_t) v24) * (uint32_t) v11);
    pto::Shape<1, 1, 1, 1, 7168> v26 = pto::Shape<1, 1, 1, 1, 7168>();
    pto::Stride<7168, 7168, 7168, 7168, 1> v27 = pto::Stride<7168, 7168, 7168, 7168, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 7168>, pto::Stride<7168, 7168, 7168, 7168, 1>, pto::Layout::ND> v28 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 7168>, pto::Stride<7168, 7168, 7168, 7168, 1>, pto::Layout::ND>(v2 + (v7 + (unsigned) v25 * (unsigned) v12), v26, v27);
    pto::Shape<1, 1, 1, 1, 7168> v29 = pto::Shape<1, 1, 1, 1, 7168>();
    pto::Stride<7168, 7168, 7168, 7168, 1> v30 = pto::Stride<7168, 7168, 7168, 7168, 1>();
    GlobalTensor<int16_t, pto::Shape<1, 1, 1, 1, 7168>, pto::Stride<7168, 7168, 7168, 7168, 1>, pto::Layout::ND> v31 = GlobalTensor<int16_t, pto::Shape<1, 1, 1, 1, 7168>, pto::Stride<7168, 7168, 7168, 7168, 1>, pto::Layout::ND>((__gm__ int16_t*) v3 + (v7 + (unsigned) v25 * (unsigned) v12), v29, v30);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    TLOAD(v18, v28);
    TLOAD(v19, v31);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    TSCATTER(v20, v18, v19);
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
  }
  set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
  TSTORE(v23, v20);
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
  #endif // __DAV_VEC__

  return;
}

