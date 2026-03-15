#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void _stage(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3, __gm__ half* v4) {
  unsigned v5 = 128;
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 0;
  int32_t v9 = 512;
  int32_t v10 = 128;
  int32_t v11 = 1;
  int64_t v12 = 256;
  int64_t v13 = 512;
  int64_t v14 = 0;
  int64_t v15 = 768;
  int64_t v16 = 1024;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v17 = get_block_idx();
  int64_t v18 = get_block_num();
  int32_t v19 = (int32_t) ((int64_t) v18);
  int32_t v20 = v9 / v19;
  int32_t v21 = v9 % v19 != v8 && v9 < v8 == v19 < v8 ? v20 + v11 : v20;
  int32_t v22 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v17) * (uint32_t) v21);
  int32_t v23 = (int32_t) ((uint32_t) v22 + (uint32_t) v21);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v24;
  TASSIGN(v24, v12);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v25;
  TASSIGN(v25, v13);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v26;
  TASSIGN(v26, v14);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v27;
  TASSIGN(v27, v15);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v28;
  TASSIGN(v28, v16);
  pto::Shape<1, 1, 1, 1, 128> v29 = pto::Shape<1, 1, 1, 1, 128>();
  pto::Stride<128, 128, 128, 128, 1> v30 = pto::Stride<128, 128, 128, 128, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v31 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v3 + (v7 + v7 * (unsigned) v10 + v7 * (unsigned) v11), v29, v30);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  TLOAD(v26, v31);
  for (size_t v32 = (size_t) v22; v32 < ((size_t) ((uint32_t) v23 < (uint32_t) v9 ? v23 : v9)); v32 += (size_t) v11) {
    int32_t v33 = (int32_t) v32;
    pto::Shape<1, 1, 1, 1, 128> v34 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v35 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v36 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v2 + (v7 + (unsigned) v33 * (unsigned) v10 + v7 * (unsigned) v11), v34, v35);
    pto::Shape<1, 1, 1, 1, 128> v37 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v38 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v39 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v4 + (v7 + (unsigned) v33 * (unsigned) v10 + v7 * (unsigned) v11), v37, v38);
    pto::Shape<1, 1, 1, 1, 128> v40 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v41 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v42 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v33 * (unsigned) v10 + v7 * (unsigned) v11), v40, v41);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v24, v36);
    TLOAD(v25, v39);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_V);
    TADD(v27, v24, v25);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TADD(v28, v27, v26);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v42, v28);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  #endif // __DAV_VEC__

  return;
}

