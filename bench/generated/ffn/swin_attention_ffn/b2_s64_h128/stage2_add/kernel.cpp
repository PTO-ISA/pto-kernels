#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void _stage(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3, __gm__ half* v4) {
  unsigned v5 = 128;
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 0;
  int32_t v9 = 128;
  int32_t v10 = 1;
  int64_t v11 = 256;
  int64_t v12 = 512;
  int64_t v13 = 0;
  int64_t v14 = 768;
  int64_t v15 = 1024;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v16 = get_block_idx();
  int64_t v17 = get_block_num();
  int32_t v18 = (int32_t) ((int64_t) v17);
  int32_t v19 = v9 / v18;
  int32_t v20 = v9 % v18 != v8 && v9 < v8 == v18 < v8 ? v19 + v10 : v19;
  int32_t v21 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v16) * (uint32_t) v20);
  int32_t v22 = (int32_t) ((uint32_t) v21 + (uint32_t) v20);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v23;
  TASSIGN(v23, v11);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v24;
  TASSIGN(v24, v12);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v25;
  TASSIGN(v25, v13);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v26;
  TASSIGN(v26, v14);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v27;
  TASSIGN(v27, v15);
  pto::Shape<1, 1, 1, 1, 128> v28 = pto::Shape<1, 1, 1, 1, 128>();
  pto::Stride<128, 128, 128, 128, 1> v29 = pto::Stride<128, 128, 128, 128, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v30 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v3 + (v7 + v7 * (unsigned) v9 + v7 * (unsigned) v10), v28, v29);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  TLOAD(v25, v30);
  for (size_t v31 = (size_t) v21; v31 < ((size_t) ((uint32_t) v22 < (uint32_t) v9 ? v22 : v9)); v31 += (size_t) v10) {
    int32_t v32 = (int32_t) v31;
    pto::Shape<1, 1, 1, 1, 128> v33 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v34 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v35 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v2 + (v7 + (unsigned) v32 * (unsigned) v9 + v7 * (unsigned) v10), v33, v34);
    pto::Shape<1, 1, 1, 1, 128> v36 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v37 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v38 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v4 + (v7 + (unsigned) v32 * (unsigned) v9 + v7 * (unsigned) v10), v36, v37);
    pto::Shape<1, 1, 1, 1, 128> v39 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v40 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v41 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v32 * (unsigned) v9 + v7 * (unsigned) v10), v39, v40);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v23, v35);
    TLOAD(v24, v38);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_V);
    TADD(v26, v23, v24);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TADD(v27, v26, v25);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v41, v27);
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

