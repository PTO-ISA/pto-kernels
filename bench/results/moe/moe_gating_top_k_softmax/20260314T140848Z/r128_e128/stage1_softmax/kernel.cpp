#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_gating_top_k_softmax_stage(__gm__ half* v1, __gm__ half* v2) {
  unsigned v3 = 128;
  unsigned v4 = 1;
  unsigned v5 = 0;
  int32_t v6 = 0;
  int32_t v7 = 128;
  int32_t v8 = 1;
  int64_t v9 = 0;
  int64_t v10 = 256;
  int64_t v11 = 512;
  int64_t v12 = 768;
  int64_t v13 = 1024;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v14 = get_block_idx();
  int64_t v15 = get_block_num();
  int32_t v16 = (int32_t) v15;
  int32_t v17 = v7 / v16;
  int32_t v18 = v7 % v16 != v6 && v7 < v6 == v16 < v6 ? v17 + v8 : v17;
  int32_t v19 = (int32_t) ((uint32_t) ((int32_t) v14) * (uint32_t) v18);
  int32_t v20 = (int32_t) ((uint32_t) v19 + (uint32_t) v18);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v21;
  TASSIGN(v21, v9);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v22;
  __ubuf__ half* v23 = v21.data();
  uint64_t v24 = reinterpret_cast<uint64_t>(v23);
  TASSIGN(v22, v24);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v25;
  TASSIGN(v25, v10);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v26;
  __ubuf__ half* v27 = v25.data();
  uint64_t v28 = reinterpret_cast<uint64_t>(v27);
  TASSIGN(v26, v28);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v29;
  TASSIGN(v29, v11);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v30;
  __ubuf__ half* v31 = v29.data();
  uint64_t v32 = reinterpret_cast<uint64_t>(v31);
  TASSIGN(v30, v32);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v33;
  TASSIGN(v33, v12);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 1, SLayout::NoneBox, 512, PadValue::Null> v34;
  __ubuf__ half* v35 = v33.data();
  uint64_t v36 = reinterpret_cast<uint64_t>(v35);
  TASSIGN(v34, v36);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v37;
  TASSIGN(v37, v13);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v38;
  __ubuf__ half* v39 = v37.data();
  uint64_t v40 = reinterpret_cast<uint64_t>(v39);
  TASSIGN(v38, v40);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  for (int32_t v41 = v19; v41 < ((uint32_t) v20 < (uint32_t) v7 ? v20 : v7); v41 += v8) {
    pto::Shape<1, 1, 1, 1, 128> v42 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v43 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v44 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v2 + (v5 + (unsigned) v41 * (unsigned) v7 + v5 * (unsigned) v8), v42, v43);
    pto::Shape<1, 1, 1, 1, 128> v45 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v46 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v47 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v1 + (v5 + (unsigned) v41 * (unsigned) v7 + v5 * (unsigned) v8), v45, v46);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v22, v44);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_V);
    TROWMAX(v34, v22, v26);
    pipe_barrier(PIPE_V);
    TROWEXPAND(v38, v34);
    pipe_barrier(PIPE_V);
    TSUB(v26, v22, v38);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_V);
    TEXP(v26, v26);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TROWSUM(v34, v26, v30);
    pipe_barrier(PIPE_V);
    TROWEXPAND(v38, v34);
    pipe_barrier(PIPE_V);
    TDIV(v30, v26, v38);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v47, v30);
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

