#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_gating_top_k_softmax_stage(__gm__ half* v1, __gm__ half* v2) {
  unsigned v3 = 64;
  unsigned v4 = 1;
  unsigned v5 = 0;
  int32_t v6 = 0;
  int32_t v7 = 64;
  int32_t v8 = 256;
  int32_t v9 = 1;
  int64_t v10 = 0;
  int64_t v11 = 128;
  int64_t v12 = 256;
  int64_t v13 = 384;
  int64_t v14 = 512;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v15 = get_block_idx();
  int64_t v16 = get_block_num();
  int32_t v17 = (int32_t) v16;
  int32_t v18 = v8 / v17;
  int32_t v19 = v8 % v17 != v6 && v8 < v6 == v17 < v6 ? v18 + v9 : v18;
  int32_t v20 = (int32_t) ((uint32_t) ((int32_t) v15) * (uint32_t) v19);
  int32_t v21 = (int32_t) ((uint32_t) v20 + (uint32_t) v19);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v22;
  TASSIGN(v22, v10);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v23;
  __ubuf__ half* v24 = v22.data();
  uint64_t v25 = reinterpret_cast<uint64_t>(v24);
  TASSIGN(v23, v25);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v26;
  TASSIGN(v26, v11);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v27;
  __ubuf__ half* v28 = v26.data();
  uint64_t v29 = reinterpret_cast<uint64_t>(v28);
  TASSIGN(v27, v29);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v30;
  TASSIGN(v30, v12);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v31;
  __ubuf__ half* v32 = v30.data();
  uint64_t v33 = reinterpret_cast<uint64_t>(v32);
  TASSIGN(v31, v33);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v34;
  TASSIGN(v34, v13);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 1, SLayout::NoneBox, 512, PadValue::Null> v35;
  __ubuf__ half* v36 = v34.data();
  uint64_t v37 = reinterpret_cast<uint64_t>(v36);
  TASSIGN(v35, v37);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v38;
  TASSIGN(v38, v14);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v39;
  __ubuf__ half* v40 = v38.data();
  uint64_t v41 = reinterpret_cast<uint64_t>(v40);
  TASSIGN(v39, v41);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  for (int32_t v42 = v20; v42 < ((uint32_t) v21 < (uint32_t) v8 ? v21 : v8); v42 += v9) {
    pto::Shape<1, 1, 1, 1, 64> v43 = pto::Shape<1, 1, 1, 1, 64>();
    pto::Stride<64, 64, 64, 64, 1> v44 = pto::Stride<64, 64, 64, 64, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v45 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v2 + (v5 + (unsigned) v42 * (unsigned) v7 + v5 * (unsigned) v9), v43, v44);
    pto::Shape<1, 1, 1, 1, 64> v46 = pto::Shape<1, 1, 1, 1, 64>();
    pto::Stride<64, 64, 64, 64, 1> v47 = pto::Stride<64, 64, 64, 64, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v48 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v1 + (v5 + (unsigned) v42 * (unsigned) v7 + v5 * (unsigned) v9), v46, v47);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v23, v45);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pipe_barrier(PIPE_V);
    TROWMAX(v35, v23, v27);
    pipe_barrier(PIPE_V);
    TROWEXPAND(v39, v35);
    pipe_barrier(PIPE_V);
    TSUB(v27, v23, v39);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_V);
    TEXP(v27, v27);
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TROWSUM(v35, v27, v31);
    pipe_barrier(PIPE_V);
    TROWEXPAND(v39, v35);
    pipe_barrier(PIPE_V);
    TDIV(v31, v27, v39);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v48, v31);
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

