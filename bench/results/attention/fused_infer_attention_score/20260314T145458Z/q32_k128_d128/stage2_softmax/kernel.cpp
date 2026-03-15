#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void dense_attention_row_softmax(__gm__ half* v1) {
  unsigned v2 = 128;
  unsigned v3 = 1;
  unsigned v4 = 0;
  int32_t v5 = 0;
  float v6 = 0.0883883461f;
  int32_t v7 = 128;
  int32_t v8 = 32;
  int32_t v9 = 1;
  int64_t v10 = 0;
  int64_t v11 = 256;
  int64_t v12 = 512;
  int64_t v13 = 768;
  int64_t v14 = 1024;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v15 = get_block_idx();
  int64_t v16 = get_block_num();
  int32_t v17 = (int32_t) v16;
  int32_t v18 = v8 / v17;
  int32_t v19 = v8 % v17 != v5 && v8 < v5 == v17 < v5 ? v18 + v9 : v18;
  int32_t v20 = (int32_t) ((uint32_t) ((int32_t) v15) * (uint32_t) v19);
  int32_t v21 = (int32_t) ((uint32_t) v20 + (uint32_t) v19);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v22;
  TASSIGN(v22, v10);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v23;
  __ubuf__ half* v24 = v22.data();
  uint64_t v25 = reinterpret_cast<uint64_t>(v24);
  TASSIGN(v23, v25);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v26;
  TASSIGN(v26, v11);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v27;
  __ubuf__ half* v28 = v26.data();
  uint64_t v29 = reinterpret_cast<uint64_t>(v28);
  TASSIGN(v27, v29);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v30;
  TASSIGN(v30, v12);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v31;
  __ubuf__ half* v32 = v30.data();
  uint64_t v33 = reinterpret_cast<uint64_t>(v32);
  TASSIGN(v31, v33);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v34;
  TASSIGN(v34, v13);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 1, SLayout::NoneBox, 512, PadValue::Null> v35;
  __ubuf__ half* v36 = v34.data();
  uint64_t v37 = reinterpret_cast<uint64_t>(v36);
  TASSIGN(v35, v37);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v38;
  TASSIGN(v38, v14);
  Tile<TileType::Vec, half, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v39;
  __ubuf__ half* v40 = v38.data();
  uint64_t v41 = reinterpret_cast<uint64_t>(v40);
  TASSIGN(v39, v41);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int32_t v42 = v20; v42 < ((uint32_t) v21 < (uint32_t) v8 ? v21 : v8); v42 += v9) {
    pto::Shape<1, 1, 1, 1, 128> v43 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v44 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v45 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v1 + (v4 + (unsigned) v42 * (unsigned) v7 + v4 * (unsigned) v9), v43, v44);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v23, v45);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMULS(v23, v23, v6);
    pipe_barrier(PIPE_V);
    TROWMAX(v35, v23, v27);
    pipe_barrier(PIPE_V);
    TROWEXPAND(v39, v35);
    pipe_barrier(PIPE_V);
    TSUB(v27, v23, v39);
    pipe_barrier(PIPE_V);
    TEXP(v27, v27);
    pipe_barrier(PIPE_V);
    TROWSUM(v35, v27, v31);
    pipe_barrier(PIPE_V);
    TROWEXPAND(v39, v35);
    pipe_barrier(PIPE_V);
    TDIV(v31, v27, v39);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v45, v31);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  return;
}

