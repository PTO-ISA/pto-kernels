#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void dense_attention_row_softmax(__gm__ half* v1) {
  unsigned v2 = 16;
  unsigned v3 = 1;
  unsigned v4 = 0;
  int32_t v5 = 0;
  int32_t v6 = 16;
  int32_t v7 = 1;
  int64_t v8 = 0;
  int64_t v9 = 32;
  int64_t v10 = 64;
  int64_t v11 = 96;
  int64_t v12 = 128;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v13 = get_block_idx();
  int64_t v14 = get_block_num();
  int32_t v15 = (int32_t) v14;
  int32_t v16 = v6 / v15;
  int32_t v17 = v6 % v15 != v5 && v6 < v5 == v15 < v5 ? v16 + v7 : v16;
  int32_t v18 = (int32_t) ((uint32_t) ((int32_t) v13) * (uint32_t) v17);
  int32_t v19 = (int32_t) ((uint32_t) v18 + (uint32_t) v17);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v20;
  TASSIGN(v20, v8);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v21;
  __ubuf__ half* v22 = v20.data();
  uint64_t v23 = reinterpret_cast<uint64_t>(v22);
  TASSIGN(v21, v23);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v24;
  TASSIGN(v24, v9);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v25;
  __ubuf__ half* v26 = v24.data();
  uint64_t v27 = reinterpret_cast<uint64_t>(v26);
  TASSIGN(v25, v27);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v28;
  TASSIGN(v28, v10);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v29;
  __ubuf__ half* v30 = v28.data();
  uint64_t v31 = reinterpret_cast<uint64_t>(v30);
  TASSIGN(v29, v31);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v32;
  TASSIGN(v32, v11);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 1, SLayout::NoneBox, 512, PadValue::Null> v33;
  __ubuf__ half* v34 = v32.data();
  uint64_t v35 = reinterpret_cast<uint64_t>(v34);
  TASSIGN(v33, v35);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v36;
  TASSIGN(v36, v12);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v37;
  __ubuf__ half* v38 = v36.data();
  uint64_t v39 = reinterpret_cast<uint64_t>(v38);
  TASSIGN(v37, v39);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  for (int32_t v40 = v18; v40 < ((uint32_t) v19 < (uint32_t) v6 ? v19 : v6); v40 += v7) {
    pto::Shape<1, 1, 1, 1, 16> v41 = pto::Shape<1, 1, 1, 1, 16>();
    pto::Stride<16, 16, 16, 16, 1> v42 = pto::Stride<16, 16, 16, 16, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v43 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v1 + (v4 + (unsigned) v40 * (unsigned) v6 + v4 * (unsigned) v7), v41, v42);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v21, v43);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWMAX(v33, v21, v25);
    pipe_barrier(PIPE_V);
    TROWEXPAND(v37, v33);
    pipe_barrier(PIPE_V);
    TSUB(v25, v21, v37);
    pipe_barrier(PIPE_V);
    TEXP(v25, v25);
    pipe_barrier(PIPE_V);
    TROWSUM(v33, v25, v29);
    pipe_barrier(PIPE_V);
    TROWEXPAND(v37, v33);
    pipe_barrier(PIPE_V);
    TDIV(v29, v25, v37);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v43, v29);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  return;
}

