#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_add_add_stage(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3) {
  unsigned v4 = 256;
  unsigned v5 = 1;
  unsigned v6 = 0;
  int32_t v7 = 0;
  int32_t v8 = 256;
  int32_t v9 = 128;
  int32_t v10 = 1;
  int64_t v11 = 0;
  int64_t v12 = 1024;
  int64_t v13 = 2048;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v14 = get_block_idx();
  int64_t v15 = get_block_num();
  int32_t v16 = (int32_t) v15;
  int32_t v17 = v9 / v16;
  int32_t v18 = v9 % v16 != v7 && v9 < v7 == v16 < v7 ? v17 + v10 : v17;
  int32_t v19 = (int32_t) ((uint32_t) ((int32_t) v14) * (uint32_t) v18);
  int32_t v20 = (int32_t) ((uint32_t) v19 + (uint32_t) v18);
  Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null> v21;
  TASSIGN(v21, v11);
  Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null> v22;
  __ubuf__ float* v23 = v21.data();
  uint64_t v24 = reinterpret_cast<uint64_t>(v23);
  TASSIGN(v22, v24);
  Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null> v25;
  TASSIGN(v25, v12);
  Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null> v26;
  __ubuf__ float* v27 = v25.data();
  uint64_t v28 = reinterpret_cast<uint64_t>(v27);
  TASSIGN(v26, v28);
  Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null> v29;
  TASSIGN(v29, v13);
  Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null> v30;
  __ubuf__ float* v31 = v29.data();
  uint64_t v32 = reinterpret_cast<uint64_t>(v31);
  TASSIGN(v30, v32);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  for (int32_t v33 = v19; v33 < ((uint32_t) v20 < (uint32_t) v9 ? v20 : v9); v33 += v10) {
    pto::Shape<1, 1, 1, 1, 256> v34 = pto::Shape<1, 1, 1, 1, 256>();
    pto::Stride<256, 256, 256, 256, 1> v35 = pto::Stride<256, 256, 256, 256, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v36 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v33 * (unsigned) v8 + v6 * (unsigned) v10), v34, v35);
    pto::Shape<1, 1, 1, 1, 256> v37 = pto::Shape<1, 1, 1, 1, 256>();
    pto::Stride<256, 256, 256, 256, 1> v38 = pto::Stride<256, 256, 256, 256, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v39 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v3 + (v6 + (unsigned) v33 * (unsigned) v8 + v6 * (unsigned) v10), v37, v38);
    pto::Shape<1, 1, 1, 1, 256> v40 = pto::Shape<1, 1, 1, 1, 256>();
    pto::Stride<256, 256, 256, 256, 1> v41 = pto::Stride<256, 256, 256, 256, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v42 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) v33 * (unsigned) v8 + v6 * (unsigned) v10), v40, v41);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v22, v36);
    TLOAD(v26, v39);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TADD(v30, v22, v26);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v42, v30);
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

