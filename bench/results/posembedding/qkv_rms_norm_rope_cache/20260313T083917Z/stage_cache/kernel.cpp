#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void qkv_rms_norm_rope_cache_cache_stage(__gm__ half* v1, __gm__ half* v2, __gm__ int8_t* v3, __gm__ int8_t* v4, int32_t v5) {
  RoundMode v6 = RoundMode::CAST_ROUND;
  unsigned v7 = 64;
  unsigned v8 = 1;
  unsigned v9 = 0;
  int32_t v10 = 0;
  int32_t v11 = 64;
  int32_t v12 = 2;
  int32_t v13 = 8;
  int32_t v14 = 1;
  int32_t v15 = 16;
  int64_t v16 = 64;
  int64_t v17 = 256;
  int64_t v18 = 192;
  int64_t v19 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v20 = get_block_idx();
  int64_t v21 = get_block_num();
  int32_t v22 = (int32_t) ((int64_t) v21);
  int32_t v23 = v12 / v22;
  int32_t v24 = v12 % v22 != v10 && v12 < v10 == v22 < v10 ? v23 + v14 : v23;
  int32_t v25 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v20) * (uint32_t) v24);
  int32_t v26 = (int32_t) ((uint32_t) v25 + (uint32_t) v24);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v27;
  TASSIGN(v27, v16);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v28;
  TASSIGN(v28, v17);
  Tile<TileType::Vec, int8_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v29;
  TASSIGN(v29, v18);
  Tile<TileType::Vec, int8_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v30;
  TASSIGN(v30, v19);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  if (v5 == v12) {
    for (size_t v31 = (size_t) v25; v31 < ((size_t) ((uint32_t) v26 < (uint32_t) v12 ? v26 : v12)); v31 += (size_t) v14) {
      int32_t v32 = (int32_t) v31;
      pto::Shape<1, 1, 1, 1, 64> v33 = pto::Shape<1, 1, 1, 1, 64>();
      pto::Stride<64, 64, 64, 64, 1> v34 = pto::Stride<64, 64, 64, 64, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v35 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v1 + (v9 + (unsigned) v32 * (unsigned) v11 + v9 * (unsigned) v14), v33, v34);
      pto::Shape<1, 1, 1, 1, 64> v36 = pto::Shape<1, 1, 1, 1, 64>();
      pto::Stride<64, 64, 64, 64, 1> v37 = pto::Stride<64, 64, 64, 64, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v38 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v2 + (v9 + (unsigned) v32 * (unsigned) v11 + v9 * (unsigned) v14), v36, v37);
      int32_t v39 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v32 * (uint32_t) v13) + (uint32_t) v32);
      pto::Shape<1, 1, 1, 1, 64> v40 = pto::Shape<1, 1, 1, 1, 64>();
      pto::Stride<64, 64, 64, 64, 1> v41 = pto::Stride<64, 64, 64, 64, 1>();
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v42 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v3 + (v9 + (unsigned) v39 * (unsigned) v11 + v9 * (unsigned) v14), v40, v41);
      pto::Shape<1, 1, 1, 1, 64> v43 = pto::Shape<1, 1, 1, 1, 64>();
      pto::Stride<64, 64, 64, 64, 1> v44 = pto::Stride<64, 64, 64, 64, 1>();
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v45 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v4 + (v9 + (unsigned) v39 * (unsigned) v11 + v9 * (unsigned) v14), v43, v44);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v27, v35);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
      TLOAD(v28, v38);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      TCVT(v29, v27, v6);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      TCVT(v30, v28, v6);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v42, v29);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      TSTORE(v45, v30);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    };
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  #endif // __DAV_VEC__

  return;
}

