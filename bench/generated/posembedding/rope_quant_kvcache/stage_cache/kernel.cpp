#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void rope_quant_kvcache_cache_stage(__gm__ half* v1, __gm__ half* v2, __gm__ int8_t* v3, __gm__ int8_t* v4, int32_t v5) {
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
  int64_t v18 = 0;
  int64_t v19 = 192;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v20 = get_block_idx();
  int64_t v21 = get_block_num();
  int32_t v22 = (int32_t) v21;
  int32_t v23 = v12 / v22;
  int32_t v24 = v12 % v22 != v10 && v12 < v10 == v22 < v10 ? v23 + v14 : v23;
  int32_t v25 = (int32_t) ((uint32_t) ((int32_t) v20) * (uint32_t) v24);
  int32_t v26 = (int32_t) ((uint32_t) v25 + (uint32_t) v24);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v27;
  TASSIGN(v27, v16);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v28;
  __ubuf__ half* v29 = v27.data();
  uint64_t v30 = reinterpret_cast<uint64_t>(v29);
  TASSIGN(v28, v30);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v31;
  TASSIGN(v31, v17);
  Tile<TileType::Vec, half, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v32;
  __ubuf__ half* v33 = v31.data();
  uint64_t v34 = reinterpret_cast<uint64_t>(v33);
  TASSIGN(v32, v34);
  Tile<TileType::Vec, int8_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v35;
  TASSIGN(v35, v18);
  Tile<TileType::Vec, int8_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v36;
  __ubuf__ int8_t* v37 = v35.data();
  uint64_t v38 = reinterpret_cast<uint64_t>(v37);
  TASSIGN(v36, v38);
  Tile<TileType::Vec, int8_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v39;
  TASSIGN(v39, v19);
  Tile<TileType::Vec, int8_t, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v40;
  __ubuf__ int8_t* v41 = v39.data();
  uint64_t v42 = reinterpret_cast<uint64_t>(v41);
  TASSIGN(v40, v42);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  if (v5 == v12) {
    for (int32_t v43 = v25; v43 < ((uint32_t) v26 < (uint32_t) v12 ? v26 : v12); v43 += v14) {
      pto::Shape<1, 1, 1, 1, 64> v44 = pto::Shape<1, 1, 1, 1, 64>();
      pto::Stride<64, 64, 64, 64, 1> v45 = pto::Stride<64, 64, 64, 64, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v46 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v1 + (v9 + (unsigned) v43 * (unsigned) v11 + v9 * (unsigned) v14), v44, v45);
      pto::Shape<1, 1, 1, 1, 64> v47 = pto::Shape<1, 1, 1, 1, 64>();
      pto::Stride<64, 64, 64, 64, 1> v48 = pto::Stride<64, 64, 64, 64, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v49 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v2 + (v9 + (unsigned) v43 * (unsigned) v11 + v9 * (unsigned) v14), v47, v48);
      int32_t v50 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v43 * (uint32_t) v13) + (uint32_t) v43);
      pto::Shape<1, 1, 1, 1, 64> v51 = pto::Shape<1, 1, 1, 1, 64>();
      pto::Stride<64, 64, 64, 64, 1> v52 = pto::Stride<64, 64, 64, 64, 1>();
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v53 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v3 + (v9 + (unsigned) v50 * (unsigned) v11 + v9 * (unsigned) v14), v51, v52);
      pto::Shape<1, 1, 1, 1, 64> v54 = pto::Shape<1, 1, 1, 1, 64>();
      pto::Stride<64, 64, 64, 64, 1> v55 = pto::Stride<64, 64, 64, 64, 1>();
      GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v56 = GlobalTensor<int8_t, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v4 + (v9 + (unsigned) v50 * (unsigned) v11 + v9 * (unsigned) v14), v54, v55);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v28, v46);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
      TLOAD(v32, v49);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      TCVT(v36, v28, v6);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
      TCVT(v40, v32, v6);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v53, v36);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
      TSTORE(v56, v40);
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

