#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void dense_attention_qk_stage(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 256;
  unsigned v5 = 16;
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 16;
  int32_t v9 = 1;
  int64_t v10 = 0;
  int64_t v11 = 512;
  using T = float;
  size_t v12 = (size_t) v9;

  #if defined(__DAV_CUBE__)
  int64_t v13 = get_block_idx();
  int64_t v14 = get_block_num();
  Tile<TileType::Mat, half, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null> v15;
  TASSIGN(v15, v10);
  Tile<TileType::Mat, half, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null> v16;
  TASSIGN(v16, v11);
  Tile<TileType::Left, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null> v17;
  TASSIGN(v17, v10);
  Tile<TileType::Right, half, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor, 512, PadValue::Null> v18;
  TASSIGN(v18, v10);
  Tile<TileType::Acc, float, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 1024, PadValue::Null> v19;
  TASSIGN(v19, v10);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  for (size_t v20 = (size_t) ((int32_t) (int64_t) v13); v20 < v12; v20 += (size_t) ((int32_t) (int64_t) v14)) {
    pto::Shape<1, 1, 1, 16, 16> v21 = pto::Shape<1, 1, 1, 16, 16>();
    pto::Stride<256, 256, 256, 16, 1> v22 = pto::Stride<256, 256, 256, 16, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND> v23 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND>(v2 + (v7 + v7 * (unsigned) v8 + v7 * (unsigned) v9), v21, v22);
    pto::Shape<1, 1, 1, 16, 16> v24 = pto::Shape<1, 1, 1, 16, 16>();
    pto::Stride<256, 256, 256, 16, 1> v25 = pto::Stride<256, 256, 256, 16, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND> v26 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND>(v3 + (v7 + v7 * (unsigned) v8 + v7 * (unsigned) v9), v24, v25);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v15, v23);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TLOAD(v16, v26);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    TMOV(v17, v15);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TMOV(v18, v16);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    TMATMUL(v19, v17, v18);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    for (size_t v27 = v12; v27 < v12; v27 += v12) {
      int32_t v28 = (int32_t) ((uint32_t) ((int32_t) v27) * (uint32_t) v8);
      pto::Shape<1, 1, 1, 16, 16> v29 = pto::Shape<1, 1, 1, 16, 16>();
      pto::Stride<256, 256, 256, 16, 1> v30 = pto::Stride<256, 256, 256, 16, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND> v31 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND>(v2 + (v7 + v7 * (unsigned) v8 + (unsigned) v28 * (unsigned) v9), v29, v30);
      pto::Shape<1, 1, 1, 16, 16> v32 = pto::Shape<1, 1, 1, 16, 16>();
      pto::Stride<256, 256, 256, 16, 1> v33 = pto::Stride<256, 256, 256, 16, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND> v34 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND>(v3 + (v7 + (unsigned) v28 * (unsigned) v8 + v7 * (unsigned) v9), v32, v33);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
      TLOAD(v15, v31);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
      TLOAD(v16, v34);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      TMOV(v17, v15);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
      TMOV(v18, v16);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      TMATMUL_ACC(v19, v19, v17, v18);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    };
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 16> v35 = pto::Shape<1, 1, 1, 16, 16>();
    pto::Stride<256, 256, 256, 16, 1> v36 = pto::Stride<256, 256, 256, 16, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND> v37 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND>(v1 + (v7 + v7 * (unsigned) v8 + v7 * (unsigned) v9), v35, v36);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(v37, v19);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  #endif // __DAV_CUBE__

  return;
}

