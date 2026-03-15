#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void dense_attention_qk_stage(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 4096;
  unsigned v5 = 64;
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 64;
  int32_t v9 = 1;
  int32_t v10 = 0;
  int64_t v11 = 0;
  int64_t v12 = 8192;
  using T = float;
  size_t v13 = (size_t) v9;

  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v14;
  TASSIGN(v14, v11);
  Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v15;
  TASSIGN(v15, v12);
  Tile<TileType::Left, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v16;
  TASSIGN(v16, v11);
  Tile<TileType::Right, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v17;
  TASSIGN(v17, v11);
  Tile<TileType::Acc, float, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 1024, PadValue::Null> v18;
  TASSIGN(v18, v11);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v19 = (size_t) v10; v19 < v13; v19 += v13) {
    int32_t v20 = (int32_t) v19;
    int32_t v21 = (int32_t) ((uint32_t) v20 * (uint32_t) v8);
    pto::Shape<1, 1, 1, 64, 64> v22 = pto::Shape<1, 1, 1, 64, 64>();
    pto::Stride<4096, 4096, 4096, 64, 1> v23 = pto::Stride<4096, 4096, 4096, 64, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<4096, 4096, 4096, 64, 1>, pto::Layout::ND> v24 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<4096, 4096, 4096, 64, 1>, pto::Layout::ND>(v2 + (v7 + v7 * (unsigned) v8 + (unsigned) v21 * (unsigned) v9), v22, v23);
    pto::Shape<1, 1, 1, 64, 64> v25 = pto::Shape<1, 1, 1, 64, 64>();
    pto::Stride<4096, 4096, 4096, 64, 1> v26 = pto::Stride<4096, 4096, 4096, 64, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<4096, 4096, 4096, 64, 1>, pto::Layout::ND> v27 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<4096, 4096, 4096, 64, 1>, pto::Layout::ND>(v3 + (v7 + (unsigned) v21 * (unsigned) v8 + v7 * (unsigned) v9), v25, v26);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v14, v24);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    TLOAD(v15, v27);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    pipe_barrier(PIPE_MTE1);
    TMOV(v16, v14);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TMOV(v17, v15);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    if (v20 == v10) {
      TMATMUL(v18, v16, v17);
    } else {
      TMATMUL_ACC(v18, v18, v16, v17);
    };
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  pto::Shape<1, 1, 1, 64, 64> v28 = pto::Shape<1, 1, 1, 64, 64>();
  pto::Stride<4096, 4096, 4096, 64, 1> v29 = pto::Stride<4096, 4096, 4096, 64, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<4096, 4096, 4096, 64, 1>, pto::Layout::ND> v30 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<4096, 4096, 4096, 64, 1>, pto::Layout::ND>(v1 + (v7 + v7 * (unsigned) v8 + v7 * (unsigned) v9), v28, v29);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v30, v18);
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  #endif // __DAV_CUBE__

  return;
}

