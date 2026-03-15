#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_dense_bf16_bf16(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int32_t v4) {
  unsigned v5 = 16384;
  unsigned v6 = 256;
  unsigned v7 = 8192;
  unsigned v8 = 128;
  unsigned v9 = 64;
  unsigned v10 = 1;
  unsigned v11 = 0;
  int32_t v12 = 0;
  int32_t v13 = 1;
  int32_t v14 = 64;
  int32_t v15 = 256;
  int32_t v16 = 2;
  int32_t v17 = 128;
  int64_t v18 = 0;
  int64_t v19 = 8192;
  using T = float;

  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, bfloat16_t, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v20;
  TASSIGN(v20, v18);
  Tile<TileType::Mat, bfloat16_t, 64, 256, BLayout::ColMajor, 64, 256, SLayout::RowMajor, 512, PadValue::Null> v21;
  TASSIGN(v21, v19);
  Tile<TileType::Left, bfloat16_t, 64, 64, BLayout::RowMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v22;
  TASSIGN(v22, v18);
  Tile<TileType::Right, bfloat16_t, 64, 256, BLayout::RowMajor, 64, 256, SLayout::ColMajor, 512, PadValue::Null> v23;
  TASSIGN(v23, v18);
  Tile<TileType::Acc, float, 64, 256, BLayout::ColMajor, 64, 256, SLayout::RowMajor, 1024, PadValue::Null> v24;
  TASSIGN(v24, v18);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v25 = (size_t) v12; v25 < ((size_t) v16); v25 += (size_t) v13) {
    int32_t v26 = (int32_t) v25;
    int32_t v27 = (int32_t) ((uint32_t) v26 * (uint32_t) v14);
    pto::Shape<1, 1, 1, 64, 64> v28 = pto::Shape<1, 1, 1, 64, 64>();
    pto::Stride<8192, 8192, 8192, 128, 1> v29 = pto::Stride<8192, 8192, 8192, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v30 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v2 + (v11 + v11 * (unsigned) v17 + (unsigned) v27 * (unsigned) v13), v28, v29);
    pto::Shape<1, 1, 1, 64, 256> v31 = pto::Shape<1, 1, 1, 64, 256>();
    pto::Stride<16384, 16384, 16384, 256, 1> v32 = pto::Stride<16384, 16384, 16384, 256, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 256>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v33 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 256>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v3 + (v11 + (unsigned) v27 * (unsigned) v15 + v11 * (unsigned) v13), v31, v32);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v20, v30);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    TLOAD(v21, v33);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    pipe_barrier(PIPE_MTE1);
    TMOV(v22, v20);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TMOV(v23, v21);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    if (v26 == v12) {
      TMATMUL(v24, v22, v23);
    } else {
      TMATMUL_ACC(v24, v24, v22, v23);
    };
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  pto::Shape<1, 1, 1, 64, 256> v34 = pto::Shape<1, 1, 1, 64, 256>();
  pto::Stride<16384, 16384, 16384, 256, 1> v35 = pto::Stride<16384, 16384, 16384, 256, 1>();
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 256>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v36 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 256>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v1 + (v11 + v11 * (unsigned) v15 + v11 * (unsigned) v13), v34, v35);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v36, v24);
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  #endif // __DAV_CUBE__

  return;
}

