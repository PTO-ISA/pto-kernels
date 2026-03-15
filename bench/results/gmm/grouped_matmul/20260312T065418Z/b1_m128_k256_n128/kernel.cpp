#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_dense_bf16_bf16(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int32_t v4) {
  unsigned v5 = 16384;
  unsigned v6 = 8192;
  unsigned v7 = 32768;
  unsigned v8 = 256;
  unsigned v9 = 64;
  unsigned v10 = 128;
  unsigned v11 = 1;
  unsigned v12 = 0;
  int32_t v13 = 0;
  int32_t v14 = 1;
  int32_t v15 = 128;
  int32_t v16 = 64;
  int32_t v17 = 4;
  int32_t v18 = 256;
  int64_t v19 = 0;
  int64_t v20 = 16384;
  using T = float;

  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, bfloat16_t, 128, 64, BLayout::ColMajor, 128, 64, SLayout::RowMajor, 512, PadValue::Null> v21;
  TASSIGN(v21, v19);
  Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null> v22;
  TASSIGN(v22, v20);
  Tile<TileType::Left, bfloat16_t, 128, 64, BLayout::RowMajor, 128, 64, SLayout::RowMajor, 512, PadValue::Null> v23;
  TASSIGN(v23, v19);
  Tile<TileType::Right, bfloat16_t, 64, 128, BLayout::RowMajor, 64, 128, SLayout::ColMajor, 512, PadValue::Null> v24;
  TASSIGN(v24, v19);
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v25;
  TASSIGN(v25, v19);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v26 = (size_t) v13; v26 < ((size_t) v17); v26 += (size_t) v14) {
    int32_t v27 = (int32_t) v26;
    int32_t v28 = (int32_t) ((uint32_t) v27 * (uint32_t) v16);
    pto::Shape<1, 1, 1, 128, 64> v29 = pto::Shape<1, 1, 1, 128, 64>();
    pto::Stride<32768, 32768, 32768, 256, 1> v30 = pto::Stride<32768, 32768, 32768, 256, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 64>, pto::Stride<32768, 32768, 32768, 256, 1>, pto::Layout::ND> v31 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 64>, pto::Stride<32768, 32768, 32768, 256, 1>, pto::Layout::ND>(v2 + (v12 + v12 * (unsigned) v18 + (unsigned) v28 * (unsigned) v14), v29, v30);
    pto::Shape<1, 1, 1, 64, 128> v32 = pto::Shape<1, 1, 1, 64, 128>();
    pto::Stride<8192, 8192, 8192, 128, 1> v33 = pto::Stride<8192, 8192, 8192, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v34 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v12 + (unsigned) v28 * (unsigned) v15 + v12 * (unsigned) v14), v32, v33);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v21, v31);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    TLOAD(v22, v34);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    pipe_barrier(PIPE_MTE1);
    TMOV(v23, v21);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TMOV(v24, v22);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    if (v27 == v13) {
      TMATMUL(v25, v23, v24);
    } else {
      TMATMUL_ACC(v25, v25, v23, v24);
    };
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  pto::Shape<1, 1, 1, 128, 128> v35 = pto::Shape<1, 1, 1, 128, 128>();
  pto::Stride<16384, 16384, 16384, 128, 1> v36 = pto::Stride<16384, 16384, 16384, 128, 1>();
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v37 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v1 + (v12 + v12 * (unsigned) v15 + v12 * (unsigned) v14), v35, v36);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v37, v25);
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  #endif // __DAV_CUBE__

  return;
}

