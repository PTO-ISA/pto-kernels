#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_dense_bf16_bf16(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int32_t v4) {
  unsigned v5 = 4096;
  unsigned v6 = 16384;
  unsigned v7 = 32;
  unsigned v8 = 128;
  unsigned v9 = 1;
  unsigned v10 = 0;
  int32_t v11 = 0;
  int32_t v12 = 1;
  int32_t v13 = 128;
  int32_t v14 = 32;
  int32_t v15 = 4;
  int64_t v16 = 0;
  int64_t v17 = 8192;
  using T = float;

  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, bfloat16_t, 128, 32, BLayout::ColMajor, 128, 32, SLayout::RowMajor, 512, PadValue::Null> v18;
  TASSIGN(v18, v16);
  Tile<TileType::Mat, bfloat16_t, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null> v19;
  TASSIGN(v19, v17);
  Tile<TileType::Left, bfloat16_t, 128, 32, BLayout::RowMajor, 128, 32, SLayout::RowMajor, 512, PadValue::Null> v20;
  TASSIGN(v20, v16);
  Tile<TileType::Right, bfloat16_t, 32, 128, BLayout::RowMajor, 32, 128, SLayout::ColMajor, 512, PadValue::Null> v21;
  TASSIGN(v21, v16);
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v22;
  TASSIGN(v22, v16);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v23 = (size_t) v11; v23 < ((size_t) v15); v23 += (size_t) v12) {
    int32_t v24 = (int32_t) v23;
    int32_t v25 = (int32_t) ((uint32_t) v24 * (uint32_t) v14);
    pto::Shape<1, 1, 1, 128, 32> v26 = pto::Shape<1, 1, 1, 128, 32>();
    pto::Stride<16384, 16384, 16384, 128, 1> v27 = pto::Stride<16384, 16384, 16384, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 32>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v28 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 32>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v2 + (v10 + v10 * (unsigned) v13 + (unsigned) v25 * (unsigned) v12), v26, v27);
    pto::Shape<1, 1, 1, 32, 128> v29 = pto::Shape<1, 1, 1, 32, 128>();
    pto::Stride<4096, 4096, 4096, 128, 1> v30 = pto::Stride<4096, 4096, 4096, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v31 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v3 + (v10 + (unsigned) v25 * (unsigned) v13 + v10 * (unsigned) v12), v29, v30);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v18, v28);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    TLOAD(v19, v31);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    pipe_barrier(PIPE_MTE1);
    TMOV(v20, v18);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TMOV(v21, v19);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    if (v24 == v11) {
      TMATMUL(v22, v20, v21);
    } else {
      TMATMUL_ACC(v22, v22, v20, v21);
    };
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  pto::Shape<1, 1, 1, 128, 128> v32 = pto::Shape<1, 1, 1, 128, 128>();
  pto::Stride<16384, 16384, 16384, 128, 1> v33 = pto::Stride<16384, 16384, 16384, 128, 1>();
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v34 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v1 + (v10 + v10 * (unsigned) v13 + v10 * (unsigned) v12), v32, v33);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v34, v22);
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  #endif // __DAV_CUBE__

  return;
}

