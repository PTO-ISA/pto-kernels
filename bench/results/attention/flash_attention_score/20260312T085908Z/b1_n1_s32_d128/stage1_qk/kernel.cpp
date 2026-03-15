#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void dense_attention_qk_stage(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 1024;
  unsigned v5 = 2048;
  unsigned v6 = 4096;
  unsigned v7 = 128;
  unsigned v8 = 64;
  unsigned v9 = 32;
  unsigned v10 = 1;
  unsigned v11 = 0;
  int32_t v12 = 2;
  int32_t v13 = 64;
  int32_t v14 = 128;
  int32_t v15 = 32;
  int32_t v16 = 1;
  int32_t v17 = 0;
  int64_t v18 = 0;
  int64_t v19 = 4096;
  using T = float;

  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, half, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v20;
  TASSIGN(v20, v18);
  Tile<TileType::Mat, half, 64, 32, BLayout::ColMajor, 64, 32, SLayout::RowMajor, 512, PadValue::Null> v21;
  TASSIGN(v21, v19);
  Tile<TileType::Left, half, 32, 64, BLayout::RowMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v22;
  TASSIGN(v22, v18);
  Tile<TileType::Right, half, 64, 32, BLayout::RowMajor, 64, 32, SLayout::ColMajor, 512, PadValue::Null> v23;
  TASSIGN(v23, v18);
  Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 1024, PadValue::Null> v24;
  TASSIGN(v24, v18);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v25 = (size_t) v17; v25 < ((size_t) v12); v25 += (size_t) v16) {
    int32_t v26 = (int32_t) v25;
    int32_t v27 = (int32_t) ((uint32_t) v26 * (uint32_t) v13);
    pto::Shape<1, 1, 1, 32, 64> v28 = pto::Shape<1, 1, 1, 32, 64>();
    pto::Stride<4096, 4096, 4096, 128, 1> v29 = pto::Stride<4096, 4096, 4096, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v30 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v2 + (v11 + v11 * (unsigned) v14 + (unsigned) v27 * (unsigned) v16), v28, v29);
    pto::Shape<1, 1, 1, 64, 32> v31 = pto::Shape<1, 1, 1, 64, 32>();
    pto::Stride<2048, 2048, 2048, 32, 1> v32 = pto::Stride<2048, 2048, 2048, 32, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 64, 32>, pto::Stride<2048, 2048, 2048, 32, 1>, pto::Layout::ND> v33 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 32>, pto::Stride<2048, 2048, 2048, 32, 1>, pto::Layout::ND>(v3 + (v11 + (unsigned) v27 * (unsigned) v15 + v11 * (unsigned) v16), v31, v32);
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
    if (v26 == v17) {
      TMATMUL(v24, v22, v23);
    } else {
      TMATMUL_ACC(v24, v24, v22, v23);
    };
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  pto::Shape<1, 1, 1, 32, 32> v34 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<1024, 1024, 1024, 32, 1> v35 = pto::Stride<1024, 1024, 1024, 32, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v36 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v1 + (v11 + v11 * (unsigned) v15 + v11 * (unsigned) v16), v34, v35);
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

