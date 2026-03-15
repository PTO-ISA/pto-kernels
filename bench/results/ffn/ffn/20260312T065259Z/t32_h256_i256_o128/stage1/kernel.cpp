#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void _stage(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 8192;
  unsigned v5 = 256;
  unsigned v6 = 32;
  unsigned v7 = 1;
  unsigned v8 = 0;
  int32_t v9 = 8;
  int32_t v10 = 256;
  int32_t v11 = 32;
  int32_t v12 = 1;
  int32_t v13 = 0;
  int64_t v14 = 0;
  int64_t v15 = 2048;
  using T = float;

  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, half, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v16;
  TASSIGN(v16, v14);
  Tile<TileType::Mat, half, 32, 256, BLayout::ColMajor, 32, 256, SLayout::RowMajor, 512, PadValue::Null> v17;
  TASSIGN(v17, v15);
  Tile<TileType::Left, half, 32, 32, BLayout::RowMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v18;
  TASSIGN(v18, v14);
  Tile<TileType::Right, half, 32, 256, BLayout::RowMajor, 32, 256, SLayout::ColMajor, 512, PadValue::Null> v19;
  TASSIGN(v19, v14);
  Tile<TileType::Acc, float, 32, 256, BLayout::ColMajor, 32, 256, SLayout::RowMajor, 1024, PadValue::Null> v20;
  TASSIGN(v20, v14);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v21 = (size_t) v13; v21 < ((size_t) v9); v21 += (size_t) v12) {
    int32_t v22 = (int32_t) v21;
    int32_t v23 = (int32_t) ((uint32_t) v22 * (uint32_t) v11);
    pto::Shape<1, 1, 1, 32, 32> v24 = pto::Shape<1, 1, 1, 32, 32>();
    pto::Stride<8192, 8192, 8192, 256, 1> v25 = pto::Stride<8192, 8192, 8192, 256, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v26 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v2 + (v8 + v8 * (unsigned) v10 + (unsigned) v23 * (unsigned) v12), v24, v25);
    pto::Shape<1, 1, 1, 32, 256> v27 = pto::Shape<1, 1, 1, 32, 256>();
    pto::Stride<8192, 8192, 8192, 256, 1> v28 = pto::Stride<8192, 8192, 8192, 256, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v29 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v3 + (v8 + (unsigned) v23 * (unsigned) v10 + v8 * (unsigned) v12), v27, v28);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v16, v26);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    TLOAD(v17, v29);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    pipe_barrier(PIPE_MTE1);
    TMOV(v18, v16);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TMOV(v19, v17);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    if (v22 == v13) {
      TMATMUL(v20, v18, v19);
    } else {
      TMATMUL_ACC(v20, v20, v18, v19);
    };
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  pto::Shape<1, 1, 1, 32, 256> v30 = pto::Shape<1, 1, 1, 32, 256>();
  pto::Stride<8192, 8192, 8192, 256, 1> v31 = pto::Stride<8192, 8192, 8192, 256, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v32 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v1 + (v8 + v8 * (unsigned) v10 + v8 * (unsigned) v12), v30, v31);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v32, v20);
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  #endif // __DAV_CUBE__

  return;
}

