#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void _stage(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 16384;
  unsigned v5 = 256;
  unsigned v6 = 8192;
  unsigned v7 = 128;
  unsigned v8 = 32;
  unsigned v9 = 64;
  unsigned v10 = 1;
  unsigned v11 = 0;
  int32_t v12 = 4;
  int32_t v13 = 32;
  int32_t v14 = 256;
  int32_t v15 = 128;
  int32_t v16 = 64;
  int32_t v17 = 1;
  int32_t v18 = 0;
  int64_t v19 = 0;
  int64_t v20 = 4096;
  using T = float;

  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, half, 64, 32, BLayout::ColMajor, 64, 32, SLayout::RowMajor, 512, PadValue::Null> v21;
  TASSIGN(v21, v19);
  Tile<TileType::Mat, half, 32, 256, BLayout::ColMajor, 32, 256, SLayout::RowMajor, 512, PadValue::Null> v22;
  TASSIGN(v22, v20);
  Tile<TileType::Left, half, 64, 32, BLayout::RowMajor, 64, 32, SLayout::RowMajor, 512, PadValue::Null> v23;
  TASSIGN(v23, v19);
  Tile<TileType::Right, half, 32, 256, BLayout::RowMajor, 32, 256, SLayout::ColMajor, 512, PadValue::Null> v24;
  TASSIGN(v24, v19);
  Tile<TileType::Acc, float, 64, 256, BLayout::ColMajor, 64, 256, SLayout::RowMajor, 1024, PadValue::Null> v25;
  TASSIGN(v25, v19);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v26 = (size_t) v18; v26 < ((size_t) v12); v26 += (size_t) v17) {
    int32_t v27 = (int32_t) v26;
    int32_t v28 = (int32_t) ((uint32_t) v27 * (uint32_t) v13);
    pto::Shape<1, 1, 1, 64, 32> v29 = pto::Shape<1, 1, 1, 64, 32>();
    pto::Stride<8192, 8192, 8192, 128, 1> v30 = pto::Stride<8192, 8192, 8192, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 64, 32>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v31 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 32>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v2 + (v11 + v11 * (unsigned) v15 + (unsigned) v28 * (unsigned) v17), v29, v30);
    pto::Shape<1, 1, 1, 32, 256> v32 = pto::Shape<1, 1, 1, 32, 256>();
    pto::Stride<8192, 8192, 8192, 256, 1> v33 = pto::Stride<8192, 8192, 8192, 256, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v34 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v3 + (v11 + (unsigned) v28 * (unsigned) v14 + v11 * (unsigned) v17), v32, v33);
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
    if (v27 == v18) {
      TMATMUL(v25, v23, v24);
    } else {
      TMATMUL_ACC(v25, v25, v23, v24);
    };
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  pto::Shape<1, 1, 1, 64, 256> v35 = pto::Shape<1, 1, 1, 64, 256>();
  pto::Stride<16384, 16384, 16384, 256, 1> v36 = pto::Stride<16384, 16384, 16384, 256, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 64, 256>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v37 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 256>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v1 + (v11 + v11 * (unsigned) v14 + v11 * (unsigned) v17), v35, v36);
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

