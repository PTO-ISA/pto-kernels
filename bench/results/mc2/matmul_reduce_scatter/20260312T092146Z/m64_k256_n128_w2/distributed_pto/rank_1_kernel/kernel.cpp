#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void matmul_reduce_scatter_local_mm(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 8192;
  unsigned v5 = 4096;
  unsigned v6 = 128;
  unsigned v7 = 16384;
  unsigned v8 = 256;
  unsigned v9 = 32;
  unsigned v10 = 64;
  unsigned v11 = 1;
  unsigned v12 = 0;
  int32_t v13 = 8;
  int32_t v14 = 32;
  int32_t v15 = 128;
  int32_t v16 = 256;
  int32_t v17 = 64;
  int32_t v18 = 1;
  int32_t v19 = 0;
  int64_t v20 = 0;
  int64_t v21 = 4096;
  using T = float;

  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, half, 64, 32, BLayout::ColMajor, 64, 32, SLayout::RowMajor, 512, PadValue::Null> v22;
  TASSIGN(v22, v20);
  Tile<TileType::Mat, half, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null> v23;
  TASSIGN(v23, v21);
  Tile<TileType::Left, half, 64, 32, BLayout::RowMajor, 64, 32, SLayout::RowMajor, 512, PadValue::Null> v24;
  TASSIGN(v24, v20);
  Tile<TileType::Right, half, 32, 128, BLayout::RowMajor, 32, 128, SLayout::ColMajor, 512, PadValue::Null> v25;
  TASSIGN(v25, v20);
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null> v26;
  TASSIGN(v26, v20);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v27 = (size_t) v19; v27 < ((size_t) v13); v27 += (size_t) v18) {
    int32_t v28 = (int32_t) v27;
    int32_t v29 = (int32_t) ((uint32_t) v28 * (uint32_t) v14);
    pto::Shape<1, 1, 1, 64, 32> v30 = pto::Shape<1, 1, 1, 64, 32>();
    pto::Stride<16384, 16384, 16384, 256, 1> v31 = pto::Stride<16384, 16384, 16384, 256, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 64, 32>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND> v32 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 32>, pto::Stride<16384, 16384, 16384, 256, 1>, pto::Layout::ND>(v2 + (v12 + v12 * (unsigned) v16 + (unsigned) v29 * (unsigned) v18), v30, v31);
    pto::Shape<1, 1, 1, 32, 128> v33 = pto::Shape<1, 1, 1, 32, 128>();
    pto::Stride<4096, 4096, 4096, 128, 1> v34 = pto::Stride<4096, 4096, 4096, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v35 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v3 + (v12 + (unsigned) v29 * (unsigned) v15 + v12 * (unsigned) v18), v33, v34);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v22, v32);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    TLOAD(v23, v35);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    pipe_barrier(PIPE_MTE1);
    TMOV(v24, v22);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TMOV(v25, v23);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    if (v28 == v19) {
      TMATMUL(v26, v24, v25);
    } else {
      TMATMUL_ACC(v26, v26, v24, v25);
    };
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  pto::Shape<1, 1, 1, 64, 128> v36 = pto::Shape<1, 1, 1, 64, 128>();
  pto::Stride<8192, 8192, 8192, 128, 1> v37 = pto::Stride<8192, 8192, 8192, 128, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v38 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v1 + (v12 + v12 * (unsigned) v15 + v12 * (unsigned) v18), v36, v37);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v38, v26);
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  #endif // __DAV_CUBE__

  return;
}

