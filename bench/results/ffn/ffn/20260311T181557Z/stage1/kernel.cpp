#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void _stage(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 8192;
  unsigned v5 = 256;
  unsigned v6 = 4096;
  unsigned v7 = 128;
  unsigned v8 = 32;
  unsigned v9 = 1;
  unsigned v10 = 0;
  int32_t v11 = 4;
  int32_t v12 = 256;
  int32_t v13 = 128;
  int32_t v14 = 32;
  int32_t v15 = 1;
  int32_t v16 = 0;
  int64_t v17 = 0;
  int64_t v18 = 2048;
  using T = float;

  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, half, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v19;
  TASSIGN(v19, v17);
  Tile<TileType::Mat, half, 32, 256, BLayout::ColMajor, 32, 256, SLayout::RowMajor, 512, PadValue::Null> v20;
  TASSIGN(v20, v18);
  Tile<TileType::Left, half, 32, 32, BLayout::RowMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v21;
  TASSIGN(v21, v17);
  Tile<TileType::Right, half, 32, 256, BLayout::RowMajor, 32, 256, SLayout::ColMajor, 512, PadValue::Null> v22;
  TASSIGN(v22, v17);
  Tile<TileType::Acc, float, 32, 256, BLayout::ColMajor, 32, 256, SLayout::RowMajor, 1024, PadValue::Null> v23;
  TASSIGN(v23, v17);
  for (size_t v24 = (size_t) v16; v24 < ((size_t) v11); v24 += (size_t) v15) {
    int32_t v25 = (int32_t) v24;
    int32_t v26 = (int32_t) ((uint32_t) v25 * (uint32_t) v14);
    pto::Shape<1, 1, 1, 32, 32> v27 = pto::Shape<1, 1, 1, 32, 32>();
    pto::Stride<4096, 4096, 4096, 128, 1> v28 = pto::Stride<4096, 4096, 4096, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v29 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v2 + (v10 + v10 * (unsigned) v13 + (unsigned) v26 * (unsigned) v15), v27, v28);
    pto::Shape<1, 1, 1, 32, 256> v30 = pto::Shape<1, 1, 1, 32, 256>();
    pto::Stride<8192, 8192, 8192, 256, 1> v31 = pto::Stride<8192, 8192, 8192, 256, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v32 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v3 + (v10 + (unsigned) v26 * (unsigned) v12 + v10 * (unsigned) v15), v30, v31);
    TLOAD(v19, v29);
    TLOAD(v20, v32);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(v21, v19);
    TMOV(v22, v20);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    if (v25 == v16) {
      TMATMUL(v23, v21, v22);
    } else {
      TMATMUL_ACC(v23, v23, v21, v22);
    };
    set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  pto::Shape<1, 1, 1, 32, 256> v33 = pto::Shape<1, 1, 1, 32, 256>();
  pto::Stride<8192, 8192, 8192, 256, 1> v34 = pto::Stride<8192, 8192, 8192, 256, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v35 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v1 + (v10 + v10 * (unsigned) v12 + v10 * (unsigned) v15), v33, v34);
  TSTORE(v35, v23);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  #endif // __DAV_CUBE__

  return;
}

