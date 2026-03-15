#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void flash_attention_score_stage3(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 2048;
  unsigned v5 = 64;
  unsigned v6 = 1024;
  unsigned v7 = 32;
  unsigned v8 = 1;
  unsigned v9 = 0;
  int32_t v10 = 64;
  int32_t v11 = 32;
  int32_t v12 = 1;
  int32_t v13 = 0;
  int64_t v14 = 0;
  int64_t v15 = 2048;
  using T = float;
  size_t v16 = (size_t) v12;

  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, half, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v17;
  TASSIGN(v17, v14);
  Tile<TileType::Mat, half, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v18;
  TASSIGN(v18, v15);
  Tile<TileType::Left, half, 32, 32, BLayout::RowMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v19;
  TASSIGN(v19, v14);
  Tile<TileType::Right, half, 32, 64, BLayout::RowMajor, 32, 64, SLayout::ColMajor, 512, PadValue::Null> v20;
  TASSIGN(v20, v14);
  Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 1024, PadValue::Null> v21;
  TASSIGN(v21, v14);
  for (size_t v22 = (size_t) v13; v22 < v16; v22 += v16) {
    int32_t v23 = (int32_t) v22;
    int32_t v24 = (int32_t) ((uint32_t) v23 * (uint32_t) v11);
    pto::Shape<1, 1, 1, 32, 32> v25 = pto::Shape<1, 1, 1, 32, 32>();
    pto::Stride<1024, 1024, 1024, 32, 1> v26 = pto::Stride<1024, 1024, 1024, 32, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v27 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v2 + (v9 + v9 * (unsigned) v11 + (unsigned) v24 * (unsigned) v12), v25, v26);
    pto::Shape<1, 1, 1, 32, 64> v28 = pto::Shape<1, 1, 1, 32, 64>();
    pto::Stride<2048, 2048, 2048, 64, 1> v29 = pto::Stride<2048, 2048, 2048, 64, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND> v30 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND>(v3 + (v9 + (unsigned) v24 * (unsigned) v10 + v9 * (unsigned) v12), v28, v29);
    TLOAD(v17, v27);
    TLOAD(v18, v30);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(v19, v17);
    TMOV(v20, v18);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    if (v23 == v13) {
      TMATMUL(v21, v19, v20);
    } else {
      TMATMUL_ACC(v21, v21, v19, v20);
    };
    set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  pto::Shape<1, 1, 1, 32, 64> v31 = pto::Shape<1, 1, 1, 32, 64>();
  pto::Stride<2048, 2048, 2048, 64, 1> v32 = pto::Stride<2048, 2048, 2048, 64, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND> v33 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND>(v1 + (v9 + v9 * (unsigned) v10 + v9 * (unsigned) v12), v31, v32);
  TSTORE(v33, v21);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  #endif // __DAV_CUBE__

  return;
}

