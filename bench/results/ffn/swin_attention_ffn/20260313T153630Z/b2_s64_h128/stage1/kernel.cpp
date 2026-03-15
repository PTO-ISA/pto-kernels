#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void _stage(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 16384;
  unsigned v5 = 128;
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 128;
  int32_t v9 = 1;
  int32_t v10 = 0;
  int64_t v11 = 0;
  int64_t v12 = 32768;
  using T = float;
  size_t v13 = (size_t) v9;

  #if defined(__DAV_CUBE__)
  int64_t v14 = get_block_idx();
  int64_t v15 = get_block_num();
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v16;
  TASSIGN(v16, v11);
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v17;
  TASSIGN(v17, v12);
  Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v18;
  TASSIGN(v18, v11);
  Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> v19;
  TASSIGN(v19, v11);
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v20;
  TASSIGN(v20, v11);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v21 = (size_t) ((int32_t) (int64_t) v14); v21 < v13; v21 += (size_t) ((int32_t) (int64_t) v15)) {
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v22 = (size_t) v10; v22 < v13; v22 += v13) {
      int32_t v23 = (int32_t) v22;
      int32_t v24 = (int32_t) ((uint32_t) v23 * (uint32_t) v8);
      pto::Shape<1, 1, 1, 128, 128> v25 = pto::Shape<1, 1, 1, 128, 128>();
      pto::Stride<16384, 16384, 16384, 128, 1> v26 = pto::Stride<16384, 16384, 16384, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v27 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v2 + (v7 + v7 * (unsigned) v8 + (unsigned) v24 * (unsigned) v9), v25, v26);
      pto::Shape<1, 1, 1, 128, 128> v28 = pto::Shape<1, 1, 1, 128, 128>();
      pto::Stride<16384, 16384, 16384, 128, 1> v29 = pto::Stride<16384, 16384, 16384, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v30 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v3 + (v7 + (unsigned) v24 * (unsigned) v8 + v7 * (unsigned) v9), v28, v29);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v16, v27);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v17, v30);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v18, v16);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v19, v17);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v23 == v10) {
        TMATMUL(v20, v18, v19);
      } else {
        TMATMUL_ACC(v20, v20, v18, v19);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 128, 128> v31 = pto::Shape<1, 1, 1, 128, 128>();
    pto::Stride<16384, 16384, 16384, 128, 1> v32 = pto::Stride<16384, 16384, 16384, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v33 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v1 + (v7 + v7 * (unsigned) v8 + v7 * (unsigned) v9), v31, v32);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v33, v20);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  #endif // __DAV_CUBE__

  return;
}

