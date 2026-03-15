#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void _stage(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 24576;
  unsigned v5 = 384;
  unsigned v6 = 8192;
  unsigned v7 = 128;
  unsigned v8 = 64;
  unsigned v9 = 1;
  unsigned v10 = 0;
  int32_t v11 = 3;
  int32_t v12 = 2;
  int32_t v13 = 384;
  int32_t v14 = 128;
  int32_t v15 = 64;
  int32_t v16 = 1;
  int32_t v17 = 0;
  int64_t v18 = 0;
  int64_t v19 = 8192;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v20 = get_block_idx();
  int64_t v21 = get_block_num();
  Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v22;
  TASSIGN(v22, v18);
  Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null> v23;
  TASSIGN(v23, v19);
  Tile<TileType::Left, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v24;
  TASSIGN(v24, v18);
  Tile<TileType::Right, half, 64, 128, BLayout::RowMajor, 64, 128, SLayout::ColMajor, 512, PadValue::Null> v25;
  TASSIGN(v25, v18);
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null> v26;
  TASSIGN(v26, v18);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v27 = (size_t) ((int32_t) (int64_t) v20); v27 < ((size_t) v11); v27 += (size_t) ((int32_t) (int64_t) v21)) {
    int32_t v28 = (int32_t) v27;
    int32_t v29 = (int32_t) ((uint32_t) (v28 == v12 ? v12 : v28 == v16 ? v16 : v17) * (uint32_t) v14);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v30 = (size_t) v17; v30 < ((size_t) v12); v30 += (size_t) v16) {
      int32_t v31 = (int32_t) v30;
      int32_t v32 = (int32_t) ((uint32_t) v31 * (uint32_t) v15);
      pto::Shape<1, 1, 1, 64, 64> v33 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<8192, 8192, 8192, 128, 1> v34 = pto::Stride<8192, 8192, 8192, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v35 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v2 + (v10 + v10 * (unsigned) v14 + (unsigned) v32 * (unsigned) v16), v33, v34);
      pto::Shape<1, 1, 1, 64, 128> v36 = pto::Shape<1, 1, 1, 64, 128>();
      pto::Stride<24576, 24576, 24576, 384, 1> v37 = pto::Stride<24576, 24576, 24576, 384, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND> v38 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND>(v3 + (v10 + (unsigned) v32 * (unsigned) v13 + (unsigned) v29 * (unsigned) v16), v36, v37);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v22, v35);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v23, v38);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v24, v22);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v25, v23);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v31 == v17) {
        TMATMUL(v26, v24, v25);
      } else {
        TMATMUL_ACC(v26, v26, v24, v25);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 64, 128> v39 = pto::Shape<1, 1, 1, 64, 128>();
    pto::Stride<24576, 24576, 24576, 384, 1> v40 = pto::Stride<24576, 24576, 24576, 384, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND> v41 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND>(v1 + (v10 + v10 * (unsigned) v13 + (unsigned) v29 * (unsigned) v16), v39, v40);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v41, v26);
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

