#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void matmul_reduce_scatter_local_mm(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 4096;
  unsigned v5 = 128;
  unsigned v6 = 8192;
  unsigned v7 = 256;
  unsigned v8 = 32;
  unsigned v9 = 1;
  unsigned v10 = 0;
  int32_t v11 = 8;
  int32_t v12 = 32;
  int32_t v13 = 128;
  int32_t v14 = 256;
  int32_t v15 = 64;
  int32_t v16 = 1;
  int32_t v17 = 0;
  int64_t v18 = 0;
  int64_t v19 = 2048;
  using T = float;
  size_t v20 = (size_t) v17;
  size_t v21 = (size_t) v16;

  #if defined(__DAV_CUBE__)
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v22 = v20; v22 < v21; v22 += v21) {
    int32_t v23 = (int32_t) ((uint32_t) ((int32_t) v22) * (uint32_t) v12);
    Tile<TileType::Mat, half, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v24;
    TASSIGN(v24, v18);
    Tile<TileType::Mat, half, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null> v25;
    TASSIGN(v25, v19);
    Tile<TileType::Left, half, 32, 32, BLayout::RowMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v26;
    TASSIGN(v26, v18);
    Tile<TileType::Right, half, 32, 128, BLayout::RowMajor, 32, 128, SLayout::ColMajor, 512, PadValue::Null> v27;
    TASSIGN(v27, v18);
    Tile<TileType::Acc, float, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 1024, PadValue::Null> v28;
    TASSIGN(v28, v18);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v29 = v20; v29 < ((size_t) v11); v29 += v21) {
      int32_t v30 = (int32_t) v29;
      int32_t v31 = (int32_t) ((uint32_t) v30 * (uint32_t) v12);
      pto::Shape<1, 1, 1, 32, 32> v32 = pto::Shape<1, 1, 1, 32, 32>();
      pto::Stride<8192, 8192, 8192, 256, 1> v33 = pto::Stride<8192, 8192, 8192, 256, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v34 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v2 + (v10 + (unsigned) ((int32_t) (uint32_t) v23 + (uint32_t) v12) * (unsigned) v14 + (unsigned) v31 * (unsigned) v16), v32, v33);
      pto::Shape<1, 1, 1, 32, 128> v35 = pto::Shape<1, 1, 1, 32, 128>();
      pto::Stride<4096, 4096, 4096, 128, 1> v36 = pto::Stride<4096, 4096, 4096, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v37 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v3 + (v10 + (unsigned) v31 * (unsigned) v13 + v10 * (unsigned) v16), v35, v36);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v24, v34);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v25, v37);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v26, v24);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v27, v25);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v30 == v17) {
        TMATMUL(v28, v26, v27);
      } else {
        TMATMUL_ACC(v28, v28, v26, v27);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 32, 128> v38 = pto::Shape<1, 1, 1, 32, 128>();
    pto::Stride<4096, 4096, 4096, 128, 1> v39 = pto::Stride<4096, 4096, 4096, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v40 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v1 + (v10 + (unsigned) v23 * (unsigned) v13 + v10 * (unsigned) v16), v38, v39);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v40, v28);
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

