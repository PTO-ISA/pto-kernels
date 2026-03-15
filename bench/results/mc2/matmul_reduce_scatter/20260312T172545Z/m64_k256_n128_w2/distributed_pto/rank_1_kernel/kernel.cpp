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
  int32_t v11 = 2;
  int32_t v12 = 8;
  int32_t v13 = 32;
  int32_t v14 = 128;
  int32_t v15 = 256;
  int32_t v16 = 64;
  int32_t v17 = 1;
  int32_t v18 = 0;
  int64_t v19 = 0;
  int64_t v20 = 2048;
  using T = float;
  size_t v21 = (size_t) v18;
  size_t v22 = (size_t) v17;

  #if defined(__DAV_CUBE__)
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v23 = v21; v23 < ((size_t) v11); v23 += v22) {
    for (size_t v24 = v21; v24 < v22; v24 += v22) {
      int32_t v25 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) ((int32_t) ((uint32_t) ((int32_t) v23) + (uint32_t) v17) % v11) * (uint32_t) v13) + (uint32_t) ((int32_t) (uint32_t) ((int32_t) v24) * (uint32_t) v13));
      Tile<TileType::Mat, half, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v26;
      TASSIGN(v26, v19);
      Tile<TileType::Mat, half, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null> v27;
      TASSIGN(v27, v20);
      Tile<TileType::Left, half, 32, 32, BLayout::RowMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v28;
      TASSIGN(v28, v19);
      Tile<TileType::Right, half, 32, 128, BLayout::RowMajor, 32, 128, SLayout::ColMajor, 512, PadValue::Null> v29;
      TASSIGN(v29, v19);
      Tile<TileType::Acc, float, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 1024, PadValue::Null> v30;
      TASSIGN(v30, v19);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      for (size_t v31 = v21; v31 < ((size_t) v12); v31 += v22) {
        int32_t v32 = (int32_t) v31;
        int32_t v33 = (int32_t) ((uint32_t) v32 * (uint32_t) v13);
        pto::Shape<1, 1, 1, 32, 32> v34 = pto::Shape<1, 1, 1, 32, 32>();
        pto::Stride<8192, 8192, 8192, 256, 1> v35 = pto::Stride<8192, 8192, 8192, 256, 1>();
        GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v36 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v2 + (v10 + (unsigned) v25 * (unsigned) v15 + (unsigned) v33 * (unsigned) v17), v34, v35);
        pto::Shape<1, 1, 1, 32, 128> v37 = pto::Shape<1, 1, 1, 32, 128>();
        pto::Stride<4096, 4096, 4096, 128, 1> v38 = pto::Stride<4096, 4096, 4096, 128, 1>();
        GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v39 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v3 + (v10 + (unsigned) v33 * (unsigned) v14 + v10 * (unsigned) v17), v37, v38);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        TLOAD(v26, v36);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        TLOAD(v27, v39);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        pipe_barrier(PIPE_MTE1);
        TMOV(v28, v26);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        TMOV(v29, v27);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        if (v32 == v18) {
          TMATMUL(v30, v28, v29);
        } else {
          TMATMUL_ACC(v30, v30, v28, v29);
        };
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      };
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      pto::Shape<1, 1, 1, 32, 128> v40 = pto::Shape<1, 1, 1, 32, 128>();
      pto::Stride<4096, 4096, 4096, 128, 1> v41 = pto::Stride<4096, 4096, 4096, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v42 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v1 + (v10 + (unsigned) v25 * (unsigned) v14 + v10 * (unsigned) v17), v40, v41);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      pipe_barrier(PIPE_FIX);
      TSTORE(v42, v30);
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    };
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

