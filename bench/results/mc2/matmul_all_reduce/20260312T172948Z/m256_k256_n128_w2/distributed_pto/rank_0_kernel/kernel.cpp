#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void matmul_all_reduce_local(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 4096;
  unsigned v5 = 128;
  unsigned v6 = 8192;
  unsigned v7 = 256;
  unsigned v8 = 64;
  unsigned v9 = 32;
  unsigned v10 = 1;
  unsigned v11 = 0;
  int32_t v12 = 8;
  int32_t v13 = 4;
  int32_t v14 = 64;
  int32_t v15 = 32;
  int32_t v16 = 128;
  int32_t v17 = 256;
  int32_t v18 = 1;
  int32_t v19 = 0;
  int64_t v20 = 0;
  int64_t v21 = 4096;
  using T = float;
  size_t v22 = (size_t) v19;
  size_t v23 = (size_t) v18;
  int64_t v24 = get_block_idx();
  int32_t v25 = (int32_t) ((int64_t) v24);

  #if defined(__DAV_CUBE__)
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  if (v25 < v13) {
    Tile<TileType::Mat, half, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v26;
    TASSIGN(v26, v20);
    Tile<TileType::Mat, half, 64, 32, BLayout::ColMajor, 64, 32, SLayout::RowMajor, 512, PadValue::Null> v27;
    TASSIGN(v27, v21);
    Tile<TileType::Left, half, 32, 64, BLayout::RowMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v28;
    TASSIGN(v28, v20);
    Tile<TileType::Right, half, 64, 32, BLayout::RowMajor, 64, 32, SLayout::ColMajor, 512, PadValue::Null> v29;
    TASSIGN(v29, v20);
    Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 1024, PadValue::Null> v30;
    TASSIGN(v30, v20);
    for (size_t v31 = v22; v31 < ((size_t) v12); v31 += v23) {
      int32_t v32 = (int32_t) ((uint32_t) ((int32_t) v31) + (uint32_t) (v25 / v13));
      if (v32 < v12) {
        int32_t v33 = (int32_t) ((uint32_t) v32 * (uint32_t) v15);
        int32_t v34 = (int32_t) ((uint32_t) (v25 % v13) * (uint32_t) v15);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        for (size_t v35 = v22; v35 < ((size_t) v13); v35 += v23) {
          int32_t v36 = (int32_t) v35;
          int32_t v37 = (int32_t) ((uint32_t) v36 * (uint32_t) v14);
          pto::Shape<1, 1, 1, 32, 64> v38 = pto::Shape<1, 1, 1, 32, 64>();
          pto::Stride<8192, 8192, 8192, 256, 1> v39 = pto::Stride<8192, 8192, 8192, 256, 1>();
          GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v40 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v2 + (v11 + (unsigned) v33 * (unsigned) v17 + (unsigned) v37 * (unsigned) v18), v38, v39);
          pto::Shape<1, 1, 1, 64, 32> v41 = pto::Shape<1, 1, 1, 64, 32>();
          pto::Stride<8192, 8192, 8192, 128, 1> v42 = pto::Stride<8192, 8192, 8192, 128, 1>();
          GlobalTensor<half, pto::Shape<1, 1, 1, 64, 32>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v43 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 32>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v11 + (unsigned) v37 * (unsigned) v16 + (unsigned) v34 * (unsigned) v18), v41, v42);
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
          TLOAD(v26, v40);
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
          TLOAD(v27, v43);
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
          wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
          pipe_barrier(PIPE_MTE1);
          TMOV(v28, v26);
          set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
          TMOV(v29, v27);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          if (v36 == v19) {
            TMATMUL(v30, v28, v29);
          } else {
            TMATMUL_ACC(v30, v30, v28, v29);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        };
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        pto::Shape<1, 1, 1, 32, 32> v44 = pto::Shape<1, 1, 1, 32, 32>();
        pto::Stride<4096, 4096, 4096, 128, 1> v45 = pto::Stride<4096, 4096, 4096, 128, 1>();
        GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v46 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v1 + (v11 + (unsigned) v33 * (unsigned) v16 + (unsigned) v34 * (unsigned) v18), v44, v45);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        pipe_barrier(PIPE_FIX);
        TSTORE(v46, v30);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      };
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

