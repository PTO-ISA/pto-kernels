#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_mat_mul_all_reduce_local(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 8192;
  unsigned v5 = 4096;
  unsigned v6 = 128;
  unsigned v7 = 64;
  unsigned v8 = 32;
  unsigned v9 = 1;
  unsigned v10 = 0;
  int32_t v11 = 4;
  int32_t v12 = 2;
  int32_t v13 = 64;
  int32_t v14 = 32;
  int32_t v15 = 128;
  int32_t v16 = 1;
  int32_t v17 = 0;
  int64_t v18 = 0;
  int64_t v19 = 4096;
  using T = float;
  size_t v20 = (size_t) v17;
  size_t v21 = (size_t) v16;
  int64_t v22 = get_block_idx();
  int32_t v23 = (int32_t) ((int64_t) v22);

  #if defined(__DAV_CUBE__)
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  if (v23 < v11) {
    Tile<TileType::Mat, half, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v24;
    TASSIGN(v24, v18);
    Tile<TileType::Mat, half, 64, 32, BLayout::ColMajor, 64, 32, SLayout::RowMajor, 512, PadValue::Null> v25;
    TASSIGN(v25, v19);
    Tile<TileType::Left, half, 32, 64, BLayout::RowMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v26;
    TASSIGN(v26, v18);
    Tile<TileType::Right, half, 64, 32, BLayout::RowMajor, 64, 32, SLayout::ColMajor, 512, PadValue::Null> v27;
    TASSIGN(v27, v18);
    Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 1024, PadValue::Null> v28;
    TASSIGN(v28, v18);
    for (size_t v29 = v20; v29 < ((size_t) v11); v29 += v21) {
      int32_t v30 = (int32_t) ((uint32_t) ((int32_t) v29) + (uint32_t) (v23 / v11));
      if (v30 < v11) {
        int32_t v31 = (int32_t) ((uint32_t) v30 * (uint32_t) v14);
        int32_t v32 = (int32_t) ((uint32_t) (v23 % v11) * (uint32_t) v14);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        for (size_t v33 = v20; v33 < ((size_t) v12); v33 += v21) {
          int32_t v34 = (int32_t) v33;
          int32_t v35 = (int32_t) ((uint32_t) v34 * (uint32_t) v13);
          pto::Shape<1, 1, 1, 32, 64> v36 = pto::Shape<1, 1, 1, 32, 64>();
          pto::Stride<4096, 4096, 4096, 128, 1> v37 = pto::Stride<4096, 4096, 4096, 128, 1>();
          GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v38 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v2 + (v10 + (unsigned) v31 * (unsigned) v15 + (unsigned) v35 * (unsigned) v16), v36, v37);
          pto::Shape<1, 1, 1, 64, 32> v39 = pto::Shape<1, 1, 1, 64, 32>();
          pto::Stride<8192, 8192, 8192, 128, 1> v40 = pto::Stride<8192, 8192, 8192, 128, 1>();
          GlobalTensor<half, pto::Shape<1, 1, 1, 64, 32>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v41 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 32>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v10 + (unsigned) v35 * (unsigned) v15 + (unsigned) v32 * (unsigned) v16), v39, v40);
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
          TLOAD(v24, v38);
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
          TLOAD(v25, v41);
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
          if (v34 == v17) {
            TMATMUL(v28, v26, v27);
          } else {
            TMATMUL_ACC(v28, v28, v26, v27);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        };
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        pto::Shape<1, 1, 1, 32, 32> v42 = pto::Shape<1, 1, 1, 32, 32>();
        pto::Stride<4096, 4096, 4096, 128, 1> v43 = pto::Stride<4096, 4096, 4096, 128, 1>();
        GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v44 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v1 + (v10 + (unsigned) v31 * (unsigned) v15 + (unsigned) v32 * (unsigned) v16), v42, v43);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        pipe_barrier(PIPE_FIX);
        TSTORE(v44, v28);
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

