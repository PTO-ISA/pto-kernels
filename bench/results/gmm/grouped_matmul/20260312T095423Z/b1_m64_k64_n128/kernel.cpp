#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_dense_bf16_bf16(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int32_t v4) {
  unsigned v5 = 2048;
  unsigned v6 = 8192;
  unsigned v7 = 128;
  unsigned v8 = 1024;
  unsigned v9 = 64;
  unsigned v10 = 16;
  unsigned v11 = 1;
  unsigned v12 = 0;
  int32_t v13 = 0;
  int32_t v14 = 1;
  int32_t v15 = 64;
  int32_t v16 = 128;
  int32_t v17 = 16;
  int32_t v18 = 4;
  int32_t v19 = 2;
  int32_t v20 = 3;
  int64_t v21 = 0;
  int64_t v22 = 2048;
  using T = float;
  size_t v23 = (size_t) v14;

  #if defined(__DAV_CUBE__)
  int64_t v24 = get_block_idx();
  int64_t v25 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v26;
  TASSIGN(v26, v21);
  Tile<TileType::Mat, bfloat16_t, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null> v27;
  TASSIGN(v27, v22);
  Tile<TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v28;
  TASSIGN(v28, v21);
  Tile<TileType::Right, bfloat16_t, 64, 128, BLayout::RowMajor, 64, 128, SLayout::ColMajor, 512, PadValue::Null> v29;
  TASSIGN(v29, v21);
  Tile<TileType::Acc, float, 16, 128, BLayout::ColMajor, 16, 128, SLayout::RowMajor, 1024, PadValue::Null> v30;
  TASSIGN(v30, v21);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v31 = (size_t) ((int32_t) (int64_t) v24); v31 < ((size_t) v18); v31 += (size_t) ((int32_t) (int64_t) v25)) {
    int32_t v32 = (int32_t) v31;
    int32_t v33 = (int32_t) ((uint32_t) (v32 == v20 ? v20 : v32 == v19 ? v19 : (v32 == v14 ? v14 : v13)) * (uint32_t) v17);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v34 = (size_t) v13; v34 < v23; v34 += v23) {
      int32_t v35 = (int32_t) v34;
      int32_t v36 = (int32_t) ((uint32_t) v35 * (uint32_t) v15);
      pto::Shape<1, 1, 1, 16, 64> v37 = pto::Shape<1, 1, 1, 16, 64>();
      pto::Stride<1024, 1024, 1024, 64, 1> v38 = pto::Stride<1024, 1024, 1024, 64, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v39 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(v2 + (v12 + (unsigned) v33 * (unsigned) v15 + (unsigned) v36 * (unsigned) v14), v37, v38);
      pto::Shape<1, 1, 1, 64, 128> v40 = pto::Shape<1, 1, 1, 64, 128>();
      pto::Stride<8192, 8192, 8192, 128, 1> v41 = pto::Stride<8192, 8192, 8192, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v42 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v12 + (unsigned) v36 * (unsigned) v16 + v12 * (unsigned) v14), v40, v41);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v26, v39);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v27, v42);
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
      if (v35 == v13) {
        TMATMUL(v30, v28, v29);
      } else {
        TMATMUL_ACC(v30, v30, v28, v29);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 128> v43 = pto::Shape<1, 1, 1, 16, 128>();
    pto::Stride<2048, 2048, 2048, 128, 1> v44 = pto::Stride<2048, 2048, 2048, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v45 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 128>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v1 + (v12 + (unsigned) v33 * (unsigned) v16 + v12 * (unsigned) v14), v43, v44);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v45, v30);
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

