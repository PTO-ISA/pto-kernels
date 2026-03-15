#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_dense_bf16_bf16(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int32_t v4) {
  unsigned v5 = 4096;
  unsigned v6 = 128;
  unsigned v7 = 2048;
  unsigned v8 = 64;
  unsigned v9 = 32;
  unsigned v10 = 1;
  unsigned v11 = 0;
  int32_t v12 = 0;
  int32_t v13 = 1;
  int32_t v14 = 64;
  int32_t v15 = 128;
  int32_t v16 = 32;
  int32_t v17 = 2;
  int32_t v18 = 4;
  int32_t v19 = 3;
  int64_t v20 = 0;
  int64_t v21 = 2048;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v22 = get_block_idx();
  int64_t v23 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v24;
  TASSIGN(v24, v20);
  Tile<TileType::Mat, bfloat16_t, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v25;
  TASSIGN(v25, v21);
  Tile<TileType::Left, bfloat16_t, 32, 32, BLayout::RowMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v26;
  TASSIGN(v26, v20);
  Tile<TileType::Right, bfloat16_t, 32, 64, BLayout::RowMajor, 32, 64, SLayout::ColMajor, 512, PadValue::Null> v27;
  TASSIGN(v27, v20);
  Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 1024, PadValue::Null> v28;
  TASSIGN(v28, v20);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v29 = (size_t) ((int32_t) (int64_t) v22); v29 < ((size_t) v18); v29 += (size_t) ((int32_t) (int64_t) v23)) {
    int32_t v30 = (int32_t) v29;
    bool v31 = v30 == v17;
    bool v32 = v30 == v19;
    int32_t v33 = (int32_t) ((uint32_t) (v32 ? v13 : v31 ? v13 : v12) * (uint32_t) v16);
    int32_t v34 = (int32_t) ((uint32_t) (v32 ? v13 : v31 ? v12 : (v30 == v13 ? v13 : v12)) * (uint32_t) v14);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v35 = (size_t) v12; v35 < ((size_t) v17); v35 += (size_t) v13) {
      int32_t v36 = (int32_t) v35;
      int32_t v37 = (int32_t) ((uint32_t) v36 * (uint32_t) v16);
      pto::Shape<1, 1, 1, 32, 32> v38 = pto::Shape<1, 1, 1, 32, 32>();
      pto::Stride<2048, 2048, 2048, 64, 1> v39 = pto::Stride<2048, 2048, 2048, 64, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND> v40 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND>(v2 + (v11 + (unsigned) v33 * (unsigned) v14 + (unsigned) v37 * (unsigned) v13), v38, v39);
      pto::Shape<1, 1, 1, 32, 64> v41 = pto::Shape<1, 1, 1, 32, 64>();
      pto::Stride<4096, 4096, 4096, 128, 1> v42 = pto::Stride<4096, 4096, 4096, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v43 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v3 + (v11 + (unsigned) v37 * (unsigned) v15 + (unsigned) v34 * (unsigned) v13), v41, v42);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v24, v40);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v25, v43);
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
      if (v36 == v12) {
        TMATMUL(v28, v26, v27);
      } else {
        TMATMUL_ACC(v28, v28, v26, v27);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 32, 64> v44 = pto::Shape<1, 1, 1, 32, 64>();
    pto::Stride<4096, 4096, 4096, 128, 1> v45 = pto::Stride<4096, 4096, 4096, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v46 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v1 + (v11 + (unsigned) v33 * (unsigned) v15 + (unsigned) v34 * (unsigned) v13), v44, v45);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v46, v28);
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

