#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_dense_bf16_bf16(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int32_t v4) {
  unsigned v5 = 8192;
  unsigned v6 = 4096;
  unsigned v7 = 128;
  unsigned v8 = 64;
  unsigned v9 = 32;
  unsigned v10 = 1;
  unsigned v11 = 0;
  int32_t v12 = 0;
  int32_t v13 = 1;
  int32_t v14 = 128;
  int32_t v15 = 32;
  int32_t v16 = 64;
  int32_t v17 = 2;
  int32_t v18 = 8;
  int32_t v19 = 3;
  int32_t v20 = 4;
  int32_t v21 = 5;
  int32_t v22 = 6;
  int32_t v23 = 7;
  int64_t v24 = 8192;
  int64_t v25 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v26 = get_block_idx();
  int64_t v27 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v28;
  TASSIGN(v28, v24);
  Tile<TileType::Mat, bfloat16_t, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v29;
  TASSIGN(v29, v25);
  Tile<TileType::Left, bfloat16_t, 32, 64, BLayout::RowMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v30;
  TASSIGN(v30, v25);
  Tile<TileType::Right, bfloat16_t, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v31;
  TASSIGN(v31, v25);
  Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 1024, PadValue::Null> v32;
  TASSIGN(v32, v25);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v33 = (size_t) ((int32_t) (int64_t) v26); v33 < ((size_t) v18); v33 += (size_t) ((int32_t) (int64_t) v27)) {
    int32_t v34 = (int32_t) v33;
    bool v35 = v34 == v17;
    bool v36 = v34 == v19;
    bool v37 = v34 == v20;
    bool v38 = v34 == v21;
    bool v39 = v34 == v22;
    bool v40 = v34 == v23;
    int32_t v41 = (int32_t) ((uint32_t) (v40 ? v19 : v39 ? v19 : (v38 ? v17 : v37 ? v17 : (v36 ? v13 : v35 ? v13 : v12))) * (uint32_t) v15);
    int32_t v42 = (int32_t) ((uint32_t) (v40 ? v13 : v39 ? v12 : (v38 ? v13 : v37 ? v12 : (v36 ? v13 : v35 ? v12 : (v34 == v13 ? v13 : v12)))) * (uint32_t) v16);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v43 = (size_t) v12; v43 < ((size_t) v17); v43 += (size_t) v13) {
      int32_t v44 = (int32_t) v43;
      int32_t v45 = (int32_t) ((uint32_t) v44 * (uint32_t) v16);
      pto::Shape<1, 1, 1, 32, 64> v46 = pto::Shape<1, 1, 1, 32, 64>();
      pto::Stride<4096, 4096, 4096, 128, 1> v47 = pto::Stride<4096, 4096, 4096, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v48 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v2 + (v11 + (unsigned) v41 * (unsigned) v14 + (unsigned) v45 * (unsigned) v13), v46, v47);
      pto::Shape<1, 1, 1, 64, 64> v49 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<8192, 8192, 8192, 128, 1> v50 = pto::Stride<8192, 8192, 8192, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v51 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v11 + (unsigned) v45 * (unsigned) v14 + (unsigned) v42 * (unsigned) v13), v49, v50);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v28, v48);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v29, v51);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v30, v28);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v31, v29);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v44 == v12) {
        TMATMUL(v32, v30, v31);
      } else {
        TMATMUL_ACC(v32, v32, v30, v31);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 32, 64> v52 = pto::Shape<1, 1, 1, 32, 64>();
    pto::Stride<4096, 4096, 4096, 128, 1> v53 = pto::Stride<4096, 4096, 4096, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v54 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v1 + (v11 + (unsigned) v41 * (unsigned) v14 + (unsigned) v42 * (unsigned) v13), v52, v53);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v54, v32);
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

