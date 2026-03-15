#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_dense_bf16_bf16(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int32_t v4) {
  unsigned v5 = 64;
  unsigned v6 = 4096;
  unsigned v7 = 128;
  unsigned v8 = 32;
  unsigned v9 = 1;
  unsigned v10 = 0;
  int32_t v11 = 0;
  int32_t v12 = 1;
  int32_t v13 = 128;
  int32_t v14 = 32;
  int32_t v15 = 64;
  int32_t v16 = 4;
  int32_t v17 = 8;
  int32_t v18 = 2;
  int32_t v19 = 3;
  int32_t v20 = 5;
  int32_t v21 = 6;
  int32_t v22 = 7;
  int64_t v23 = 4096;
  int64_t v24 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v25 = get_block_idx();
  int64_t v26 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v27;
  TASSIGN(v27, v23);
  Tile<TileType::Mat, bfloat16_t, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v28;
  TASSIGN(v28, v24);
  Tile<TileType::Left, bfloat16_t, 32, 32, BLayout::RowMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v29;
  TASSIGN(v29, v24);
  Tile<TileType::Right, bfloat16_t, 32, 64, BLayout::RowMajor, 32, 64, SLayout::ColMajor, 512, PadValue::Null> v30;
  TASSIGN(v30, v24);
  Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 1024, PadValue::Null> v31;
  TASSIGN(v31, v24);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v32 = (size_t) ((int32_t) (int64_t) v25); v32 < ((size_t) v17); v32 += (size_t) ((int32_t) (int64_t) v26)) {
    int32_t v33 = (int32_t) v32;
    bool v34 = v33 == v18;
    bool v35 = v33 == v19;
    bool v36 = v33 == v16;
    bool v37 = v33 == v20;
    bool v38 = v33 == v21;
    bool v39 = v33 == v22;
    int32_t v40 = (int32_t) ((uint32_t) (v39 ? v19 : v38 ? v19 : (v37 ? v18 : v36 ? v18 : (v35 ? v12 : v34 ? v12 : v11))) * (uint32_t) v14);
    int32_t v41 = (int32_t) ((uint32_t) (v39 ? v12 : v38 ? v11 : (v37 ? v12 : v36 ? v11 : (v35 ? v12 : v34 ? v11 : (v33 == v12 ? v12 : v11)))) * (uint32_t) v15);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v42 = (size_t) v11; v42 < ((size_t) v16); v42 += (size_t) v12) {
      int32_t v43 = (int32_t) v42;
      int32_t v44 = (int32_t) ((uint32_t) v43 * (uint32_t) v14);
      pto::Shape<1, 1, 1, 32, 32> v45 = pto::Shape<1, 1, 1, 32, 32>();
      pto::Stride<4096, 4096, 4096, 128, 1> v46 = pto::Stride<4096, 4096, 4096, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v47 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v2 + (v10 + (unsigned) v40 * (unsigned) v13 + (unsigned) v44 * (unsigned) v12), v45, v46);
      pto::Shape<1, 1, 1, 32, 64> v48 = pto::Shape<1, 1, 1, 32, 64>();
      pto::Stride<4096, 4096, 4096, 128, 1> v49 = pto::Stride<4096, 4096, 4096, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v50 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v3 + (v10 + (unsigned) v44 * (unsigned) v13 + (unsigned) v41 * (unsigned) v12), v48, v49);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v27, v47);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v28, v50);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v29, v27);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v30, v28);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v43 == v11) {
        TMATMUL(v31, v29, v30);
      } else {
        TMATMUL_ACC(v31, v31, v29, v30);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 32, 64> v51 = pto::Shape<1, 1, 1, 32, 64>();
    pto::Stride<4096, 4096, 4096, 128, 1> v52 = pto::Stride<4096, 4096, 4096, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v53 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v1 + (v10 + (unsigned) v40 * (unsigned) v13 + (unsigned) v41 * (unsigned) v12), v51, v52);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v53, v31);
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

