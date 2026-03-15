#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void dense_attention_pv_stage(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 2048;
  unsigned v5 = 4096;
  unsigned v6 = 128;
  unsigned v7 = 512;
  unsigned v8 = 32;
  unsigned v9 = 16;
  unsigned v10 = 1;
  unsigned v11 = 0;
  int32_t v12 = 8;
  int32_t v13 = 16;
  int32_t v14 = 128;
  int32_t v15 = 32;
  int32_t v16 = 1;
  int32_t v17 = 0;
  int32_t v18 = 2;
  int32_t v19 = 3;
  int32_t v20 = 4;
  int32_t v21 = 5;
  int32_t v22 = 6;
  int32_t v23 = 7;
  int64_t v24 = 0;
  int64_t v25 = 1024;
  using T = float;
  size_t v26 = (size_t) v16;

  #if defined(__DAV_CUBE__)
  int64_t v27 = get_block_idx();
  int64_t v28 = get_block_num();
  Tile<TileType::Mat, half, 16, 32, BLayout::ColMajor, 16, 32, SLayout::RowMajor, 512, PadValue::Null> v29;
  TASSIGN(v29, v24);
  Tile<TileType::Mat, half, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v30;
  TASSIGN(v30, v25);
  Tile<TileType::Left, half, 16, 32, BLayout::RowMajor, 16, 32, SLayout::RowMajor, 512, PadValue::Null> v31;
  TASSIGN(v31, v24);
  Tile<TileType::Right, half, 32, 32, BLayout::RowMajor, 32, 32, SLayout::ColMajor, 512, PadValue::Null> v32;
  TASSIGN(v32, v24);
  Tile<TileType::Acc, float, 16, 32, BLayout::ColMajor, 16, 32, SLayout::RowMajor, 1024, PadValue::Null> v33;
  TASSIGN(v33, v24);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v34 = (size_t) ((int32_t) (int64_t) v27); v34 < ((size_t) v12); v34 += (size_t) ((int32_t) (int64_t) v28)) {
    int32_t v35 = (int32_t) v34;
    bool v36 = v35 == v20;
    bool v37 = v35 == v21;
    bool v38 = v35 == v22;
    bool v39 = v35 == v23;
    int32_t v40 = (int32_t) ((uint32_t) (v39 ? v16 : v38 ? v16 : (v37 ? v16 : v36 ? v16 : v17)) * (uint32_t) v13);
    int32_t v41 = (int32_t) ((uint32_t) (v39 ? v19 : v38 ? v18 : (v37 ? v16 : v36 ? v17 : (v35 == v19 ? v19 : v35 == v18 ? v18 : (v35 == v16 ? v16 : v17)))) * (uint32_t) v15);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v42 = (size_t) v17; v42 < v26; v42 += v26) {
      int32_t v43 = (int32_t) v42;
      int32_t v44 = (int32_t) ((uint32_t) v43 * (uint32_t) v15);
      pto::Shape<1, 1, 1, 16, 32> v45 = pto::Shape<1, 1, 1, 16, 32>();
      pto::Stride<512, 512, 512, 32, 1> v46 = pto::Stride<512, 512, 512, 32, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<512, 512, 512, 32, 1>, pto::Layout::ND> v47 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<512, 512, 512, 32, 1>, pto::Layout::ND>(v2 + (v11 + (unsigned) v40 * (unsigned) v15 + (unsigned) v44 * (unsigned) v16), v45, v46);
      pto::Shape<1, 1, 1, 32, 32> v48 = pto::Shape<1, 1, 1, 32, 32>();
      pto::Stride<4096, 4096, 4096, 128, 1> v49 = pto::Stride<4096, 4096, 4096, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v50 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v3 + (v11 + (unsigned) v44 * (unsigned) v14 + (unsigned) v41 * (unsigned) v16), v48, v49);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v29, v47);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v30, v50);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v31, v29);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v32, v30);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v43 == v17) {
        TMATMUL(v33, v31, v32);
      } else {
        TMATMUL_ACC(v33, v33, v31, v32);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 32> v51 = pto::Shape<1, 1, 1, 16, 32>();
    pto::Stride<2048, 2048, 2048, 128, 1> v52 = pto::Stride<2048, 2048, 2048, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v53 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 32>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v1 + (v11 + (unsigned) v40 * (unsigned) v14 + (unsigned) v41 * (unsigned) v16), v51, v52);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v53, v33);
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

