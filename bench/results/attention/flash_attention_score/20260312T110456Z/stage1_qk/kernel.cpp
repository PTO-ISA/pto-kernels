#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void dense_attention_qk_stage(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 512;
  unsigned v5 = 2048;
  unsigned v6 = 32;
  unsigned v7 = 1024;
  unsigned v8 = 64;
  unsigned v9 = 16;
  unsigned v10 = 1;
  unsigned v11 = 0;
  int32_t v12 = 4;
  int32_t v13 = 16;
  int32_t v14 = 64;
  int32_t v15 = 32;
  int32_t v16 = 1;
  int32_t v17 = 0;
  int32_t v18 = 2;
  int32_t v19 = 3;
  int64_t v20 = 2048;
  int64_t v21 = 0;
  using T = float;
  size_t v22 = (size_t) v16;

  #if defined(__DAV_CUBE__)
  int64_t v23 = get_block_idx();
  int64_t v24 = get_block_num();
  Tile<TileType::Mat, half, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v25;
  TASSIGN(v25, v20);
  Tile<TileType::Mat, half, 64, 16, BLayout::ColMajor, 64, 16, SLayout::RowMajor, 512, PadValue::Null> v26;
  TASSIGN(v26, v21);
  Tile<TileType::Left, half, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v27;
  TASSIGN(v27, v21);
  Tile<TileType::Right, half, 64, 16, BLayout::RowMajor, 64, 16, SLayout::ColMajor, 512, PadValue::Null> v28;
  TASSIGN(v28, v21);
  Tile<TileType::Acc, float, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 1024, PadValue::Null> v29;
  TASSIGN(v29, v21);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v30 = (size_t) ((int32_t) (int64_t) v23); v30 < ((size_t) v12); v30 += (size_t) ((int32_t) (int64_t) v24)) {
    int32_t v31 = (int32_t) v30;
    bool v32 = v31 == v18;
    bool v33 = v31 == v19;
    int32_t v34 = (int32_t) ((uint32_t) (v33 ? v16 : v32 ? v16 : v17) * (uint32_t) v13);
    int32_t v35 = (int32_t) ((uint32_t) (v33 ? v16 : v32 ? v17 : (v31 == v16 ? v16 : v17)) * (uint32_t) v13);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (size_t v36 = (size_t) v17; v36 < v22; v36 += v22) {
      int32_t v37 = (int32_t) v36;
      int32_t v38 = (int32_t) ((uint32_t) v37 * (uint32_t) v14);
      pto::Shape<1, 1, 1, 16, 64> v39 = pto::Shape<1, 1, 1, 16, 64>();
      pto::Stride<1024, 1024, 1024, 64, 1> v40 = pto::Stride<1024, 1024, 1024, 64, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v41 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(v2 + (v11 + (unsigned) v34 * (unsigned) v14 + (unsigned) v38 * (unsigned) v16), v39, v40);
      pto::Shape<1, 1, 1, 64, 16> v42 = pto::Shape<1, 1, 1, 64, 16>();
      pto::Stride<2048, 2048, 2048, 32, 1> v43 = pto::Stride<2048, 2048, 2048, 32, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 16>, pto::Stride<2048, 2048, 2048, 32, 1>, pto::Layout::ND> v44 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 16>, pto::Stride<2048, 2048, 2048, 32, 1>, pto::Layout::ND>(v3 + (v11 + (unsigned) v38 * (unsigned) v15 + (unsigned) v35 * (unsigned) v16), v42, v43);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v25, v41);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v26, v44);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v27, v25);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v28, v26);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v37 == v17) {
        TMATMUL(v29, v27, v28);
      } else {
        TMATMUL_ACC(v29, v29, v27, v28);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 16> v45 = pto::Shape<1, 1, 1, 16, 16>();
    pto::Stride<512, 512, 512, 32, 1> v46 = pto::Stride<512, 512, 512, 32, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<512, 512, 512, 32, 1>, pto::Layout::ND> v47 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<512, 512, 512, 32, 1>, pto::Layout::ND>(v1 + (v11 + (unsigned) v34 * (unsigned) v15 + (unsigned) v35 * (unsigned) v16), v45, v46);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v47, v29);
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

