#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void grouped_matmul_dense_bf16_bf16(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3, int32_t v4) {
  unsigned v5 = 8192;
  unsigned v6 = 2048;
  unsigned v7 = 128;
  unsigned v8 = 64;
  unsigned v9 = 16;
  unsigned v10 = 1;
  unsigned v11 = 0;
  int32_t v12 = 0;
  int32_t v13 = 1;
  int32_t v14 = 128;
  int32_t v15 = 16;
  int32_t v16 = 64;
  int32_t v17 = 2;
  int64_t v18 = 0;
  int64_t v19 = 2048;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v20 = get_block_idx();
  int64_t v21 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v22;
  TASSIGN(v22, v18);
  Tile<TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v23;
  __cbuf__ bfloat16_t* v24 = v22.data();
  uint64_t v25 = reinterpret_cast<uint64_t>(v24);
  TASSIGN(v23, v25);
  Tile<TileType::Mat, bfloat16_t, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v26;
  TASSIGN(v26, v19);
  Tile<TileType::Mat, bfloat16_t, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v27;
  __cbuf__ bfloat16_t* v28 = v26.data();
  uint64_t v29 = reinterpret_cast<uint64_t>(v28);
  TASSIGN(v27, v29);
  Tile<TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v30;
  TASSIGN(v30, v18);
  Tile<TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v31;
  __ca__ bfloat16_t* v32 = v30.data();
  uint64_t v33 = reinterpret_cast<uint64_t>(v32);
  TASSIGN(v31, v33);
  Tile<TileType::Right, bfloat16_t, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v34;
  TASSIGN(v34, v18);
  Tile<TileType::Right, bfloat16_t, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v35;
  __cb__ bfloat16_t* v36 = v34.data();
  uint64_t v37 = reinterpret_cast<uint64_t>(v36);
  TASSIGN(v35, v37);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v38;
  TASSIGN(v38, v18);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v39;
  __cc__ float* v40 = v38.data();
  uint64_t v41 = reinterpret_cast<uint64_t>(v40);
  TASSIGN(v39, v41);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (int32_t v42 = (int32_t) v20; v42 < v15; v42 += (int32_t) v21) {
    int32_t v43 = (int32_t) ((uint32_t) (v42 / v17) * (uint32_t) v15);
    int32_t v44 = (int32_t) ((uint32_t) (v42 % v17) * (uint32_t) v16);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int32_t v45 = v12; v45 < v17; v45 += v13) {
      int32_t v46 = (int32_t) ((uint32_t) v45 * (uint32_t) v16);
      pto::Shape<1, 1, 1, 16, 64> v47 = pto::Shape<1, 1, 1, 16, 64>();
      pto::Stride<2048, 2048, 2048, 128, 1> v48 = pto::Stride<2048, 2048, 2048, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v49 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v2 + (v11 + (unsigned) v43 * (unsigned) v14 + (unsigned) v46 * (unsigned) v13), v47, v48);
      pto::Shape<1, 1, 1, 64, 64> v50 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<8192, 8192, 8192, 128, 1> v51 = pto::Stride<8192, 8192, 8192, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v52 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v11 + (unsigned) v46 * (unsigned) v14 + (unsigned) v44 * (unsigned) v13), v50, v51);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v23, v49);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v27, v52);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v31, v23);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v35, v27);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v45 == v12) {
        TMATMUL(v39, v31, v35);
      } else {
        TMATMUL_ACC(v39, v39, v31, v35);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 64> v53 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<2048, 2048, 2048, 128, 1> v54 = pto::Stride<2048, 2048, 2048, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v55 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v1 + (v11 + (unsigned) v43 * (unsigned) v14 + (unsigned) v44 * (unsigned) v13), v53, v54);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v55, v39);
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

