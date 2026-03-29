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
  int32_t v18 = 8;
  int64_t v19 = 0;
  int64_t v20 = 2048;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v21 = get_block_idx();
  int64_t v22 = get_block_num();
  Tile<TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v23;
  TASSIGN(v23, v19);
  Tile<TileType::Mat, bfloat16_t, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v24;
  __cbuf__ bfloat16_t* v25 = v23.data();
  uint64_t v26 = reinterpret_cast<uint64_t>(v25);
  TASSIGN(v24, v26);
  Tile<TileType::Mat, bfloat16_t, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v27;
  TASSIGN(v27, v20);
  Tile<TileType::Mat, bfloat16_t, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v28;
  __cbuf__ bfloat16_t* v29 = v27.data();
  uint64_t v30 = reinterpret_cast<uint64_t>(v29);
  TASSIGN(v28, v30);
  Tile<TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v31;
  TASSIGN(v31, v19);
  Tile<TileType::Left, bfloat16_t, 16, 64, BLayout::RowMajor, 16, 64, SLayout::RowMajor, 512, PadValue::Null> v32;
  __ca__ bfloat16_t* v33 = v31.data();
  uint64_t v34 = reinterpret_cast<uint64_t>(v33);
  TASSIGN(v32, v34);
  Tile<TileType::Right, bfloat16_t, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v35;
  TASSIGN(v35, v19);
  Tile<TileType::Right, bfloat16_t, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null> v36;
  __cb__ bfloat16_t* v37 = v35.data();
  uint64_t v38 = reinterpret_cast<uint64_t>(v37);
  TASSIGN(v36, v38);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v39;
  TASSIGN(v39, v19);
  Tile<TileType::Acc, float, 16, 64, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 1024, PadValue::Null> v40;
  __cc__ float* v41 = v39.data();
  uint64_t v42 = reinterpret_cast<uint64_t>(v41);
  TASSIGN(v40, v42);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (int32_t v43 = (int32_t) v21; v43 < v15; v43 += (int32_t) v22) {
    int32_t v44 = v43 / v15;
    int32_t v45 = v43 % v15;
    int32_t v46 = (int32_t) ((uint32_t) v44 * (uint32_t) v17);
    int32_t v47 = v44 == v12 ? (int32_t) ((uint32_t) v17 - (uint32_t) v46) : v17;
    int32_t v48 = v45 / v47;
    int32_t v49 = (int32_t) ((uint32_t) (v44 % v17 == v13 ? (int32_t) ((uint32_t) ((int32_t) (uint32_t) v18 - (uint32_t) v48) - (uint32_t) v13) : v48) * (uint32_t) v15);
    int32_t v50 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v46 + (uint32_t) (v45 % v47)) * (uint32_t) v16);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int32_t v51 = v12; v51 < v17; v51 += v13) {
      int32_t v52 = (int32_t) ((uint32_t) v51 * (uint32_t) v16);
      pto::Shape<1, 1, 1, 16, 64> v53 = pto::Shape<1, 1, 1, 16, 64>();
      pto::Stride<2048, 2048, 2048, 128, 1> v54 = pto::Stride<2048, 2048, 2048, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v55 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v2 + (v11 + (unsigned) v49 * (unsigned) v14 + (unsigned) v52 * (unsigned) v13), v53, v54);
      pto::Shape<1, 1, 1, 64, 64> v56 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<8192, 8192, 8192, 128, 1> v57 = pto::Stride<8192, 8192, 8192, 128, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v58 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v11 + (unsigned) v52 * (unsigned) v14 + (unsigned) v50 * (unsigned) v13), v56, v57);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v24, v55);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v28, v58);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v32, v24);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v36, v28);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v51 == v12) {
        TMATMUL(v40, v32, v36);
      } else {
        TMATMUL_ACC(v40, v40, v32, v36);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 16, 64> v59 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<2048, 2048, 2048, 128, 1> v60 = pto::Stride<2048, 2048, 2048, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND> v61 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<2048, 2048, 2048, 128, 1>, pto::Layout::ND>(v1 + (v11 + (unsigned) v49 * (unsigned) v14 + (unsigned) v50 * (unsigned) v13), v59, v60);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v61, v40);
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

