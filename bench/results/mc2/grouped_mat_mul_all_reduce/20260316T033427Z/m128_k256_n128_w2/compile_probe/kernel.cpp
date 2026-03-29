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
  int64_t v20 = get_block_idx();
  int32_t v21 = (int32_t) v20;

  #if defined(__DAV_CUBE__)
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  if (v21 < v11) {
    Tile<TileType::Mat, half, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v22;
    TASSIGN(v22, v18);
    Tile<TileType::Mat, half, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v23;
    __cbuf__ half* v24 = v22.data();
    uint64_t v25 = reinterpret_cast<uint64_t>(v24);
    TASSIGN(v23, v25);
    Tile<TileType::Mat, half, 64, 32, BLayout::ColMajor, 64, 32, SLayout::RowMajor, 512, PadValue::Null> v26;
    TASSIGN(v26, v19);
    Tile<TileType::Mat, half, 64, 32, BLayout::ColMajor, 64, 32, SLayout::RowMajor, 512, PadValue::Null> v27;
    __cbuf__ half* v28 = v26.data();
    uint64_t v29 = reinterpret_cast<uint64_t>(v28);
    TASSIGN(v27, v29);
    Tile<TileType::Left, half, 32, 64, BLayout::RowMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v30;
    TASSIGN(v30, v18);
    Tile<TileType::Left, half, 32, 64, BLayout::RowMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v31;
    __ca__ half* v32 = v30.data();
    uint64_t v33 = reinterpret_cast<uint64_t>(v32);
    TASSIGN(v31, v33);
    Tile<TileType::Right, half, 64, 32, BLayout::RowMajor, 64, 32, SLayout::ColMajor, 512, PadValue::Null> v34;
    TASSIGN(v34, v18);
    Tile<TileType::Right, half, 64, 32, BLayout::RowMajor, 64, 32, SLayout::ColMajor, 512, PadValue::Null> v35;
    __cb__ half* v36 = v34.data();
    uint64_t v37 = reinterpret_cast<uint64_t>(v36);
    TASSIGN(v35, v37);
    Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 1024, PadValue::Null> v38;
    TASSIGN(v38, v18);
    Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 1024, PadValue::Null> v39;
    __cc__ float* v40 = v38.data();
    uint64_t v41 = reinterpret_cast<uint64_t>(v40);
    TASSIGN(v39, v41);
    for (int32_t v42 = v17; v42 < v11; v42 += v16) {
      int32_t v43 = (int32_t) ((uint32_t) v42 + (uint32_t) (v21 / v11));
      if (v43 < v11) {
        int32_t v44 = (int32_t) ((uint32_t) v43 * (uint32_t) v14);
        int32_t v45 = (int32_t) ((uint32_t) (v21 % v11) * (uint32_t) v14);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        for (int32_t v46 = v17; v46 < v12; v46 += v16) {
          int32_t v47 = (int32_t) ((uint32_t) v46 * (uint32_t) v13);
          pto::Shape<1, 1, 1, 32, 64> v48 = pto::Shape<1, 1, 1, 32, 64>();
          pto::Stride<4096, 4096, 4096, 128, 1> v49 = pto::Stride<4096, 4096, 4096, 128, 1>();
          GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v50 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v2 + (v10 + (unsigned) v44 * (unsigned) v15 + (unsigned) v47 * (unsigned) v16), v48, v49);
          pto::Shape<1, 1, 1, 64, 32> v51 = pto::Shape<1, 1, 1, 64, 32>();
          pto::Stride<8192, 8192, 8192, 128, 1> v52 = pto::Stride<8192, 8192, 8192, 128, 1>();
          GlobalTensor<half, pto::Shape<1, 1, 1, 64, 32>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v53 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 32>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v10 + (unsigned) v47 * (unsigned) v15 + (unsigned) v45 * (unsigned) v16), v51, v52);
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
          TLOAD(v23, v50);
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
          TLOAD(v27, v53);
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
          if (v46 == v17) {
            TMATMUL(v39, v31, v35);
          } else {
            TMATMUL_ACC(v39, v39, v31, v35);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        };
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        pto::Shape<1, 1, 1, 32, 32> v54 = pto::Shape<1, 1, 1, 32, 32>();
        pto::Stride<4096, 4096, 4096, 128, 1> v55 = pto::Stride<4096, 4096, 4096, 128, 1>();
        GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v56 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v1 + (v10 + (unsigned) v44 * (unsigned) v15 + (unsigned) v45 * (unsigned) v16), v54, v55);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        pipe_barrier(PIPE_FIX);
        TSTORE(v56, v39);
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

