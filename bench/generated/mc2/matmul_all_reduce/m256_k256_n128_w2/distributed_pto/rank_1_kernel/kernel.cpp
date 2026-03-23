#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void matmul_all_reduce_local(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 4096;
  unsigned v5 = 128;
  unsigned v6 = 8192;
  unsigned v7 = 256;
  unsigned v8 = 64;
  unsigned v9 = 32;
  unsigned v10 = 1;
  unsigned v11 = 0;
  int32_t v12 = 8;
  int32_t v13 = 4;
  int32_t v14 = 64;
  int32_t v15 = 32;
  int32_t v16 = 128;
  int32_t v17 = 256;
  int32_t v18 = 1;
  int32_t v19 = 0;
  int64_t v20 = 0;
  int64_t v21 = 4096;
  using T = float;
  int64_t v22 = get_block_idx();
  int32_t v23 = (int32_t) v22;

  #if defined(__DAV_CUBE__)
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  if (v23 < v13) {
    Tile<TileType::Mat, half, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v24;
    TASSIGN(v24, v20);
    Tile<TileType::Mat, half, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v25;
    __cbuf__ half* v26 = v24.data();
    uint64_t v27 = reinterpret_cast<uint64_t>(v26);
    TASSIGN(v25, v27);
    Tile<TileType::Mat, half, 64, 32, BLayout::ColMajor, 64, 32, SLayout::RowMajor, 512, PadValue::Null> v28;
    TASSIGN(v28, v21);
    Tile<TileType::Mat, half, 64, 32, BLayout::ColMajor, 64, 32, SLayout::RowMajor, 512, PadValue::Null> v29;
    __cbuf__ half* v30 = v28.data();
    uint64_t v31 = reinterpret_cast<uint64_t>(v30);
    TASSIGN(v29, v31);
    Tile<TileType::Left, half, 32, 64, BLayout::RowMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v32;
    TASSIGN(v32, v20);
    Tile<TileType::Left, half, 32, 64, BLayout::RowMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v33;
    __ca__ half* v34 = v32.data();
    uint64_t v35 = reinterpret_cast<uint64_t>(v34);
    TASSIGN(v33, v35);
    Tile<TileType::Right, half, 64, 32, BLayout::RowMajor, 64, 32, SLayout::ColMajor, 512, PadValue::Null> v36;
    TASSIGN(v36, v20);
    Tile<TileType::Right, half, 64, 32, BLayout::RowMajor, 64, 32, SLayout::ColMajor, 512, PadValue::Null> v37;
    __cb__ half* v38 = v36.data();
    uint64_t v39 = reinterpret_cast<uint64_t>(v38);
    TASSIGN(v37, v39);
    Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 1024, PadValue::Null> v40;
    TASSIGN(v40, v20);
    Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 1024, PadValue::Null> v41;
    __cc__ float* v42 = v40.data();
    uint64_t v43 = reinterpret_cast<uint64_t>(v42);
    TASSIGN(v41, v43);
    for (int32_t v44 = v19; v44 < v12; v44 += v18) {
      int32_t v45 = (int32_t) ((uint32_t) v44 + (uint32_t) (v23 / v13));
      if (v45 < v12) {
        int32_t v46 = (int32_t) ((uint32_t) v45 * (uint32_t) v15);
        int32_t v47 = (int32_t) ((uint32_t) (v23 % v13) * (uint32_t) v15);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        for (int32_t v48 = v19; v48 < v13; v48 += v18) {
          int32_t v49 = (int32_t) ((uint32_t) v48 * (uint32_t) v14);
          pto::Shape<1, 1, 1, 32, 64> v50 = pto::Shape<1, 1, 1, 32, 64>();
          pto::Stride<8192, 8192, 8192, 256, 1> v51 = pto::Stride<8192, 8192, 8192, 256, 1>();
          GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v52 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v2 + (v11 + (unsigned) v46 * (unsigned) v17 + (unsigned) v49 * (unsigned) v18), v50, v51);
          pto::Shape<1, 1, 1, 64, 32> v53 = pto::Shape<1, 1, 1, 64, 32>();
          pto::Stride<8192, 8192, 8192, 128, 1> v54 = pto::Stride<8192, 8192, 8192, 128, 1>();
          GlobalTensor<half, pto::Shape<1, 1, 1, 64, 32>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v55 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 32>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v11 + (unsigned) v49 * (unsigned) v16 + (unsigned) v47 * (unsigned) v18), v53, v54);
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
          TLOAD(v25, v52);
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
          TLOAD(v29, v55);
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
          wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
          pipe_barrier(PIPE_MTE1);
          TMOV(v33, v25);
          set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
          TMOV(v37, v29);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          if (v48 == v19) {
            TMATMUL(v41, v33, v37);
          } else {
            TMATMUL_ACC(v41, v41, v33, v37);
          };
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        };
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        pto::Shape<1, 1, 1, 32, 32> v56 = pto::Shape<1, 1, 1, 32, 32>();
        pto::Stride<4096, 4096, 4096, 128, 1> v57 = pto::Stride<4096, 4096, 4096, 128, 1>();
        GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v58 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v1 + (v11 + (unsigned) v46 * (unsigned) v16 + (unsigned) v47 * (unsigned) v18), v56, v57);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        pipe_barrier(PIPE_FIX);
        TSTORE(v58, v41);
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

