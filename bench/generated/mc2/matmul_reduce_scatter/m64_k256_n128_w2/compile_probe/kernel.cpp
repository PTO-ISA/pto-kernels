#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void matmul_reduce_scatter_local_mm(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 4096;
  unsigned v5 = 128;
  unsigned v6 = 8192;
  unsigned v7 = 256;
  unsigned v8 = 32;
  unsigned v9 = 1;
  unsigned v10 = 0;
  int32_t v11 = 2;
  int32_t v12 = 8;
  int32_t v13 = 32;
  int32_t v14 = 128;
  int32_t v15 = 256;
  int32_t v16 = 64;
  int32_t v17 = 1;
  int32_t v18 = 0;
  int64_t v19 = 0;
  int64_t v20 = 2048;
  using T = float;

  #if defined(__DAV_CUBE__)
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (int32_t v21 = v18; v21 < v11; v21 += v17) {
    for (int32_t v22 = v18; v22 < v17; v22 += v17) {
      int32_t v23 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) (v21 % v11) * (uint32_t) v13) + (uint32_t) ((int32_t) (uint32_t) v22 * (uint32_t) v13));
      Tile<TileType::Mat, half, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v24;
      TASSIGN(v24, v19);
      Tile<TileType::Mat, half, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v25;
      __cbuf__ half* v26 = v24.data();
      uint64_t v27 = reinterpret_cast<uint64_t>(v26);
      TASSIGN(v25, v27);
      Tile<TileType::Mat, half, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null> v28;
      TASSIGN(v28, v20);
      Tile<TileType::Mat, half, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null> v29;
      __cbuf__ half* v30 = v28.data();
      uint64_t v31 = reinterpret_cast<uint64_t>(v30);
      TASSIGN(v29, v31);
      Tile<TileType::Left, half, 32, 32, BLayout::RowMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v32;
      TASSIGN(v32, v19);
      Tile<TileType::Left, half, 32, 32, BLayout::RowMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v33;
      __ca__ half* v34 = v32.data();
      uint64_t v35 = reinterpret_cast<uint64_t>(v34);
      TASSIGN(v33, v35);
      Tile<TileType::Right, half, 32, 128, BLayout::RowMajor, 32, 128, SLayout::ColMajor, 512, PadValue::Null> v36;
      TASSIGN(v36, v19);
      Tile<TileType::Right, half, 32, 128, BLayout::RowMajor, 32, 128, SLayout::ColMajor, 512, PadValue::Null> v37;
      __cb__ half* v38 = v36.data();
      uint64_t v39 = reinterpret_cast<uint64_t>(v38);
      TASSIGN(v37, v39);
      Tile<TileType::Acc, float, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 1024, PadValue::Null> v40;
      TASSIGN(v40, v19);
      Tile<TileType::Acc, float, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 1024, PadValue::Null> v41;
      __cc__ float* v42 = v40.data();
      uint64_t v43 = reinterpret_cast<uint64_t>(v42);
      TASSIGN(v41, v43);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      for (int32_t v44 = v18; v44 < v12; v44 += v17) {
        int32_t v45 = (int32_t) ((uint32_t) v44 * (uint32_t) v13);
        pto::Shape<1, 1, 1, 32, 32> v46 = pto::Shape<1, 1, 1, 32, 32>();
        pto::Stride<8192, 8192, 8192, 256, 1> v47 = pto::Stride<8192, 8192, 8192, 256, 1>();
        GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v48 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v2 + (v10 + (unsigned) v23 * (unsigned) v15 + (unsigned) v45 * (unsigned) v17), v46, v47);
        pto::Shape<1, 1, 1, 32, 128> v49 = pto::Shape<1, 1, 1, 32, 128>();
        pto::Stride<4096, 4096, 4096, 128, 1> v50 = pto::Stride<4096, 4096, 4096, 128, 1>();
        GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v51 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v3 + (v10 + (unsigned) v45 * (unsigned) v14 + v10 * (unsigned) v17), v49, v50);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        TLOAD(v25, v48);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        TLOAD(v29, v51);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        pipe_barrier(PIPE_MTE1);
        TMOV(v33, v25);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        TMOV(v37, v29);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        if (v44 == v18) {
          TMATMUL(v41, v33, v37);
        } else {
          TMATMUL_ACC(v41, v41, v33, v37);
        };
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      };
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      pto::Shape<1, 1, 1, 32, 128> v52 = pto::Shape<1, 1, 1, 32, 128>();
      pto::Stride<4096, 4096, 4096, 128, 1> v53 = pto::Stride<4096, 4096, 4096, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v54 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v1 + (v10 + (unsigned) v23 * (unsigned) v14 + v10 * (unsigned) v17), v52, v53);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      pipe_barrier(PIPE_FIX);
      TSTORE(v54, v41);
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
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

