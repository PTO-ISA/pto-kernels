#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void all_gather_matmul_dense(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 4096;
  unsigned v5 = 128;
  unsigned v6 = 8192;
  unsigned v7 = 256;
  unsigned v8 = 64;
  unsigned v9 = 32;
  unsigned v10 = 1;
  unsigned v11 = 0;
  int32_t v12 = 4;
  int32_t v13 = 32;
  int32_t v14 = 64;
  int32_t v15 = 256;
  int32_t v16 = 128;
  int32_t v17 = 1;
  int32_t v18 = 0;
  int32_t v19 = 2;
  int32_t v20 = 3;
  int64_t v21 = 0;
  int64_t v22 = 4096;
  using T = float;
  int64_t v23 = get_block_idx();
  int64_t v24 = get_block_num();

  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, half, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v25;
  TASSIGN(v25, v21);
  Tile<TileType::Mat, half, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v26;
  __cbuf__ half* v27 = v25.data();
  uint64_t v28 = reinterpret_cast<uint64_t>(v27);
  TASSIGN(v26, v28);
  Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null> v29;
  TASSIGN(v29, v22);
  Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null> v30;
  __cbuf__ half* v31 = v29.data();
  uint64_t v32 = reinterpret_cast<uint64_t>(v31);
  TASSIGN(v30, v32);
  Tile<TileType::Left, half, 32, 64, BLayout::RowMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v33;
  TASSIGN(v33, v21);
  Tile<TileType::Left, half, 32, 64, BLayout::RowMajor, 32, 64, SLayout::RowMajor, 512, PadValue::Null> v34;
  __ca__ half* v35 = v33.data();
  uint64_t v36 = reinterpret_cast<uint64_t>(v35);
  TASSIGN(v34, v36);
  Tile<TileType::Right, half, 64, 128, BLayout::RowMajor, 64, 128, SLayout::ColMajor, 512, PadValue::Null> v37;
  TASSIGN(v37, v21);
  Tile<TileType::Right, half, 64, 128, BLayout::RowMajor, 64, 128, SLayout::ColMajor, 512, PadValue::Null> v38;
  __cb__ half* v39 = v37.data();
  uint64_t v40 = reinterpret_cast<uint64_t>(v39);
  TASSIGN(v38, v40);
  Tile<TileType::Acc, float, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 1024, PadValue::Null> v41;
  TASSIGN(v41, v21);
  Tile<TileType::Acc, float, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 1024, PadValue::Null> v42;
  __cc__ float* v43 = v41.data();
  uint64_t v44 = reinterpret_cast<uint64_t>(v43);
  TASSIGN(v42, v44);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (int32_t v45 = (int32_t) v23; v45 < v12; v45 += (int32_t) v24) {
    bool v46 = v45 == v19;
    bool v47 = v45 == v20;
    int32_t v48 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) (v47 ? v17 : v46 ? v17 : v18) * (uint32_t) v14) + (uint32_t) ((int32_t) (uint32_t) (v47 ? v17 : v46 ? v18 : (v45 == v17 ? v17 : v18)) * (uint32_t) v13));
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int32_t v49 = v18; v49 < v12; v49 += v17) {
      int32_t v50 = (int32_t) ((uint32_t) v49 * (uint32_t) v14);
      pto::Shape<1, 1, 1, 32, 64> v51 = pto::Shape<1, 1, 1, 32, 64>();
      pto::Stride<8192, 8192, 8192, 256, 1> v52 = pto::Stride<8192, 8192, 8192, 256, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v53 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v2 + (v11 + (unsigned) v48 * (unsigned) v15 + (unsigned) v50 * (unsigned) v17), v51, v52);
      pto::Shape<1, 1, 1, 64, 128> v54 = pto::Shape<1, 1, 1, 64, 128>();
      pto::Stride<8192, 8192, 8192, 128, 1> v55 = pto::Stride<8192, 8192, 8192, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v56 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v3 + (v11 + (unsigned) v50 * (unsigned) v16 + v11 * (unsigned) v17), v54, v55);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v26, v53);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v30, v56);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v34, v26);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v38, v30);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v49 == v18) {
        TMATMUL(v42, v34, v38);
      } else {
        TMATMUL_ACC(v42, v42, v34, v38);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 32, 128> v57 = pto::Shape<1, 1, 1, 32, 128>();
    pto::Stride<4096, 4096, 4096, 128, 1> v58 = pto::Stride<4096, 4096, 4096, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v59 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v1 + (v11 + (unsigned) v48 * (unsigned) v16 + v11 * (unsigned) v17), v57, v58);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v59, v42);
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

