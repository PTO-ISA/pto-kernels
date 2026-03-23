#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void _stage(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3) {
  unsigned v4 = 24576;
  unsigned v5 = 384;
  unsigned v6 = 8192;
  unsigned v7 = 128;
  unsigned v8 = 64;
  unsigned v9 = 1;
  unsigned v10 = 0;
  int32_t v11 = 3;
  int32_t v12 = 2;
  int32_t v13 = 384;
  int32_t v14 = 128;
  int32_t v15 = 64;
  int32_t v16 = 1;
  int32_t v17 = 0;
  int64_t v18 = 0;
  int64_t v19 = 8192;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v20 = get_block_idx();
  int64_t v21 = get_block_num();
  Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v22;
  TASSIGN(v22, v18);
  Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v23;
  __cbuf__ half* v24 = v22.data();
  uint64_t v25 = reinterpret_cast<uint64_t>(v24);
  TASSIGN(v23, v25);
  Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null> v26;
  TASSIGN(v26, v19);
  Tile<TileType::Mat, half, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 512, PadValue::Null> v27;
  __cbuf__ half* v28 = v26.data();
  uint64_t v29 = reinterpret_cast<uint64_t>(v28);
  TASSIGN(v27, v29);
  Tile<TileType::Left, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v30;
  TASSIGN(v30, v18);
  Tile<TileType::Left, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> v31;
  __ca__ half* v32 = v30.data();
  uint64_t v33 = reinterpret_cast<uint64_t>(v32);
  TASSIGN(v31, v33);
  Tile<TileType::Right, half, 64, 128, BLayout::RowMajor, 64, 128, SLayout::ColMajor, 512, PadValue::Null> v34;
  TASSIGN(v34, v18);
  Tile<TileType::Right, half, 64, 128, BLayout::RowMajor, 64, 128, SLayout::ColMajor, 512, PadValue::Null> v35;
  __cb__ half* v36 = v34.data();
  uint64_t v37 = reinterpret_cast<uint64_t>(v36);
  TASSIGN(v35, v37);
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null> v38;
  TASSIGN(v38, v18);
  Tile<TileType::Acc, float, 64, 128, BLayout::ColMajor, 64, 128, SLayout::RowMajor, 1024, PadValue::Null> v39;
  __cc__ float* v40 = v38.data();
  uint64_t v41 = reinterpret_cast<uint64_t>(v40);
  TASSIGN(v39, v41);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (int32_t v42 = (int32_t) v20; v42 < v11; v42 += (int32_t) v21) {
    int32_t v43 = (int32_t) ((uint32_t) (v42 == v12 ? v12 : v42 == v16 ? v16 : v17) * (uint32_t) v14);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    for (int32_t v44 = v17; v44 < v12; v44 += v16) {
      int32_t v45 = (int32_t) ((uint32_t) v44 * (uint32_t) v15);
      pto::Shape<1, 1, 1, 64, 64> v46 = pto::Shape<1, 1, 1, 64, 64>();
      pto::Stride<8192, 8192, 8192, 128, 1> v47 = pto::Stride<8192, 8192, 8192, 128, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND> v48 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<8192, 8192, 8192, 128, 1>, pto::Layout::ND>(v2 + (v10 + v10 * (unsigned) v14 + (unsigned) v45 * (unsigned) v16), v46, v47);
      pto::Shape<1, 1, 1, 64, 128> v49 = pto::Shape<1, 1, 1, 64, 128>();
      pto::Stride<24576, 24576, 24576, 384, 1> v50 = pto::Stride<24576, 24576, 24576, 384, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND> v51 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND>(v3 + (v10 + (unsigned) v45 * (unsigned) v13 + (unsigned) v43 * (unsigned) v16), v49, v50);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v23, v48);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v27, v51);
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
      if (v44 == v17) {
        TMATMUL(v39, v31, v35);
      } else {
        TMATMUL_ACC(v39, v39, v31, v35);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 64, 128> v52 = pto::Shape<1, 1, 1, 64, 128>();
    pto::Stride<24576, 24576, 24576, 384, 1> v53 = pto::Stride<24576, 24576, 24576, 384, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND> v54 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<24576, 24576, 24576, 384, 1>, pto::Layout::ND>(v1 + (v10 + v10 * (unsigned) v13 + (unsigned) v43 * (unsigned) v16), v52, v53);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pipe_barrier(PIPE_FIX);
    TSTORE(v54, v39);
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

