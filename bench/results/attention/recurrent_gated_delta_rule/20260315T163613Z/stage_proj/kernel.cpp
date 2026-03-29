#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void recurrent_state_key_proj(__gm__ float* v1, __gm__ bfloat16_t* v2, __gm__ bfloat16_t* v3) {
  unsigned v4 = 256;
  unsigned v5 = 16;
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 0;
  int32_t v9 = 512;
  int32_t v10 = 16;
  int32_t v11 = 32;
  int32_t v12 = 1;
  int64_t v13 = 0;
  int64_t v14 = 512;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v15 = get_block_idx();
  int64_t v16 = get_block_num();
  int32_t v17 = (int32_t) v16;
  int32_t v18 = v11 / v17;
  int32_t v19 = v11 % v17 != v8 && v11 < v8 == v17 < v8 ? v18 + v12 : v18;
  int32_t v20 = (int32_t) ((uint32_t) ((int32_t) v15) * (uint32_t) v19);
  int32_t v21 = (int32_t) ((uint32_t) v20 + (uint32_t) v19);
  Tile<TileType::Mat, bfloat16_t, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null> v22;
  TASSIGN(v22, v13);
  Tile<TileType::Mat, bfloat16_t, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null> v23;
  __cbuf__ bfloat16_t* v24 = v22.data();
  uint64_t v25 = reinterpret_cast<uint64_t>(v24);
  TASSIGN(v23, v25);
  Tile<TileType::Left, bfloat16_t, 16, 16, BLayout::RowMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null> v26;
  TASSIGN(v26, v13);
  Tile<TileType::Left, bfloat16_t, 16, 16, BLayout::RowMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null> v27;
  __ca__ bfloat16_t* v28 = v26.data();
  uint64_t v29 = reinterpret_cast<uint64_t>(v28);
  TASSIGN(v27, v29);
  Tile<TileType::Mat, bfloat16_t, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor, 512, PadValue::Null> v30;
  TASSIGN(v30, v14);
  Tile<TileType::Mat, bfloat16_t, 16, 16, BLayout::ColMajor, 1, 16, SLayout::RowMajor, 512, PadValue::Null> v31;
  __cbuf__ bfloat16_t* v32 = v30.data();
  uint64_t v33 = reinterpret_cast<uint64_t>(v32);
  TASSIGN(v31, v33);
  Tile<TileType::Right, bfloat16_t, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor, 512, PadValue::Null> v34;
  TASSIGN(v34, v13);
  Tile<TileType::Right, bfloat16_t, 16, 16, BLayout::RowMajor, 1, 16, SLayout::ColMajor, 512, PadValue::Null> v35;
  __cb__ bfloat16_t* v36 = v34.data();
  uint64_t v37 = reinterpret_cast<uint64_t>(v36);
  TASSIGN(v35, v37);
  Tile<TileType::Acc, float, 1, 16, BLayout::ColMajor, 1, 16, SLayout::RowMajor, 1024, PadValue::Null> v38;
  TASSIGN(v38, v13);
  Tile<TileType::Acc, float, 1, 16, BLayout::ColMajor, 1, 16, SLayout::RowMajor, 1024, PadValue::Null> v39;
  __cc__ float* v40 = v38.data();
  uint64_t v41 = reinterpret_cast<uint64_t>(v40);
  TASSIGN(v39, v41);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  for (int32_t v42 = v20; v42 < ((uint32_t) v21 < (uint32_t) v11 ? v21 : v11); v42 += v12) {
    pto::Shape<1, 1, 1, 16, 16> v43 = pto::Shape<1, 1, 1, 16, 16>();
    pto::Stride<256, 256, 256, 16, 1> v44 = pto::Stride<256, 256, 256, 16, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND> v45 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND>(v2 + (v7 + (unsigned) ((int32_t) (uint32_t) v42 * (uint32_t) v10) * (unsigned) v10 + v7 * (unsigned) v12), v43, v44);
    pto::Shape<1, 1, 1, 1, 16> v46 = pto::Shape<1, 1, 1, 1, 16>();
    pto::Stride<16, 16, 16, 16, 1> v47 = pto::Stride<16, 16, 16, 16, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v48 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v3 + (v7 + (unsigned) v42 * (unsigned) v10 + v7 * (unsigned) v12), v46, v47);
    pto::Shape<1, 1, 1, 1, 16> v49 = pto::Shape<1, 1, 1, 1, 16>();
    pto::Stride<16, 16, 16, 16, 1> v50 = pto::Stride<16, 16, 16, 16, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v51 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v42 * (unsigned) v10 + v7 * (unsigned) v12), v49, v50);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v23, v45);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    TLOAD(v31, v48);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    TMOV(v27, v23);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TMOV(v35, v31);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    TGEMV(v39, v27, v35);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(v51, v39);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  #endif // __DAV_CUBE__

  return;
}

