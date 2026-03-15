#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void ffn_dense_relu_fp16(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3, __gm__ half* v4, __gm__ half* v5) {
  unsigned v6 = 8192;
  unsigned v7 = 256;
  unsigned v8 = 4096;
  unsigned v9 = 128;
  unsigned v10 = 32;
  unsigned v11 = 1;
  unsigned v12 = 0;
  int32_t v13 = 8;
  int32_t v14 = 4;
  int32_t v15 = 256;
  int32_t v16 = 128;
  int32_t v17 = 32;
  int32_t v18 = 1;
  int32_t v19 = 0;
  int64_t v20 = 0;
  int64_t v21 = 2048;
  int64_t v22 = 512;
  int64_t v23 = 18432;
  int64_t v24 = 20480;
  int64_t v25 = 16384;
  int64_t v26 = 32768;
  using T = float;
  size_t v27 = (size_t) v19;
  size_t v28 = (size_t) v18;

  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, half, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v29;
  TASSIGN(v29, v20);
  Tile<TileType::Mat, half, 32, 256, BLayout::ColMajor, 32, 256, SLayout::RowMajor, 512, PadValue::Null> v30;
  TASSIGN(v30, v21);
  Tile<TileType::Left, half, 32, 32, BLayout::RowMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v31;
  TASSIGN(v31, v20);
  Tile<TileType::Right, half, 32, 256, BLayout::RowMajor, 32, 256, SLayout::ColMajor, 512, PadValue::Null> v32;
  TASSIGN(v32, v20);
  Tile<TileType::Acc, float, 32, 256, BLayout::ColMajor, 32, 256, SLayout::RowMajor, 1024, PadValue::Null> v33;
  TASSIGN(v33, v20);
  for (size_t v34 = v27; v34 < ((size_t) v14); v34 += v28) {
    int32_t v35 = (int32_t) v34;
    int32_t v36 = (int32_t) ((uint32_t) v35 * (uint32_t) v17);
    pto::Shape<1, 1, 1, 32, 32> v37 = pto::Shape<1, 1, 1, 32, 32>();
    pto::Stride<4096, 4096, 4096, 128, 1> v38 = pto::Stride<4096, 4096, 4096, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v39 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v3 + (v12 + v12 * (unsigned) v16 + (unsigned) v36 * (unsigned) v18), v37, v38);
    pto::Shape<1, 1, 1, 32, 256> v40 = pto::Shape<1, 1, 1, 32, 256>();
    pto::Stride<8192, 8192, 8192, 256, 1> v41 = pto::Stride<8192, 8192, 8192, 256, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v42 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v4 + (v12 + (unsigned) v36 * (unsigned) v15 + v12 * (unsigned) v18), v40, v41);
    TLOAD(v29, v39);
    TLOAD(v30, v42);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(v31, v29);
    TMOV(v32, v30);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    if (v35 == v19) {
      TMATMUL(v33, v31, v32);
    } else {
      TMATMUL_ACC(v33, v33, v31, v32);
    };
    set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  pto::Shape<1, 1, 1, 32, 256> v43 = pto::Shape<1, 1, 1, 32, 256>();
  pto::Stride<8192, 8192, 8192, 256, 1> v44 = pto::Stride<8192, 8192, 8192, 256, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v45 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v2 + (v12 + v12 * (unsigned) v15 + v12 * (unsigned) v18), v43, v44);
  TSTORE(v45, v33);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  #endif // __DAV_CUBE__


  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, half, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null> v46;
  TASSIGN(v46, v20);
  Tile<TileType::Vec, half, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null> v47;
  TASSIGN(v47, v22);
  for (size_t v48 = v27; v48 < ((size_t) v17); v48 += v28) {
    pto::Shape<1, 1, 1, 1, 256> v49 = pto::Shape<1, 1, 1, 1, 256>();
    pto::Stride<256, 256, 256, 256, 1> v50 = pto::Stride<256, 256, 256, 256, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v51 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v2 + (v12 + (unsigned) ((int32_t) v48) * (unsigned) v15 + v12 * (unsigned) v18), v49, v50);
    TLOAD(v46, v51);
    TRELU(v47, v46);
    TSTORE(v51, v47);
  }
  #endif // __DAV_VEC__


  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, half, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v52;
  TASSIGN(v52, v23);
  Tile<TileType::Mat, half, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null> v53;
  TASSIGN(v53, v24);
  Tile<TileType::Left, half, 32, 32, BLayout::RowMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v54;
  TASSIGN(v54, v21);
  Tile<TileType::Right, half, 32, 128, BLayout::RowMajor, 32, 128, SLayout::ColMajor, 512, PadValue::Null> v55;
  TASSIGN(v55, v25);
  Tile<TileType::Acc, float, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 1024, PadValue::Null> v56;
  TASSIGN(v56, v26);
  for (size_t v57 = v27; v57 < ((size_t) v13); v57 += v28) {
    int32_t v58 = (int32_t) v57;
    int32_t v59 = (int32_t) ((uint32_t) v58 * (uint32_t) v17);
    pto::Shape<1, 1, 1, 32, 32> v60 = pto::Shape<1, 1, 1, 32, 32>();
    pto::Stride<8192, 8192, 8192, 256, 1> v61 = pto::Stride<8192, 8192, 8192, 256, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v62 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v2 + (v12 + v12 * (unsigned) v15 + (unsigned) v59 * (unsigned) v18), v60, v61);
    pto::Shape<1, 1, 1, 32, 128> v63 = pto::Shape<1, 1, 1, 32, 128>();
    pto::Stride<4096, 4096, 4096, 128, 1> v64 = pto::Stride<4096, 4096, 4096, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v65 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v5 + (v12 + (unsigned) v59 * (unsigned) v16 + v12 * (unsigned) v18), v63, v64);
    TLOAD(v52, v62);
    TLOAD(v53, v65);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TMOV(v54, v52);
    TMOV(v55, v53);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    if (v58 == v19) {
      TMATMUL(v56, v54, v55);
    } else {
      TMATMUL_ACC(v56, v56, v54, v55);
    };
    set_flag(PIPE_M, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID1);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  pto::Shape<1, 1, 1, 32, 128> v66 = pto::Shape<1, 1, 1, 32, 128>();
  pto::Stride<4096, 4096, 4096, 128, 1> v67 = pto::Stride<4096, 4096, 4096, 128, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v68 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v1 + (v12 + v12 * (unsigned) v16 + v12 * (unsigned) v18), v66, v67);
  TSTORE(v68, v56);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  #endif // __DAV_CUBE__

  return;
}

