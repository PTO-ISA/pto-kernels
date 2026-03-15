#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_gating_top_k_softmax_v2_select_stage(__gm__ half* v1, __gm__ int32_t* v2, __gm__ half* v3, __gm__ half* v4) {
  int32_t v5 = 16;
  int32_t v6 = 0;
  int32_t v7 = 1;
  int32_t v8 = 2;
  int32_t v9 = 3;
  int32_t v10 = 4;
  int32_t v11 = 5;
  int32_t v12 = 6;
  int32_t v13 = 7;
  int32_t v14 = 8;
  int32_t v15 = 9;
  int32_t v16 = 10;
  int32_t v17 = 11;
  int32_t v18 = 12;
  int32_t v19 = 13;
  int32_t v20 = 14;
  int32_t v21 = 15;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v22 = get_block_idx();
  int64_t v23 = get_block_num();
  int32_t v24 = (int32_t) ((int64_t) v23);
  int32_t v25 = v14 / v24;
  int32_t v26 = v14 % v24 != v6 && v14 < v6 == v24 < v6 ? v25 + v7 : v25;
  int32_t v27 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v22) * (uint32_t) v26);
  int32_t v28 = (int32_t) ((uint32_t) v27 + (uint32_t) v26);
  for (size_t v29 = (size_t) v27; v29 < ((size_t) ((uint32_t) v28 < (uint32_t) v14 ? v28 : v14)); v29 += (size_t) v7) {
    int32_t v30 = (int32_t) ((uint32_t) ((int32_t) v29) * (uint32_t) v5);
    half v31 = v4[v30];
    half v32 = v4[(int32_t) ((uint32_t) v30 + (uint32_t) v7)];
    bool v33 = v32 != v32 || v31 != v31;
    bool v34 = (v32 == v32 && v31 == v31) && v32 > v31;
    half v35 = v34 ? v32 : v31;
    half v36 = v4[(int32_t) ((uint32_t) v30 + (uint32_t) v8)];
    bool v37 = v36 != v36 || v35 != v35;
    bool v38 = (v36 == v36 && v35 == v35) && v36 > v35;
    half v39 = v38 ? v36 : v35;
    half v40 = v4[(int32_t) ((uint32_t) v30 + (uint32_t) v9)];
    bool v41 = v40 != v40 || v39 != v39;
    bool v42 = (v40 == v40 && v39 == v39) && v40 > v39;
    half v43 = v42 ? v40 : v39;
    half v44 = v4[(int32_t) ((uint32_t) v30 + (uint32_t) v10)];
    bool v45 = v44 != v44 || v43 != v43;
    bool v46 = (v44 == v44 && v43 == v43) && v44 > v43;
    half v47 = v46 ? v44 : v43;
    half v48 = v4[(int32_t) ((uint32_t) v30 + (uint32_t) v11)];
    bool v49 = v48 != v48 || v47 != v47;
    bool v50 = (v48 == v48 && v47 == v47) && v48 > v47;
    half v51 = v50 ? v48 : v47;
    half v52 = v4[(int32_t) ((uint32_t) v30 + (uint32_t) v12)];
    bool v53 = v52 != v52 || v51 != v51;
    bool v54 = (v52 == v52 && v51 == v51) && v52 > v51;
    half v55 = v54 ? v52 : v51;
    half v56 = v4[(int32_t) ((uint32_t) v30 + (uint32_t) v13)];
    bool v57 = v56 != v56 || v55 != v55;
    bool v58 = (v56 == v56 && v55 == v55) && v56 > v55;
    half v59 = v58 ? v56 : v55;
    half v60 = v4[(int32_t) ((uint32_t) v30 + (uint32_t) v14)];
    bool v61 = v60 != v60 || v59 != v59;
    bool v62 = (v60 == v60 && v59 == v59) && v60 > v59;
    half v63 = v62 ? v60 : v59;
    half v64 = v4[(int32_t) ((uint32_t) v30 + (uint32_t) v15)];
    bool v65 = v64 != v64 || v63 != v63;
    bool v66 = (v64 == v64 && v63 == v63) && v64 > v63;
    half v67 = v66 ? v64 : v63;
    half v68 = v4[(int32_t) ((uint32_t) v30 + (uint32_t) v16)];
    bool v69 = v68 != v68 || v67 != v67;
    bool v70 = (v68 == v68 && v67 == v67) && v68 > v67;
    half v71 = v70 ? v68 : v67;
    half v72 = v4[(int32_t) ((uint32_t) v30 + (uint32_t) v17)];
    bool v73 = v72 != v72 || v71 != v71;
    bool v74 = (v72 == v72 && v71 == v71) && v72 > v71;
    half v75 = v74 ? v72 : v71;
    half v76 = v4[(int32_t) ((uint32_t) v30 + (uint32_t) v18)];
    bool v77 = v76 != v76 || v75 != v75;
    bool v78 = (v76 == v76 && v75 == v75) && v76 > v75;
    half v79 = v78 ? v76 : v75;
    half v80 = v4[(int32_t) ((uint32_t) v30 + (uint32_t) v19)];
    bool v81 = v80 != v80 || v79 != v79;
    bool v82 = (v80 == v80 && v79 == v79) && v80 > v79;
    half v83 = v82 ? v80 : v79;
    half v84 = v4[(int32_t) ((uint32_t) v30 + (uint32_t) v20)];
    bool v85 = v84 != v84 || v83 != v83;
    bool v86 = (v84 == v84 && v83 == v83) && v84 > v83;
    half v87 = v86 ? v84 : v83;
    half v88 = v4[(int32_t) ((uint32_t) v30 + (uint32_t) v21)];
    bool v89 = v88 != v88 || v87 != v87;
    int32_t v90 = (v88 == v88 && v87 == v87) && v88 > v87 ? v21 : (v86 ? v20 : v82 ? v19 : (v78 ? v18 : v74 ? v17 : (v70 ? v16 : v66 ? v15 : (v62 ? v14 : v58 ? v13 : (v54 ? v12 : v50 ? v11 : (v46 ? v10 : v42 ? v9 : (v38 ? v8 : v34 ? v7 : v6)))))));
    half v91 = v3[(int32_t) ((uint32_t) v30 + (uint32_t) v90)];
    v1[v29] = v91;
    v2[v29] = v90;
  }
  pipe_barrier(PIPE_ALL);
  #endif // __DAV_VEC__

  return;
}

