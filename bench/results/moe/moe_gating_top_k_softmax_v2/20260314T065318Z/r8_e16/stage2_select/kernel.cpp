#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void moe_gating_top_k_softmax_v2_select_stage(__gm__ half* v1, __gm__ int32_t* v2, __gm__ half* v3) {
  int32_t v4 = 16;
  int32_t v5 = 0;
  int32_t v6 = 1;
  int32_t v7 = 2;
  int32_t v8 = 3;
  int32_t v9 = 4;
  int32_t v10 = 5;
  int32_t v11 = 6;
  int32_t v12 = 7;
  int32_t v13 = 8;
  int32_t v14 = 9;
  int32_t v15 = 10;
  int32_t v16 = 11;
  int32_t v17 = 12;
  int32_t v18 = 13;
  int32_t v19 = 14;
  int32_t v20 = 15;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v21 = get_block_idx();
  int64_t v22 = get_block_num();
  int32_t v23 = (int32_t) ((int64_t) v22);
  int32_t v24 = v13 / v23;
  int32_t v25 = v13 % v23 != v5 && v13 < v5 == v23 < v5 ? v24 + v6 : v24;
  int32_t v26 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v21) * (uint32_t) v25);
  int32_t v27 = (int32_t) ((uint32_t) v26 + (uint32_t) v25);
  for (size_t v28 = (size_t) v26; v28 < ((size_t) ((uint32_t) v27 < (uint32_t) v13 ? v27 : v13)); v28 += (size_t) v6) {
    int32_t v29 = (int32_t) ((uint32_t) ((int32_t) v28) * (uint32_t) v4);
    half v30 = v3[v29];
    half v31 = v3[(int32_t) ((uint32_t) v29 + (uint32_t) v6)];
    bool v32 = v31 != v31 || v30 != v30;
    bool v33 = (v31 == v31 && v30 == v30) && v31 > v30;
    half v34 = v33 ? v31 : v30;
    half v35 = v3[(int32_t) ((uint32_t) v29 + (uint32_t) v7)];
    bool v36 = v35 != v35 || v34 != v34;
    bool v37 = (v35 == v35 && v34 == v34) && v35 > v34;
    half v38 = v37 ? v35 : v34;
    half v39 = v3[(int32_t) ((uint32_t) v29 + (uint32_t) v8)];
    bool v40 = v39 != v39 || v38 != v38;
    bool v41 = (v39 == v39 && v38 == v38) && v39 > v38;
    half v42 = v41 ? v39 : v38;
    half v43 = v3[(int32_t) ((uint32_t) v29 + (uint32_t) v9)];
    bool v44 = v43 != v43 || v42 != v42;
    bool v45 = (v43 == v43 && v42 == v42) && v43 > v42;
    half v46 = v45 ? v43 : v42;
    half v47 = v3[(int32_t) ((uint32_t) v29 + (uint32_t) v10)];
    bool v48 = v47 != v47 || v46 != v46;
    bool v49 = (v47 == v47 && v46 == v46) && v47 > v46;
    half v50 = v49 ? v47 : v46;
    half v51 = v3[(int32_t) ((uint32_t) v29 + (uint32_t) v11)];
    bool v52 = v51 != v51 || v50 != v50;
    bool v53 = (v51 == v51 && v50 == v50) && v51 > v50;
    half v54 = v53 ? v51 : v50;
    half v55 = v3[(int32_t) ((uint32_t) v29 + (uint32_t) v12)];
    bool v56 = v55 != v55 || v54 != v54;
    bool v57 = (v55 == v55 && v54 == v54) && v55 > v54;
    half v58 = v57 ? v55 : v54;
    half v59 = v3[(int32_t) ((uint32_t) v29 + (uint32_t) v13)];
    bool v60 = v59 != v59 || v58 != v58;
    bool v61 = (v59 == v59 && v58 == v58) && v59 > v58;
    half v62 = v61 ? v59 : v58;
    half v63 = v3[(int32_t) ((uint32_t) v29 + (uint32_t) v14)];
    bool v64 = v63 != v63 || v62 != v62;
    bool v65 = (v63 == v63 && v62 == v62) && v63 > v62;
    half v66 = v65 ? v63 : v62;
    half v67 = v3[(int32_t) ((uint32_t) v29 + (uint32_t) v15)];
    bool v68 = v67 != v67 || v66 != v66;
    bool v69 = (v67 == v67 && v66 == v66) && v67 > v66;
    half v70 = v69 ? v67 : v66;
    half v71 = v3[(int32_t) ((uint32_t) v29 + (uint32_t) v16)];
    bool v72 = v71 != v71 || v70 != v70;
    bool v73 = (v71 == v71 && v70 == v70) && v71 > v70;
    half v74 = v73 ? v71 : v70;
    half v75 = v3[(int32_t) ((uint32_t) v29 + (uint32_t) v17)];
    bool v76 = v75 != v75 || v74 != v74;
    bool v77 = (v75 == v75 && v74 == v74) && v75 > v74;
    half v78 = v77 ? v75 : v74;
    half v79 = v3[(int32_t) ((uint32_t) v29 + (uint32_t) v18)];
    bool v80 = v79 != v79 || v78 != v78;
    bool v81 = (v79 == v79 && v78 == v78) && v79 > v78;
    half v82 = v81 ? v79 : v78;
    half v83 = v3[(int32_t) ((uint32_t) v29 + (uint32_t) v19)];
    bool v84 = v83 != v83 || v82 != v82;
    bool v85 = (v83 == v83 && v82 == v82) && v83 > v82;
    half v86 = v85 ? v83 : v82;
    half v87 = v3[(int32_t) ((uint32_t) v29 + (uint32_t) v20)];
    bool v88 = v87 != v87 || v86 != v86;
    bool v89 = (v87 == v87 && v86 == v86) && v87 > v86;
    half v90 = v89 ? v87 : v86;
    int32_t v91 = v89 ? v20 : (v85 ? v19 : v81 ? v18 : (v77 ? v17 : v73 ? v16 : (v69 ? v15 : v65 ? v14 : (v61 ? v13 : v57 ? v12 : (v53 ? v11 : v49 ? v10 : (v45 ? v9 : v41 ? v8 : (v37 ? v7 : v33 ? v6 : v5)))))));
    v1[v28] = v90;
    v2[v28] = v91;
  }
  pipe_barrier(PIPE_ALL);
  #endif // __DAV_VEC__

  return;
}

