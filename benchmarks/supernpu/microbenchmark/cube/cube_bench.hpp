#ifndef CUBE_BENCH_HPP
#define CUBE_BENCH_HPP

// Named CUBE direct-operation micro-bench templates using v0.58 API names:
//   TMATMUL / TMATMUL_ACC / TMATMUL_BIAS / TMATMUL_MX
//   TGEMV  / TGEMV_ACC  / TGEMV_BIAS  / TGEMV_MX

#include <common/pto_tileop.hpp>
#include <cstdint>
#include "benchmark.h"
#include "bench_utils.hpp"

using namespace pto;

template <typename D, int M, int K>
using gmA_t = global_tensor<D, RowMajor<M, K>>;
template <typename D, int K, int N>
using gmB_t = global_tensor<D, RowMajor<K, N>>;
template <typename D, int M, int N>
using gmC_t = global_tensor<D, RowMajor<M, N>>;

template <typename D>
struct cube_accumulator_element {
    using type = std::conditional_t<
        std::is_same_v<D, __half> || std::is_floating_point_v<D>, float,
        std::conditional_t<std::is_signed_v<D>, int32_t, uint32_t>>;
};

template <typename D>
using cube_accumulator_element_t = typename cube_accumulator_element<D>::type;

template <typename D, int M, int K>
using tL_t = std::conditional_t<(M <= 16), CubeTileM16<D, M, K>,
                                CubeTileM32<D, M, K>>;
template <typename D, int K, int N>
using tR_t = CubeTileN8<D, K, N>;
template <typename D, int M, int N>
using tAcc_t = std::conditional_t<
    (M <= 16), CubeAccumulatorM16<cube_accumulator_element_t<D>, M, N>,
    CubeAccumulatorM32<cube_accumulator_element_t<D>, M, N>>;
template <typename D, int M, int N>
using tOut_t = std::conditional_t<(M <= 16), CubeAccumulatorM16<D, M, N>,
                                  CubeAccumulatorM32<D, M, N>>;
template <typename D, int N>
using tBias_t = Tile<Location::Vec, cube_accumulator_element_t<D>, 1, N,
                       BLayout::RowMajor>;

// C = A * B   (TMATMUL -> Tile -> GM)
template <typename D, int M, int N, int K>
void bench_matmul(D *c, D *a, D *b) {
    using itA = global_iterator<gmA_t<D, M, K>, tL_t<D, M, K>>;
    using itB = global_iterator<gmB_t<D, K, N>, tR_t<D, K, N>>;
    using itC = global_iterator<gmC_t<D, M, N>, tOut_t<D, M, N>>;
    itA gA(a); itB gB(b); itC gC(c);
    auto gA0 = gA(0, 0), gB0 = gB(0, 0), gC0 = gC(0, 0);
    tL_t<D, M, K> tA; tR_t<D, K, N> tB;
    tOut_t<D, M, N> tOut;
    TLOAD_CUBE(tA, gA0);
    TLOAD_CUBE(tB, gB0);
    if constexpr (std::is_same_v<D, __half>) {
        TMATMUL<FixpAttr::f16()>(tOut, tA, tB);
    } else {
        TMATMUL(tOut, tA, tB);
    }
    TSTORE_CUBE(gC0, tOut);
}

// C += A * B  (TMATMUL_ACC accumulate into ACC)
template <typename D, int M, int N, int K>
void bench_matmul_acc(D *c, D *a, D *b) {
    using itA = global_iterator<gmA_t<D, M, K>, tL_t<D, M, K>>;
    using itB = global_iterator<gmB_t<D, K, N>, tR_t<D, K, N>>;
    using itC = global_iterator<gmC_t<D, M, N>, tOut_t<D, M, N>>;
    itA gA(a); itB gB(b); itC gC(c);
    auto gA0 = gA(0, 0), gB0 = gB(0, 0), gC0 = gC(0, 0);
    tL_t<D, M, K> tA; tR_t<D, K, N> tB;
    tAcc_t<D, M, N> tAcc; tOut_t<D, M, N> tOut;
    TLOAD_CUBE(tA, gA0);
    TLOAD_CUBE(tB, gB0);
    TLOAD_CUBE(tAcc, gC0);
    TMATMUL_ACC(tOut, tAcc, tA, tB);
    TSTORE_CUBE(gC0, tOut);
}

// C = A * B + bias  (TMATMUL_BIAS)
template <typename D, int M, int N, int K>
void bench_matmul_bias(D *c, D *a, D *b,
                       cube_accumulator_element_t<D> *bias) {
    using itA = global_iterator<gmA_t<D, M, K>, tL_t<D, M, K>>;
    using itB = global_iterator<gmB_t<D, K, N>, tR_t<D, K, N>>;
    using itC = global_iterator<gmC_t<D, M, N>, tOut_t<D, M, N>>;
    using AccD = cube_accumulator_element_t<D>;
    using itBias = global_iterator<gmC_t<AccD, 1, N>, tBias_t<D, N>>;
    itA gA(a); itB gB(b); itC gC(c); itBias gBias(bias);
    auto gA0 = gA(0, 0);
    auto gB0 = gB(0, 0);
    auto gC0 = gC(0, 0);
    auto gBias0 = gBias(0, 0);
    tL_t<D, M, K> tA; tR_t<D, K, N> tB; tBias_t<D, N> tBias;
    tAcc_t<D, M, N> tAcc; tOut_t<D, M, N> tOut;
    TLOAD_CUBE(tA, gA0);
    TLOAD_CUBE(tB, gB0);
    TLOAD(tBias, gBias0);
    if constexpr (std::is_same_v<D, __half>) {
        TMATMUL_BIAS<FixpAttr::f16()>(tOut, tA, tB, tBias);
    } else {
        TMATMUL_BIAS(tOut, tA, tB, tBias);
    }
    TSTORE_CUBE(gC0, tOut);
}

// PTO ISA 0.58.3 exposes TMATMULMX/TGEMVMX, but the workload does not invent
// scale operands: E8M0 scale tiles are present exactly when the selected MX
// input type requires them. Re-enable MX cases only with a matching TileOP
// overload and an independent result oracle.

// TGEMV templates remain excluded from the generated smoke corpus until the
// workload adds M=1 fixtures and independent result oracles. The API surface is
// available; this is a workload-coverage boundary, not an ISA/toolchain claim.
#if 0
// C = A * v  (TGEMV matrix-vector)
template <typename D, int M, int N, int K>
void bench_gemv(D *c, D *a, D *b) {
    using itA = global_iterator<gmA_t<D, M, K>, tL_t<D, M, K>>;
    using itB = global_iterator<gmB_t<D, K, N>, tR_t<D, K, N>>;
    using itC = global_iterator<gmC_t<D, M, N>, tOut_t<D, M, N>>;
    itA gA(a); itB gB(b); itC gC(c);
    auto gA0 = gA(0, 0), gB0 = gB(0, 0), gC0 = gC(0, 0);
    tL_t<D, M, K> tA; tR_t<D, K, N> tB;
    tAcc_t<D, M, N> tAcc; tOut_t<D, M, N> tOut;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    TGEMV(tAcc, tA, tB);
    TSTORE(gC0, tOut);
}

// C += A * v  (TGEMV_ACC)
template <typename D, int M, int N, int K>
void bench_gemv_acc(D *c, D *a, D *b) {
    using itA = global_iterator<gmA_t<D, M, K>, tL_t<D, M, K>>;
    using itB = global_iterator<gmB_t<D, K, N>, tR_t<D, K, N>>;
    using itC = global_iterator<gmC_t<D, M, N>, tOut_t<D, M, N>>;
    itA gA(a); itB gB(b); itC gC(c);
    auto gA0 = gA(0, 0), gB0 = gB(0, 0), gC0 = gC(0, 0);
    tL_t<D, M, K> tA; tR_t<D, K, N> tB;
    tAcc_t<D, M, N> tAcc; tOut_t<D, M, N> tOut;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    TGEMV_ACC(tAcc, tA, tB);
    TSTORE(gC0, tOut);
}

// C = A * v + bias  (TGEMV_BIAS)
template <typename D, int M, int N, int K>
void bench_gemv_bias(D *c, D *a, D *b, D *bias) {
    using itA = global_iterator<gmA_t<D, M, K>, tL_t<D, M, K>>;
    using itB = global_iterator<gmB_t<D, K, N>, tR_t<D, K, N>>;
    using itC = global_iterator<gmC_t<D, M, N>, tOut_t<D, M, N>>;
    using itBias = global_iterator<gmC_t<D, 1, N>, tOut_t<D, 1, N>>;
    itA gA(a); itB gB(b); itC gC(c); itBias gBias(bias);
    auto gA0 = gA(0, 0), gB0 = gB(0, 0), gC0 = gC(0, 0), gBias0 = gBias(0, 0);
    tL_t<D, M, K> tA; tR_t<D, K, N> tB; tOut_t<D, 1, N> tBias;
    tAcc_t<D, M, N> tAcc; tOut_t<D, M, N> tOut;
    TLOAD(tA, gA0);
    TLOAD(tB, gB0);
    TLOAD(tBias, gBias0);
    TGEMV_BIAS(tAcc, tA, tB, tBias);
    TSTORE(gC0, tOut);
}

// C = A * v  microscaling (TGEMV_MX, with per-tile scale factors)
template <typename D, int M, int N, int K>
void bench_gemv_mx(D *c, D *a, D *as, D *b, D *bs) {
    using itA = global_iterator<gmA_t<D, M, K>, tL_t<D, M, K>>;
    using itB = global_iterator<gmB_t<D, K, N>, tR_t<D, K, N>>;
    using itC = global_iterator<gmC_t<D, M, N>, tOut_t<D, M, N>>;
    itA gA(a), gAs(as); itB gB(b), gBs(bs); itC gC(c);
    auto gA0 = gA(0, 0), gAs0 = gAs(0, 0), gB0 = gB(0, 0), gBs0 = gBs(0, 0), gC0 = gC(0, 0);
    tL_t<D, M, K> tA, tAs; tR_t<D, K, N> tB, tBs;
    tAcc_t<D, M, N> tAcc; tOut_t<D, M, N> tOut;
    TLOAD(tA, gA0); TLOAD(tAs, gAs0);
    TLOAD(tB, gB0); TLOAD(tBs, gBs0);
    TGEMV_MX(tAcc, tA, tAs, tB, tBs);
    TSTORE(gC0, tOut);
}
#endif  // FIXME: TGEMV* not exposed yet

#endif
