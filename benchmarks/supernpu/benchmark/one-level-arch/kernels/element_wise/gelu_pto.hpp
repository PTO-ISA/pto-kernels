// PTO/LinxISA v0.58 GELU workload using VEC elementwise and SFU operations.
#include <common/pto_tile.hpp>
#include <common/global_iterator.hpp>

#include <cstdint>
#include <cstdio>

// ============================================================================
// GELU 多项式拟合系数 (与原始 gelu.hpp 一致)
// GELU(x) = x / (1 + exp(t * P(t²)))
// P(t²) = A5*t²⁵ + A4*t²⁴ + A3*t²³ + A2*t²² + A1*t² + A0 + AM1/t²
// (Horner: p = ((((A5*t2 + A4)*t2 + A3)*t2 + A2)*t2 + A1)*t2 + A0)*t2 + AM1)
// ============================================================================
namespace gelu_coeffs {
    constexpr float A5  = -3.5123395303315874e-09f;
    constexpr float A4  =  2.6452661927578447e-07f;
    constexpr float A3  = -7.9294877650681883e-06f;
    constexpr float A2  =  1.1061238183174282e-04f;
    constexpr float A1  =  6.5189960878342390e-05f;
    constexpr float A0  = -7.2666168212890625e-02f;
    constexpr float AM1 = -1.5957698822021484e+00f;
    constexpr float CLAMP_MAX = 5.75f;
}

// ----------------------------------------------------------------------------
// gelu_impl: 用 PTO ISA Tile 指令计算 GELU (一层编程, 无 __vec__ 块)
//
// 输入:  inTile  — fp16 tile, shape (1, tM)
// 输出:  outTile — fp16 tile, shape (1, tM)
// 中间:  全部在 fp32 tile 上计算
//
// 算法:
//   x  = (float)in
//   t  = clamp(x, -5.75, 5.75)
//   t2 = t * t
//   p  = Horner(t2, [A5,A4,A3,A2,A1,A0,AM1])
//   e  = exp(t * p)
//   y  = x * (1 / (1 + e))           // 用 TRECIP + TMUL 代替除法
//   out = (half)y
// ----------------------------------------------------------------------------
template<typename tile_shapeData, typename tile_shapeFP32>
void gelu_impl(
    tile_shapeData  &inTile,
    tile_shapeData  &outTile,
    tile_shapeFP32  &tmpCvt          // TCVT 需要的临时 tile
) {
    using fp_t = typename tile_shapeFP32::DType;   // float

    tile_shapeFP32 xTile;        // x = (float)input
    tile_shapeFP32 tTile;        // t = clamp(x)
    tile_shapeFP32 t2Tile;       // t²
    tile_shapeFP32 pTile;        // 多项式值
    tile_shapeFP32 scratchTile;  // 复用: tp -> exp -> denom -> recip -> y

    // ---- Step 1: fp16 -> fp32 ----
    TCVT(xTile, inTile);

    // ---- Step 2: clamp x to [-5.75, 5.75] ----
    TMAXS(tTile, xTile, (fp_t)(-gelu_coeffs::CLAMP_MAX));   // t = max(x, -5.75)
    TMINS(tTile, tTile, (fp_t)gelu_coeffs::CLAMP_MAX);       // t = min(t, 5.75)

    // ---- Step 3: t² = t * t ----
    TMUL(t2Tile, tTile, tTile);

    // ---- Step 4: 多项式 Horner 法 ----
    // p = A5*t2 + A4
    TMULS(pTile, t2Tile, gelu_coeffs::A5);
    TADDS(pTile, pTile, gelu_coeffs::A4);

    // p = p*t2 + A3
    TMUL(pTile, pTile, t2Tile);
    TADDS(pTile, pTile, gelu_coeffs::A3);

    // p = p*t2 + A2
    TMUL(pTile, pTile, t2Tile);
    TADDS(pTile, pTile, gelu_coeffs::A2);

    // p = p*t2 + A1
    TMUL(pTile, pTile, t2Tile);
    TADDS(pTile, pTile, gelu_coeffs::A1);

    // p = p*t2 + A0
    TMUL(pTile, pTile, t2Tile);
    TADDS(pTile, pTile, gelu_coeffs::A0);

    // p = p*t2 + AM1
    TMUL(pTile, pTile, t2Tile);
    TADDS(pTile, pTile, gelu_coeffs::AM1);

    // ---- Step 5: exp_val = exp(t * p) ----
    // scratch = t * p
    TMUL(scratchTile, tTile, pTile);
    // exp_val = exp(scratch)
    // v0.58 SFU TEXP
    TEXP(scratchTile, scratchTile);          // scratch = exp(t*p)

    // ---- Step 6: y = x / (1 + exp_val) ----
    // denom = 1 + exp_val
    TADDS(scratchTile, scratchTile, (fp_t)1.0f);   // scratch = 1 + exp
    // recip = 1 / denom
    // v0.58 SFU TRECIP
    TRECIP(scratchTile, scratchTile);               // scratch = 1 / (1+exp)
    // y = x * recip
    TMUL(scratchTile, xTile, scratchTile);           // scratch = x * recip = y

    // ---- Step 7: fp32 -> fp16 ----
    TCVT(outTile, scratchTile);
}


// ----------------------------------------------------------------------------
// gelu: 主入口, 接口与原 gelu.hpp 一致
// ----------------------------------------------------------------------------
template<typename dtype, int gM, int tM>
void gelu(
    dtype *in_ptr,
    dtype *out_ptr,
    bool approximate = false
    ) {
    using gm_shape       = global_tensor<dtype, RowMajor<1, gM>>;
    using tile_shapeData = Tile<Location::Vec, dtype, 1, tM, BLayout::RowMajor>;
    using tile_shapeFP32 = Tile<Location::Vec, float, 1, tM, BLayout::RowMajor>;
    using tile_shapeData_rmd = Tile<Location::Vec, dtype, 1, tM, BLayout::RowMajor, 1, gM % tM>;
    using tile_shapeFP32_rmd = Tile<Location::Vec, float, 1, tM, BLayout::RowMajor, 1, gM % tM>;

    const int Mb    = gM / tM;
    const int rmd_M = gM % tM;

    using itIn  = global_iterator<gm_shape, tile_shapeData>;
    using itOut = global_iterator<gm_shape, tile_shapeData>;

    itIn  gIIter(in_ptr);
    itOut gOIter(out_ptr);

    tile_shapeData inTile, outTile;
    tile_shapeFP32 tmpCvt;                          // TCVT 临时 tile
    tile_shapeData_rmd inTile_rmd, outTile_rmd;
    tile_shapeFP32_rmd tmpCvt_rmd;

    for (int i = 0; i < Mb; ++i) {
        auto gI = gIIter(0, i);
        auto gO = gOIter(0, i);

        // TLOAD: GM -> UB
        TLOAD(inTile, gI);

        gelu_impl<tile_shapeData, tile_shapeFP32>(inTile, outTile, tmpCvt);

        // TSTORE: UB -> GM
        TSTORE(gO, outTile);
    }
    if constexpr (rmd_M) {
        auto gI = gIIter(0, Mb);
        auto gO = gOIter(0, Mb);

        TLOAD(inTile_rmd, gI);
        gelu_impl<tile_shapeData_rmd, tile_shapeFP32_rmd>(inTile_rmd, outTile_rmd, tmpCvt_rmd);
        TSTORE(gO, outTile_rmd);
    }
}
