// PTO/LinxISA v0.58 broadcast workload using the canonical TileOP API.
#include <common/pto_tile.hpp>
#include <common/global_iterator.hpp>

#include <cstdint>
#include <cstdio>

// =====================================================================
// Broadcast (B,1,K) -> (B,N,K) via TLOAD + TINSERT×N + TSTORE
//
// Data layout (row-major):
//   Input:  [B][1][K] -> flat = B*K,       batch b data at offset b*K
//   Output: [B][N][K] -> flat = B*N*K,     batch b data at offset b*N*K
//   Broadcast along dim1: 1 -> N, dim0 & dim2 preserved.
//
// Template params (与原 broadcast_vec_039 一致):
//   dtype      - data type (__half, float, etc.)
//   MAX_DIM    - max dimensions (kept for compat)
//   IN_DIM     - input dim count (kept for compat)
//   OUT_DIM    - output dim count (kept for compat)
//   gIM        - total input elements  = B * K      (e.g. 8192*16 = 131072)
//   gOM        - total output elements = B * N * K  (e.g. 8192*8*16 = 1048576)
//   kTileBatch - batches per tile, power-of-2 (e.g. 1,2,4,8,16,32,64)
//   kInner     - inner dimension K, power-of-2      (e.g. 16)
// =====================================================================

template<typename dtype, size_t MAX_DIM = 8, size_t IN_DIM, size_t OUT_DIM,
         size_t gIM, size_t gOM, size_t kTileBatch, size_t kInner>
void broadcast(dtype *in_ptr, dtype *out_ptr,
               const size_t * /*in_shape*/, const size_t * /*out_shape*/) {
    constexpr size_t kBCast = gOM / gIM;
    constexpr size_t kBatch = gIM / kInner;
    constexpr size_t tileCols = 256;

    static_assert(gOM % gIM == 0,
                  "gOM must be divisible by gIM for (B,1,K)->(B,N,K) broadcast");
    static_assert(gIM % kInner == 0,
                  "gIM must be divisible by kInner (B = gIM/kInner must be integer)");
    static_assert((kInner & (kInner - 1)) == 0,
                  "kInner must be power of 2 for bit-op index decomposition");
    static_assert((kTileBatch & (kTileBatch - 1)) == 0,
                  "kTileBatch must be power of 2 for 512B tile alignment");
    static_assert(tileCols >= kBCast * kInner,
                  "padded tileCols (256) must >= broadcast target width (N*K)");

    using tile_in  = Tile<Location::Vec, dtype, kTileBatch, tileCols,
                          BLayout::RowMajor, kTileBatch, kInner>;
    using tile_out = Tile<Location::Vec, dtype, kTileBatch, tileCols,
                          BLayout::RowMajor, kTileBatch, kBCast * kInner>;
    using gm_in    = global_tensor<dtype, RowMajor<kTileBatch, kInner>>;
    using gm_out   = global_tensor<dtype, RowMajor<kTileBatch, kBCast * kInner>>;

    constexpr size_t Nb  = kBatch / kTileBatch;
    constexpr size_t rmd = kBatch % kTileBatch;

    tile_in inTile;
    tile_out outTile;

    for (size_t i = 0; i < Nb; i++) {
        gm_in gsrc(in_ptr + i * kTileBatch * kInner);
        gm_out gdst(out_ptr + i * kTileBatch * kBCast * kInner);

        // TLOAD: GM -> UB, 加载 (kTileBatch, kInner) 输入 tile
        TLOAD(inTile, gsrc);

        // TINSERT × kBCast: 将输入 tile 插入输出 tile 的 N 个列偏移
        // 每次 TINSERT 写入 kInner 列, N 次互不重叠, 合起来填满 N*kInner 列
        #pragma clang loop unroll(full)
        for (size_t c = 0; c < kBCast; c++) {
            TINSERT(outTile, inTile, /*indexRow=*/0, /*indexCol=*/(uint16_t)(c * kInner));
        }

        // TSTORE: UB -> GM, 写回 (kTileBatch, kBCast*kInner) 输出 tile
        TSTORE(gdst, outTile);
    }

    using tile_in_r  = Tile<Location::Vec, dtype, kTileBatch, tileCols,
                            BLayout::RowMajor, rmd, kInner>;
    using tile_out_r = Tile<Location::Vec, dtype, kTileBatch, tileCols,
                            BLayout::RowMajor, rmd, kBCast * kInner>;
    tile_in_r inTile_rmd;
    tile_out_r outTile_rmd;
    if constexpr (rmd > 0) {
        gm_in gsrc(in_ptr + Nb * kTileBatch * kInner);
        gm_out gdst(out_ptr + Nb * kTileBatch * kBCast * kInner);

        TLOAD(inTile_rmd, gsrc);

        #pragma clang loop unroll(full)
        for (size_t c = 0; c < kBCast; c++) {
            TINSERT(outTile_rmd, inTile_rmd, 0, (uint16_t)(c * kInner));
        }

        TSTORE(gdst, outTile_rmd);
    }
}
