// PTO/LinxISA v0.58 broadcast workload using the canonical TileOP API.
#include <common/pto_tile.hpp>
#include <common/global_iterator.hpp>

#include <cstdint>
#include <cstdio>

// =====================================================================
// Broadcast (N,1) -> (N,C) via TLOAD + TROWEXPAND + TSTORE
//
// Template params (与原 broadcast_vec_07 一致):
//   dtype     - data type (__half, float, etc.)
//   MAX_DIM   - max dimensions (unused, kept for compat)
//   IN_DIM    - input dim count (unused)
//   OUT_DIM   - output dim count (unused)
//   gIM       - total input elements = N*1 = N   (e.g. 1443)
//   gOM       - total output elements = N*C       (e.g. 1443*129)
//   kTileRows - rows per tile, must be power-of-2 (e.g. 1,2,4,..,64)
// =====================================================================

template<typename dtype, size_t MAX_DIM = 8, size_t IN_DIM, size_t OUT_DIM,
         size_t gIM, size_t gOM, size_t kTileRows>
void broadcast(dtype *in_ptr, dtype *out_ptr,
               const size_t * /*in_shape*/, const size_t * /*out_shape*/) {
    constexpr size_t kN = gIM;
    constexpr size_t kC = gOM / gIM;
    constexpr size_t tileCols = 256;

    static_assert(gOM % gIM == 0,
                  "gOM must be divisible by gIM for (N,1)->(N,C) broadcast");
    static_assert(tileCols >= kC,
                  "padded tileCols (256) must >= broadcast target columns");
    static_assert((kTileRows & (kTileRows - 1)) == 0,
                  "kTileRows must be power of 2 for 512B tile alignment");

    using tile_in  = Tile<Location::Vec, dtype, kTileRows, tileCols,
                          BLayout::RowMajor, kTileRows, 1>;
    using tile_out = Tile<Location::Vec, dtype, kTileRows, tileCols,
                          BLayout::RowMajor, kTileRows, kC>;
    using gm_in    = global_tensor<dtype, RowMajor<kTileRows, 1>>;
    using gm_out   = global_tensor<dtype, RowMajor<kTileRows, kC>>;

    constexpr size_t Nb  = kN / kTileRows;
    constexpr size_t rmd = kN % kTileRows;

    tile_in inTile;
    tile_out outTile;

    for (size_t i = 0; i < Nb; i++) {
        gm_in gsrc(in_ptr + i * kTileRows);
        gm_out gdst(out_ptr + i * kTileRows * kC);

        // TLOAD: GM -> UB, 加载 (kTileRows, 1) 输入 tile
        TLOAD(inTile, gsrc);

        // TROWEXPAND: 将每行 col 0 广播到全部 kC 列 -> (kTileRows, kC)
        TROWEXPAND(outTile, inTile);

        // TSTORE: UB -> GM, 写回 (kTileRows, kC) 输出 tile
        TSTORE(gdst, outTile);
    }

    using tile_in_r  = Tile<Location::Vec, dtype, kTileRows, tileCols,
                            BLayout::RowMajor, rmd, 1>;
    using tile_out_r = Tile<Location::Vec, dtype, kTileRows, tileCols,
                            BLayout::RowMajor, rmd, kC>;
    tile_in_r inTile_rmd;
    tile_out_r outTile_rmd;
    if constexpr (rmd > 0) {
        gm_in gsrc(in_ptr + Nb * kTileRows);
        gm_out gdst(out_ptr + Nb * kTileRows * kC);

        TLOAD(inTile_rmd, gsrc);
        TROWEXPAND(outTile_rmd, inTile_rmd);
        TSTORE(gdst, outTile_rmd);
    }
}
