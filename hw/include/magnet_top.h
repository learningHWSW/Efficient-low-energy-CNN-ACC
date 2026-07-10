// =============================================================================
// magnet_top.h — top-level kernel declaration (shared by testbench/host)
// =============================================================================
#pragma once

#include "accel_types.h"

// Data layout (all prepared by the host, see mapper.py):
//   gmem_ia0..3: IA  NHWC, in words [H][W][C1]        (C padded to a VECTOR_SIZE multiple)
//                All four pointers reference the SAME buffer — the loader
//                reads C-slices through N_IA_PORTS AXI masters in parallel.
//   gmem_w    : W   KRSC,  in words [Kpad][R][S][C1]  (K padded to an N_LANES multiple)
//   gmem_oa   : OA  NHWC,  in words [P][Q][K1]        (word = N_LANES int8)
//   gmem_bias : int32 [Kpad]
extern "C" void magnet_top(
    const iavec_t *gmem_ia0,
    const iavec_t *gmem_ia1,
    const iavec_t *gmem_ia2,
    const iavec_t *gmem_ia3,
    const wvec_t  *gmem_w,
    oavec_t       *gmem_oa,
    const bias_t  *gmem_bias,
    int H, int W, int C1, int K1, int K,
    int P, int Q, int R, int S,
    int stride, int pad, int CT1,
    int KP, int CP,
    int mult, int shift, int relu_en);
