// =============================================================================
// global_pe.h — Global PE kernel declaration (shared by testbench/runtime)
// =============================================================================
#pragma once

#include "accel_types.h"

#define GPE_ELTWISE 0
#define GPE_MAXPOOL 1
#define GPE_AVGPOOL 2

// Data: same NHWC 64b words as the conv kernel ([H][W][C1], 8 int8/word)
//   ELTWISE: O = sat(relu((A*multA + B*multB) >> shift)); B/R/S/stride/pad unused
//   MAXPOOL: window max (no requantization; B/mult* unused)
//   AVGPOOL: window sum then sat((sum*multA) >> shift) — 1/count folded into multA
extern "C" void global_pe_top(
    const oavec_t *gmem_a,
    const oavec_t *gmem_b,
    oavec_t       *gmem_o,
    int mode,
    int H, int W, int C1,
    int R, int S, int stride, int pad,
    int multA, int multB, int shift, int relu_en);
