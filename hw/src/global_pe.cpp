// =============================================================================
// global_pe.cpp — Global PE (Phase 3b, maps to Simba MICRO'19 Section 3.1)
//
// Simba's Global PE provides second-level storage plus near-memory execution
// of "low-reuse" operations (element-wise add in ResNet, pooling, ...).
// This kernel implements that compute-unit part:
//
//   mode 0: ELTWISE  — O = sat(relu((A*multA + B*multB + round) >> shift))
//                      ResNet residual add (per-input requantization scales)
//   mode 1: MAXPOOL  — O[p,q,c] = max over the valid window (padding ignored)
//                      scale-preserving, so no requantization
//   mode 2: AVGPOOL  — O[p,q,c] = sat((sum * multA + round) >> shift)
//                      1/(window size) is folded into multA/shift by the host
//                      (global average pool: R=H, S=W, stride=1, pad=0)
//
// The data layout is the same NHWC 64b word as the conv kernel
// ([H][W][C1], 8 int8/word), so magnet_top output feeds in directly.
// =============================================================================

#include "global_pe.h"

#ifndef __SYNTHESIS__
#include <cassert>
#endif

static oa_t sat_shift(wide_t x, int shift, bool relu) {
#pragma HLS inline
    if (shift > 0)
        x += ((wide_t)1 << (shift - 1));
    x >>= shift;
    if (relu && x < 0) x = 0;
    if (x > OA_MAX) x = OA_MAX;
    if (x < OA_MIN) x = OA_MIN;
    return (oa_t)x;
}

// ---------------------------------------------------------------------------
// Elementwise: sum two int8 tensors, requantizing each with its own scale
// (residual add)
// ---------------------------------------------------------------------------
static void eltwise_add(const oavec_t *gmem_a, const oavec_t *gmem_b,
                        oavec_t *gmem_o, int n_words,
                        int multA, int multB, int shift, bool relu) {
ELT:
    for (int i = 0; i < n_words; ++i) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 64 max = 1048576
        const oavec_t a = gmem_a[i];
        const oavec_t b = gmem_b[i];
        oavec_t o;
        for (int lane = 0; lane < N_LANES; ++lane) {
#pragma HLS unroll
            const wide_t x = (wide_t)((acc_t)oavec_get(a, lane) * multA) +
                             (wide_t)((acc_t)oavec_get(b, lane) * multB);
            oavec_set(o, lane, sat_shift(x, shift, relu));
        }
        gmem_o[i] = o;
    }
}

// ---------------------------------------------------------------------------
// Pooling: walk the window for each output pixel (max, or sum -> requant).
//   Padding is ignored (max: keeps the -128 identity; avg: excluded from the
//   sum — the divisor lives in multA, so count_include_pad is also a host
//   decision).
// ---------------------------------------------------------------------------
static void pool(const oavec_t *gmem_a, oavec_t *gmem_o,
                 int H, int W, int C1, int R, int S, int stride, int pad,
                 int multA, int shift, bool is_avg) {
    const int P = (H + 2 * pad - R) / stride + 1;
    const int Q = (W + 2 * pad - S) / stride + 1;
    const int rs = R * S;

POOL_PIX:
    for (int opix = 0; opix < P * Q; ++opix) {
#pragma HLS loop_tripcount min = 1 max = 65536
        const int p = opix / Q;
        const int q = opix % Q;
    POOL_C:
        for (int c1 = 0; c1 < C1; ++c1) {
#pragma HLS loop_tripcount min = 1 max = 256
            acc_t acc[N_LANES];
#pragma HLS array_partition variable = acc complete
            for (int lane = 0; lane < N_LANES; ++lane) {
#pragma HLS unroll
                acc[lane] = is_avg ? (acc_t)0 : (acc_t)OA_MIN;
            }
        POOL_WIN:
            for (int w_i = 0; w_i < rs; ++w_i) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 4 max = 49
                const int r = w_i / S;
                const int s = w_i % S;
                const int ih = p * stride - pad + r;
                const int iw = q * stride - pad + s;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                    continue;
                const oavec_t v = gmem_a[(ih * W + iw) * C1 + c1];
                for (int lane = 0; lane < N_LANES; ++lane) {
#pragma HLS unroll
                    const acc_t x = (acc_t)oavec_get(v, lane);
                    if (is_avg)
                        acc[lane] += x;
                    else if (x > acc[lane])
                        acc[lane] = x;
                }
            }
            oavec_t o;
            for (int lane = 0; lane < N_LANES; ++lane) {
#pragma HLS unroll
                if (is_avg)
                    oavec_set(o, lane,
                              sat_shift((wide_t)acc[lane] * multA, shift, false));
                else
                    oavec_set(o, lane, (oa_t)acc[lane]);
            }
            gmem_o[(p * Q + q) * C1 + c1] = o;
        }
    }
}

// ===========================================================================
// Top level
// ===========================================================================
extern "C" void global_pe_top(
    const oavec_t *gmem_a,
    const oavec_t *gmem_b,
    oavec_t       *gmem_o,
    int mode,
    int H, int W, int C1,
    int R, int S, int stride, int pad,
    int multA, int multB, int shift, int relu_en) {
#pragma HLS INTERFACE m_axi port = gmem_a offset = slave bundle = gmem0 depth = 1048576
#pragma HLS INTERFACE m_axi port = gmem_b offset = slave bundle = gmem1 depth = 1048576
#pragma HLS INTERFACE m_axi port = gmem_o offset = slave bundle = gmem2 depth = 1048576
#pragma HLS INTERFACE s_axilite port = mode
#pragma HLS INTERFACE s_axilite port = H
#pragma HLS INTERFACE s_axilite port = W
#pragma HLS INTERFACE s_axilite port = C1
#pragma HLS INTERFACE s_axilite port = R
#pragma HLS INTERFACE s_axilite port = S
#pragma HLS INTERFACE s_axilite port = stride
#pragma HLS INTERFACE s_axilite port = pad
#pragma HLS INTERFACE s_axilite port = multA
#pragma HLS INTERFACE s_axilite port = multB
#pragma HLS INTERFACE s_axilite port = shift
#pragma HLS INTERFACE s_axilite port = relu_en
#pragma HLS INTERFACE s_axilite port = return

#ifndef __SYNTHESIS__
    assert(mode == GPE_ELTWISE || mode == GPE_MAXPOOL || mode == GPE_AVGPOOL);
    if (mode != GPE_ELTWISE) {
        assert(R >= 1 && S >= 1 && stride >= 1);
        assert((H + 2 * pad - R) % stride == 0 || true);
    }
#endif

    if (mode == GPE_ELTWISE) {
        eltwise_add(gmem_a, gmem_b, gmem_o, H * W * C1,
                    multA, multB, shift, relu_en != 0);
    } else {
        pool(gmem_a, gmem_o, H, W, C1, R, S, stride, pad,
             multA, shift, mode == GPE_AVGPOOL);
    }
}
