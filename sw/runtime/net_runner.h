// =============================================================================
// net_runner.h — network execution runtime (Phase 3b)
//
// Plays the role of Simba's RISC-V controller: layer sequencing plus kernel
// configuration-register computation. For now it calls the kernels as plain
// C functions (csim/g++ verification); in Phase 4 only the call sites get
// replaced with XRT enqueues (the configuration logic stays as-is).
//
// - Tensor: same NHWC 64b-word layout as the kernel contract. A conv output
//   (K1 words) becomes the next layer's input (C1 words) directly, so no
//   repacking happens between layers.
// - plan_conv(): C++ mirror of sw/mapper/mapper.py eval_mapping — picks the
//   spatial mapping (KP,CP) and C-tile (CT1) per layer.
//   Note: when chaining, the tensor layout (C1) is fixed by the previous
//   layer, so candidates with C1 % CP != 0 are excluded (unlike the mapper,
//   repadding is not possible here).
// =============================================================================
#pragma once

#include "magnet_top.h"
#include "global_pe.h"
#include <vector>
#include <cassert>
#include <climits>

namespace nr {

struct Tensor {
    int H = 0, W = 0, C1 = 0;      // NHWC, C1 = channel words (8 int8/word)
    std::vector<iavec_t> data;

    void alloc(int h, int w, int c1) {
        H = h; W = w; C1 = c1;
        data.assign((size_t)h * w * c1, iavec_t());
        for (auto &word : data)
            for (int v = 0; v < VECTOR_SIZE; ++v)
                iavec_set(word, v, (ia_t)0);
    }
    oavec_t *as_oa() { return reinterpret_cast<oavec_t *>(data.data()); }
    const oavec_t *as_oa() const {
        return reinterpret_cast<const oavec_t *>(data.data());
    }
};
static_assert(sizeof(iavec_t) == sizeof(oavec_t),
              "IA/OA word layout must match for layer chaining");

struct ConvParams {
    int K = 0, R = 1, S = 1, stride = 1, pad = 0;
    int mult = 1, shift = 0, relu = 0;
    std::vector<wvec_t> w;      // [Kpad][R][S][C1] (C1 = input tensor's C1)
    std::vector<bias_t> bias;   // [Kpad]
};

struct ConvPlan {
    int KP = N_PES, CP = 1, CT1 = 1;
    bool valid = false;
    long long est_cycles = 0;
};

// Mirror of the mapper.eval_mapping cycle model (unifying the two copies is
// a Phase 4 TODO)
inline ConvPlan eval_mapping(int C1, int K1, int P, int Q, int R, int S,
                             int stride, int KP, int CP) {
    ConvPlan pl;
    pl.KP = KP;
    pl.CP = CP;
    if (C1 % CP != 0)
        return pl;                       // chaining: no repadding -> exclude
    const int rs = R * S;
    const int QB = (Q < MIN_QB) ? MIN_QB : Q;
    const int win_cols = (QB - 1) * stride + S;
    const int c1s = C1 / CP;
    int ct1 = c1s;
    if (ct1 > WBUF_DEPTH / rs) ct1 = WBUF_DEPTH / rs;
    if (ct1 > IABUF_DEPTH / (R * win_cols)) ct1 = IABUF_DEPTH / (R * win_cols);
    if (ct1 < 1 || Q > Q_MAX)
        return pl;
    pl.CT1 = ct1;
    pl.valid = true;

    const int C2 = (c1s + ct1 - 1) / ct1;
    const int P2 = (P + PT_ROWS - 1) / PT_ROWS;
    const int K1G = (K1 + KP - 1) / KP;
    const long long blk_compute = (long long)PT_ROWS * rs * c1s * QB;
    const int cp_beats = (CP + N_IA_PORTS - 1) / N_IA_PORTS; // parallel ports
    const long long blk_load = (long long)PT_ROWS * R * win_cols * c1s * cp_beats;
    const long long blk_gather = (long long)PT_ROWS * Q * KP;
    long long blk = blk_compute;
    if (blk_load > blk) blk = blk_load;
    if (blk_gather > blk) blk = blk_gather;
    const long long body = (long long)K1G * P2 * blk;
    const long long n_wloads = (long long)K1G * (C2 == 1 ? 1 : P2 * C2);
    const long long load_w =
        n_wloads * N_PES * N_LANES * rs * (C2 == 1 ? c1s : ct1);
    pl.est_cycles = body + load_w;
    return pl;
}

inline ConvPlan plan_conv(int C1, int K1, int P, int Q, int R, int S,
                          int stride) {
    ConvPlan best;
    best.est_cycles = LLONG_MAX;
    for (int kp = 1; kp <= N_PES; kp *= 2) {
        if (N_PES % kp)
            continue;
        ConvPlan pl = eval_mapping(C1, K1, P, Q, R, S, stride, kp, N_PES / kp);
        if (pl.valid && pl.est_cycles < best.est_cycles)
            best = pl;
    }
    return best;
}

// ---------------------------------------------------------------------------
// Layer execution (kernel call sites — replaced with XRT enqueues in Phase 4)
// ---------------------------------------------------------------------------
inline void run_conv(const Tensor &in, const ConvParams &cp, Tensor &out) {
    const int P = (in.H + 2 * cp.pad - cp.R) / cp.stride + 1;
    const int Q = (in.W + 2 * cp.pad - cp.S) / cp.stride + 1;
    const int K1 = (cp.K + N_LANES - 1) / N_LANES;
    const ConvPlan pl = plan_conv(in.C1, K1, P, Q, cp.R, cp.S, cp.stride);
    assert(pl.valid && "no valid spatial mapping for this layer");
    assert((int)cp.w.size() == K1 * N_LANES * cp.R * cp.S * in.C1);
    assert((int)cp.bias.size() == K1 * N_LANES);

    out.alloc(P, Q, K1);
    // the four IA pointers alias the same buffer (multi-port loader)
    magnet_top(in.data.data(), in.data.data(), in.data.data(), in.data.data(),
               cp.w.data(), out.as_oa(), cp.bias.data(),
               in.H, in.W, in.C1, K1, cp.K, P, Q, cp.R, cp.S,
               cp.stride, cp.pad, pl.CT1, pl.KP, pl.CP,
               cp.mult, cp.shift, cp.relu);
}

inline void run_eltwise(const Tensor &a, const Tensor &b, Tensor &out,
                        int multA, int multB, int shift, int relu) {
    assert(a.H == b.H && a.W == b.W && a.C1 == b.C1);
    out.alloc(a.H, a.W, a.C1);
    global_pe_top(a.as_oa(), b.as_oa(), out.as_oa(), GPE_ELTWISE,
                  a.H, a.W, a.C1, 1, 1, 1, 0, multA, multB, shift, relu);
}

inline void run_pool(const Tensor &in, Tensor &out, int mode,
                     int R, int S, int stride, int pad,
                     int multA = 0, int shift = 0) {
    const int P = (in.H + 2 * pad - R) / stride + 1;
    const int Q = (in.W + 2 * pad - S) / stride + 1;
    out.alloc(P, Q, in.C1);
    global_pe_top(in.as_oa(), in.as_oa(), out.as_oa(), mode,
                  in.H, in.W, in.C1, R, S, stride, pad, multA, 0, shift, 0);
}

} // namespace nr
