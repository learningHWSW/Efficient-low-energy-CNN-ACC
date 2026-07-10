// =============================================================================
// tb_network.cpp — network-level end-to-end test (Phase 3b)
//
// Runs a ResNet bottleneck block as a chain through net_runner and compares
// bit-exactly against golden models:
//
//   x(16x16x32) - conv1x1 32->16 relu - conv3x3 16->16 relu - conv1x1 16->32
//        \_________________ residual _________________________
//                                          eltwise add + relu <-'
//                                          maxpool 3x3 s2 p1  -> 8x8x32
//                                          global avgpool 8x8 -> 1x1x32
//
// What this verifies: data-layout compatibility between kernels (chaining
// without repacking), plan_conv's automatic KP/CP/CT1 selection, and
// bit-exactness at every stage.
//
// Run: bundled g++ (run_hls.ps1 net) — links both kernels, hence gcc mode
// =============================================================================

#include "../../sw/runtime/net_runner.h"
#include <cstdio>
#include <random>

// ---- golden helpers (same arithmetic as the kernels) -----------------------
static int8_t g_requant(int64_t x, int mult, int shift, bool relu) {
    x *= mult;
    if (shift > 0) x += ((int64_t)1 << (shift - 1));
    x >>= shift;
    if (relu && x < 0) x = 0;
    if (x > 127) x = 127;
    if (x < -128) x = -128;
    return (int8_t)x;
}

// Golden conv: NHWC int8 -> int8 (assumes C and K are multiples of 8)
static std::vector<int8_t> g_conv(const std::vector<int8_t> &in, int H, int W,
                                  int C, const std::vector<int8_t> &w,
                                  const std::vector<int32_t> &bias, int K,
                                  int R, int S, int stride, int pad,
                                  int mult, int shift, int relu,
                                  int &P_out, int &Q_out) {
    const int P = (H + 2 * pad - R) / stride + 1;
    const int Q = (W + 2 * pad - S) / stride + 1;
    std::vector<int8_t> out((size_t)P * Q * K);
    for (int p = 0; p < P; ++p)
        for (int q = 0; q < Q; ++q)
            for (int k = 0; k < K; ++k) {
                int32_t acc = bias[k];
                for (int r = 0; r < R; ++r)
                    for (int s = 0; s < S; ++s) {
                        const int ih = p * stride - pad + r;
                        const int iw = q * stride - pad + s;
                        if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                        for (int c = 0; c < C; ++c)
                            acc += (int32_t)w[((k * R + r) * S + s) * C + c] *
                                   (int32_t)in[(ih * W + iw) * C + c];
                    }
                out[(p * Q + q) * K + k] = g_requant(acc, mult, shift, relu != 0);
            }
    P_out = P; Q_out = Q;
    return out;
}

// ---- Tensor <-> int8 array conversion ---------------------------------------
static nr::Tensor to_tensor(const std::vector<int8_t> &v, int H, int W, int C) {
    nr::Tensor t;
    t.alloc(H, W, C / VECTOR_SIZE);
    for (int i = 0; i < H * W; ++i)
        for (int c = 0; c < C; ++c)
            iavec_set(t.data[i * t.C1 + c / VECTOR_SIZE], c % VECTOR_SIZE,
                      (ia_t)v[i * C + c]);
    return t;
}

static int compare(const char *name, const nr::Tensor &t,
                   const std::vector<int8_t> &gold, int C) {
    int errors = 0;
    for (int i = 0; i < t.H * t.W; ++i)
        for (int c = 0; c < C; ++c) {
            const int8_t hw = (int8_t)iavec_get(
                t.data[i * t.C1 + c / VECTOR_SIZE], c % VECTOR_SIZE);
            if (hw != gold[i * C + c] && errors++ < 5)
                printf("  [%s] MISMATCH pix=%d c=%d hw=%d sw=%d\n",
                       name, i, c, (int)hw, (int)gold[i * C + c]);
        }
    printf("[NET] %-18s %dx%dx%-3d -> %s (%d errors)\n",
           name, t.H, t.W, C, errors ? "FAIL" : "PASS", errors);
    return errors;
}

// Prepare conv parameters (random weight/bias generation + device packing)
static nr::ConvParams make_conv(std::mt19937 &rng, int C, int K, int R, int S,
                                int stride, int pad, int mult, int shift,
                                int relu, std::vector<int8_t> &w_raw,
                                std::vector<int32_t> &b_raw) {
    std::uniform_int_distribution<int> d8(-128, 127);
    std::uniform_int_distribution<int> db(-1000, 1000);
    w_raw.resize((size_t)K * R * S * C);
    b_raw.assign(K, 0);
    for (auto &v : w_raw) v = (int8_t)d8(rng);
    for (auto &v : b_raw) v = db(rng);

    nr::ConvParams cp;
    cp.K = K; cp.R = R; cp.S = S; cp.stride = stride; cp.pad = pad;
    cp.mult = mult; cp.shift = shift; cp.relu = relu;
    const int C1 = C / VECTOR_SIZE;
    const int K1 = (K + N_LANES - 1) / N_LANES;
    cp.w.assign((size_t)K1 * N_LANES * R * S * C1, wvec_t());
    for (auto &word : cp.w)
        for (int v = 0; v < VECTOR_SIZE; ++v) wvec_set(word, v, (w_t)0);
    for (int k = 0; k < K; ++k)
        for (int r = 0; r < R; ++r)
            for (int s = 0; s < S; ++s)
                for (int c = 0; c < C; ++c)
                    wvec_set(cp.w[((k * R + r) * S + s) * C1 + c / VECTOR_SIZE],
                             c % VECTOR_SIZE,
                             (w_t)w_raw[((k * R + r) * S + s) * C + c]);
    cp.bias.assign(K1 * N_LANES, 0);
    for (int k = 0; k < K; ++k) cp.bias[k] = b_raw[k];
    return cp;
}

int main() {
    std::mt19937 rng(20260709);
    std::uniform_int_distribution<int> d8(-128, 127);
    int e = 0;

    // ---- input x: 16x16x32 ----
    const int H = 16, W = 16, C = 32;
    std::vector<int8_t> x((size_t)H * W * C);
    for (auto &v : x) v = (int8_t)d8(rng);
    nr::Tensor t_x = to_tensor(x, H, W, C);

    // ---- L0: conv1x1 32->16 relu ----
    std::vector<int8_t> w0r; std::vector<int32_t> b0r;
    nr::ConvParams cp0 = make_conv(rng, 32, 16, 1, 1, 1, 0, 9, 11, 1, w0r, b0r);
    int P, Q;
    std::vector<int8_t> g0 = g_conv(x, H, W, 32, w0r, b0r, 16, 1, 1, 1, 0,
                                    9, 11, 1, P, Q);
    nr::Tensor t0;
    nr::run_conv(t_x, cp0, t0);
    e += compare("conv1x1_32->16", t0, g0, 16);

    // ---- L1: conv3x3 16->16 relu ----
    std::vector<int8_t> w1r; std::vector<int32_t> b1r;
    nr::ConvParams cp1 = make_conv(rng, 16, 16, 3, 3, 1, 1, 5, 11, 1, w1r, b1r);
    std::vector<int8_t> g1 = g_conv(g0, P, Q, 16, w1r, b1r, 16, 3, 3, 1, 1,
                                    5, 11, 1, P, Q);
    nr::Tensor t1;
    nr::run_conv(t0, cp1, t1);
    e += compare("conv3x3_16->16", t1, g1, 16);

    // ---- L2: conv1x1 16->32 (no relu, feeds the residual add) ----
    std::vector<int8_t> w2r; std::vector<int32_t> b2r;
    nr::ConvParams cp2 = make_conv(rng, 16, 32, 1, 1, 1, 0, 7, 11, 0, w2r, b2r);
    std::vector<int8_t> g2 = g_conv(g1, P, Q, 16, w2r, b2r, 32, 1, 1, 1, 0,
                                    7, 11, 0, P, Q);
    nr::Tensor t2;
    nr::run_conv(t1, cp2, t2);
    e += compare("conv1x1_16->32", t2, g2, 32);

    // ---- E: residual add (t2*mA + x*mB) >> sh, relu ----
    const int mA = 200, mB = 71, sh = 8;
    std::vector<int8_t> gE((size_t)H * W * C);
    for (size_t i = 0; i < gE.size(); ++i) {
        int64_t v = (int64_t)g2[i] * mA + (int64_t)x[i] * mB;
        if (sh > 0) v += (int64_t)1 << (sh - 1);
        v >>= sh;
        if (v < 0) v = 0;
        if (v > 127) v = 127;
        gE[i] = (int8_t)v;
    }
    nr::Tensor tE;
    nr::run_eltwise(t2, t_x, tE, mA, mB, sh, 1);
    e += compare("residual_add", tE, gE, 32);

    // ---- M: maxpool 3x3 s2 p1 -> 8x8x32 ----
    const int PP = (H + 2 - 3) / 2 + 1;
    std::vector<int8_t> gM((size_t)PP * PP * C);
    for (int p = 0; p < PP; ++p)
        for (int q = 0; q < PP; ++q)
            for (int c = 0; c < C; ++c) {
                int mx = -128;
                for (int r = 0; r < 3; ++r)
                    for (int s = 0; s < 3; ++s) {
                        const int ih = p * 2 - 1 + r, iw = q * 2 - 1 + s;
                        if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                        const int v = gE[(ih * W + iw) * C + c];
                        if (v > mx) mx = v;
                    }
                gM[(p * PP + q) * C + c] = (int8_t)mx;
            }
    nr::Tensor tM;
    nr::run_pool(tE, tM, GPE_MAXPOOL, 3, 3, 2, 1);
    e += compare("maxpool_3x3s2", tM, gM, 32);

    // ---- A: global avgpool 8x8 -> 1x1x32 (multA = round(2^15/64) = 512) ----
    std::vector<int8_t> gA(C);
    for (int c = 0; c < C; ++c) {
        int64_t sum = 0;
        for (int i = 0; i < PP * PP; ++i) sum += gM[i * C + c];
        sum *= 512;
        sum += (int64_t)1 << 14;
        sum >>= 15;
        if (sum > 127) sum = 127;
        if (sum < -128) sum = -128;
        gA[c] = (int8_t)sum;
    }
    nr::Tensor tA;
    nr::run_pool(tM, tA, GPE_AVGPOOL, PP, PP, 1, 0, 512, 15);
    e += compare("global_avgpool", tA, gA, 32);

    printf(e == 0 ? "\n=== NETWORK E2E PASSED ===\n"
                  : "\n=== NETWORK E2E FAILED: %d errors ===\n", e);
    return e ? 1 : 0;
}
