// =============================================================================
// tb_resnet50.cpp — full ResNet-50 network test (Phase 4)
//
// Two parts:
//   1) FULL-SIZE DRY RUN: plans every real ResNet-50 conv shape (224x224
//      input, 53 convs incl. downsample shortcuts) with plan_conv and asserts
//      a valid mapping exists. Prints the per-layer plan and a total cycle /
//      fps estimate. No computation — validates constraint coverage.
//   2) REDUCED-SIZE E2E: executes the complete ResNet-50 v1.5 topology
//      (stem + 16 bottleneck blocks + pools) at 32x32 input with the REAL
//      channel widths (64..2048) through net_runner, comparing every block
//      output bit-exactly against a golden int8 chain. Exercises deep layer
//      chaining, residual/downsample wiring, and the tiny-Q padding paths.
//
// Run: bundled g++ (run_hls.ps1 resnet) — links both kernels
// =============================================================================

#include "../../sw/runtime/net_runner.h"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// golden helpers (same arithmetic as the kernels)
// ---------------------------------------------------------------------------
static int8_t g_sat(int64_t x, int shift, bool relu) {
    if (shift > 0) x += ((int64_t)1 << (shift - 1));
    x >>= shift;
    if (relu && x < 0) x = 0;
    if (x > OA_MAX) x = OA_MAX;
    if (x < OA_MIN) x = OA_MIN;
    return (int8_t)x;
}

static std::vector<int8_t> g_conv(const std::vector<int8_t> &in, int H, int W,
                                  int C, const std::vector<int8_t> &w,
                                  const std::vector<int32_t> &bias, int K,
                                  int R, int S, int stride, int pad,
                                  int mult, int shift, int relu) {
    const int P = (H + 2 * pad - R) / stride + 1;
    const int Q = (W + 2 * pad - S) / stride + 1;
    std::vector<int8_t> out((size_t)P * Q * K);
    for (int p = 0; p < P; ++p)
        for (int q = 0; q < Q; ++q)
            for (int k = 0; k < K; ++k) {
                int64_t acc = bias[k];
                for (int r = 0; r < R; ++r)
                    for (int s = 0; s < S; ++s) {
                        const int ih = p * stride - pad + r;
                        const int iw = q * stride - pad + s;
                        if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                        for (int c = 0; c < C; ++c)
                            acc += (int32_t)w[((k * R + r) * S + s) * C + c] *
                                   (int32_t)in[(ih * W + iw) * C + c];
                    }
                out[(p * Q + q) * K + k] = g_sat(acc * mult, shift, relu != 0);
            }
    return out;
}

static std::vector<int8_t> g_eltwise(const std::vector<int8_t> &a,
                                     const std::vector<int8_t> &b,
                                     int mA, int mB, int sh, int relu) {
    std::vector<int8_t> o(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        o[i] = g_sat((int64_t)a[i] * mA + (int64_t)b[i] * mB, sh, relu != 0);
    return o;
}

static std::vector<int8_t> g_maxpool(const std::vector<int8_t> &in, int H,
                                     int W, int C, int R, int S, int stride,
                                     int pad) {
    const int P = (H + 2 * pad - R) / stride + 1;
    const int Q = (W + 2 * pad - S) / stride + 1;
    std::vector<int8_t> out((size_t)P * Q * C);
    for (int p = 0; p < P; ++p)
        for (int q = 0; q < Q; ++q)
            for (int c = 0; c < C; ++c) {
                int mx = OA_MIN;
                for (int r = 0; r < R; ++r)
                    for (int s = 0; s < S; ++s) {
                        const int ih = p * stride - pad + r;
                        const int iw = q * stride - pad + s;
                        if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                        const int v = in[(ih * W + iw) * C + c];
                        if (v > mx) mx = v;
                    }
                out[(p * Q + q) * C + c] = (int8_t)mx;
            }
    return out;
}

// ---------------------------------------------------------------------------
// host <-> device helpers
// ---------------------------------------------------------------------------
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
            if (hw != gold[i * C + c] && errors++ < 3)
                printf("  [%s] MISMATCH pix=%d c=%d hw=%d sw=%d\n",
                       name, i, c, (int)hw, (int)gold[i * C + c]);
        }
    if (errors)
        printf("[E2E] %-24s %dx%dx%-4d FAIL (%d errors)\n", name, t.H, t.W, C,
               errors);
    return errors;
}

// heuristic requant shift keeping activations lively through a deep chain
// (SHIFT_DELTA compensates for the smaller products at lower precision)
static int conv_shift(int C, int R, int S) {
    return 9 + SHIFT_DELTA +
           (int)std::round(std::log2(std::sqrt((double)C * R * S)));
}

// random conv params (device packing) + golden weight copy
static nr::ConvParams make_conv(std::mt19937 &rng, int C, int K, int R, int S,
                                int stride, int pad, int relu,
                                std::vector<int8_t> &w_raw,
                                std::vector<int32_t> &b_raw) {
    std::uniform_int_distribution<int> d8(W_RMIN, W_RMAX);
    std::uniform_int_distribution<int> db(-1000, 1000);
    w_raw.resize((size_t)K * R * S * C);
    b_raw.assign(K, 0);
    for (auto &v : w_raw) v = (int8_t)d8(rng);
    for (auto &v : b_raw) v = db(rng);

    nr::ConvParams cp;
    cp.K = K; cp.R = R; cp.S = S; cp.stride = stride; cp.pad = pad;
    cp.mult = 3; cp.shift = conv_shift(C, R, S); cp.relu = relu;
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

// one conv through DUT + golden, with per-layer compare
static int conv_step(std::mt19937 &rng, const char *name, nr::Tensor &t,
                     std::vector<int8_t> &gold, int &H, int &W, int C, int K,
                     int R, int S, int stride, int pad, int relu) {
    std::vector<int8_t> w_raw;
    std::vector<int32_t> b_raw;
    nr::ConvParams cp = make_conv(rng, C, K, R, S, stride, pad, relu,
                                  w_raw, b_raw);
    gold = g_conv(gold, H, W, C, w_raw, b_raw, K, R, S, stride, pad,
                  cp.mult, cp.shift, relu);
    nr::Tensor out;
    nr::run_conv(t, cp, out);
    t = std::move(out);
    H = (H + 2 * pad - R) / stride + 1;
    W = (W + 2 * pad - S) / stride + 1;
    return compare(name, t, gold, K);
}

// ===========================================================================
// Part 1: full-size dry run (mapping validation for real ResNet-50 shapes)
// ===========================================================================
struct Shape { const char *name; int H, W, C, K, R, S, st, pad; int repeat; };

static int dry_run() {
    // ResNet-50 v1.5, 224x224 input; downsample shortcut convs included
    const Shape shapes[] = {
        {"conv1_7x7",     224, 224,    3,   64, 7, 7, 2, 3, 1},
        {"res2_1x1a",      56,  56,   64,   64, 1, 1, 1, 0, 3},
        {"res2_3x3",       56,  56,   64,   64, 3, 3, 1, 1, 3},
        {"res2_1x1b",      56,  56,   64,  256, 1, 1, 1, 0, 3},
        {"res2_down",      56,  56,   64,  256, 1, 1, 1, 0, 1},
        {"res3_1x1a",      56,  56,  256,  128, 1, 1, 1, 0, 1},
        {"res3_3x3s2",     56,  56,  128,  128, 3, 3, 2, 1, 1},
        {"res3_1x1a_r",    28,  28,  512,  128, 1, 1, 1, 0, 3},
        {"res3_3x3",       28,  28,  128,  128, 3, 3, 1, 1, 3},
        {"res3_1x1b",      28,  28,  128,  512, 1, 1, 1, 0, 4},
        {"res3_down",      56,  56,  256,  512, 1, 1, 2, 0, 1},
        {"res4_1x1a",      28,  28,  512,  256, 1, 1, 1, 0, 1},
        {"res4_3x3s2",     28,  28,  256,  256, 3, 3, 2, 1, 1},
        {"res4_1x1a_r",    14,  14, 1024,  256, 1, 1, 1, 0, 5},
        {"res4_3x3",       14,  14,  256,  256, 3, 3, 1, 1, 5},
        {"res4_1x1b",      14,  14,  256, 1024, 1, 1, 1, 0, 6},
        {"res4_down",      28,  28,  512, 1024, 1, 1, 2, 0, 1},
        {"res5_1x1a",      14,  14, 1024,  512, 1, 1, 1, 0, 1},
        {"res5_3x3s2",     14,  14,  512,  512, 3, 3, 2, 1, 1},
        {"res5_1x1a_r",     7,   7, 2048,  512, 1, 1, 1, 0, 2},
        {"res5_3x3",        7,   7,  512,  512, 3, 3, 1, 1, 2},
        {"res5_1x1b",       7,   7,  512, 2048, 1, 1, 1, 0, 3},
        {"res5_down",      14,  14, 1024, 2048, 1, 1, 2, 0, 1},
    };

    printf("=== Part 1: full-size ResNet-50 dry run (mapping check) ===\n");
    printf("%-14s %5s %5s %6s %5s %4s %12s\n",
           "layer", "KPxCP", "CT1", "C2", "P", "Q", "cycles(x rep)");
    long long total = 0;
    int fails = 0;
    for (const auto &s : shapes) {
        const int P = (s.H + 2 * s.pad - s.R) / s.st + 1;
        const int Q = (s.W + 2 * s.pad - s.S) / s.st + 1;
        const int C1 = (s.C + VECTOR_SIZE - 1) / VECTOR_SIZE;
        const int K1 = (s.K + N_LANES - 1) / N_LANES;
        const nr::ConvPlan pl = nr::plan_conv(C1, K1, P, Q, s.R, s.S, s.st);
        if (!pl.valid) {
            printf("%-14s  NO VALID MAPPING\n", s.name);
            ++fails;
            continue;
        }
        const int c1s = C1 / pl.CP;
        const int C2 = (c1s + pl.CT1 - 1) / pl.CT1;
        printf("%-14s %2dx%-2d %5d %6d %5d %4d %12lld\n",
               s.name, pl.KP, pl.CP, pl.CT1, C2, P, Q,
               pl.est_cycles * s.repeat);
        total += pl.est_cycles * s.repeat;
    }
    printf("dry run: total %lld cycles = %.2f ms @ 200MHz (%.1f fps est.)\n\n",
           total, total / 200e3, 200e6 / (double)total);
    return fails;
}

// ===========================================================================
// Part 2: reduced-size end-to-end (32x32 input, real channel widths)
// ===========================================================================
static int e2e() {
    printf("=== Part 2: reduced-size ResNet-50 e2e (32x32, real channels) ===\n");
    std::mt19937 rng(20260710);
    std::uniform_int_distribution<int> d8(IA_RMIN, IA_RMAX);
    int e = 0;

    // input 32x32xVECTOR_SIZE (real ResNet input is 3 channels padded up to
    // a memory word; use VECTOR_SIZE dense channels — 8 int8 / 16 int4)
    int H = 32, W = 32;
    int C = VECTOR_SIZE;
    std::vector<int8_t> gold((size_t)H * W * C);
    for (auto &v : gold) v = (int8_t)d8(rng);
    nr::Tensor t = to_tensor(gold, H, W, C);

    // ---- stem: conv1 7x7 s2 p3 -> maxpool 3x3 s2 p1 ----
    e += conv_step(rng, "conv1", t, gold, H, W, C, 64, 7, 7, 2, 3, 1);
    {
        gold = g_maxpool(gold, H, W, 64, 3, 3, 2, 1);
        nr::Tensor out;
        nr::run_pool(t, out, GPE_MAXPOOL, 3, 3, 2, 1);
        t = std::move(out);
        H = (H + 2 - 3) / 2 + 1;
        W = (W + 2 - 3) / 2 + 1;
        e += compare("maxpool", t, gold, 64);
    }

    // ---- 4 stages of bottleneck blocks: [3,4,6,3] ----
    const int n_blocks[4] = {3, 4, 6, 3};
    const int c_mid[4] = {64, 128, 256, 512};
    int Cin = 64;
    for (int st = 0; st < 4; ++st) {
        const int Cout = c_mid[st] * 4;
        for (int b = 0; b < n_blocks[st]; ++b) {
            const int stride = (st > 0 && b == 0) ? 2 : 1;
            char nm[64];

            // keep the block input for the shortcut
            nr::Tensor t_in = t;                 // copy (host-side)
            std::vector<int8_t> gold_in = gold;
            const int Hin = H, Win = W;

            snprintf(nm, sizeof nm, "s%d_b%d_1x1a", st + 2, b);
            e += conv_step(rng, nm, t, gold, H, W, Cin, c_mid[st], 1, 1, 1, 0, 1);
            snprintf(nm, sizeof nm, "s%d_b%d_3x3", st + 2, b);
            e += conv_step(rng, nm, t, gold, H, W, c_mid[st], c_mid[st], 3, 3,
                           stride, 1, 1);
            snprintf(nm, sizeof nm, "s%d_b%d_1x1b", st + 2, b);
            e += conv_step(rng, nm, t, gold, H, W, c_mid[st], Cout, 1, 1, 1, 0, 0);

            // shortcut: identity, or 1x1 (stride) downsample conv
            nr::Tensor t_sc;
            std::vector<int8_t> gold_sc;
            if (b == 0) {
                int Hs = Hin, Ws = Win;
                t_sc = std::move(t_in);
                gold_sc = std::move(gold_in);
                snprintf(nm, sizeof nm, "s%d_b%d_down", st + 2, b);
                e += conv_step(rng, nm, t_sc, gold_sc, Hs, Ws, Cin, Cout, 1, 1,
                               stride, 0, 0);
            } else {
                t_sc = std::move(t_in);
                gold_sc = std::move(gold_in);
            }

            // residual add + relu
            const int mA = 180, mB = 90, sh = 8;
            gold = g_eltwise(gold, gold_sc, mA, mB, sh, 1);
            nr::Tensor out;
            nr::run_eltwise(t, t_sc, out, mA, mB, sh, 1);
            t = std::move(out);
            snprintf(nm, sizeof nm, "s%d_b%d_add", st + 2, b);
            e += compare(nm, t, gold, Cout);

            Cin = Cout;
        }
    }

    // ---- global average pool ----
    {
        const int mult = (int)((1 << 15) / (H * W));
        std::vector<int8_t> g(Cin);
        for (int c = 0; c < Cin; ++c) {
            int64_t sum = 0;
            for (int i = 0; i < H * W; ++i) sum += gold[(size_t)i * Cin + c];
            g[c] = g_sat(sum * mult, 15, false);
        }
        gold = g;
        nr::Tensor out;
        nr::run_pool(t, out, GPE_AVGPOOL, H, W, 1, 0, mult, 15);
        t = std::move(out);
        e += compare("global_avgpool", t, gold, Cin);
    }

    printf("e2e: 1 stem + 16 bottleneck blocks + pools, final 1x1x%d\n", Cin);
    return e;
}

int main() {
    int e = dry_run();
    e += e2e();
    printf(e == 0 ? "\n=== RESNET-50 ALL PASSED ===\n"
                  : "\n=== RESNET-50 FAILED: %d errors ===\n", e);
    return e ? 1 : 0;
}
