// =============================================================================
// tb_global_pe.cpp — Global PE C-simulation testbench
//
// Compares eltwise add / maxpool / avgpool bit-exactly against golden models.
// Run: run_hls.ps1 gpe-csim (Vitis) or gcc mode
// =============================================================================

#include "global_pe.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>

static int8_t golden_sat_shift(int64_t x, int shift, bool relu) {
    if (shift > 0)
        x += ((int64_t)1 << (shift - 1));
    x >>= shift;
    if (relu && x < 0) x = 0;
    if (x > 127)  x = 127;
    if (x < -128) x = -128;
    return (int8_t)x;
}

static void pack(std::vector<oavec_t> &dst, const std::vector<int8_t> &src,
                 int n_words) {
    for (int i = 0; i < n_words; ++i)
        for (int l = 0; l < N_LANES; ++l)
            oavec_set(dst[i], l, (oa_t)src[i * N_LANES + l]);
}

static int test_eltwise(int H, int W, int C1, int multA, int multB, int shift,
                        int relu, unsigned seed) {
    const int n = H * W * C1;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d8(-128, 127);
    std::vector<int8_t> a(n * N_LANES), b(n * N_LANES);
    for (auto &v : a) v = (int8_t)d8(rng);
    for (auto &v : b) v = (int8_t)d8(rng);

    std::vector<oavec_t> d_a(n), d_b(n), d_o(n);
    pack(d_a, a, n);
    pack(d_b, b, n);

    global_pe_top(d_a.data(), d_b.data(), d_o.data(), GPE_ELTWISE,
                  H, W, C1, 1, 1, 1, 0, multA, multB, shift, relu);

    int errors = 0;
    for (int i = 0; i < n; ++i)
        for (int l = 0; l < N_LANES; ++l) {
            const int8_t sw = golden_sat_shift(
                (int64_t)a[i * N_LANES + l] * multA +
                (int64_t)b[i * N_LANES + l] * multB, shift, relu != 0);
            const int8_t hw = (int8_t)oavec_get(d_o[i], l);
            if (hw != sw && errors++ < 5)
                printf("  ELT MISMATCH i=%d l=%d hw=%d sw=%d\n", i, l, hw, sw);
        }
    printf("[TB] eltwise %dx%dx%d relu=%d -> %s (%d errors)\n",
           H, W, C1, relu, errors ? "FAIL" : "PASS", errors);
    return errors;
}

static int test_pool(int mode, int H, int W, int C1, int R, int S, int stride,
                     int pad, int multA, int shift, unsigned seed) {
    const int P = (H + 2 * pad - R) / stride + 1;
    const int Q = (W + 2 * pad - S) / stride + 1;
    const int n = H * W * C1;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d8(-128, 127);
    std::vector<int8_t> a(n * N_LANES);
    for (auto &v : a) v = (int8_t)d8(rng);

    std::vector<oavec_t> d_a(n), d_o(P * Q * C1);
    pack(d_a, a, n);

    global_pe_top(d_a.data(), d_a.data(), d_o.data(), mode,
                  H, W, C1, R, S, stride, pad, multA, 0, shift, 0);

    int errors = 0;
    for (int p = 0; p < P; ++p)
        for (int q = 0; q < Q; ++q)
            for (int c1 = 0; c1 < C1; ++c1)
                for (int l = 0; l < N_LANES; ++l) {
                    int32_t mx = -128, sum = 0;
                    for (int r = 0; r < R; ++r)
                        for (int s = 0; s < S; ++s) {
                            const int ih = p * stride - pad + r;
                            const int iw = q * stride - pad + s;
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W)
                                continue;
                            const int32_t x =
                                a[((ih * W + iw) * C1 + c1) * N_LANES + l];
                            mx = std::max(mx, x);
                            sum += x;
                        }
                    const int8_t sw = (mode == GPE_MAXPOOL)
                        ? (int8_t)mx
                        : golden_sat_shift((int64_t)sum * multA, shift, false);
                    const int8_t hw =
                        (int8_t)oavec_get(d_o[(p * Q + q) * C1 + c1], l);
                    if (hw != sw && errors++ < 5)
                        printf("  POOL MISMATCH p=%d q=%d c=%d l=%d hw=%d sw=%d\n",
                               p, q, c1, l, hw, sw);
                }
    printf("[TB] %s %dx%dx%d R=%d S=%d st=%d pad=%d -> P=%d Q=%d %s (%d errors)\n",
           mode == GPE_MAXPOOL ? "maxpool" : "avgpool",
           H, W, C1, R, S, stride, pad, P, Q,
           errors ? "FAIL" : "PASS", errors);
    return errors;
}

int main() {
    int e = 0;
    unsigned seed = 20260709;
    // eltwise (residual add)
    e += test_eltwise(16, 16, 8, 77, 51, 8, 1, seed++);
    e += test_eltwise(16, 16, 8, 128, 256, 9, 0, seed++);
    e += test_eltwise(7, 7, 32, 1, 1, 1, 1, seed++);
    // maxpool: ResNet stem 3x3 s2 p1, VGG 2x2 s2
    e += test_pool(GPE_MAXPOOL, 16, 16, 4, 3, 3, 2, 1, 0, 0, seed++);
    e += test_pool(GPE_MAXPOOL, 16, 16, 2, 2, 2, 2, 0, 0, 0, seed++);
    // avgpool: global 7x7 (multA = round(2^15/49), shift=15)
    e += test_pool(GPE_AVGPOOL, 7, 7, 8, 7, 7, 1, 0, 669, 15, seed++);
    // avgpool 3x3 s1 p1 (count_include_pad=True style: fixed divisor of 9)
    e += test_pool(GPE_AVGPOOL, 12, 12, 2, 3, 3, 1, 1, 3641, 15, seed++);

    printf(e == 0 ? "=== ALL TESTS PASSED ===\n"
                  : "=== FAILED: %d errors ===\n", e);
    return e ? 1 : 0;
}
