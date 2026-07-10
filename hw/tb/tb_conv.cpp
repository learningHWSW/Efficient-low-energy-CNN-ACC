// =============================================================================
// tb_conv.cpp — magnet_top C-simulation testbench (multi-PE)
//
// Compares bit-exactly against a plain C++ golden convolution (including int8
// quantization). Each layer runs under three spatial mappings
// (KP x CP = N_PES): K-split (IA multicast), C-split (cross-PE reduction),
// and a mixed configuration.
//
// Run: vitis-run --mode hls --csim (run_hls.ps1 csim), or bundled g++
// (gcc mode)
// =============================================================================

#include "magnet_top.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

struct TestCase {
    const char *name;
    int H, W, C, K;      // actual (pre-padding) dimensions
    int R, S, stride, pad;
    int mult, shift, relu;
    int ct1;             // input-channel tile (words). 0 = whole slice (no tiling)
};

// Same arithmetic as the hardware requantize (round-half-up, saturate)
static int8_t golden_requant(int32_t acc, int mult, int shift, bool relu) {
    int64_t x = (int64_t)acc * (int64_t)mult;
    if (shift > 0)
        x += ((int64_t)1 << (shift - 1));
    x >>= shift;
    if (relu && x < 0) x = 0;
    if (x > 127)  x = 127;
    if (x < -128) x = -128;
    return (int8_t)x;
}

static int run_test(const TestCase &tc, int KP, int CP, unsigned seed) {
    const int P   = (tc.H + 2 * tc.pad - tc.R) / tc.stride + 1;
    const int Q   = (tc.W + 2 * tc.pad - tc.S) / tc.stride + 1;
    const int C1r = (tc.C + VECTOR_SIZE - 1) / VECTOR_SIZE;
    const int C1  = ((C1r + CP - 1) / CP) * CP;   // pad to a CP multiple (host contract)
    const int K1  = (tc.K + N_LANES - 1) / N_LANES;
    const int Kp  = K1 * N_LANES;
    const int c1s = C1 / CP;
    const int CT1 = (tc.ct1 > 0 && tc.ct1 <= c1s) ? tc.ct1 : c1s;

    printf("[TB] %-22s KPxCP=%dx%d  H=%d W=%d C=%d K=%d R=%d S=%d st=%d pad=%d "
           "C1=%d CT1=%d -> P=%d Q=%d\n",
           tc.name, KP, CP, tc.H, tc.W, tc.C, tc.K, tc.R, tc.S, tc.stride,
           tc.pad, C1, CT1, P, Q);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d8(-128, 127);
    std::uniform_int_distribution<int> db(-1000, 1000);

    // ---- host arrays ----
    std::vector<int8_t> ia(tc.H * tc.W * tc.C);
    std::vector<int8_t> w(tc.K * tc.R * tc.S * tc.C);
#ifdef COSIM_SMALL
    std::vector<int32_t> bias(4096, 0);   // sized to the bias m_axi depth
#else
    std::vector<int32_t> bias(Kp, 0);
#endif
    for (auto &v : ia) v = (int8_t)d8(rng);
    for (auto &v : w)  v = (int8_t)d8(rng);
    for (int k = 0; k < tc.K; ++k) bias[k] = db(rng);

    // ---- device arrays (NHWC / KRSC; C1 zero-padded to CP*VECTOR_SIZE) ----
    // The cosim wrapper copies m_axi-depth elements, so allocate at least that
    size_t n_ia = tc.H * tc.W * C1, n_w = (size_t)Kp * tc.R * tc.S * C1;
    size_t n_oa = (size_t)P * Q * K1;
#ifdef COSIM_SMALL
    if (n_ia < GMEM_DEPTH) n_ia = GMEM_DEPTH;
    if (n_w < GMEM_DEPTH)  n_w = GMEM_DEPTH;
    if (n_oa < GMEM_DEPTH) n_oa = GMEM_DEPTH;
#endif
    std::vector<iavec_t> d_ia(n_ia);
    std::vector<wvec_t>  d_w(n_w);
    std::vector<oavec_t> d_oa(n_oa);

    for (auto &word : d_ia)
        for (int v = 0; v < VECTOR_SIZE; ++v) iavec_set(word, v, (ia_t)0);
    for (auto &word : d_w)
        for (int v = 0; v < VECTOR_SIZE; ++v) wvec_set(word, v, (w_t)0);

    for (int h = 0; h < tc.H; ++h)
        for (int x = 0; x < tc.W; ++x)
            for (int c = 0; c < tc.C; ++c)
                iavec_set(d_ia[(h * tc.W + x) * C1 + c / VECTOR_SIZE],
                          c % VECTOR_SIZE,
                          (ia_t)ia[(h * tc.W + x) * tc.C + c]);

    for (int k = 0; k < tc.K; ++k)
        for (int r = 0; r < tc.R; ++r)
            for (int s = 0; s < tc.S; ++s)
                for (int c = 0; c < tc.C; ++c)
                    wvec_set(d_w[((k * tc.R + r) * tc.S + s) * C1 + c / VECTOR_SIZE],
                             c % VECTOR_SIZE,
                             (w_t)w[((k * tc.R + r) * tc.S + s) * tc.C + c]);

    // ---- golden model ----
    std::vector<int8_t> gold(P * Q * tc.K);
    for (int p = 0; p < P; ++p)
        for (int q = 0; q < Q; ++q)
            for (int k = 0; k < tc.K; ++k) {
                int32_t acc = bias[k];
                for (int r = 0; r < tc.R; ++r)
                    for (int s = 0; s < tc.S; ++s) {
                        const int ih = p * tc.stride - tc.pad + r;
                        const int iw = q * tc.stride - tc.pad + s;
                        if (ih < 0 || ih >= tc.H || iw < 0 || iw >= tc.W)
                            continue;
                        for (int c = 0; c < tc.C; ++c)
                            acc += (int32_t)w[((k * tc.R + r) * tc.S + s) * tc.C + c] *
                                   (int32_t)ia[(ih * tc.W + iw) * tc.C + c];
                    }
                gold[(p * Q + q) * tc.K + k] =
                    golden_requant(acc, tc.mult, tc.shift, tc.relu != 0);
            }

    // ---- run DUT (the four IA pointers alias the same buffer) ----
    magnet_top(d_ia.data(), d_ia.data(), d_ia.data(), d_ia.data(),
               d_w.data(), d_oa.data(), bias.data(),
               tc.H, tc.W, C1, K1, tc.K, P, Q, tc.R, tc.S,
               tc.stride, tc.pad, CT1, KP, CP, tc.mult, tc.shift, tc.relu);

    // ---- compare ----
    int errors = 0;
    for (int p = 0; p < P; ++p)
        for (int q = 0; q < Q; ++q)
            for (int k = 0; k < tc.K; ++k) {
                const int8_t hw =
                    (int8_t)oavec_get(d_oa[(p * Q + q) * K1 + k / N_LANES],
                                      k % N_LANES);
                const int8_t sw = gold[(p * Q + q) * tc.K + k];
                if (hw != sw && errors++ < 10)
                    printf("  MISMATCH p=%d q=%d k=%d : hw=%d gold=%d\n",
                           p, q, k, (int)hw, (int)sw);
            }

    printf("  -> %s (%d errors)\n", errors ? "FAIL" : "PASS", errors);
    return errors;
}

int main() {
#ifdef COSIM_SMALL
    // Reduced set for RTL co-simulation: small sizes covering QB padding,
    // C-tiling, K padding, and stride
    const TestCase tests[] = {
        // name                     H   W   C    K   R  S  st pad  mult shift relu ct1
        {"cs_3x3_s1_p1",            8,  8, 16,  16,  3, 3, 1, 1,    5,  10,  1,  0}, // Q=8<16
        {"cs_1x1",                  8,  8, 32,  16,  1, 1, 1, 0,    9,  11,  0,  0},
        {"cs_ctile_s2_K20",         9,  9, 32,  20,  3, 3, 2, 1,   17,  11,  1,  1}, // C2>1, K padding
    };
#else
    const TestCase tests[] = {
        // name                     H   W   C    K   R  S  st pad  mult shift relu ct1
        {"3x3_s1_p1_basic",        16, 16, 16,  16,  3, 3, 1, 1,    5,  10,  1,  0},
        {"1x1_pointwise",          16, 16, 64,  32,  1, 1, 1, 0,    9,  11,  0,  0},
        {"3x3_s2_downsample",      15, 15,  8,   8,  3, 3, 2, 1,   17,  10,  1,  0},
        {"7x7_s2_p3_stem",         32, 32,  3,   8,  7, 7, 2, 3,    3,  10,  1,  0},
        {"small_Q_lt_MIN_QB",       8,  8, 16,  16,  3, 3, 1, 1,    5,   9,  0,  0},
        {"K_not_multiple_lanes",   12, 12, 24,  20,  3, 3, 1, 1,    7,  10,  1,  0},
        {"C_not_multiple_vec",     12, 12, 12,   8,  3, 3, 1, 1,    7,  10,  0,  0},
        {"5x5_s1_p2",              14, 14, 16,  16,  5, 5, 1, 2,    3,  11,  1,  0},
        {"ctile_3x3_C48_CT2",      12, 12, 48,  16,  3, 3, 1, 1,    5,  11,  1,  2},
        {"ctile_nondiv_C40_CT2",   12, 12, 40,  16,  3, 3, 1, 1,    5,  11,  0,  2},
        {"ctile_1x1_C128_CT4",     16, 16, 128, 24,  1, 1, 1, 0,    9,  12,  1,  4},
        {"rowgroup_P13",           13, 13, 16,   8,  3, 3, 1, 1,    5,  10,  1,  0},
    };
#endif
    // KP x CP = N_PES: K-split / mixed / C-split
    const int spatial[][2] = {{N_PES, 1}, {4, N_PES / 4}, {1, N_PES}};

    int total_errors = 0;
    unsigned seed = 20260709;
    for (const auto &sp : spatial)
        for (const auto &tc : tests)
            total_errors += run_test(tc, sp[0], sp[1], seed++);

    if (total_errors == 0)
        printf("\n=== ALL TESTS PASSED ===\n");
    else
        printf("\n=== FAILED: %d total errors ===\n", total_errors);
    return total_errors ? 1 : 0;
}
