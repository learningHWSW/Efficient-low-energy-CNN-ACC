// =============================================================================
// magnet_top.cpp — MAGNet-style CNN accelerator (single PE, OS-LWS dataflow)
//
// [ARCHIVED Phase 2 version — kept for reference; superseded by the multi-PE
//  hw/src/magnet_top.cpp and no longer built. Uses the old LayerCfg without
//  KP/CP.]
//
// Phase 2: C-tiling + double buffering (dataflow) + DSP packing
//
// Implements the PE of the MAGNet paper (Venkatesan et al., ICCAD 2019)
// Fig.2(b) in Vitis HLS.
//   - Vector MAC unit : VECTOR_SIZE-wide dot product x N_LANES lanes
//                       (DSP packing: adjacent lane pairs share one DSP, WP487)
//   - Weight buffer   : per-lane banks, (k1, c2) tile cache
//   - IA row buffer   : dataflow PIPO — load(p+1) overlaps compute(p)
//   - Accum buffer    : psums of a PT_ROWS row group, resident across the c2
//                       loop (RMW)
//   - PPU             : bias + requantization (scale/round) + ReLU + int8 sat
//
// Dataflow = OS-LWS (paper Fig.4(e)). Temporal order:
//   k1 -> p1 (row group) -> c2 (channel tile) -> p0 -> (r,s,c1) -> q0 (innermost)
//   A weight word stays fixed per (r,s,c1) and is reused across the q0 loop
//   (LWS); partial sums stay in acc_buf and are reused R*S*C1 times
//   (including all c2 tiles) (OS).
//
// Constraints (checked by mapper.py):
//   R*S*CT1             <= WBUF_DEPTH
//   R*win_cols*CT1      <= IABUF_DEPTH   (win_cols = (QB-1)*stride + S)
//   Q                   <= Q_MAX,  CT1 <= C1
//
// The algorithm corresponds 1:1 to sw/model/reference_model.py (bit-exactly
// verified).
// =============================================================================

#include "magnet_top.h"

#ifndef __SYNTHESIS__
#include <cassert>
#endif

// ---------------------------------------------------------------------------
// DSP-packed MAC: two products from one 27x18 multiply (DSP48E2/DSP58)
//   pr = ((w1<<18) + w0) * a
//   lo = sign_extend_18(pr[17:0]) == w0*a,  hi = (pr>>18) + (lo<0) == w1*a
// Exhaustively verified over all (w0,w1,a) combinations
// (scratch: verify_dsp_pack.py)
// ---------------------------------------------------------------------------
static void mac_pair(w_t w0, w_t w1, ia_t a, acc_t &prod0, acc_t &prod1) {
#pragma HLS inline
#ifdef USE_AP_TYPES
    ap_int<27> wp = ((ap_int<27>)w1 << 18) + (ap_int<27>)w0;
    ap_int<36> pr = wp * a;
    ap_int<18> lo = pr.range(17, 0);          // low 18b reinterpreted == w0*a
    ap_int<18> hi = (ap_int<18>)(pr >> 18) + (ap_uint<1>)lo[17];
    prod0 = (acc_t)lo;
    prod1 = (acc_t)hi;
#else
    const int64_t pr = (((int64_t)w1 << 18) + (int64_t)w0) * (int64_t)a;
    const int32_t lo_raw = (int32_t)(pr & 0x3FFFF);
    const int32_t lo = (lo_raw >= (1 << 17)) ? lo_raw - (1 << 18) : lo_raw;
    const int32_t hi = (int32_t)(pr >> 18) + (lo < 0 ? 1 : 0);
    prod0 = lo;
    prod1 = hi;
#endif
}

// ---------------------------------------------------------------------------
// PPU: requantize int32 -> int8  (scale = mult / 2^shift, round-half-up)
// ---------------------------------------------------------------------------
static oa_t requantize(acc_t acc, int mult, int shift, bool relu) {
#pragma HLS inline
    wide_t x = (wide_t)acc * (wide_t)mult;
    if (shift > 0)
        x += ((wide_t)1 << (shift - 1));
    x >>= shift;
    if (relu && x < 0) x = 0;
    if (x > 127)  x = 127;
    if (x < -128) x = -128;
    return (oa_t)x;
}

// ---------------------------------------------------------------------------
// Weight load: (k1, c2) tile — per-lane banks, ct1-word bursts per (r,s)
//   gmem_w layout: [Kpad][R][S][C1]
// ---------------------------------------------------------------------------
static void load_weights(const wvec_t *gmem_w, wvec_t wbuf[WBUF_DEPTH][N_LANES],
                         const LayerCfg &cfg, int k1, int c2, int ct1) {
    const int rs_total = cfg.R * cfg.S;
LOAD_W_LANE:
    for (int lane = 0; lane < N_LANES; ++lane) {
        const int k = k1 * N_LANES + lane;
    LOAD_W_RS:
        for (int rs = 0; rs < rs_total; ++rs) {
#pragma HLS loop_tripcount min = 1 max = 49
            const int src = (k * rs_total + rs) * cfg.C1 + c2 * cfg.CT1;
            const int dst = rs * ct1;
        LOAD_W_BURST:
            for (int i = 0; i < ct1; ++i) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1 max = 256
                wbuf[dst + i][lane] = gmem_w[src + i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// IA load: loads the R input rows needed for output row p (channel-tile c2
// portion), including zero padding.
//   rowbuf address = (r*win_cols + col)*ct1 + c1t
// ---------------------------------------------------------------------------
static void load_ia_rows(const iavec_t *gmem_ia, iavec_t rowbuf[IABUF_DEPTH],
                         LayerCfg cfg, int p, int c2, int ct1, int win_cols) {
    const int row_words = win_cols * ct1;

LOAD_IA_ROW:
    for (int r = 0; r < cfg.R; ++r) {
#pragma HLS loop_tripcount min = 1 max = 7
        const int ih   = p * cfg.stride - cfg.pad + r;
        const int base = r * row_words;

        // 1) zero-fill the whole row (handles padding + unloaded regions)
    IA_ZERO:
        for (int i = 0; i < row_words; ++i) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 16 max = 8192
            iavec_t z;
            for (int v = 0; v < VECTOR_SIZE; ++v) {
#pragma HLS unroll
                iavec_set(z, v, (ia_t)0);
            }
            rowbuf[base + i] = z;
        }

        if (ih < 0 || ih >= cfg.H)
            continue;

        int cols = cfg.W;
        if (cols > win_cols - cfg.pad) cols = win_cols - cfg.pad;

        if (ct1 == cfg.C1) {
            // fast path (C2==1): the whole row is contiguous -> one burst
            const int n   = cols * cfg.C1;
            const int src = (ih * cfg.W) * cfg.C1;
            const int dst = base + cfg.pad * cfg.C1;
        IA_COPY_ROW:
            for (int i = 0; i < n; ++i) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 16 max = 8192
                rowbuf[dst + i] = gmem_ia[src + i];
            }
        } else {
            // C-tile path: ct1-word burst per pixel (stride of C1 between)
        IA_COPY_PX:
            for (int x = 0; x < cols; ++x) {
#pragma HLS loop_tripcount min = 1 max = 256
                const int src = (ih * cfg.W + x) * cfg.C1 + c2 * cfg.CT1;
                const int dst = base + (cfg.pad + x) * ct1;
            IA_COPY_CT:
                for (int i = 0; i < ct1; ++i) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1 max = 256
                    rowbuf[dst + i] = gmem_ia[src + i];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Compute: computes output row p0 (within the group) in OS-LWS order
//   flattened loop: (r, s, c1) x q0 with q0 innermost
//   Accumulates psums into acc_buf[p0][q][lane]; initialized with bias on the
//   first pass of c2==0.
// ---------------------------------------------------------------------------
static void compute_row(const wvec_t wbuf[WBUF_DEPTH][N_LANES],
                        const iavec_t rowbuf[IABUF_DEPTH],
                        acc_t acc_buf[PT_ROWS][Q_MAX][N_LANES],
                        const bias_t bias_reg[N_LANES],
                        LayerCfg cfg, int p0, int qb, int ct1, int win_cols,
                        bool first_c2) {
    const int rsct  = cfg.R * cfg.S * ct1;
    const int total = rsct * qb;

    int r = 0, s = 0, c1 = 0, q = 0;
    int rsc = 0; // linear index of (r,s,c1) = weight buffer address

MAC_LOOP:
    for (int it = 0; it < total; ++it) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 1024 max = 262144
#pragma HLS dependence variable = acc_buf type = inter direction = RAW distance = 16 dependent = true
#pragma HLS dependence variable = acc_buf type = inter direction = WAW distance = 16 dependent = true
        const int col     = q * cfg.stride + s;
        const int ia_addr = (r * win_cols + col) * ct1 + c1;
        const iavec_t iav = rowbuf[ia_addr];

        const bool first = first_c2 && (rsc == 0);

#if USE_DSP_PACKING
    PAIR:
        for (int j = 0; j < N_LANES / 2; ++j) {
#pragma HLS unroll
            acc_t d0 = 0, d1 = 0;
        DOT2:
            for (int v = 0; v < VECTOR_SIZE; ++v) {
#pragma HLS unroll
                acc_t lo, hi;
                mac_pair(wvec_get(wbuf[rsc][2 * j], v),
                         wvec_get(wbuf[rsc][2 * j + 1], v),
                         iavec_get(iav, v), lo, hi);
                d0 += lo;
                d1 += hi;
            }
            const acc_t prev0 =
                first ? (acc_t)bias_reg[2 * j] : acc_buf[p0][q][2 * j];
            const acc_t prev1 =
                first ? (acc_t)bias_reg[2 * j + 1] : acc_buf[p0][q][2 * j + 1];
            acc_buf[p0][q][2 * j]     = prev0 + d0;
            acc_buf[p0][q][2 * j + 1] = prev1 + d1;
        }
#else
    LANE:
        for (int lane = 0; lane < N_LANES; ++lane) {
#pragma HLS unroll
            acc_t dot = 0;
        DOT:
            for (int v = 0; v < VECTOR_SIZE; ++v) {
#pragma HLS unroll
                dot += (acc_t)(wvec_get(wbuf[rsc][lane], v) * iavec_get(iav, v));
            }
            const acc_t prev =
                first ? (acc_t)bias_reg[lane] : acc_buf[p0][q][lane];
            acc_buf[p0][q][lane] = prev + dot;
        }
#endif

        // counter update: q0 innermost -> c1 -> s -> r
        if (q == qb - 1) {
            q = 0;
            rsc++;
            if (c1 == ct1 - 1) {
                c1 = 0;
                if (s == cfg.S - 1) { s = 0; r++; }
                else                { s++; }
            } else {
                c1++;
            }
        } else {
            q++;
        }
    }
}

// ---------------------------------------------------------------------------
// PPU + store: valid rows of the group -> requant -> pack -> DRAM
//   gmem_oa layout: [P][Q][K1], word = N_LANES int8
// ---------------------------------------------------------------------------
static void ppu_store_group(oavec_t *gmem_oa,
                            const acc_t acc_buf[PT_ROWS][Q_MAX][N_LANES],
                            const LayerCfg &cfg, int k1, int p1, int rows) {
STORE_ROW:
    for (int p0 = 0; p0 < rows; ++p0) {
#pragma HLS loop_tripcount min = 1 max = 8
        const int p = p1 * PT_ROWS + p0;
    PPU_STORE:
        for (int q = 0; q < cfg.Q; ++q) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 8 max = 256
            oavec_t ov;
            for (int lane = 0; lane < N_LANES; ++lane) {
#pragma HLS unroll
                oavec_set(ov, lane,
                          requantize(acc_buf[p0][q][lane], cfg.mult, cfg.shift,
                                     cfg.relu_en != 0));
            }
            gmem_oa[(p * cfg.Q + q) * cfg.K1 + k1] = ov;
        }
    }
}

// ===========================================================================
// Top level
// ===========================================================================
extern "C" void magnet_top(
    const iavec_t *gmem_ia,
    const wvec_t  *gmem_w,
    oavec_t       *gmem_oa,
    const bias_t  *gmem_bias,
    int H, int W, int C1, int K1, int K,
    int P, int Q, int R, int S,
    int stride, int pad, int CT1,
    int mult, int shift, int relu_en) {
#pragma HLS INTERFACE m_axi port = gmem_ia   offset = slave bundle = gmem0 depth = 1048576
#pragma HLS INTERFACE m_axi port = gmem_w    offset = slave bundle = gmem1 depth = 1048576
#pragma HLS INTERFACE m_axi port = gmem_oa   offset = slave bundle = gmem2 depth = 1048576
#pragma HLS INTERFACE m_axi port = gmem_bias offset = slave bundle = gmem3 depth = 4096
#pragma HLS INTERFACE s_axilite port = H
#pragma HLS INTERFACE s_axilite port = W
#pragma HLS INTERFACE s_axilite port = C1
#pragma HLS INTERFACE s_axilite port = K1
#pragma HLS INTERFACE s_axilite port = K
#pragma HLS INTERFACE s_axilite port = P
#pragma HLS INTERFACE s_axilite port = Q
#pragma HLS INTERFACE s_axilite port = R
#pragma HLS INTERFACE s_axilite port = S
#pragma HLS INTERFACE s_axilite port = stride
#pragma HLS INTERFACE s_axilite port = pad
#pragma HLS INTERFACE s_axilite port = CT1
#pragma HLS INTERFACE s_axilite port = mult
#pragma HLS INTERFACE s_axilite port = shift
#pragma HLS INTERFACE s_axilite port = relu_en
#pragma HLS INTERFACE s_axilite port = return

    LayerCfg cfg;
    cfg.H = H;   cfg.W = W;   cfg.C1 = C1; cfg.K1 = K1; cfg.K = K;
    cfg.P = P;   cfg.Q = Q;   cfg.R = R;   cfg.S = S;
    cfg.stride = stride;      cfg.pad = pad;    cfg.CT1 = CT1;
    cfg.mult = mult;          cfg.shift = shift;  cfg.relu_en = relu_en;

    // Layers with short Q are padded to MIN_QB (guarantees the acc_buf RMW
    // distance; the excess is discarded)
    const int qb       = (Q < MIN_QB) ? MIN_QB : Q;
    const int win_cols = (qb - 1) * stride + S;
    const int C2       = (C1 + CT1 - 1) / CT1;        // input-channel tiles
    const int P2       = (P + PT_ROWS - 1) / PT_ROWS; // row groups

#ifndef __SYNTHESIS__
    assert(R * S * CT1 <= WBUF_DEPTH);
    assert(R * win_cols * CT1 <= IABUF_DEPTH);
    assert(Q <= Q_MAX && qb <= Q_MAX);
    assert(CT1 >= 1 && CT1 <= C1);
    assert(K <= K1 * N_LANES);
    assert(N_LANES % 2 == 0);
#endif

    // ---- PE local memories (MAGNet Fig.2(b)) ----
    static wvec_t wbuf[WBUF_DEPTH][N_LANES];
#pragma HLS bind_storage variable = wbuf type = ram_2p impl = bram
#pragma HLS array_partition variable = wbuf complete dim = 2
    static acc_t acc_buf[PT_ROWS][Q_MAX][N_LANES];
#pragma HLS bind_storage variable = acc_buf type = ram_2p impl = bram
#pragma HLS array_partition variable = acc_buf complete dim = 3
    bias_t bias_reg[N_LANES];
#pragma HLS array_partition variable = bias_reg complete

    // weight tile cache tag (loads once per k1 when C2==1)
    int w_tag_k1 = -1, w_tag_c2 = -1;

K1_LOOP:
    for (int k1 = 0; k1 < K1; ++k1) {
#pragma HLS loop_tripcount min = 1 max = 64
    LOAD_BIAS:
        for (int lane = 0; lane < N_LANES; ++lane) {
#pragma HLS pipeline II = 1
            bias_reg[lane] = gmem_bias[k1 * N_LANES + lane];
        }

    P1_LOOP:
        for (int p1 = 0; p1 < P2; ++p1) {
#pragma HLS loop_tripcount min = 1 max = 32
            int rows = P - p1 * PT_ROWS;
            if (rows > PT_ROWS) rows = PT_ROWS;

        C2_LOOP:
            for (int c2 = 0; c2 < C2; ++c2) {
#pragma HLS loop_tripcount min = 1 max = 16
                int ct1 = C1 - c2 * CT1;
                if (ct1 > CT1) ct1 = CT1;

                if (k1 != w_tag_k1 || c2 != w_tag_c2) {
                    load_weights(gmem_w, wbuf, cfg, k1, c2, ct1);
                    w_tag_k1 = k1;
                    w_tag_c2 = c2;
                }

                // p0 loop: dataflow — load_ia(p0+1) overlaps compute(p0).
                // Runs with a fixed bound (PT_ROWS) for canonical form;
                // rows with p >= P (ih out of range -> zeros) are computed
                // and then dropped at store time.
            P0_LOOP:
                for (int p0 = 0; p0 < PT_ROWS; ++p0) {
#pragma HLS dataflow
                    iavec_t rowbuf[IABUF_DEPTH]; // per-iteration PIPO (x2 buffer)
#pragma HLS bind_storage variable = rowbuf type = ram_2p impl = bram
                    load_ia_rows(gmem_ia, rowbuf, cfg, p1 * PT_ROWS + p0, c2,
                                 ct1, win_cols);
                    compute_row(wbuf, rowbuf, acc_buf, bias_reg, cfg, p0, qb,
                                ct1, win_cols, c2 == 0);
                }
            }

            ppu_store_group(gmem_oa, acc_buf, cfg, k1, p1, rows);
        }
    }
}
