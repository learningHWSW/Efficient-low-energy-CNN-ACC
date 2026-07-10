// =============================================================================
// magnet_top.cpp — MAGNet-style CNN accelerator (multi-PE, N_PES processing
// elements)
//
// A scaled-down version of the Simba (MICRO 2019) chiplet implemented as an
// HLS dataflow pipeline:
//
//   ia_loader ──ia_st[pe]──▶ ┌─────┐
//   (multicast / C-slice)    │ PE0 │─psum_st[0]─┐
//   w_loader  ──w_st[pe]───▶ │ ... │            ├──▶ gather_store
//   (K/C tile distribution)  │ PEn │─psum_st[n]─┘    (cross-PE reduction
//                            └─────┘                  + bias + PPU + store)
//
// Spatial mapping (runtime, Simba Listing 2 chiplet-level parallel_for):
//   KP x CP = N_PES.  pe = kp*CP + cp
//   kp: split along K — PE groups work on different k1 tiles (IA is multicast)
//   cp: split along C — PEs within a group share input-channel slices; their
//       psums are summed in gather (= Simba's cross-PE partial-sum reduction)
//
// Each PE runs the same OS-LWS engine as Phase 2 (C-tiling, DSP packing).
// Bias is added exactly once in gather after the reduction (PEs init to 0).
//
// Constraints (checked by mapper.py; every task derives loop bounds from the
// same formulas = lockstep):
//   KP*CP == N_PES,  C1 % CP == 0 (host padding),  CT1 <= C1/CP
//   R*S*CT1 <= WBUF_DEPTH,  R*win_cols*CT1 <= IABUF_DEPTH,  Q <= Q_MAX
//
// Algorithm and stream word counts correspond 1:1 to
// sw/model/reference_model.py (verified incl. stream-drain checks).
// =============================================================================

#include "magnet_top.h"

#ifndef __SYNTHESIS__
#include <cassert>
#endif

// ---------------------------------------------------------------------------
// Shared derived parameters (all tasks must use the same formulas so that
// stream word counts match)
// ---------------------------------------------------------------------------
struct Derived {
    int qb;        // max(Q, MIN_QB)
    int win_cols;  // (qb-1)*stride + S
    int c1s;       // C1 / CP — C-slice words per PE
    int C2;        // ceil(c1s / CT1) — channel tiles per slice
    int P2;        // ceil(P / PT_ROWS)
    int K1G;       // ceil(K1 / KP)
};

static Derived derive(const LayerCfg &cfg) {
#pragma HLS inline
    Derived d;
    d.qb = (cfg.Q < MIN_QB) ? MIN_QB : cfg.Q;
    d.win_cols = (d.qb - 1) * cfg.stride + cfg.S;
    d.c1s = cfg.C1 / cfg.CP;
    d.C2 = (d.c1s + cfg.CT1 - 1) / cfg.CT1;
    d.P2 = (cfg.P + PT_ROWS - 1) / PT_ROWS;
    d.K1G = (cfg.K1 + cfg.KP - 1) / cfg.KP;
    return d;
}

static int ct1_of(int c2, int c1s, int CT1) {
#pragma HLS inline
    int ct1 = c1s - c2 * CT1;
    return (ct1 > CT1) ? CT1 : ct1;
}

// Write to a stream array with a runtime index (resolved as unrolled
// constant-index compares)
template <typename T>
static void st_write(hls::stream<T> st[N_PES], int idx, const T &v) {
#pragma HLS inline
    for (int i = 0; i < N_PES; ++i) {
#pragma HLS unroll
        if (i == idx) st[i].write(v);
    }
}

// ---------------------------------------------------------------------------
// DSP-packed MAC (same as Phase 2, exhaustively verified)
// ---------------------------------------------------------------------------
static void mac_pair(w_t w0, w_t w1, ia_t a, acc_t &prod0, acc_t &prod1) {
#pragma HLS inline
#ifdef USE_AP_TYPES
    ap_int<27> wp = ((ap_int<27>)w1 << 18) + (ap_int<27>)w0;
    ap_int<36> pr = wp * a;
    ap_int<18> lo = pr.range(17, 0);
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

// ===========================================================================
// Task 1: IA loader — reads row windows and distributes C-slices to the PEs.
//   K-split PEs (same cp) receive the same word via multicast (one DRAM read).
//   With C-split, all CP slices of a (r,col,i) position are read in the SAME
//   iteration through N_IA_PORTS parallel AXI masters (all pointing at the
//   same buffer) — loader time scales with ceil(CP/N_IA_PORTS) instead of CP.
//   Per-PE send order is unchanged: r -> col -> i (linear rowbuf order).
// ===========================================================================
static void ia_loader(const iavec_t *gmem_ia0, const iavec_t *gmem_ia1,
                      const iavec_t *gmem_ia2, const iavec_t *gmem_ia3,
                      hls::stream<iavec_t> ia_st[N_PES], LayerCfg cfg) {
    const Derived d = derive(cfg);
    const iavec_t *ports[N_IA_PORTS] = {gmem_ia0, gmem_ia1, gmem_ia2, gmem_ia3};

IA_K1G:
    for (int k1g = 0; k1g < d.K1G; ++k1g) {
#pragma HLS loop_tripcount min = 1 max = 16
    IA_P1:
        for (int p1 = 0; p1 < d.P2; ++p1) {
#pragma HLS loop_tripcount min = 1 max = 32
        IA_C2:
            for (int c2 = 0; c2 < d.C2; ++c2) {
#pragma HLS loop_tripcount min = 1 max = 16
                const int ct1 = ct1_of(c2, d.c1s, cfg.CT1);
            IA_P0:
                for (int p0 = 0; p0 < PT_ROWS; ++p0) {
                    const int p = p1 * PT_ROWS + p0;
                    // flattened: r -> col -> i -> beat. Each beat serves
                    // N_IA_PORTS slices, one read per port per iteration
                    // (keeps II=1; loader time = R*win_cols*ct1*cp_beats).
                    const int cp_beats =
                        (cfg.CP + N_IA_PORTS - 1) / N_IA_PORTS;
                    const int total = cfg.R * d.win_cols * ct1 * cp_beats;
                    int r = 0, col = 0, i = 0, b = 0;
                IA_SEND:
                    for (int it = 0; it < total; ++it) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 64 max = 16384
                        const int ih = p * cfg.stride - cfg.pad + r;
                        const int w  = col - cfg.pad;
                        const bool valid = (ih >= 0) && (ih < cfg.H) &&
                                           (w >= 0) && (w < cfg.W);
                        const int pix_base = (ih * cfg.W + w) * cfg.C1 +
                                             c2 * cfg.CT1 + i;
                        // one read per port into registers (exactly one
                        // static reader per m_axi bundle)
                        iavec_t port_word[N_IA_PORTS];
#pragma HLS array_partition variable = port_word complete
                    IA_PORT:
                        for (int pp = 0; pp < N_IA_PORTS; ++pp) {
#pragma HLS unroll
                            const int cp = b * N_IA_PORTS + pp;
                            if (valid && cp < cfg.CP) {
                                port_word[pp] = ports[pp][pix_base + cp * d.c1s];
                            } else {
                                for (int v = 0; v < VECTOR_SIZE; ++v) {
#pragma HLS unroll
                                    iavec_set(port_word[pp], v, (ia_t)0);
                                }
                            }
                        }
                        // exactly ONE static write site per PE stream
                        // (keeps the FIFO write port conflict-free -> II=1).
                        // CP is a power of two, so pe % CP == pe & (CP-1).
                    IA_MCAST:
                        for (int pe = 0; pe < N_PES; ++pe) {
#pragma HLS unroll
                            const int cp_pe = pe & (cfg.CP - 1);
                            if (cp_pe >= b * N_IA_PORTS &&
                                cp_pe < b * N_IA_PORTS + N_IA_PORTS) {
                                ia_st[pe].write(
                                    port_word[cp_pe - b * N_IA_PORTS]);
                            }
                        }
                        // counters: b -> i -> col -> r
                        if (b == cp_beats - 1) {
                            b = 0;
                            if (i == ct1 - 1) {
                                i = 0;
                                if (col == d.win_cols - 1) { col = 0; r++; }
                                else                        { col++; }
                            } else {
                                i++;
                            }
                        } else {
                            b++;
                        }
                    }
                }
            }
        }
    }
}

// ===========================================================================
// Task 2: Weight loader — distributes the (k1g, c2) tile to each PE.
//   PE (kp,cp) receives C-slice cp of filter k1 = k1g*KP + kp.
//   k1 >= K1 (KP padding) sends zeros (no DRAM read).
//   Send order = PE receive order: lane -> rs -> i
// ===========================================================================
static void w_loader(const wvec_t *gmem_w, hls::stream<wvec_t> w_st[N_PES],
                     LayerCfg cfg) {
    const Derived d = derive(cfg);
    const int rs = cfg.R * cfg.S;
    int tag_k1g = -1, tag_c2 = -1;

W_K1G:
    for (int k1g = 0; k1g < d.K1G; ++k1g) {
#pragma HLS loop_tripcount min = 1 max = 16
    W_P1:
        for (int p1 = 0; p1 < d.P2; ++p1) {
#pragma HLS loop_tripcount min = 1 max = 32
        W_C2:
            for (int c2 = 0; c2 < d.C2; ++c2) {
#pragma HLS loop_tripcount min = 1 max = 16
                if (k1g == tag_k1g && c2 == tag_c2)
                    continue;
                tag_k1g = k1g;
                tag_c2 = c2;
                const int ct1 = ct1_of(c2, d.c1s, cfg.CT1);
            W_PE:
                for (int pe = 0; pe < N_PES; ++pe) {
                    const int kp = pe / cfg.CP;
                    const int cp = pe % cfg.CP;
                    const int k1 = k1g * cfg.KP + kp;
                    const bool valid_k = (k1 < cfg.K1);
                    const int n = N_LANES * rs * ct1;
                    int lane = 0, rs_i = 0, i = 0;
                W_SEND:
                    for (int it = 0; it < n; ++it) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 8 max = 4096
                        wvec_t word;
                        if (valid_k) {
                            const int k = k1 * N_LANES + lane;
                            word = gmem_w[(k * rs + rs_i) * cfg.C1 +
                                          cp * d.c1s + c2 * cfg.CT1 + i];
                        } else {
                            for (int v = 0; v < VECTOR_SIZE; ++v) {
#pragma HLS unroll
                                wvec_set(word, v, (w_t)0);
                            }
                        }
                        st_write(w_st, pe, word);
                        if (i == ct1 - 1) {
                            i = 0;
                            if (rs_i == rs - 1) { rs_i = 0; lane++; }
                            else                { rs_i++; }
                        } else {
                            i++;
                        }
                    }
                }
            }
        }
    }
}

// ===========================================================================
// Task 3: PE worker — stream-fed version of the Phase 2 OS-LWS engine.
//   Receive counts/conditions use the same formulas as the loaders (lockstep).
//   acc initializes to 0 (bias is added once in gather after the reduction).
//   After all c2 tiles of a p1 group are accumulated, pushes the psum rows
//   (PT_ROWS x Q).
// ===========================================================================
static void pe_worker(hls::stream<iavec_t> &ia_st, hls::stream<wvec_t> &w_st,
                      hls::stream<psum_t> &psum_st, LayerCfg cfg) {
    const Derived d = derive(cfg);
    const int rs = cfg.R * cfg.S;

    wvec_t wbuf[WBUF_DEPTH][N_LANES];
#pragma HLS bind_storage variable = wbuf type = ram_2p impl = bram
#pragma HLS array_partition variable = wbuf complete dim = 2
    iavec_t rowbuf[IABUF_DEPTH];
#pragma HLS bind_storage variable = rowbuf type = ram_2p impl = bram
    acc_t acc_buf[PT_ROWS][Q_MAX][N_LANES];
#pragma HLS bind_storage variable = acc_buf type = ram_2p impl = bram
#pragma HLS array_partition variable = acc_buf complete dim = 3

    int tag_k1g = -1, tag_c2 = -1;

PE_K1G:
    for (int k1g = 0; k1g < d.K1G; ++k1g) {
#pragma HLS loop_tripcount min = 1 max = 16
    PE_P1:
        for (int p1 = 0; p1 < d.P2; ++p1) {
#pragma HLS loop_tripcount min = 1 max = 32
        PE_C2:
            for (int c2 = 0; c2 < d.C2; ++c2) {
#pragma HLS loop_tripcount min = 1 max = 16
                const int ct1 = ct1_of(c2, d.c1s, cfg.CT1);
                const int rsct = rs * ct1;

                // ---- receive weights (same tag condition as the loader) ----
                if (!(k1g == tag_k1g && c2 == tag_c2)) {
                    tag_k1g = k1g;
                    tag_c2 = c2;
                    const int n = N_LANES * rs * ct1;
                    int lane = 0, rs_i = 0, i = 0;
                PE_RECV_W:
                    for (int it = 0; it < n; ++it) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 8 max = 4096
                        wbuf[rs_i * ct1 + i][lane] = w_st.read();
                        if (i == ct1 - 1) {
                            i = 0;
                            if (rs_i == rs - 1) { rs_i = 0; lane++; }
                            else                { rs_i++; }
                        } else {
                            i++;
                        }
                    }
                }

            PE_P0:
                for (int p0 = 0; p0 < PT_ROWS; ++p0) {
                    // ---- receive IA rows (loader send order = linear) ----
                    const int n = cfg.R * d.win_cols * ct1;
                PE_RECV_IA:
                    for (int it = 0; it < n; ++it) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 16 max = 2048
                        rowbuf[it] = ia_st.read();
                    }

                    // ---- compute row (OS-LWS, flattened counters) ----
                    const int total = rsct * d.qb;
                    int r = 0, s = 0, c1 = 0, q = 0, rsc = 0;
                MAC_LOOP:
                    for (int it = 0; it < total; ++it) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 256 max = 131072
#pragma HLS dependence variable = acc_buf type = inter direction = RAW distance = 16 dependent = true
#pragma HLS dependence variable = acc_buf type = inter direction = WAW distance = 16 dependent = true
                        const int col     = q * cfg.stride + s;
                        const int ia_addr = (r * d.win_cols + col) * ct1 + c1;
                        const iavec_t iav = rowbuf[ia_addr];
                        const bool first  = (c2 == 0) && (rsc == 0);

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
                                first ? (acc_t)0 : acc_buf[p0][q][2 * j];
                            const acc_t prev1 =
                                first ? (acc_t)0 : acc_buf[p0][q][2 * j + 1];
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
                                dot += (acc_t)(wvec_get(wbuf[rsc][lane], v) *
                                               iavec_get(iav, v));
                            }
                            const acc_t prev =
                                first ? (acc_t)0 : acc_buf[p0][q][lane];
                            acc_buf[p0][q][lane] = prev + dot;
                        }
#endif
                        if (q == d.qb - 1) {
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
            }

            // ---- psum push: whole row group (fixed count PT_ROWS x Q) ----
        PE_PUSH:
            for (int p0 = 0; p0 < PT_ROWS; ++p0) {
            PE_PUSH_Q:
                for (int q = 0; q < cfg.Q; ++q) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 8 max = 256
                    psum_t ps;
                    for (int lane = 0; lane < N_LANES; ++lane) {
#pragma HLS unroll
                        ps.v[lane] = acc_buf[p0][q][lane];
                    }
                    psum_st.write(ps);
                }
            }
        }
    }
}

// ===========================================================================
// Task 4: Gather — cross-PE reduction + bias + PPU + store.
//   Sums the CP psums of each kp group, adds bias once, then requantizes.
//   p >= P (row-group tail) and k1 >= K1 (KP padding) only drain the streams.
//
//   II=1 loop over flattened (q, kp): one output word per cycle with a fixed
//   set of N_LANES requant units — gather DSP/LUT stays flat as N_PES grows.
//   The psum streams are read once per q (at kp==0) into registers.
// ===========================================================================
static void gather_store(hls::stream<psum_t> psum_st[N_PES], oavec_t *gmem_oa,
                         const bias_t *gmem_bias, LayerCfg cfg) {
    const Derived d = derive(cfg);
    bias_t bias_reg[N_PES][N_LANES];
#pragma HLS array_partition variable = bias_reg complete dim = 0

G_K1G:
    for (int k1g = 0; k1g < d.K1G; ++k1g) {
#pragma HLS loop_tripcount min = 1 max = 16
        // ---- load bias (0 for k1 >= K1) ----
    G_BIAS:
        for (int kp = 0; kp < N_PES; ++kp) {
            const int k1 = k1g * cfg.KP + kp;
            const bool valid = (kp < cfg.KP) && (k1 < cfg.K1);
        G_BIAS_L:
            for (int lane = 0; lane < N_LANES; ++lane) {
#pragma HLS pipeline II = 1
                bias_reg[kp][lane] =
                    valid ? gmem_bias[k1 * N_LANES + lane] : (bias_t)0;
            }
        }

    G_P1:
        for (int p1 = 0; p1 < d.P2; ++p1) {
#pragma HLS loop_tripcount min = 1 max = 32
        G_P0:
            for (int p0 = 0; p0 < PT_ROWS; ++p0) {
                const int p = p1 * PT_ROWS + p0;
                const int total = cfg.Q * cfg.KP;
                psum_t vals[N_PES];
#pragma HLS array_partition variable = vals complete
                int q = 0, kp = 0, base = 0; // base = kp*CP

            G_QKP:
                for (int it = 0; it < total; ++it) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 8 max = 4096
                    // read every PE stream only at the first kp of each q
                    // (keeps word counts matched)
                    if (kp == 0) {
                    G_READ:
                        for (int pe = 0; pe < N_PES; ++pe) {
#pragma HLS unroll
                            vals[pe] = psum_st[pe].read();
                        }
                    }

                    const int k1 = k1g * cfg.KP + kp;
                    if (p < cfg.P && k1 < cfg.K1) {
                        oavec_t ov;
                    G_LANE:
                        for (int lane = 0; lane < N_LANES; ++lane) {
#pragma HLS unroll
                            acc_t sum = bias_reg[kp][lane];
                        G_CP:
                            for (int pe = 0; pe < N_PES; ++pe) {
#pragma HLS unroll
                                if (pe >= base && pe < base + cfg.CP)
                                    sum += vals[pe].v[lane];
                            }
                            oavec_set(ov, lane,
                                      requantize(sum, cfg.mult, cfg.shift,
                                                 cfg.relu_en != 0));
                        }
                        gmem_oa[(p * cfg.Q + q) * cfg.K1 + k1] = ov;
                    }

                    // counters: kp -> q
                    if (kp == cfg.KP - 1) { kp = 0; base = 0; q++; }
                    else                  { kp++; base += cfg.CP; }
                }
            }
        }
    }
}

// ===========================================================================
// Dataflow core (canonical: task calls only)
// ===========================================================================
static void magnet_core(const iavec_t *gmem_ia0, const iavec_t *gmem_ia1,
                        const iavec_t *gmem_ia2, const iavec_t *gmem_ia3,
                        const wvec_t *gmem_w,
                        oavec_t *gmem_oa, const bias_t *gmem_bias,
                        LayerCfg cfg) {
#pragma HLS dataflow
    hls::stream<iavec_t> ia_st[N_PES];
#pragma HLS stream variable = ia_st depth = 512
    hls::stream<wvec_t> w_st[N_PES];
#pragma HLS stream variable = w_st depth = 64
    hls::stream<psum_t> psum_st[N_PES];
#pragma HLS stream variable = psum_st depth = 64
#pragma HLS bind_storage variable = psum_st type = fifo impl = srl

    ia_loader(gmem_ia0, gmem_ia1, gmem_ia2, gmem_ia3, ia_st, cfg);
    w_loader(gmem_w, w_st, cfg);
PE_INST:
    for (int pe = 0; pe < N_PES; ++pe) {
#pragma HLS unroll
        pe_worker(ia_st[pe], w_st[pe], psum_st[pe], cfg);
    }
    gather_store(psum_st, gmem_oa, gmem_bias, cfg);
}

// ===========================================================================
// Top level
// ===========================================================================
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
    int mult, int shift, int relu_en) {
#pragma HLS INTERFACE m_axi port = gmem_ia0  offset = slave bundle = gmem0 depth = GMEM_DEPTH
#pragma HLS INTERFACE m_axi port = gmem_ia1  offset = slave bundle = gmem4 depth = GMEM_DEPTH
#pragma HLS INTERFACE m_axi port = gmem_ia2  offset = slave bundle = gmem5 depth = GMEM_DEPTH
#pragma HLS INTERFACE m_axi port = gmem_ia3  offset = slave bundle = gmem6 depth = GMEM_DEPTH
#pragma HLS INTERFACE m_axi port = gmem_w    offset = slave bundle = gmem1 depth = GMEM_DEPTH
#pragma HLS INTERFACE m_axi port = gmem_oa   offset = slave bundle = gmem2 depth = GMEM_DEPTH
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
#pragma HLS INTERFACE s_axilite port = KP
#pragma HLS INTERFACE s_axilite port = CP
#pragma HLS INTERFACE s_axilite port = mult
#pragma HLS INTERFACE s_axilite port = shift
#pragma HLS INTERFACE s_axilite port = relu_en
#pragma HLS INTERFACE s_axilite port = return

    LayerCfg cfg;
    cfg.H = H;   cfg.W = W;   cfg.C1 = C1; cfg.K1 = K1; cfg.K = K;
    cfg.P = P;   cfg.Q = Q;   cfg.R = R;   cfg.S = S;
    cfg.stride = stride;      cfg.pad = pad;    cfg.CT1 = CT1;
    cfg.KP = KP;              cfg.CP = CP;
    cfg.mult = mult;          cfg.shift = shift;  cfg.relu_en = relu_en;

#ifndef __SYNTHESIS__
    {
        const Derived d = derive(cfg);
        assert(KP * CP == N_PES);
        assert((CP & (CP - 1)) == 0); // power of two (loader uses & (CP-1))
        assert(C1 % CP == 0);
        assert(CT1 >= 1 && CT1 <= d.c1s);
        assert(R * S * CT1 <= WBUF_DEPTH);
        assert(R * d.win_cols * CT1 <= IABUF_DEPTH);
        assert(Q <= Q_MAX && d.qb <= Q_MAX);
        assert(K <= K1 * N_LANES);
        assert(N_LANES % 2 == 0);
    }
#endif

    magnet_core(gmem_ia0, gmem_ia1, gmem_ia2, gmem_ia3,
                gmem_w, gmem_oa, gmem_bias, cfg);
}
