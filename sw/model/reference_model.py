#!/usr/bin/env python3
"""Bit-exact Python mirror of the magnet_top.cpp algorithm (multi-PE).

Reproduces the dataflow task structure of hw/src/magnet_top.cpp exactly:
  ia_loader  : reads IA and multicasts each PE's C-slice (stream = list)
  w_loader   : distributes the (k1,c2) tile per PE (zeros for k1 >= K1)
  pe_worker  : receives rowbuf/wbuf -> OS-LWS MAC -> pushes psum row stream
  gather     : sums the CP psums (cross-PE reduction) + bias + requant + store

Streams are modeled as lists and 'fully consumed' is asserted at the end,
so producer/consumer count mismatches (the cause of hardware deadlocks) are
caught here as well.

Spatial mapping: KP x CP = N_PES. kp = pe//CP (K-split), cp = pe%CP (C-split).
Constraints: C1 % CP == 0 (host pads), CT1 <= C1/CP.

Run: python sw/model/reference_model.py
"""
import random

VECTOR_SIZE = 8
N_LANES = 8
N_PES = 8
WBUF_DEPTH = 256
IABUF_DEPTH = 2048
Q_MAX = 128
MIN_QB = 16
PT_ROWS = 8


def clamp_i8(x):
    return max(-128, min(127, x))


def requantize(acc, mult, shift, relu):
    x = acc * mult
    if shift > 0:
        x += 1 << (shift - 1)
    x >>= shift
    if relu and x < 0:
        x = 0
    return clamp_i8(x)


def mac_pair(w0, w1, a):
    """DSP packing: two products from one 27x18 multiply (mirrors
    magnet_top.cpp mac_pair)."""
    pr = ((w1 << 18) + w0) * a
    lo_raw = pr & 0x3FFFF
    lo = lo_raw - (1 << 18) if lo_raw >= (1 << 17) else lo_raw
    hi = (pr >> 18) + (1 if lo < 0 else 0)
    return lo, hi


class Stream:
    """hls::stream mirror — used to verify consumption counts."""

    def __init__(self, name):
        self.name = name
        self.data = []
        self.rd = 0

    def write(self, v):
        self.data.append(v)

    def read(self):
        assert self.rd < len(self.data), f"stream {self.name} underflow (deadlock)"
        v = self.data[self.rd]
        self.rd += 1
        return v

    def check_drained(self):
        assert self.rd == len(self.data), \
            f"stream {self.name}: {len(self.data) - self.rd} words unconsumed (deadlock)"


def derive(cfg):
    """Shared derived parameters (all tasks use the same formulas — this is
    what makes the lockstep hold)."""
    qb = max(cfg["Q"], MIN_QB)
    win_cols = (qb - 1) * cfg["stride"] + cfg["S"]
    c1s = cfg["C1"] // cfg["CP"]                      # C-slice words per PE
    C2 = (c1s + cfg["CT1"] - 1) // cfg["CT1"]
    P2 = (cfg["P"] + PT_ROWS - 1) // PT_ROWS
    K1G = (cfg["K1"] + cfg["KP"] - 1) // cfg["KP"]
    return qb, win_cols, c1s, C2, P2, K1G


def ct1_of(c2, c1s, CT1):
    return min(CT1, c1s - c2 * CT1)


def ia_loader(d_ia, ia_st, cfg):
    qb, win_cols, c1s, C2, P2, K1G = derive(cfg)
    H, W, C1, stride, pad, R = (cfg[k] for k in ("H", "W", "C1", "stride", "pad", "R"))
    for k1g in range(K1G):
        for p1 in range(P2):
            for c2 in range(C2):
                ct1 = ct1_of(c2, c1s, cfg["CT1"])
                for p0 in range(PT_ROWS):
                    p = p1 * PT_ROWS + p0
                    for r in range(R):
                        ih = p * stride - pad + r
                        for col in range(win_cols):
                            w = col - pad
                            valid = (0 <= ih < H) and (0 <= w < W)
                            for cp in range(cfg["CP"]):
                                base = cp * c1s + c2 * cfg["CT1"]
                                for i in range(ct1):
                                    word = d_ia[(ih * W + w) * C1 + base + i] \
                                        if valid else [0] * VECTOR_SIZE
                                    for pe in range(N_PES):   # multicast (same cp)
                                        if pe % cfg["CP"] == cp:
                                            ia_st[pe].write(word)


def w_loader(d_w, w_st, cfg):
    qb, win_cols, c1s, C2, P2, K1G = derive(cfg)
    rs = cfg["R"] * cfg["S"]
    tag = (-1, -1)
    for k1g in range(K1G):
        for p1 in range(P2):
            for c2 in range(C2):
                if tag == (k1g, c2):
                    continue
                tag = (k1g, c2)
                ct1 = ct1_of(c2, c1s, cfg["CT1"])
                for pe in range(N_PES):
                    kp, cp = pe // cfg["CP"], pe % cfg["CP"]
                    k1 = k1g * cfg["KP"] + kp
                    base = cp * c1s + c2 * cfg["CT1"]
                    for lane in range(N_LANES):
                        k = k1 * N_LANES + lane
                        for rs_i in range(rs):
                            for i in range(ct1):
                                word = d_w[(k * rs + rs_i) * cfg["C1"] + base + i] \
                                    if k1 < cfg["K1"] else [0] * VECTOR_SIZE
                                w_st[pe].write(word)


def pe_worker(ia_st, w_st, psum_st, cfg, pe):
    qb, win_cols, c1s, C2, P2, K1G = derive(cfg)
    R, S, stride, Q = cfg["R"], cfg["S"], cfg["stride"], cfg["Q"]
    rs = R * S
    rowbuf = [None] * IABUF_DEPTH
    wbuf = [None] * WBUF_DEPTH
    acc_buf = [[[0] * N_LANES for _ in range(Q_MAX)] for _ in range(PT_ROWS)]
    tag = (-1, -1)

    for k1g in range(K1G):
        for p1 in range(P2):
            for c2 in range(C2):
                ct1 = ct1_of(c2, c1s, cfg["CT1"])
                rsct = rs * ct1
                assert rsct <= WBUF_DEPTH
                assert R * win_cols * ct1 <= IABUF_DEPTH

                # ---- recv weights (same tag condition as loader = lockstep) --
                if tag != (k1g, c2):
                    tag = (k1g, c2)
                    for lane in range(N_LANES):
                        for rs_i in range(rs):
                            for i in range(ct1):
                                wbuf_idx = rs_i * ct1 + i
                                if wbuf[wbuf_idx] is None:
                                    wbuf[wbuf_idx] = [None] * N_LANES
                                wbuf[wbuf_idx][lane] = w_st.read()

                for p0 in range(PT_ROWS):
                    # ---- recv IA rows (linear rowbuf fill = loader order) ----
                    n = R * win_cols * ct1
                    for i in range(n):
                        rowbuf[i] = ia_st.read()

                    # ---- compute_row (flattened counters, OS-LWS,
                    #      zero-init without bias) ----
                    total = rsct * qb
                    r = s = c1 = q = rsc = 0
                    for _ in range(total):
                        col = q * stride + s
                        ia_addr = (r * win_cols + col) * ct1 + c1
                        iav = rowbuf[ia_addr]
                        first = (c2 == 0 and rsc == 0)
                        for j in range(N_LANES // 2):
                            l0, l1 = 2 * j, 2 * j + 1
                            d0 = d1 = 0
                            for v in range(VECTOR_SIZE):
                                lo, hi = mac_pair(wbuf[rsc][l0][v],
                                                  wbuf[rsc][l1][v], iav[v])
                                d0 += lo
                                d1 += hi
                            prev0 = 0 if first else acc_buf[p0][q][l0]
                            prev1 = 0 if first else acc_buf[p0][q][l1]
                            acc_buf[p0][q][l0] = prev0 + d0
                            acc_buf[p0][q][l1] = prev1 + d1
                        if q == qb - 1:
                            q = 0
                            rsc += 1
                            if c1 == ct1 - 1:
                                c1 = 0
                                if s == S - 1:
                                    s = 0
                                    r += 1
                                else:
                                    s += 1
                            else:
                                c1 += 1
                        else:
                            q += 1

            # ---- psum row push (after all c2 tiles of the group; fixed count)
            for p0 in range(PT_ROWS):
                for q_ in range(Q):
                    psum_st.write(list(acc_buf[p0][q_]))


def gather_store(psum_st, d_bias, d_mult, d_oa, cfg):
    qb, win_cols, c1s, C2, P2, K1G = derive(cfg)
    KP, CP, K1, Q, P = cfg["KP"], cfg["CP"], cfg["K1"], cfg["Q"], cfg["P"]
    for k1g in range(K1G):
        bias_reg = [[0] * N_LANES for _ in range(N_PES)]
        mult_reg = [[0] * N_LANES for _ in range(N_PES)]
        for kp in range(KP):
            k1 = k1g * KP + kp
            if k1 < K1:
                for lane in range(N_LANES):
                    bias_reg[kp][lane] = d_bias[k1 * N_LANES + lane]
                    mult_reg[kp][lane] = d_mult[k1 * N_LANES + lane]
        for p1 in range(P2):
            for p0 in range(PT_ROWS):
                p = p1 * PT_ROWS + p0
                for q in range(Q):
                    vals = [psum_st[pe].read() for pe in range(N_PES)]  # always all
                    if p >= P:
                        continue
                    for kp in range(KP):
                        k1 = k1g * KP + kp
                        if k1 >= K1:
                            continue
                        word = []
                        for lane in range(N_LANES):
                            s = sum(vals[kp * CP + cp][lane] for cp in range(CP))
                            word.append(requantize(s + bias_reg[kp][lane],
                                                   mult_reg[kp][lane],
                                                   cfg["shift"],
                                                   cfg["relu"]))
                        d_oa[(p * Q + q) * K1 + k1] = word


def dut(d_ia, d_w, d_bias, d_mult, cfg):
    """Mirror of magnet_top(): runs the tasks sequentially (same semantics as
    csim's dataflow)."""
    assert cfg["KP"] * cfg["CP"] == N_PES
    assert cfg["C1"] % cfg["CP"] == 0
    assert cfg["CT1"] <= cfg["C1"] // cfg["CP"]
    assert cfg["Q"] <= Q_MAX

    ia_st = [Stream(f"ia{p}") for p in range(N_PES)]
    w_st = [Stream(f"w{p}") for p in range(N_PES)]
    psum_st = [Stream(f"ps{p}") for p in range(N_PES)]
    d_oa = [None] * (cfg["P"] * cfg["Q"] * cfg["K1"])

    ia_loader(d_ia, ia_st, cfg)
    w_loader(d_w, w_st, cfg)
    for pe in range(N_PES):
        pe_worker(ia_st[pe], w_st[pe], psum_st[pe], cfg, pe)
    gather_store(psum_st, d_bias, d_mult, d_oa, cfg)

    for st in ia_st + w_st + psum_st:
        st.check_drained()
    return d_oa


def run_test(name, H, W, C, K, R, S, stride, pad, mult, shift, relu, seed,
             KP, CP, CT1=None):
    P = (H + 2 * pad - R) // stride + 1
    Q = (W + 2 * pad - S) // stride + 1
    C1r = -(-C // VECTOR_SIZE)
    C1 = -(-C1r // CP) * CP          # pad to a CP multiple (host contract)
    K1 = -(-K // N_LANES)
    Kp = K1 * N_LANES
    c1s = C1 // CP
    if CT1 is None or CT1 > c1s:
        CT1 = c1s  # clamp an explicit CT1 larger than the slice (same as TB)

    rng = random.Random(seed)
    ia = [rng.randint(-128, 127) for _ in range(H * W * C)]
    w = [rng.randint(-128, 127) for _ in range(K * R * S * C)]
    bias = [rng.randint(-1000, 1000) if k < K else 0 for k in range(Kp)]

    d_ia = [[0] * VECTOR_SIZE for _ in range(H * W * C1)]
    d_w = [[0] * VECTOR_SIZE for _ in range(Kp * R * S * C1)]
    for h in range(H):
        for x in range(W):
            for c in range(C):
                d_ia[(h * W + x) * C1 + c // VECTOR_SIZE][c % VECTOR_SIZE] = \
                    ia[(h * W + x) * C + c]
    for k in range(K):
        for r in range(R):
            for s in range(S):
                for c in range(C):
                    d_w[((k * R + r) * S + s) * C1 + c // VECTOR_SIZE][c % VECTOR_SIZE] = \
                        w[((k * R + r) * S + s) * C + c]

    gold = {}
    for p in range(P):
        for q in range(Q):
            for k in range(K):
                acc = bias[k]
                for r in range(R):
                    for s in range(S):
                        ih = p * stride - pad + r
                        iw = q * stride - pad + s
                        if 0 <= ih < H and 0 <= iw < W:
                            for c in range(C):
                                acc += w[((k * R + r) * S + s) * C + c] * \
                                       ia[(ih * W + iw) * C + c]
                gold[(p, q, k)] = requantize(acc, mult, shift, relu)

    cfg = dict(H=H, W=W, C1=C1, K1=K1, K=K, P=P, Q=Q, R=R, S=S,
               stride=stride, pad=pad, CT1=CT1, KP=KP, CP=CP,
               mult=mult, shift=shift, relu=relu)
    d_mult = [mult] * Kp   # uniform per-channel table (per-tensor behavior)
    d_oa = dut(d_ia, d_w, bias, d_mult, cfg)

    errors = 0
    for p in range(P):
        for q in range(Q):
            for k in range(K):
                hw = d_oa[(p * Q + q) * K1 + k // N_LANES][k % N_LANES]
                if hw != gold[(p, q, k)]:
                    if errors < 5:
                        print(f"  MISMATCH p={p} q={q} k={k}: hw={hw} gold={gold[(p,q,k)]}")
                    errors += 1
    C2 = (c1s + CT1 - 1) // CT1
    print(f"[{'PASS' if errors == 0 else 'FAIL'}] {name:<24} KPxCP={KP}x{CP} "
          f"(P={P} Q={Q} C1={C1} CT1={CT1} C2={C2}, {errors} errors)")
    return errors


BASE_TESTS = [
    #  name                    H   W    C   K  R  S st pd mult sh relu  CT1
    ("3x3_s1_p1_basic",       16, 16,  16, 16, 3, 3, 1, 1,  5, 10, 1, None),
    ("1x1_pointwise",         16, 16,  64, 32, 1, 1, 1, 0,  9, 11, 0, None),
    ("3x3_s2_downsample",     15, 15,   8,  8, 3, 3, 2, 1, 17, 10, 1, None),
    ("7x7_s2_p3_stem",        32, 32,   3,  8, 7, 7, 2, 3,  3, 10, 1, None),
    ("K_not_multiple_lanes",  12, 12,  24, 20, 3, 3, 1, 1,  7, 10, 1, None),
    ("ctile_3x3_C48_CT2",     12, 12,  48, 16, 3, 3, 1, 1,  5, 11, 1, 2),
    ("ctile_1x1_C128_CT4",    16, 16, 128, 24, 1, 1, 1, 0,  9, 12, 1, 4),
    ("rowgroup_P13",          13, 13,  16,  8, 3, 3, 1, 1,  5, 10, 1, None),
]

# K-split / mixed / C-split. Includes many K1 < KP (idle-group padding) cases
SPATIAL = [(N_PES, 1), (4, N_PES // 4), (1, N_PES)]

if __name__ == "__main__":
    total = 0
    seed = 20260709
    for KP, CP in SPATIAL:
        for t in BASE_TESTS:
            ct1 = t[12]
            # With C-split, CT1 cannot exceed the slice -> None = auto (whole slice)
            total += run_test(*t[:12], seed, KP=KP, CP=CP, CT1=ct1)
            seed += 1
    print("=== ALL PASSED ===" if total == 0 else f"=== {total} ERRORS ===")
