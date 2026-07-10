#!/usr/bin/env python3
# =============================================================================
# mapper.py — lightweight implementation of the MAGNet Mapper (Timeloop-lite)
#
# Role (maps to MAGNet paper Section V):
#   1. Verify that a layer fits the current hardware config (buffer sizes)
#   2. Compute padded dimensions (C1, K1) and runtime config-register values
#   3. Estimate cycles / DRAM traffic / MAC utilization -> mapping quality
#
# Usage:
#   python mapper.py --layer H,W,C,K,R,S,stride,pad
#   python mapper.py --network resnet50
#   python mapper.py --network resnet50 --json out.json
# =============================================================================

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict


# --- hardware design-time parameters (must match hw/include/accel_config.h) --
@dataclass
class HwConfig:
    vector_size: int = 8
    n_lanes: int = 8
    n_pes: int = 8         # number of PEs (KP*CP == n_pes, matches N_PES)
    wbuf_depth: int = 256
    iabuf_depth: int = 2048
    q_max: int = 128
    min_qb: int = 16
    pt_rows: int = 8       # accumulation-buffer row-group size
    n_ia_ports: int = 4    # parallel IA loader AXI masters (N_IA_PORTS)
    freq_mhz: float = 200.0
    word_bytes: int = 8    # 64-bit AXI word


@dataclass
class Layer:
    name: str
    H: int; W: int; C: int; K: int
    R: int; S: int; stride: int = 1; pad: int = 0


def ceil_div(a, b):
    return (a + b - 1) // b


def spatial_candidates(n_pes):
    """All (KP, CP) combinations with KP*CP == n_pes."""
    out = []
    kp = 1
    while kp <= n_pes:
        if n_pes % kp == 0:
            out.append((kp, n_pes // kp))
        kp *= 2
    return out


def eval_mapping(layer: Layer, hw: HwConfig, KP, CP):
    """Evaluate validity/cycles/traffic of one spatial mapping (mirrors
    magnet_top.cpp)."""
    P = (layer.H + 2 * layer.pad - layer.R) // layer.stride + 1
    Q = (layer.W + 2 * layer.pad - layer.S) // layer.stride + 1
    C1r = ceil_div(layer.C, hw.vector_size)
    C1 = ceil_div(C1r, CP) * CP          # pad to a CP multiple (host contract)
    K1 = ceil_div(layer.K, hw.n_lanes)
    QB = max(Q, hw.min_qb)
    win_cols = (QB - 1) * layer.stride + layer.S
    rs = layer.R * layer.S
    c1s = C1 // CP

    CT1 = min(c1s,
              hw.wbuf_depth // rs,
              hw.iabuf_depth // (layer.R * win_cols))
    violations = []
    if CT1 < 1:
        violations.append(
            f"CT1<1: R*S={rs} > WBUF {hw.wbuf_depth} or "
            f"R*win_cols={layer.R * win_cols} > IABUF {hw.iabuf_depth}")
        CT1 = 1
    if Q > hw.q_max:
        violations.append(f"Q {Q} > Q_MAX {hw.q_max}")

    C2 = ceil_div(c1s, CT1)
    P2 = ceil_div(P, hw.pt_rows)
    K1G = ceil_div(K1, KP)

    # ---- cycle model ----
    # Per (k1g,p1) block: max over the three pipeline stages:
    #   PE compute   = PT*rs*c1s*QB (sum over all c2 tiles, PEs in parallel)
    #   loader send  = PT*R*win_cols*c1s*ceil(CP/n_ia_ports)
    #                  (CP slices per position read through parallel ports)
    #   gather store = PT*Q*KP (II=1, one word per cycle)
    blk_compute = hw.pt_rows * rs * c1s * QB
    cp_beats = ceil_div(CP, hw.n_ia_ports)
    blk_load = hw.pt_rows * layer.R * win_cols * c1s * cp_beats
    blk_gather = hw.pt_rows * Q * KP
    body = K1G * P2 * max(blk_compute, blk_load, blk_gather)

    n_wloads = K1G * (1 if C2 == 1 else P2 * C2)
    load_w = n_wloads * hw.n_pes * hw.n_lanes * rs * \
        (c1s if C2 == 1 else CT1)
    cycles = body + load_w

    real_macs = P * Q * layer.K * rs * layer.C
    peak = K1G * P2 * hw.pt_rows * rs * c1s * QB * \
        hw.vector_size * hw.n_lanes * hw.n_pes
    util = real_macs / peak if peak else 0.0

    wb = hw.word_bytes
    dram_ia = K1G * P2 * hw.pt_rows * layer.R * min(layer.W, win_cols) * C1 * wb
    dram_w = load_w * wb
    dram_oa = P * Q * K1 * wb
    dram_bias = K1G * KP * hw.n_lanes * 4

    return {
        "KP": KP, "CP": CP, "C1": C1, "K1": K1, "CT1": CT1, "C2": C2,
        "P": P, "Q": Q, "K1G": K1G,
        "violations": violations,
        "cycles": cycles,
        "util": util,
        "dram": {"ia": dram_ia, "w": dram_w, "oa": dram_oa, "bias": dram_bias,
                 "total": dram_ia + dram_w + dram_oa + dram_bias},
        "real_macs": real_macs,
    }


def map_layer(layer: Layer, hw: HwConfig):
    """Evaluate all spatial-mapping candidates and pick the fewest cycles."""
    cands = [eval_mapping(layer, hw, kp, cp)
             for kp, cp in spatial_candidates(hw.n_pes)]
    valid = [c for c in cands if not c["violations"]]
    best = min(valid, key=lambda c: c["cycles"]) if valid \
        else min(cands, key=lambda c: c["cycles"])

    latency_ms = best["cycles"] / (hw.freq_mhz * 1e6) * 1e3
    gops = (2 * best["real_macs"]) / (best["cycles"] / (hw.freq_mhz * 1e6)) / 1e9 \
        if best["cycles"] else 0

    return {
        "name": layer.name,
        "cfg_regs": {  # in magnet_top() argument order
            "H": layer.H, "W": layer.W, "C1": best["C1"], "K1": best["K1"],
            "K": layer.K, "P": best["P"], "Q": best["Q"],
            "R": layer.R, "S": layer.S,
            "stride": layer.stride, "pad": layer.pad, "CT1": best["CT1"],
            "KP": best["KP"], "CP": best["CP"],
        },
        "C2": best["C2"],
        "valid": not best["violations"],
        "violations": best["violations"],
        "est_cycles": best["cycles"],
        "est_latency_ms": round(latency_ms, 3),
        "est_gops": round(gops, 1),
        "mac_utilization": round(best["util"], 3),
        "dram_bytes": best["dram"],
        "alternatives": [{"KP": c["KP"], "CP": c["CP"], "cycles": c["cycles"],
                          "valid": not c["violations"]} for c in cands],
    }


# ---- network presets --------------------------------------------------------
def resnet50_layers():
    """Representative ResNet-50 conv shapes (repeat counts noted for
    duplicated shapes)."""
    L = Layer
    return [
        L("conv1_7x7",      224, 224,    3,   64, 7, 7, 2, 3),
        L("res2_1x1a",       56,  56,   64,   64, 1, 1, 1, 0),
        L("res2_3x3",        56,  56,   64,   64, 3, 3, 1, 1),   # x3
        L("res2_1x1b",       56,  56,   64,  256, 1, 1, 1, 0),   # x3
        L("res3_1x1a",       56,  56,  256,  128, 1, 1, 1, 0),
        L("res3_3x3",        28,  28,  128,  128, 3, 3, 1, 1),   # x4
        L("res3_1x1b",       28,  28,  128,  512, 1, 1, 1, 0),   # x4
        L("res4_1x1a",       28,  28,  512,  256, 1, 1, 1, 0),
        L("res4_3x3",        14,  14,  256,  256, 3, 3, 1, 1),   # x6
        L("res4_1x1b",       14,  14,  256, 1024, 1, 1, 1, 0),   # x6
        L("res5_1x1a",       14,  14, 1024,  512, 1, 1, 1, 0),
        L("res5_3x3",         7,   7,  512,  512, 3, 3, 1, 1),   # x3
        L("res5_1x1b",        7,   7,  512, 2048, 1, 1, 1, 0),   # x3
    ]


NETWORKS = {"resnet50": resnet50_layers}


def main():
    ap = argparse.ArgumentParser(description="MAGNet-lite mapper")
    ap.add_argument("--layer", help="H,W,C,K,R,S,stride,pad")
    ap.add_argument("--network", choices=NETWORKS.keys())
    ap.add_argument("--json", help="save results to a JSON file")
    ap.add_argument("--freq", type=float, default=200.0, help="clock (MHz)")
    args = ap.parse_args()

    hw = HwConfig(freq_mhz=args.freq)
    layers = []
    if args.layer:
        v = [int(x) for x in args.layer.split(",")]
        if len(v) != 8:
            sys.exit("--layer requires 8 values: H,W,C,K,R,S,stride,pad")
        layers.append(Layer("custom", *v))
    elif args.network:
        layers = NETWORKS[args.network]()
    else:
        ap.print_help()
        return

    results = [map_layer(l, hw) for l in layers]

    hdr = (f"{'layer':<14}{'valid':<7}{'KPxCP':>6}{'P':>4}{'Q':>5}{'C1':>5}"
           f"{'K1':>5}{'CT1':>5}{'C2':>4}"
           f"{'cycles':>12}{'ms':>9}{'GOPS':>8}{'util':>7}{'DRAM MB':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        c = r["cfg_regs"]
        print(f"{r['name']:<14}{str(r['valid']):<7}"
              f"{str(c['KP']) + 'x' + str(c['CP']):>6}{c['P']:>4}{c['Q']:>5}"
              f"{c['C1']:>5}{c['K1']:>5}{c['CT1']:>5}{r['C2']:>4}"
              f"{r['est_cycles']:>12,}"
              f"{r['est_latency_ms']:>9}{r['est_gops']:>8}"
              f"{r['mac_utilization']:>7}"
              f"{r['dram_bytes']['total'] / 1e6:>9.2f}")
        for v in r["violations"]:
            print(f"    !! {v}")

    total_cycles = sum(r["est_cycles"] for r in results)
    print(f"\nTotal (each unique shape once): {total_cycles:,} cycles "
          f"= {total_cycles / (hw.freq_mhz * 1e3):.2f} ms @ {hw.freq_mhz:.0f} MHz")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"hw": asdict(hw), "layers": results}, f,
                      ensure_ascii=False, indent=2)
        print(f"JSON written: {args.json}")


if __name__ == "__main__":
    main()
