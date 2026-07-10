#!/usr/bin/env python3
# =============================================================================
# tuner.py — design-space exploration (MAGNet Tuner-lite, Phase 4)
#
# Plays the role of the MAGNet Tuner (paper Section VI): searches the
# design-time parameter space and reports performance/resource trade-offs.
# The original uses Bayesian optimization; the space here is small enough to
# sweep exhaustively, so this version evaluates every point with the mapper's
# cycle model plus an analytical resource model calibrated against the
# Phase 4 csynth reports.
#
# Usage:
#   python tuner.py                    # sweep + Pareto table for ResNet-50
#   python tuner.py --device zu9eg     # fit check against another device
# =============================================================================

import argparse
import itertools
import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "mapper"))
from mapper import HwConfig, map_layer, resnet50_layers, ceil_div  # noqa: E402

# ResNet-50 repeat counts for the mapper's unique-shape table (same order)
RESNET50_REPEATS = [1, 1, 3, 3, 1, 4, 4, 1, 6, 6, 1, 3, 3]

DEVICES = {
    #            BRAM18  DSP   kLUT
    "zu7ev": (624, 1728, 230),
    "zu9eg": (1824, 2520, 274),
}


def bram36_for(depth, width_bits):
    """BRAM36 count for one bank: width-dominated below 36b x 1K."""
    if depth <= 0:
        return 0
    width_cost = ceil_div(width_bits, 72)          # 72b max per BRAM36 (SDP)
    depth_cost = ceil_div(depth * width_bits, 36864)
    return max(width_cost, depth_cost)


def resources(hw: HwConfig):
    """Analytical resource model, calibrated against csynth (8 PE ~ 394-410
    BRAM18 / ~490 DSP / ~110k LUT with wbuf=256, iabuf=2048, q_max=128)."""
    lanes = hw.n_lanes
    # per PE
    wbuf = lanes * bram36_for(hw.wbuf_depth, 64)
    rowbuf = bram36_for(hw.iabuf_depth, 64)
    accbuf = lanes * bram36_for(hw.pt_rows * hw.q_max, 32)
    ia_fifo = bram36_for(512, 64)                  # ia_st depth 512
    pe_b36 = wbuf + rowbuf + accbuf + ia_fifo
    bram18 = 2 * (pe_b36 * hw.n_pes) + 60          # + AXI adapters/loaders
    dsp = hw.n_pes * 35 + 6 * hw.n_ia_ports + 18 + 38 + 40
    klut = round(hw.n_pes * 7.5 + 5 * hw.n_ia_ports + 45)
    return bram18, dsp, klut


def resnet50_cycles(hw: HwConfig):
    layers = resnet50_layers()
    total = 0
    valid = True
    for layer, rep in zip(layers, RESNET50_REPEATS):
        r = map_layer(layer, hw)
        if not r["valid"]:
            valid = False
        total += r["est_cycles"] * rep
    return total, valid


def main():
    ap = argparse.ArgumentParser(description="MAGNet Tuner-lite (exhaustive DSE)")
    ap.add_argument("--device", choices=DEVICES.keys(), default="zu7ev")
    ap.add_argument("--freq", type=float, default=200.0)
    args = ap.parse_args()
    dev_b, dev_d, dev_l = DEVICES[args.device]

    space = {
        "n_pes": [4, 8, 16],
        "wbuf_depth": [256, 512],
        "iabuf_depth": [2048, 4096],
        "q_max": [128, 256],
    }
    points = []
    for n_pes, wbuf, iabuf, qmax in itertools.product(*space.values()):
        hw = HwConfig(n_pes=n_pes, wbuf_depth=wbuf, iabuf_depth=iabuf,
                      q_max=qmax, freq_mhz=args.freq)
        cycles, valid = resnet50_cycles(hw)
        bram, dsp, klut = resources(hw)
        fits = bram <= dev_b and dsp <= dev_d and klut <= dev_l
        points.append(dict(n_pes=n_pes, wbuf=wbuf, iabuf=iabuf, qmax=qmax,
                           cycles=cycles, valid=valid, bram=bram, dsp=dsp,
                           klut=klut, fits=fits))

    # Pareto front over (cycles, bram) among valid+fitting points
    front = []
    for p in sorted((p for p in points if p["valid"] and p["fits"]),
                    key=lambda p: p["cycles"]):
        if not front or p["bram"] < front[-1]["bram"]:
            front.append(p)
    pareto = {(p["n_pes"], p["wbuf"], p["iabuf"], p["qmax"]) for p in front}

    hdr = (f"{'PEs':>4}{'wbuf':>6}{'iabuf':>7}{'qmax':>6}"
           f"{'cycles':>13}{'ms':>8}{'fps':>7}"
           f"{'BRAM18':>8}{'DSP':>6}{'kLUT':>6}{'fits':>6}{'pareto':>8}")
    print(f"ResNet-50 DSE @ {args.freq:.0f} MHz, device={args.device} "
          f"(BRAM18 {dev_b}, DSP {dev_d}, kLUT {dev_l})")
    print(hdr)
    print("-" * len(hdr))
    for p in sorted(points, key=lambda p: p["cycles"]):
        ms = p["cycles"] / (args.freq * 1e3)
        fps = 1e3 / ms if ms else 0
        mark = "*" if (p["n_pes"], p["wbuf"], p["iabuf"], p["qmax"]) in pareto \
            else ""
        note = "yes" if p["fits"] else "no"
        if not p["valid"]:
            note = "INVALID"
        print(f"{p['n_pes']:>4}{p['wbuf']:>6}{p['iabuf']:>7}{p['qmax']:>6}"
              f"{p['cycles']:>13,}{ms:>8.2f}{fps:>7.1f}"
              f"{p['bram']:>8}{p['dsp']:>6}{p['klut']:>6}{note:>6}{mark:>8}")

    print("\n'*' = Pareto-optimal (cycles vs BRAM) among valid+fitting points")
    print("'INVALID' = some ResNet-50 layer has no legal mapping "
          "(e.g. Q > q_max)")


if __name__ == "__main__":
    main()
