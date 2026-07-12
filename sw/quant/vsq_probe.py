#!/usr/bin/env python3
# =============================================================================
# vsq_probe.py — does per-vector weight scaling (VS-Quant) recover int4?
#
# A make-or-break experiment BEFORE touching the HLS datapath. Runs the full
# quantized ResNet-50 integer pipeline (same integer arithmetic as the kernel)
# with per-(output-channel, input-channel-group) weight scales, and reports
# int4 top-1 agreement vs the float model. If this recovers accuracy, the
# hardware change is a single `acc += dot * u[cgroup][lane]` multiply.
#
# Scheme (two-level, VS-Quant style, weights only):
#   real w = S_k (coarse fp, per output channel) * u[k,cg] (int, per C-group,
#            shared over kernel r,s) * w_int
#   acc_int[k] = sum_cg u[k,cg] * partial_conv_cg(w_int, a_int)   (a per-tensor)
#   out = requant(acc_int * S_k * sx / sy)     # coarse fold = existing mult
#
# Usage: python vsq_probe.py [--bits 4] [--ubits 4] [--pct 99.99]
# =============================================================================

import argparse
import os
import sys

import numpy as np
import torch
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_resnet50 as ex


def ceil_div(a, b):
    return (a + b - 1) // b


def conv_group(a, wq, stride, pad):
    """Integer partial conv over one C-group: a[V,H,W], wq[K,V,R,S] -> [K,P,Q]."""
    return ex.conv2d_np(a.astype(np.float64), wq.astype(np.float64),
                        np.zeros(wq.shape[0]), stride, pad)


def vsq_conv(a_int, wf, bf, K, R, S, stride, pad, sx, sy, qmax, qmin, V, U_MAX,
             per_channel_fallback, relu):
    """a_int: [C,H,W] int; wf: float weights [K,C,R,S]. Returns int8 [K,P,Q]
    plus the coarse per-channel mult/shift used (for reporting)."""
    C = wf.shape[1]
    ng = ceil_div(C, V)
    Cp = ng * V
    wpad = np.zeros((K, Cp, R, S), np.float64)
    wpad[:, :C] = wf
    apad = np.zeros((Cp,) + a_int.shape[1:], np.int64)
    apad[:C] = a_int

    # per-vector float scale: max|w| over each (k, cg) vector (shared over r,s)
    wv = wpad.reshape(K, ng, V, R, S)
    vecmax = np.abs(wv).max(axis=(2, 3, 4))            # [K, ng]
    sv_float = np.where(vecmax > 0, vecmax / qmax, 1.0)  # per (k,cg)

    if per_channel_fallback:
        # collapse to one scale per k (this reproduces the plain per-channel run)
        sk = sv_float.max(axis=1, keepdims=True)
        u = np.ones_like(sv_float)
        sv_float = sk * u
    else:
        # two-level: coarse S_k, per-vector integer u in [1, U_MAX]
        sk = sv_float.max(axis=1, keepdims=True) / U_MAX
        u = np.clip(np.round(sv_float / sk), 1, U_MAX)
        sv_float = sk * u                              # effective scale

    # quantize weights with the effective per-vector scale
    wq = np.clip(np.round(wpad.reshape(K, ng, V, R, S) /
                          sv_float[:, :, None, None, None]),
                 qmin, qmax).astype(np.int64)

    # integer accumulation: sum_cg u[k,cg] * partial_conv_cg
    P = (a_int.shape[1] + 2 * pad - R) // stride + 1
    Q = (a_int.shape[2] + 2 * pad - S) // stride + 1
    acc = np.zeros((K, P, Q), np.int64)
    for cg in range(ng):
        part = conv_group(apad[cg * V:(cg + 1) * V],
                          wq[:, cg], stride, pad)       # [K,P,Q] float(int-valued)
        acc += np.round(part).astype(np.int64) * u[:, cg][:, None, None].astype(np.int64)

    # bias into the coarse-scale integer domain (acc_int units = S_k * sx)
    bq = np.round(bf / (sk[:, 0] * sx)).astype(np.int64)
    acc += bq[:, None, None]

    # coarse requant: mult = round(S_k * sx / sy * 2^shift), per output channel
    M = (sk[:, 0] * sx / sy)                            # [K]
    _m, shift = ex.mult_shift(float(M.max()))
    mult = np.maximum(1, np.round(M * (1 << shift))).astype(np.int64)
    out = ex.sat_shift_np(acc, mult[:, None, None], shift, relu, qmax, qmin)
    return out.astype(np.int64)


def run(args):
    bits = args.bits
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    V = 16 if bits == 4 else 8
    U_MAX = (1 << args.ubits) - 1

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet50(weights=weights).eval()
    cats = weights.meta["categories"]
    graph = ex.build_graph(model)

    rng = np.random.default_rng(20260710)
    imgs = ex.calib_inputs(args.images, args.calib + args.eval, rng)
    n_cal = min(args.calib, len(imgs))

    # activation scales (per-tensor, percentile) via the float pass
    stats = {}
    for i in range(n_cal):
        ex.float_forward(graph, imgs[i], stats, args.pct)
    keep = ("input", "logits")
    scale = {k: ((s[0] if k in keep else s[1]) / qmax if
                 (s[0] if k in keep else s[1]) > 0 else 1.0)
             for k, s in stats.items()}

    def int_net(x, per_channel_only):
        xq = np.clip(np.round(x / scale["input"]), qmin, qmax).astype(np.int64)
        t = {"input": xq}
        for L in graph:
            if L["op"] == "conv":
                Kk, _Cc, Rr, Ss = L["w"].shape
                t[L["dst"]] = vsq_conv(
                    t[L["src"]], L["w"], L["b"], Kk, Rr, Ss,
                    L["stride"], L["pad"], scale[L["src"]], scale[L["dst"]],
                    qmax, qmin, V, U_MAX, per_channel_only, L["relu"])
            elif L["op"] == "maxpool":
                t[L["dst"]] = ex.maxpool_np(
                    t[L["src"]].astype(np.float64), L["R"], L["S"],
                    L["stride"], L["pad"]).astype(np.int64)
            elif L["op"] == "eltwise":
                sa, sb, so = scale[L["src"]], scale[L["src2"]], scale[L["dst"]]
                sh = 12
                mA = int(round(sa / so * (1 << sh)))
                mB = int(round(sb / so * (1 << sh)))
                acc = t[L["src"]] * mA + t[L["src2"]] * mB
                t[L["dst"]] = ex.sat_shift_np(acc, 1, sh, L["relu"], qmax, qmin)
            elif L["op"] == "gavgpool":
                si, so = scale[L["src"]], scale[L["dst"]]
                sh = 15
                C, H, W = t[L["src"]].shape
                mA = int(round(si / so / (H * W) * (1 << sh)))
                s = t[L["src"]].sum(axis=(1, 2), keepdims=True)
                t[L["dst"]] = ex.sat_shift_np(s, mA, sh, False, qmax, qmin)
        return t["logits"].reshape(-1)

    for mode, pc in (("per-channel", True), ("per-vector VSQ", False)):
        a1 = a5 = 0
        for x in imgs:
            with torch.no_grad():
                fl = model(torch.from_numpy(x).float().unsqueeze(0)).numpy().ravel()
            il = int_net(x, pc)
            ft1 = int(np.argmax(fl))
            a1 += (ft1 == int(np.argmax(il)))
            a5 += (ft1 in np.argsort(il)[-5:])
        n = len(imgs)
        print(f"int{bits} {mode:<16}: top-1 {a1}/{n}, float-top1 in top-5 {a5}/{n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--ubits", type=int, default=4)
    ap.add_argument("--pct", type=float, default=99.99)
    ap.add_argument("--calib", type=int, default=8)
    ap.add_argument("--eval", type=int, default=8)
    ap.add_argument("--images", default=os.path.join(
        os.path.dirname(__file__), "calib"))
    run(ap.parse_args())
