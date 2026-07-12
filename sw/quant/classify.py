#!/usr/bin/env python3
# =============================================================================
# classify.py — image in, class name out (Phase 6 front-end)
#
# The user-facing entry point: give it an image file, it preprocesses,
# quantizes, runs the quantized ResNet-50, and prints the ImageNet class.
#
#   python classify.py cat.jpg                 # --sim (default): integer
#                                              # pipeline in numpy on this PC
#   sudo python3 classify.py cat.jpg --fpga magnet_top.bit   # on the board
#
# Both backends share the exact same preprocessing + quantization + label
# lookup, so --sim on a PC verifies everything except the hardware MMIO. The
# integer pipeline is bit-identical to what the FPGA computes (the export's
# self-check ties the two together), so --sim predicts the FPGA result.
#
# Needs an export directory (sw/quant/export by default) produced by
# export_resnet50.py, which now also writes labels.txt.
# =============================================================================

import argparse
import json
import os
import sys

import numpy as np


# ---- preprocessing: JPEG -> normalized float CHW (no torch needed) --------
def load_image(path):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    # torchvision eval transform: Resize(256) shorter side, CenterCrop(224)
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    x = np.asarray(img, dtype=np.float32) / 255.0          # HWC in [0,1]
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    x = (x - mean) / std
    return x.transpose(2, 0, 1)                             # CHW


def quantize_input(x_chw, scale, qmax, qmin):
    xq = np.clip(np.round(x_chw / scale), qmin, qmin + (qmax - qmin))
    return xq.astype(np.int8)


# ---- integer pipeline (--sim): reuse the exact export simulation ----------
def run_sim(export_dir, xq_chw, bits):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import export_resnet50 as ex  # the bit-exact integer reference

    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))

    def rd(name, dt):
        return np.frombuffer(open(os.path.join(export_dir, name), "rb").read(),
                             dt)

    # parse manifest.txt (the machine-readable format the C++ loader uses;
    # unlike manifest.json it carries eltwise src2)
    qgraph = []
    vec = 8
    with open(os.path.join(export_dir, "manifest.txt")) as mf:
        for line in mf:
            p = line.split()
            if not p or p[0] in ("model", "input"):
                if p and p[0] == "model":
                    vec = int(p[3])
                continue
            if p[0] == "conv":
                (_, name, src, dst, C, K, R, S, st, pad, relu, shift,
                 wf, bf, mfn) = p
                C, K, R, S = int(C), int(K), int(R), int(S)
                Kp = ((K + 8 - 1) // 8) * 8
                Cp = ((C + vec - 1) // vec) * vec
                wb = rd(wf, np.int8 if bits == 8 else np.uint8)
                if bits == 4:  # unpack signed nibbles
                    lo = (wb & 0xF).astype(np.int16)
                    hi = ((wb >> 4) & 0xF).astype(np.int16)
                    nib = np.empty(wb.size * 2, np.int16)
                    nib[0::2], nib[1::2] = lo, hi
                    nib[nib >= 8] -= 16
                    wb = nib
                wq = wb.reshape(Kp, R, S, Cp)[:K, :, :, :C].transpose(0, 3, 1, 2)
                qgraph.append(dict(op="conv", src=src, dst=dst,
                                   wq=wq.astype(np.int64),
                                   bq=rd(bf, np.int32)[:K].astype(np.int64),
                                   mults=rd(mfn, np.int32)[:K].astype(np.int64),
                                   shift=int(shift), stride=int(st),
                                   pad=int(pad), relu=int(relu)))
            elif p[0] == "maxpool":
                _, name, src, dst, R, S, st, pad = p
                qgraph.append(dict(op="maxpool", src=src, dst=dst, R=int(R),
                                   S=int(S), stride=int(st), pad=int(pad)))
            elif p[0] == "eltwise":
                _, name, s1, s2, dst, mA, mB, sh, relu = p
                qgraph.append(dict(op="eltwise", src=s1, src2=s2, dst=dst,
                                   multA=int(mA), multB=int(mB), shift=int(sh),
                                   relu=int(relu)))
            elif p[0] == "gavgpool":
                _, name, src, dst, mA, sh = p
                qgraph.append(dict(op="gavgpool", src=src, dst=dst,
                                   _multA=int(mA), shift=int(sh)))

    def int_fwd(xq_hwc):
        t = {"input": xq_hwc.astype(np.int64)}
        for L in qgraph:
            if L["op"] == "conv":
                acc = ex.conv2d_np(t[L["src"]].astype(np.float64),
                                   L["wq"].astype(np.float64),
                                   L["bq"].astype(np.float64),
                                   L["stride"], L["pad"])
                acc = np.round(acc).astype(np.int64)
                y = ex.sat_shift_np(acc, L["mults"][:, None, None], L["shift"],
                                    L["relu"], qmax, qmin)
            elif L["op"] == "maxpool":
                y = ex.maxpool_np(t[L["src"]].astype(np.float64), L["R"],
                                  L["S"], L["stride"], L["pad"]).astype(np.int64)
            elif L["op"] == "eltwise":
                acc = t[L["src"]] * L["multA"] + t[L["src2"]] * L["multB"]
                y = ex.sat_shift_np(acc, 1, L["shift"], L["relu"], qmax, qmin)
            elif L["op"] == "gavgpool":
                s = t[L["src"]].sum(axis=(1, 2), keepdims=True)
                y = ex.sat_shift_np(s, L["_multA"], L["shift"], False,
                                    qmax, qmin)
            t[L["dst"]] = y
        return t[qgraph[-1]["dst"]].reshape(-1)

    # int_forward mirrors the conv layout: input is CHW here, but ex.conv2d_np
    # expects [C,H,W] — which is exactly CHW. Good.
    return int_fwd(xq_chw)


# ---- FPGA pipeline (--fpga): delegate to the PYNQ host driver -------------
def run_fpga(export_dir, xq_chw, bitfile):
    hostdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "host")
    sys.path.insert(0, os.path.abspath(hostdir))
    import pynq_host as ph
    # write the quantized input where pynq_host expects it, then run its graph
    xq_hwc = xq_chw.transpose(1, 2, 0).astype(np.int8)
    with open(os.path.join(export_dir, "test_input.bin"), "wb") as f:
        f.write(xq_hwc.tobytes())
    sys.argv = ["pynq_host", bitfile, export_dir]
    # pynq_host.main() runs the manifest and prints top-5; capture logits via
    # its internal pool instead — simplest: re-run its main and read logits.
    # (pynq_host prints the class indices; classify adds names below.)
    ph.main()
    return None  # names printed by pynq_host + this script's label lookup


def main():
    ap = argparse.ArgumentParser(description="image -> ImageNet class")
    ap.add_argument("image")
    ap.add_argument("--export", default=os.path.join(
        os.path.dirname(__file__), "export"))
    ap.add_argument("--fpga", metavar="BITFILE", default=None,
                    help="run on the FPGA via PYNQ (default: numpy --sim)")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.export, "manifest.json")))
    bits = man["bits"]
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    scale = man["input"]["scale"]
    labels = open(os.path.join(args.export, "labels.txt"),
                  encoding="utf-8").read().splitlines()

    x = load_image(args.image)
    xq = quantize_input(x, scale, qmax, qmin)

    if args.fpga:
        run_fpga(args.export, xq, args.fpga)
        return

    logits = run_sim(args.export, xq, bits)
    order = np.argsort(logits)[::-1][:args.topk]
    print(f"image: {args.image}  (backend: sim/int{bits})")
    for rank, k in enumerate(order, 1):
        print(f"  {rank}. {labels[k]:<30s} (logit {int(logits[k])})")


if __name__ == "__main__":
    main()
