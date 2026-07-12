#!/usr/bin/env python3
# =============================================================================
# export_resnet50.py — PTQ quantization + export to the accelerator layout
#                      (Phase 6: real-weight pipeline, step 1)
#
# Pipeline:
#   1. Load torchvision's pretrained ResNet-50 and FOLD BatchNorm into the
#      preceding conv (the hardware has no BN).
#   2. Calibrate activation scales per tensor (images from sw/quant/calib/
#      if present, otherwise synthetic inputs — replace with real images for
#      meaningful accuracy).
#   3. Quantize weights per-tensor symmetric (int8 default, --int4).
#   4. Derive the hardware requantization constants:
#        conv    : out = sat(relu((acc + bias_q) * mult >> shift))
#                  M = S_x*S_w/S_y,  mult = round(M * 2^shift)
#        eltwise : out = sat(relu((A*multA + B*multB) >> shift))
#        avgpool : multA folds 1/count and the scale ratio
#   5. Pack weights/bias into the accelerator layout ([Kpad][R][S][C1] words,
#      C padded to VECTOR_SIZE, K padded to N_LANES) and write binaries plus
#      a manifest.json describing the full layer graph for the runtime.
#   6. Self-check: simulate the EXACT integer pipeline (same arithmetic as
#      magnet_top/global_pe) in numpy and compare top-1 vs the float model.
#
# Usage:
#   python export_resnet50.py [--int4] [--out DIR] [--calib N] [--images DIR]
# =============================================================================

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torchvision


# ---------------------------------------------------------------------------
# Hardware layout parameters (must match hw/include/accel_config.h)
# ---------------------------------------------------------------------------
N_LANES = 8


def hw_params(int4):
    bits = 4 if int4 else 8
    vec = 16 if int4 else 8
    qmax = (1 << (bits - 1)) - 1        # 127 / 7
    qmin = -(1 << (bits - 1))           # -128 / -8
    return bits, vec, qmax, qmin


# ---------------------------------------------------------------------------
# Step 1: BN folding — walk resnet50 and emit a flat layer graph
# ---------------------------------------------------------------------------
def fold_bn(conv_w, bn):
    gamma = bn.weight.detach().numpy()
    beta = bn.bias.detach().numpy()
    mean = bn.running_mean.detach().numpy()
    var = bn.running_var.detach().numpy()
    scale = gamma / np.sqrt(var + bn.eps)
    w = conv_w.detach().numpy() * scale[:, None, None, None]
    b = beta - mean * scale
    return w, b


def build_graph(model):
    """Flatten ResNet-50 into ops: conv / maxpool / eltwise / gavgpool.
    Tensors are named; convs carry folded float weights."""
    g = []  # list of dicts

    def conv(name, src, dst, w, b, stride, pad, relu):
        g.append(dict(op="conv", name=name, src=src, dst=dst, w=w, b=b,
                      stride=stride, pad=pad, relu=relu))

    conv("conv1", "input", "t_conv1",
         *fold_bn(model.conv1.weight, model.bn1), 2, 3, 1)
    g.append(dict(op="maxpool", name="maxpool", src="t_conv1", dst="t_pool",
                  R=3, S=3, stride=2, pad=1))

    src = "t_pool"
    for li, layer in enumerate(
            [model.layer1, model.layer2, model.layer3, model.layer4], start=1):
        for bi, blk in enumerate(layer):
            p = f"l{li}b{bi}"
            conv(f"{p}_1", src, f"t_{p}_1",
                 *fold_bn(blk.conv1.weight, blk.bn1), 1, 0, 1)
            conv(f"{p}_2", f"t_{p}_1", f"t_{p}_2",
                 *fold_bn(blk.conv2.weight, blk.bn2), blk.conv2.stride[0], 1, 1)
            conv(f"{p}_3", f"t_{p}_2", f"t_{p}_3",
                 *fold_bn(blk.conv3.weight, blk.bn3), 1, 0, 0)
            if blk.downsample is not None:
                conv(f"{p}_ds", src, f"t_{p}_ds",
                     *fold_bn(blk.downsample[0].weight, blk.downsample[1]),
                     blk.downsample[0].stride[0], 0, 0)
                sc = f"t_{p}_ds"
            else:
                sc = src
            g.append(dict(op="eltwise", name=f"{p}_add", src=f"t_{p}_3",
                          src2=sc, dst=f"t_{p}_out", relu=1))
            src = f"t_{p}_out"

    g.append(dict(op="gavgpool", name="gavgpool", src=src, dst="t_gap"))
    # fc as 1x1 conv on a 1x1 spatial tensor
    fc_w = model.fc.weight.detach().numpy()[:, :, None, None]  # [1000,2048,1,1]
    fc_b = model.fc.bias.detach().numpy()
    conv("fc", "t_gap", "logits", fc_w, fc_b, 1, 0, 0)
    return g


# ---------------------------------------------------------------------------
# Step 2: float reference forward (records per-tensor abs-max for calibration)
# ---------------------------------------------------------------------------
def conv2d_np(x, w, b, stride, pad):
    """x: [C,H,W] float64, w: [K,C,R,S] -> [K,P,Q] (im2col + matmul)."""
    C, H, W = x.shape
    K, _, R, S = w.shape
    P = (H + 2 * pad - R) // stride + 1
    Q = (W + 2 * pad - S) // stride + 1
    xp = np.zeros((C, H + 2 * pad, W + 2 * pad), dtype=x.dtype)
    xp[:, pad:pad + H, pad:pad + W] = x
    cols = np.empty((C * R * S, P * Q), dtype=x.dtype)
    idx = 0
    for c in range(C):
        for r in range(R):
            for s in range(S):
                cols[idx] = xp[c, r:r + P * stride:stride,
                               s:s + Q * stride:stride].reshape(-1)
                idx += 1
    out = w.reshape(K, -1) @ cols + b[:, None]
    return out.reshape(K, P, Q)


def maxpool_np(x, R, S, stride, pad):
    C, H, W = x.shape
    P = (H + 2 * pad - R) // stride + 1
    Q = (W + 2 * pad - S) // stride + 1
    xp = np.full((C, H + 2 * pad, W + 2 * pad), -np.inf, dtype=x.dtype)
    xp[:, pad:pad + H, pad:pad + W] = x
    out = np.full((C, P, Q), -np.inf, dtype=x.dtype)
    for r in range(R):
        for s in range(S):
            out = np.maximum(out, xp[:, r:r + P * stride:stride,
                                     s:s + Q * stride:stride])
    return out


def float_forward(graph, x, stats, pct=100.0):
    """Float forward. Records per-tensor stats[name] = [abs-max, pct-max]:
    the true abs-max and the pct-th percentile of |values| (per image,
    max-reduced across images). pct>=100 makes both identical."""
    def rec(name, v):
        a = float(np.abs(v).max())
        p = a if pct >= 100.0 else float(np.percentile(np.abs(v), pct))
        s = stats.setdefault(name, [0.0, 0.0])
        s[0] = max(s[0], a)
        s[1] = max(s[1], p)

    t = {"input": x}
    rec("input", x)
    for L in graph:
        if L["op"] == "conv":
            y = conv2d_np(t[L["src"]], L["w"].astype(np.float64),
                          L["b"].astype(np.float64), L["stride"], L["pad"])
            if L["relu"]:
                y = np.maximum(y, 0)
        elif L["op"] == "maxpool":
            y = maxpool_np(t[L["src"]], L["R"], L["S"], L["stride"], L["pad"])
        elif L["op"] == "eltwise":
            y = t[L["src"]] + t[L["src2"]]
            if L["relu"]:
                y = np.maximum(y, 0)
        elif L["op"] == "gavgpool":
            y = t[L["src"]].mean(axis=(1, 2), keepdims=True)
        t[L["dst"]] = y
        rec(L["dst"], y)
    return t["logits"].reshape(-1)


# ---------------------------------------------------------------------------
# Step 3/4: quantization parameter derivation
# ---------------------------------------------------------------------------
def mult_shift(M, max_mult=1 << 15):
    """mult = round(M * 2^shift) with mult in [max_mult/2, max_mult)."""
    if M <= 0:
        return 0, 1
    shift = 0
    while round(M * (1 << shift)) < max_mult // 2 and shift < 40:
        shift += 1
    m = int(round(M * (1 << shift)))
    if m >= max_mult:
        m //= 2
        shift -= 1
    return m, shift


def quantize(graph, amax, qmax):
    """Attach integer weights and requant constants to every layer.
    Weights are quantized PER OUTPUT CHANNEL; the requant multiplier is a
    per-channel table with a shared per-layer shift (matches the hardware:
    gmem_mult[Kpad] + scalar shift)."""
    scale = {name: (v / qmax if v > 0 else 1.0) for name, v in amax.items()}
    q = []
    for L in graph:
        e = dict(L)  # shallow copy
        if L["op"] == "conv":
            w = L["w"]                                # [K,C,R,S]
            K = w.shape[0]
            swc = np.abs(w).reshape(K, -1).max(axis=1) / qmax
            swc[swc == 0] = 1.0                       # dead channels
            # Floor the per-channel scale spread: near-dead channels would
            # otherwise blow up bq = b/(sx*swc) past int32 (the hardware
            # accumulates bias in 32 bits). /256 keeps virtually all of the
            # per-channel benefit while bounding bias and psum headroom.
            swc = np.maximum(swc, swc.max() / 256.0)
            wq = np.clip(np.round(w / swc[:, None, None, None]),
                         -qmax - 1, qmax).astype(np.int32)
            sx, sy = scale[L["src"]], scale[L["dst"]]
            bq = np.round(L["b"] / (sx * swc)).astype(np.int64)
            if np.abs(bq).max() >= 2**30:
                print(f"  WARNING {L['name']}: |bias_q| max "
                      f"{np.abs(bq).max():.3g} nears int32 range")
            M = sx * swc / sy                         # per-channel
            _m, sh = mult_shift(float(M.max()))       # shared shift
            mults = np.maximum(
                1, np.round(M * (1 << sh))).astype(np.int64)
            e.update(wq=wq, bq=bq, mults=mults, shift=sh, sw=swc)
        elif L["op"] == "eltwise":
            sa, sb = scale[L["src"]], scale[L["src2"]]
            so = scale[L["dst"]]
            sh = 12
            e.update(multA=int(round(sa / so * (1 << sh))),
                     multB=int(round(sb / so * (1 << sh))), shift=sh)
        elif L["op"] == "gavgpool":
            si, so = scale[L["src"]], scale[L["dst"]]
            sh = 15
            # count is resolved at runtime (H*W); store the scale ratio
            e.update(scale_ratio=si / so, shift=sh)
        q.append(e)
    return q, scale


# ---------------------------------------------------------------------------
# Step 6: exact integer pipeline simulation (mirrors magnet_top / global_pe)
# ---------------------------------------------------------------------------
def sat_shift_np(x, mult, shift, relu, qmax, qmin):
    y = x.astype(np.int64) * np.asarray(mult, dtype=np.int64)
    if shift > 0:
        y += 1 << (shift - 1)
    y >>= shift
    if relu:
        y = np.maximum(y, 0)
    return np.clip(y, qmin, qmax)


def int_forward(qgraph, xq, qmax, qmin, keep_tensors=False):
    t = {"input": xq.astype(np.int64)}
    for L in qgraph:
        if L["op"] == "conv":
            # float64 matmul is exact here (all values integers << 2^53)
            acc = conv2d_np(t[L["src"]].astype(np.float64),
                            L["wq"].astype(np.float64),
                            L["bq"].astype(np.float64), L["stride"], L["pad"])
            acc = np.round(acc).astype(np.int64)
            y = sat_shift_np(acc, L["mults"][:, None, None], L["shift"],
                             L["relu"], qmax, qmin)
        elif L["op"] == "maxpool":
            y = maxpool_np(t[L["src"]].astype(np.float64), L["R"], L["S"],
                           L["stride"], L["pad"]).astype(np.int64)
        elif L["op"] == "eltwise":
            acc = t[L["src"]] * L["multA"] + t[L["src2"]] * L["multB"]
            y = sat_shift_np(acc, 1, L["shift"], L["relu"], qmax, qmin)
        elif L["op"] == "gavgpool":
            C, H, W = t[L["src"]].shape
            multA = int(round(L["scale_ratio"] / (H * W) * (1 << L["shift"])))
            s = t[L["src"]].sum(axis=(1, 2), keepdims=True)
            y = sat_shift_np(s, multA, L["shift"], False, qmax, qmin)
            L["multA"] = multA  # resolved value for the manifest
        t[L["dst"]] = y
    if keep_tensors:
        return t["logits"].reshape(-1), t
    return t["logits"].reshape(-1)


# ---------------------------------------------------------------------------
# Step 5: pack into the accelerator layout and write files
# ---------------------------------------------------------------------------
def pack_nibbles(vals):
    """int4: two signed nibbles per byte, low nibble = even element."""
    v = (np.asarray(vals, dtype=np.int64) & 0xF).astype(np.uint8)
    if len(v) % 2:
        v = np.append(v, 0)
    return (v[0::2] | (v[1::2] << 4)).astype(np.uint8)


def pack_weights(wq, vec, bits):
    """[K,C,R,S] int -> bytes in [Kpad][R][S][C1] word order."""
    K, C, R, S = wq.shape
    Kp = ((K + N_LANES - 1) // N_LANES) * N_LANES
    Cp = ((C + vec - 1) // vec) * vec
    buf = np.zeros((Kp, R, S, Cp), dtype=np.int64)
    buf[:K, :, :, :C] = wq.transpose(0, 2, 3, 1)  # KRSC
    flat = buf.reshape(-1)
    if bits == 4:
        return pack_nibbles(flat).tobytes()
    return flat.astype(np.int8).tobytes()


def write_export(qgraph, scale, meta, outdir, vec, bits):
    os.makedirs(outdir, exist_ok=True)
    layers = []
    for L in qgraph:
        e = {k: L[k] for k in ("op", "name", "src", "dst") if k in L}
        if L["op"] == "conv":
            K, C, R, S = L["wq"].shape
            e.update(C=C, K=K, R=R, S=S, stride=L["stride"], pad=L["pad"],
                     relu=L["relu"], shift=L["shift"])
            wfile = f"{L['name']}_w.bin"
            bfile = f"{L['name']}_b.bin"
            mfile = f"{L['name']}_m.bin"
            with open(os.path.join(outdir, wfile), "wb") as f:
                f.write(pack_weights(L["wq"], vec, bits))
            Kp = ((K + N_LANES - 1) // N_LANES) * N_LANES
            bq = np.zeros(Kp, dtype=np.int32)
            bq[:K] = L["bq"].astype(np.int32)
            with open(os.path.join(outdir, bfile), "wb") as f:
                f.write(bq.tobytes())
            mq = np.zeros(Kp, dtype=np.int32)
            mq[:K] = L["mults"].astype(np.int32)
            with open(os.path.join(outdir, mfile), "wb") as f:
                f.write(mq.tobytes())
            e.update(w_file=wfile, b_file=bfile, m_file=mfile)
        elif L["op"] == "maxpool":
            e.update(R=L["R"], S=L["S"], stride=L["stride"], pad=L["pad"])
        elif L["op"] == "eltwise":
            e.update(src2=L["src2"], multA=L["multA"], multB=L["multB"],
                     shift=L["shift"], relu=L["relu"])
        elif L["op"] == "gavgpool":
            e.update(multA=L.get("multA", 0), shift=L["shift"])
        layers.append(e)

    manifest = dict(
        model="resnet50", bits=bits, vector_size=vec, n_lanes=N_LANES,
        input=dict(shape=[3, 224, 224], scale=scale["input"],
                   preprocess="torchvision imagenet (mean/std normalize)"),
        output=dict(tensor="logits", scale=scale["logits"]),
        layers=layers, **meta)
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    # machine-readable manifest for the C++ loader (space-separated)
    with open(os.path.join(outdir, "manifest.txt"), "w") as f:
        f.write(f"model resnet50 {bits} {vec} {N_LANES}\n")
        f.write("input 224 224 3 test_input.bin test_logits.bin\n")
        for L in qgraph:
            if L["op"] == "conv":
                K, C, R, S = L["wq"].shape
                f.write(f"conv {L['name']} {L['src']} {L['dst']} "
                        f"{C} {K} {R} {S} {L['stride']} {L['pad']} "
                        f"{L['relu']} {L['shift']} "
                        f"{L['name']}_w.bin {L['name']}_b.bin "
                        f"{L['name']}_m.bin\n")
            elif L["op"] == "maxpool":
                f.write(f"maxpool {L['name']} {L['src']} {L['dst']} "
                        f"{L['R']} {L['S']} {L['stride']} {L['pad']}\n")
            elif L["op"] == "eltwise":
                f.write(f"eltwise {L['name']} {L['src']} {L['src2']} "
                        f"{L['dst']} {L['multA']} {L['multB']} "
                        f"{L['shift']} {L['relu']}\n")
            elif L["op"] == "gavgpool":
                f.write(f"gavgpool {L['name']} {L['src']} {L['dst']} "
                        f"{L.get('multA', 0)} {L['shift']}\n")
    return manifest


def write_test_vectors(outdir, xq, tensors, logits):
    """Test input (HWC int8), expected logits (int8) and per-tensor
    checksums from the integer simulation — consumed by the C++ loader."""
    with open(os.path.join(outdir, "test_input.bin"), "wb") as f:
        f.write(xq.transpose(1, 2, 0).astype(np.int8).tobytes())  # CHW->HWC
    with open(os.path.join(outdir, "test_logits.bin"), "wb") as f:
        f.write(logits.astype(np.int8).tobytes())
    with open(os.path.join(outdir, "checks.txt"), "w") as f:
        for name, v in tensors.items():
            if name == "input":
                continue
            f.write(f"{name} {v.size} {int(v.sum())}\n")


# ---------------------------------------------------------------------------
# Calibration inputs
# ---------------------------------------------------------------------------
def calib_inputs(images_dir, n, rng):
    tfm_available = True
    try:
        from PIL import Image
        from torchvision import transforms
        tfm = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
    except ImportError:
        tfm_available = False

    imgs = []
    if images_dir and os.path.isdir(images_dir) and tfm_available:
        for fn in sorted(os.listdir(images_dir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                img = Image.open(os.path.join(images_dir, fn)).convert("RGB")
                imgs.append(tfm(img).numpy().astype(np.float64))
                if len(imgs) >= n:
                    break
    if imgs:
        print(f"calibration: {len(imgs)} real image(s) from {images_dir}")
        return imgs
    print(f"calibration: {n} SYNTHETIC input(s) -- drop .jpg files into "
          f"sw/quant/calib/ for meaningful scales/accuracy")
    return [rng.standard_normal((3, 224, 224)) for _ in range(n)]


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ResNet-50 PTQ export")
    ap.add_argument("--int4", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "export"))
    ap.add_argument("--calib", type=int, default=8, help="calibration inputs")
    ap.add_argument("--eval", type=int, default=8,
                    help="extra images for float-vs-int agreement eval")
    ap.add_argument("--images", default=os.path.join(
        os.path.dirname(__file__), "calib"))
    ap.add_argument("--pct", type=float, default=99.99,
                    help="activation calibration percentile "
                         "(100 = abs-max; weights/input/logits always abs-max)."
                         " Swept on fruits262: 99.99 best (11/16), abs-max"
                         " 10/16, 99.9 too aggressive (7/16)")
    args = ap.parse_args()
    bits, vec, qmax, qmin = hw_params(args.int4)

    print("loading torchvision resnet50 (downloads weights on first run)...")
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet50(weights=weights)
    model.eval()
    cats = weights.meta["categories"]

    graph = build_graph(model)
    print(f"graph: {sum(1 for l in graph if l['op'] == 'conv')} convs, "
          f"{sum(1 for l in graph if l['op'] == 'eltwise')} eltwise, "
          f"2 pools")

    rng = np.random.default_rng(20260710)
    inputs = calib_inputs(args.images, args.calib + args.eval, rng)
    n_cal = min(args.calib, len(inputs))

    # calibration: numpy float forward on the first n_cal inputs collects
    # per-tensor [abs-max, percentile] (names align with the quantized graph)
    stats = {}
    for i in range(n_cal):
        float_forward(graph, inputs[i], stats, args.pct)
        print(f"  calib {i + 1}/{n_cal} done")

    # percentile clipping for intermediate activations; abs-max where the
    # extreme value itself matters (input quantization, logits argmax)
    keep_max = ("input", "logits")
    amax = {n: (s[0] if n in keep_max else s[1]) for n, s in stats.items()}
    clipped = [(n, s[1] / s[0]) for n, s in stats.items()
               if n not in keep_max and s[0] > 0]
    if clipped and args.pct < 100.0:
        ratios = sorted(r for _n, r in clipped)
        print(f"  percentile {args.pct}: scale tightening ratio "
              f"median {ratios[len(ratios) // 2]:.3f}, "
              f"min {ratios[0]:.3f} over {len(clipped)} tensors")

    qgraph, scale = quantize(graph, amax, qmax)

    # agreement eval over ALL loaded inputs: float logits via torch (fast),
    # integer pipeline via the exact numpy simulation. The first input's
    # tensors/logits become the C++ loader's test vectors.
    agree1 = agree5 = 0
    tv = None
    for i, x in enumerate(inputs):
        with torch.no_grad():
            fl = model(torch.from_numpy(x).float().unsqueeze(0)) \
                .numpy().ravel()
        xq = np.clip(np.round(x / scale["input"]), qmin, qmax)
        if i == 0:
            il, tensors = int_forward(qgraph, xq, qmax, qmin,
                                      keep_tensors=True)
            tv = (xq, tensors, il)
            f5 = [cats[k] for k in np.argsort(fl)[-5:][::-1]]
            i5 = [cats[k] for k in np.argsort(il)[-5:][::-1]]
            print(f"  test image float top-5: {f5}")
            print(f"  test image int{bits} top-5: {i5}")
        else:
            il = int_forward(qgraph, xq, qmax, qmin)
        ft1 = int(np.argmax(fl))
        agree1 += (ft1 == int(np.argmax(il)))
        agree5 += (ft1 in np.argsort(il)[-5:])
        print(f"  eval {i + 1}/{len(inputs)}: float='{cats[ft1]}' "
              f"int='{cats[int(np.argmax(il))]}'")
    n = len(inputs)
    print(f"quantization fidelity (int{bits} vs float, {n} images): "
          f"top-1 agreement {agree1}/{n}, float-top1 in int-top5 {agree5}/{n}")

    meta = dict(selfcheck_top1_agree=f"{agree1}/{n}",
                calibration_pct=args.pct,
                calibration="real" if os.path.isdir(args.images) and
                any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                    for f in os.listdir(args.images)) else "synthetic")
    manifest = write_export(qgraph, scale, meta, args.out, vec, bits)
    write_test_vectors(args.out, *tv)
    # ImageNet class names (index -> label) for the classifier front-end
    with open(os.path.join(args.out, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(cats))
    total = sum(os.path.getsize(os.path.join(args.out, f))
                for f in os.listdir(args.out))
    print(f"export: {len(manifest['layers'])} layers -> {args.out} "
          f"({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
