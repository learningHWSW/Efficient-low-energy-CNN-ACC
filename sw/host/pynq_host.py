#!/usr/bin/env python3
# =============================================================================
# pynq_host.py — drive the magnet_top conv accelerator from PYNQ (Phase 4b)
#
# Runs ON THE BOARD (ZCU104 with the PYNQ image, aarch64). This is the
# XRT-free deployment path that pairs with the Windows/Vivado bitstream
# (hw/scripts/build_vivado.tcl): the kernel is driven directly through its
# s_axilite register map, so no Vitis platform / XRT runtime is required.
#
# The default bitstream contains magnet_top only (the conv engine). This
# script therefore runs every conv layer on the FPGA and the pooling /
# element-wise ops on the ARM in numpy — a hybrid that still executes the
# full ResNet-50 from sw/quant/export. (Add global_pe_top to the bitstream,
# via KERNELS in build_vivado.tcl, to offload those too.)
#
# Usage (on the board):
#   sudo python3 pynq_host.py magnet_top.bit export_dir/
#
# where export_dir/ is a copy of sw/quant/export (manifest.txt + *.bin).
# =============================================================================

import os
import sys
import struct
import numpy as np

VECTOR_SIZE = 8      # must match the synthesized bitstream (int8 default)
N_LANES = 8
N_PES = 8
WBUF_DEPTH = 256
IABUF_DEPTH = 2048
Q_MAX = 128
MIN_QB = 16
PT_ROWS = 8
N_IA_PORTS = 4

# ---- s_axilite register offsets (build/hls/hls/syn/report/csynth.rpt) ------
R_CTRL = 0x00
R_PTR = dict(ia0=0x10, ia1=0x1c, ia2=0x28, ia3=0x34,
             w=0x40, oa=0x4c, bias=0x58, mult=0x64)
R_SCAL = dict(H=0x70, W=0x78, C1=0x80, K1=0x88, K=0x90, P=0x98, Q=0xa0,
              R=0xa8, S=0xb0, stride=0xb8, pad=0xc0, CT1=0xc8, KP=0xd0,
              CP=0xd8, mult=0xe0, shift=0xe8, relu_en=0xf0)


# ---- plan_conv: mirror of sw/runtime/net_runner.h -------------------------
def ceil_div(a, b):
    return (a + b - 1) // b


def plan_conv(C1, K1, P, Q, R, S, stride):
    best = None
    kp = 1
    while kp <= N_PES:
        if N_PES % kp == 0:
            cp = N_PES // kp
            if C1 % cp == 0:
                rs = R * S
                qb = max(Q, MIN_QB)
                win = (qb - 1) * stride + S
                c1s = C1 // cp
                ct1 = min(c1s, WBUF_DEPTH // rs, IABUF_DEPTH // (R * win))
                if ct1 >= 1 and Q <= Q_MAX:
                    C2 = ceil_div(c1s, ct1)
                    P2 = ceil_div(P, PT_ROWS)
                    K1G = ceil_div(K1, kp)
                    blk = max(PT_ROWS * rs * c1s * qb,
                              PT_ROWS * R * win * c1s * ceil_div(cp, N_IA_PORTS),
                              PT_ROWS * Q * kp)
                    cyc = K1G * P2 * blk + \
                        K1G * (1 if C2 == 1 else P2 * C2) * N_PES * N_LANES * \
                        rs * (c1s if C2 == 1 else ct1)
                    if best is None or cyc < best[3]:
                        best = (kp, cp, ct1, cyc)
        kp *= 2
    return best  # (KP, CP, CT1, cycles) or None


class MagnetConv:
    def __init__(self, bitfile):
        from pynq import Overlay, allocate
        self._allocate = allocate
        self.ol = Overlay(bitfile)
        # the HLS IP appears as <name>_0; grab the first magnet_top instance
        ipname = next(n for n in self.ol.ip_dict if "magnet_top" in n)
        self.ip = getattr(self.ol, ipname)
        print(f"overlay loaded, kernel = {ipname}")

    def _wptr(self, off, buf):
        addr = buf.device_address
        self.ip.write(off, addr & 0xffffffff)
        self.ip.write(off + 4, (addr >> 32) & 0xffffffff)

    def run_conv(self, in_hwc_i8, C, wq_bytes, bias_i32, mult_i32,
                 K, R, S, stride, pad, shift, relu):
        """in_hwc_i8: [H,W,C] int8 numpy. Returns [P,Q,K] int8."""
        H, W = in_hwc_i8.shape[:2]
        C1 = C // VECTOR_SIZE
        K1 = ceil_div(K, N_LANES)
        Kp = K1 * N_LANES
        P = (H + 2 * pad - R) // stride + 1
        Q = (W + 2 * pad - S) // stride + 1
        plan = plan_conv(C1, K1, P, Q, R, S, stride)
        assert plan, "no valid mapping"
        KP, CP, CT1, _ = plan

        alloc = self._allocate
        # IA words are contiguous [H][W][C1] int8 = the HWC byte stream
        bo_ia = alloc((H * W * C,), dtype=np.int8)
        bo_ia[:] = in_hwc_i8.reshape(-1)
        bo_w = alloc((len(wq_bytes),), dtype=np.int8)
        bo_w[:] = np.frombuffer(wq_bytes, dtype=np.int8)
        bo_oa = alloc((P * Q * K1 * N_LANES,), dtype=np.int8)
        bo_b = alloc((Kp,), dtype=np.int32); bo_b[:] = bias_i32
        bo_m = alloc((Kp,), dtype=np.int32); bo_m[:] = mult_i32
        for b in (bo_ia, bo_w, bo_b, bo_m):
            b.sync_to_device()

        ia_addr = bo_ia.device_address
        for name in ("ia0", "ia1", "ia2", "ia3"):     # all alias one buffer
            self.ip.write(R_PTR[name], ia_addr & 0xffffffff)
            self.ip.write(R_PTR[name] + 4, (ia_addr >> 32) & 0xffffffff)
        self._wptr(R_PTR["w"], bo_w)
        self._wptr(R_PTR["oa"], bo_oa)
        self._wptr(R_PTR["bias"], bo_b)
        self._wptr(R_PTR["mult"], bo_m)
        for name, val in dict(H=H, W=W, C1=C1, K1=K1, K=K, P=P, Q=Q, R=R, S=S,
                              stride=stride, pad=pad, CT1=CT1, KP=KP, CP=CP,
                              mult=1, shift=shift, relu_en=relu).items():
            self.ip.write(R_SCAL[name], int(val) & 0xffffffff)

        self.ip.write(R_CTRL, 1)                        # ap_start
        while (self.ip.read(R_CTRL) & 0x2) == 0:        # poll ap_done
            pass
        bo_oa.sync_from_device()
        out = np.array(bo_oa).reshape(P, Q, K1 * N_LANES)[:, :, :K].copy()
        for b in (bo_ia, bo_w, bo_oa, bo_b, bo_m):
            b.freebuffer()
        return out.astype(np.int8)


# ---- ARM-side ops (not in the conv-only bitstream) ------------------------
def sat8(x, shift, relu):
    y = x.astype(np.int64)
    if shift > 0:
        y = y + (1 << (shift - 1))
    y >>= shift
    if relu:
        y = np.maximum(y, 0)
    return np.clip(y, -128, 127).astype(np.int8)


def maxpool(x, R, S, st, pad):
    H, W, C = x.shape
    P = (H + 2 * pad - R) // st + 1
    Q = (W + 2 * pad - S) // st + 1
    xp = np.full((H + 2 * pad, W + 2 * pad, C), -128, np.int32)
    xp[pad:pad + H, pad:pad + W] = x
    o = np.full((P, Q, C), -128, np.int32)
    for r in range(R):
        for s in range(S):
            o = np.maximum(o, xp[r:r + P * st:st, s:s + Q * st:st])
    return o.astype(np.int8)


def main():
    if len(sys.argv) < 3:
        print("usage: sudo python3 pynq_host.py <bitfile> <export_dir>")
        return
    bitfile, export = sys.argv[1], sys.argv[2]
    dev = MagnetConv(bitfile)

    def rd(name):
        return open(os.path.join(export, name), "rb").read()

    tensors = {}
    logits_name = None
    with open(os.path.join(export, "manifest.txt")) as mf:
        for line in mf:
            p = line.split()
            if not p:
                continue
            if p[0] == "input":
                H, W, C = int(p[1]), int(p[2]), int(p[3])
                x = np.frombuffer(rd(p[4]), np.int8).reshape(H, W, C)
                tensors["input"] = x
            elif p[0] == "conv":
                (_, name, src, dst, C, K, R, S, st, pad, relu, shift,
                 wf, bf, mfn) = p
                y = dev.run_conv(
                    tensors[src], int(C), rd(wf),
                    np.frombuffer(rd(bf), np.int32),
                    np.frombuffer(rd(mfn), np.int32),
                    int(K), int(R), int(S), int(st), int(pad),
                    int(shift), int(relu))
                tensors[dst] = y
                logits_name = dst
            elif p[0] == "maxpool":
                _, name, src, dst, R, S, st, pad = p
                tensors[dst] = maxpool(tensors[src], int(R), int(S),
                                       int(st), int(pad))
            elif p[0] == "eltwise":
                _, name, s1, s2, dst, mA, mB, sh, relu = p
                a, b = tensors[s1].astype(np.int64), tensors[s2].astype(np.int64)
                tensors[dst] = sat8(a * int(mA) + b * int(mB), int(sh),
                                    int(relu))
            elif p[0] == "gavgpool":
                _, name, src, dst, mA, sh = p
                s = tensors[src].astype(np.int64).sum(axis=(0, 1))
                tensors[dst] = sat8(s * int(mA), int(sh), 0).reshape(1, 1, -1)

    logits = tensors[logits_name].reshape(-1)
    top5 = np.argsort(logits)[-5:][::-1]
    print("top-5 class indices:", top5.tolist())
    exp = os.path.join(export, "test_logits.bin")
    if os.path.exists(exp):
        want = np.frombuffer(rd("test_logits.bin"), np.int8)
        n = min(len(want), len(logits))
        mism = int(np.sum(want[:n] != logits[:n].astype(np.int8)))
        print(f"logits vs expected: {mism}/{n} mismatches")


if __name__ == "__main__":
    main()
