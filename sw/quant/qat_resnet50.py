#!/usr/bin/env python3
# =============================================================================
# qat_resnet50.py — quantization-aware training for the int4 accelerator
#
# PTQ (even per-channel + per-vector) can't recover int4 on ResNet-50 (the
# 4-bit ACTIVATION resolution is the bottleneck; see vsq_probe.py). The fix is
# QAT: fine-tune with fake-quantization in the loop so the network learns to be
# robust to 4-bit. The fake-quant here mirrors the FPGA exactly — per-output-
# channel symmetric int weights, per-tensor symmetric int activations, STE
# gradients — so training directly targets the hardware's behavior.
#
# Labels are NOT required: this uses knowledge distillation from the frozen
# FP32 ResNet-50 (teacher) to the quantized model (student), so ANY images
# work (e.g. sw/quant/calib). Full accuracy recovery needs a large image set
# and a GPU; on CPU this script runs as a smoke test proving the mechanism.
#
# Usage:
#   python qat_resnet50.py --data sw/quant/calib --epochs 3 --bits 4
#   python qat_resnet50.py --smoke            # tiny CPU run: wiring + loss check
#   python qat_resnet50.py --data ... --save qat_int4.pth   # then feed to export
# =============================================================================

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# ---------------------------------------------------------------------------
# fake-quant (straight-through estimator), matching the accelerator's integer
# quantization: symmetric, round-to-nearest, clamp to [qmin, qmax]
# ---------------------------------------------------------------------------
def _ste_round(x):
    return x + (torch.round(x) - x).detach()


def fake_quant(x, scale, qmin, qmax):
    q = torch.clamp(_ste_round(x / scale), qmin, qmax)
    return q * scale


class QConv2d(nn.Module):
    """Conv2d with per-output-channel fake-quant weights and per-tensor
    fake-quant inputs, mirroring magnet_top's integer datapath."""

    def __init__(self, conv, bits):
        super().__init__()
        self.conv = conv
        self.qmax = (1 << (bits - 1)) - 1
        self.qmin = -(1 << (bits - 1))
        # per-tensor input activation scale, calibrated then fine-tuned (LSQ)
        self.act_scale = nn.Parameter(torch.ones(1))
        self.act_calibrated = False

    def calibrate_act(self, x, pct=99.99):
        a = torch.quantile(x.abs().flatten().float(),
                           torch.tensor(pct / 100.0))
        self.act_scale.data.fill_(float(a) / self.qmax + 1e-8)
        self.act_calibrated = True

    def forward(self, x):
        if not self.act_calibrated:
            self.calibrate_act(x)
        xq = fake_quant(x, self.act_scale.abs(), self.qmin, self.qmax)
        w = self.conv.weight
        # per-output-channel symmetric weight scale from the current weights
        sw = w.abs().amax(dim=(1, 2, 3), keepdim=True) / self.qmax + 1e-8
        wq = fake_quant(w, sw, self.qmin, self.qmax)
        return F.conv2d(xq, wq, self.conv.bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation,
                        self.conv.groups)


def quantize_model(model, bits):
    """Replace every Conv2d (except the 3-channel stem, kept higher fidelity
    is optional — here we quantize all) with a QConv2d."""
    for name, m in model.named_children():
        if isinstance(m, nn.Conv2d):
            setattr(model, name, QConv2d(m, bits))
        else:
            quantize_model(m, bits)
    return model


# ---------------------------------------------------------------------------
# data: unlabeled images -> preprocessed tensors (no labels needed)
# ---------------------------------------------------------------------------
def load_batch(paths):
    from PIL import Image
    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    xs = []
    for p in paths:
        xs.append(tfm(Image.open(p).convert("RGB")))
    return torch.stack(xs)


def image_paths(d, limit=None):
    fs = [os.path.join(d, f) for f in sorted(os.listdir(d))
          if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return fs[:limit] if limit else fs


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="int4 QAT via distillation")
    ap.add_argument("--data", default=os.path.join(
        os.path.dirname(__file__), "calib"))
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=4.0, help="distill temp")
    ap.add_argument("--save", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU run: 16 images, 8 steps, wiring/loss check")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")

    w = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    teacher = torchvision.models.resnet50(weights=w).eval().to(dev)
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = torchvision.models.resnet50(weights=w)
    student = quantize_model(student, args.bits).to(dev)

    paths = image_paths(args.data, limit=16 if args.smoke else None)
    if not paths:
        raise SystemExit(f"no images in {args.data}")
    epochs = 1 if args.smoke else args.epochs
    max_steps = 8 if args.smoke else 10 ** 9
    print(f"{len(paths)} images, {epochs} epoch(s), batch {args.batch}, "
          f"int{args.bits} QAT via distillation")

    # calibration pass (sets activation scales) — one batch, no grad
    student.eval()
    with torch.no_grad():
        student(load_batch(paths[:args.batch]).to(dev))

    opt = torch.optim.Adam([p for p in student.parameters()
                            if p.requires_grad], lr=args.lr)
    student.train()
    step = 0
    for ep in range(epochs):
        order = np.random.permutation(len(paths))
        losses = []
        for i in range(0, len(paths) - args.batch + 1, args.batch):
            idx = order[i:i + args.batch]
            x = load_batch([paths[j] for j in idx]).to(dev)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            T = args.temp
            loss = F.kl_div(F.log_softmax(s_logits / T, 1),
                            F.softmax(t_logits / T, 1),
                            reduction="batchmean") * (T * T)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            step += 1
            if step >= max_steps:
                break
        print(f"epoch {ep + 1}: distill KL loss "
              f"{losses[0]:.4f} -> {losses[-1]:.4f} "
              f"(mean {np.mean(losses):.4f}, {len(losses)} steps)")
        if step >= max_steps:
            break

    # quick agreement check on the training images (teacher vs student top-1)
    student.eval()
    agree = 0
    with torch.no_grad():
        for i in range(0, len(paths), args.batch):
            x = load_batch(paths[i:i + args.batch]).to(dev)
            agree += int((teacher(x).argmax(1) == student(x).argmax(1)).sum())
    print(f"teacher/student top-1 agreement on {len(paths)} train images: "
          f"{agree}/{len(paths)}")

    if args.save:
        torch.save(student.state_dict(), args.save)
        print(f"saved QAT weights -> {args.save} "
              f"(load into resnet50 + export_resnet50.py to deploy)")


if __name__ == "__main__":
    main()
