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
import math
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
def _round_ste(x):
    return (torch.round(x) - x).detach() + x


def _grad_scale(x, g):
    # pass-through value, scale the gradient by g (LSQ, Esser et al. 2020)
    return (x - x * g).detach() + x * g


def lsq_quant(v, scale, qmin, qmax, n):
    """Learned-step-size quantization: `scale` is a learnable parameter, its
    gradient scaled by 1/sqrt(n*qmax) for stable low-bit training."""
    s = scale.abs() + 1e-8
    s = _grad_scale(s, 1.0 / math.sqrt(max(1, n) * qmax))
    q = _round_ste(torch.clamp(v / s, qmin, qmax))
    return q * s


class QConv2d(nn.Module):
    """Conv2d with LSQ fake-quant: learnable per-output-channel weight scale
    and a learnable per-tensor activation scale, mirroring the accelerator's
    integer datapath. Per-instance `bits` allows mixed precision (int8 stem)."""

    def __init__(self, conv, bits):
        super().__init__()
        self.conv = conv
        self.bits = bits
        self.qmax = (1 << (bits - 1)) - 1
        self.qmin = -(1 << (bits - 1))
        K = conv.weight.shape[0]
        # weight scale (per output channel), init from max — learnable (LSQ)
        sw = conv.weight.detach().abs().amax(dim=(1, 2, 3), keepdim=True)
        self.w_scale = nn.Parameter(sw / self.qmax + 1e-8)
        self.a_scale = nn.Parameter(torch.ones(1))
        self.a_ready = False

    def forward(self, x):
        if not self.a_ready:                        # one-shot activation calib
            v = x.detach().abs().flatten().float()
            if v.numel() > 200000:
                v = v[torch.randint(0, v.numel(), (200000,), device=v.device)]
            a = torch.quantile(v, torch.tensor(0.9999, device=v.device))
            self.a_scale.data.fill_(float(a) / self.qmax + 1e-8)
            self.a_ready = True
        xq = lsq_quant(x, self.a_scale, self.qmin, self.qmax, x[0].numel())
        w = self.conv.weight
        wq = lsq_quant(w, self.w_scale, self.qmin, self.qmax,
                       w[0].numel())
        return F.conv2d(xq, wq, self.conv.bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation,
                        self.conv.groups)


def quantize_model(model, bits, stem_bits=None):
    """Wrap every Conv2d in a QConv2d at `bits`. The stem conv (in_channels==3)
    uses `stem_bits` if given (int8 stem is standard for usable int4); the
    final fc (Linear) is left in FP. Both sensitive layers can run at higher
    precision on the ARM / a small int8 pass at deploy time."""
    for name, m in model.named_children():
        if isinstance(m, nn.Conv2d):
            b = stem_bits if (stem_bits and m.in_channels == 3) else bits
            setattr(model, name, QConv2d(m, b))
        else:
            quantize_model(m, bits, stem_bits)
    return model


# ---------------------------------------------------------------------------
# data: unlabeled images -> preprocessed tensors (no labels needed)
# ---------------------------------------------------------------------------
from torchvision import transforms as _T

_TFM = _T.Compose([
    _T.Resize(256), _T.CenterCrop(224), _T.ToTensor(),
    _T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class ImageSet(torch.utils.data.Dataset):
    """Unlabeled image dataset (module-level for Windows DataLoader workers)."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        return _TFM(Image.open(self.paths[i]).convert("RGB"))


def load_batch(paths):
    return torch.stack([ImageSet(paths)[i] for i in range(len(paths))])


def image_paths(d, limit=None, seed=0):
    # recurse (datasets store images in per-class subdirs)
    fs = []
    for root, _dirs, files in os.walk(d):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                fs.append(os.path.join(root, f))
    fs.sort()
    if limit and len(fs) > limit:
        # RANDOM subsample across the whole (class-sorted) list so every class
        # is represented — a strided/prefix cut would miss whole classes
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(fs), limit, replace=False))
        fs = [fs[i] for i in idx]
    return fs


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
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of training images (subsampled evenly)")
    ap.add_argument("--eval-every", type=int, default=1,
                    help="report teacher/student agreement every N epochs")
    ap.add_argument("--stem-bits", type=int, default=None,
                    help="precision for the 3-channel stem conv (e.g. 8); "
                         "int8 stem is standard for usable int4")
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
    student = quantize_model(student, args.bits, args.stem_bits).to(dev)
    if args.stem_bits:
        print(f"mixed precision: stem conv int{args.stem_bits}, "
              f"rest int{args.bits}, fc FP")

    all_paths = image_paths(args.data, limit=16 if args.smoke else args.limit)
    if len(all_paths) < 8:
        raise SystemExit(f"need >=8 images in {args.data}, found {len(all_paths)}")
    # shuffle before splitting so train/eval both span classes (images are
    # stored in per-class subdirs; a sorted tail would be one class)
    np.random.default_rng(0).shuffle(all_paths)
    n_eval = min(256, max(4, len(all_paths) // 10))
    eval_paths = all_paths[-n_eval:]
    paths = all_paths[:-n_eval]
    epochs = 1 if args.smoke else args.epochs
    max_steps = 8 if args.smoke else 10 ** 9
    print(f"{len(paths)} train + {len(eval_paths)} eval images, "
          f"{epochs} epoch(s), batch {args.batch}, int{args.bits} QAT distill")

    use_cuda = dev.type == "cuda"
    loader = torch.utils.data.DataLoader(
        ImageSet(paths), batch_size=args.batch, shuffle=True,
        num_workers=4 if use_cuda else 0, pin_memory=use_cuda, drop_last=True)

    def eval_agree():
        student.eval()
        agree = 0
        with torch.no_grad():
            for i in range(0, len(eval_paths), args.batch):
                x = load_batch(eval_paths[i:i + args.batch]).to(dev)
                agree += int((teacher(x).argmax(1) ==
                              student(x).argmax(1)).sum())
        student.train()
        return agree

    # calibration pass (sets activation scales) — one batch, no grad
    student.eval()
    with torch.no_grad():
        student(load_batch(paths[:args.batch]).to(dev))
    print(f"before training: eval agreement {eval_agree()}/{len(eval_paths)}")

    opt = torch.optim.Adam([p for p in student.parameters()
                            if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs * len(loader)))
    student.train()
    T = args.temp
    step = 0
    for ep in range(epochs):
        losses = []
        for x in loader:
            x = x.to(dev, non_blocking=use_cuda)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            loss = F.kl_div(F.log_softmax(s_logits / T, 1),
                            F.softmax(t_logits / T, 1),
                            reduction="batchmean") * (T * T)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step >= max_steps:
                break
        ea = eval_agree()
        print(f"epoch {ep + 1}: KL loss mean {np.mean(losses):.4f} "
              f"(last {losses[-1]:.4f}, {len(losses)} steps) | "
              f"eval agreement {ea}/{len(eval_paths)}")
        if args.save:
            torch.save(student.state_dict(), args.save)
        if step >= max_steps:
            break

    if args.save:
        print(f"saved QAT weights -> {args.save} "
              f"(load into resnet50 + export_resnet50.py to deploy)")


if __name__ == "__main__":
    main()
