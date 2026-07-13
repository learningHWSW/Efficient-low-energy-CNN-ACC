#!/usr/bin/env python3
# =============================================================================
# eval_imagenet.py — real ImageNet-1k top-1/top-5 for FP vs int4 PTQ vs int4 QAT
#
# The kagglehub imagenet1k-val set stores images in per-class WNID subdirs
# (n01440764, ...). Sorted WNIDs map 1:1 to the torchvision class index
# (n01440764 = tench = 0), so the folder gives the ground-truth label.
#
# Reports three models on the same images:
#   FP32           — torchvision pretrained ResNet-50
#   int4 PTQ       — fake-quant model, activation-calibrated, NO training
#   int4 QAT       — fake-quant model with weights from qat_resnet50.py --save
#
# Usage:
#   python eval_imagenet.py --data <val_dir> --qat build/qat_int4_imagenet.pth
#   # add --classes 600 999  to eval only classes 600..999 (strictly held out
#   # from a --limit 30000 sorted-first QAT run)
# =============================================================================

import argparse
import os
import sys

import numpy as np
import torch
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qat_resnet50 import quantize_model, ImageSet, image_paths


def find_val_root(d):
    # descend to the dir that directly contains the nXXXXXXXX class folders
    for root, dirs, _files in os.walk(d):
        if sum(1 for x in dirs if x.startswith("n") and x[1:].isdigit()) >= 100:
            return root
    return d


def build_eval_set(root, per_class, cls_lo, cls_hi, seed, exclude=None):
    exclude = exclude or set()
    wnids = sorted(x for x in os.listdir(root)
                   if x.startswith("n") and x[1:].isdigit())
    assert len(wnids) == 1000, f"expected 1000 classes, got {len(wnids)}"
    rng = np.random.default_rng(seed)
    paths, labels = [], []
    for idx, wn in enumerate(wnids):
        if not (cls_lo <= idx <= cls_hi):
            continue
        fs = [f for f in sorted(os.listdir(os.path.join(root, wn)))
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        fs = [f for f in fs
              if os.path.join(root, wn, f) not in exclude]  # held-out only
        if not fs:
            continue
        pick = rng.choice(len(fs), min(per_class, len(fs)), replace=False)
        for j in pick:
            paths.append(os.path.join(root, wn, fs[j]))
            labels.append(idx)
    return paths, np.array(labels)


@torch.no_grad()
def top1_top5(model, paths, labels, dev, batch):
    model.eval()
    loader = torch.utils.data.DataLoader(
        ImageSet(paths), batch_size=batch, num_workers=4,
        pin_memory=(dev.type == "cuda"))
    top1 = top5 = seen = 0
    for i, x in enumerate(loader):
        x = x.to(dev, non_blocking=True)
        logits = model(x)
        t5 = logits.topk(5, 1).indices.cpu().numpy()
        y = labels[seen:seen + len(x)]
        top1 += int((t5[:, 0] == y).sum())
        top5 += int(sum(y[j] in t5[j] for j in range(len(y))))
        seen += len(x)
    return top1 / seen, top5 / seen, seen


def calibrate(model, paths, dev, batch, n=512):
    """One pass to set the QConv activation scales (int4 PTQ baseline)."""
    model.eval()
    loader = torch.utils.data.DataLoader(
        ImageSet(paths[:n]), batch_size=batch, num_workers=4)
    with torch.no_grad():
        for x in loader:
            model(x.to(dev))
            break  # first batch triggers per-layer calibration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--qat", default=None, help="QAT state_dict (.pth)")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--classes", type=int, nargs=2, default=[0, 999],
                    metavar=("LO", "HI"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stem-bits", type=int, default=None,
                    help="stem-conv precision to match a mixed-precision QAT")
    ap.add_argument("--exclude-train-limit", type=int, default=None,
                    help="reconstruct qat train set (image_paths limit/seed 0) "
                         "and exclude it -> strictly held-out eval")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = find_val_root(args.data)
    exclude = None
    if args.exclude_train_limit:
        exclude = set(image_paths(args.data, limit=args.exclude_train_limit,
                                  seed=0))
        print(f"excluding {len(exclude)} training images from eval")
    paths, labels = build_eval_set(root, args.per_class, args.classes[0],
                                   args.classes[1], args.seed, exclude)
    print(f"eval set: {len(paths)} images, classes {args.classes[0]}.."
          f"{args.classes[1]}, device {dev}")

    w = torchvision.models.ResNet50_Weights.IMAGENET1K_V1

    fp = torchvision.models.resnet50(weights=w).to(dev)
    a1, a5, _ = top1_top5(fp, paths, labels, dev, args.batch)
    print(f"FP32       top-1 {a1*100:5.1f}%  top-5 {a5*100:5.1f}%")

    ptq = quantize_model(torchvision.models.resnet50(weights=w),
                         args.bits, args.stem_bits).to(dev)
    calibrate(ptq, paths, dev, args.batch)
    a1, a5, _ = top1_top5(ptq, paths, labels, dev, args.batch)
    print(f"int{args.bits} PTQ   top-1 {a1*100:5.1f}%  top-5 {a5*100:5.1f}%")

    if args.qat and os.path.exists(args.qat):
        qat = quantize_model(torchvision.models.resnet50(weights=w),
                             args.bits, args.stem_bits).to(dev)
        sd = torch.load(args.qat, map_location=dev, weights_only=True)
        qat.load_state_dict(sd)
        a1, a5, _ = top1_top5(qat, paths, labels, dev, args.batch)
        print(f"int{args.bits} QAT   top-1 {a1*100:5.1f}%  top-5 {a5*100:5.1f}%"
              f"   (fc/biases in FP)")


if __name__ == "__main__":
    main()
