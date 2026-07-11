#!/usr/bin/env python3
# =============================================================================
# fetch_calib_images.py — download the fruits262 dataset via kagglehub and
# sample a deterministic subset into sw/quant/calib/ for PTQ calibration.
#
# Usage: python fetch_calib_images.py [--n 48]
# =============================================================================

import argparse
import os
import shutil

import kagglehub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48, help="images to sample")
    args = ap.parse_args()

    path = kagglehub.dataset_download("aelchimminut/fruits262")
    print("dataset:", path)

    # collect all image paths (class subdirectories)
    imgs = []
    for root, _dirs, files in os.walk(path):
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                imgs.append(os.path.join(root, fn))
    imgs.sort()
    print(f"found {len(imgs)} images")
    if not imgs:
        raise SystemExit("no images found in the dataset")

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib")
    os.makedirs(outdir, exist_ok=True)
    # evenly spaced deterministic sample across the (class-sorted) list
    step = max(1, len(imgs) // args.n)
    picked = imgs[::step][:args.n]
    for i, src in enumerate(picked):
        cls = os.path.basename(os.path.dirname(src))
        dst = os.path.join(outdir, f"{i:03d}_{cls}.jpg")
        shutil.copyfile(src, dst)
    print(f"copied {len(picked)} images -> {outdir}")


if __name__ == "__main__":
    main()
