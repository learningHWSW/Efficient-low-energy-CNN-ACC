#!/usr/bin/env python3
# =============================================================================
# fetch_imagenet_val.py — download the ImageNet-1k validation set (50k images,
# 1000 classes, ~7GB) via kagglehub, for label-free distillation QAT.
#
# Distillation needs no labels, only diverse images spanning the teacher's
# output distribution — the val set covers all 1000 classes, far more diverse
# than fruits262. Prints the extracted path (feed it to qat_resnet50.py --data).
#
# Usage: python fetch_imagenet_val.py
# =============================================================================

import os
import kagglehub

path = kagglehub.dataset_download("titericz/imagenet1k-val")
print("dataset path:", path)

n = 0
for root, _dirs, files in os.walk(path):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            n += 1
print(f"images: {n}")
print("feed to: qat_resnet50.py --data", path)
