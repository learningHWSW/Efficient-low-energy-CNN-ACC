Drop a handful of representative .jpg/.png images here (any natural photos,
ideally ImageNet-like). export_resnet50.py uses them to calibrate the
activation scales; without them it falls back to synthetic inputs, which
produces a working export but unrepresentative scales (poor accuracy).

10-30 images are typically enough for per-tensor PTQ calibration.
