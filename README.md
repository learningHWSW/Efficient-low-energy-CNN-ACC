# Efficient Low-Energy CNN Accelerator (MAGNet/Simba-style, AMD Vitis HLS)

An int8/int4 CNN inference accelerator for AMD Zynq UltraScale+ (ZCU104),
modeled on NVIDIA's **MAGNet** (ICCAD 2019) configurable PE template and
**Simba** (MICRO 2019) multi-chip system. Design references live in
`resources/` (MAGNet, Simba, Buffets, and Sze's *Efficient Processing of DNNs*).
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

The toolchain is closed **end to end on Windows**: PyTorch weights → PTQ export
→ HLS → csim/cosim → ZCU104 bitstream (timing clean). Only running on physical
hardware remains.

## Highlights

- **8-PE multi-PE system** (Simba chiplet structure): IA multicast loader +
  weight loader + 8×PE + cross-PE reduction (gather) as an HLS dataflow
  pipeline. **512 MAC/cycle = 204.8 GOPS @ 200 MHz** (int8); BRAM 56% / DSP 29%
  / LUT 27% on xczu7ev.
- **Runtime spatial mapping** KP×CP — K-split (8×1) / mixed (4×2) / C-split
  (1×8), chosen per layer by the mapper. Gather uses a flattened (q,kp) II=1
  loop, so resources stay flat as N_PES grows.
- **16-PE build fits the free-license ZCU104** — `hls_config_pe16_zu7ev.cfg`
  (row buffers → URAM + `PT_ROWS=4`): BRAM 85% / URAM 16% / DSP 52% / LUT 76%,
  all MAC loops II=1, no performance loss (~28.8 fps est.). Alternatives:
  xcku5p (free, no ARM PS) or zu9eg/ZCU102 (roomy, Enterprise license).
- **INT8 vs INT4** — int8 (`-DUSE_INT4` off) is essentially lossless with plain
  PTQ; int4 (`-DUSE_INT4`, 1024 MAC/cycle = 409.6 GOPS, *fewer* DSPs than int8)
  doubles throughput but trades accuracy and needs QAT. Real ImageNet-1k top-1
  on 8,910 strictly held-out images (all 1000 classes, training images excluded;
  `eval_imagenet.py`):

  | model | top-1 | top-5 | notes |
  |---|---|---|---|
  | FP32 | 75.9% | 93.2% | reference |
  | **int8 PTQ** | **76.1%** | 93.1% | lossless, no QAT needed |
  | int4 PTQ (all int4) | 14.3% | 30.2% | unusable |
  | int4 PTQ (int8 stem) | 42.0% | 66.7% | int8 stem is the biggest single lever |
  | int4 QAT (basic) | 44.0% | 70.3% | plain fake-quant, val-only distill |
  | **int4 QAT (LSQ + int8 stem)** | **56.2%** | 79.7% | learned scales + mixed precision |

  Path to int4: PTQ leaves it at top-1 0/16 (activation resolution is the
  bottleneck — `vsq_probe.py` shows per-vector weight scaling alone doesn't fix
  it). Label-free distillation QAT (`qat_resnet50.py`, fake-quant matched to the
  datapath, teacher = FP ResNet-50) plus **LSQ** (learned step sizes) and a
  **mixed-precision int8 stem** lifts int4 to 56.2% top-1 (80% top-5). Trained on
  the ImageNet-1k val set (`fetch_imagenet_val.py`, 40k images, 6 epochs) on an
  RTX 5060 (CUDA 12.8). Still below int8 because this is val-only distillation
  for a few epochs and keeps the final fc in FP; proper QAT on the 1.2M-image
  train set reaches ~74%.
- **Real-weight pipeline** — `sw/quant/export_resnet50.py` (pretrained ResNet-50
  → BN folding → per-channel PTQ → accelerator layout). The **full 224×224
  ResNet-50 (72 layers incl. fc)** runs through the kernels bit-exactly against
  an integer reference (logits 0/1000 mismatches). Quantization fidelity vs the
  float model on 16 real images: **top-1 14/16, float-top1 within int8 top-5
  16/16** (per-channel + percentile calibration).
- **Windows-native bitstream** — `hw/scripts/build_vivado.tcl` (classic Vivado
  flow, no XRT/Linux): HLS IP → ZCU104 block design → synth/impl → `.bit`+`.xsa`.
  Both configs close timing at 200 MHz on the ZCU104 (post-route): **8 PE — LUT
  27%, BRAM 56%, WNS +0.405 ns; 16 PE — LUT 42%, BRAM 68%, URAM 17%, DSP 52%,
  WNS +0.032 ns**, DRC 0 errors. Board bring-up via `sw/host/pynq_host.py`
  (PYNQ, drives the s_axilite register map directly).
- **Verified** at every level: bit-exact Python reference model (incl. stream
  deadlock checks), g++ testbenches, Vitis csim, csynth (all MAC loops II=1),
  and RTL cosim (9 runs pass).

## Repository layout

```
hw/
  include/accel_config.h    # design-time params (VECTOR_SIZE, N_LANES, N_PES, buffers, precision)
  include/accel_types.h     # ap_int/hls_stream types (fall back to std types off-tool)
  include/magnet_top.h      # conv kernel decl + DRAM layout contract
  include/global_pe.h       # Global PE kernel decl (eltwise/pool)
  src/magnet_top.cpp        # multi-PE conv system (loader / PE×N / gather dataflow)
  src/global_pe.cpp         # Global PE (residual add, maxpool, avgpool)
  src/archive/              # earlier-phase implementations (reference only)
  tb/tb_conv.cpp            # conv kernel TB (golden compare, 12 cases × 3 spatial)
  tb/tb_global_pe.cpp       # Global PE TB
  tb/tb_network.cpp         # network e2e TB (ResNet bottleneck block)
  tb/tb_resnet50.cpp        # full ResNet-50 dry run + reduced-size e2e
  tb/tb_resnet50_real.cpp   # real quantized weights through the kernels
  scripts/build_vivado.tcl  # classic Vivado flow -> bitstream (Windows)
  scripts/run_hls.ps1       # HLS build wrapper (csim/synth/cosim/gpe-*/int4-*/pe16-*/gcc/net/resnet/real)
  scripts/hls_config*.cfg   # Vitis unified-flow configs (per kernel/target)
sw/
  quant/export_resnet50.py  # pretrained ResNet-50 PTQ -> accelerator layout export
  quant/classify.py         # image -> ImageNet class (sim or FPGA backend)
  quant/qat_resnet50.py     # int4 quantization-aware training (distillation, no labels)
  quant/fetch_imagenet_val.py # download ImageNet-1k val (kagglehub) for QAT distillation
  quant/eval_imagenet.py    # real ImageNet top-1/top-5: FP vs int4 PTQ vs int4 QAT
  quant/vsq_probe.py        # per-vector-scale experiment (why int4 needs QAT)
  quant/fetch_calib_images.py # sample fruits262 calibration images via kagglehub
  mapper/mapper.py          # MAGNet Mapper (spatial/tile selection + cycle/traffic model)
  tuner/tuner.py            # design-space exploration (MAGNet Tuner-lite)
  model/reference_model.py  # bit-exact Python model (with stream-count checks)
  runtime/net_runner.h      # layer sequencer + plan_conv (C++ mirror of the mapper)
  host/pynq_host.py         # board driver (PYNQ, XRT-free)
  host/xrt_host.cpp         # board driver (XRT, Linux) — alternative
docs/ARCHITECTURE.md        # design doc (MAGNet/Simba mapping, dataflow, roadmap)
resources/                  # reference papers and Buffets RTL (git-ignored)
```

## Requirements

- **AMD Vivado/Vitis 2026.1**. From 2024.2 the classic `vitis_hls` binary is
  gone; the HLS scripts use the unified flow (`v++ --mode hls`, `vitis-run`).
  A free license is needed for csim/synth/cosim (see below); without it the
  `gcc` mode still runs the testbenches with the bundled g++ + real ap_int headers.
- Default target: **ZCU104** (`xczu7ev-ffvc1156-2-e`); change `part=` in the
  `hls_config*.cfg` files for other boards.
- Python 3.10+ for the mapper / reference model / quantization pipeline
  (PyTorch + torchvision only for `export_resnet50.py`).

### Free license activation (once)

Since 2025.1 the free Standard Edition also needs a license file:
1. Sign in at https://account.amd.com/en/licenses (AMD account).
2. Generate a "Vivado/Vitis Standard Edition" free license — the Host ID is the
   Ethernet MAC (`Get-NetAdapter`).
3. Save the `Xilinx.lic` to `C:\Users\<user>\.Xilinx\Xilinx.lic` and point
   `XILINXD_LICENSE_FILE` at it.

## Quick start

```powershell
# 1) Algorithm check (no Vitis/license needed)
python sw\model\reference_model.py

# 2) Mapping / performance estimates
python sw\mapper\mapper.py --network resnet50            # 8 PE int8
python sw\mapper\mapper.py --network resnet50 --pes 16 --int4

# 3) Testbenches via the bundled g++ + real ap_int headers (no license)
.\hw\scripts\run_hls.ps1 gcc        # conv kernel, 36 cases
.\hw\scripts\run_hls.ps1 net        # ResNet bottleneck-block e2e

# 4) HLS flow (after license activation)
.\hw\scripts\run_hls.ps1 csim       # C simulation
.\hw\scripts\run_hls.ps1 synth      # synthesis report (II=1, resources, timing)
.\hw\scripts\run_hls.ps1 cosim      # RTL co-simulation

# 5) Real quantized ResNet-50 through the kernels
python sw\quant\fetch_calib_images.py --n 48    # calibration images (optional)
python sw\quant\export_resnet50.py              # PTQ export -> sw/quant/export
.\hw\scripts\run_hls.ps1 real                   # run it through the kernels

# 6) Windows bitstream (after run_hls.ps1 synth produces the IP)
& 'C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat' -mode batch -source hw/scripts/build_vivado.tcl
```

(For legacy Vitis ≤2024.1, see `hw/scripts/run_hls.tcl`.)

## Using it: image in, class out

The user-facing entry point is `sw/quant/classify.py` — give it an image, it
preprocesses (resize/crop/normalize), quantizes, runs the quantized ResNet-50,
and prints the ImageNet class:

```powershell
python sw\quant\classify.py cat.jpg              # --sim: numpy integer pipeline on a PC
# 1. Egyptian cat        (logit ...)
# 2. tabby, tabby cat    (logit ...)
```

```bash
sudo python3 sw/quant/classify.py cat.jpg --fpga magnet_top.bit   # on the board
```

Both backends share the exact same preprocessing, quantization, and label
lookup. The `--sim` integer pipeline is **bit-identical to what the FPGA
computes** (verified: kernel-path logits match `--sim` 0/1000 on a real image),
so running `--sim` on a PC predicts the FPGA result — only the hardware MMIO is
board-only. End-to-end operation on the board:

```
image.jpg
  → [ARM] preprocess + int8 quantize (input scale from manifest.json)
  → [FPGA] conv layers (magnet_top driven via s_axilite register map)
  → [ARM] pooling / residual add (conv-only bitstream; numpy fallback)
  → [ARM] argmax + labels.txt → class name
```

Prerequisite: an export directory from `export_resnet50.py` (weights + manifest
+ `labels.txt`). Note int8 logits saturate at ±127, so several top classes can
tie at the cap — a larger-precision final layer would sharpen top-1 (a future
refinement); the ranking is otherwise correct.

## Design summary

Three-tier compute hierarchy from MAGNet: a **Vector MAC** (VECTOR_SIZE-wide
dot product) × **N_LANES** lanes forms one PE; N_PES PEs form the array. The
input-channel dimension (C) maps spatially to VectorSize and the output-channel
dimension (K) to NLanes. Temporally the PE uses the **OS-LWS** order the paper
reports as most energy-efficient: weights stay fixed across the innermost output
loop while partial sums are read-modify-written in the accumulation buffer. A
post-processing unit does bias + int8/int4 requantization + ReLU for layer
fusion, with per-output-channel requant multipliers for accuracy. On top of the
PE array sit the Simba-style system pieces: multicast IA loader, cross-PE
partial-sum reduction, a Global PE (residual add / pooling), and a host runtime
that sequences layers and picks each layer's spatial mapping.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Single-PE OS-LWS conv engine + verification infra | ✅ |
| 1 | Vitis csim/csynth/cosim, II=1 @ 200 MHz | ✅ |
| 2 | C-tiling, double buffering (dataflow PIPO), DSP packing | ✅ |
| 3a | Multi-PE + cross-PE reduction + KP×CP spatial mapping | ✅ |
| 3b | Global PE (eltwise/pool) + host runtime + network e2e | ✅ |
| 4a | 8-PE scale-up + gather redesign + XRT/link deliverables | ✅ |
| 4b | Windows-native ZCU104 bitstream + PYNQ host | ✅ (bring-up on real HW pending) |
| 4c | Multi-port IA loader + full ResNet-50 + Tuner-lite | ✅ |
| 5 | INT4 datapath + 16-PE build (incl. free-license ZCU104) | ✅ |
| 6 | Real-weight ResNet-50 pipeline + per-channel PTQ | ✅ |
| — | On-board execution; int4 fidelity (per-vector scale / QAT) | future |

**Constraints:** Q ≤ 128 (covers every ResNet-50 layer; VGG-224 would need
W-tiling); KP and CP are powers of two.
