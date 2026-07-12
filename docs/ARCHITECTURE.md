# Architecture — MAGNet/Simba-style CNN accelerator (Vitis HLS)

This document describes the design, which ports MAGNet's PE architectural
template to AMD Vitis HLS (FPGA) and adds Simba's system-level structure, based
on the NVIDIA papers in `resources/` (MAGNet, Simba, Buffets) and Sze's
*Efficient Processing of Deep Neural Networks*.

---

## 1. MAGNet mapping

MAGNet (ICCAD 2019) has three components:

| MAGNet component | Role | This project |
|---|---|---|
| **Designer** | C++/SystemC template → RTL via HLS | `hw/src/magnet_top.cpp` + `hw/include/accel_config.h` (compile-time params) |
| **Mapper** | Layer→hardware mapping (tiling, ordering) | `sw/mapper/mapper.py` (constraint check + cycle/traffic model) |
| **Tuner** | Bayesian design-space exploration | `sw/tuner/tuner.py` (exhaustive DSE) |

### Compute hierarchy (paper Fig. 2)

```
System ── N_PES PEs + cross-PE reduction / Global PE
  └─ PE ── N_LANES Vector MACs + W/IA/Accum buffers + PPU
       └─ Vector MAC ── VECTOR_SIZE-wide dot product
```

### Design-time parameters (paper Table I ↔ `accel_config.h`)

| Paper | Code | Default | Meaning |
|---|---|---|---|
| VectorSize | `VECTOR_SIZE` | 8 (16 for int4) | dot-product width = **spatial map of C (input channels)** |
| NLanes | `N_LANES` | 8 | vector MACs per PE = **spatial map of K (output channels)** |
| NPEs | `N_PES` | 8 (build switch) | PE array size |
| WPrecision | `W_BITS` | 8 / 4 | int8 or int4 weights |
| IAPrecision | `IA_BITS` | 8 / 4 | int8 or int4 activations |
| AccumPrecision | `ACC_BITS` | 32 | 32b is natural on FPGA (paper uses 24b) |
| WeightBufSize | `WBUF_DEPTH` | 256 | per-lane bank, 64b word |
| IABufSize | `IABUF_DEPTH` | 2048 | 64b word |
| AccumBufSize | `Q_MAX` × `PT_ROWS` | 128 × 8 | partial sums of a row group |

Default throughput: **512 MAC/cycle = 204.8 GOPS @ 200 MHz** (8 PEs, int8;
1024 MAC/cycle with int4 or 16 PEs).

---

## 2. Dataflow: OS-LWS (Output Stationary – Local Weight Stationary)

Following the paper's Fig. 12 result — **OS-LWS is most energy-efficient in
nearly every configuration** — OS-LWS is the default dataflow (paper Fig. 4(e)
loop nest):

```
for k1 = [0 : K1)          # output-channel tile (N_LANES each)  — temporal
  for p1 = [0 : P2)        # output-row group (PT_ROWS each)     — temporal
    for c2 = [0 : C2)      # input-channel tile (CT1 words)      — temporal (psum accumulate)
      for p0 = [0 : PT)    # row within group — load ‖ compute overlap (dataflow)
        for (r, s, c1)     # kernel position × channel word in tile — temporal
          for q0 = [0 : QB) # output column (innermost)          — temporal
            acc[p0][q0][lane] += W[k1,lane, r,s,c2·CT1+c1]
                                 · IA[p·st+r, q0·st+s, c2·CT1+c1]
            # lane ∈ N_LANES (spatial), · is a VECTOR_SIZE-wide dot (spatial)
```

- **LWS**: while `(r,s,c1)` is fixed the same weight word is reused across the
  whole q0 loop (QB times). This is the paper's weight-collector reuse; on FPGA,
  BRAM reads are cheap, so the same ordering is obtained by holding the address
  fixed, without a separate latch tier.
- **OS**: partial sums stay in `acc_buf[p0][q][lane]` and are read-modify-written
  R·S·C1 times across all c2 tiles. No psum round-trip to DRAM.
- **C-tiling** (Phase 2): if a filter is larger than the weight buffer, the input
  channels are split into CT1-word tiles (c2 loop) while psums stay resident in
  acc_buf. When C2>1 the weight tile is reloaded per (k1,p1,c2) (a tag cache
  loads it once per k1 when C2==1) — this is the price of psum-stationary. With
  the default buffers every ResNet-50 layer has C2=1; C-tiling activates when the
  buffers are shrunk for the multi-PE array.
- **Double buffering** (Phase 2): the p0 loop is an HLS dataflow region — the IA
  row buffer becomes a per-iteration PIPO (ping-pong) so load_ia(p0+1) overlaps
  compute(p0). For stride-1 layers the IA load is fully hidden behind compute.
- **DSP packing** (Phase 2): adjacent lane pairs share the same activation, so
  `(w1≪18 + w0)×a` in one 27×18 multiply yields two MACs (Xilinx WP487). The
  borrow-correction arithmetic is exhaustively verified over all 16.7M (w0,w1,a)
  combinations. DSP demand 64→32 per PE.
- The PPU requantizes each row group (bias + scale·round) with **per-output-channel
  multipliers** + ReLU + int8/int4 saturation.

### Pipeline safety (`MIN_QB`)

The acc_buf RMW revisits the same address every QB cycles. To guarantee II=1 the
revisit distance must exceed the pipeline depth, so QB is padded to
`max(Q, MIN_QB=16)` and excess results are discarded at store time. An HLS
`dependence distance=16` pragma communicates this to the scheduler.

---

## 3. Memory system

### On-chip buffers (paper Fig. 2(b))

| Buffer | Structure | Paper equivalent |
|---|---|---|
| `wbuf[WBUF_DEPTH][N_LANES]` | per-lane partition (BRAM banks), (k1,c2) tile cache | Weight buffer |
| `rowbuf[IABUF_DEPTH]` | dataflow PIPO (×2), R input rows incl. zero-pad; optionally in URAM | Input-activation buffer (buffet fill/read split) |
| `acc_buf[PT_ROWS][Q_MAX][N_LANES]` | per-lane partition, psum resident across the c2 loop | Accumulation buffer (RMW, buffet update) |
| `bias_reg` / `mult_reg` | registers | PPU bias + per-channel requant multipliers |

Of the Buffets fill/read/update semantics, **update** (RMW) is handled by
`acc_buf`, and the **fill→read split** by separating the load and compute stages;
explicit credit-based synchronization is approximated by the load/compute/store
overlap (double buffering).

### DRAM layout (host-prepared, word = 64b = 8×int8 / 16×int4)

| Array | Layout | Padding |
|---|---|---|
| IA | `[H][W][C1]` (NHWC) | C zero-padded to a `VECTOR_SIZE` multiple |
| W | `[Kpad][R][S][C1]` (KRSC) | K zero-padded to an `N_LANES` multiple |
| OA | `[P][Q][K1]`, word = N_LANES elements | lanes ≥ K ignored |
| bias, mult | `int32 [Kpad]` | k ≥ K is 0 |

### Mapping constraints (checked by mapper.py)

```
R·S·CT1               ≤ WBUF_DEPTH
R·win_cols·CT1        ≤ IABUF_DEPTH     # win_cols = (QB-1)·stride + S
Q                     ≤ Q_MAX
```
Every ResNet-50 conv layer fits the default configuration
(`python sw/mapper/mapper.py --network resnet50`).

---

## 3.5 Multi-PE system (Phase 3a — Simba chiplet structure)

The Simba (MICRO 2019) chiplet interior is implemented as four HLS dataflow tasks:

```
ia_loader ──ia_st[pe]──▶ ┌─────┐
(multicast / C-slice)    │ PE0 │─psum_st[0]─┐
w_loader  ──w_st[pe]───▶ │ ... │            ├──▶ gather_store
(K/C tile distribution)  │ PEn │─psum_st[n]─┘    (cross-PE reduction
                         └─────┘                  + bias + PPU + store)
```

- **Spatial mapping** (runtime KP×CP=N_PES, Simba Listing 2's parallel_for):
  - `KP` (K-split): PE groups own different k1 tiles. IA is **multicast**
    (one DRAM read → KP streams) — IA traffic ÷KP vs a single PE.
  - `CP` (C-split): PEs within a group share input-channel slices; their psums
    are summed in gather = **Simba's cross-PE partial-sum reduction**. Bias is
    added once, after the reduction.
  - The mapper picks the fewest-cycle mapping per layer.
- **Lockstep contract**: every task derives its loop bounds from the same
  formulas (`derive()`), so stream production/consumption counts match by
  construction (reference_model.py asserts on stream drain). k1≥K1 (KP padding)
  sends zero weights; p≥P (row-group tail) is consumed and discarded.
- **Constraints**: C1 is host-padded to a CP multiple; CT1 ≤ C1/CP.
- **Multi-port IA loader** (Phase 4c): N_IA_PORTS (4) AXI masters all pointing at
  the same DDR buffer read the C-slices in parallel — loader time scales as
  ceil(CP/4) instead of CP. One read per port per iteration plus exactly one
  static write per PE stream keeps II=1 (multiple static writers to a stream
  cause FIFO port conflicts that raise II). This makes the mapper favor C-split
  (2×4, 1×8).

## 3.6 Global PE and host runtime (Phase 3b)

Maps to Simba's Global PE (second-level storage + near-memory low-reuse ops) and
its RISC-V controller:

- **`global_pe_top`** (hw/src/global_pe.cpp, a separate HLS component):
  - `GPE_ELTWISE`: O = sat(relu((A·multA + B·multB) ≫ shift)) — ResNet residual
    add with per-input requantization scales, II=1 streaming.
  - `GPE_MAXPOOL` / `GPE_AVGPOOL`: window walk (II=1), padding ignored. For avg
    the 1/count folds into multA/shift (including global average pool).
  - Same NHWC 64b-word layout as the conv kernel → **chaining without repacking**.
- **`sw/runtime/net_runner.h`**: the RISC-V controller role — layer sequencer +
  kernel configuration. `plan_conv()` mirrors the mapper's cycle model to pick
  KP/CP/CT1 per layer (chaining constraint: the tensor's C1 is fixed by the
  previous layer, so C1 % CP != 0 candidates are excluded). It calls the kernels
  as C functions for csim; on the board the call sites become PYNQ MMIO or XRT
  enqueues.
- **Verification**: tb_global_pe (7 unit cases) + tb_network (ResNet
  bottleneck-block e2e: conv×3 → residual add → maxpool → global avgpool,
  bit-exact at every stage).

## 4. Execution flow

```
k1 tile loop:
  load bias + per-channel mult : N_LANES each
  p1 (row group) loop:
    c2 (channel tile) loop:
      load_weights : burst the (k1,c2) tile into per-lane banks (tag cache)
      p0 loop [HLS dataflow]:
        load_ia_rows ‖ compute_row   # PIPO overlap, II=1
    ppu_store_group: requant→ReLU→pack→DRAM (whole row group)
```

Residual inefficiencies (targeted by later phases):
1. **IA row reload** — adjacent output rows share (R−stride) input rows but reload
   them; the dataflow overlap hides the latency but DRAM bandwidth/energy is ×R.
   Multi-PE multicast (Global PE) absorbs most of it.
2. **PT_ROWS/Q<16 waste** — tail rows and short-Q padding lower utilization on
   small output layers; K/C spatial splitting (multi-PE) is the fundamental fix.

---

## 5. Roadmap

| Phase | Scope | Status |
|---|---|---|
| **0** | Single-PE OS-LWS conv engine + golden TB + mapper | ✅ |
| **1** | Vitis csim/csynth pass (II=1, 200 MHz, packing) | ✅ |
| **2** | C-tiling, double buffering, DSP packing (2 MAC/DSP) | ✅ |
| **3a** | Multi-PE dataflow + cross-PE reduction + KP×CP mapping + Simba-scale buffers | ✅ (csim 36 cases) |
| **3b** | Global PE (eltwise/pool) + net_runner runtime + network e2e | ✅ (csim/csynth clean) |
| **—** | RTL cosim (reduced TB, 9 runs) | ✅ PASS (no deadlock) |
| **4a** | 8-PE scale-up (512 MAC/cycle) + gather (q,kp) flatten redesign + XRT/link deliverables | ✅ |
| **4c** | Multi-port IA loader + full ResNet-50 (53-conv dry run + reduced e2e) + Tuner-lite | ✅ |
| **5** | INT4 datapath (-DUSE_INT4) + 16-PE build (incl. free-license ZCU104) | ✅ |
| **4b** | Windows-native ZCU104 bitstream (classic Vivado flow) + PYNQ host | ✅ (.bit generated, timing clean; on-board bring-up pending) |
| **6** | Real-weight ResNet-50 pipeline + per-channel PTQ | ✅ |
| **—** | On-board execution; Q>128 (W-tiling); int4 per-vector scale / QAT | future |

### INT4 configuration (Phase 5)

MAGNet's headline feature (better perf/area at 4-bit) as a compile switch:
- `-DUSE_INT4`: W_BITS/IA_BITS/OA_BITS=4, **VECTOR_SIZE 8→16**. The 64b word
  carries 16 elements, doubling per-PE throughput (128 MAC/cycle) →
  **1024 MAC/cycle = 409.6 GOPS @ 200 MHz** for 8 PEs.
- **DSP packing still holds at 4-bit**: |w0·a| ≤ 8·8 = 64 ≪ 2^17, so the
  borrow-correction proof is unchanged; only VECTOR_SIZE grows (DOT2 runs 16×).
  Measured DSP is actually *lower* than int8 (small multiplies map to fabric).
- Saturation bounds (OA_MAX/MIN) and TB value ranges (IA_RMAX, …) are derived
  from the precision macros.
- **Layer chaining**: at int4 the OA word (8×4b=32b) differs from the IA word
  (16×4b=64b), but the byte stream is identical — net_runner converts word
  counts by `VECTOR_SIZE·IA_BITS / (N_LANES·OA_BITS)` (×1 int8, ×2 int4). K must
  be a VECTOR_SIZE multiple.
- Mapper estimate for ResNet-50: int8 20.3 ms → **int4 11.8 ms** (unique shapes,
  once each).

### 16-PE configuration (`-DN_PES=16`)

Functionally verified (reference model 24/24, g++ int8/int4 36 cases each) and
synthesized. Three device options (synthesis-measured):

| Target | License | Condition | BRAM/URAM/DSP/LUT | Notes |
|---|---|---|---|---|
| **zu7ev (ZCU104)** | **free** | `USE_URAM_ROWBUF` + `PT_ROWS=4` (`hls_config_pe16_zu7ev.cfg`) | **85% / 16% / 52% / 76%** | no perf loss (mapper 34.8 ms); LUT 76% may need P&R timing effort |
| xcku5p | free | default config | 82% / 0 / 50% / 81% | no ARM PS → different deployment; boards rare |
| zu9eg (ZCU102) | Enterprise | default (`hls_config_pe16.cfg`) | 43% / 0 / 36% / 63% | most headroom |

BRAM savings: rowbuf→URAM (−128 BRAM18), PT_ROWS=4 (acc banks BRAM36→BRAM18,
−128). `mapper.py --pes 16 [--int4]`. Estimated full-network throughput: 16 PE
int8 ~28.8 fps; combine with int4 for 2048 MAC/cycle.

### Windows-native deployment flow (Phase 4b, no XRT)

The Vitis acceleration flow (`v++ -l` + XRT) is Linux-only, but the **classic
Vivado flow completes on Windows**:
1. HLS csynth → kernel IP (`build/hls/.../impl/ip`, `run_hls.ps1 synth`).
2. `hw/scripts/build_vivado.tcl`: ZCU104 board-preset PS + kernel + AXI
   SmartConnect (7 data masters → HP0, control → HPM0) block design → synth →
   impl → `.bit` + `.xsa`. Set `VIV_BD_ONLY=1` to validate the block design
   quickly first.
3. On the board, drive the s_axilite register map directly with **PYNQ**
   (`sw/host/pynq_host.py`) instead of XRT (register offsets from the csynth
   report). The default bitstream is conv-only, so pooling/eltwise fall back to
   the ARM in numpy (hybrid); add global_pe_top to KERNELS to offload them too.

`build_vivado.tcl` takes env overrides — `MAGNET_IP_DIR` (kernel IP dir),
`MAGNET_FREQ` (PL clock MHz), `MAGNET_OUT` (output subdir) — so the same
script builds the 8-PE or 16-PE bitstream. Both close timing at 200 MHz on the
ZCU104 (place & route measured):

| Config | LUT | BRAM | URAM | DSP | Post-route WNS |
|---|---|---|---|---|---|
| 8 PE (build/hls) | 26.8% | 55.6% | 0% | 28.7% | +0.405 ns |
| 16 PE (build/hls_zu7final, URAM+PT4) | 42.0% | 68.4% | 16.7% | 51.9% | +0.032 ns |

Both: DRC 0 errors, .bit 19 MB. The 16-PE post-route LUT (42%) is far below the
csynth estimate (76%) — Vivado packs the fabric int8-packing logic much better
than HLS estimates. WNS +0.032 ns is tight but met.

### References (resources/)
- MAGNet: A Modular Accelerator Generator for Neural Networks — ICCAD 2019
- Simba: Scaling DL Inference with MCM-Based Architecture — MICRO 2019 (system level)
- Buffets: An Efficient and Composable Storage Idiom — ASPLOS 2019 (+ `buffets-master/` RTL)
- Sze et al., Efficient Processing of Deep Neural Networks (dataflow taxonomy)
