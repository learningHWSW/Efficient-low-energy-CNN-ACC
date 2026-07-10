// =============================================================================
// accel_config.h — MAGNet-style accelerator design-time parameters
//
// Mapping to MAGNet Table I (design-time parameters):
//   VectorSize      -> VECTOR_SIZE   (vector MAC dot-product width, C-dim unroll)
//   NLanes          -> N_LANES       (vector MACs per PE, K-dim unroll)
//   WPrecision      -> W_BITS
//   IAPrecision     -> IA_BITS
//   AccumPrecision  -> ACC_BITS
//   WeightBufSize   -> WBUF_DEPTH    (word = W_BITS*VECTOR_SIZE per lane)
//   IABufSize       -> IABUF_DEPTH   (word = IA_BITS*VECTOR_SIZE)
//   AccumBufSize    -> Q_MAX         (acc line: Q_MAX x N_LANES x ACC_BITS)
//   NPEs            -> N_PES         (multi-PE system since Phase 3a)
//
// Powers of two recommended for all values. When changing them, update the
// mapper's HwConfig defaults as well.
// =============================================================================
#pragma once

// ---- Compute hierarchy ------------------------------------------------------
#define VECTOR_SIZE   8    // dot-product width  (input-channel unroll)
#define N_LANES       8    // vector MACs per PE (output-channel unroll)
#define N_PES         8    // PE array (Phase 4). Runtime spatial mapping KP*CP = N_PES.
                           // 8 fits zu7ev/ZCU104 (covered by the free license tier).
                           // 16 (Simba chiplet class) needs zu9eg/ZCU102 or larger —
                           // the algorithm is verified up to 16 PEs
                           // (reference_model with N_PES=16).

// MACs/cycle = VECTOR_SIZE * N_LANES * N_PES = 512  (@200MHz -> 204.8 GOPS)

// ---- Precision --------------------------------------------------------------
#define W_BITS        8    // weight precision (int8)
#define IA_BITS       8    // input-activation precision (int8)
#define OA_BITS       8    // output-activation precision (int8)
#define ACC_BITS      32   // accumulation precision (paper uses 24b; 32b is free on FPGA)

// ---- On-chip buffer sizing (per PE) -----------------------------------------
// Shrunk to Simba scale in Phase 3a (BRAM budget of the multi-PE array).
// Larger layers are handled by C-tiling (CT1) — the mapper picks CT1.
// Weight buffer  : N_LANES banks, each word = VECTOR_SIZE x W_BITS = 64b
//                  constraint: R*S*CT1 <= WBUF_DEPTH
//                  (256 keeps 8 PEs within zu7ev BRAM; CT1<=28 for 3x3 kernels)
#define WBUF_DEPTH    256

// IA row buffer  : word = VECTOR_SIZE x IA_BITS = 64b
//                  constraint: R * win_cols * CT1 <= IABUF_DEPTH
//                  (win_cols = (QB-1)*stride + S, QB = max(Q, MIN_QB))
#define IABUF_DEPTH   2048

// Accumulation buffer: holds partial sums of one output-row group (OS-LWS).
// 128 covers every ResNet-50 layer (max Q=112) within the 8-PE BRAM budget.
// Layers with Q>128 (e.g. VGG at 224) would need W-direction tiling
// (not implemented — the mapper flags them).
#define Q_MAX         128

// Number of output rows kept in the accumulation buffer (row-group size).
// With C-tiling (C2>1) psums stay resident here across the c2 loop.
// If P is not a multiple of PT_ROWS, the tail rows of the last group are
// computed and then discarded.
#define PT_ROWS       8

// Maximum output channels (bias buffer sizing)
#define K_MAX         2048

// ---- IA loader ports --------------------------------------------------------
// Number of read ports of the IA loader (all point at the same DDR buffer;
// the host passes the same pointer N_IA_PORTS times). With C-split (CP>1)
// the loader reads up to N_IA_PORTS slices per cycle, removing the
// single-port bottleneck: loader time scales with ceil(CP/N_IA_PORTS).
#define N_IA_PORTS    4

// ---- Pipeline safety --------------------------------------------------------
// Minimum revisit distance of the acc_buf read-modify-write. Layers with
// Q < MIN_QB are padded to QB = MIN_QB and the excess results are discarded
// (handled automatically in hardware).
#define MIN_QB        16

// ---- Simulation options -----------------------------------------------------
// Depth of the m_axi interfaces (the cosim wrapper copies this many elements
// per buffer). COSIM_SMALL builds shrink it to match the reduced TB — the TB
// must allocate buffers of at least GMEM_DEPTH elements (otherwise the cosim
// wrapper SIGSEGVs).
#ifdef COSIM_SMALL
#define GMEM_DEPTH 4096
#else
#define GMEM_DEPTH 1048576
#endif

// ---- Datapath options -------------------------------------------------------
// 1 packs two int8 MACs into one 27x18 DSP multiply (Xilinx WP487 technique).
// Two adjacent lanes share the same activation: compute (w1<<18 + w0)*a and
// split the products with borrow correction. The arithmetic is exhaustively
// verified over all 16.7M (w0,w1,a) combinations. N_LANES must be even.
// Set to 0 to fall back to plain multiplies when debugging.
#define USE_DSP_PACKING 1
