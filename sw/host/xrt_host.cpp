// =============================================================================
// xrt_host.cpp — XRT host for the board (Phase 4 deliverable; build/run on
// the board)
//
// NOTE: this file is built with XRT on the board's Linux (aarch64,
//   ZCU104/KV260). It has NOT been execution-verified yet — no board or
//   platform is available on this PC.
//
// Build (on the board):
//   g++ -std=c++17 xrt_host.cpp -o magnet_host \
//       -I$XILINX_XRT/include -L$XILINX_XRT/lib -lxrt_coreutil
//
// Run:
//   ./magnet_host magnet.xclbin
//
// Structure: uses the same configuration computation as the plan_conv /
// sequencer logic of sw/runtime/net_runner.h, but enqueues XRT runs instead
// of calling the kernel C functions. Follows the verified net_runner data
// layout contract (NHWC 64b words; conv output = next input).
// =============================================================================

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Hardware configuration (must match hw/include/accel_config.h)
static constexpr int VECTOR_SIZE = 8;
static constexpr int N_LANES = 8;
static constexpr int N_PES = 8;
static constexpr int WBUF_DEPTH = 256;
static constexpr int IABUF_DEPTH = 2048;
static constexpr int Q_MAX = 128;
static constexpr int MIN_QB = 16;
static constexpr int PT_ROWS = 8;
static constexpr int N_IA_PORTS = 4;

// ---------------------------------------------------------------------------
// plan_conv — mirror of sw/runtime/net_runner.h (automatic KP/CP/CT1 choice)
// ---------------------------------------------------------------------------
struct ConvPlan { int KP = N_PES, CP = 1, CT1 = 1; bool valid = false; long long cyc = 0; };

static ConvPlan eval_mapping(int C1, int K1, int P, int Q, int R, int S,
                             int stride, int KP, int CP) {
    ConvPlan pl; pl.KP = KP; pl.CP = CP;
    if (C1 % CP != 0) return pl;
    const int rs = R * S;
    const int QB = (Q < MIN_QB) ? MIN_QB : Q;
    const int win_cols = (QB - 1) * stride + S;
    const int c1s = C1 / CP;
    int ct1 = c1s;
    if (ct1 > WBUF_DEPTH / rs) ct1 = WBUF_DEPTH / rs;
    if (ct1 > IABUF_DEPTH / (R * win_cols)) ct1 = IABUF_DEPTH / (R * win_cols);
    if (ct1 < 1 || Q > Q_MAX) return pl;
    pl.CT1 = ct1; pl.valid = true;
    const int C2 = (c1s + ct1 - 1) / ct1;
    const int P2 = (P + PT_ROWS - 1) / PT_ROWS;
    const int K1G = (K1 + KP - 1) / KP;
    long long blk = (long long)PT_ROWS * rs * c1s * QB;
    const int cp_beats = (CP + N_IA_PORTS - 1) / N_IA_PORTS;
    const long long bl = (long long)PT_ROWS * R * win_cols * c1s * cp_beats;
    const long long bg = (long long)PT_ROWS * Q * KP;
    if (bl > blk) blk = bl;
    if (bg > blk) blk = bg;
    pl.cyc = (long long)K1G * P2 * blk +
             (long long)K1G * (C2 == 1 ? 1 : P2 * C2) * N_PES * N_LANES * rs *
                 (C2 == 1 ? c1s : ct1);
    return pl;
}

static ConvPlan plan_conv(int C1, int K1, int P, int Q, int R, int S, int stride) {
    ConvPlan best; best.cyc = LLONG_MAX;
    for (int kp = 1; kp <= N_PES; kp *= 2) {
        if (N_PES % kp) continue;
        ConvPlan pl = eval_mapping(C1, K1, P, Q, R, S, stride, kp, N_PES / kp);
        if (pl.valid && pl.cyc < best.cyc) best = pl;
    }
    return best;
}

// ---------------------------------------------------------------------------
// Demo: run one conv layer + a global avgpool. (A full network walks a layer
// table repeating run_conv/run_gpe — extend with the same structure as
// net_runner.)
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: %s <xclbin>\n", argv[0]); return 1; }

    auto device = xrt::device(0);
    auto uuid = device.load_xclbin(argv[1]);
    auto conv = xrt::kernel(device, uuid, "magnet_top");
    auto gpe  = xrt::kernel(device, uuid, "global_pe_top");

    // ---- example layer: 56x56x64 -> 64, 3x3 s1 p1 (ResNet res2_3x3) ----
    const int H = 56, W = 56, C = 64, K = 64, R = 3, S = 3, stride = 1, pad = 1;
    const int C1 = C / VECTOR_SIZE, K1 = (K + N_LANES - 1) / N_LANES;
    const int P = (H + 2 * pad - R) / stride + 1;
    const int Q = (W + 2 * pad - S) / stride + 1;
    const int mult = 5, shift = 11, relu = 1;

    const ConvPlan pl = plan_conv(C1, K1, P, Q, R, S, stride);
    if (!pl.valid) { printf("no valid mapping\n"); return 1; }
    printf("plan: KPxCP=%dx%d CT1=%d (est %lld cycles)\n",
           pl.KP, pl.CP, pl.CT1, pl.cyc);

    // ---- buffers (8B word units; contract in hw/include/magnet_top.h) ----
    const size_t ia_bytes = (size_t)H * W * C1 * 8;
    const size_t w_bytes  = (size_t)K1 * N_LANES * R * S * C1 * 8;
    const size_t oa_bytes = (size_t)P * Q * K1 * 8;
    const size_t b_bytes  = (size_t)K1 * N_LANES * 4;

    // arg order: ia0..ia3 (aliases of one buffer), w, oa, bias
    auto bo_ia = xrt::bo(device, ia_bytes, conv.group_id(0));
    auto bo_w  = xrt::bo(device, w_bytes,  conv.group_id(4));
    auto bo_oa = xrt::bo(device, oa_bytes, conv.group_id(5));
    auto bo_b  = xrt::bo(device, b_bytes,  conv.group_id(6));

    // (real use: fill quantized weight/bias/input here)
    std::vector<uint8_t> zeros(ia_bytes, 0);
    bo_ia.write(zeros.data());
    bo_ia.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ---- run conv (argument order = magnet_top.h) ----
    auto run = conv(bo_ia, bo_ia, bo_ia, bo_ia, bo_w, bo_oa, bo_b,
                    H, W, C1, K1, K, P, Q, R, S,
                    stride, pad, pl.CT1, pl.KP, pl.CP,
                    mult, shift, relu);
    run.wait();
    printf("conv done\n");

    // ---- global avgpool example (GPE_AVGPOOL=2, multA=round(2^15/(P*Q))) ----
    auto bo_pool = xrt::bo(device, (size_t)K1 * 8, gpe.group_id(2));
    auto run2 = gpe(bo_oa, bo_oa, bo_pool, /*mode=*/2,
                    P, Q, K1, P, Q, 1, 0,
                    (int)((1 << 15) / (P * Q)), 0, 15, 0);
    run2.wait();
    bo_pool.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    printf("pool done\n");
    return 0;
}
