// =============================================================================
// accel_types.h — data types and memory-word definitions
//
// Uses ap_int/ap_uint under Vitis HLS; falls back to standard types elsewhere
// (plain g++). A 64b memory word carries VECTOR_SIZE (=8) int8 elements.
// =============================================================================
#pragma once

#include "accel_config.h"
#include <cstdint>

#if defined(__SYNTHESIS__)
#  define USE_AP_TYPES 1
#elif defined(__has_include)
#  if __has_include("ap_int.h")
#    define USE_AP_TYPES 1
#  endif
#endif

#ifdef USE_AP_TYPES
// ---------------------------------------------------------------------------
// Vitis HLS path
// ---------------------------------------------------------------------------
#include "ap_int.h"

typedef ap_int<W_BITS>    w_t;      // weight element
typedef ap_int<IA_BITS>   ia_t;     // input activation element
typedef ap_int<OA_BITS>   oa_t;     // output activation element
typedef ap_int<ACC_BITS>  acc_t;    // partial sum
typedef ap_int<64>        wide_t;   // requantization intermediate

typedef ap_uint<IA_BITS * VECTOR_SIZE>  iavec_t;   // 64b IA word
typedef ap_uint<W_BITS  * VECTOR_SIZE>  wvec_t;    // 64b W word
typedef ap_uint<OA_BITS * N_LANES>      oavec_t;   // 64b OA word
typedef int32_t                          bias_t;

static inline ia_t iavec_get(const iavec_t &v, int i) {
#pragma HLS inline
    return (ia_t)v.range(IA_BITS * i + IA_BITS - 1, IA_BITS * i);
}
static inline w_t wvec_get(const wvec_t &v, int i) {
#pragma HLS inline
    return (w_t)v.range(W_BITS * i + W_BITS - 1, W_BITS * i);
}
static inline void oavec_set(oavec_t &v, int i, oa_t x) {
#pragma HLS inline
    v.range(OA_BITS * i + OA_BITS - 1, OA_BITS * i) = (ap_uint<OA_BITS>)x;
}
static inline void iavec_set(iavec_t &v, int i, ia_t x) {
#pragma HLS inline
    v.range(IA_BITS * i + IA_BITS - 1, IA_BITS * i) = (ap_uint<IA_BITS>)x;
}
static inline void wvec_set(wvec_t &v, int i, w_t x) {
#pragma HLS inline
    v.range(W_BITS * i + W_BITS - 1, W_BITS * i) = (ap_uint<W_BITS>)x;
}
static inline oa_t oavec_get(const oavec_t &v, int i) {
#pragma HLS inline
    return (oa_t)v.range(OA_BITS * i + OA_BITS - 1, OA_BITS * i);
}

#include "hls_stream.h"

#else
// ---------------------------------------------------------------------------
// Plain C++ path (logic verification without ap_int)
// ---------------------------------------------------------------------------
#include <queue>

namespace hls {
template <typename T> class stream {
    std::queue<T> q;
public:
    void write(const T &v) { q.push(v); }
    T read() { T v = q.front(); q.pop(); return v; }
    bool empty() const { return q.empty(); }
};
} // namespace hls

typedef int8_t   w_t;
typedef int8_t   ia_t;
typedef int8_t   oa_t;
typedef int32_t  acc_t;
typedef int64_t  wide_t;

struct iavec_t { int8_t b[VECTOR_SIZE]; };
struct wvec_t  { int8_t b[VECTOR_SIZE]; };
struct oavec_t { int8_t b[N_LANES];     };
typedef int32_t bias_t;

static inline ia_t iavec_get(const iavec_t &v, int i) { return v.b[i]; }
static inline w_t  wvec_get (const wvec_t  &v, int i) { return v.b[i]; }
static inline void oavec_set(oavec_t &v, int i, oa_t x) { v.b[i] = x; }
static inline void iavec_set(iavec_t &v, int i, ia_t x) { v.b[i] = x; }
static inline void wvec_set (wvec_t  &v, int i, w_t  x) { v.b[i] = x; }
static inline oa_t oavec_get(const oavec_t &v, int i) { return v.b[i]; }

#endif // USE_AP_TYPES

// ---------------------------------------------------------------------------
// Partial-sum row word sent from a PE to gather (N_LANES accumulators)
// ---------------------------------------------------------------------------
struct psum_t {
    acc_t v[N_LANES];
};

// ---------------------------------------------------------------------------
// Layer execution configuration (MAGNet run-time parameters, Table III).
// Decided by the host/mapper and delivered as AXI-Lite scalars.
// ---------------------------------------------------------------------------
struct LayerCfg {
    int H, W;        // input activation height/width
    int C1;          // ceil(C / VECTOR_SIZE) — input-channel words (padded to a CP multiple)
    int K1;          // ceil(K / N_LANES)     — output-channel tiles
    int K;           // actual output channels (<= K1*N_LANES)
    int P, Q;        // output height/width
    int R, S;        // kernel height/width
    int stride;
    int pad;
    int CT1;         // input-channel tile size (words, <= C1/CP). Chosen by the mapper.
    int KP;          // spatial split along K (number of PE groups)
    int CP;          // spatial split along C (PEs per group). KP*CP == N_PES
    int32_t mult;    // requantization multiplier
    int shift;       // requantization shift (>= 0)
    int relu_en;     // apply ReLU when 1
};
