// =============================================================================
// tb_resnet50_real.cpp — run the REAL quantized ResNet-50 through the
// accelerator kernels (Phase 6 step 2)
//
// Consumes the export of sw/quant/export_resnet50.py:
//   manifest.txt    : layer graph + requant constants (space-separated)
//   <layer>_w.bin   : weights packed in the accelerator layout
//   <layer>_b.bin   : int32 bias (padded to N_LANES multiple)
//   test_input.bin  : quantized input (HWC int8)
//   test_logits.bin : expected logits from the Python integer simulation
//   checks.txt      : per-tensor (count, sum) checksums for localization
//
// Every layer runs through net_runner (plan_conv picks KP/CP/CT1 per layer)
// and is checked against the Python simulation checksum; the final logits
// must match bit-exactly.
//
// Run: run_hls.ps1 real   (g++; int8 export by default)
//      tb_resnet50_real.exe [export_dir]
// =============================================================================

#include "../../sw/runtime/net_runner.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

static std::string g_dir;

static std::vector<char> read_bin(const std::string &name) {
    std::ifstream f(g_dir + "/" + name, std::ios::binary);
    if (!f) {
        printf("FATAL: cannot open %s/%s\n", g_dir.c_str(), name.c_str());
        exit(1);
    }
    return std::vector<char>((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
}

// tensor pool: name -> (Tensor, real channel count)
struct Entry {
    nr::Tensor t;
    int C = 0;
};
static std::map<std::string, Entry> pool;

// per-tensor checksums from the Python integer simulation
static std::map<std::string, std::pair<long long, long long>> checks;

static int check_tensor(const std::string &name) {
    const Entry &e = pool[name];
    long long cnt = (long long)e.t.H * e.t.W * e.C, sum = 0;
    for (int i = 0; i < e.t.H * e.t.W; ++i)
        for (int c = 0; c < e.C; ++c)
            sum += (int8_t)iavec_get(e.t.data[i * e.t.C1 + c / VECTOR_SIZE],
                                     c % VECTOR_SIZE);
    auto it = checks.find(name);
    if (it == checks.end()) {
        printf("  %-16s (no reference checksum)\n", name.c_str());
        return 0;
    }
    const bool ok = (it->second.first == cnt && it->second.second == sum);
    if (!ok)
        printf("  %-16s FAIL count %lld/%lld sum %lld/%lld\n", name.c_str(),
               cnt, it->second.first, sum, it->second.second);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    g_dir = (argc > 1) ? argv[1] : "sw/quant/export";
    std::ifstream mf(g_dir + "/manifest.txt");
    if (!mf) {
        printf("FATAL: %s/manifest.txt not found (run export_resnet50.py)\n",
               g_dir.c_str());
        return 1;
    }

    // reference checksums
    {
        std::ifstream cf(g_dir + "/checks.txt");
        std::string name;
        long long cnt, sum;
        while (cf >> name >> cnt >> sum)
            checks[name] = {cnt, sum};
    }

    std::string tok;
    int errors = 0, layers = 0;
    std::vector<int8_t> expected_logits;
    std::string logits_tensor;
    int logits_K = 0;

    while (mf >> tok) {
        if (tok == "model") {
            std::string m;
            int bits, vec, lanes;
            mf >> m >> bits >> vec >> lanes;
            if (vec != VECTOR_SIZE || lanes != N_LANES) {
                printf("FATAL: export is %d-bit vec=%d but the build has "
                       "VECTOR_SIZE=%d (rebuild or re-export)\n",
                       bits, vec, VECTOR_SIZE);
                return 1;
            }
        } else if (tok == "input") {
            int H, W, C;
            std::string ifile, lfile;
            mf >> H >> W >> C >> ifile >> lfile;
            std::vector<char> raw = read_bin(ifile);
            Entry e;
            e.C = C;
            e.t.alloc(H, W, (C + VECTOR_SIZE - 1) / VECTOR_SIZE);
            for (int i = 0; i < H * W; ++i)
                for (int c = 0; c < C; ++c)
                    iavec_set(e.t.data[i * e.t.C1 + c / VECTOR_SIZE],
                              c % VECTOR_SIZE, (ia_t)raw[i * C + c]);
            pool["input"] = std::move(e);
            std::vector<char> lr = read_bin(lfile);
            expected_logits.assign(lr.begin(), lr.end());
            printf("input %dx%dx%d, %zu expected logits\n", H, W, C,
                   expected_logits.size());
        } else if (tok == "conv") {
            std::string name, src, dst, wf, bf, mfn;
            int C, K, R, S, stride, pad, relu, shift;
            mf >> name >> src >> dst >> C >> K >> R >> S >> stride >> pad >>
                relu >> shift >> wf >> bf >> mfn;
            const Entry &in = pool[src];
            nr::ConvParams cp;
            cp.K = K; cp.R = R; cp.S = S; cp.stride = stride; cp.pad = pad;
            cp.shift = shift; cp.relu = relu;
            // weights: bytes are already in [Kpad][R][S][C1] element order
            std::vector<char> wb = read_bin(wf);
            const size_t n_words = wb.size() / VECTOR_SIZE;
            cp.w.resize(n_words);
            for (size_t i = 0; i < n_words; ++i)
                for (int v = 0; v < VECTOR_SIZE; ++v)
                    wvec_set(cp.w[i], v, (w_t)wb[i * VECTOR_SIZE + v]);
            std::vector<char> bb = read_bin(bf);
            cp.bias.resize(bb.size() / 4);
            std::memcpy(cp.bias.data(), bb.data(), bb.size());
            // per-channel requant multiplier table
            std::vector<char> mb = read_bin(mfn);
            cp.mults.resize(mb.size() / 4);
            std::memcpy(cp.mults.data(), mb.data(), mb.size());

            Entry out;
            out.C = K;
            nr::run_conv(in.t, cp, out.t);
            pool[dst] = std::move(out);
            errors += check_tensor(dst);
            logits_tensor = dst;
            logits_K = K;
        } else if (tok == "maxpool") {
            std::string name, src, dst;
            int R, S, stride, pad;
            mf >> name >> src >> dst >> R >> S >> stride >> pad;
            Entry out;
            out.C = pool[src].C;
            nr::run_pool(pool[src].t, out.t, GPE_MAXPOOL, R, S, stride, pad);
            pool[dst] = std::move(out);
            errors += check_tensor(dst);
        } else if (tok == "eltwise") {
            std::string name, src, src2, dst;
            int mA, mB, sh, relu;
            mf >> name >> src >> src2 >> dst >> mA >> mB >> sh >> relu;
            Entry out;
            out.C = pool[src].C;
            nr::run_eltwise(pool[src].t, pool[src2].t, out.t, mA, mB, sh, relu);
            pool[dst] = std::move(out);
            errors += check_tensor(dst);
        } else if (tok == "gavgpool") {
            std::string name, src, dst;
            int multA, shift;
            mf >> name >> src >> dst >> multA >> shift;
            const Entry &in = pool[src];
            Entry out;
            out.C = in.C;
            nr::run_pool(in.t, out.t, GPE_AVGPOOL, in.t.H, in.t.W, 1, 0,
                         multA, shift);
            pool[dst] = std::move(out);
            errors += check_tensor(dst);
        }
        if (tok == "conv" || tok == "maxpool" || tok == "eltwise" ||
            tok == "gavgpool") {
            ++layers;
            if (layers % 10 == 0)
                printf("  ... %d layers done (%d errors)\n", layers, errors);
        }
    }

    // final logits: bit-exact compare + top-5
    const Entry &lg = pool[logits_tensor];
    int lerr = 0;
    std::vector<std::pair<int, int>> scored;
    for (int k = 0; k < logits_K; ++k) {
        const int8_t hw = (int8_t)iavec_get(
            lg.t.data[k / VECTOR_SIZE], k % VECTOR_SIZE);
        if ((size_t)k < expected_logits.size() && hw != expected_logits[k])
            ++lerr;
        scored.push_back({(int)hw, k});
    }
    std::sort(scored.rbegin(), scored.rend());
    printf("\nlogits: %d/%d mismatches vs Python integer simulation\n",
           lerr, logits_K);
    printf("top-5 class indices: ");
    for (int i = 0; i < 5; ++i)
        printf("%d(%d) ", scored[i].second, scored[i].first);
    printf("\n");

    errors += lerr;
    printf(errors == 0 ? "\n=== REAL-WEIGHT RESNET-50 PASSED "
                         "(%d layers, bit-exact) ===\n"
                       : "\n=== FAILED: %d errors over %d layers ===\n",
           errors ? errors : layers, errors ? layers : layers);
    return errors ? 1 : 0;
}
