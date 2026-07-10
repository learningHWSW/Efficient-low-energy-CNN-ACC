# Efficient AI CNN Accelerator (MAGNet-style, AMD Vitis HLS)

NVIDIA **MAGNet** (ICCAD 2019) 논문의 모듈형 가속기 템플릿을 AMD Vitis HLS로
구현하는 프로젝트. `resources/`의 논문들(MAGNet, Simba, Buffets, Sze 교과서)을
설계 근거로 삼는다. 상세 설계는 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 참조.

## 현재 상태 (Phase 4a까지 완료)

- ✅ **PE 8개 멀티 PE 시스템** (Simba chiplet 구조): IA multicast 로더 + W 분배
  로더 + PE×8 + **cross-PE reduction**(gather) — HLS dataflow 태스크 파이프라인.
  **512 MAC/cycle = 204.8 GOPS @ 200MHz**, ZU7EV에 BRAM 63%/DSP 28%/LUT 47%
- ✅ **런타임 spatial 매핑** KP×CP: K-split(8×1)/혼합(4×2)/C-split(1×8),
  mapper가 레이어별 자동 선택. gather는 (q,kp) flatten II=1 구조라
  N_PES를 키워도 리소스가 늘지 않음
- ✅ **16 PE 구성** (`-DN_PES=16`, `hls_config_pe16.cfg`): 기능 검증 완료
  (g++ int8/int4 각 36케이스) + 구조 합성 확인. **zu9eg(ZCU102)급 +
  Enterprise 라이선스 필요** (zu7ev는 BRAM 초과). `mapper.py --pes 16`
- ✅ PE 내부: OS-LWS 데이터플로우, C-타일링, DSP packing(2MAC/DSP, 전수 검증)
- ✅ **Vitis 2026.1 csim 통과**: 12 레이어 × 3 spatial = 36 케이스 골든 일치.
  16 PE 구성도 알고리즘 검증 완료(모델 24/24) — zu9eg + Enterprise 라이선스
  환경이면 config 두 줄로 확장
- ✅ Python 레퍼런스 모델이 스트림 생산/소비 개수까지 검증 (데드락 방지)
- ✅ Mapper: ResNet-50 추정 **~19.8ms/img** (8 PE; 4 PE 36.4ms, 단일 PE 191ms)
- ✅ **멀티포트 IA 로더** (4× AXI 마스터, 같은 버퍼 별칭): C-split 로더 시간이
  CP → ceil(CP/4)로 축소, IA_SEND II=1 유지. mapper가 C-split을 적극 선택
- ✅ **ResNet-50 전체 네트워크**: 실제 크기 53개 conv 전부 유효 매핑
  (dry-run, ~77ms = 13 fps 추정) + 축소 해상도(32×32, 실제 채널 폭) 16개
  bottleneck 블록 e2e 비트 일치 (`run_hls.ps1 resnet`)
- ✅ **Tuner-lite** (`sw/tuner/tuner.py`): 설계공간 탐색 — 현재 구성(8 PE,
  wbuf 256, qmax 128)이 zu7ev에서 Pareto 최적임을 확인
- ✅ **INT4 구성** (`-DUSE_INT4`): W/IA/OA 4-bit + VECTOR_SIZE 16 → **1024
  MAC/cycle = 409.6 GOPS**. int8/int4 양쪽 회귀 PASS(g++ + Vitis csim),
  ResNet-50 추정 int8 20ms → int4 12ms. `run_hls.ps1 int4-csim|int4-synth`,
  `mapper.py --int4`
- ✅ **XRT 호스트/링크 딜리버러블**: `sw/host/xrt_host.cpp` + `hw/scripts/system.cfg`
  (보드·플랫폼 미보유로 실기 검증은 보류 — Phase 4b)
- ℹ 제약: Q ≤ 128 (ResNet-50 전 레이어 커버; VGG-224급은 W 타일링 필요),
  KP/CP는 2의 거듭제곱
- ✅ **Global PE** (`global_pe_top`, 별도 커널): residual add(eltwise, 스케일별
  재양자화)·maxpool·avgpool — csim/csynth 클린 (II=1)
- ✅ **호스트 런타임** (`sw/runtime/net_runner.h`): 레이어 시퀀서 + C++판
  매퍼(plan_conv). **ResNet bottleneck 블록 e2e** (conv×3 → residual →
  maxpool → global avgpool) 전 스테이지 비트 일치
- ✅ **RTL co-simulation PASS** — 축소 TB(3 레이어 × 3 spatial = 9런)로 XSIM
  실행, 전부 비트 일치. 유한 스트림 depth의 RTL에서 완주 = **데드락 부재 확인**

## 폴더 구조

```
hw/
  include/accel_config.h   # 설계시점 파라미터 (VectorSize, NLanes, N_PES, 버퍼, 정밀도)
  include/accel_types.h    # ap_int/hls_stream 타입 (없으면 표준 타입 대체)
  include/magnet_top.h     # conv 커널 선언 + DRAM 레이아웃 규약
  include/global_pe.h      # Global PE 커널 선언 (eltwise/pool)
  src/magnet_top.cpp       # 멀티 PE conv 시스템 (loader/PE×4/gather dataflow)
  src/global_pe.cpp        # Global PE (residual add, maxpool, avgpool)
  src/archive/             # 이전 Phase 구현 보관
  tb/tb_conv.cpp           # conv 커널 TB (골든 비교, 12케이스 × 3 spatial)
  tb/tb_global_pe.cpp      # Global PE TB (7케이스)
  tb/tb_network.cpp        # 네트워크 e2e TB (ResNet bottleneck 블록)
  scripts/run_hls.ps1      # 빌드 래퍼 (csim/synth/gpe-*/gcc/net)
  scripts/hls_config*.cfg  # Vitis 통합 플로우 설정 (커널별)
sw/
  mapper/mapper.py         # MAGNet Mapper (spatial/타일 선택 + 사이클/트래픽 모델)
  model/reference_model.py # 비트-정확 Python 모델 (스트림 개수 검증 포함)
  runtime/net_runner.h     # 레이어 시퀀서 + plan_conv (Phase 4에서 XRT로 교체)
docs/ARCHITECTURE.md       # 설계 문서 (MAGNet/Simba 대응표, dataflow, 로드맵)
resources/                 # 참고 논문 및 Buffets RTL
```

## 요구 환경

- **AMD Vivado/Vitis 2026.1** — 설치 확인: `C:\AMDDesignTools\2026.1`
  - 2024.2+에는 클래식 `vitis_hls`가 없고 **통합 플로우**(`v++ --mode hls`, `vitis-run`)만 있음
  - **무료 라이선스 활성화 필요** (아래 참조). 미활성화 시 `gcc` 모드로 TB 검증은 가능
  - 설치된 디바이스: xczu7ev(ZCU104)·xczu9eg·xczu19eg — KV260(xck26)은 미설치라
    현재 타깃은 **ZCU104** (`hw/scripts/hls_config.cfg`의 `part=`로 변경)
- Python 3.10+ (mapper / reference model)

### 무료 라이선스 활성화 (1회)

2025.1부터 무료 Standard Edition도 라이선스 파일이 필요하다:
1. https://account.amd.com/en/licenses 로그인 (AMD 계정)
2. "Vivado/Vitis Standard Edition" 무료 라이선스 생성 — Host ID는 이더넷 MAC
   (`Get-NetAdapter`로 확인, 이 PC: `34-5A-60-BE-67-C7`)
3. 받은 `Xilinx.lic`를 `C:\Users\<사용자>\.Xilinx\Xilinx.lic` 에 저장

## Quick Start

```powershell
# 1) 알고리즘 검증 (Vitis/라이선스 없이 즉시 실행 가능)
python sw\model\reference_model.py

# 2) 매핑 확인
python sw\mapper\mapper.py --network resnet50
python sw\mapper\mapper.py --layer 56,56,64,64,3,3,1,1

# 3) TB 실행 — 번들 g++ + 실제 ap_int 헤더 (라이선스 불필요, 13케이스 PASS 확인됨)
.\hw\scripts\run_hls.ps1 gcc

# 4) HLS (라이선스 활성화 후)
.\hw\scripts\run_hls.ps1 csim    # C simulation
.\hw\scripts\run_hls.ps1 synth   # 합성 리포트 (II=1, 리소스, 타이밍)
.\hw\scripts\run_hls.ps1 cosim   # RTL co-simulation
```

(구버전 Vitis(≤2024.1)용 Tcl 스크립트는 `hw/scripts/run_hls.tcl` 참조)

## 설계 요약

MAGNet의 3계층 컴퓨트 구조를 따른다:
**Vector MAC** (VECTOR_SIZE=8 dot product) × **N_LANES=8** → 64 MAC/cycle PE.
입력채널(C)을 VectorSize로, 출력채널(K)을 NLanes로 공간 매핑하고,
논문에서 에너지 최적으로 보고된 **OS-LWS** 순서(가중치를 q 루프 동안 고정 재사용,
partial sum은 accumulation buffer에서 read-modify-write)로 시간 매핑한다.
PPU가 bias·requantize(int8)·ReLU를 수행해 레이어 융합을 지원한다.

## 로드맵

| Phase | 내용 |
|---|---|
| 0 ✅ | 단일 PE 엔진 + 검증 인프라 |
| 2 ✅ | C-타일링, double buffering(PIPO), DSP packing |
| 1 | Vitis csim/csynth/cosim 통과, II=1·200MHz 확인 |
| 3a | PE 4개 + cross-PE reduction + K/C spatial 매핑 (Simba 방향) |
| 3b | Global PE (multicast, elementwise/pooling), PS 호스트 런타임 |
| 4 | 16 PE + KV260 보드 통합 (XRT), ResNet-50 end-to-end, INT4 탐색 (Tuner) |
