# 아키텍처 설계 문서 — MAGNet-style CNN 가속기 (Vitis HLS)

이 문서는 `resources/`의 NVIDIA 논문들(MAGNet, Simba, Buffets)과
Sze의 *Efficient Processing of Deep Neural Networks* 를 근거로,
MAGNet의 PE 아키텍처 템플릿을 AMD Vitis HLS(FPGA)로 옮긴 설계를 설명한다.

---

## 1. MAGNet 요약과 본 설계의 대응 관계

MAGNet (ICCAD 2019)은 세 가지 컴포넌트로 구성된다.

| MAGNet 컴포넌트 | 역할 | 본 프로젝트 대응 |
|---|---|---|
| **Designer** | C++/SystemC 템플릿 → HLS로 RTL 생성 | `hw/src/magnet_top.cpp` + `hw/include/accel_config.h` (컴파일타임 파라미터) |
| **Mapper** | 레이어→하드웨어 매핑(타일링, 순서) 탐색 | `sw/mapper/mapper.py` (제약 검증 + 사이클/트래픽 모델) |
| **Tuner** | 베이지안 최적화 기반 설계공간 탐색 | Phase 4 (config sweep 스크립트로 시작) |

### 컴퓨트 계층 (논문 Fig. 2)

```
System ── N_PES 개의 PE + Global Buffer  (Phase 3)
  └─ PE ── N_LANES 개의 Vector MAC + W/IA/Accum buffer + PPU   ← 현재 구현
       └─ Vector MAC ── VECTOR_SIZE-wide dot product
```

### 설계시점 파라미터 (논문 Table I ↔ `accel_config.h`)

| 논문 | 코드 | 기본값 | 의미 |
|---|---|---|---|
| VectorSize | `VECTOR_SIZE` | 8 | dot-product 폭 = **C(입력채널) 차원 공간 매핑** |
| NLanes | `N_LANES` | 8 | PE 내 vector MAC 수 = **K(출력채널) 차원 공간 매핑** |
| WPrecision | `W_BITS` | 8 | int8 가중치 |
| IAPrecision | `IA_BITS` | 8 | int8 activation |
| AccumPrecision | `ACC_BITS` | 32 | FPGA에서는 32b가 자연스러움 (논문 24b) |
| WeightBufSize | `WBUF_DEPTH` | 1024 | lane별 뱅크, word=64b → lane당 8KB |
| IABufSize | `IABUF_DEPTH` | 8192 | word=64b → 64KB |
| AccumBufSize | `Q_MAX` | 256 | 출력 한 행의 partial sum 라인 |
| NPEs | `N_PES` | 1 | Phase 3에서 확장 |

기본 구성의 처리량: **64 MAC/cycle = 25.6 GOPS @ 200 MHz** (MAC=2op 기준).

---

## 2. 데이터플로우: OS-LWS (Output Stationary – Local Weight Stationary)

논문 Fig. 12의 결론 — **OS-LWS가 거의 모든 구성에서 에너지 최적** — 을 따라
OS-LWS를 기본 데이터플로우로 구현했다 (논문 Fig. 4(e)의 loop nest).

```
for k1 = [0 : K1)          # 출력채널 타일 (N_LANES 개씩)     — temporal
  for p1 = [0 : P2)         # 출력 행 그룹 (PT_ROWS 개씩)      — temporal
    for c2 = [0 : C2)       # 입력채널 타일 (CT1 word씩)       — temporal (psum 누적)
      for p0 = [0 : PT)     # 그룹 내 행 — load ‖ compute 오버랩(dataflow)
        for (r, s, c1)      # 커널 위치 × 타일 내 채널 word    — temporal
          for q0 = [0 : QB) # 출력 열 (innermost)              — temporal
            acc[p0][q0][lane] += W[k1,lane, r,s,c2·CT1+c1]
                                 · IA[p·st+r, q0·st+s, c2·CT1+c1]
            # lane ∈ N_LANES (공간), · 는 VECTOR_SIZE-wide dot (공간)
```

- **LWS**: `(r,s,c1)`이 고정된 동안 같은 weight word가 q0 루프 전체(QB회)에서 재사용된다.
  논문의 weight collector(래치 어레이)에 해당하는 재사용이며, FPGA에서는 BRAM 읽기
  에너지가 싸므로 별도 latch 계층 없이 주소 고정으로 같은 순서를 얻는다.
- **OS**: partial sum이 `acc_buf[p0][q][lane]`(accumulation buffer)에 머물며
  모든 c2 타일에 걸쳐 R·S·C1 회 read-modify-write 누적된다. psum의 DRAM 왕복이 없다.
- **C-타일링** (Phase 2): 필터가 weight buffer보다 크면 입력채널을 CT1 word
  타일로 쪼개고(c2 루프), psum은 acc_buf에 상주한 채 타일들을 누적한다.
  C2>1일 때 weight 타일은 (k1,p1,c2)마다 재로드되며(태그 캐시로 C2==1이면 k1당
  1회), 이것이 psum-stationary의 대가다. 현재 기본 버퍼에선 ResNet-50 전
  레이어가 C2=1이고, 멀티 PE를 위해 버퍼를 줄일 때 C-타일링이 활성화된다.
- **Double buffering** (Phase 2): p0 루프가 HLS dataflow 영역 — IA row buffer가
  iteration별 PIPO(ping-pong)가 되어 load_ia(p0+1)와 compute(p0)가 오버랩된다.
  stride 1 레이어에서 IA 로드는 compute에 완전히 가려진다.
- **DSP packing** (Phase 2): 이웃 lane 쌍이 같은 activation을 공유하므로
  `(w1≪18 + w0)×a` 27×18 곱셈 1개로 MAC 2개를 얻는다(WP487). borrow 보정
  산술은 (w0,w1,a) 전조합 16.7M 전수 검증. DSP 수요 64→32/PE.
- PPU는 행 그룹 단위로 bias 포함 값을 requantize(scale·round) + ReLU + int8 saturate 한다.

### 파이프라인 안전장치 (`MIN_QB`)

`acc_line` RMW는 같은 주소를 QB 사이클마다 재방문한다. II=1을 보장하려면
재방문 거리가 파이프라인 깊이보다 커야 하므로, `QB = max(Q, MIN_QB=16)` 으로
패딩하고 초과분 결과는 store 단계에서 버린다. HLS `dependence distance=16`
pragma가 이 보장을 스케줄러에 전달한다.

---

## 3. 메모리 시스템

### 온칩 버퍼 (MAGNet Fig. 2(b) 대응)

| 버퍼 | 구조 | 크기(기본) | 논문 대응 |
|---|---|---|---|
| `wbuf[WBUF_DEPTH][N_LANES]` | lane별 완전 분할(BRAM 뱅크 8개), (k1,c2) 타일 캐시 | 64KB | Weight buffer (read port = W_BITS×NLanes×VectorSize) |
| `rowbuf[IABUF_DEPTH]` | dataflow PIPO(×2), zero-pad 포함 R개 입력 행 | 64KB×2 | Input activation buffer (buffet fill/read 분리) |
| `acc_buf[PT_ROWS][Q_MAX][N_LANES]` | lane별 완전 분할, c2 루프 동안 psum 상주 | 64KB | Accumulation buffer (RMW, buffet update) |
| `bias_reg[N_LANES]` | 레지스터 | — | (PPU bias) |

Buffets 논문의 fill/read/update 시맨틱 중 **update**(RMW)는 `acc_line`이,
**fill→read 분리**는 load/compute 단계 분리가 담당한다. 명시적 credit 기반
동기화(buffet)는 Phase 2의 load/compute/store 오버랩(double buffering)에서 도입한다.

### DRAM 레이아웃 (호스트가 준비, word = 64b = 8×int8)

| 배열 | 레이아웃 | 패딩 규칙 |
|---|---|---|
| IA | `[H][W][C1]` (NHWC) | C를 `VECTOR_SIZE` 배수로 zero-pad |
| W | `[Kpad][R][S][C1]` (KRSC) | K를 `N_LANES` 배수로 zero-pad |
| OA | `[P][Q][K1]`, word = N_LANES×int8 | lane ≥ K 부분은 무시 |
| bias | `int32 [Kpad]` | k ≥ K 는 0 |

### 매핑 제약 (mapper.py가 검증)

```
R·S·C1                ≤ WBUF_DEPTH      # k1 타일의 필터가 통째로 적재
R·win_cols·C1         ≤ IABUF_DEPTH     # win_cols = (QB-1)·stride + S
Q                     ≤ Q_MAX
```
ResNet-50 / VGG-16의 모든 conv 레이어가 기본 구성에 들어간다
(`python sw/mapper/mapper.py --network resnet50` 으로 확인).

---

## 3.5 멀티 PE 시스템 (Phase 3a — Simba chiplet 구조)

Simba(MICRO 2019) chiplet 내부를 HLS dataflow 태스크 4종으로 구현:

```
ia_loader ──ia_st[pe]──▶ ┌─────┐
(multicast/C-slice 분배) │ PE0 │─psum_st[0]─┐
w_loader  ──w_st[pe]───▶ │ ... │            ├──▶ gather_store
(K/C 타일 분배)          │ PE3 │─psum_st[3]─┘    (cross-PE reduction
                         └─────┘                  + bias + PPU + store)
```

- **Spatial 매핑** (런타임 KP×CP=N_PES, Simba Listing 2의 parallel_for):
  - `KP` (K-split): PE 그룹이 서로 다른 k1 타일 담당. IA는 **multicast**
    (DRAM 1회 읽기 → KP개 스트림) — 단일 PE 대비 IA 트래픽 ÷KP.
  - `CP` (C-split): 그룹 내 PE가 입력채널 슬라이스 분담, psum을 gather에서
    합산 = **Simba cross-PE partial-sum reduction**. bias는 reduction 후 1회.
  - mapper가 레이어별로 (4×1)/(2×2)/(1×4) 중 최소 사이클 매핑 선택.
- **Lockstep 규약**: 모든 태스크가 동일 수식(derive())으로 루프 경계를
  파생하므로 스트림 생산/소비 개수가 구조적으로 일치 (reference_model.py가
  스트림 잔량 assert로 검증). k1≥K1(KP 패딩)은 zero 가중치, p≥P(행그룹
  꼬리)는 psum 소비 후 폐기.
- **제약**: C1은 CP 배수로 호스트 패딩, CT1 ≤ C1/CP.
- **멀티포트 IA 로더** (Phase 4c): 같은 DDR 버퍼를 가리키는 N_IA_PORTS(4)개
  AXI 마스터로 C-슬라이스를 병렬 읽기 — 로더 시간이 CP에서
  ceil(CP/4)로 축소. 반복당 포트별 1읽기 + PE 스트림별 정적 write 1개로
  II=1 유지 (스트림 writer가 여러 개면 FIFO 포트 충돌로 II가 늘어남에 유의).
  이 개선으로 mapper가 C-split(2×4, 1×8)을 적극 선택하게 됨.

## 3.6 Global PE와 호스트 런타임 (Phase 3b)

Simba의 Global PE(2차 저장소 + 저재사용 연산 근접 처리)와 RISC-V 컨트롤러 대응:

- **`global_pe_top`** (hw/src/global_pe.cpp, 별도 HLS 컴포넌트):
  - `GPE_ELTWISE`: O = sat(relu((A·multA + B·multB) ≫ shift)) — ResNet
    residual add. 입력별 재양자화 스케일 지원, II=1 스트리밍.
  - `GPE_MAXPOOL` / `GPE_AVGPOOL`: 윈도우 순회(II=1), 패딩 무시.
    avg의 1/count는 호스트가 multA/shift에 반영 (global avgpool 포함).
  - conv 커널과 동일한 NHWC 64b word 레이아웃 → **재패킹 없는 체이닝**.
- **`sw/runtime/net_runner.h`**: Simba RISC-V 컨트롤러의 역할 —
  레이어 시퀀서 + 커널 설정 계산. `plan_conv()`가 mapper의 사이클 모델을
  미러하여 레이어별 KP/CP/CT1을 자동 선택한다 (체이닝 제약: 텐서 C1이
  이전 레이어에서 고정되므로 C1 % CP != 0 후보는 제외). 현재는 커널 C 함수
  직접 호출(csim 검증), Phase 4에서 호출부만 XRT enqueue로 교체.
- **검증**: tb_global_pe(단위 7케이스) + tb_network(ResNet bottleneck 블록
  e2e: conv×3 → residual add → maxpool → global avgpool, 전 스테이지 비트 일치).

## 4. 실행 흐름 (현재 구현)

```
k1 타일 루프:
  load bias        : N_LANES 개
  p1 (행 그룹) 루프:
    c2 (채널 타일) 루프:
      load_weights : (k1,c2) 타일을 lane별 뱅크로 burst 로드 (태그 캐시)
      p0 루프 [HLS dataflow]:
        load_ia_rows ‖ compute_row   # PIPO로 오버랩, II=1, 64 MAC/cycle
    ppu_store_group: requant→ReLU→pack→DRAM (행 그룹 전체)
```

알려진 잔여 비효율 (Phase 3에서 해결):
1. **IA 행 재로드** — 인접 출력 행이 (R−stride)개 입력 행을 공유하지만 매번 재로드.
   dataflow 오버랩으로 레이턴시에선 가려지지만 DRAM 대역폭/에너지는 ×R.
   → 스트림 기반 라인 버퍼로 해결 (PIPO 구조 재설계 필요).
2. **k1 루프마다 IA 전체 재로드** → 멀티 PE에서 IA multicast(Global PE)로 흡수.
3. **P가 PT_ROWS 비배수일 때 꼬리 행 낭비, Q<16 패딩 낭비** — 작은 출력 레이어의
   utilization 저하. → K/C 공간 분할(멀티 PE)이 근본 해결책.

---

## 5. 로드맵

| Phase | 내용 | 상태 |
|---|---|---|
| **0** | 단일 PE OS-LWS conv 엔진 + 골든모델 TB + mapper | ✅ |
| **2** | C-타일링, double buffering, DSP packing(2MAC/DSP) | ✅ |
| **1** | Vitis csim/csynth 통과 (II=1, 200MHz, packing 확인) | ✅ (cosim은 보류) |
| **3a** | PE 4개 dataflow + cross-PE reduction + KP×CP spatial 매핑 + 버퍼 Simba 스케일 축소 | ✅ (csim 36케이스 PASS) |
| **3b** | Global PE(elementwise/pooling) + net_runner 호스트 런타임 + 네트워크 e2e 검증 | ✅ (csim/csynth 클린) |
| **—** | cosim (RTL 시뮬레이션, 축소 TB 9런) | ✅ PASS (데드락 없음 확인) |
| **4a** | **8 PE 확장** (512 MAC/cycle, ZU7EV 타깃) + gather (q,kp) flatten 재설계(N_PES 증가에도 리소스 평평) + XRT 호스트/링크 설정 딜리버러블 | ✅ |
| **4c** | **멀티포트 IA 로더**(4포트, C-split 최대 4×) + **ResNet-50 전체 네트워크**(실크기 dry-run 53 conv + 축소 e2e 16블록) + **Tuner-lite**(sw/tuner/tuner.py, 설계공간 탐색) | ✅ |
| **4b** | 보드 실행: 플랫폼 설치 → v++ 링크(system.cfg) → XRT 호스트 실기 검증 | 보드/플랫폼 필요 |
| **—** | INT4 데이터패스(W/IA_BITS=4), Q>128 지원(W 타일링) | 향후 |

**16 PE 관련**: 알고리즘은 N_PES=16으로 레퍼런스 모델 24/24 검증 완료.
합성은 zu9eg(ZCU102)급이 필요한데 무료 BASIC 라이선스가 zu9eg를 커버하지
않아 8 PE(zu7ev)를 합성 타깃으로 선택. 대학/기업 라이선스가 있으면
`N_PES=16` + `part=xczu9eg-ffvb1156-2-e` 두 줄 변경으로 확장된다.

### 참고 문헌 (resources/)
- MAGNet: A Modular Accelerator Generator for Neural Networks — ICCAD 2019
- Simba: Scaling DL Inference with MCM-Based Architecture — MICRO 2019 (Phase 3 시스템 레벨)
- Buffets: An Efficient and Composable Storage Idiom — ASPLOS 2019 (+ `buffets-master/` RTL)
- Sze et al., Efficient Processing of Deep Neural Networks (dataflow 분류 체계)
