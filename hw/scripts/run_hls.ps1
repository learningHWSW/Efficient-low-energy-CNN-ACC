# =============================================================================
# run_hls.ps1 — Vitis unified HLS flow wrapper (2024.2+ / 2026.1)
#
# Usage (from the repo root):
#   .\hw\scripts\run_hls.ps1 csim       # conv kernel C simulation
#   .\hw\scripts\run_hls.ps1 synth      # conv kernel C synthesis
#   .\hw\scripts\run_hls.ps1 cosim      # conv kernel RTL co-simulation
#                                       # (reduced TB; synthesizes first)
#   .\hw\scripts\run_hls.ps1 pe16-csim  # 16-PE config csim (zu9eg license req.)
#   .\hw\scripts\run_hls.ps1 pe16-synth # 16-PE config synthesis
#   .\hw\scripts\run_hls.ps1 int4-csim  # conv kernel csim, int4 config
#   .\hw\scripts\run_hls.ps1 int4-synth # conv kernel synthesis, int4 config
#   .\hw\scripts\run_hls.ps1 gpe-csim   # Global PE C simulation
#   .\hw\scripts\run_hls.ps1 gpe-synth  # Global PE C synthesis
#   .\hw\scripts\run_hls.ps1 gcc        # run conv TB with bundled g++ (no license)
#   .\hw\scripts\run_hls.ps1 net        # network e2e test (links both kernels, g++)
#
# Note: csim/synth/cosim require the free AMD license activation (see README)
# =============================================================================
param([string]$Mode = "csim")

$ErrorActionPreference = "Stop"
$XILINX = "C:\AMDDesignTools\2026.1"
$RepoRoot = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $RepoRoot

$cfg = "hw/scripts/hls_config.cfg"
$work = "build/hls"

switch ($Mode) {
    "csim" {
        & "$XILINX\Vitis\bin\vitis-run.bat" --mode hls --csim --config $cfg --work_dir $work
    }
    "synth" {
        & "$XILINX\Vitis\bin\v++.bat" -c --mode hls --config $cfg --work_dir $work
    }
    "cosim" {
        # dedicated config: COSIM_SMALL shrinks m_axi depths + reduced test set
        & "$XILINX\Vitis\bin\v++.bat" -c --mode hls `
            --config hw/scripts/hls_config_cosim.cfg --work_dir build/hls_cosim
        if ($LASTEXITCODE -eq 0) {
            & "$XILINX\Vitis\bin\vitis-run.bat" --mode hls --cosim `
                --config hw/scripts/hls_config_cosim.cfg --work_dir build/hls_cosim
        }
    }
    "pe16-csim" {
        # 16-PE config (zu9eg target — needs an Enterprise-tier license)
        & "$XILINX\Vitis\bin\vitis-run.bat" --mode hls --csim `
            --config hw/scripts/hls_config_pe16.cfg --work_dir build/hls_pe16
    }
    "pe16-synth" {
        & "$XILINX\Vitis\bin\v++.bat" -c --mode hls `
            --config hw/scripts/hls_config_pe16.cfg --work_dir build/hls_pe16
    }
    "int4-csim" {
        & "$XILINX\Vitis\bin\vitis-run.bat" --mode hls --csim `
            --config hw/scripts/hls_config_int4.cfg --work_dir build/hls_int4
    }
    "int4-synth" {
        & "$XILINX\Vitis\bin\v++.bat" -c --mode hls `
            --config hw/scripts/hls_config_int4.cfg --work_dir build/hls_int4
    }
    "gpe-csim" {
        & "$XILINX\Vitis\bin\vitis-run.bat" --mode hls --csim `
            --config hw/scripts/hls_config_gpe.cfg --work_dir build/hls_gpe
    }
    "gpe-synth" {
        & "$XILINX\Vitis\bin\v++.bat" -c --mode hls `
            --config hw/scripts/hls_config_gpe.cfg --work_dir build/hls_gpe
    }
    "gcc" {
        # no license needed: bundled g++ + the real ap_int headers
        $gxx = "$XILINX\Model_Composer\tps\mingw\10.0.0\win64.o\nt\bin\g++.exe"
        New-Item -ItemType Directory -Force build | Out-Null
        & $gxx -std=c++14 -O2 -I hw/include -I "$XILINX\Vitis\include" `
            hw/src/magnet_top.cpp hw/tb/tb_conv.cpp -o build/tb_conv.exe
        if ($LASTEXITCODE -eq 0) { & ".\build\tb_conv.exe" }
    }
    "net" {
        # network e2e: links the conv + global_pe kernels together (g++)
        $gxx = "$XILINX\Model_Composer\tps\mingw\10.0.0\win64.o\nt\bin\g++.exe"
        New-Item -ItemType Directory -Force build | Out-Null
        & $gxx -std=c++14 -O2 -I hw/include -I "$XILINX\Vitis\include" `
            hw/src/magnet_top.cpp hw/src/global_pe.cpp hw/tb/tb_network.cpp `
            -o build/tb_network.exe
        if ($LASTEXITCODE -eq 0) { & ".\build\tb_network.exe" }
    }
    "resnet" {
        # full ResNet-50: mapping dry run + reduced-size e2e (g++)
        $gxx = "$XILINX\Model_Composer\tps\mingw\10.0.0\win64.o\nt\bin\g++.exe"
        New-Item -ItemType Directory -Force build | Out-Null
        & $gxx -std=c++14 -O2 -I hw/include -I "$XILINX\Vitis\include" `
            hw/src/magnet_top.cpp hw/src/global_pe.cpp hw/tb/tb_resnet50.cpp `
            -o build/tb_resnet50.exe
        if ($LASTEXITCODE -eq 0) { & ".\build\tb_resnet50.exe" }
    }
    default { Write-Output "Unknown mode: $Mode (csim|synth|cosim|gpe-csim|gpe-synth|gcc|net)" }
}
