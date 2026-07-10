# =============================================================================
# run_hls.tcl — Vitis HLS build script (legacy classic flow, Vitis <= 2024.1)
#
# NOTE: Vitis 2024.2+ removed the classic `vitis_hls` executable — use
#   run_hls.ps1 / hls_config*.cfg (unified flow) instead. Kept for older
#   tool versions.
#
# Usage (from anywhere in the repo):
#   vitis_hls -f hw/scripts/run_hls.tcl csim     ;# C simulation (golden compare)
#   vitis_hls -f hw/scripts/run_hls.tcl synth    ;# C synthesis (reports)
#   vitis_hls -f hw/scripts/run_hls.tcl cosim    ;# RTL co-simulation (slow)
#   vitis_hls -f hw/scripts/run_hls.tcl export   ;# Vitis kernel (.xo) export
#
# To change the target board, edit the PART variable below:
#   Kria KV260 : xck26-sfvc784-2LV-c
#   ZCU104     : xczu7ev-ffvc1156-2-e
#   Pynq-Z2    : xc7z020clg400-1  (7-10ns clock recommended)
# =============================================================================

set MODE "csim"
if { [llength $::argv] >= 1 } { set MODE [lindex $::argv 0] }

set PART   {xck26-sfvc784-2LV-c}
set PERIOD 5.0  ;# ns -> 200 MHz

set SCRIPT_DIR [file dirname [file normalize [info script]]]
set HW_DIR     [file dirname $SCRIPT_DIR]

open_project -reset proj_magnet
set_top magnet_top
add_files     "$HW_DIR/src/magnet_top.cpp" -cflags "-I$HW_DIR/include -std=c++14"
add_files -tb "$HW_DIR/tb/tb_conv.cpp"     -cflags "-I$HW_DIR/include -std=c++14"

open_solution -reset "sol1" -flow_target vitis
set_part $PART
create_clock -period $PERIOD -name default

switch $MODE {
    "csim" {
        csim_design
    }
    "synth" {
        csynth_design
    }
    "cosim" {
        csynth_design
        cosim_design -rtl verilog
    }
    "export" {
        csynth_design
        export_design -format xo -output "$HW_DIR/../build/magnet_top.xo"
    }
    default {
        puts "Unknown mode: $MODE (csim|synth|cosim|export)"
        exit 1
    }
}
exit
