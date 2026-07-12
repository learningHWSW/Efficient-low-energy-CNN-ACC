# =============================================================================
# build_vivado.tcl — classic Vivado flow (Windows/PowerShell friendly):
#   HLS kernel IP -> block design (PS + kernel + AXI) -> synth/impl -> .bit/.xsa
#
# This is the WINDOWS-native path to a real bitstream (the Linux-only
# restriction applies to the Vitis `v++ -l`/XRT flow, not to this one).
# The kernel is controlled through its s_axilite register map (addresses in
# build/hls/hls/syn/report/csynth.rpt); on the board drive it with PYNQ or
# /dev/mem instead of XRT.
#
# Prereqs: run the HLS csynth first so the IP exists:
#   .\hw\scripts\run_hls.ps1 synth        # -> build/hls/hls/impl/ip
#   (optional) .\hw\scripts\run_hls.ps1 gpe-synth
#
# Run (from the repo root):
#   & 'C:\AMDDesignTools\2026.1\Vivado\bin\vivado.bat' -mode batch `
#       -source hw/scripts/build_vivado.tcl
#
# Outputs under build/vivado/:
#   magnet_top_wrapper.bit   (bitstream)
#   magnet_top.xsa           (hardware handoff for PYNQ / Vitis SW)
#   post_route_util.rpt, post_route_timing.rpt
# =============================================================================

set REPO   [file normalize [file dirname [info script]]/../..]
set BOARD  xilinx.com:zcu104:part0:1.1

# Env overrides (default = 8-PE build/hls at 200 MHz -> build/vivado):
#   MAGNET_IP_DIR  : magnet_top HLS impl/ip dir (e.g. build/hls_zu7final for 16 PE)
#   MAGNET_FREQ    : PL clock in MHz (lower for timing-tight configs)
#   MAGNET_OUT     : output subdir + bitstream prefix under build/
proc envdef {name def} {
    return [expr {[info exists ::env($name)] ? $::env($name) : $def}]
}
set FREQ    [envdef MAGNET_FREQ 200.0]
set OUTNAME [envdef MAGNET_OUT  vivado]
set OUTDIR  $REPO/build/$OUTNAME

# Kernels to place in the design. Add global_pe_top for the full network:
#   set KERNELS {magnet_top global_pe_top}
set KERNELS {magnet_top}

# IP repository per kernel (HLS impl/ip dirs)
array set IPREPO [list \
    magnet_top    [envdef MAGNET_IP_DIR $REPO/build/hls/hls/impl/ip] \
    global_pe_top $REPO/build/hls_gpe/hls/impl/ip]

file mkdir $OUTDIR
create_project magnet_bd $OUTDIR/magnet_bd -force
set_property board_part $BOARD [current_project]

# ---- IP repos ----
set repos {}
foreach k $KERNELS { lappend repos $IPREPO($k) }
set_property ip_repo_paths $repos [current_project]
update_ip_catalog

# ---- Block design ----
create_bd_design "system"

# Zynq UltraScale+ PS with the ZCU104 board preset (correct DDR/MIO)
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset "1"} $ps
# 200 MHz PL clock; enable one HPM master (control) + one HP slave (data)
set_property -dict [list \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $FREQ \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH {128}] $ps

# kernel instances
foreach k $KERNELS {
    create_bd_cell -type ip -vlnv xilinx.com:hls:$k:1.0 ${k}_0
}

# ---- clocking / reset on the 200 MHz PL domain ----
set clk  [get_bd_pins ps/pl_clk0]
set rstn [get_bd_pins ps/pl_resetn0]
set rst  [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst_200]
connect_bd_net $clk  [get_bd_pins rst_200/slowest_sync_clk]
connect_bd_net $rstn [get_bd_pins rst_200/ext_reset_in]
connect_bd_net $clk  [get_bd_pins ps/maxihpm0_fpd_aclk]
connect_bd_net $clk  [get_bd_pins ps/saxihp0_fpd_aclk]
set arstn_ic [get_bd_pins rst_200/interconnect_aresetn]
set arstn_pr [get_bd_pins rst_200/peripheral_aresetn]

# collect all kernel data masters and control slaves
set masters {}
set ctrls   {}
foreach k $KERNELS {
    foreach mp [get_bd_intf_pins -quiet ${k}_0/m_axi_*] { lappend masters $mp }
    lappend ctrls [get_bd_intf_pins ${k}_0/s_axi_control]
}

# ---- data path: kernel masters -> SmartConnect -> PS S_AXI_HP0 ----
set nm [llength $masters]
set smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect axi_smc]
set_property -dict [list CONFIG.NUM_SI $nm CONFIG.NUM_MI {1}] $smc
connect_bd_net $clk [get_bd_pins axi_smc/aclk]
connect_bd_net $arstn_ic [get_bd_pins axi_smc/aresetn]
connect_bd_intf_net [get_bd_intf_pins axi_smc/M00_AXI] \
    [get_bd_intf_pins ps/S_AXI_HP0_FPD]
set i 0
foreach mp $masters {
    connect_bd_intf_net $mp \
        [get_bd_intf_pins axi_smc/S[format %02d $i]_AXI]
    incr i
}

# ---- control path: PS M_AXI_HPM0 -> SmartConnect -> kernel s_axi_control ----
set nc [llength $ctrls]
set smcc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect axi_smc_ctrl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI $nc] $smcc
connect_bd_net $clk [get_bd_pins axi_smc_ctrl/aclk]
connect_bd_net $arstn_ic [get_bd_pins axi_smc_ctrl/aresetn]
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] \
    [get_bd_intf_pins axi_smc_ctrl/S00_AXI]
set i 0
foreach cp $ctrls {
    connect_bd_intf_net [get_bd_intf_pins axi_smc_ctrl/M[format %02d $i]_AXI] $cp
    incr i
}

# ---- clocks/resets on the kernels ----
foreach k $KERNELS {
    connect_bd_net $clk [get_bd_pins ${k}_0/ap_clk]
    connect_bd_net $arstn_pr [get_bd_pins ${k}_0/ap_rst_n]
}

# map the control register block(s) into the PS address space
assign_bd_address
regenerate_bd_layout
validate_bd_design
save_bd_design

# Fast pre-check: set env VIV_BD_ONLY=1 to stop after building/validating the
# block design (seconds) before committing to the long synth/impl run.
if {[info exists ::env(VIV_BD_ONLY)] && $::env(VIV_BD_ONLY) eq "1"} {
    puts "BD_ONLY_DONE (block design validated, skipping synth/impl)"
    return
}

# ---- wrapper + synth/impl/bitstream ----
set bd_file [get_files system.bd]
make_wrapper -files $bd_file -top
add_files -norecurse $OUTDIR/magnet_bd/magnet_bd.gen/sources_1/bd/system/hdl/system_wrapper.v
set_property top system_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 8
wait_on_run synth_1

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

open_run impl_1
report_utilization -file $OUTDIR/post_route_util.rpt
report_timing_summary -file $OUTDIR/post_route_timing.rpt
set wns [get_property SLACK [get_timing_paths -delay_type max]]
puts "POST_ROUTE_WNS $wns"

# collect artifacts
file copy -force \
    $OUTDIR/magnet_bd/magnet_bd.runs/impl_1/system_wrapper.bit \
    $OUTDIR/magnet_top_wrapper.bit
write_hw_platform -fixed -include_bit -force -file $OUTDIR/magnet_top.xsa
puts "BITSTREAM_DONE $OUTDIR/magnet_top_wrapper.bit"
