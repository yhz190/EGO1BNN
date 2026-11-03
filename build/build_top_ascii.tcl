# ============================================================================
# build_top_ascii.tcl -- full-design synth (+ optional impl/bitstream) for top_bnn
# All paths are ASCII (Vivado Tcl on zh-CN reads scripts in the system codepage,
# so non-ASCII source paths break file operations).
# usage (synth):  vivado -mode batch -source build_top_ascii.tcl
# usage (build):  vivado -mode batch -source build_top_ascii.tcl -tclargs impl
# ============================================================================
set src   <local>/build_top/src
set prj   <local>/build_top/patch2
set rep   <local>/build_top/patch2/reports
set bit   <local>/EGO1_BNN/05_delivery/ego1_bnn_patch2.bit
file mkdir $prj
file mkdir $rep

create_project -force patch2 $prj -part xc7a35tcsg324-1
add_files -norecurse [list \
    $src/picorv32.v \
    $src/uart_phy.v $src/frame_rx_tx.v \
    $src/pcpi_bnn.v $src/sfr_bnn.v $src/vga_simple.v $src/top_bnn.v \
    $src/popcnt9.v $src/lane_array.v $src/accum.v $src/theta_cmp.v \
    $src/win_buf.v $src/act_bram.v $src/l3_mix.v $src/l4_pool.v \
    $src/l5_fc.v $src/layer_seq.v $src/bnn_engine.v ]
add_files -fileset constrs_1 -norecurse $src/board_ego1_top_bnn.xdc
set_property top top_bnn [current_fileset]
set_property include_dirs [list $src] [get_filesets sources_1]
update_compile_order -fileset sources_1
# make $readmemh sources visible inside the synthesis run directory
set_property STEPS.SYNTH_DESIGN.TCL.PRE <local>/build_top/pre_synth.tcl [get_runs synth_1]

puts "\[tcl\] === synth ==="
launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} { error "synth failed" }
open_run synth_1 -name synth_1

# clk50 is now constrained persistently in board_ego1_top_bnn.xdc
update_timing -quiet

if {[catch {
    report_utilization       -file $rep/synth_util.rpt
    report_timing_summary    -file $rep/synth_timing.rpt -delay_type max
    report_clock_utilization  -file $rep/synth_clocks.rpt
    report_utilization -hierarchical -file $rep/synth_hier.rpt
    set dsp [llength [get_cells -hier -filter {REF_NAME =~ DSP48E1*}]]
    set wns "n/a"
    set tp [get_timing_paths -max_paths 1 -delay_type max -quiet]
    if {[llength $tp] > 0} { set wns [get_property SLACK [lindex $tp 0]] }
    puts "GATE_SYNTH DSP=$dsp WNS=$wns"
} err]} { puts "GATE_SYNTH_WARN $err" }
foreach c {u_eng u_cpu u_l4 u_l5} { puts "CELL $c [llength [get_cells -hier -quiet -filter NAME=~*$c*]]" }
close_design

if {[llength $argv] > 0 && [lindex $argv 0] eq "impl"} {
    puts "\[tcl\] === impl + bitstream ==="
    launch_runs impl_1 -to_step write_bitstream -jobs 8
    wait_on_run impl_1
    if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { error "impl failed" }
    open_run impl_1
    report_utilization    -file $rep/impl_util.rpt
    report_timing_summary -file $rep/impl_timing.rpt
    set dsp2 [llength [get_cells -hier -filter {REF_NAME =~ DSP48E1*}]]
    set wns2 "n/a"
    set tp2 [get_timing_paths -max_paths 1 -delay_type max -quiet]
    if {[llength $tp2] > 0} { set wns2 [get_property SLACK [lindex $tp2 0]] }
    puts "GATE_IMPL DSP=$dsp2 WNS=$wns2"
    set bit_src [file join [get_property DIRECTORY [get_runs impl_1]] top_bnn.bit]
    if {[file exists $bit_src]} {
        file copy -force $bit_src $bit
        puts "\[tcl\] bitstream copied: $bit"
    }
    close_design
}
puts "BUILD_TOP_DONE"
