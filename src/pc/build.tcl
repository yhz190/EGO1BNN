# build.tcl -- synth + impl + bitstream + reports
# usage: vivado -mode batch -source src/pc/build.tcl
set root [file normalize [file join [file dirname [info script]] .. ..]]
set prj   [file join $root build ego1_bnn ego1_bnn.xpr]
set rep   [file join $root build reports]
set bit   [file join $root report ego1_bnn.bit]
file mkdir $rep
file mkdir [file join $root report]

open_project $prj

puts "\[tcl\] === synthesize ==="
reset_run synth_1
launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} { error "synth failed" }
open_run synth_1 -name synth_1
report_utilization      -file [file join $rep synth_util.rpt]
report_timing_summary   -file [file join $rep synth_timing.rpt] -delay_type max
report_dsp              -file [file join $rep synth_dsp.rpt]

puts "\[tcl\] === implement ==="
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { error "impl failed" }
open_run impl_1
report_utilization    -file [file join $rep impl_util.rpt]
report_timing_summary -file [file join $rep impl_timing.rpt]
report_dsp            -file [file join $rep impl_dsp.rpt]
write_bitstream -force $bit

puts "\[tcl\] === done: $bit ==="
close_project