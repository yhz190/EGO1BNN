open_hw_manager
connect_hw_server -url localhost:3121 -allow_non_jtag
set targets [get_hw_targets]
puts "TARGETS $targets"
if {[llength $targets] == 0} { puts "NO_TARGET"; exit 1 }
open_hw_target [lindex $targets 0]
set dev [lindex [get_hw_devices] 0]
current_hw_device $dev
puts "DEVICE [get_property PART $dev]"
puts "IDCODE [get_property IDCODE $dev]"
close_hw_target
disconnect_hw_server
puts "SCAN_DONE"