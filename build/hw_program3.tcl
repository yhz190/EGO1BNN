open_hw_manager
connect_hw_server -url localhost:3121 -allow_non_jtag
open_hw_target [lindex [get_hw_targets] 0]
set dev [lindex [get_hw_devices] 0]
current_hw_device $dev
set_property PROGRAM.FILE {<local>/build_top/patch2/patch2.runs/impl_1/top_bnn.bit} $dev
program_hw_devices $dev
puts "PROGRAM_OK"
close_hw_target
disconnect_hw_server
puts "FLASH_DONE"