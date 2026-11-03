# pre_synth.tcl -- copy the $readmemh sources into the synthesis run directory
# (Vivado resolves $readmemh relative to the run directory; missing files silently
# initialise the memories to zero, which then lets the optimiser fold the whole BNN
# datapath away -- this hook prevents that.)
set src_dir <local>/build_top
foreach f {fw.hex weights.hex} {
    if {[file exists $src_dir/$f]} {
        file copy -force $src_dir/$f [pwd]/$f
        puts "\[pre_synth\] copied $f -> [pwd]/$f"
    } else {
        puts "\[pre_synth\] ERROR: $src_dir/$f missing"
    }
}
