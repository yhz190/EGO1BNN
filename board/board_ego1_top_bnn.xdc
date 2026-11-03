## ============================================================================
## board_ego1_top_bnn.xdc -- top_bnn 专用引脚约束（patch2, 2026-11-01）
## ----------------------------------------------------------------------------
## 引脚来源：E-Elements EGO1 用户手册 v1.2（文献[9]）逐脚核对；官方 EGo1.xdc 未随包提供，详细比对逻辑见 src/pc/check_xdc.py
## 本文件只覆盖 top_bnn 实际使用的端口：
##   sys_clk_100m P17 | sys_rst_n P15 | pc_uart_rxd N5 | pc_uart_txd T4
##   led[3:0] = {F6,G4,G3,J4} | VGA: hs D7, vs C4, rgb[11:0]
## 三条历史警告仍然有效（见早期 SoC 工程注释；该文件未随包提供）：
##   1) 不要使用板卡目录里 board_part.xml 的引脚定义（错）
##   2) UART 方向：N5=FPGA RX, T4=FPGA TX（已实测）
##   3) LED 丝印与 led_pin 位序为降序（本设计只用 4 位，按 led[0]=F6 起）
## ============================================================================

## ---------------- 系统时钟 / 复位 ----------------
set_property -dict {PACKAGE_PIN P17 IOSTANDARD LVCMOS33} [get_ports sys_clk_100m]
set_property -dict {PACKAGE_PIN P15 IOSTANDARD LVCMOS33} [get_ports sys_rst_n]

## ---------------- UART（一根 USB 的 FT2232 第二通道）----------------
set_property -dict {PACKAGE_PIN N5 IOSTANDARD LVCMOS33} [get_ports pc_uart_rxd]
set_property -dict {PACKAGE_PIN T4 IOSTANDARD LVCMOS33} [get_ports pc_uart_txd]

## ---------------- LED[3:0] = {act, err, done, busy} ----------------
set_property -dict {PACKAGE_PIN F6 IOSTANDARD LVCMOS33} [get_ports {led[0]}]
set_property -dict {PACKAGE_PIN G4 IOSTANDARD LVCMOS33} [get_ports {led[1]}]
set_property -dict {PACKAGE_PIN G3 IOSTANDARD LVCMOS33} [get_ports {led[2]}]
set_property -dict {PACKAGE_PIN J4 IOSTANDARD LVCMOS33} [get_ports {led[3]}]

## ---------------- VGA（12bit 色 4:4:4 + HS + VS）----------------
set_property -dict {PACKAGE_PIN D7 IOSTANDARD LVCMOS33} [get_ports vga_hs]
set_property -dict {PACKAGE_PIN C4 IOSTANDARD LVCMOS33} [get_ports vga_vs]
set_property -dict {PACKAGE_PIN F5 IOSTANDARD LVCMOS33} [get_ports {vga_rgb[0]} ]
set_property -dict {PACKAGE_PIN C6 IOSTANDARD LVCMOS33} [get_ports {vga_rgb[1]} ]
set_property -dict {PACKAGE_PIN C5 IOSTANDARD LVCMOS33} [get_ports {vga_rgb[2]} ]
set_property -dict {PACKAGE_PIN B7 IOSTANDARD LVCMOS33} [get_ports {vga_rgb[3]} ]
set_property -dict {PACKAGE_PIN B6 IOSTANDARD LVCMOS33} [get_ports {vga_rgb[4]} ]
set_property -dict {PACKAGE_PIN A6 IOSTANDARD LVCMOS33} [get_ports {vga_rgb[5]} ]
set_property -dict {PACKAGE_PIN A5 IOSTANDARD LVCMOS33} [get_ports {vga_rgb[6]} ]
set_property -dict {PACKAGE_PIN D8 IOSTANDARD LVCMOS33} [get_ports {vga_rgb[7]} ]
set_property -dict {PACKAGE_PIN C7 IOSTANDARD LVCMOS33} [get_ports {vga_rgb[8]} ]
set_property -dict {PACKAGE_PIN E6 IOSTANDARD LVCMOS33} [get_ports {vga_rgb[9]} ]
set_property -dict {PACKAGE_PIN E5 IOSTANDARD LVCMOS33} [get_ports {vga_rgb[10]}]
set_property -dict {PACKAGE_PIN E7 IOSTANDARD LVCMOS33} [get_ports {vga_rgb[11]}]

## ---------------- 时钟约束 ----------------
## 板载晶振 100MHz；系统内部再产生 50MHz 引擎时钟（top_bnn 内二分频+BUFG），
## 50MHz 域在本文件用 generated clock 持久约束（见下方 clk50）。
create_clock -period 10.000 -name sys_clk_100m [get_ports sys_clk_100m]

## 50 MHz engine domain: clk_div -> BUFG (persistent generated clock)
## XDC does not support "if"; keep direct commands (BUFG pinned by DONT_TOUCH).
set_property DONT_TOUCH TRUE [get_cells -hier -quiet -filter {NAME =~ *clk_div_reg}]
set_property DONT_TOUCH TRUE [get_cells -hier -quiet -filter {NAME =~ *u_bufg}]
create_generated_clock -name clk50 -source [get_ports sys_clk_100m] -divide_by 2 [get_pins -hier -quiet -filter {NAME =~ *u_bufg/O}]


## ---------------- IO 时序 ----------------
## 1) 慢速人机 IO：led / VGA 都是"给人看"的，无固定输入输出关系
set_false_path -to   [get_ports {led[*]}]
## 2) UART 异步串行
set_false_path -from [get_ports {pc_uart_rxd}]
set_false_path -to   [get_ports {pc_uart_txd}]
## 3) VGA：25MHz 像素节拍，多周期约束（与 D2/D3 版一致，避免误报）
set_output_delay -clock sys_clk_100m 2.000 [get_ports {vga_hs vga_vs}]
set_output_delay -clock sys_clk_100m 2.000 [get_ports {vga_rgb[*]}]
set_multicycle_path 4 -setup -to [get_ports {vga_rgb[*] vga_hs vga_vs}]
set_multicycle_path 3 -hold  -to [get_ports {vga_rgb[*] vga_hs vga_vs}]
