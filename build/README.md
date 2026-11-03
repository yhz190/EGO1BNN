# build/ 说明

- `impl_util.rpt`：**最终实现资源报告**：LUT **4,405**（logic 4,184 + LUTRAM 221）、FF 5,578、BRAM 5.0、DSP 0。
- `impl_timing.rpt`：**最终实现时序报告**：`clk50` 引擎域 WNS **+4.136 ns**（13,226 端点）；`sys_clk_100m` 源时钟域 WNS **+8.892 ns**（1 端点）。
- `synth_util.rpt` / `synth_timing.rpt`：综合阶段资源/时序报告，供对照。
- 50 MHz 引擎域约束已写入 `board/board_ego1_top_bnn.xdc`（generated clock + DONT_TOUCH），实现阶段不再丢失。

- `build_top_ascii.tcl`：完整 synth+impl 流程脚本；clk50 约束已持久化在 `board/board_ego1_top_bnn.xdc`，脚本不再临时创建该时钟。

- 时序报告中的 `no_output_delay`（5 个）为 UART/LED，已用 `set_false_path` 覆盖；`no_input_delay` 有 1 个 HIGH 为 `sys_rst_n` 复位输入（异步慢速，未加 IO 约束；设计内部已做同步释放，不影响功能）。
