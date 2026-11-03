# VGA 实现与上板记录（T5，2026-11-01 17:05）

> 实现：子 agent（RTL），Lead 独立复验 + 综合/实现/烧录。判据 T5="拔掉串口仍显示"。

## 1. 独立复验（Lead 自己跑）

| 项 | 结果 |
|---|---|
| `tb_vga`（Lead 空目录重跑） | **48/48 PASS，0 FAIL** |
| `tb_top_bnn`（含 VGA 集成，147 项） | **147/147 PASS** |
| VGA 时序（TB 实测） | 640×480，像素时钟 25.000 MHz（50M÷2），HS 32.000 µs，VS 16.800 ms，**刷新率 59.5238 Hz**，消隐期 RGB=0 |
| 负控 | `-d NEGCTL` → 2 FAIL（证明 TB 有牙） |
| TB 抓到的真 bug | 字符索引仅 2bit，`SCISSORS` 第 6/7 字被截断 → 修后墨点数 2240 ✅ |

## 2. 综合/实现门禁（route 后，含 VGA）

| 门 | patch2（无 VGA） | **含 VGA** | 判据 |
|---|---|---|---|
| LUT | 4,042 | **4,405**（logic 4,184 + LUTRAM 221） | ≤5,000 ✅ |
| FF | 5,542 | **5,578** | ✅ |
| BRAM | 5.0 当量（4×RAMB36+2×RAMB18） | **5.0 当量（不变）** | ≤6.0 ✅ |
| DSP | 0 | **0** | ✅ |
| WNS | +4.136 ns | **+4.136 ns** | ≥0 ✅ |

> VGA 增量：**+365 LUT / +36 FF / 0 BRAM**（子 agent 估算 ~450 LUT，实测更低；无 Framebuffer、无行缓冲）。

## 3. 上板

| 步 | 结果 |
|---|---|
| 烧录 | `PROGRAM_OK` |
| 推理功能复测（B6 模型，5 帧） | **5/5 与模型一致**（VGA 未干扰数据通路） |
| bitstream | `ego1_bnn_vga_b6.bit`（未随包，历史版本），md5 `ce59fd35564c301c37de0f5e4f24c305`（历史版本，已于 11-02 被 cur16 取代，见交付清单 1.2） |

## 4. Lead 对三个实现决策的裁定（全部接受）

| # | 决策 | 裁定 |
|---|---|---|
| 1 | 状态字增加 `IDLE`（BUSY>ERR>DONE>IDLE 优先级） | ✅ 接受——T5 判据是"拔串口仍显示"，无引擎活动时必须有可显示状态 |
| 2 | `pix_clk_en` 的 ÷2 放在 `vga_simple` 模块内 | ✅ 接受——top 只允许加实例+连线，端口签名保持；将来若要外部使能，接 25 MHz 单周期脉冲即可 |
| 3 | `err` 只接 `sfr_status[2]`（不接 `errors_o`） | ✅ 接受——`errors_o` 只在复位清，会让 ERR 永久点亮；STATUS.bit2 才是 CPU 可见的粘滞错误位，可被 CTRL.CLR 清 |

## 5. 待目视确认（板端，需要人看显示器）

预期画面：
- 分辨率 640×480，刷新 59.52 Hz；
- 顶部：**标题彩条**（颜色顺序：白 黄 青 绿 品红 红 蓝 黑）；
- 中央：大号分类字（`IDLE` / `ROCK` / `PAPER` / `SCISSORS` / `JUNK`）；
- 下方：**4 条分数条**（长度随 SFR SCORE0..3 变化）；
- 左下：状态字（`IDLE`/`BUSY`/`DONE`/`ERR`）。

> 若颜色顺序或字形与预期不符，记录"实际 vs 预期"给我，我按位序/ROM 修正。

## 6. 复跑

```powershell
# 仿真（48 项）
powershell -NoProfile -ExecutionPolicy Bypass -File "<repo>\sim\20261101_VGA仿真\run_vga.ps1" -SkipTop
# 综合+实现+烧录（Lead 侧，~8 分钟）
cd <local>\build_top ; vivado -mode batch -source build_top_ascii.tcl -tclargs impl
vivado -mode batch -source hw_program_vga.tcl
```
