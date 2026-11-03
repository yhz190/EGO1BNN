> 证据文件说明：`data/t8_rtl/run620.log` 为包内可核对文件；xvlog/xelab 编译日志、`run20.log`、`run_negctl.log`、`tb_t8_nframes.txt`、`20261101_make_data.log` 为开发机中间产物，未随源码包提供。

# T8 RTL 级回归记录：620 帧 golden vs RTL 逐帧一致（2026-11-01）

> **判据（验收矩阵 T8）**：与 golden **位精确，差异 0**。
> **方法**：只跑 xsim，620 帧真实向量逐帧过 `bnn_engine`，4 个 19-bit Q6 分数与 golden **逐 bit** 比对；**RTL 零改动**（11 个模块 SHA256 与项目源一致），未综合、未实现、未生成 bitstream、未用 git。
> **结论**：**PASS — 620/620 帧逐位一致（差异 0）**；首个差异：**无**。20 帧快检 20/20。负控（故意改坏 1 帧）0/1 并报出差异，证明 TB 有效。

## 1. 结果摘要

| 项 | 结果 |
|---|---|
| 全量回归 | **`[RESULT] 620/620` / `[PASS]`** |
| 每帧 4 个分数完全相等 | 620 / 620 |
| 不一致帧数 / 首个不一致帧 | **0 / 无** |
| 按类差异计数 | f0=0 f1=0 f2=0 f3=0 |
| `cls_o` == golden `pred` | 620 / 620 |
| 每帧拍数（L1..L5+done） | 620/620 帧均 16,911（与 16,911 拍预算一致） |
| 仿真时长（620 帧） | 130,246,600 ns（130.25 ms 仿真时间），**墙钟 67.6 s**，xsim CPU 62.3 s |
| 20 帧快检 | **`[RESULT] 20/20` / `[PASS]`**，墙钟 5.1 s |
| 负控（frame 0 改 1024 像素） | **0/1 FAIL**，4 类分数全不同 → TB 对数据错误敏感 |

## 2. 被测对象与只读边界

- DUT：`src/rtl/bnn/bnn_engine.v`（含 `bnn_wblob`）及其 10 个子模块
  `popcnt9 / lane_array / accum / theta_cmp / win_buf / act_bram / l3_mix / l4_pool / l5_fc / layer_seq`。
- 本次**新建**：`src/rtl/bnn/tb_t8_620.v`（SHA256 `75CEA906…D8712D1`）。
  TB 只驱动冻结端口（`rst_n/start/w_*/frame_*/th_*`）并采样 `score_o/cls_o/all_done`，**无层次化 force、无 RTL 修改**。
- 11 个 RTL 模块与项目源逐文件 SHA256 比对：**全部 SAME**（`<local>\build_sim_bnn\t8_rtl\INPUT_SHA256.txt` 为输入清单）。
- 全程未运行 synth/impl/write_bitstream（工具只用 `xvlog.bat / xelab.bat / xsim.bat`）。

## 3. 数据（先文件核对，再用）

| 文件 | 内容 | SHA256 |
|---|---|---|
| `data/t8_rtl/frame_words.hex` | 620 帧 × 128 字（512 B/帧，字 i = 字节 4i..4i+3 小端），79,360 行 | `EEF55059…C3AE687D` |
| `data/t8_rtl/scores.txt` | 620 行 `f0 f1 f2 f3 pred`（Q6，19-bit 有符号） | `B5B27120…F5790722` |
| `data/t8_rtl/tb_t8_expected.hex` | **新增**：620 行，每行 1 个 80-bit 字，供 `$readmemh` | `020000EB…42B075BF` |
| `board/weights/hw/weights.hex` | 271 字 v5 beat-major blob | `4474A4FC…D969B3482C` |
| `src/pc/make_t8_rtl_data.py` | **扩展**：生成 expected hex + 4 组自检 | `B805D05C…0D999A66` |

**`tb_t8_expected.hex` 位布局**（每帧一行，20 个 hex 字符）：

```
[79:76] pred (0..3, 4 个填充位)   [75:57] f3   [56:38] f2   [37:19] f1   [18:0] f0
```

每个分数为 19-bit 二补码，TB 用 `score_o === ex_mem[k][75:0]` 一次比完整 76 bit。

**数据自检（`20261101_make_data.log`，4 组全 `[OK]`，脚本退出码 0）**

1. `frame_words.hex` 79,360 字，逐帧回读与 `data/vectors_t1a/*.bin` 的 620×512 B **完全相等**（不匹配字数 0）；
2. `scores.txt` 620 行；2,480 个分数全部落在 19-bit 有符号区间；`pred == argmax(f0..f3)`（并列取低索引，与 `cls_o` 同规则）且 `f == golden.npz L5` **620/620**；expected hex 回读 **620/620**；
3. `weights.hex` 271 字；blob bits [0,8616) 的 CRC16（CCITT-FALSE，poly 0x1021，init 0xFFFF）= **0xB8C7**，与 v5 规范/manifest 一致；ID 字段 = 0x0001；
4. 用 `golden.npz` 的 L4 位 + `weights.hex` 的 FC 域**独立重算** Q6 分数（popcount 模型）== `scores.txt` **620/620**，Q6 bias = [-1, 0, 0, 2]。

> 说明：`data/t8_rtl/README.txt` 旧版把 CRC16 写成 `0x613A`（陈旧），本次按实测改为 **0xB8C7**；`frame_words.hex` 与 `scores.txt` 重新生成后 SHA256 与改前**逐字节相同**（确定性可复现）。

## 4. TB 设计（`tb_t8_620.v`）

1. **时钟/复位**：100 MHz（10 ns），异步低有效复位 8 拍。
2. **权重装载**：`$readmemh("weights.hex")` → 271 字在 **271 个连续周期** 内以 `w_we=1, w_addr=0..270` 喂入（`bnn_wblob` 要求连续），随后 36 拍 theta push；`ready=1` 出现在 **3.18 µs（约 307 拍）**，之后才允许 `start`。
3. **帧写入**：每帧 4096 bit，1 bit/拍写入 **FRAME bank 0**（`frame_bank=0`），写完后 `frame_rd_bank=0` 并给 1 拍 `start`。位序与 `preprocess.py` 的 `np.packbits`（MSB first）及 `top_bnn.v` 的 bit 串行器一致：

   ```
   word = fw_mem[frame*128 + (p>>5)]       // p = 像素号 0..4095
   bit  = (word >> (8*((p>>3)&3) + (7-(p&7)))) & 1
   ```
4. **运行与判据**：`start` 后等 `all_done`（1 拍，此时 `score_o` 已稳定），
   - `score_o[75:0] === ex_mem[k][75:0]` → 帧通过（逐 bit，含符号位）；
   - `cls_o === ex_mem[k][78:76]` → 附加检查（argmax 与 golden pred）；
   - 统计每帧拍数（应为 16,911），不一致时打印帧号、期望/实际/xor 及 4 个十进制分数（最多 10 条）。
5. **快检帧数**：参数 `NFRAMES=620`；运行期可用 `tb_t8_nframes.txt`（单个整数）覆盖。
   > 不用 `+NFRAMES=v` 的原因：Vivado 2022.2 的 `.bat` loader 会把参数里的 `=` 拆成两个 token（实测 `-testplusarg NFRAMES=20` → `ERROR: Expected a switch but found 2`），文件覆盖无此问题。

## 5. 编译与运行（全新空目录 `<local>\build_sim_bnn\t8_rtl\`）

该目录 2026-11-01 15:04 创建（创建时 0 项），拷入 11 个 RTL、TB、`weights.hex / frame_words.hex / tb_t8_expected.hex` 后编译：

```bat
set XBIN=<local>\Xilinx\Vivado\2022.2\bin
cd /d <local>\build_sim_bnn\t8_rtl

%XBIN%\xvlog.bat popcnt9.v lane_array.v accum.v theta_cmp.v win_buf.v act_bram.v ^
                 l3_mix.v l4_pool.v l5_fc.v layer_seq.v bnn_engine.v tb_t8_620.v
%XBIN%\xelab.bat -L unisims_ver tb_t8_620 -s tb_t8_620

%XBIN%\xsim.bat tb_t8_620 -R                       :: 620 帧全量
echo 20 > tb_t8_nframes.txt
%XBIN%\xsim.bat tb_t8_620 -R                       :: 20 帧快检
```

实测：`xvlog` 1.1 s、`xelab` 2.1 s（Vivado Simulator v2022.2，30 线程）；全量 `xsim` **67.6 s 墙钟**。
日志已留存在 `data/t8_rtl/logs/`：`xvlog.log / xelab.log / xvlog_stdout.log / xelab_stdout.log / run620.log / run20.log / run_negctl.log / INPUT_SHA256.txt`。

## 6. 结果

### 6.1 620 帧全量（`data/t8_rtl/run620.log`）

```
[INFO] weight blob loaded (271 words + theta push), ready=1 at 3180000
[INFO] 50/620 frames done: score-exact 50, mismatch 0, t=10506700000
...
[INFO] 620/620 frames done: score-exact 620, mismatch 0, t=130246600000
================================================================
[INFO] frames run          : 620
[INFO] score-exact frames  : 620
[INFO] mismatched frames   : 0
[INFO] per-class mismatches : f0=0 f1=0 f2=0 f3=0
[INFO] cls_o == pred        : 620/620
[INFO] frame latency        : 16911..16911 cycles (expect 16911)
[RESULT] 620/620
[PASS] T8 RTL regression: 620/620 frames bit-exact
```

> `t=` 为 `%t`（1 ps 分辨率），130,246,600,000 ps = 130.25 ms；每帧 210.07 µs = **21,007 拍** = 4,096 拍写帧 + 16,911 拍推理（无额外空拍）。
> 全量日志第 11 行的 `WARNING: file tb_t8_nframes.txt could not be opened` 是**预期**的：该文件只是可选快检覆盖，缺失时默认跑满 620 帧。

### 6.2 20 帧快检（`logs/run20.log`）

```
[INFO] tb_t8_nframes.txt: fscanf ret=1 value=20
[INFO] tb_t8_620: 20 frame(s)  (NFRAMES=620)
[INFO] 20/20 frames done: score-exact 20, mismatch 0, t=4204600000
[RESULT] 20/20
[PASS] T8 RTL regression: 20/20 frames bit-exact
```

### 6.3 负控（证明 TB 不是空转）`logs/run_negctl.log`

把 frame 0 的前 32 个字（1,024 个像素）取反后跑 1 帧：

```
[FAIL] frame 0 (latency 16911): score_o=fe905fda03fd800037f expected=fee05ffc0004c0006ff xor=0070002603f94000580
       exp f0..f3 = 1791 2432 -256 -2302   pred=1
       got f0..f3 = 895 -1280 -2432 -2942   cls_o=0
[RESULT] 0/1
[FAIL] T8 RTL regression: 1/1 frames differ (first 0)
```

负控目录 `<local>\build_sim_bnn\t8_rtl_negctl\`（与正式目录同源，仅改 `frame_words.hex` 前 32 行 + 1 帧配置），**不属于验收结论**，仅作 TB 灵敏度自检。

### 6.4 首个差异分析

**无差异**（620/620 逐 bit 相等），无需按任务 4 做数据顺序 vs RTL 问题定位。

## 7. 耗时

| 阶段 | 实测 |
|---|---|
| xvlog（12 个 .v） | 1.1 s |
| xelab + snapshot | 2.1 s |
| xsim 620 帧 | **67.6 s 墙钟**（xsim CPU 62.3 s；130.25 ms 仿真时间） |
| xsim 20 帧 | 5.1 s 墙钟 |
| 数据生成 + 4 组自检 | < 5 s（numpy） |

远低于 10 分钟预算，无需裁剪帧数；`--frames` 能力仍保留（`tb_t8_nframes.txt`）。

## 8. 复现

1. 运行 `make_t8_rtl_data.py`（需 numpy）重新生成并自检 4 个数据文件（应打印 `[RESULT] ALL DATA CHECKS PASS`）；
2. 新建空目录，按 §5 拷入 11 个 RTL + `tb_t8_620.v` + 3 个 hex，执行 §5 命令；
3. 期望：全量打印 `[RESULT] 620/620` 与 `[PASS]`；20 帧打印 `[RESULT] 20/20` 与 `[PASS]`；每帧 `frame latency 16911`。
4. 输入校验：与 `logs/INPUT_SHA256.txt` 逐项比对（TB `75CEA906…`、`frame_words.hex EEF55059…`、`tb_t8_expected.hex 020000EB…`、`weights.hex 4474A4FC…`）。

## 9. 未覆盖 / 边界（明确声明）

- **不涉及**综合、时序、资源、比特流、上板；本记录只证明 `bnn_engine` RTL 功能与 golden 一致。
- 本 TB 固定写 **FRAME bank 0** 并固定 `frame_rd_bank=0`；D5 双缓冲的"A 帧运行中写 B 帧"行为不在本 TB 范围（由 `tb_top_bnn.v` 覆盖）。
- 不覆盖 UART 采集链路、`top_bnn`/PCPI、CPU 交互（T4a/T4b/T11 各自覆盖）。
- 620 帧全部来自 `vectors_t1a`（与板端 620 帧同一批），未含训练集外新样本。
- `NFRAMES` 的运行期覆盖走文件而非命令行（原因见 §4.5）。

## 10. 对验收矩阵 T8 的意义

验收矩阵 T8 行此前为"620 帧 RTL 级回归 TB 未建"。本记录补齐最后一块：

- **TB 已建**：`src/rtl/bnn/tb_t8_620.v`（可复现、数据自检、负控有效）；
- **判据达成**：620/620 帧 4×19-bit Q6 分数**逐 bit 相等，差异 0**；附加 `cls_o` 620/620；
- **建议**：T8 行状态由 🟨 更新为 ✅，证据指向本文件与 `data/t8_rtl/run620.log`。
