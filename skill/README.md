# skill/ 技能包：RISC-V + 零 DSP BNN FPGA 软硬协同项目

## 通用层（换题目、换板子仍可直接复用）

| 项 | 内容 | 已验证效果 |
|---|---|---|
| 单一数字源与过期数字扫描 | `tools/check_numbers.py` | 在本仓库约 50 篇、7,799 行文档上运行，过期数字从 15 处归零；可独立用于任何“文档数字必须与产物一致”的项目 |
| 引脚约束结构校验 | `tools/check_xdc.py`（与 `src/pc/check_xdc.py` 同源，修改后请同步） | `[PASS] 22 ports: structure OK`；官方 XDC 缺失时跳过官方引脚比对 |
| 敏感/凭据字符串扫描 | `tools/check_secrets.py` | 无 `.git` 时自动遍历包内文件；扫描后 `[PASS]`/`[FAIL]` |
| DSP 使用量与时序门禁 | `tools/check_dsp.py` | 读取 `build/` 的 synth/impl 报告；DSP=0 且 WNS≥0 才 PASS，缺报告判 FAIL |
| 现场标定检查清单 | `checklists/onsite_calibration_checklist.md` | 在 4 次现场重标定中复用（每次约 25 分钟），覆盖光照/机位/交错采集/门禁/录制；可复用于任何“PC 采集→训练→部署→板端自检”的视觉项目 |
| 打包前隐私门禁方法 | 本 README 的“打包自检”小节 | 用“逐字节扫描 + 逐图目视”发现并清除 `.pyc` 中的本机路径、Office 元数据与截图提示符 |

## 项目层（与 BNN/EGO1/PCPI 绑定，换题需改写）

- `src/pc/onsite_train_aug.py`：现场增广训练 + G4 门禁 + blob 导出；
- `src/pc/deploy_onsite.ps1`：只换权重重建 bitstream + 烧录 + 板端自检；
- `src/pc/board_smoke.py` / `record_board_evidence.cmd`：板端逐帧一致性回放；
- `src/pc/weights_hex.py`：从权重 blob 反解模型并校验 CRC16。

## 适用范围

- 纯 FPGA（Artix-7 等）+ RISC-V 软核 + 自定义指令加速器的软硬协同项目；
- 需要在低成本器件上部署二值/定点神经网络、并做板端一致性验证的项目；
- 需要“PC 端采集—训练/量化—导出—只换权重重建—板端自检”现场标定闭环的项目。

## 使用方法

1. 读 `checklists/onsite_calibration_checklist.md`，按清单准备场景与采集；
2. 用 `tools/check_numbers.py` 建立单一数字源（numbers.json）并扫描过期数字；
3. 复用项目层的训练/部署/回放脚本（路径见上）；
4. 换板/换题时：保留“单一数字源 + 逐帧一致性门禁 + 现场标定闭环”的方法部分，替换 RTL/协议/模型相关脚本。

## 失效条件

- 目标板无 UART 或无法从 PC 注入帧（本技能包依赖 PC→板帧协议）；
- 使用 PYNQ/Zynq 等带 ARM 的 SoC 方案（通用层可复用；PCPI/手写 RTL 部分需重写）；
- 验收口径与本文不一致（判据/帧数/数据划分必须重定，不要照搬数字）；
- 器件资源远小于本方案（<4,400 LUT/5 BRAM）时，需重新评估网络结构与资源预算。

## 打包自检（可复用）

```bash
# 1) 无编译缓存
find . \( -name "*.pyc" -o -name "__pycache__" \) -print
# 2) 逐字节扫本机路径（grep -a 可扫二进制）
grep -ra "Users\\\\" --include="*" .
# 3) 逐图目视 + 文字通读（MD5 只能证明“同一文件”，不能证明“没有人脸”）
```
