# 工具说明（`src/pc`，PC 端，不依赖 Vivado）

主链路工具按用途分组；双击入口为三个 `.cmd`。

## 一键入口

| 文件 | 用途 |
|---|---|
| `start_capture.cmd` | 打开采集工具（`capture_live.py`）：选类、拍帧、环境锁定/域探针 |
| `start_demo_rec.cmd` | 打开演示窗口（`demo_live.py`）：实时识别、目标框、3 帧滞回、AE 锁定 |
| `record_board_evidence.cmd` | 板端证据回放：371 帧 `371/371` 与模型一致，约 58 fps |

## 训练与权重

| 文件 | 用途 |
|---|---|
| `train_bnn10.py` | 训练主脚本（含 `hw_eval_q6` 定点评估） |
| `train_l3x8.py` | 8 通道 L3 网络（最终结构）与 `hw_eval8` |
| `train_fix.py` | 训练数据构造/定点化辅助 |
| `train_formal_g2.py` | G2 正式训练口径 |
| `gen_weights.py` | 导出 v5 权重 blob（`--ckpt/--outdir`），带 CRC16 与反解自检 |
| `onsite_train.py` | 现场标定（时间窗 → 切分 → 训练 → G4 → 导出 blob） |
| `onsite_train_aug.py` | 现场标定 + 手部尺度/位置增广（推荐） |
| `video_adapt_train.py` | 演示域适配：录屏→去叠加层→伪标签→微调 |
| `deploy_onsite.ps1` | 只换权重重建 bitstream → 烧录 → 板端自检 |

## 采集与预处理

| 文件 | 用途 |
|---|---|
| `capture_live.py` | 采集工具（tkinter）：分类拍摄、`B` 空背景、`S` 锁环境、`A` 连拍 |
| `preprocess.py` | 核心预处理（中心裁方 64×64 + Otsu），训练/部署必须一致 |
| `preprocess2.py` / `preprocess3.py` / `preprocess4.py` | 备选预处理（bbox/背景差分/曝光归一化），按需选用 |
| `env_stats.py` | 环境指纹（背景亮度/纹理/光照对比/边暗区）+ 参考锁定判定 |
| `validate_session.py` | 采集批次合规检查（数量/背景/出框/距离覆盖/环境一致率） |

## 板端 / 协议

| 文件 | 用途 |
|---|---|
| `board_smoke.py` | 板端逐帧回放比对（`RESULT: N/N frames match the model exactly`） |
| `uart_inject.py` | 批量注入 + 回读，落 CSV（T1a/T4a 数据源） |
| `t4b_stream.py` | 预打包流式注入（T4b 吞吐判据） |
| `demo_live.py` | 实时演示管线（相机→512B→板→结果叠字） |

## 数字与检查

| 文件 | 用途 |
|---|---|
| `gen_numbers.py` / `check_numbers.py` | 唯一数字来源生成 / 全文档过期数字扫描 |
| `check_dsp.py` / `check_xdc.py` / `check_secrets.py` | DSP=0 门禁 / 引脚约束校验 / 敏感文件扫描 |
| `class_map.py` | 类别编码唯一来源（0=rock 1=paper 2=scissors 3=junk） |

## 仿真 / golden

| 文件 | 用途 |
|---|---|
| `golden.py` / `golden_hw.py` | Python golden 前向（逐层）与硬件定点复现 |
| `gen_golden_t8_v2.py` / `make_t8_rtl_data.py` | T8 回归向量生成 |
| `hwsim_gate.py` / `verify_q6_chain.py` / `eval_exact.py` | 仿真门/定点链路校验 |
| `export_l1_signoff.py` / `export_layer_signoff.py` | 逐层签核数据导出 |

## 环境

- Python：`.venv\Scripts\python.exe`（numpy / torch CPU / pyserial / Pillow / opencv）
- 依赖：`pip install pyserial`（仅串口注入需要）
- Windows 下 `python` 无反应时用完整 venv 路径

## 协议约定（回环验证用）

- 下行：帧头 `A5 10 00 02` + 512 B payload + **CRC8（poly 0x07）**，共 517 B；
- 上行：`A5 83 cls s0 s1 s2 flags fps seq CRC8`，共 10 B；
- `uart_inject.py` 常用参数：`--dir --csv --dry-run --classes --baud`；
- `board_smoke.py` 在 `--ckpt` 缺失时自动回退到入库的权重 blob（`board/weights/hw_adapt_20261102/weights.hex`），
  证据命令可只用 git 仓库复现。
