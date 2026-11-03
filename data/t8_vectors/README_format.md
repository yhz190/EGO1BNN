# T8 golden 逐层向量格式说明（2026-11-01，v1 权威版）

> 生成器：`src/pc/gen_golden_t8.py`
> 权威模型：`board/weights/E_E1_fixsplit_seed1.pt`（同域 0.9257 / 跨域 0.7016，浮点参考口径）
> **本目录替代旧向量**（旧向量如 `L2_act.hex` 来自队友 golden-hw 分支，为开发机中间产物、未随包提供；基于旧的 float-θ 语义，已作废，不得用于 T8 签核）

## 0. 语义（与硬件一致，位精确）

| 项 | 定义 |
|---|---|
| 权重二值化 | `w_bit = 1 if w_real > 0 else 0`（`w_bit=1 ↔ +1`，`0 ↔ −1`） |
| 激活 | `bit=1 ↔ +1`（亮/手部），`bit=0 ↔ −1` |
| **θ 取整** | **`θ_int = floor(θ_float)`**（因为 popcount 是整数，`count > θ_float ⟺ count > floor(θ_float)`；四舍五入会整体抬高 1，实测会错 299/620 帧） |
| L1/L2 padding | bit 域补 **0**（= ±1 域补 −1），65×65 扫描 |
| L3 | 1×1 卷积，**无 padding**，4,096 拍 |
| L4 | 4×4 块求和 ≥ 8 记 1（多数判决） |
| L5 | `score_int = 2*matches − feat + bias_int`，`bias_int = round(bias_float)`（9bit 有符号） |
| 类别 | 0=rock 1=paper 2=scissors 3=junk |

## 1. 文件

| 文件 | 内容 |
|---|---|
| `golden.npz` | 全部 620 帧的逐层结果（见 §2） |
| `report.json` | θ 表、FC 偏置、精度、自洽结论 |
| `sample_frame_{0,1,2}.hex` | 3 帧的人可读文本向量（供模块级 TB 直接读） |
| `README_format.md` | 本文件 |

## 2. golden.npz 字段

| 键 | 形状 | 含义 |
|---|---|---|
| `files` | (620,) | 帧文件名（与 `data/vectors_t1a/labels.csv` 同序） |
| `labels` | (620,) int64 | 真实类别 |
| `L1` | (620, 8, 512) uint8 | L1 激活位图，`np.packbits` **MSB-first**（`byte.bit7` = 该行第 0 列） |
| `L2` | (620, 8, 512) uint8 | L2 激活位图 |
| `L3` | (620, 4, 512) uint8 | L3 激活位图 |
| `c1` / `c2` / `c3` | (620, C, 64, 64) uint8 | 各层匹配计数（0..9 / 0..8） |
| `L4` | (620, 128) uint8 | 1024 bit 特征，packbits MSB-first（`bit0` = 特征 0） |
| `L5` | (620, 4) int16 | 整数得分 `2*matches − feat + bias_int` |
| `pred` | (620,) uint8 | 整数口径预测类别 |

解包示例：

```python
z = np.load('golden.npz')
l1 = np.unpackbits(z['L1'][0], axis=1)          # (8, 4096)
l1 = l1.reshape(8, 64, 64)                      # [channel][y][x]
l4 = np.unpackbits(z['L4'][0])[:1024]
```

## 3. 自检结论（生成时）

| 检查 | 结果 |
|---|---|
| 向量重放自洽（从 L4 重算 FC，argmax 与 `pred` 一致） | **620/620** ✅ |
| 整数口径 vs `hw_eval`（float-θ + float-bias）argmax | **599/620**（21 帧近差，见 §4，**已知且可解释**） |
| 逐层位图（L1/L2/L3/L4）与 float-θ 流水线差异 | **0**（完全一致） |
| 整数口径准确率 | **0.7113**（vs float 参考 0.7016） |

## 4. 已知差异：FC 偏置的小数部分

`hw_eval` 的 FC 打分是 `(2*matches − feat) + bias_float`，bias 含小数（本模型为 +0.95 / +0.971 量级）；
硬件用 9bit 整数 `bias_int = round(bias)`。当两类得分差 < 1 时，整数化会改变 argmax（21 帧，3.4%）。

**决策**：以**硬件整数语义为准**（这也是板端真实行为），故 T8 向量用整数 θ + 整数 bias；
浮点参考值 0.7016 仅作为对照记录，不再作为签核依据。
