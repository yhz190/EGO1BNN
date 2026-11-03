# 权重布局与导出规范 v5（gen_weights.py 的唯一依据）

> 版本：v5（2026-11-01，v2.2-patch2 + beat-major FC）｜**唯一正文口径 = 8 通道 + Q6 定点**
> 权威模型：**8 通道 ckpt**（当前 `C_l3x8_80ep_s2.pt`；最终交付 ckpt 待 G2 验证集口径重选后钉死）
> 4 通道旧版（4,448 bit）**归档为附录 A，不参与任何计算**（单一事实源原则）
> 类编码：见 §9（唯一事实源）
>
> **v5 变更（2026-11-01）**：FC 段（bit 208..8400）由 v4 的 **class-major** 改为 **beat-major**，
> 数值、字段偏移、宽度、CRC16 覆盖范围**全部不变**，只改 8,192 bit FC 的存储顺序。
> 目的：让 `bnn_wblob` 把 32bit 权重字**直推** `l5_fc` 内部 RAM，删掉 256×32 的 `fc_ram` 缓存与
> 256×4 拍反交织（该模块综合后为 **2,769 LUT / 8,788 FF**，主因就是这份缓存）。
> 证据见 `sim/20261101_beatmajor/`。

## 1. 网络结构（Net8：pool=4，L3=8 通道）

| 层 | 运算 | 输出尺寸 | 权重形状 | bit 数 |
|---|---|---|---|---|
| L1 | conv 3×3, 1→8, pad=1（bit 域补 0） | 8×64×64 | [8,1,3,3] | **72** |
| L2 | depthwise 3×3, 8→8, pad=1（±1 域补 −1 ≡ bit 0） | 8×64×64 | [8,1,3,3] | **72** |
| L3 | conv 1×1, 8→**8** | **8**×64×64 | [8,8,1,1] | **64** |
| L4 | 4×4 average-pool → 多数判决（≥8/16 记 1） | 8×16×16 → **2,048 bit** | — | 0 |
| L5 | dense **2,048**→4（含 junk 类） | 4 | [4,2048] | **8,192** |

激活约定：`w_bit = 1 if w_real > 0 else 0`；激活 `1 = 亮`；全链路只用 XNOR + popcount。

## 2. bit 流布局（总计 **8,648 bit = 271 个 32-bit 字 = 1,084 B**）

| 偏移(bit) | 字段 | 位数 | 编码顺序 |
|---|---|---|---|
| 0 | L1 权重 | 72 | `[out_ch][ky][kx]` |
| 72 | L2 权重 | 72 | `[ch][ky][kx]` |
| 144 | L3 权重 | 64 | `[out_ch][in_ch]` |
| 208 | L5/FC 权重 | 8,192 | **v5 beat-major**：`[beat q][class c][feat_in_byte j]`，见 §3.3 |
| 8,400 | θ 表 | 216 | 见 §3 |
| 8,616 | ID | 16 | 固定 `0x0001` |
| 8,632 | CRC16 | 16 | 见 §4 |

对齐：8,648 → **271 字**（8,672 bit）。

> **v4 → v5 唯一差异**：FC 段的 8,192 bit 重排（v4 为 `[class c][feat f]` → `208+2048c+f`；
> v5 为 `208+32q+8c+j`）。L1/L2/L3/θ/ID 偏移与 CRC16 覆盖范围**逐 bit 不变**；
> 因为 FC 内容参与 CRC，**CRC16 值必然改变**（v4 `0x613A` → v5 `0xB8C7`），这不是错误。

## 3. θ 表（216 bit）与 FC 偏置（Q6）

| 子段 | 通道数 | 位宽 | 偏移 | 说明 |
|---|---|---|---|---|
| θ1 | 8 | 4 | 8,400 | L1 每通道阈值（popcount 0~9） |
| θ2 | 8 | 4 | 8,432 | L2 每通道阈值（0~9） |
| θ3 | 8 | 4 | 8,464 | L3 每通道阈值（0~8） |
| θ4 | 8 | 7 | 8,496 | L4 多数判决阈值（≥8/16；7bit 余量） |
| θFC | 4 | **16（Q6 有符号）** | 8,552 | `bias_q = round(bias_float × 64)` |

### 3.1 θ 导出规则（**必须 floor**）

```
θ_int = floor(θ_float)          # 不是 round！
count > θ_float  ⟺  count > floor(θ_float)     (count 为整数)
```
> 实测：用 `round` 会把阈值整体抬高 1，导致 **299/620 帧**预测改变。
> θ 来源 `train_bnn10.theta_of(bn, w, K, pm1)`：L1（0/1 域）`θ=K/2−Σs/2+c`；L2/L3（±1 域）`θ=(K+c)/2`。

### 3.2 FC 打分与 argmax（Q6 定案，E6）

```
score_q[c] = 64 × (2·matches[c] − 2048) + θFC[c]      # 19bit 有符号
pred       = argmax(score_q)
```
> Q6 实测：6 个 ckpt × 2 个测试集，`int9` 全部 argmax 不一致（6/6），**Q6 全部一致（0/6）**。
> 相关：MARGIN SFR 同量纲 ×64（旧 24 → 1,536，字段 ≥12bit）。

### 3.3 FC 存储顺序：v5 **beat-major**（唯一口径）

```
beat q = 0..255                    (= l5_fc 的 beat 序号 / 权重地址 w_addr)
word(q) = { b3, b2, b1, b0 }       (32 bit，b3 在最高字节)
b_c     = fc_w[c][8q+7 : 8q]       (class c 的第 8q..8q+7 个特征，LSB 在右)
blob bit 208 + 32q + 8c + j = fc_w[c][8q + j]
```

即：每个 beat 字装 4 个 class 在**同一 beat**上的各 8 个特征位；字内 `bit 8c+j` = class c 的特征 `8q+j`。
FC 起点 `208 = 6*32 + 16`，因此 beat q 仍跨两个字：

```
beat_word(q) = { words[7+q][15:0], words[6+q][31:16] }      q = 0..255
```

**RTL 对应**：`bnn_wblob` 在 blob word `7+q` 到达的那一拍直接
`l5_we=1 / l5_addr=q / l5_data={d[15:0], fc_hi}`（`fc_hi` = 上一拍 word `6+q` 的 `[31:16]`），
直接写进 `l5_fc` 的 256×32 内部 RAM。**v5 不需要 `fc_ram` 缓存，也没有 256×4 拍的反交织第二遍**。
`l5_fc` 模块本身（端口、内部 RAM、语义）**一个字节都没改**；它源码注释里的 "class-major" 是 v4
时点说明，按本节 v5 输入理解即可（冻结模块，不在本次可改清单内）。

**数值自检**：`weights.hex` 反解成 `fc_w[4][2048]` 后重跑 Q6 前向，argmax 与
`train_bnn10.hw_eval_q6` 在 620 帧上 **620/620** 一致（`gen_weights.py` 内置断言 +
`src/pc/weights_hex.py`（v5 布局与 CRC16 校验脚本，已随包） 独立复核；TB 侧另用 Verilog 阵列参考复核）。

## 4. CRC16

- 多项式 **0x1021**（CRC-16/CCITT-FALSE），init **0xFFFF**，不反转，xorout 0x0000
- 覆盖：**bit 0 ~ 8,615**（ID 之前全部内容）；CRC16 自身不参与
- 存放：bit 8,632~8,647，高字节在前

## 5. 文件格式

`board/weights/hw/weights.hex`：**271 行，每行 8 个 hex（32 bit 字）**。

**位序（唯一口径，已由 `gen_weights.py` 实现并自检）**：

```
blob 位索引 i  ->  字 w = i // 32，字内位 k = i % 32（LSB 先）
hex 第 w 行    =  该 32bit 字的值（8 hex，高位在左）
```

- 8,648 bit 装入 **271 字 = 8,672 bit**，末尾 **24 bit 填充 0**；
- 各字段内低位先：L1 的 lane c / tap t → 字段内 bit `c*9+t`；L3 的 out o / in i → `o*8+i`；
  **FC（v5）→ 字段内 bit `32q+8c+j`（beat-major，见 §3.3）**；θ 按 `ch*W`；θFC 为 16bit 二进制补码；
- **CRC16 计算方式**：对 blob 前 **8,616 bit（=1,077 字节，按 LSB-first 组字节）** 做
  CRC-16/CCITT-FALSE（poly 0x1021 / init 0xFFFF），存于 bit 8,632~8,647。

`manifest.json`：

```json
{
  "layout_version": "v5",
  "bits": 8648,
  "words": 271,
  "arch": "L3=8ch, feat=2048, FC=4x2048",
  "fc_bias": "Q6 signed 16-bit",
  "fc_order": "beat-major",
  "fc_bit_map": "bit 208+32*q+8*c+j = fc_w[c][8*q+j]; beat word q = {b3,b2,b1,b0}",
  "theta": "floor",
  "crc16": "0x????",
  "model": "<final ckpt>",
  "model_md5": "<md5>",
  "accuracy_same_domain": null,
  "accuracy_cross_domain": null,
  "float_exact_agree": "n/n",
  "git_commit": "<生成时 commit>"
}
```

## 6. 自检（必须全过才算交付）

1. bit 总数 = **8,648**、行数 = **271**
2. **Q6 一致性**：反解析后用整数 Q6 前向，argmax 与 `hw_eval`（float 参考）**逐帧 100% 一致**
3. CRC16 重算一致
4. 连续两次运行输出**逐字节相同**
5. `manifest.json` 字段齐全（精度两栏按 §7 口径填写）
6. **v5 新增**：逐 beat 校验 `words[6+q][31:16]`/`words[7+q][15:0]` 拼出的字 == `{b3,b2,b1,b0}`
   （256 个 beat 全过；`gen_weights.py` 写文件后立即断言，`src/pc/weights_hex.py`（v5 布局与 CRC16 校验脚本，已随包） 从落盘文件独立复核）

## 7. 交付精度口径（G2 硬规则）

- **epoch 选择只用验证集**（训练集内切 10%，split seed 写死），或固定 epoch；
- 测试集**只对最终 ckpt 跑一次**，该数即承诺值；
- `best over test epochs` 只许出现在消融/调试记录，**验收矩阵/答辩/文档不得引用**；
- ckpt 准入门：**全程 `float == exact`**，否则不得进入交付池（G4）。

## 8. 参考实现（可直接复用，不要重写前向）

| 用途 | 位置 |
|---|---|
| 模型 `Net8(pool=4, cout3=8)` | `src/pc/train_l3x8.py` |
| θ 导出 `theta_of()` / 硬件精确前向 `hw_eval()` | `src/pc/train_bnn10.py` |
| 类编码断言 | `src/pc/class_map.py` |
| T8 向量生成 | `src/pc/gen_golden_t8.py` |
| 测试向量 | `data/t8_vectors/` |

## 9. 类编码（**单一事实源**）

> 任何其它文档不得复述下列数字，只能引用本节。

| 编号 | 名称 | 语义 |
|---|---|---|
| **0** | `rock` | 石头 |
| **1** | `paper` | 布 |
| **2** | `scissors` | 剪刀 |
| **3** | `junk` | 第 4 类（协议层旧称 `UNKNOWN`，语义等同） |

代码保护：`class_map.assert_class_map()` 在向量生成/权重导出/训练入口断言一致性。

---

## 附录 A（历史，不参与计算）：4 通道旧版

| 字段 | 位数 |
|---|---|
| L1 权重 | 72 |
| L2 权重 | 72 |
| L3 权重（4out×8in） | 32 |
| FC 权重（4×1,024） | 4,096 |
| θ1/θ2/θ3/θ4 | 32/32/16/28 |
| θFC（9bit 整数） | 36 |
| ID + CRC16 | 32 |
| **合计** | **4,448 bit = 139 字 = 556 B**（4 通道历史口径，已被 v2.2-patch2 取代） |

> 4 通道口径已于 2026-11-01 被 v2.2-patch2 取代（同域差 2.4pp、跨域约差 9pp）；
> 保留此表仅为历史追溯，**不得用于任何新导出**。
