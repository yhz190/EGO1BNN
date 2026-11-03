> 证据文件说明：`data/live/_manifest.csv` 等现场采集清单为开发机数据，未随源码包提供；本记录的方法与脚本（`check_numbers.py`）在包内。

# T10：唯一数字来源 `numbers.json`（建立记录）

<!-- numbers-check: ignore-file —— 本文件本身就是旧值清单，check_numbers.py 扫到该标记会整篇跳过（见 §5.1 规则 6） -->

> 日期：2026-11-01 18:27（Asia/Shanghai）｜任务：T10「双赛道一致 / 数字单一来源」
> 边界：只新建/修改 4 个文件（§1）；**未改任何 `.v` / Python 源码 / RTL / 权重**，**未使用 git**（仅以只读文件方式读 `.git/HEAD` 与 `refs/tags`）。
>
> **结论**：`numbers.json`（50 条现行值 + 9 条旧值黑名单）+ 生成器 + 检查器已就绪，JSON 内部自洽 **11/11**；
> `check_numbers.py` 当前 **FAIL**：A 类现行文档 **9 行 / 21 个过期值**，B 类记录文档 **13 行 / 15 个过期值**
> （合计 **22 行 / 36 个过期值**）。
> **所有过期数字均未自行修改**（改别的文档属 L2 且 T10 不是改口径的授权），清单见 §6，交 Lead 处置。

---

## 1. 产物清单

| 文件 | 作用 | 生成方式 |
|---|---|---|
| `report/design_notes/numbers.json` | **唯一数字来源**（UTF-8、`indent=2`、每条叶子带 `source`） | `gen_numbers.py` 生成，**禁止手改** |
| `src/pc/gen_numbers.py` | 从权威产物自动取值 → 写 `numbers.json`；派生量自检 | 新增 |
| `src/pc/check_numbers.py` | 扫 `report/design_notes/*.md` + `sim/*.md`，只报过期数字 | 新增 |
| `sim/t10_single_source_of_truth.md` | 本文件（结构说明 + 来源 + 真实运行输出） | 新增 |

未触碰：`src/rtl/**`、`src/firmware/**`、`board/weights/**`、`report/design_notes/09_…v2.2…md` 及其它任何 `.md`。

---

## 2. `numbers.json` 结构

### 2.1 顶层键

| 键 | 内容 |
|---|---|
| `schema` | `ego1-bnn/numbers@1`（版本化，便于以后加字段） |
| `generated_at` | 生成时刻（+08:00） |
| `generator` | 生成脚本相对路径 |
| `policy` | 一句话纪律：文档只能引用现行值，旧值见 `superseded` |
| `t10_key8` | T10 判据「8 个关键数字」的 8 个键路径（见 §2.4） |
| `project` | 基线文档、git HEAD、tag 列表、最新 tag（只读文件解析） |
| `performance` | 性能：拍数 / 时钟 / 帧时 / 三个 fps 口径 |
| `resource` | 资源：LUT / FF / BRAM / DSP / WNS；含 `patch2_no_vga` 子项 |
| `model` | 精度：T1a / T1b / 部署 seed / RPS 诊断 / 板端一致性与准确率 |
| `protocol` | 协议：帧长 / CRC8 多项式 / blob 字数位数 / 三个 CRC16 |
| `acceptance` | 验收：TB 项数 / T8 帧数 / T4b 判据 |
| `data` | 数据：现场采集帧数（快照）/ 类分布 / `class_map` |
| `superseded` | **旧值黑名单**（9 条），每条含 `replaced_by` 与 `replace_with` |

### 2.2 叶子格式（所有数字统一）

```json
"lut_total": {
  "value": 4405,                 // 值（数字或 "180/180" 这类比值串）
  "unit": "LUT",                 // 单位（可省略）
  "source": "sim/vga_implementation.md#§2（含 VGA，route 后）",
  "auto": false,                 // true=程序从产物读取；false=人工常量（但有文档锚点）
  "note": "= lut_logic 4,184 + lut_ram 221；判据 ≤5,000"
}
```

- `status: "superseded"` + `superseded_by`：仅用于 `resource.patch2_no_vga` 三个子项（4042 / 5542 / 8.375）。
  检查器**不把这类叶子当现行值**，而把它们并入过期黑名单。
- `volatile: true`：仅 `data.live_frames_total` —— 采集仍在追加，它是**快照**而非稳定值。

### 2.3 现行值 vs 旧值

- **现行值** = 除 `superseded` 段与 `status=superseded` 叶子之外的全部叶子（50 条）。
- **旧值** = `superseded.items` 9 条：

| 旧值 | 含义 | 应为 | 对应键 |
|---|---|---|---|
| 19,410 | patch1 每帧拍数 | 16,911 | `performance.engine_cycles` |
| 2,576 | patch1 引擎 fps | 2,957 | `performance.fps_engine` |
| 1,244 | v2.1 权重表 bit 数 | 8,648 | `protocol.blob_bits` |
| 4,448 | 4 通道权重表 bit 数 | 8,648 | `protocol.blob_bits` |
| 8,693 | 优化前整机 Slice LUTs | 4,405 | `resource.lut_total` |
| 3,821 | patch2 无 VGA 的 Logic LUT | 4,184 | `resource.lut_logic` |
| 4,042 | patch2 无 VGA 的 LUT 总量 | 4,405 | `resource.lut_total` |
| 5,542 | patch2 无 VGA 的 FF | 5,578 | `resource.ff` |
| 8.375 | patch2 无 VGA 的 WNS | 4.136 | `resource.wns_ns` |

> `superseded.allow_coexist = [59.55, 180.56]`：这两个帧率**分属 T4a / T4b 两个口径**，允许并存，不是新旧关系。

### 2.4 T10 八个关键数字

| # | 键 | 值 | 口径 |
|---|---|---|---|
| 1 | `performance.engine_cycles` | 16,911 拍 | [xsim] 与 [route] 同值 |
| 2 | `performance.fps_engine` | 2,957 fps | 16,911 @50 MHz 独占 |
| 3 | `resource.lut_total` | 4,405 LUT | 含 VGA route 后 |
| 4 | `resource.ff` | 5,578 FF | 含 VGA route 后 |
| 5 | `model.t1a` | 96.07% ±0.47 | G2 正式口径，3 有效 seed |
| 6 | `model.t1b` | 98.89% ±0.56 | B6 同环境留出 180 帧 |
| 7 | `model.board_acc` | 99.44% | 板端 179/180 |
| 8 | `protocol.downlink_bytes` | 517 B | 1+1+2+512+1；CRC8 覆盖全帧 |

---

## 3. 数据来源：自动 vs 文档常量

### 3.1 自动读取（`auto: true`，每次生成都重读权威产物）

| 键 | 自动来源 | 说明 |
|---|---|---|
| `protocol.blob_words` / `blob_bits` | `board/weights/hw_b6/manifest.json` | 271 字 / 8,648 bit |
| `protocol.blob_crc16_b6` | `board/weights/hw_b6/manifest.json#crc16` | **0x3569**（当前部署 blob） |
| `protocol.blob_crc16_formal` | `board/weights/hw/manifest.json#crc16` | 0xB8C7（FORMAL 权重） |
| `protocol.blob_crc16_b5` | `board/weights/hw_b5/manifest.json#crc16` | 0x020B（B5 会话） |
| `model.rps_vectors_diag` | `hw_b6/manifest.json#acc_q6_on_vectors` | 原始 0.7790322 → 记 0.779（仅诊断） |
| `data.live_frames_total` | `data/live/_manifest.csv` **行数** | 按 manifest 行数记（含 271 个重复文件名） |
| `data.live_frames_unique` | 同上，唯一文件名 | 2,681（重复 271，跨会话重名） |
| `data.live_class_counts` | 同上，`class` 列 | junk 835 / paper 795 / rock 765 / scissors 557 |
| `data.class_map` | `src/pc/class_map.py#CLASS_MAP` | 解析源码并校验顺序 = rock/paper/scissors/junk |
| `project.git_*` | `.git/HEAD`、`.git/refs/tags` | **纯文件读取，不调用 git 命令**；最新 tag `v2.2-demo-b6` |

### 3.2 文档常量（`auto: false`，每条带 `source = 路径#章节`）

> 下表 `09_…v2.2…md` 指 `report/design_notes/v2.2_baseline_spec.md`（基线冻结版，唯一权威方案）；其余同理省略前缀目录。

| 键 | 值 | source |
|---|---|---|
| `performance.engine_cycles` | 16,911 | `09_…v2.2…md#§6.3` |
| `performance.freq_mhz` | 50 | `09_…v2.2…md#§3.1 与 §15.B` |
| `performance.frame_us` | 338.2 | `09_…v2.2…md#§6.3` |
| `performance.fps_engine` | 2,957 | `09_…v2.2…md#§6.3` |
| `performance.fps_end_to_end` | 59.55 | `t4b_stream_injection.md#§1（T4a）` |
| `performance.fps_streaming` | 180.56 | `t4b_stream_injection.md#§2（T4b）` |
| `resource.lut_total / lut_logic / lut_ram` | 4,405 / 4,184 / 221 | `vga_implementation.md#§2` |
| `resource.ff / bram_equiv / dsp / wns_ns` | 5,578 / 5.0 / 0 / 4.136 | 同上 |
| `resource.patch2_no_vga.*` | 4,042 / 5,542 / 8.375 | 同上，对照列（superseded 子项） |
| `model.t1a / t1a_sigma` | 96.07 / 0.47 | `20261101_G2正式结果与域适配.md#§1` |
| `model.t1b / t1b_sigma / t1b_frames / t1b_seeds` | 98.89 / 0.56 / 180 / 2 | `b6_calibration_final_deploy.md#§1` |
| `model.t1b_deployed_s1 / board_match / board_acc` | 99.44 / 180/180 / 99.44 | `b6_calibration_final_deploy.md#§3` |
| `protocol.downlink_bytes / uplink_bytes` | 517 / 10 | `09_…md#§7.1` + `18_协议细节确认_CRC覆盖范围.md` |
| `protocol.crc8_poly` | 0x07 | `09_…md#§7.1` |
| `acceptance.checks_pipeline` | 115 | `20261101_patch2_RTL适配记录.md#§1/§4` |
| `acceptance.checks_top` | 147 | `vga_implementation.md#§1`、`top_bnn集成记录#§14.3` |
| `acceptance.t8_frames` | 620 | `t8_rtl_regression.md#§1` |
| `acceptance.t4b_criterion` | 150 | `09_…md#§1.3`（T4b 判据，未改） |

> 说明：`18_协议细节确认_CRC覆盖范围.md` 给出**下行 517 B / 上行 10 B** 与「CRC8 覆盖帧头~末字节」，
> `09_…md#§7.1` 原来只写多项式 0x07、没写覆盖范围 —— numbers.json 以 18 号文为帧长/覆盖口径。

---

## 4. `gen_numbers.py`

```powershell
cd <repo>\src/pc
..\.venv\Scripts\python.exe gen_numbers.py
```

生成时的自检（任一不过只告警不中断，除非权威产物缺失 → 退出 2）：

1. `engine_cycles / freq_mhz` 与 `frame_us` 差 ≤0.1 µs（16,911/50 = 338.22 vs 338.2）；
2. `1e6/frame_us` 与 `fps_engine` 差 ≤5 fps（2,956.6 vs 2,957）；
3. `blob_words × 32 ≥ blob_bits`（271×32 = 8,672 ≥ 8,648，对齐 1,084 B）；
4. 下行帧长 = 512 载荷 + 5 头/CRC = 517；上行 = 10；
5. 三个 `manifest.json` 必含 `bits/words/crc16/model/acc_q6_on_vectors`；
6. `class_map.py` 的 `CLASS_MAP` 必须是 rock/paper/scissors/junk，否则退出 2。

**实际输出（18:27）**

```text
[OK] 自动来源：manifest×3 / live CSV / git tags×6 / class_map.py
[OK] 现行叶子 50 条，旧值 9 条
[OK] live_frames_total=2952（唯一 2681，重复 271） 快照 2026-11-01T18:26:50+08:00
[OK] blob: 8648 bit / 271 字；CRC16 B6=0x3569
[OK] git tag latest = v2.2-demo-b6（head ref: refs/heads/master）

[OK] 写出 report/design_notes/numbers.json（15,609 B，UTF-8 indent=2）
     生成时间 2026-11-01T18:27:57+08:00
[OK] 派生量自检全部通过
```

---

## 5. `check_numbers.py`

### 5.1 判定模型（只报过期数字）

1. **先遮罩再取数**：先遮掉十六进制（`0xB8C7`）、日期（`2026-11-01`/`20261101`）、版本（`v2.2`）、
   章节号（`§6.3`）、文件名（`…md`）、带数字标识符（`L1`/`B6`/`COM34`）→ 再匹配整数/百分比/小数。
2. **命中现行值** → 正常（本次共 2,472 处命中，说明文档大量引用的是新值）。
3. **命中过期值**（§2.3 的 9 条）→ 再判三种豁免：
   - **(a) 同行已有新值**：如 VGA 记录表「patch2（无 VGA）4,042 ｜ **4,405**」，属前后对照，不报；
   - **(b) 行内含历史上下文**：历史/旧版/已被/归档/提案/审查/`v2.1`/`patch1`/无 VGA/4 通道/`→`/`vs` 等；
   - **(c) 整篇历史文档**：文件名含 `v2.0`/`v2.1`/提案/审查/评估/答复/复盘/`20261031_`/容量实验/骨架记录 等
     （本次 29 篇，`--all` 会打印完整名单）。
4. **既不在现行值、也不在黑名单**的数字（如早期估算 3,560 / 4,900）→ **不算错**，不报。
5. **文档分级**（便于处置，不影响判定）：
   - **A 现行文档**：文件名含 进度/总表/验收矩阵/计划/标定/部署/上板 —— 状态类，**必须改**；
   - **B 记录类文档**：其它（实验/集成/吞吐/验收记录）—— 是历史实测，**建议加注「patch2 无 VGA 快照」，不要改数**。
6. **自述文档跳过**：文件前 5 行含 `<!-- numbers-check: ignore-file -->` 标记的整篇跳过
   —— 本报告要列旧值清单，否则会「自己报自己」；这是唯一豁免整个非历史文档的机制（见文件头）。
7. **附加检查**（同样计入 PASS/FAIL）：
   - `numbers.json` 内部自洽 11 项（LUT 加法、拍数/时钟/帧时、帧率、517=512+5、blob 对齐、
     179/180=99.44、T6 四门限、T4b 实测≥判据、T1a/T1b 红线、T10 八键齐全）；
   - 权威产物漂移：三个 `manifest.json` 的 CRC16 与 JSON 不一致 → `[WARN]`；
   - 易变源：`live/_manifest.csv` 行数与快照不一致 → `[INFO]`，**仅 `--all` 显示**
     （采集仍在追加，属正常，不影响 PASS/FAIL）。

### 5.2 用法与退出码

```powershell
cd <repo>\src/pc
..\.venv\Scripts\python.exe check_numbers.py          # 只报过期数字（默认）
..\.venv\Scripts\python.exe check_numbers.py --all    # 连被豁免的历史引用一起列出
..\.venv\Scripts\python.exe check_numbers.py --quiet  # 只打 RESULT 行（给 CI/门禁用）
```

退出码：`0 = PASS`，`1 = FAIL`（有 A/B 过期值或内部自洽失败）。
也可从项目根运行：`& '.venv\Scripts\python.exe' 'src/pc\check_numbers.py'`（脚本按 `__file__` 定位根目录）。

### 5.3 当前扫描结果（真实输出，18:27）

```text
==== T10 check_numbers ====
numbers.json : report/design_notes/numbers.json（schema=ego1-bnn/numbers@1，generated 2026-11-01T18:27:57+08:00）
现行值       : 46 条｜过期黑名单 9 条｜文档中命中现行值 2472 处
扫描         : report/design_notes/*.md + sim/*.md = 50 篇，7,799 行（豁免历史文档 29 篇 + 自述文档 1 篇）
内部自洽     : 11/11 OK

【A】现行文档（状态/进度/标定类，必须改）：9 行 / 21 个过期值
  report/design_notes/20_进度与执行计划_20261101.md:138
    4042 → 应为 4,405（resource.lut_total）；5542 → 应为 5,578（resource.ff）；8.375 → 应为 4.136（resource.wns_ns）
  report/design_notes/22_D4引擎集成计划.md:89
    4042 → 应为 4,405（resource.lut_total）；5542 → 应为 5,578（resource.ff）；8.375 → 应为 4.136（resource.wns_ns）
  report/design_notes/28_项目进度总表与剩余计划_20261101.md:9
    4042 → 应为 4,405（resource.lut_total）；8.375 → 应为 4.136（resource.wns_ns）
  report/design_notes/28_项目进度总表与剩余计划_20261101.md:23
    4042 → 应为 4,405（resource.lut_total）；5542 → 应为 5,578（resource.ff）
  report/design_notes/28_项目进度总表与剩余计划_20261101.md:24
    8.375 → 应为 4.136（resource.wns_ns）
  sim/b6_calibration_final_deploy.md:32
    4042 → 应为 4,405（resource.lut_total）；5542 → 应为 5,578（resource.ff）；8.375 → 应为 4.136（resource.wns_ns）
  sim/20261101_T1b同环境标定与部署.md:54
    4042 → 应为 4,405（resource.lut_total）；5542 → 应为 5,578（resource.ff）；8.375 → 应为 4.136（resource.wns_ns）
  sim/records/acceptance_matrix.md:19
    4042 → 应为 4,405（resource.lut_total）；5542 → 应为 5,578（resource.ff）；8.375 → 应为 4.136（resource.wns_ns）
  sim/records/acceptance_matrix.md:96
    4042 → 应为 4,405（resource.lut_total）

【B】记录类文档（历史实测，建议加注'patch2 无 VGA 快照'，勿改数）：13 行 / 15 个过期值
  report/design_notes/26_patch2执行报告_待签核.md:23
    19410 → 应为 16,911（performance.engine_cycles）
  report/design_notes/26_patch2执行报告_待签核.md:117
    4042 → 应为 4,405（resource.lut_total）
  report/design_notes/26_patch2执行报告_待签核.md:121
    4042 → 应为 4,405（resource.lut_total）；3821 → 应为 4,184（resource.lut_logic）
  report/design_notes/26_patch2执行报告_待签核.md:138
    4042 → 应为 4,405（resource.lut_total）
  sim/20261101_top_bnn集成记录.md:722
    8693 → 应为 4,405（resource.lut_total）
  sim/20261101_top_bnn集成记录.md:724
    8693 → 应为 4,405（resource.lut_total）
  sim/20261101_top_bnn集成记录.md:730
    8693 → 应为 4,405（resource.lut_total）
  sim/20261101_top_bnn集成记录.md:742
    8693 → 应为 4,405（resource.lut_total）
  sim/20261101_top_bnn集成记录.md:743
    8693 → 应为 4,405（resource.lut_total）
  sim/20261101_吞吐升级与数据二期结论.md:19
    4042 → 应为 4,405（resource.lut_total）；3821 → 应为 4,184（resource.lut_logic）
  sim/20261101_吞吐升级与数据二期结论.md:20
    5542 → 应为 5,578（resource.ff）
  sim/20261101_吞吐升级与数据二期结论.md:23
    8.375 → 应为 4.136（resource.wns_ns）
  sim/20261101_板端验收记录_patch2.md:14
    5542 → 应为 5,578（resource.ff）

豁免：历史文档 29 篇｜自述文档 1 篇｜同行含新值 13 处｜行级历史上下文 13 处

JSON key 名直查 0 次（文档目前按值引用，未按 key 名引用）

RESULT: FAIL  （A 现行 21 个/9 行，B 记录 15 个/13 行，内部自洽失败 0 项）
```

> 说明：`现行值 46 条` 是**参与文档比对的**数值叶子数 = 全部现行叶子 50 条 − `project` 段 4 条（git/baseline，与数字比对无关）；
> `gen_numbers.py` 打印的 `50 条` 是全部现行叶子（含 project / class_map / mtime）。

---

## 6. 过期数字清单（**待 Lead 修文档**）

> 本节数字与 §5.3 完全同源。**T10 不改其它文档**：A 类由 Lead 决定改数或加「VGA 前」限定；
> B 类是历史实测记录，按 N3 可追溯原则**不应改数**，建议在文档首行加「patch2 无 VGA 快照」限定。

### 6.1 A 类：现行文档必须改（9 行 / 21 值）

| # | 文件:行 | 过期值 → 应为 | 备注 |
|---|---|---|---|
| 1 | `report/design_notes/20_进度与执行计划_20261101.md:138` | 4,042→4,405；5,542→5,578；8.375→4.136 | G5 门禁行，仍是 VGA 前数字 |
| 2 | `report/design_notes/22_D4引擎集成计划.md:89` | 4,042→4,405；5,542→5,578；8.375→4.136 | 资源/时序结论行 |
| 3 | `report/design_notes/28_项目进度总表与剩余计划_20261101.md:9` | 4,042→4,405；8.375→4.136 | 总表首段 |
| 4 | `report/design_notes/28_项目进度总表与剩余计划_20261101.md:23` | 4,042→4,405；5,542→5,578 | 资源行 |
| 5 | `report/design_notes/28_项目进度总表与剩余计划_20261101.md:24` | 8.375→4.136 | 时序行 |
| 6 | `sim/b6_calibration_final_deploy.md:32` | 4,042→4,405；5,542→5,578；8.375→4.136 | 部署记录写「门禁不变」→ 易被当成当前值 |
| 7 | `sim/20261101_T1b同环境标定与部署.md:54` | 4,042→4,405；5,542→5,578；8.375→4.136 | B5 会话重建记录 |
| 8 | `sim/records/acceptance_matrix.md:19` | 4,042→4,405；5,542→5,578；8.375→4.136 | **T6 行与同表 T5 行（4,405/5,578/4.136）自相矛盾**，最需修 |
| 9 | `sim/records/acceptance_matrix.md:96` | 4,042→4,405 | T6 历史对比行 |

### 6.2 B 类：记录类文档（建议加注，**勿改数**）（13 行 / 15 值）

| # | 文件:行 | 过期值 | 处置建议 |
|---|---|---|---|
| 1-4 | `report/design_notes/26_patch2执行报告_待签核.md:23/117/121/138` | 19,410 / 4,042 / 3,821 | 该文是 patch2（无 VGA）执行报告，建议标题加「VGA 前」 |
| 5-9 | `sim/20261101_top_bnn集成记录.md:722/724/730/742/743` | 8,693 | §14.4 是「优化前基线 vs v5 估算」，建议该节标题加「历史基线」 |
| 10-12 | `sim/20261101_吞吐升级与数据二期结论.md:19/20/23` | 4,042 / 3,821 / 5,542 / 8.375 | 921600 升级时的 route 快照（VGA 前） |
| 13 | `sim/20261101_板端验收记录_patch2.md:14` | 5,542 | patch2 板端验收记录 |

> 另有 13 处「同行含新值」对照（VGA 记录表、09 方案 v2.2 的修订留痕、验收矩阵历史留档）与 13 处「行级历史上下文」
> 被自动豁免；`--all` 可逐条打印。29 篇历史文档（v2.0/v2.1、提案、审查、20261031 记录等）整体不参与告警。

---

## 7. 人工复核发现（不在给定黑名单，**工具不报**，供 Lead 参考）

1. `sim/records/acceptance_matrix.md:65,74` 写「UART 理论上限 ~174 fps（**527B**@921600）」；
   协议帧长是 **517 B**（`09_…md#§7.1`、18 号文），517×10/921600 = 5.61 ms → **178.3 fps**。
   527 可能是「517 下行 + 10 上行」被误当单帧字节数（或笔误），建议统一到 517 B / 178.3 fps。
2. `sim/records/acceptance_matrix.md:66` 写「板端准确率 **72.90%**（= 模型跨域精度）」；
   B6 部署后板端准确率为 **99.44%**（179/180）。该行是 B5/RPS 口径残留。
3. `sim/records/acceptance_matrix.md:18` T5 行已是 4,405/5,578/4.136，而 `:19` T6 行仍是 4,042/5,542/8.375（已在 A 类）。
4. `report/design_notes/09_…md` §6.4 表写「WBRAM 8,648 bit = **1,081 B**」，§6.5 标题写「271 字 = **1,084 B**」：
   1,081 B = 8,648/8（bit 直除），1,084 B = 271×4（按 32 bit 字对齐存储）。两者都对但口径不同，
   建议 §6.4 补「（未对齐）」以免被当成矛盾；`numbers.json` 的 blob 段以 manifest 的 271 字为准。

以上均**未改**，仅记录。若 Lead 认为应并入黑名单，在 `gen_numbers.py::LEGACY` 加一条即可（工具自动跟随）。

---

## 8. 局限与注意事项

1. **文档尚未按 JSON key 名引用**：`JSON key 名直查 0 次`。因此本检查器的主判定是「值归属 + 旧值黑名单」，
   而不是「同名 key 逐行比对」；等文档开始写 `engine_cycles=16911` 这类形式，直查分支会自动生效。
2. **`live_frames_total` 是快照**：生成时 2,952 行（唯一 2,681，重复 271）。采集随时可能继续追加，
   交付冻结后请重跑 `gen_numbers.py`；检查器对行数变化只给 `[INFO]`（默认隐藏，`--all` 可见），不判 FAIL。
3. **历史文档整篇豁免**是人工规则（文件名正则），不是语义理解：若某篇历史文档被再次当作现状引用，
   需要人工判断。豁免规则集中在 `check_numbers.py` 顶部 `ARCHIVED_RE` / `HIST_RE`，可审计、可收紧。
4. **扫描范围**仅 `report/design_notes/*.md`、`sim/*.md`（非递归）；`report/`、`README.md` 未扫。
5. **CRC16 三个 blob 分列**（FORMAL 0xB8C7 / B5 0x020B / B6 0x3569）：检查器不比较历史 blob，
   只做「manifest 现读值 vs JSON」漂移告警（防部署后忘更新 numbers.json）。
6. `numbers.json` 是**生成物**：任何数字异议应改 `gen_numbers.py` 的常量（带 source）后重跑，
   不要手改 JSON。

---

## 9. 复跑与门禁建议

```powershell
cd <repo>\src/pc

# 1) 刷新唯一数字来源（采集冻结/部署换 blob 后必须跑）
..\.venv\Scripts\python.exe gen_numbers.py

# 2) 一致性门禁（退出码 0 = PASS）
..\.venv\Scripts\python.exe check_numbers.py

# 3) 排查被豁免的历史引用
..\.venv\Scripts\python.exe check_numbers.py --all
```

建议把第 2 步加进交付前检查单（与 T10 一起签核）：**numbers.json 生成成功 + check_numbers.py 退出码 0** 才算 T10 关闭。
当前状态：生成 ✅，检查 ❌（A 类 9 行待修）。

---

## 7. Lead 处置（2026-11-01 18:40）

按"现行文档改新值、记录文档只加注"的原则处理完毕，复跑结果 **PASS**。

| 类别 | 处理 | 结果 |
|---|---|---|
| A 类 9 行/21 值 | 改为含 VGA 当前值（4,405 / 5,578 / 4.136），同行保留"无 VGA 版 4,042 / 5,542 / 8.375"作对照 | ✅ 清零 |
| B 类 13 行/15 值 | **不改数字**，逐行加注"（patch2 无 VGA 快照，历史实测；含 VGA 版见 VGA 实现与上板记录）" | ✅ 清零 |
| 人工发现 1 | 验收矩阵 T4b 行 `527B/~174fps` → **517B/178.3fps** | ✅ |
| 人工发现 2 | 验收矩阵板端准确率 72.90% 加注"B6 部署后 99.44%" | ✅ |
| 人工发现 3 | 09 方案 §6.4/§6.5 的 1,081B 与 1,084B 分别加注"口径A：bit 直除"与"口径B：32-bit 字对齐" | ✅ |

复跑（真实输出）：

```
RESULT: PASS  （A 现行 0 个/0 行，B 记录 0 个/0 行，内部自洽失败 0 项）
```

> 注意：`numbers.json` 的 `live_frames_total` 是**快照**，每次采集追加后需重跑 `gen_numbers.py`；
> 检查器默认不因行数漂移报警。
