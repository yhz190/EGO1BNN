#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T10 数字单一来源生成器 —— 产出 data/design_notes/numbers.json。

原则
----
1. **能自动读的绝不手抄**：weights/*/manifest.json、data/live/_manifest.csv、
   .git/refs/tags（纯文件读取，不调用 git 命令）一律现场读取；
2. 读不到的（验收判据、VGA 资源、精度指标）用常量，但每条都带
   ``source = "<文档相对路径>#<章节>"``，供 check_numbers.py 与人工回溯；
3. 每条叶子字段结构固定：``{"value", "unit"?, "source", "auto", "note"?}``；
4. 旧值集中放在 ``superseded`` 段（含 patch2 无 VGA 子项），**不进入当前口径**；
   check_numbers.py 只把 ``status != "superseded"`` 的叶子当作现行值。

用法
----
    cd <repo>/src/pc
    ..\\.venv\\Scripts\\python.exe gen_numbers.py
退出码：0 = 生成成功（允许有派生量警告）；2 = 权威产物缺失/自检失败。
"""
from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):          # Windows 控制台 UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]       # <项目根>/src/pc/gen_numbers.py
OUT = ROOT / "data/design_notes" / "numbers.json"
TZ = timezone(timedelta(hours=8))

# --------------------------------------------------------------------------
# 权威产物路径
# --------------------------------------------------------------------------
SPEC = "data/design_notes/v2.2_baseline_spec.md"
REL_T4B = "sim/t4b_stream_injection.md"
REL_VGA = "sim/vga_implementation.md"
REL_B6 = "sim/b6_calibration_final_deploy.md"
REL_G2 = "sim/20261101_G2正式结果与域适配.md"
REL_PATCH2 = "sim/20261101_patch2_RTL适配记录.md"
REL_TOP = "sim/20261101_top_bnn集成记录.md"
REL_T8 = "sim/t8_rtl_regression.md"
REL_CRC = "data/design_notes/18_协议细节确认_CRC覆盖范围.md"

W_FORMAL = ROOT / "src" / "weights" / "hw" / "manifest.json"
W_B5 = ROOT / "src" / "weights" / "hw_b5" / "manifest.json"
W_B6 = ROOT / "src" / "weights" / "hw_b6" / "manifest.json"
LIVE_CSV = ROOT / "src" / "data" / "live" / "_manifest.csv"
CLASS_MAP_PY = ROOT / "src" / "scripts" / "class_map.py"
GIT = ROOT / ".git"

WARNINGS: list[str] = []


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"[WARN] {msg}")


def leaf(value, *, source: str, unit: str | None = None, auto: bool = False,
         note: str | None = None, **extra) -> dict:
    """统一叶子：value / unit? / source / auto / note? / extra..."""
    d: dict = {"value": value}
    if unit:
        d["unit"] = unit
    d["source"] = source
    d["auto"] = auto
    if note:
        d["note"] = note
    d.update(extra)
    return d


def read_manifest(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"[FAIL] 缺失权威产物：{path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


# --------------------------------------------------------------------------
# 自动来源 1：权重 manifest（blob 规格 / CRC16 / 诊断精度）
# --------------------------------------------------------------------------
def load_blob_manifests() -> tuple[dict, dict, dict]:
    mf, m5, m6 = read_manifest(W_FORMAL), read_manifest(W_B5), read_manifest(W_B6)
    for name, m in (("hw", mf), ("hw_b5", m5), ("hw_b6", m6)):
        for k in ("bits", "words", "crc16", "model", "acc_q6_on_vectors"):
            if k not in m:
                raise SystemExit(f"[FAIL] {name}/manifest.json 缺字段 {k}")
    bits, words = m6["bits"], m6["words"]
    if words * 32 < bits:
        warn(f"manifest: words*32={words*32} < bits={bits}，blob 未按 32bit 对齐")
    return mf, m5, m6


# --------------------------------------------------------------------------
# 自动来源 2：现场采集 manifest（行数 = 总帧数，文件名可重复）
# --------------------------------------------------------------------------
def load_live_counts() -> dict:
    if not LIVE_CSV.is_file():
        raise SystemExit(f"[FAIL] 缺失采集清单：{LIVE_CSV}")
    rows, names, classes = 0, set(), {}
    with LIVE_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rows += 1
            names.add(row.get("file", ""))
            c = row.get("class", "")
            classes[c] = classes.get(c, 0) + 1
    mtime = datetime.fromtimestamp(LIVE_CSV.stat().st_mtime, TZ).isoformat(timespec="seconds")
    return {"rows": rows, "unique": len(names), "dup": rows - len(names),
            "classes": classes, "mtime": mtime}


# --------------------------------------------------------------------------
# 自动来源 3：git 标签（纯文件读取；任务要求不执行 git 命令）
# --------------------------------------------------------------------------
def read_git_tags() -> dict:
    tags = {}
    tag_dir = GIT / "refs" / "tags"
    if tag_dir.is_dir():
        for p in tag_dir.iterdir():
            if p.is_file():
                try:
                    tags[p.name] = {"sha": p.read_text(encoding="utf-8").strip(),
                                    "mtime": p.stat().st_mtime}
                except OSError:
                    continue
    packed = GIT / "packed-refs"
    if packed.is_file():                       # 松散 ref 优先；补充打包 ref
        for line in packed.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split(" ")
            if len(parts) == 2 and parts[1].startswith("refs/tags/"):
                tags.setdefault(parts[1][len("refs/tags/"):], {"sha": parts[0], "mtime": 0.0})
    head = (GIT / "HEAD").read_text(encoding="utf-8").strip() if (GIT / "HEAD").is_file() else ""
    latest = max(tags, key=lambda n: tags[n]["mtime"]) if tags else None
    return {"tags": tags, "latest": latest, "head": head}


# --------------------------------------------------------------------------
# 自动来源 4：class_map.py（唯一可执行类编码定义）
# --------------------------------------------------------------------------
def load_class_map() -> dict:
    text = CLASS_MAP_PY.read_text(encoding="utf-8-sig", errors="replace")
    m = re.search(r"CLASS_MAP\s*=\s*\(([^)]*)\)", text)
    if not m:
        raise SystemExit("[FAIL] class_map.py 里找不到 CLASS_MAP")
    names = re.findall(r"['\"]([a-z]+)['\"]", m.group(1))
    expect = ["rock", "paper", "scissors", "junk"]
    if names != expect:
        raise SystemExit(f"[FAIL] CLASS_MAP 漂移：{names} != {expect}")
    return {str(i): n for i, n in enumerate(names)}


# --------------------------------------------------------------------------
# 旧值黑名单（任务指定）—— 唯一存放处；patch2 无 VGA 子项由它派生
# --------------------------------------------------------------------------
LEGACY = [
    dict(value=19410, unit="cycle", label="每帧拍数（patch1 行缓冲 3×65 版）",
         replaced_by="performance.engine_cycles",
         note="patch2 实测 16,911 拍；19,410 仅历史留痕"),
    dict(value=2576, unit="fps", label="引擎峰值帧率（patch1）",
         replaced_by="performance.fps_engine",
         note="patch1 19,410 拍 @50MHz 的倒数值；patch2 为 2,957"),
    dict(value=1244, unit="bit", label="权重表位数（v2.1，junk 四类初期）",
         replaced_by="protocol.blob_bits",
         note="v2.1 口径 1,244 bit = 156 B；patch2 8 通道 Q6 为 8,648 bit"),
    dict(value=4448, unit="bit", label="权重表位数（4 通道版）",
         replaced_by="protocol.blob_bits",
         note="4 通道归档值；8 通道 patch2 为 8,648 bit"),
    dict(value=8693, unit="LUT", label="整机 Slice LUTs（patch2 优化前全集）",
         replaced_by="resource.lut_total",
         note="含 LUT as Memory 1,761；优化后 4,042（无 VGA）/4,405（含 VGA）"),
    dict(value=3821, unit="LUT", label="LUT as Logic（patch2 无 VGA）",
         replaced_by="resource.lut_logic",
         note="含 VGA 后 Logic 4,184（+365）"),
    dict(value=4042, unit="LUT", label="LUT 总量（patch2 无 VGA）",
         replaced_by="resource.lut_total", subitem="lut_total",
         note="含 VGA 版为 4,405（+365）"),
    dict(value=5542, unit="FF", label="FF（patch2 无 VGA）",
         replaced_by="resource.ff", subitem="ff",
         note="含 VGA 版为 5,578（+36）"),
    dict(value=8.375, unit="ns", label="WNS（patch2 无 VGA）",
         replaced_by="resource.wns_ns", subitem="wns_ns",
         note="含 VGA 版为 +4.136 ns"),
]


def build() -> dict:
    mf, m5, m6 = load_blob_manifests()
    live = load_live_counts()
    git = read_git_tags()
    cmap = load_class_map()

    # ---------------- 现行值 ----------------
    perf = {
        "engine_cycles": leaf(16911, unit="cycle", source=f"{SPEC}#§6.3",
                              note="L1..L5+done 拍数求和；T8 620 帧逐帧实测同为 16,911"),
        "freq_mhz": leaf(50, unit="MHz", source=f"{SPEC}#§3.1 与 §15.B",
                         note="clk_sys = 寄存器二分频+BUFG 100/2；VGA 等效 25 MHz 使能"),
        "frame_us": leaf(338.2, unit="us", source=f"{SPEC}#§6.3",
                         note="16,911 / 50 MHz = 338.22 µs，文档口径 338.2"),
        "fps_engine": leaf(2957, unit="fps", source=f"{SPEC}#§6.3",
                           note="引擎独占峰值；1e6/338.22 = 2,956.6 → 取 2,957"),
        "fps_end_to_end": leaf(59.55, unit="fps", source=f"{REL_T4B}#§1（T4a 口径）",
                               note="620 帧 / 10.4 s，含 PC 逐帧预处理 + 停等；T4a 判据 ≥30"),
        "fps_streaming": leaf(180.56, unit="fps", source=f"{REL_T4B}#§2（T4b 口径）",
                              note="预打包流式注入 3.434 s/620 帧；理论 178.3；T4b 判据 ≥150"),
    }

    res = {
        "lut_total": leaf(4405, unit="LUT", source=f"{REL_VGA}#§2（含 VGA，route 后）",
                          note="= lut_logic 4,184 + lut_ram 221；判据 ≤5,000"),
        "lut_logic": leaf(4184, unit="LUT", source=f"{REL_VGA}#§2"),
        "lut_ram": leaf(221, unit="LUT", source=f"{REL_VGA}#§2（LUTRAM）"),
        "ff": leaf(5578, unit="FF", source=f"{REL_VGA}#§2"),
        "bram_equiv": leaf(5.0, unit="RAMB36-equiv", source=f"{REL_VGA}#§2",
                           note="4×RAMB36 + 2×RAMB18；VGA 未增加 BRAM；判据 ≤6.0"),
        "dsp": leaf(0, unit="DSP", source=f"{REL_VGA}#§2", note="双闸门保证"),
        "wns_ns": leaf(4.136, unit="ns", source=f"{REL_VGA}#§2（route 后）",
                       note="含 VGA；判据 ≥0"),
        # 子项：patch2 无 VGA 版（旧口径，check_numbers 视为 superseded）
        "patch2_no_vga": {
            "lut_total": leaf(4042, unit="LUT", source=f"{REL_VGA}#§2 对照列",
                              status="superseded", superseded_by="resource.lut_total"),
            "ff": leaf(5542, unit="FF", source=f"{REL_VGA}#§2 对照列",
                       status="superseded", superseded_by="resource.ff"),
            "wns_ns": leaf(8.375, unit="ns", source=f"{REL_VGA}#§2 对照列",
                           status="superseded", superseded_by="resource.wns_ns"),
        },
    }

    acc_q6 = m6["acc_q6_on_vectors"]
    model = {
        "t1a": leaf(96.07, unit="%", source=f"{REL_G2}#§1（G2 正式口径）",
                    note="3 有效 seed：95.74/95.74/96.74；seed3 按 G4 拒收；判据 ≥95"),
        "t1a_sigma": leaf(0.47, unit="pp", source=f"{REL_G2}#§1"),
        "t1b": leaf(98.89, unit="%", source=f"{REL_B6}#§1",
                    note="B6 同环境留出 180 帧、2 seed；判据 ≥85"),
        "t1b_sigma": leaf(0.56, unit="pp", source=f"{REL_B6}#§1"),
        "t1b_frames": leaf(180, unit="frame", source=f"{REL_B6}#§1"),
        "t1b_seeds": leaf(2, unit="seed", source=f"{REL_B6}#§1"),
        "t1b_deployed_s1": leaf(99.44, unit="%", source=f"{REL_B6}#§3",
                                note="部署 ckpt B6_DA_s1 单 seed；= 179/180"),
        "rps_vectors_diag": leaf(round(acc_q6, 3), unit="ratio",
                                 source=f"{rel(W_B6)}#acc_q6_on_vectors", auto=True,
                                 note=f"RPS 620 向量跨域诊断（原始 {acc_q6:.7f}，仅诊断不参与验收）"),
        "board_match": leaf("180/180", source=f"{REL_B6}#§3", unit="frame",
                            note="板端 B6 测试集逐帧与模型一致（CRC/cls/sat8 全对）",
                            numerator=180, denominator=180),
        "board_acc": leaf(99.44, unit="%", source=f"{REL_B6}#§3",
                          note="= 179/180 板端真实标签准确率"),
    }

    proto = {
        "downlink_bytes": leaf(517, unit="B", source=f"{SPEC}#§7.1 与 {REL_CRC}#决定",
                               note="1+1+2+512+1；CRC8 覆盖 [0..515]"),
        "uplink_bytes": leaf(10, unit="B", source=f"{SPEC}#§7.1 与 {REL_CRC}#决定",
                             note="1+1+1+3+1+1+1+1；CRC8 覆盖 [0..8]"),
        "crc8_poly": leaf("0x07", source=f"{SPEC}#§7.1"),
        "blob_words": leaf(m6["words"], unit="word", source=f"{rel(W_B6)}#words", auto=True),
        "blob_bits": leaf(m6["bits"], unit="bit", source=f"{rel(W_B6)}#bits", auto=True,
                          note="8,648 bit → 271 字（对齐 8,672 bit = 1,084 B）"),
        "blob_crc16_formal": leaf(mf["crc16"], source=f"{rel(W_FORMAL)}#crc16", auto=True,
                                  note=f"模型 {mf['model']}"),
        "blob_crc16_b5": leaf(m5["crc16"], source=f"{rel(W_B5)}#crc16", auto=True,
                              note=f"模型 {m5['model']}"),
        "blob_crc16_b6": leaf(m6["crc16"], source=f"{rel(W_B6)}#crc16", auto=True,
                              note=f"模型 {m6['model']}（当前部署 blob）"),
    }

    accept = {
        "checks_pipeline": leaf(115, unit="check", source=f"{REL_PATCH2}#§1/§4",
                                note="tb_pipeline；负控 4 变异体被抓 + 空目录复跑一致"),
        "checks_top": leaf(147, unit="check", source=f"{REL_VGA}#§1 与 {REL_TOP}#§14.3",
                           note="tb_top_bnn（含 VGA 集成）147/147 PASS"),
        "t8_frames": leaf(620, unit="frame", source=f"{REL_T8}#§1",
                          note="620/620 逐 bit 一致；每帧 16,911 拍"),
        "t4b_criterion": leaf(150, unit="fps", source=f"{SPEC}#§1.3（T4b 判据）",
                              note="≥150 fps；实测 180.56"),
    }

    data = {
        "live_frames_total": leaf(live["rows"], unit="frame", source=f"{rel(LIVE_CSV)}（行数）", auto=True,
                                  volatile=True,
                                  note=f"按 manifest 行数记（含重复文件名）；快照时间 {live['mtime']}"),
        "live_frames_unique": leaf(live["unique"], unit="frame", source=f"{rel(LIVE_CSV)}（唯一文件名）", auto=True,
                                   note=f"重复文件名 {live['dup']} 个（跨会话重名）"),
        "live_manifest_mtime": leaf(live["mtime"], source=f"{rel(LIVE_CSV)}（mtime）", auto=True,
                                    note="采集仍在追加时该值会变，重跑 gen_numbers.py 刷新"),
        "live_class_counts": {k: leaf(v, unit="frame", source=f"{rel(LIVE_CSV)}#class", auto=True)
                              for k, v in sorted(live["classes"].items())},
        "class_map": {k: leaf(v, source=f"{rel(CLASS_MAP_PY)}#CLASS_MAP", auto=True)
                      for k, v in cmap.items()},
    }

    project = {
        "git_head": leaf(git["head"], source=".git/HEAD", auto=True,
                         note="纯文件读取，未调用 git 命令"),
        "git_tag_latest": leaf(git["latest"] or "", source=".git/refs/tags（mtime 最新）", auto=True),
        "git_tags": leaf(sorted(git["tags"]), source=".git/refs/tags", auto=True),
        "baseline_spec": leaf(SPEC, source=SPEC),
    }

    numbers = {
        "schema": "ego1-bnn/numbers@1",
        "generated_at": datetime.now(TZ).isoformat(timespec="seconds"),
        "generator": "src/pc/gen_numbers.py",
        "policy": "文档只能引用本文件现行值；旧值见 superseded 段，禁止再写入验收口径",
        # T10 判据"8 个关键数字逐项一致"的 8 个键（报告/答辩统一引用这 8 条）
        "t10_key8": [
            "performance.engine_cycles",
            "performance.fps_engine",
            "resource.lut_total",
            "resource.ff",
            "model.t1a",
            "model.t1b",
            "model.board_acc",
            "protocol.downlink_bytes",
        ],
        "project": project,
        "performance": perf,
        "resource": res,
        "model": model,
        "protocol": proto,
        "acceptance": accept,
        "data": data,
    }

    # ---------------- 旧值（含 patch2 无 VGA 子项引用） ----------------
    def current(path: str):
        node = numbers
        for p in path.split("."):
            node = node[p]
        return node["value"]

    items = []
    for e in LEGACY:
        items.append({
            "value": e["value"],
            "unit": e["unit"],
            "label": e["label"],
            "replaced_by": e["replaced_by"],
            "replace_with": current(e["replaced_by"]),
            "note": e["note"],
            "source": f"{REL_VGA}#§2（对照列）" if e.get("subitem") else f"{SPEC}#§0 历史修订 / 验收矩阵执行表#§四",
        })
    numbers["superseded"] = {
        "note": "旧值黑名单：文档中出现这些值且无历史上下文/同行无新值时，check_numbers.py 报过期",
        "allow_coexist": [59.55, 180.56],
        "items": items,
    }

    # ---------------- 自检：派生量与文档口径 ----------------
    fc = numbers["performance"]["engine_cycles"]["value"] / numbers["performance"]["freq_mhz"]["value"]
    if abs(fc - numbers["performance"]["frame_us"]["value"]) > 0.1:
        warn(f"帧时自检：{fc:.2f} µs 与文档 338.2 µs 差 > 0.1")
    fps = 1e6 / fc
    if abs(fps - numbers["performance"]["fps_engine"]["value"]) > 5:
        warn(f"帧率自检：{fps:.1f} fps 与文档 2,957 差 > 5")
    if numbers["protocol"]["blob_words"]["value"] * 32 < numbers["protocol"]["blob_bits"]["value"]:
        warn("blob 字数不足 32bit 对齐")
    if numbers["protocol"]["downlink_bytes"]["value"] != 512 + 5:
        warn("下行帧长与 512B 载荷 + 5B 头/CRC 不符")
    if numbers["protocol"]["uplink_bytes"]["value"] != 10:
        warn("上行帧长应为 10 B")
    if not numbers["data"]["class_map"]:
        warn("class_map 为空")

    n_leaf = count_leaves({k: v for k, v in numbers.items() if k != "superseded"})
    print(f"[OK] 自动来源：manifest×3 / live CSV / git tags×{len(git['tags'])} / class_map.py")
    print(f"[OK] 现行叶子 {n_leaf} 条，旧值 {len(items)} 条")
    print(f"[OK] live_frames_total={live['rows']}（唯一 {live['unique']}，重复 {live['dup']}）"
          f" 快照 {live['mtime']}")
    print(f"[OK] blob: {numbers['protocol']['blob_bits']['value']} bit / "
          f"{numbers['protocol']['blob_words']['value']} 字；CRC16 B6={numbers['protocol']['blob_crc16_b6']['value']}")
    print(f"[OK] git tag latest = {git['latest']}（head {git['head']}）")
    return numbers


def count_leaves(node, *, skip_superseded: bool = True) -> int:
    if isinstance(node, dict):
        if "value" in node and "source" in node:
            if skip_superseded and node.get("status") == "superseded":
                return 0
            return 1
        if node is None:
            return 0
        return sum(count_leaves(v, skip_superseded=skip_superseded) for v in node.values())
    if isinstance(node, list):
        return sum(count_leaves(v, skip_superseded=skip_superseded) for v in node)
    return 0


def main() -> int:
    numbers = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(numbers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    size = OUT.stat().st_size
    print(f"\n[OK] 写出 {rel(OUT)}（{size:,} B，UTF-8 indent=2）")
    print(f"     生成时间 {numbers['generated_at']}")
    if WARNINGS:
        print(f"[WARN] 共 {len(WARNINGS)} 条派生量警告（见上）")
        return 0
    print("[OK] 派生量自检全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
