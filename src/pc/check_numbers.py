#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T10 数字一致性检查器 —— 扫描文档，只报"过期数字"。

规则（与 sim/t10_single_source_of_truth.md 一致）
----------------------------------------------------
1. **现行值**：numbers.json 里 ``status != "superseded"`` 的叶子。文档出现这些值 → 正常。
2. **过期值**：``superseded.items`` + ``resource.patch2_no_vga`` 等 superseded 叶子。
   文档出现且**不满足豁免** → 报"过期数字"。
3. **豁免**（三种，逐条打印统计，``--all`` 可展开）：
   a. 同一行已含替换后的新值 → 属于"前后对照表"，不算错；
   b. 同一行含历史上下文词（历史/旧版/已被/取代/归档/提案/审查/v2.1/patch1/无 VGA/4 通道…）；
   c. 整篇文档属于历史文档（文件名含 v2.0/v2.1/提案/审查/评估/答复/复盘/20261031_ 等）。
4. 文档出现但既不在现行值、也不在过期黑名单里的数字（如 3,560 的早期估算）**不算错**。
5. **自述文档**：前 5 行含 ``numbers-check: ignore-file`` 的文件整篇跳过
   （T10 报告本身要列旧值清单，否则会自己报自己）。
6. 输出 PASS/FAIL + ``文件:行号``；退出码 0=PASS，1=FAIL。

用法
----
    cd <repo>/src/pc
    ..\\.venv\\Scripts\\python.exe check_numbers.py          # 只报过期数字
    ..\\.venv\\Scripts\\python.exe check_numbers.py --all    # 连豁免的一起列出
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
NUMBERS = ROOT / "data/design_notes" / "numbers.json"
SCAN_DIRS = ("data/design_notes", "sim")

# --------------------------------------------------------------------------
# 文档分级（文件名 → 历史文档 / 现行状态文档 / 记录类文档）
# --------------------------------------------------------------------------
ARCHIVED_RE = re.compile(
    r"v2\.[01](?!\d)|提案|审查|评估|答复|复核|决策评审|复盘|仿真门|20261031_"
    r"|D4数据通路|容量实验|误命名-勿用|拍数定界|骨架记录"
)
STATUS_RE = re.compile(r"进度|总表|验收矩阵|计划|标定|部署|上板")

# 行级历史上下文（出现即视为历史引用/对照，不报）
HIST_RE = re.compile(
    r"历史|旧版|旧值|旧表|旧文档|旧口径|旧数字|已被|已废|废弃|留档|归档|取代|不再|残留"
    r"|提案|审查|评估|答复|复核|决策评审|复盘|变更记录|基线|优化前|已过时"
    r"|v2\.0|v2\.1|patch1|无\s*VGA|不含\s*VGA|4\s*通道|4ch"
    r"|→|vs|superseded|deprecated",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------
# 数字抽取：先遮掉日期/版本/文件名/十六进制/标识符，再匹配数字
# --------------------------------------------------------------------------
MASK_PATTERNS = [
    re.compile(r"0[xX][0-9A-Fa-f]+"),                                   # 0xB8C7
    re.compile(r"https?://[^\s)>\]]+"),                                 # URL
    re.compile(r"\d{4}-\d{2}-\d{2}(?:T[\d:+\-]+)?"),                    # 日期
    re.compile(r"\b20\d{6}\b"),                                         # 20261101
    re.compile(r"\bv\d+(?:\.\d+)+"),                                     # v2.2
    re.compile(r"\b\d+(?:\.\d+)+(?=-\w)"),                               # 2.2-patch
    re.compile(r"§\d+(?:\.\d+)*"),                                      # §6.3
    re.compile(r"[\w\-./\\]+\.(?:md|py|v|sv|json|csv|hex|txt|png|bit|log|bin|xdc|tcl|ps1|pt|rpt|docx)\b"),
    re.compile(r"[A-Za-z_][A-Za-z_0-9]*\d[A-Za-z_0-9]*"),               # L1 / B6 / R34 / COM6
]
NUM_RE = re.compile(r"(?<![\w.§])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def mask_line(line: str) -> str:
    chars = list(line)
    for pat in MASK_PATTERNS:
        for m in pat.finditer(line):
            for i in range(m.start(), m.end()):
                chars[i] = " "
    return "".join(chars)


def numbers_in(line: str) -> list[tuple[str, float]]:
    masked = mask_line(line)
    out = []
    for m in NUM_RE.finditer(masked):
        tok = m.group(0)
        try:
            out.append((tok, float(tok.replace(",", ""))))
        except ValueError:
            continue
    return out


# --------------------------------------------------------------------------
# numbers.json 读取
# --------------------------------------------------------------------------
def flatten(node, prefix: str = ""):
    if isinstance(node, dict):
        if "value" in node and "source" in node:
            yield prefix, node
            return
        for k, v in node.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from flatten(v, f"{prefix}[{i}]")


def resolve_path(root: dict, path: str):
    node = root
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def load_numbers():
    if not NUMBERS.is_file():
        raise SystemExit(f"[FAIL] 找不到 {NUMBERS}；先跑 gen_numbers.py")
    data = json.loads(NUMBERS.read_text(encoding="utf-8-sig"))

    current: dict[str, dict] = {}
    superseded_leaves: list[tuple[str, dict]] = []
    for path, leaf in flatten(data):
        if path.startswith("superseded") or path.startswith("project"):
            continue
        if leaf.get("status") == "superseded":
            superseded_leaves.append((path, leaf))
        else:
            current[path] = leaf

    # 现行数值集合（含 "180/180" 这类比值）
    current_values: dict[float, list[str]] = {}

    def add_value(v, path):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            current_values.setdefault(float(v), []).append(path)
        elif isinstance(v, str):
            m = re.fullmatch(r"(\d+)/(\d+)", v.strip())
            if m:
                for x in m.groups():
                    current_values.setdefault(float(x), []).append(path)

    for path, leaf in current.items():
        add_value(leaf.get("value"), path)

    # 过期值表：superseded.items + superseded 叶子
    stale: dict[float, dict] = {}
    for path, leaf in superseded_leaves:
        target = leaf.get("superseded_by", "")
        new_leaf = resolve_path(data, target) if target else None
        stale[round(float(leaf["value"]), 6)] = {
            "display": leaf["value"],
            "label": path,
            "replaced_by": target,
            "replace_with": new_leaf["value"] if new_leaf else None,
            "note": leaf.get("note", ""),
        }
    for item in data.get("superseded", {}).get("items", []):
        stale[round(float(item["value"]), 6)] = {
            "display": item["value"],
            "label": item.get("label", ""),
            "replaced_by": item.get("replaced_by", ""),
            "replace_with": item.get("replace_with"),
            "note": item.get("note", ""),
        }
    return data, current, current_values, stale


# --------------------------------------------------------------------------
# 扫描
# --------------------------------------------------------------------------
def scan(current: dict, current_values, stale):
    files = []
    for d in SCAN_DIRS:
        files += sorted((ROOT / d).glob("*.md"))
    findings = {"A": [], "B": []}
    exempt = {"doc": 0, "doc_names": [], "self": 0, "same_line": [], "context": []}
    key_hits = 0
    current_hits = 0
    total_lines = 0
    scanned = 0

    # JSON key 名直查索引（同一短名取第一个）
    # 只收 performance/resource/model/protocol/acceptance 下的数值叶子，
    # 且短名必须含下划线 —— 避免把类名 rock/paper/scissors/junk 当 key。
    key_index: dict[str, tuple[str, dict]] = {}
    for p, leaf in current.items():
        if p.split(".")[0] not in ("performance", "resource", "model", "protocol", "acceptance"):
            continue
        short = p.split(".")[-1]
        if "_" in short and isinstance(leaf.get("value"), (int, float)):
            key_index.setdefault(short, (p, leaf))

    for path in files:
        name = path.name
        try:
            lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        except OSError as exc:
            print(f"[WARN] 读不了 {path}: {exc}")
            continue
        rel = path.relative_to(ROOT).as_posix()

        # 自述文档（如 T10 报告本身，内含旧值清单）：显式标记后整篇跳过，
        # 否则"列出过期数字"的报告会把自己报成过期。
        if any("numbers-check: ignore-file" in ln for ln in lines[:5]):
            exempt["self"] += 1
            continue
        if ARCHIVED_RE.search(name):
            exempt["doc"] += 1
            exempt["doc_names"].append(rel)
            continue
        total_lines += len(lines)
        bucket = "A" if STATUS_RE.search(name) else "B"
        scanned += 1

        for lineno, line in enumerate(lines, 1):
            toks = numbers_in(line)
            if not toks:
                continue
            vals = [v for _, v in toks]

            # ---- (0) key 名直查：文档若直接写 JSON key 名，取其后第一个数字比对 ----
            for key, (kpath, kleaf) in key_index.items():
                km = re.search(rf"(?<![\w`]){re.escape(key)}(?![\w`])", line)
                if not km:
                    continue
                key_hits += 1
                tail = numbers_in(line[km.end():])
                exp = kleaf["value"]
                if tail and isinstance(exp, (int, float)):
                    got = tail[0][1]
                    if abs(got - float(exp)) > max(1e-9, abs(float(exp)) * 1e-3):
                        findings[bucket].append((rel, lineno, [{
                            "display": tail[0][0], "label": kpath, "replaced_by": kpath,
                            "replace_with": exp, "note": "JSON key 同名直查",
                        }]))

            # ---- (1) 过期值扫描 ----
            hits = []
            for tok, val in toks:
                if round(val, 6) in current_values:
                    current_hits += 1
                entry = stale.get(round(val, 6))
                if entry is None:
                    continue
                # 豁免 a：同行已有替换后的新值（前后对照）
                newv = entry.get("replace_with")
                if isinstance(newv, (int, float)) and any(abs(v - float(newv)) < 1e-6 for v in vals):
                    exempt["same_line"].append((rel, lineno, tok, entry))
                    continue
                # 豁免 b：行级历史上下文
                if HIST_RE.search(line):
                    exempt["context"].append((rel, lineno, tok, entry))
                    continue
                hits.append({"display": tok, **entry})
            if hits:
                seen = set()
                uniq = []
                for h in hits:
                    k = round(float(str(h["display"]).replace(",", "")), 6)
                    if k in seen:
                        continue
                    seen.add(k)
                    uniq.append(h)
                findings[bucket].append((rel, lineno, uniq))

    return findings, exempt, scanned, total_lines, key_hits, current_hits


# --------------------------------------------------------------------------
# numbers.json 内部自洽（算术关系 / 验收门限）
# --------------------------------------------------------------------------
def internal_checks(data) -> list[tuple[str, bool, str]]:
    def g(path):
        node = resolve_path(data, path)
        return None if node is None else node.get("value")

    out: list[tuple[str, bool, str]] = []

    def chk(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    lut, logic, lutram = g("resource.lut_total"), g("resource.lut_logic"), g("resource.lut_ram")
    chk("LUT 总量 = Logic + LUTRAM",
        isinstance(lut, int) and lut == (logic or 0) + (lutram or 0),
        f"{lut} = {logic} + {lutram}")
    cyc, mhz, us = g("performance.engine_cycles"), g("performance.freq_mhz"), g("performance.frame_us")
    chk("拍数/时钟 = 帧时",
        isinstance(cyc, int) and abs(cyc / mhz - us) <= 0.1, f"{cyc}/{mhz} = {cyc/mhz:.2f} µs vs {us}")
    fps = g("performance.fps_engine")
    chk("1e6/帧时 ≈ 引擎 fps", abs(1e6 / us - fps) <= 2, f"{1e6/us:.1f} vs {fps}")
    chk("下行 517 B = 512 载荷 + 5 头/CRC",
        g("protocol.downlink_bytes") == 512 + 5, str(g("protocol.downlink_bytes")))
    bits, words = g("protocol.blob_bits"), g("protocol.blob_words")
    chk("blob 字数足够装 bit", isinstance(words, int) and words * 32 >= bits, f"{words}*32 ≥ {bits}")
    chk("板端准确率 = 179/180",
        abs(g("model.board_acc") - round(179 / 180 * 100, 2)) < 0.01,
        f"{g('model.board_acc')} vs {round(179/180*100, 2)}")
    m = g("model.board_match")
    chk("板端逐帧 180/180", m == "180/180", str(m))
    chk("T6 门限：LUT ≤5,000 / BRAM ≤6 / DSP=0 / WNS ≥0",
        lut <= 5000 and g("resource.bram_equiv") <= 6 and g("resource.dsp") == 0 and g("resource.wns_ns") >= 0,
        f"LUT {lut} / BRAM {g('resource.bram_equiv')} / DSP {g('resource.dsp')} / WNS {g('resource.wns_ns')}")
    chk("T4b：实测 ≥ 判据 150 fps",
        g("performance.fps_streaming") >= g("acceptance.t4b_criterion"),
        f"{g('performance.fps_streaming')} ≥ {g('acceptance.t4b_criterion')}")
    chk("T1a/T1b 均达红线（≥95% / ≥85%）",
        g("model.t1a") >= 95 and g("model.t1b") >= 85,
        f"T1a {g('model.t1a')} / T1b {g('model.t1b')}")
    key8 = data.get("t10_key8", [])
    missing = [p for p in key8 if resolve_path(data, p) is None]
    chk("T10 八个关键数字齐全", len(key8) == 8 and not missing,
        f"{len(key8)} 条" + (f"，缺 {missing}" if missing else ""))
    return out


# --------------------------------------------------------------------------
# 源漂移：numbers.json 生成后权威产物是否又变了
# --------------------------------------------------------------------------
def source_drift(data) -> tuple[list[str], list[str]]:
    warn, info = [], []
    for key, path in (("blob_crc16_b6", "board/weights/hw_b6/manifest.json"),
                      ("blob_crc16_formal", "board/weights/hw/manifest.json"),
                      ("blob_crc16_b5", "board/weights/hw_b5/manifest.json")):
        p = ROOT / path
        if p.is_file():
            try:
                m = json.loads(p.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError:
                continue
            got = m.get("crc16")
            exp = data["protocol"][key]["value"]
            if got != exp:
                warn.append(f"{path}#crc16={got}，numbers.json 记录 {exp}")
    csv_path = ROOT / "src" / "data" / "live" / "_manifest.csv"
    if csv_path.is_file():
        rows = sum(1 for _ in csv_path.open("r", encoding="utf-8-sig")) - 1
        exp = data["data"]["live_frames_total"]["value"]
        if rows != exp:
            info.append(f"live/_manifest.csv 行数 {rows}，numbers.json 快照 {exp}"
                        f"（采集在增长则属正常；冻结后重跑 gen_numbers.py）")
    return warn, info


# --------------------------------------------------------------------------
def fmt_entry(h: dict) -> str:
    new = h.get("replace_with")
    new_s = f"{new:,}" if isinstance(new, int) else ("—" if new is None else str(new))
    label = h.get("replaced_by") or h.get("label")
    return f"{h['display']} → 应为 {new_s}（{label}）"


def main() -> int:
    ap = argparse.ArgumentParser(description="T10 数字单一来源一致性检查")
    ap.add_argument("--all", action="store_true", help="同时列出被豁免的历史引用")
    ap.add_argument("--quiet", action="store_true", help="只打印结果行")
    args = ap.parse_args()

    data, current, current_values, stale = load_numbers()
    findings, exempt, scanned, total_lines, key_hits, current_hits = scan(
        current, current_values, stale)
    internal = internal_checks(data)
    n_bad_internal = sum(1 for _, ok, _ in internal if not ok)

    n_a = sum(len(x[2]) for x in findings["A"])
    n_b = sum(len(x[2]) for x in findings["B"])
    lines_a, lines_b = len(findings["A"]), len(findings["B"])

    if not args.quiet:
        print("==== T10 check_numbers ====")
        print(f"numbers.json : data/design_notes/numbers.json"
              f"（schema={data.get('schema')}，generated {data.get('generated_at')}）")
        print(f"现行值       : {len(current)} 条｜过期黑名单 {len(stale)} 条"
              f"｜文档中命中现行值 {current_hits} 处")
        print(f"扫描         : data/design_notes/*.md + sim/*.md = {scanned} 篇，{total_lines:,} 行"
              f"（豁免历史文档 {exempt['doc']} 篇 + 自述文档 {exempt['self']} 篇）")
        print(f"内部自洽     : {len(internal) - n_bad_internal}/{len(internal)} OK")
        for name, ok, detail in internal:
            if not ok:
                print(f"  [FAIL] {name}  （{detail}）")
        print()

    def dump(bucket: str, title: str) -> None:
        rows = findings[bucket]
        if not rows:
            print(f"{title}：无")
            return
        print(f"{title}：{len(rows)} 行 / {sum(len(r[2]) for r in rows)} 个过期值")
        for rel, lineno, items in rows:
            print(f"  {rel}:{lineno}")
            print("    " + "；".join(fmt_entry(h) for h in items))
        print()

    if not args.quiet:
        dump("A", "【A】现行文档（状态/进度/标定类，必须改）")
        dump("B", "【B】记录类文档（历史实测，建议加注'patch2 无 VGA 快照'，勿改数）")

        print(f"豁免：历史文档 {exempt['doc']} 篇｜自述文档 {exempt['self']} 篇"
              f"｜同行含新值 {len(exempt['same_line'])} 处"
              f"｜行级历史上下文 {len(exempt['context'])} 处")
        if args.all:
            print("  [历史文档] " + "、".join(exempt["doc_names"]))
            seen = set()
            for rel, lineno, tok, e in exempt["same_line"]:
                if (rel, lineno, tok) in seen:
                    continue
                seen.add((rel, lineno, tok))
                print(f"  [同行对照] {rel}:{lineno}  {tok}（{e['replaced_by']} 已是 {e['replace_with']}）")
            seen = set()
            for rel, lineno, tok, e in exempt["context"]:
                if (rel, lineno, tok) in seen:
                    continue
                seen.add((rel, lineno, tok))
                print(f"  [历史上下文] {rel}:{lineno}  {tok}")

        drift_warn, drift_info = source_drift(data)
        if drift_warn:
            print("\n[WARN] 权威产物已变化，numbers.json 过期，请重跑 gen_numbers.py：")
            for d in drift_warn:
                print(f"  - {d}")
        if drift_info and args.all:
            print("\n[INFO] 易变数据源（快照，不影响 PASS/FAIL）：")
            for d in drift_info:
                print(f"  - {d}")
        print(f"\nJSON key 名直查 {key_hits} 次（文档目前按值引用，未按 key 名引用）")
        print()

    status = "PASS" if (n_a + n_b + n_bad_internal) == 0 else "FAIL"
    print(f"RESULT: {status}  （A 现行 {n_a} 个/{lines_a} 行，B 记录 {n_b} 个/{lines_b} 行，"
          f"内部自洽失败 {n_bad_internal} 项）")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
