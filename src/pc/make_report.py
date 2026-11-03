#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_report.py — 读 uart_inject.py 的 CSV，生成 Markdown 精度报告。

门限一律从 sim/records/acceptance_matrix.md 抓；抓不到就报错退出，绝不自己编数字。
默认输出 sim/<今天>_T1a_混淆矩阵.md
"""
from __future__ import annotations

import argparse
import csv
import datetime
import math
import os
import re
import sys
from collections import Counter, defaultdict

DEFAULT_SPEC = os.path.join("sim", "records", "acceptance_matrix.md")
SPEC_PATTERNS = [
    r"(?:≥|>=|>|不低于|至少)\s*([0-9]+(?:\.[0-9]+)?)\s*%",
    r"([0-9]+(?:\.[0-9]+)?)\s*%",
]
SPECIAL = ["REJECT", "TIMEOUT", "(空)"]


def load_threshold(spec_path: str, item: str = "T1a"):
    if not os.path.exists(spec_path):
        raise SystemExit(
            f"ERROR: 找不到判据文件 {spec_path}。\n"
            f"       报告的门限必须来自验收矩阵，请用 --spec 指定正确路径。")
    with open(spec_path, encoding="utf-8") as fh:
        lines = fh.readlines()
    hits = [ln.strip() for ln in lines if item in ln]
    if not hits:
        raise SystemExit(f"ERROR: {spec_path} 里没有出现 {item}，无法引用判据")
    for ln in hits:
        for pat in SPEC_PATTERNS:
            m = re.search(pat, ln)
            if m:
                return float(m.group(1)), ln
    raise SystemExit(f"ERROR: {item} 那一行里找不到百分比门限：{hits[0]!r}")


def wilson(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def read_rows(path: str):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def build_report(rows, threshold, quote, *, item="T1a", spec_path=DEFAULT_SPEC,
                 csv_path="", title=None, failures=False) -> str:
    n = len(rows)
    if n == 0:
        raise SystemExit("ERROR: CSV 里没有数据行")

    tl = lambda r: (r.get("true_class") or "").strip() or "(空)"
    pl = lambda r: (r.get("pred_class") or "").strip() or "(空)"

    correct = sum(1 for r in rows if tl(r) == pl(r))
    acc = correct / n
    lo, hi = wilson(correct, n)
    passed = acc * 100.0 >= threshold - 1e-9

    present = {tl(r) for r in rows} | {pl(r) for r in rows}
    labels = sorted(present - set(SPECIAL))
    cols = labels + [c for c in SPECIAL if c in present]

    mat = defaultdict(Counter)
    for r in rows:
        mat[tl(r)][pl(r)] += 1

    L = []
    L.append(f"# {title or (item + ' 分类精度测试报告')}")
    L.append("")
    L.append(f"- 生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- 数据来源：`{csv_path}`（{n} 帧）")
    L.append(f"- 判据来源：`{spec_path}`")
    L.append("")

    L.append("## 1. 结论")
    L.append("")
    L.append(f"**{'满足' if passed else '不满足'} {item}** —— 实测总准确率 "
             f"**{acc*100:.2f}%**（{correct}/{n}），门限 **≥ {threshold:g}%**。")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("| --- | --- |")
    L.append(f"| 判据编号 | {item} |")
    L.append(f"| 判据原文 | `{quote}` |")
    L.append(f"| 门限 | ≥ {threshold:g}% |")
    L.append(f"| 实测准确率 | {acc*100:.2f}%（{correct}/{n}） |")
    L.append(f"| 95% Wilson 置信区间 | {lo*100:.1f}% ~ {hi*100:.1f}% |")
    L.append(f"| 结论 | {'满足' if passed else '不满足'} |")
    L.append("")
    if lo * 100 < threshold:
        L.append(f"> 注：点估计{'已' if passed else '未'}达门限，但置信区间下界 "
                 f"{lo*100:.1f}% < {threshold:g}%，n={n} 的样本量不足以在 95% 置信水平下"
                 f"宣称稳定达标，建议加大测试集。")
        L.append("")

    L.append("## 2. 混淆矩阵")
    L.append("")
    L.append("行 = 真实类别，列 = 板端判定结果。")
    L.append("")
    L.append("| 真实 \\ 预测 | " + " | ".join(cols) + " | 合计 | 该类准确率 |")
    L.append("|" + " --- |" * (len(cols) + 3))
    for t in labels:
        vals = [mat[t][c] for c in cols]
        tot = sum(vals)
        a = (mat[t][t] / tot) if tot else 0.0
        L.append(f"| **{t}** | " + " | ".join(str(x) for x in vals)
                 + f" | {tot} | {a*100:.1f}% |")
    col_tot = [sum(mat[t][c] for t in labels) for c in cols]
    L.append("| **列合计** | " + " | ".join(str(x) for x in col_tot)
             + f" | {sum(col_tot)} | {acc*100:.1f}% |")
    L.append("")

    L.append("## 3. 每类指标")
    L.append("")
    L.append("| 类别 | 样本数 | 判对 | 准确率(召回) | 精确率 | 95% CI |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for t in labels:
        tot = sum(mat[t][c] for c in cols)
        tp = mat[t][t]
        a = tp / tot if tot else 0.0
        colsum = sum(mat[x].get(t, 0) for x in labels)
        prec = tp / colsum if colsum else 0.0
        p_lo, p_hi = wilson(tp, tot)
        L.append(f"| {t} | {tot} | {tp} | {a*100:.1f}% | {prec*100:.1f}% "
                 f"| {p_lo*100:.1f}~{p_hi*100:.1f}% |")
    L.append("")

    timeouts = sum(1 for r in rows if pl(r) == "TIMEOUT")
    rejects = sum(1 for r in rows if pl(r) == "REJECT")
    seq_bad = sum(1 for r in rows if str(r.get("seq_ok", "")).strip() in ("0", "False"))
    L.append("## 4. 异常帧统计")
    L.append("")
    L.append(f"- 无响应 / 超时：{timeouts} 帧")
    L.append(f"- 板端拒识（flags 置位）：{rejects} 帧")
    L.append(f"- seq 不匹配：{seq_bad} 帧")
    L.append("")

    if failures:
        bad = [r for r in rows if tl(r) != pl(r)]
        L.append("## 5. 判错帧清单")
        L.append("")
        if bad:
            L.append("| seq | file | 真实 | 预测 | s0 | s1 | s2 | flags | note |")
            L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for r in bad:
                L.append(f"| {r.get('seq','')} | {r.get('file','')} | {tl(r)} | {pl(r)} "
                         f"| {r.get('s0','')} | {r.get('s1','')} | {r.get('s2','')} "
                         f"| {r.get('flags','')} | {r.get('note','')} |")
        else:
            L.append("无。")
        L.append("")

    L.append("## 6. 复现方式")
    L.append("")
    L.append("```bash")
    L.append(f"python src/pc/make_report.py {csv_path} --spec {spec_path} --item {item}")
    L.append("```")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="生成混淆矩阵 Markdown 报告")
    ap.add_argument("csv_path")
    ap.add_argument("--spec", default=DEFAULT_SPEC, help="acceptance_matrix.md 路径")
    ap.add_argument("--item", default="T1a", help="判据编号")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--failures", action="store_true", help="附判错帧清单")
    a = ap.parse_args(argv)

    threshold, quote = load_threshold(a.spec, a.item)
    rows = read_rows(a.csv_path)
    md = build_report(rows, threshold, quote, item=a.item, spec_path=a.spec,
                      csv_path=a.csv_path, title=a.title, failures=a.failures)

    out = a.out or os.path.join(
        "sim", f"{datetime.date.today().isoformat()}_{a.item}_混淆矩阵.md")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(md)

    print(f"[ok] 报告已写入 {out}")
    print(f"[ok] 门限 ≥{threshold:g}% 引自 {a.spec}")
    for line in md.splitlines():
        if line.startswith("**满足") or line.startswith("**不满足"):
            print(line)
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
