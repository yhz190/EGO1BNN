#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_acceptance.py — 一键跑全部验收用例，输出 JSON + Markdown 报告"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT_DIR = ROOT / "sim"


class Collector:
    def __init__(self):
        self.cases = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            msg = ""
            if report.outcome != "passed" and report.longreprtext:
                lines = [l for l in report.longreprtext.strip().splitlines() if l.strip()]
                msg = lines[-1][:200] if lines else ""
            self.cases.append({"nodeid": report.nodeid,
                               "outcome": report.outcome,
                               "duration": round(report.duration, 4),
                               "message": msg})
        elif report.when == "setup" and report.outcome == "skipped":
            self.cases.append({"nodeid": report.nodeid,
                               "outcome": "skipped",
                               "duration": round(report.duration, 4),
                               "message": str(report.longrepr)[-200:]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(HERE))
    ap.add_argument("--out", default=None, help="报告输出路径前缀")
    a = ap.parse_args()

    import pytest

    col = Collector()
    t0 = time.time()
    code = pytest.main([a.path, "-q", "--no-header", "-p", "no:cacheprovider"],
                       plugins=[col])
    elapsed = round(time.time() - t0, 2)

    cases = col.cases
    passed = sum(1 for c in cases if c["outcome"] == "passed")
    skipped = sum(1 for c in cases if c["outcome"] == "skipped")
    failed_cases = [c for c in cases if c["outcome"] == "failed"]
    failed = len(failed_cases)
    total = len(cases)

    stamp = datetime.date.today().isoformat()
    base = Path(a.out) if a.out else OUT_DIR / ("%s_验收自动化报告" % stamp)
    base.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "date": stamp,
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s": elapsed,
        "exit_code": int(code),
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "pass_rate": round(passed / total * 100, 2) if total else 0.0,
        "cases": cases,
    }
    json_path = base.with_suffix(".json")
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    L = []
    L.append("# 验收自动化报告（%s）" % stamp)
    L.append("")
    L.append("- 生成时间：%s" % data["generated_at"])
    L.append("- 总耗时：%s 秒" % elapsed)
    L.append("- 用例总数：%d" % total)
    L.append("")
    L.append("| 结果 | 数量 |")
    L.append("| --- | --- |")
    L.append("| PASS | %d |" % passed)
    L.append("| FAIL | %d |" % failed)
    L.append("| SKIP | %d |" % skipped)
    L.append("")
    L.append("**结论：%s**" % ("全部通过" if failed == 0 else "%d 项失败" % failed))
    L.append("")

    if failed_cases:
        L.append("## 失败用例")
        L.append("")
        for c in failed_cases:
            L.append("- `%s`" % c["nodeid"])
            if c["message"]:
                L.append("  - %s" % c["message"].replace("|", "\\|"))
        L.append("")

    skipped_cases = [c for c in cases if c["outcome"] == "skipped"]
    if skipped_cases:
        L.append("## 跳过用例（不计失败）")
        L.append("")
        for c in skipped_cases:
            tail = c["message"].splitlines()[-1] if c["message"] else ""
            L.append("- `%s` — %s" % (c["nodeid"], tail))
        L.append("")

    L.append("## 全部用例")
    L.append("")
    L.append("| 用例 | 结果 | 耗时(s) |")
    L.append("| --- | --- | --- |")
    for c in cases:
        L.append("| `%s` | %s | %.4f |" % (c["nodeid"], c["outcome"].upper(), c["duration"]))

    md_path = base.with_suffix(".md")
    md_path.write_text("\n".join(L) + "\n", encoding="utf-8")

    print("")
    print("用例总数 %d：PASS %d / FAIL %d / SKIP %d，耗时 %.2fs"
          % (total, passed, failed, skipped, elapsed))
    print("JSON: %s" % json_path)
    print("MD  : %s" % md_path)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
