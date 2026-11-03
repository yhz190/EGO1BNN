"""全链路：组帧 →（mock）回包 → 统计口径（对应验收：统计口径 >= 3 个用例）"""
import csv
from pathlib import Path

import pytest

import uart_inject as ui
import make_report as mr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VECTORS = ROOT / "sim" / "vectors_t1a"
LABELS = VECTORS / "labels.csv"
LABELS_LIST = ["0", "1", "2", "3"]

pytestmark = pytest.mark.skipif(not LABELS.exists(),
                                reason="缺少 data/vectors_t1a/labels.csv")

_CACHE = {}


class _Serial:
    def __init__(self, replies):
        self.rx = bytearray()
        self.tx = bytearray()
        self.replies = list(replies)
        self.writes = 0

    @property
    def in_waiting(self):
        return len(self.rx)

    def write(self, d):
        self.writes += 1
        self.tx += d
        if self.replies:
            r = self.replies.pop(0)
            if r:
                self.rx += r
        return len(d)

    def read(self, n=1):
        n = min(n, len(self.rx))
        if n <= 0:
            return b""
        o = bytes(self.rx[:n])
        del self.rx[:n]
        return o

    def flush(self):
        pass

    def close(self):
        pass

    def reset_input_buffer(self):
        self.rx.clear()


def up(cls, seq):
    """上行帧只有 s0/s1/s2 三个分数，判定类 = argmax。
    class 3 (UNKNOWN) 无法用分数表达，只能靠 flags 拒识，故这里不模拟。"""
    s = [10, 10, 10]
    s[cls % 3] = 200
    body = bytes([0xA5, 0x83, cls % 3, s[0], s[1], s[2], 0, 7, seq & 0xFF])
    return body + bytes([ui.crc8(body)])


def load_rows():
    with open(LABELS, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def run_all():
    """整条链跑一次并缓存：620 帧 → mock 回包 → CSV 记录"""
    if "v" in _CACHE:
        return _CACHE["v"]

    rows = load_rows()
    frames = []
    replies = []
    for i, r in enumerate(rows):
        cls = int(r["class"])
        p = VECTORS / r["file"]
        frames.append(dict(path=str(p), name=r["file"], payload=p.read_bytes(),
                           seq=i & 0xFF, true_class=str(cls)))
        replies.append(up(cls, i & 0xFF))

    got = ui.run_batch(_Serial(replies), frames, None, timeout=0.2, retries=0,
                       class_names=LABELS_LIST)
    _CACHE["v"] = (rows, got)
    return rows, got


def count_matrix(got):
    m = {t: {p: 0 for p in LABELS_LIST} for t in LABELS_LIST}
    for r in got:
        m[r["true_class"]][r["pred_class"]] += 1
    return m


def expected_correct(rows):
    """class 0/1/2 能用分数表达，class 3 不能 → 只有前三类算对"""
    return sum(1 for r in rows if int(r["class"]) in (0, 1, 2))


def test_vectors_dataset_complete():
    rows = load_rows()
    assert len(rows) == 620
    for r in rows:
        assert (VECTORS / r["file"]).stat().st_size == 512
    assert {int(r["class"]) for r in rows} == {0, 1, 2, 3}


def test_pipeline_accuracy():
    rows, got = run_all()
    assert len(got) == 620
    assert all(r["pred_class"] in LABELS_LIST for r in got), "出现 TIMEOUT/REJECT"

    correct = sum(1 for r in got if r["pred_class"] == r["true_class"])
    exp = expected_correct(rows)
    assert correct == exp, "实测 %d，期望 %d" % (correct, exp)
    assert abs(correct / len(got) - exp / len(rows)) < 1e-12


def test_confusion_matrix_diagonal_equals_correct():
    _, got = run_all()
    m = count_matrix(got)
    diag = sum(m[c][c] for c in LABELS_LIST)
    correct = sum(1 for r in got if r["pred_class"] == r["true_class"])
    assert diag == correct


def test_confusion_matrix_totals():
    _, got = run_all()
    m = count_matrix(got)

    row_total = sum(sum(m[t].values()) for t in LABELS_LIST)
    col_total = sum(sum(m[t][c] for t in LABELS_LIST) for c in LABELS_LIST)

    assert row_total == len(got), "行合计应为总帧数"
    assert col_total == len(got), "列合计应为总帧数"
    assert row_total == col_total


def test_report_accuracy_matches_pipeline():
    rows, got = run_all()
    exp = expected_correct(rows)
    pct = "%.2f%%" % (exp / len(rows) * 100.0)

    md = mr.build_report(got, 95.0, ">=95%", item="T1a", csv_path="run.csv")
    assert pct in md, "报告里应有 %s" % pct
    assert str(len(rows)) in md


def test_wilson_bounds_cover_point_estimate():
    rows, got = run_all()
    correct = sum(1 for r in got if r["pred_class"] == r["true_class"])
    lo, hi = mr.wilson(correct, len(got))
    p = correct / len(got)
    assert lo <= p <= hi
    assert 0.0 <= lo and hi <= 1.0


def test_real_serial_optional(real_serial):
    """有 COM6 才跑实链，否则 SKIP（SKIP 不计失败）"""
    real_serial.write(ui.build_downlink(bytes(512)))
    u = ui.UplinkReader(real_serial).read(0.5)
    assert u is not None, "板子未回包（无板时本用例应被 SKIP）"
