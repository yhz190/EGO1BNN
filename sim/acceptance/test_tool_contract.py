"""uart_inject.py / make_report.py 的 CLI 契约（对应验收：工具契约 >= 5 个用例）"""
import csv
from pathlib import Path

import pytest

import uart_inject as ui
import make_report as mr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPTS = ROOT / "src" / "scripts"
README_TOOLS_MD = SCRIPTS / "README.md"
SPEC_MD = ROOT / "sim" / "records" / "acceptance_matrix.md"


def up(cls, seq, s0=200, s1=10, s2=5):
    body = bytes([0xA5, 0x83, cls, s0, s1, s2, 0, 7, seq & 0xFF])
    return body + bytes([ui.crc8(body)])


def test_cli_help_lists_required_flags(capsys):
    with pytest.raises(SystemExit) as exc:
        ui.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--dir", "--csv", "--dry-run", "--classes", "--baud"):
        assert flag in out, "帮助里缺少 %s" % flag


def test_dry_run_reports_517(tmp_path, capsys):
    for i in range(2):
        (tmp_path / ("circle_%04d.bin" % i)).write_bytes(bytes(512))
    rc = ui.main(["--dir", str(tmp_path), "--classes", "circle", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "517" in out and "512B" in out


def test_limit_is_honoured(tmp_path):
    for i in range(5):
        (tmp_path / ("circle_%04d.bin" % i)).write_bytes(bytes(512))
    frames, _ = ui.collect_frames(str(tmp_path), ["circle"], limit=2)
    assert len(frames) == 2


def test_unknown_class_rejected(tmp_path):
    (tmp_path / "circle_0001.bin").write_bytes(bytes(512))
    with pytest.raises(SystemExit):
        ui.collect_frames(str(tmp_path), ["paper"])


def test_csv_header_matches_contract(tmp_path):
    frames = [dict(path="a", name="circle_0001.bin", payload=bytes(512),
                   seq=1, true_class="circle")]
    out = tmp_path / "o.csv"
    ui.run_batch(_mk_serial([up(0, 1)]), frames, str(out),
                 timeout=0.1, retries=0, class_names=["circle", "cross"])
    with open(out, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == ui.CSV_FIELDS
    assert rows[0]["pred_class"] == "circle"


def _mk_serial(replies):
    class S:
        def __init__(self, rs):
            self.rx, self.tx, self.replies, self.writes = bytearray(), bytearray(), list(rs), 0

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

    return S(replies)


def test_make_report_matrix_and_accuracy():
    mk = lambda t, p: dict(seq="1", true_class=t, pred_class=p, s0="1", s1="2",
                           s2="3", sj="3", flags="0", rx_seq="1", fps="7",
                           seq_ok="1", note="", file="f.bin")
    rows = [mk("circle", "circle"), mk("circle", "cross"),
            mk("cross", "cross"), mk("triangle", "triangle")]
    md = mr.build_report(rows, 95.0, "| T1a | >=95% |", item="T1a", csv_path="x.csv")
    assert "75.00%" in md
    assert "混淆矩阵" in md
    assert "**不满足 T1a**" in md


def test_threshold_missing_spec_raises(tmp_path):
    with pytest.raises(SystemExit):
        mr.load_threshold(str(tmp_path / "nope.md"), "T1a")


def test_readme_documents_cli():
    if not README_TOOLS_MD.exists():
        pytest.skip("缺少 README.md")
    text = README_TOOLS_MD.read_text(encoding="utf-8", errors="ignore")
    for flag in ("--dir", "--csv", "--dry-run", "--classes", "--baud"):
        assert flag in text, "README 未说明 %s" % flag


def test_spec_file_parses_t1a():
    if not SPEC_MD.exists():
        pytest.skip("缺少 sim/records/acceptance_matrix.md")
    thr, quote = mr.load_threshold(str(SPEC_MD), "T1a")
    assert 0 < thr <= 100
    assert "T1a" in quote
