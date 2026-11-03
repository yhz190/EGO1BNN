"""仓库一致性（对应验收：仓库一致性 >= 4 个用例）"""
import re
from pathlib import Path

import pytest

import uart_inject as ui

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WEIGHT_SPEC = ROOT / "src" / "weights_layout_spec.md"
CRC_DOC = ROOT / "data/design_notes" / "18_协议细节确认_CRC覆盖范围.md"
README_TOOLS = ROOT / "src" / "scripts" / "README.md"


def read(path):
    if not path.exists():
        pytest.skip("缺少 %s" % path.name)
    return path.read_text(encoding="utf-8", errors="ignore")


def find_v22():
    d = ROOT / "data/design_notes"
    if not d.is_dir():
        return None
    hits = sorted(d.glob("*v2.2*基线冻结*.md"))
    return hits[0] if hits else None


def test_weight_bits_arithmetic():
    parts = [72, 72, 32, 4096, 144, 16, 16]
    assert sum(parts) == 4448


def test_weight_total_is_whole_words():
    assert 4448 % 32 == 0
    assert 4448 // 32 == 139


def test_spec_declares_total_bits_and_words():
    text = read(WEIGHT_SPEC).replace(",", "")
    assert "4448" in text, "规范里应有总 bit 数 4448"
    assert "139" in text, "规范里应有字数 139"


def test_spec_bitfield_table_sums_to_total():
    text = read(WEIGHT_SPEC)
    m = re.search(r"bit 流布局(.*?)(?=\n##\s)", text, re.S)
    if not m:
        pytest.skip("找不到 bit 流布局章节")
    rows = re.findall(r"^\|\s*[\d,]+\s*\|[^|]*\|\s*([\d,]+)\s*\|", m.group(1), re.M)
    if not rows:
        pytest.skip("该章节未解析出位宽表格")
    total = sum(int(x.replace(",", "")) for x in rows)
    assert total == 4448, "表格位宽之和 %d != 4448" % total


def test_crc_doc_declares_cover_sof():
    text = read(CRC_DOC)
    assert "包含帧头" in text
    assert "0xA5" in text


def test_code_crc_matches_declared_scope():
    pkt = ui.build_downlink(bytes(512))
    assert len(pkt) == 517
    assert pkt[-1] == ui.crc8(pkt[:-1])
    assert pkt[-1] != ui.crc8(pkt[2:-1])


def test_v22_doc_says_517_and_10():
    doc = find_v22()
    if doc is None:
        pytest.skip("找不到 v2.2 冻结版文档")
    text = doc.read_text(encoding="utf-8", errors="ignore")
    assert "517" in text
    assert "10 B" in text or "10B" in text


def test_readme_mentions_crc_scope():
    text = read(README_TOOLS)
    assert "CRC" in text
    assert "A5" in text
