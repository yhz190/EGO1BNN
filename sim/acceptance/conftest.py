"""sim/acceptance/conftest.py — 公共 fixture 与路径

不写死绝对路径：全部由本文件位置推出仓库根。
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent          # sim/acceptance
ROOT = HERE.parents[1]                          # 仓库根
SCRIPTS = ROOT / "src" / "scripts"

for _p in (str(SCRIPTS), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VECTORS_DIR = ROOT / "sim" / "vectors_t1a"
LABELS_CSV = VECTORS_DIR / "labels.csv"
SPEC_MD = ROOT / "sim" / "acceptance_matrix.md"
WEIGHT_SPEC_MD = ROOT / "src" / "weights_layout_spec.md"
CRC_DOC_MD = ROOT / "data/design_notes" / "18_协议细节确认_CRC覆盖范围.md"
README_TOOLS_MD = SCRIPTS / "README.md"


def find_v22_doc():
    """找 v2.2_baseline_spec.md（名字可能有前缀，用通配）"""
    d = ROOT / "data/design_notes"
    if not d.is_dir():
        return None
    hits = sorted(d.glob("*v2.2*基线冻结*.md"))
    return hits[0] if hits else None


class FakeSerial:
    """假串口：write() 时把预设回复塞进接收缓冲；chunk 控制每次 read 最多返回多少字节。"""

    def __init__(self, replies=None, chunk=64):
        self.rx = bytearray()
        self.tx = bytearray()
        self.replies = list(replies or [])
        self.chunk = chunk
        self.timeout = 0.0
        self.writes = 0

    @property
    def in_waiting(self):
        return len(self.rx)

    def write(self, data):
        self.writes += 1
        self.tx += data
        if self.replies:
            r = self.replies.pop(0)
            if r:
                self.rx += r
        return len(data)

    def read(self, n=1):
        n = min(n, self.chunk, len(self.rx))
        if n <= 0:
            return b""
        out = bytes(self.rx[:n])
        del self.rx[:n]
        return out

    def flush(self):
        pass

    def close(self):
        pass

    def reset_input_buffer(self):
        self.rx.clear()


@pytest.fixture
def mock_serial():
    """返回 FakeSerial 类（工厂），测试里自行实例化。"""
    return FakeSerial


@pytest.fixture
def real_serial():
    """真串口 COM6；不存在就 SKIP（SKIP 不计失败）。"""
    try:
        import serial
        from serial.tools import list_ports
    except ImportError:
        pytest.skip("未安装 pyserial")

    ports = [p.device.upper() for p in list_ports.comports()]
    if "COM6" not in ports:
        pytest.skip("未检测到 COM6（当前: %s）" % (", ".join(ports) or "无"))

    ser = serial.Serial("COM6", 921600, timeout=0.05)
    try:
        yield ser
    finally:
        ser.close()
