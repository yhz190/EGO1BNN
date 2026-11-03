"""帧协议边界测试（对应验收：协议边界 >= 8 个用例）"""
import pytest

import uart_inject as ui


def up_frame(cls, s0=10, s1=20, s2=30, flags=0, fps=7, seq=0):
    body = bytes([0xA5, 0x83, cls, s0, s1, s2, flags, fps, seq])
    return body + bytes([ui.crc8(body)])


def reader_with(mock_serial, data, chunk=64):
    s = mock_serial(chunk=chunk)
    s.rx += data
    return ui.UplinkReader(s), s


def test_uplink_frame_decodes_all_fields(mock_serial):
    r, _ = reader_with(mock_serial, up_frame(2, 1, 2, 3, flags=0x01, fps=42, seq=9))
    u = r.read(0.2)
    assert u is not None
    assert (u.cls, u.s0, u.s1, u.s2, u.flags, u.fps, u.seq) == (2, 1, 2, 3, 0x01, 42, 9)


def test_uplink_frame_byte_by_byte(mock_serial):
    r, _ = reader_with(mock_serial, up_frame(1, seq=3), chunk=1)
    u = r.read(0.2)
    assert u is not None and u.seq == 3 and u.cls == 1


def test_uplink_frame_random_chunks(mock_serial):
    r, _ = reader_with(mock_serial, up_frame(3, seq=77), chunk=3)
    u = r.read(0.2)
    assert u is not None and u.seq == 77 and u.cls == 3


def test_uplink_skips_leading_garbage(mock_serial):
    junk = b"\x00\x11\x22\x33\xFF"
    r, _ = reader_with(mock_serial, junk + up_frame(0, seq=5))
    u = r.read(0.2)
    assert u is not None and u.seq == 5


def test_uplink_resync_after_fake_header(mock_serial):
    """假帧头 0xA5 + 非 0x83，必须跳过并找到真帧"""
    r, _ = reader_with(mock_serial, b"\xA5\x84" + up_frame(2, seq=6))
    u = r.read(0.2)
    assert u is not None and u.seq == 6 and u.cls == 2


def test_uplink_handles_double_sof(mock_serial):
    """连续两个 0xA5，以后一个为帧头"""
    r, _ = reader_with(mock_serial, b"\xA5" + up_frame(1, seq=8))
    u = r.read(0.2)
    assert u is not None and u.seq == 8


def test_uplink_two_frames_back_to_back(mock_serial):
    """粘包：两帧首尾相接"""
    data = up_frame(1, seq=1) + up_frame(2, seq=2)
    r, _ = reader_with(mock_serial, data)
    a = r.read(0.2)
    b = r.read(0.2)
    assert a is not None and a.seq == 1
    assert b is not None and b.seq == 2


def test_uplink_bad_crc_rejected(mock_serial):
    bad = bytearray(up_frame(0, seq=4))
    bad[-1] ^= 0xFF
    r, _ = reader_with(mock_serial, bytes(bad))
    assert r.read(0.05) is None


def test_uplink_truncated_frame_times_out(mock_serial):
    """少一个字节（长度不足）不能解出"""
    r, _ = reader_with(mock_serial, up_frame(0, seq=4)[:-1])
    assert r.read(0.05) is None


def test_uplink_trailing_bytes_ignored(mock_serial):
    r, _ = reader_with(mock_serial, up_frame(3, seq=12) + b"\x00\x00\x00")
    u = r.read(0.2)
    assert u is not None and u.seq == 12


def test_uplink_seq_wraps_255_to_0(mock_serial):
    data = up_frame(0, seq=255) + up_frame(1, seq=0)
    r, _ = reader_with(mock_serial, data)
    assert r.read(0.2).seq == 255
    assert r.read(0.2).seq == 0


def test_downlink_rejects_wrong_payload_len():
    with pytest.raises(ValueError):
        ui.build_downlink(bytes(511))
    with pytest.raises(ValueError):
        ui.build_downlink(bytes(513))


def test_downlink_crc_covers_sof():
    """CRC 覆盖 A5..payload（18_协议细节确认）"""
    pkt = ui.build_downlink(bytes(512))
    assert pkt[-1] == ui.crc8(pkt[:-1])
    assert pkt[-1] != ui.crc8(pkt[2:-1])


def test_collect_frames_rejects_wrong_size(tmp_path):
    """长度字段异常的帧文件必须被拒"""
    (tmp_path / "circle_0001.bin").write_bytes(bytes(511))
    with pytest.raises(SystemExit):
        ui.collect_frames(str(tmp_path), ["circle"])
