#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""uart_inject.py — v2.2 帧批量注入 + 回读判定 + CSV 记录

下行(PC->板): A5 10 len(2,LE) payload(512) CRC8        = 517 B
上行(板->PC): A5 83 class s0 s1 s2 flags fps seq CRC8   = 10 B

用法:
  # 批量注入一个目录并出 CSV（板子在线）
  python uart_inject.py --port COM7 --baud 115200 \
      --dir data/all --classes circle,cross,triangle --csv out.csv

  # 无板自检：只校验文件尺寸与组帧
  python uart_inject.py --dir data/all --classes circle,cross,triangle --dry-run
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# ------------------------- 协议常量 -------------------------
SOF = 0xA5
CMD_DOWN = 0x10
CMD_UP = 0x83

PAYLOAD_LEN = 512
DOWN_FRAME_LEN = 2 + 2 + PAYLOAD_LEN + 1     # 517
UP_FRAME_LEN = 10                            # 见 README「帧格式假设」

# CRC8/ATM: poly=0x07, init=0x00, 不反转, xorout=0x00
# 必须与 src/rtl/ 里的 crc8 实现一致；不一致只改这两行
CRC8_POLY = 0x07
CRC8_INIT = 0x00

FLAG_REJECT = 0x01          # flags bit0 = 1 表示板端拒识（按错处理）

SOF_UP = bytes((SOF, CMD_UP))

CSV_FIELDS = [
    "seq", "true_class", "pred_class", "s0", "s1", "s2", "sj", "flags",
    "rx_seq", "fps", "seq_ok", "note", "file",
]

# ------------------------- 组帧 / 解帧 -------------------------


def crc8(data: bytes, crc: int = CRC8_INIT) -> int:
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ CRC8_POLY) & 0xFF if (crc & 0x80) else ((crc << 1) & 0xFF)
    return crc


def build_downlink(payload: bytes) -> bytes:
    if len(payload) != PAYLOAD_LEN:
        raise ValueError(f"payload 必须 {PAYLOAD_LEN} 字节，实际 {len(payload)}")
    body = bytes([SOF, CMD_DOWN]) + PAYLOAD_LEN.to_bytes(2, "little") + payload
    return body + bytes([crc8(body)])


@dataclass
class Uplink:
    cls: int
    s0: int
    s1: int
    s2: int
    flags: int
    fps: int
    seq: int


class UplinkReader:
    """带缓冲的上行帧解析器：自动跳垃圾、校验 CRC8、支持单字节分片到达。"""

    def __init__(self, ser, frame_len: int = UP_FRAME_LEN):
        self.ser = ser
        self.frame_len = frame_len
        self.buf = bytearray()

    def reset(self) -> None:
        self.buf.clear()

    def _pump(self) -> bool:
        try:
            n = self.ser.in_waiting
        except Exception:
            n = 0
        data = self.ser.read(n) if n else self.ser.read(1)
        if data:
            self.buf += data
            return True
        return False

    def read(self, timeout: float = 1.0) -> Optional[Uplink]:
        deadline = time.monotonic() + timeout
        while True:
            up = self._try_parse()
            if up is not None:
                return up
            if time.monotonic() >= deadline:
                return None
            if not self._pump():
                time.sleep(0.001)

    def _try_parse(self) -> Optional[Uplink]:
        while True:
            i = self.buf.find(SOF_UP)
            if i < 0:
                if len(self.buf) > 1:             # 保留最后 1 字节，防 A5 被切开
                    del self.buf[: len(self.buf) - 1]
                return None
            if i:
                del self.buf[:i]
            if len(self.buf) < self.frame_len:
                return None
            cand = bytes(self.buf[: self.frame_len])
            if crc8(cand[:-1]) == cand[-1]:
                del self.buf[: self.frame_len]
                return Uplink(cls=cand[2], s0=cand[3], s1=cand[4], s2=cand[5],
                              flags=cand[6], fps=cand[7], seq=cand[8])
            del self.buf[:1]                      # 假帧头，丢弃 1 字节重找


# ------------------------- 文件名 -> 标签 / 序号 -------------------------


def _natural_key(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(p))]


def derive_label(path):
    """circle_0007.bin / 0007_circle.bin / circle/0007.bin 都能认。"""
    p = Path(path)
    stem = p.stem
    cls = seq = None
    m = re.match(r"^([A-Za-z]+)[^0-9]*?(\d+)$", stem)
    if m:
        cls, seq = m.group(1), int(m.group(2))
    if cls is None:
        for t in re.split(r"[^0-9A-Za-z]+", stem):
            if t and not t.isdigit():
                cls = t
                break
    if cls is None and p.parent.name not in ("", "."):
        cls = p.parent.name
    if seq is None:
        digs = re.findall(r"\d+", stem)
        seq = int(digs[-1]) if digs else None
    return cls, seq


def collect_frames(dirpath: str, class_names=None, limit: Optional[int] = None):
    paths = sorted(glob.glob(os.path.join(dirpath, "**", "*.bin"), recursive=True),
                   key=_natural_key)
    if not paths:
        raise SystemExit(f"ERROR: {dirpath} 下没有找到 *.bin 帧文件")

    items = [[p, *derive_label(p)] for p in paths]      # [path, cls, seq]

    if not class_names:
        names = sorted({(it[1] or "unknown") for it in items})
        print("[info] 未给 --classes，按文件名/目录名推断类别映射："
              + "  ".join(f"{i}={n}" for i, n in enumerate(names)))
        print("[info] 请确认该顺序与板端 class 编码一致，不一致就用 --classes 显式指定")
    else:
        names = list(class_names)
        known = {n.lower() for n in names}
        bad = sorted({(it[1] or "?") for it in items if (it[1] or "").lower() not in known})
        if bad:
            raise SystemExit(f"ERROR: 出现 --classes 未声明的类别: {bad}")

    frames = []
    for p, cls, seq in items:
        data = Path(p).read_bytes()
        if len(data) != PAYLOAD_LEN:
            raise SystemExit(f"ERROR: {p} 为 {len(data)} 字节，应为 {PAYLOAD_LEN} 字节 payload")
        frames.append(dict(path=p, name=os.path.basename(p), payload=data,
                           seq=seq, true_class=cls or "unknown"))

    if limit:
        frames = frames[:limit]

    filled = 0
    for i, fr in enumerate(frames):
        if fr["seq"] is None:
            fr["seq"] = i
            filled += 1
    if filled:
        print(f"[warn] {filled} 个文件无序号信息，已按遍历顺序补 seq")
    return frames, names


# ------------------------- 批量注入 -------------------------


def _make_row(fr, up, dt_ms, class_names, signed_scores):
    row = dict(seq=fr["seq"], true_class=fr["true_class"], pred_class="",
               s0="", s1="", s2="", sj="", flags="", rx_seq="", fps="",
               seq_ok="", note="", file=fr["name"])
    if up is None:
        row["pred_class"] = "TIMEOUT"
        row["note"] = f"no uplink frame within {dt_ms}ms"
        return row

    def cv(b):
        return b - 256 if (signed_scores and b >= 128) else b

    s = [cv(up.s0), cv(up.s1), cv(up.s2)]
    j = max(range(3), key=lambda k: s[k])
    row.update(s0=s[0], s1=s[1], s2=s[2], flags=up.flags,
               rx_seq=up.seq, fps=up.fps,
               seq_ok=int(fr["seq"] is not None and up.seq == fr["seq"]))
    if up.flags & FLAG_REJECT:
        row["pred_class"] = "REJECT"
        row["sj"] = max(s)                 # 拒识时 sj 记最大分，便于看阈值余量
        row["note"] = "board asserted reject flag"
    else:
        row["pred_class"] = class_names[j] if j < len(class_names) else str(j)
        row["sj"] = s[j]
    return row


def run_batch(ser, frames, csv_path=None, timeout=1.0, retries=2, delay=0.0,
              class_names=(), signed_scores=False, verbose=False) -> List[dict]:
    rows: List[dict] = []
    fh = writer = None
    if csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        fh = open(csv_path, "w", newline="", encoding="utf-8-sig")
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()

    reader = UplinkReader(ser)
    try:
        for i, fr in enumerate(frames):
            pkt = build_downlink(fr["payload"])
            up = None
            t0 = time.monotonic()
            for _ in range(retries + 1):
                reader.reset()
                if hasattr(ser, "reset_input_buffer"):
                    try:
                        ser.reset_input_buffer()
                    except Exception:
                        pass
                ser.write(pkt)
                if hasattr(ser, "flush"):
                    try:
                        ser.flush()
                    except Exception:
                        pass
                up = reader.read(timeout)
                if up is not None:
                    break
            dt_ms = int((time.monotonic() - t0) * 1000)

            row = _make_row(fr, up, dt_ms, class_names, signed_scores)
            rows.append(row)
            if writer:
                writer.writerow(row)
                fh.flush()                       # 长跑不丢数据
            if verbose:
                print(f"[{i+1}/{len(frames)}] {fr['name']:<26} seq={fr['seq']} "
                      f"true={fr['true_class']:<10} pred={row['pred_class']:<10} "
                      f"s=({row['s0']},{row['s1']},{row['s2']}) flags={row['flags']} {dt_ms}ms")
            if delay:
                time.sleep(delay)
    finally:
        if fh:
            fh.close()
    return rows


def summarize(rows, class_names=()):
    if not rows:
        return
    n = len(rows)
    ok = sum(1 for r in rows if r["pred_class"] == r["true_class"])
    to = sum(1 for r in rows if r["pred_class"] == "TIMEOUT")
    rj = sum(1 for r in rows if r["pred_class"] == "REJECT")
    print(f"\n==== {n} 帧  正确 {ok}  准确率 {ok/n*100:.2f}%  超时 {to}  拒识 {rj} ====")

    present = {r["true_class"] for r in rows} | {r["pred_class"] for r in rows}
    labels = [c for c in class_names if c in present] + \
             sorted(present - set(class_names) - {"REJECT", "TIMEOUT"})
    cols = labels + [c for c in ("REJECT", "TIMEOUT") if c in present]

    print("混淆矩阵(行=真实, 列=预测):")
    print("        " + "".join(f"{c:>10}" for c in cols) + f"{'合计':>8}")
    for t in labels:
        cnt = {c: 0 for c in cols}
        for r in rows:
            if r["true_class"] == t:
                cnt[r["pred_class"]] = cnt.get(r["pred_class"], 0) + 1
        print(f"{t:<8}" + "".join(f"{cnt.get(c, 0):>10}" for c in cols)
              + f"{sum(cnt.values()):>8}")


# ------------------------- CLI -------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description="v2.2 帧批量注入工具（PC 端）")
    ap.add_argument("--dir", help="帧目录（递归找 *.bin，512B payload）")
    ap.add_argument("--file", help="单帧 payload 文件（兼容旧用法）")
    ap.add_argument("--classes", default="", help="类别顺序，逗号分隔，对应 s0,s1,s2")
    ap.add_argument("--port", help="如 COM7 或 /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--csv", help="输出 CSV 路径")
    ap.add_argument("--timeout", type=float, default=1.0, help="单帧回读超时(s)")
    ap.add_argument("--retries", type=int, default=2, help="超时重发次数")
    ap.add_argument("--delay", type=float, default=0.0, help="帧间隔(s)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--signed-scores", action="store_true", help="s0..s2 视为 int8 补码")
    ap.add_argument("--dry-run", action="store_true", help="不连串口，只校验文件与组帧")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    class_names = [c.strip() for c in a.classes.split(",") if c.strip()]

    if a.dir:
        frames, class_names = collect_frames(a.dir, class_names, a.limit)
    elif a.file:
        data = Path(a.file).read_bytes()
        if len(data) != PAYLOAD_LEN:
            raise SystemExit(f"ERROR: {a.file} 为 {len(data)} 字节，应为 {PAYLOAD_LEN}")
        frames = [dict(path=a.file, name=os.path.basename(a.file), payload=data,
                       seq=0, true_class=(class_names[0] if class_names else "unknown"))]
    else:
        ap.error("需要 --dir 或 --file")

    if a.dry_run:
        print(f"[dry-run] {len(frames)} 帧  payload={PAYLOAD_LEN}B  下行帧长={DOWN_FRAME_LEN}B")
        for fr in frames[:5]:
            pkt = build_downlink(fr["payload"])
            print(f"  {fr['name']:<26} seq={fr['seq']:<4} true={fr['true_class']:<10} "
                  f"head={pkt[:4].hex(' ')} crc=0x{pkt[-1]:02X}")
        if len(frames) > 5:
            print(f"  ... 其余 {len(frames) - 5} 帧略")
        return 0

    if not a.port:
        raise SystemExit("ERROR: 需要 --port（或用 --dry-run 无板自检）")

    import serial
    ser = serial.Serial(a.port, a.baud, timeout=0.01)
    try:
        rows = run_batch(ser, frames, a.csv, a.timeout, a.retries, a.delay,
                         class_names, a.signed_scores, a.verbose)
    finally:
        ser.close()

    summarize(rows, class_names)
    if a.csv:
        print(f"[ok] CSV 已写入 {a.csv}")
        print(f"     下一步：python make_report.py {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
