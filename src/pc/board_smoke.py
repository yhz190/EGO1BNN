#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""board_smoke.py -- direct board test: send 512B frames over UART, decode the 10B result frames,
and compare with the exact Q6 model prediction (hw_eval_q6) computed from the same .bin files.

No assumptions about uart_inject's class table: prints raw bytes.
"""
import os, sys, time, glob
import numpy as np
import serial
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'scripts'))
import train_bnn10 as T10
import train_l3x8 as L8
from class_map import CLASS_MAP

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('port', nargs='?', default='COM6')
_ap.add_argument('baud', nargs='?', type=int, default=921600)
_ap.add_argument('n', nargs='?', type=int, default=5)
_ap.add_argument('--dir', default=os.path.join(ROOT, 'sim', 'vectors_t1a'))
_ap.add_argument('--ckpt', default=os.path.join(ROOT, 'src', 'weights', 'FORMAL_l3x8_s4.pt'))
_A = _ap.parse_args()
PORT, BAUD, N = _A.port, _A.baud, _A.n
VEC = _A.dir
CK = _A.ckpt


def crc8(data):
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else ((crc << 1) & 0xFF)
    return crc


def downlink(payload):
    body = bytes([0xA5, 0x10, 0x00, 0x02]) + payload
    return body + bytes([crc8(body)])


HEX = os.path.join(ROOT, 'src', 'weights', 'hw_adapt_20261102', 'weights.hex')
MODE = 'ckpt'
if CK and os.path.exists(CK):
    sd = torch.load(CK, map_location='cpu')
    m = L8.Net8(); m.load_state_dict(sd); m.eval()
    print('  model: ckpt %s' % os.path.basename(CK))
else:
    import weights_hex as WH
    assert os.path.exists(HEX), 'model reference missing: %s' % HEX
    _W = WH.load_hex(HEX)
    MODE = 'hex'
    print('  model: tracked weights blob %s (CRC 0x%04X)' % (HEX, _W['crc']))


def expected(bits):
    """bits: 64x64 uint8 -> (q6 scores, sat8 bytes, cls)"""
    X = torch.from_numpy(bits[None, None, :, :].astype(np.float32))
    if MODE == 'ckpt':
        with torch.no_grad():
            sc = T10.hw_eval_q6(m, X)[0].long()
    else:
        sc = torch.from_numpy(WH.forward_q6(_W, X)[0]).long()
    sat = [max(-128, min(127, int(v) >> 10)) & 0xFF for v in sc.tolist()]
    cls = int(sc.argmax())
    return [int(v) for v in sc.tolist()], sat, cls


files = sorted(glob.glob(os.path.join(VEC, '*.bin')))[:N]
t_start = time.time()
ser = serial.Serial(PORT, BAUD, timeout=1.5)
print('  port=%s baud=%d frames=%d' % (PORT, BAUD, len(files)))
time.sleep(0.3)
ok = 0
for i, f in enumerate(files):
    raw = open(f, 'rb').read()
    assert len(raw) == 512, len(raw)
    bits = np.unpackbits(np.frombuffer(raw, np.uint8))[:4096].reshape(64, 64)
    exp_sc, exp_sat, exp_cls = expected(bits)
    ser.reset_input_buffer()
    ser.write(downlink(raw))
    ser.flush()
    rsp = ser.read(10)
    if len(rsp) != 10:
        print('  [%d] %-14s TIMEOUT (got %d bytes)' % (i, os.path.basename(f), len(rsp)))
        continue
    crc_ok = (crc8(rsp[:9]) == rsp[9]) and rsp[0] == 0xA5 and rsp[1] == 0x83
    cls = rsp[2]
    s = [int(b - 256 if b >= 128 else b) for b in rsp[3:6]]
    flags, fps, seq = rsp[6], rsp[7], rsp[8]
    # the 10 B frame carries only s0..s2 (classes 0..2); the 4th score lives in SFR SCORE3
    exp3 = [(b - 256 if b >= 128 else b) for b in exp_sat[:3]]
    match = (s == exp3) and (cls == exp_cls)
    ok += 1 if match else 0
    if 'CSV' in os.environ:
        with open(os.environ['CSV'], 'a', encoding='utf-8') as fh:
            fh.write('%d,%s,%d,%s,%d,%s,%s\n' % (i, os.path.basename(f), cls,
                     ' '.join(map(str, s)), exp_cls, ' '.join(map(str, exp3)),
                     'MATCH' if match else 'DIFF'))
    print('  [%d] %-14s crc=%s cls=%d sat=%s flags=0x%02x fps=%d seq=%d | model cls=%d sat=%s %s'
          % (i, os.path.basename(f), 'OK' if crc_ok else 'BAD', cls, s, flags, fps, seq,
             exp_cls, exp3, 'MATCH' if match else 'DIFF'))
ser.close()
elapsed = time.time() - t_start
print('  elapsed=%.1fs  throughput=%.2f frames/s (%.1f ms/frame)' % (elapsed, len(files)/elapsed, 1000*elapsed/len(files)))
print('\n  RESULT: %d/%d frames match the model exactly' % (ok, len(files)))
sys.exit(0 if ok == len(files) else 1)
