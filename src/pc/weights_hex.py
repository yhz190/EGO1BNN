#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""weights_hex.py -- parse the tracked v5 weights blob (weights.hex) and run the exact Q6 forward.

Why: training checkpoints (*.pt) are intentionally NOT shipped in the repo; the only tracked
model reference is board/weights/<version>/weights.hex.  board_smoke.py falls back to this
module when --ckpt is missing, so the board evidence command stays reproducible from git.

Blob layout v5 (8,648 bits / 271 words, LSB-first within a word):
  L1 72 / L2 72 / L3 64 / FC 8192 / theta1 32 / theta2 32 / theta3 32 / theta4 56 /
  thetaFC 64 / ID 16 / CRC16 16
FC is beat-major in the blob; de-interleaved back to class-major [4][2048].
"""
import os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'scripts'))
import train_bnn10 as T10

FEAT = 2048
FC_SCALE = 64
TOTAL = 8648


def crc16_ccitt_false(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else ((crc << 1) & 0xFFFF)
    return crc


def load_hex(path):
    words = [int(line.strip(), 16) for line in open(path) if line.strip()]
    bits = []
    for v in words:
        for k in range(32):
            bits.append((v >> k) & 1)
    assert len(bits) >= TOTAL, 'weights blob too short'
    bits = bits[:TOTAL]

    pos = 0

    def take(width, n):
        nonlocal pos
        out = []
        for _ in range(n):
            v = 0
            for k in range(width):
                v |= bits[pos + k] << k
            pos += width
            out.append(v)
        return out

    w1 = np.array(take(1, 72), np.uint8).reshape(8, 1, 3, 3)
    w2 = np.array(take(1, 72), np.uint8).reshape(8, 1, 3, 3)
    w3 = np.array(take(1, 64), np.uint8).reshape(8, 8, 1, 1)
    fcb = take(1, 8192)
    wf = np.zeros((4, FEAT), np.uint8)
    for q in range(256):
        for c in range(4):
            for j in range(8):
                wf[c, 8 * q + j] = fcb[32 * q + 8 * c + j]
    t1 = np.array(take(4, 8))
    t2 = np.array(take(4, 8))
    t3 = np.array(take(4, 8))
    t4 = np.array(take(7, 8))
    bias = [v - 65536 if v >= 32768 else v for v in take(16, 4)]
    ident = take(16, 1)[0]
    crc = take(16, 1)[0]
    raw = bytearray()
    for i in range(0, 8616, 8):
        v = 0
        for k in range(8):
            v |= bits[i + k] << k          # LSB-first within byte (as gen_weights writes it)
        raw.append(v)
    assert ident == 0x0001, 'ID mismatch'
    assert crc16_ccitt_false(bytes(raw)) == crc, 'CRC mismatch'
    return dict(w1=w1, w2=w2, w3=w3, wf=wf, t1=t1, t2=t2, t3=t3, t4=t4, bias=bias, crc=crc)


def forward_q6(P, X):
    """X: torch (N,1,64,64) -> np.int64 scores (N,4), same units as T10.hw_eval_q6()."""
    w1 = torch.from_numpy(P['w1']).float()
    w2 = torch.from_numpy(P['w2']).float()
    w3 = torch.from_numpy(P['w3']).float()
    wf = torch.from_numpy(P['wf']).float()
    t1 = torch.tensor(P['t1']).view(1, -1, 1, 1)
    t2 = torch.tensor(P['t2']).view(1, -1, 1, 1)
    t3 = torch.tensor(P['t3']).view(1, -1, 1, 1)
    bias = torch.tensor(P['bias'])
    with torch.no_grad():
        x1 = (T10.matches_fixed(X, w1, 1) > t1).float()
        x2 = (T10.matches_fixed(x1, w2, 1, groups=8) > t2).float()
        x3 = (T10.matches_fixed(x2, w3, 0) > t3).float()
        s = torch.nn.functional.avg_pool2d(x3, 4) * 16.0
        fh = (s >= 8.0).float().flatten(1)
        mf = T10.matches_fixed(fh.reshape(-1, FEAT, 1, 1), wf.reshape(4, FEAT, 1, 1), 0).reshape(-1, 4)
        sc = (2 * mf - FEAT).long() * FC_SCALE + bias.view(1, 4)
    return sc.numpy()


if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'src', 'weights', 'hw_adapt_20261102', 'weights.hex')
    P = load_hex(p)
    print('[ok] %s  CRC=0x%04X' % (p, P['crc']))
