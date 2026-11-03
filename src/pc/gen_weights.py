#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gen_weights.py -- export the patch2 (8ch + Q6) weight blob for RTL.

Layout: src/weights_layout_spec.md v5  (8,648 bit = 271 x 32-bit words)

v5 (2026-11-01, beat-major FC): the FC field is stored in the *beat-major* order that
l5_fc consumes directly, so bnn_wblob needs no fc_ram cache and no de-interleave pass
(that cache was 2,769 LUT / 8,788 FF after synthesis -- see sim/20261101_beatmajor/).
Only the storage order changed; every weight value, the offsets, the field widths and
the CRC16 coverage are identical to v4.

Bit order (authoritative, unambiguous):
    blob bit index i        -> word w = i // 32, bit k = i % 32 (LSB-first within the word)
    hex line w              -> the 32-bit word value printed as 8 hex chars
Fields:
    [0,72)      L1 weights       8 ch x 9 tap, lane c tap t -> bit c*9+t
    [72,144)    L2 weights       same packing
    [144,208)   L3 weights       8 out x 8 in, out o in i -> bit o*8+i
    [208,8400)  FC weights       beat-major (v5):
                                  bit 208+32*q+8*c+j = fc_w[c][8*q+j]
                                  i.e. beat word q = {b3,b2,b1,b0}, b_c = fc_w[c][8q+7:8q]
    [8400,8432) theta1           8 x 4bit  (ch*4)
    [8432,8464) theta2           8 x 4bit
    [8464,8496) theta3           8 x 4bit
    [8496,8552) theta4           8 x 7bit
    [8552,8616) thetaFC          4 x 16bit Q6 signed two's complement
    [8616,8632) ID               0x0001
    [8632,8648) CRC16            CCITT-FALSE over blob bits [0,8616) = 1077 bytes
"""
import os, sys, json, hashlib
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'scripts'))
import train_bnn10 as T10
import train_l3x8 as L8
from class_map import assert_class_map

assert_class_map()
FC_SCALE = 64

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--ckpt', default=os.path.join(ROOT, 'src', 'weights', 'FORMAL_l3x8_s4.pt'))
_ap.add_argument('--outdir', default=os.path.join(ROOT, 'src', 'weights', 'hw'))
_A = _ap.parse_args()
CKPT = _A.ckpt
OUTDIR = _A.outdir
os.makedirs(OUTDIR, exist_ok=True)

FIELDS = [
    ('L1', 72), ('L2', 72), ('L3', 64), ('FC', 8192),
    ('theta1', 32), ('theta2', 32), ('theta3', 32), ('theta4', 56), ('thetaFC', 64),
    ('ID', 16), ('CRC16', 16),
]
TOTAL = sum(b for _, b in FIELDS)
assert TOTAL == 8648, TOTAL


def crc16_ccitt_false(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else ((crc << 1) & 0xFFFF)
    return crc


def main():
    sd = torch.load(CKPT, map_location='cpu')
    m = L8.Net8(); m.load_state_dict(sd); m.eval()
    feat = m.feat
    assert feat == 2048 and m.c3.weight.shape[0] == 8, 'expect 8ch/2048 model'

    w1 = (m.c1.weight > 0).numpy().astype(np.uint8).reshape(-1)     # 72
    w2 = (m.dw.weight > 0).numpy().astype(np.uint8).reshape(-1)     # 72
    w3 = (m.c3.weight > 0).numpy().astype(np.uint8).reshape(-1)     # 64
    wf = (m.fc.weight > 0).numpy().astype(np.uint8).reshape(-1)     # 8192
    t1 = T10.theta_of(m.b1, m.c1.weight, 9, False).floor().long()
    t2 = T10.theta_of(m.b2, m.dw.weight, 9, True).floor().long()
    t3 = T10.theta_of(m.b3, m.c3.weight, 8, True).floor().long()
    t4 = torch.full((8,), 8, dtype=torch.long)                       # 4x4 block majority >= 8/16
    bias_q = (m.fc.bias.detach() * FC_SCALE).round().long()

    bits = []
    def put(vals, width):
        for v in vals:
            v = int(v)
            for k in range(width):
                bits.append((v >> k) & 1)

    put(w1, 1); put(w2, 1); put(w3, 1)
    # ---- FC field, layout v5: beat-major ----
    # beat q packs the 4 classes' beat-q bytes as {b3,b2,b1,b0}; within a byte the
    # feature bit order stays LSB-first (feature 8q+j -> byte bit j).  This is
    # exactly the 32-bit word l5_fc writes on beat q, so the RTL can stream the
    # blob straight into l5_fc's wmem without a cache / de-interleave pass.
    for q in range(256):
        for c in range(4):
            for j in range(8):
                bits.append(int(wf[c * feat + 8 * q + j]))
    assert len(bits) == 8400, len(bits)
    put(t1.tolist(), 4); put(t2.tolist(), 4); put(t3.tolist(), 4)
    put(t4.tolist(), 7)
    put([(int(v) & 0xFFFF) for v in bias_q.tolist()], 16)
    put([0x0001], 16)
    assert len(bits) == 8632, len(bits)

    # CRC16 over the first 8616 bits = 1077 bytes
    def bits_to_bytes(bs):
        out = bytearray()
        for i in range(0, len(bs), 8):
            chunk = bs[i:i + 8]
            while len(chunk) < 8:
                chunk.append(0)
            v = 0
            for k, b in enumerate(chunk):
                v |= (b & 1) << k          # LSB-first within byte
            out.append(v)
        return bytes(out)

    payload = bits_to_bytes(bits[:8616])
    assert len(payload) == 1077, len(payload)
    crc = crc16_ccitt_false(payload)
    put([crc], 16)
    assert len(bits) == 8648, len(bits)

    # pad to a whole number of 32-bit words (271 words = 8,672 bits; 24 pad bits)
    NW = (TOTAL + 31) // 32
    while len(bits) < NW * 32:
        bits.append(0)
    words = []
    for w0 in range(NW):
        v = 0
        for k in range(32):
            v |= bits[w0 * 32 + k] << k
        words.append(v)

    hex_path = os.path.join(OUTDIR, 'weights.hex')
    with open(hex_path, 'w') as fh:
        for w in words:
            fh.write('%08x\n' % w)

    # ---- structural check of the v5 FC beat layout (independent of the readback) ----
    # FC starts at bit 208 = word 6 bit 16, so beat q straddles words 6+q / 7+q:
    #   beat_word(q) = { words[7+q][15:0], words[6+q][31:16] }
    # and must equal {fc_w[3][8q+:8], fc_w[2][8q+:8], fc_w[1][8q+:8], fc_w[0][8q+:8]}.
    for q in range(256):
        exp_w = 0
        for c in range(4):
            for j in range(8):
                exp_w |= int(wf[c * feat + 8 * q + j]) << (8 * c + j)
        got_w = ((words[6 + q] >> 16) | ((words[7 + q] & 0xFFFF) << 16)) & 0xFFFFFFFF
        assert got_w == exp_w, 'beat %d word mismatch: %08x != %08x' % (q, got_w, exp_w)

    manifest = {
        'layout_version': 'v5',
        'bits': TOTAL,
        'words': len(words),
        'arch': 'L3=8ch, feat=2048, FC=4x2048',
        'fc_bias': 'Q6 signed 16-bit (FC_SCALE=64)',
        'fc_order': 'beat-major',
        'fc_bit_map': 'bit 208+32*q+8*c+j = fc_w[c][8*q+j]; beat word q = {b3,b2,b1,b0}',
        'fc_word_map': 'words[6+q][31:16] = beat q low 16, words[7+q][15:0] = beat q high 16',
        'theta': 'floor',
        'crc16': '0x%04X' % crc,
        'crc16_poly': '0x1021', 'crc16_init': '0xFFFF', 'crc16_cover_bits': 8616,
        'model': os.path.basename(CKPT),
        'model_md5': hashlib.md5(open(CKPT, 'rb').read()).hexdigest(),
        'theta_int': {'L1': t1.tolist(), 'L2': t2.tolist(), 'L3': t3.tolist(), 'L4': t4.tolist()},
        'bias_q': bias_q.tolist(),
        'bit_order': 'blob bit i -> word i//32 bit i%32 (LSB-first within word)',
    }

    # ---------------- self-check: re-parse hex, rebuild weights, compare Q6 forward ----------------
    back = []
    for line in open(hex_path):
        v = int(line.strip(), 16)
        for k in range(32):
            back.append((v >> k) & 1)
    # 8,648 bits are packed into 271 words = 8,672 bits; the last 24 bits are padding
    assert len(back) == len(words) * 32 == 8672, len(back)
    back = back[:TOTAL]
    pos = 0
    def take(width, n):
        nonlocal pos
        vals = []
        for _ in range(n):
            v = 0
            for k in range(width):
                v |= back[pos + k] << k
            pos += width
            vals.append(v)
        return vals

    w1b = np.array(take(1, 72), np.uint8)
    w2b = np.array(take(1, 72), np.uint8)
    w3b = np.array(take(1, 64), np.uint8)
    # v5: the FC field in the hex is beat-major; rebuild the class-major [4][2048]
    # matrix exactly as l5_fc will consume it (fc_w[c][8q+j] at field bit 32q+8c+j).
    fcb = take(1, 8192)
    wfb = np.zeros(8192, np.uint8)
    for q in range(256):
        for c in range(4):
            for j in range(8):
                wfb[c * feat + 8 * q + j] = fcb[32 * q + 8 * c + j]
    t1b = take(4, 8); t2b = take(4, 8); t3b = take(4, 8); t4b = take(7, 8)
    bq = [v - 65536 if v >= 32768 else v for v in take(16, 4)]
    ident = take(16, 1)[0]
    crc_rb = take(16, 1)[0]
    assert ident == 0x0001 and crc_rb == crc, 'ID/CRC readback mismatch'
    assert w1b.tolist() == w1.tolist() and wfb.tolist() == wf.tolist(), 'weight readback mismatch'
    manifest['self_check'] = ('hex readback == source weights/theta; '
                              'FC beat-major de-interleave == fc_w; ID/CRC ok')

    # Q6 forward with the parsed weights, on the 620-frame test set
    import csv
    VEC = os.path.join(ROOT, 'sim', 'vectors_t1a')
    rows = list(csv.DictReader(open(os.path.join(VEC, 'labels.csv'), encoding='utf-8-sig')))
    X = np.zeros((len(rows), 1, 64, 64), np.float32)
    for i, r in enumerate(rows):
        X[i, 0] = np.unpackbits(np.fromfile(os.path.join(VEC, r['file'].strip()), np.uint8))[:4096].reshape(64, 64)
    Xt = torch.from_numpy(X)
    yt = torch.tensor([int(r['class']) for r in rows])
    w1t = torch.from_numpy(w1b).float().reshape(8, 1, 3, 3)
    w2t = torch.from_numpy(w2b).float().reshape(8, 1, 3, 3)
    w3t = torch.from_numpy(w3b).float().reshape(8, 8, 1, 1)
    wft = torch.from_numpy(wfb).float().reshape(4, feat)
    with torch.no_grad():
        x1 = (T10.matches_fixed(Xt, w1t, 1) > torch.tensor(t1b).view(1, -1, 1, 1)).float()
        x2 = (T10.matches_fixed(x1, w2t, 1, groups=8) > torch.tensor(t2b).view(1, -1, 1, 1)).float()
        x3 = (T10.matches_fixed(x2, w3t, 0) > torch.tensor(t3b).view(1, -1, 1, 1)).float()
        s = torch.nn.functional.avg_pool2d(x3, 4) * 16.0
        fh = (s >= 8.0).float().flatten(1)
        mf = T10.matches_fixed(fh.reshape(-1, feat, 1, 1), wft.reshape(4, feat, 1, 1), 0).reshape(-1, 4)
        sc = (2 * mf - feat).long() * FC_SCALE + torch.tensor(bq).view(1, 4)
        pred_from_hex = sc.argmax(1)
        pred_ref = T10.hw_eval_q6(m, Xt).argmax(1)
    agree = int((pred_from_hex == pred_ref).sum())
    acc = float((pred_from_hex == yt).float().mean())
    manifest['acc_q6_on_vectors'] = acc
    manifest['hex_vs_hw_eval_q6_agree'] = '%d/%d' % (agree, len(yt))
    print('[hex->weights] readback OK, CRC 0x%04X, ID 0x0001' % crc)
    print('[check] hex-forward vs hw_eval_q6 argmax: %d/%d' % (agree, len(yt)))
    print('[acc]   %.4f (vectors_t1a, cross-domain)' % acc)
    if agree != len(yt):
        print('[FAIL] hex rebuild disagrees with reference -> blob not usable')
        return 1

    json.dump(manifest, open(os.path.join(OUTDIR, 'manifest.json'), 'w'), indent=2, ensure_ascii=False)
    print('[out]', hex_path, '(%d lines)' % len(words))
    print('[out]', os.path.join(OUTDIR, 'manifest.json'))
    print('[PASS] gen_weights v5 (beat-major FC) complete')
    return 0


if __name__ == '__main__':
    sys.exit(main())
