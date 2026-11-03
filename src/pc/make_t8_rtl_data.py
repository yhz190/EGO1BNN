#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_t8_rtl_data.py -- pack the T8 vectors for the 620-frame RTL regression.

Inputs
  data/vectors_t1a/*.bin + labels.csv          620 x 512 B frames (row-major, MSB first)
  data/t8_vectors_v2/golden.npz       L4 bits + L5 Q6 scores + pred
  board/weights/hw/weights.hex                  271 word v5 beat-major blob

Outputs (data/t8_rtl/)
  frame_words.hex     620 x 128 words, one word per line ($readmemh).  Word i of a
                      frame = bytes 4i..4i+3, little endian.
  scores.txt          620 lines "<f0> <f1> <f2> <f3> <pred>"  (Q6 integer, 19 bit signed)
  tb_t8_expected.hex  620 lines, one 80-bit word per frame for $readmemh in tb_t8_620.v:
                        [79:76] pred (4 pad bits, 0..3)
                        [75:57] f3   [56:38] f2   [37:19] f1   [18:0] f0
                      each score 19-bit two's complement
  README.txt

Self-checks (all must print [OK]; the script exits 1 on any [FAIL])
  [1] frame_words.hex has 79,360 words and round-trips to the 620 source .bin files
  [2] scores.txt has 620 lines, every score fits 19-bit signed, pred == argmax(f0..f3)
      (ties -> lower index, same rule as bnn_engine.cls_o)
  [3] weights.hex has 271 words and CRC16 (CCITT-FALSE over blob bits [0,8616)) == 0xB8C7
  [4] L5 Q6 scores recomputed from golden L4 + the FC field of weights.hex == scores.txt

usage:
  <python with numpy> make_t8_rtl_data.py
"""
import os
import sys
import csv
import hashlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VEC = os.path.join(ROOT, 'sim', 'vectors_t1a')
T8 = os.path.join(ROOT, 'sim', '20261101_t8_vectors_v2')
WHEX = os.path.join(ROOT, 'src', 'weights', 'hw', 'weights.hex')
OUT = os.path.join(ROOT, 'sim', '20261101_t8_rtl')
os.makedirs(OUT, exist_ok=True)

N_FRAMES = 620
N_WORDS = 128          # 128 x 32 bit = 512 B per frame
SCORE_MIN = -(1 << 18)
SCORE_MAX = (1 << 18) - 1
CRC16_EXPECT = 0xB8C7   # weights_layout_spec.md v5 / manifest.json
FC_BIT0, FC_BITS = 208, 8192
THFC_BIT0, THFC_BITS = 8552, 64

failures = []


def check(cond, msg):
    if cond:
        print('  [OK]   ' + msg)
    else:
        print('  [FAIL] ' + msg)
        failures.append(msg)
    return bool(cond)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def crc16_ccitt_false(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else ((crc << 1) & 0xFFFF)
    return crc


def read_hex_words(path):
    words = []
    for line in open(path, encoding='ascii'):
        line = line.strip()
        if line:
            words.append(int(line, 16))
    return words


def blob_bits(words, first, count):
    """bits[i] = blob bit (first+i), bit i of word i//32 (LSB first within word)."""
    out = np.empty(count, np.uint8)
    for i in range(count):
        b = first + i
        out[i] = (words[b >> 5] >> (b & 31)) & 1
    return out


def s19(v):
    return int(v) & 0x7FFFF


def main():
    rows = list(csv.DictReader(open(os.path.join(VEC, 'labels.csv'), encoding='utf-8-sig')))
    z = np.load(os.path.join(T8, 'golden.npz'))
    check(len(rows) == N_FRAMES, 'labels.csv frames = %d' % len(rows))
    check(len(z['labels']) == N_FRAMES, 'golden.npz frames = %d' % len(z['labels']))

    # ------------------------------------------------------------------ frames
    words = []
    raws = []
    for r in rows:
        raw = np.fromfile(os.path.join(VEC, r['file'].strip()), np.uint8)
        raws.append(raw)
        if len(raw) != 512:
            print('  [FAIL] %s size %d != 512' % (r['file'], len(raw)))
            failures.append('frame size')
        for i in range(128):
            w = (int(raw[4 * i]) | (int(raw[4 * i + 1]) << 8) |
                 (int(raw[4 * i + 2]) << 16) | (int(raw[4 * i + 3]) << 24))
            words.append(w)
    check(len(words) == N_FRAMES * N_WORDS, 'frame words = %d' % len(words))

    fw_path = os.path.join(OUT, 'frame_words.hex')
    with open(fw_path, 'w') as fh:
        for w in words:
            fh.write('%08x\n' % w)

    # ------------------------------------------------------------------ scores
    L5 = z['L5'].astype(np.int64)
    pred = z['pred'].astype(np.int64)
    sc_path = os.path.join(OUT, 'scores.txt')
    with open(sc_path, 'w') as fh:
        for i in range(N_FRAMES):
            fh.write('%d %d %d %d %d\n' % (L5[i, 0], L5[i, 1], L5[i, 2], L5[i, 3], pred[i]))

    # ------------------------------------------------- expected hex (80 bit/line)
    exp_path = os.path.join(OUT, 'tb_t8_expected.hex')
    with open(exp_path, 'w') as fh:
        for i in range(N_FRAMES):
            v = ((int(pred[i]) & 0xF) << 76 |
                 (s19(L5[i, 3]) << 57) | (s19(L5[i, 2]) << 38) |
                 (s19(L5[i, 1]) << 19) | s19(L5[i, 0]))
            fh.write('%020x\n' % v)

    with open(os.path.join(OUT, 'README.txt'), 'w') as fh:
        fh.write('frame_words.hex     : 620 x 128 words (512 B/frame, word i = bytes 4i..4i+3 little endian)\n')
        fh.write('scores.txt          : per line: f0 f1 f2 f3 pred  (Q6 integer scores from gen_golden_t8_v2.py)\n')
        fh.write('tb_t8_expected.hex  : 620 lines, 80 bit/frame: [79:76]=pred [75:57]=f3 [56:38]=f2 [37:19]=f1 [18:0]=f0\n')
        fh.write('                      (19-bit two\'s complement; one line per frame for $readmemh in tb_t8_620.v)\n')
        fh.write('weights             : board/weights/hw/weights.hex (271 words, v5 beat-major, CRC16 0xB8C7)\n')
        fh.write('frames              : data/vectors_t1a/*.bin, order = labels.csv order\n')
        fh.write('generator           : src/pc/make_t8_rtl_data.py (self-checks + SHA256 below)\n')

    print('\n[check 1] frame_words.hex round-trip vs source .bin')
    back = read_hex_words(fw_path)
    check(len(back) == N_FRAMES * N_WORDS, 'readmemh layout: %d words' % len(back))
    bad_bytes = 0
    for k in range(N_FRAMES):
        for i in range(N_WORDS):
            w = back[k * N_WORDS + i]
            b = bytes((w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF, (w >> 24) & 0xFF))
            if b != raws[k][4 * i:4 * i + 4].tobytes():
                bad_bytes += 1
    check(bad_bytes == 0, 'all 620 frames x 512 B equal to vectors_t1a (mismatched words %d)' % bad_bytes)

    print('\n[check 2] scores.txt / tb_t8_expected.hex')
    srows = [list(map(int, ln.split())) for ln in open(sc_path) if ln.strip()]
    check(len(srows) == N_FRAMES, 'scores.txt lines = %d' % len(srows))
    bad_rng = sum(1 for r in srows for v in r[:4] if not (SCORE_MIN <= v <= SCORE_MAX))
    check(bad_rng == 0, 'all %d scores inside 19-bit signed [%d,%d]' % (N_FRAMES * 4, SCORE_MIN, SCORE_MAX))
    bad_pred = 0
    for k, r in enumerate(srows):
        am = int(np.argmax(r[:4]))          # first maximum -> same tie rule as cls_o
        if am != r[4] or r[:4] != [int(v) for v in L5[k]]:
            bad_pred += 1
    check(bad_pred == 0, 'pred == argmax(f0..f3) (ties->lower) and f == golden.npz L5 for 620/620')

    expw = read_hex_words(exp_path)
    check(len(expw) == N_FRAMES, 'tb_t8_expected.hex lines = %d' % len(expw))
    bad_exp = 0
    for k in range(N_FRAMES):
        v = expw[k]
        e0 = v & 0x7FFFF
        e1 = (v >> 19) & 0x7FFFF
        e2 = (v >> 38) & 0x7FFFF
        e3 = (v >> 57) & 0x7FFFF
        ep = (v >> 76) & 0xF
        packed = [e0, e1, e2, e3]
        if packed != [s19(x) for x in L5[k]] or ep != int(pred[k]):
            bad_exp += 1
    check(bad_exp == 0, 'expected hex round-trips to golden L5/pred for 620/620')

    print('\n[check 3] weights.hex')
    w = read_hex_words(WHEX)
    check(len(w) == 271, 'weights.hex words = %d' % len(w))
    bits = blob_bits(w, 0, 8616)
    payload = np.packbits(bits, bitorder='little').tobytes()   # LSB-first within byte
    crc = crc16_ccitt_false(payload)
    check(len(payload) == 1077, 'CRC payload bytes = %d' % len(payload))
    check(crc == CRC16_EXPECT, 'CRC16 over blob bits [0,8616) = 0x%04X (expect 0x%04X)' % (crc, CRC16_EXPECT))
    idb = blob_bits(w, 8616, 16)
    idv = sum(int(idb[k]) << k for k in range(16))
    check(idv == 0x0001, 'blob ID field [8616,8632) = 0x%04X' % idv)

    print('\n[check 4] L5 Q6 recomputed from golden L4 + FC field of weights.hex')
    fc_bits = blob_bits(w, FC_BIT0, FC_BITS)          # [q][c][j] -> feature 8q+j
    grp = fc_bits.reshape(256, 4, 8)
    fc = np.zeros((4, 2048), np.uint8)
    for c in range(4):
        fc[c] = grp[:, c, :].reshape(2048)
    thfc_bits = blob_bits(w, THFC_BIT0, THFC_BITS)
    bias = np.zeros(4, np.int64)
    for c in range(4):
        v = sum(int(thfc_bits[c * 16 + k]) << k for k in range(16))
        bias[c] = v - 65536 if v >= 32768 else v
    feat = np.unpackbits(z['L4'], axis=1)[:, :2048].astype(np.uint8)
    mf = np.zeros((N_FRAMES, 4), np.int64)
    for c in range(4):
        mf[:, c] = (feat == fc[c][None, :]).sum(1)
    score = (2 * mf - 2048) * 64 + bias[None, :]
    check(bool(np.array_equal(score, L5)), 'recomputed Q6 == scores.txt for %d/%d frames'
          % (int((score == L5).all(1).sum()), N_FRAMES))
    check(bias.tolist() == [-1, 0, 0, 2], 'Q6 bias = %s' % bias.tolist())

    print('\n[sha256]')
    for name in ('frame_words.hex', 'scores.txt', 'tb_t8_expected.hex', 'README.txt'):
        p = os.path.join(OUT, name)
        print('  %-20s %s  %d B' % (name, sha256(p), os.path.getsize(p)))
    print('  %-20s %s' % ('weights.hex', sha256(WHEX)))

    print('\n[RESULT] %s' % ('ALL DATA CHECKS PASS' if not failures else
                             'FAIL (%d): %s' % (len(failures), '; '.join(failures))))
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
