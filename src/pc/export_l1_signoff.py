#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""export_l1_signoff.py -- export RTL-readable vectors for the L1 module-level sign-off.

Outputs (into <local>/build_sim_bnn/l1_signoff/data):
  frame_<k>.hex    : 64 lines x 16 hex chars  (frame bits, MSB-first per line)
  w1.hex           : 18 hex chars = 72 bit     (L1 weights, order [out_ch][ky][kx])
  theta.hex        : 2 hex chars per channel (8 lines) integer theta (floor)
  expect_<k>.txt   : 4096 lines, each "c0 c1 ... c7 act0 ... act7"
Source: data/t8_vectors/golden.npz + weights/E_E1_fixsplit_seed1.pt
"""
import os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get('EGO1_LOCAL_DIR', r'<local>') + r'/build_sim_bnn/l1_signoff/data'
sys.path.insert(0, os.path.join(ROOT, 'src', 'scripts'))
import train_bnn10 as T10

os.makedirs(OUT, exist_ok=True)
CK = os.path.join(ROOT, 'src', 'weights', 'E_E1_fixsplit_seed1.pt')
z = np.load(os.path.join(ROOT, 'sim', '20261101_t8_vectors', 'golden.npz'))

m = T10.Net(4)
m.load_state_dict(torch.load(CK, map_location='cpu'))
m.eval()

w1 = (m.c1.weight > 0).float().numpy().astype(np.uint8).reshape(-1)     # [8,1,3,3] -> 72
assert w1.size == 72, w1.size
t1 = T10.theta_of(m.b1, m.c1.weight, 9, False).floor().long().numpy()   # 8 ints
print('[l1] weights one bits =', int(w1.sum()), '/72')
print('[l1] theta int =', t1.tolist())

with open(os.path.join(OUT, 'w1.hex'), 'w') as fh:
    # 72 bit as 18 hex chars, MSB = w1[0]
    bits = ''.join(str(int(b)) for b in w1)[::-1]   # reversed: w1[0] -> LSB (lane tap0)
    bits = bits.ljust(72, '0')
    fh.write(''.join('%x' % int(bits[i:i + 4], 2) for i in range(0, 72, 4)) + '\n')
with open(os.path.join(OUT, 'theta.hex'), 'w') as fh:
    for v in t1.tolist():
        fh.write('%02x\n' % (int(v) & 0xFF))

# frames: golden.npz has no raw frame, rebuild from vectors_t1a
import csv
VEC = os.path.join(ROOT, 'sim', 'vectors_t1a')
rows = list(csv.DictReader(open(os.path.join(VEC, 'labels.csv'), encoding='utf-8-sig')))
NF = 4
for k in range(NF):
    fn = z['files'][k]
    raw = np.fromfile(os.path.join(VEC, fn), np.uint8)
    bits = np.unpackbits(raw)[:4096].reshape(64, 64).astype(np.uint8)
    with open(os.path.join(OUT, 'frame_%d.hex' % k), 'w') as fh:
        for y in range(64):
            fh.write(''.join('%x' % int(''.join(str(int(v)) for v in bits[y, x * 4:x * 4 + 4]), 2)
                             for x in range(16)) + '\n')
    c1 = z['c1'][k]                                  # [8,64,64]
    l1 = np.unpackbits(z['L1'][k], axis=1).reshape(8, 64, 64)
    with open(os.path.join(OUT, 'expect_%d.txt' % k), 'w') as fh:
        for y in range(64):
            for x in range(64):
                cs = ' '.join('%d' % c1[c, y, x] for c in range(8))
                acts = ' '.join('%d' % l1[c, y, x] for c in range(8))
                fh.write('%s %s\n' % (cs, acts))
    print('[frame %d] %s  labels=%d pred=%d  c1 sum=%d act sum=%d'
          % (k, str(fn), int(z['labels'][k]), int(z['pred'][k]), int(c1.sum()), int(l1.sum())))
print('[out]', OUT)
