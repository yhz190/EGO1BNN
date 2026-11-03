#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""export_layer_signoff.py -- export RTL-readable vectors for L2/L3 module-level sign-off.

L2 (8 planes, depthwise 3x3): inputs are the L1 activation bitmaps from the T8 golden set.
L3 (4 out x 8 in, 1x1):      inputs are the L2 activation bitmaps; no window buffer.

Files written into <local>/build_sim_bnn/l2_signoff/data (and l3_signoff/data):
  l2: w2.hex (72b), theta2.hex (8 x 2 hex), plane_<k>_<p>.hex (64x16 hex per plane),
      expect2_<k>.txt (4096 lines: 8 counts + 8 acts)
  l3: w3.hex (32b), theta3.hex (4 x 2 hex), in_<k>_<p>.hex (8 planes, 64x16 hex),
      expect3_<k>.txt (4096 lines: 4 counts + 4 acts)
"""
import os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'scripts'))
import train_bnn10 as T10

GOLD = os.path.join(ROOT, 'sim', '20261101_t8_vectors', 'golden.npz')
CK = os.path.join(ROOT, 'src', 'weights', 'E_E1_fixsplit_seed1.pt')
z = np.load(GOLD)

m = T10.Net(4)
m.load_state_dict(torch.load(CK, map_location='cpu'))
m.eval()

NF = 4


def hex_rows(bits2d):                      # (64,64) -> 64 lines of 16 hex chars, MSB=x0
    out = []
    for y in range(64):
        row = bits2d[y]
        out.append(''.join('%x' % int(''.join(str(int(v)) for v in row[x*4:x*4+4]), 2) for x in range(64 // 4)))
    return out


def wbits_hex(warr, nbits, reverse=True):
    b = ''.join(str(int(v)) for v in warr.reshape(-1))
    if reverse:
        b = b[::-1]
    b = b.ljust(nbits, '0')
    return ''.join('%x' % int(b[i:i+4], 2) for i in range(0, nbits, 4))


# ---------------- L2 ----------------
D2 = os.environ.get('EGO1_LOCAL_DIR', r'<local>') + r'/build_sim_bnn/l2_signoff/data'
os.makedirs(D2, exist_ok=True)
w2 = (m.dw.weight > 0).float().numpy().astype(np.uint8)      # [8,1,3,3]
t2 = T10.theta_of(m.b2, m.dw.weight, 9, True).floor().long().numpy()
open(os.path.join(D2, 'w2.hex'), 'w').write(wbits_hex(w2, 72) + '\n')
open(os.path.join(D2, 'theta2.hex'), 'w').write('\n'.join('%02x' % int(v) for v in t2.tolist()) + '\n')
print('[L2] weights one bits =', int(w2.sum()), '/72   theta =', t2.tolist())

for k in range(NF):
    l1 = np.unpackbits(z['L1'][k], axis=1).reshape(8, 64, 64)     # (8,64,64)
    l2 = np.unpackbits(z['L2'][k], axis=1).reshape(8, 64, 64)
    c2 = z['c2'][k]
    for p in range(8):
        open(os.path.join(D2, 'plane_%d_%d.hex' % (k, p)), 'w').write('\n'.join(hex_rows(l1[p])) + '\n')
    with open(os.path.join(D2, 'expect2_%d.txt' % k), 'w') as fh:
        for y in range(64):
            for x in range(64):
                cs = ' '.join('%d' % c2[c, y, x] for c in range(8))
                acts = ' '.join('%d' % l2[c, y, x] for c in range(8))
                fh.write('%s %s\n' % (cs, acts))
    print('[L2] frame %d c2sum=%d act2sum=%d' % (k, int(c2.sum()), int(l2.sum())))

# ---------------- L3 ----------------
D3 = os.environ.get('EGO1_LOCAL_DIR', r'<local>') + r'/build_sim_bnn/l3_signoff/data'
os.makedirs(D3, exist_ok=True)
w3 = (m.c3.weight > 0).float().numpy().astype(np.uint8)      # [4,8,1,1]
t3 = T10.theta_of(m.b3, m.c3.weight, 8, True).floor().long().numpy()
open(os.path.join(D3, 'w3.hex'), 'w').write(wbits_hex(w3, 32) + '\n')
open(os.path.join(D3, 'theta3.hex'), 'w').write('\n'.join('%02x' % int(v) for v in t3.tolist()) + '\n')
print('[L3] weights one bits =', int(w3.sum()), '/32   theta =', t3.tolist())

for k in range(NF):
    l2 = np.unpackbits(z['L2'][k], axis=1).reshape(8, 64, 64)
    l3 = np.unpackbits(z['L3'][k], axis=1).reshape(4, 64, 64)
    c3 = z['c3'][k]
    for p in range(8):
        open(os.path.join(D3, 'in_%d_%d.hex' % (k, p)), 'w').write('\n'.join(hex_rows(l2[p])) + '\n')
    with open(os.path.join(D3, 'expect3_%d.txt' % k), 'w') as fh:
        for y in range(64):
            for x in range(64):
                cs = ' '.join('%d' % c3[c, y, x] for c in range(4))
                acts = ' '.join('%d' % l3[c, y, x] for c in range(4))
                fh.write('%s %s\n' % (cs, acts))
    print('[L3] frame %d c3sum=%d act3sum=%d' % (k, int(c3.sum()), int(l3.sum())))

print('[out]', D2, 'and', D3)
