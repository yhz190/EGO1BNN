#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""golden.py -- bit-exact reference model of the 5-layer BNN (64x64 binary input).

Layers (matches plan v2.2):
  L1  conv 3x3, 1 -> 8 ch,  binarize
  L2  depthwise 3x3, 8 -> 8, binarize
  L3  conv 1x1, 8 -> 4,     binarize
  L4  pool 8x8 majority      -> 4 ch x 8 x 8 = 256 bits
  L5  dense 256 -> 4 (incl. junk channel)

Weight budget (plan v2.2): 1244 bits total
  L1 8*9=72 | L2 8*9=72 | L3 4*8=32 | L5 4*256=1024
  thresholds 4*8=32 | reserved 12   -> 72+72+32+1024+32+12 = 1244
"""
import argparse, struct, sys
import numpy as np

H = W = 64
C1, C3, C5 = 8, 4, 4
W_TOTAL_BITS = 1244

def load_frame(path):
    raw = open(path, 'rb').read()
    assert len(raw) == 512, 'frame must be 512 bytes, got %d' % len(raw)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    return bits.reshape(H, W).astype(np.uint8)

def pattern(name):
    img = np.zeros((H, W), dtype=np.uint8)
    if name == 'checker':
        img[::8, ::8] = 1
        for y in range(0, H, 8):
            for x in range(0, W, 8):
                img[y:y+8, x:x+8] = ((y//8 + x//8) & 1)
    elif name == 'circle':
        yy, xx = np.mgrid[0:H, 0:W]
        img[((xx-32)**2 + (yy-32)**2) <= 400] = 1
    return img

def make_weights(seed=1234):
    rng = np.random.default_rng(seed)
    w = {
        'L1': rng.integers(0, 2, size=(C1, 3, 3)).astype(np.uint8),
        'L2': rng.integers(0, 2, size=(C1, 3, 3)).astype(np.uint8),
        'L3': rng.integers(0, 2, size=(C3, C1)).astype(np.uint8),
        'L5': rng.integers(0, 2, size=(C5, 256)).astype(np.uint8),
        'th': rng.integers(1, 8, size=(4,)).astype(np.int32),
    }
    nbits = w['L1'].size + w['L2'].size + w['L3'].size + w['L5'].size
    w['nbits_weights'] = nbits
    return w

def conv3x3_bin(img, kern):
    """binary 3x3 conv with SAME padding (zero pad), returns popcount score map.

    XNOR convention: out-of-range pixels are 0; when kern bit==0 we count (1-pixel).
    """
    h, w = img.shape
    padded = np.pad(img, 1, mode='constant', constant_values=0)
    out = np.zeros((h, w), dtype=np.int32)
    for dy in range(3):
        for dx in range(3):
            sub = padded[dy:dy+h, dx:dx+w]
            if kern[dy, dx]:
                out += sub
            else:
                out += (1 - sub)
    return out

def binarize(x, th):
    return (x >= th).astype(np.uint8)

def forward(img, w, verbose=True):
    # L1: 1->8
    l1 = np.zeros((C1, H, W), dtype=np.uint8)
    for c in range(C1):
        l1[c] = binarize(conv3x3_bin(img, w['L1'][c]), w['th'][0])
    # L2: depthwise
    l2 = np.zeros((C1, H, W), dtype=np.uint8)
    for c in range(C1):
        l2[c] = binarize(conv3x3_bin(l1[c], w['L2'][c]), w['th'][1])
    # L3: 1x1
    h3 = l2.shape[1]
    l3 = np.zeros((C3, h3, h3), dtype=np.uint8)
    for o in range(C3):
        acc = np.zeros((h3, h3), dtype=np.int32)
        for i in range(C1):
            acc += np.where(w['L3'][o, i] == 1, l2[i], 1 - l2[i])
        l3[o] = binarize(acc, w['th'][2])
    # L4: 8x8 majority pool
    s = h3 // 8
    l4 = np.zeros((C3, s, s), dtype=np.uint8)
    for c in range(C3):
        blk = l3[c, :s*8, :s*8].reshape(s, 8, s, 8).sum(axis=(1, 3))
        l4[c] = (blk >= 32).astype(np.uint8)
    feat = l4.reshape(-1)
    # L5: dense 256 -> 4
    scores = []
    for o in range(C5):
        same = np.where(w['L5'][o] == 1, feat, 1 - feat).sum()
        scores.append(int(same))
    return dict(l1=l1, l2=l2, l3=l3, l4=l4, feat=feat, scores=scores)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame', help='raw 512-byte binary frame')
    ap.add_argument('--pattern', default='circle', choices=['circle', 'checker'])
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--dump', default=None, help='prefix to dump layer bitmaps')
    a = ap.parse_args()

    img = load_frame(a.frame) if a.frame else pattern(a.pattern)
    w = make_weights(a.seed)
    print('[weights] bits = %d (plan total %d incl. thresholds/reserved)'
          % (w['nbits_weights'], W_TOTAL_BITS))
    r = forward(img, w)
    print('[L1] %s  ones=%d' % (r['l1'].shape, int(r['l1'].sum())))
    print('[L2] %s  ones=%d' % (r['l2'].shape, int(r['l2'].sum())))
    print('[L3] %s  ones=%d' % (r['l3'].shape, int(r['l3'].sum())))
    print('[L4] %s  ones=%d' % (r['l4'].shape, int(r['l4'].sum())))
    print('[L5] feat_bits=%d scores=%s' % (r['feat'].size, r['scores']))
    s = r['scores']
    cls = int(np.argmax(s))
    print('[out] class=%d (0=rock 1=paper 2=scissors 3=junk)' % cls)
    if a.dump:
        np.save(a.dump + '_l4.npy', r['l4'])
        print('[dump] %s_l4.npy' % a.dump)
    return 0

if __name__ == '__main__':
    sys.exit(main())