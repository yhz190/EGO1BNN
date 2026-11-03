#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_bnn2.py -- improved recipe: real-valued CNN + BN, then binarize + calibrate.

Why: training with hard sign() from layer 1 (train_bnn.py) converged to ~0.5.
New recipe:
  Phase A  train real-valued CNN (conv/BN) with CE loss             -> high accuracy
  Phase B  binarize weights (sign) + calibrate per-layer thresholds  -> keep accuracy
  Phase C  evaluate binarized model, export 1244-bit weights.hex
Junk class is derived from REAL images (partial crops / rotated / blank), not noise only.

usage:
  python train_bnn2.py --data <dir> --epochs 30 --out <weights dir>
"""
import os
import argparse, os, json, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SIZE = 64
CLASSES = ['rock', 'paper', 'scissors', 'junk']
LAYOUT = {'c1': (8, 1, 3, 3), 'dw': (8, 1, 3, 3), 'c3': (4, 8, 1, 1), 'fc': (4, 256)}
BITS_ORDER = [('c1', 72), ('dw', 72), ('c3', 32), ('fc', 1024)]

# ------------------------------------------------------------------ data
def otsu(a):
    hist, _ = np.histogram(a, bins=256, range=(0.0, 1.0))
    total = a.size; sum_all = np.dot(np.arange(256), hist)
    wb = 0.0; sb = 0.0; best = -1.0; bt = 127
    for t in range(256):
        wb += hist[t]
        if wb == 0: continue
        wf = total - wb
        if wf == 0: break
        sb += t * hist[t]
        mb = sb / wb; mf = (sum_all - sb) / wf
        v = wb * wf * (mb - mf) ** 2
        if v > best: best, bt = v, t
    return bt / 255.0

def load_bin(path, flip=False, shift=0):
    g = Image.open(path).convert('L')
    w, h = g.size; s = min(w, h)
    g = g.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2)).resize((SIZE, SIZE), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32) / 255.0
    t = otsu(a)
    b = (a > t).astype(np.float32)          # 1 = bright (hand)
    if flip:
        b = b[:, ::-1]
    if shift:
        b = np.roll(b, shift, axis=1)
    return b

def junk_from_images(paths, n, rng):
    out = []
    for _ in range(n):
        kind = rng.integers(0, 3)
        if kind == 0:                                   # partial crop (hand cut off)
            b = load_bin(paths[rng.integers(len(paths))])
            h = b.shape[0]; crop = int(h * rng.uniform(0.35, 0.6))
            x0 = rng.integers(0, h - crop); y0 = rng.integers(0, h - crop)
            patch = b[y0:y0+crop, x0:x0+crop]
            b = np.asarray(Image.fromarray((patch*255).astype(np.uint8)).resize((SIZE, SIZE)), np.float32)/255.0
            b = (b > 0.5).astype(np.float32)  # FIX: binarize after resize
        elif kind == 1:                                 # rotated (side hand)
            b = load_bin(paths[rng.integers(len(paths))])
            b = np.rot90(b, k=int(rng.integers(1, 4)))
        else:                                           # blank / flat background
            v = rng.uniform(0.0, 0.6)
            b = np.full((SIZE, SIZE), v, np.float32)
            b = (b > 0.45).astype(np.float32)
        out.append(b)
    # FIX 2026-11-01: junk must be strictly binary (1-bit FRAME)
    out = [(x > 0.5).astype(np.float32) for x in out]
    for _x in out:
        assert set(np.unique(_x)).issubset({0.0, 1.0}), "junk must be binary"

    return out

def build_dataset(roots, per_class, seed=0, augment=True):
    rng = np.random.default_rng(seed)
    X, y = [], []
    real = []
    for ci, cls in enumerate(CLASSES[:3]):
        paths = []
        for r in roots:
            d = os.path.join(r, cls)
            if os.path.isdir(d):
                paths += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(('.png', '.jpg'))]
        real += paths
        if per_class:
            idx = rng.permutation(len(paths))[:per_class]
            paths = [paths[i] for i in idx]
        for p in paths:
            X.append(load_bin(p)); y.append(ci)
            if augment:
                X.append(load_bin(p, flip=True)); y.append(ci)
    nj = per_class * 2 if per_class else len(real)
    for b in junk_from_images(real, nj, rng):
        X.append(b); y.append(3)
    X = np.stack(X)[:, None, :, :]
    y = np.array(y)
    p = rng.permutation(len(X))
    return torch.from_numpy(X[p]).float(), torch.from_numpy(y[p]).long()

# ------------------------------------------------------------------ model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(8)
        self.dw = nn.Conv2d(8, 8, 3, padding=1, groups=8, bias=False)
        self.b2 = nn.BatchNorm2d(8)
        self.c3 = nn.Conv2d(8, 4, 1, bias=False)
        self.b3 = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4 * 8 * 8, 4)
    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.dw(x)))
        x = F.relu(self.b3(self.c3(x)))
        x = F.adaptive_avg_pool2d(x, 8)
        return self.fc(x.flatten(1))
    def acts(self, x):
        a = []
        x = F.relu(self.b1(self.c1(x))); a.append(x.detach())
        x = F.relu(self.b2(self.dw(x))); a.append(x.detach())
        x = F.relu(self.b3(self.c3(x))); a.append(x.detach())
        return a

# ------------------------------------------------------------------ binarize
def bin_forward(model, X, taus):
    """binary activations, binary weights; returns logits"""
    def sgn(x, t): return torch.where(x - t >= 0, 1.0, -1.0)
    w = model
    x = X
    # L1
    y = F.conv2d(x, w.c1.weight.sign(), None, 1, 1); x = sgn(y, taus[0])
    y = F.conv2d(x, w.dw.weight.sign(), None, 1, 1, groups=8); x = sgn(y, taus[1])
    y = F.conv2d(x, w.c3.weight.sign(), None, 1, 0); x = sgn(y, taus[2])
    x = F.avg_pool2d(x, 8)
    f = x.flatten(1)
    f = sgn(f, taus[3]) if taus[3] is not None else f
    return F.linear(f, w.fc.weight.sign(), None)

def evaluate(model, X, y, taus, bs=256):
    model.eval(); correct = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out = bin_forward(model, X[i:i+bs], taus)
            correct += (out.argmax(1) == y[i:i+bs]).sum().item()
    return correct / len(X)

def _init_taus(model, X):
    model.eval()
    with torch.no_grad():
        acts = model.acts(X)
    taus = []
    for a in acts:
        pos = a[a > 0]
        taus.append(float(torch.median(pos)) if pos.numel() else 0.0)
    taus.append(0.0)
    return taus

def calibrate(model, X, y, rounds=3):
    """joint coordinate-ascent: all 4 thresholds optimised together"""
    taus = _init_taus(model, X)
    print('  [calib] init taus =', ['%.3f' % t for t in taus])
    model.eval()
    with torch.no_grad():
        acts = model.acts(X)
    best_acc = evaluate(model, X, y, taus, bs=512)
    for r in range(rounds):
        for k in range(4):
            if k < 3:
                a = acts[k]
                pos = a[a > 0]
                hi = float(pos.max()) if pos.numel() else 1.0
                grid = np.linspace(0.0, max(hi * 0.9, 0.1), 13)
            else:
                grid = np.linspace(-0.6, 0.6, 13)
            for cand in grid:
                trial = list(taus); trial[k] = float(cand)
                acc = evaluate(model, X, y, trial, bs=512)
                if acc > best_acc:
                    best_acc, taus = acc, trial
            print('  [calib r%d] L%d tau=%.3f best_acc=%.4f' % (r + 1, k + 1, taus[k], best_acc))
    print('  [calib] final taus =', ['%.3f' % t for t in taus])
    return taus
def export(model, taus, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    bits = []
    for name, n in BITS_ORDER:
        w = getattr(model, {'c1':'c1','dw':'dw','c3':'c3','fc':'fc'}[name]).weight.detach()
        bits.append((w.reshape(-1) > 0).to(torch.int64))
    thr = []
    for t in taus[:4]:
        v = int(max(0, min(255, round(float(t) * 16))))
        thr += [(v >> b) & 1 for b in range(8)]
    total = torch.cat(bits).tolist() + thr + [0] * 12
    assert len(total) == 1244, len(total)
    words = []
    for i in range(0, 1244, 32):
        c = total[i:i+32] + [0] * (32 - len(total[i:i+32]))
        v = 0
        for b in c: v = (v << 1) | b
        words.append(v)
    with open(os.path.join(out_dir, 'weights.hex'), 'w', encoding='ascii') as f:
        for w in words: f.write('%08x\n' % w)
    json.dump({'layout': 'c1(8x9) dw(8x9) c3(4x8) fc(4x256) thr(4x8b) reserved(12)',
               'total_bits': 1244, 'words': len(words), 'thresholds': [float(t) for t in taus]},
              open(os.path.join(out_dir, 'weights.json'), 'w'), indent=2)
    return words

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(ROOT, r'data'))
    ap.add_argument('--per-class', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', default=os.path.join(ROOT, r'board/weights'))
    a = ap.parse_args()

    roots = [os.path.join(a.data, s) for s in ['rps', 'rps-test-set'] if os.path.isdir(os.path.join(a.data, s))]
    Xtr, ytr = build_dataset(roots, a.per_class, seed=1, augment=True)
    Xte, yte = build_dataset(roots, 124, seed=99, augment=False)
    print('[data] train=%d test=%d  per-class(train)=%s' % (len(Xtr), len(Xte), np.bincount(ytr.numpy(), minlength=4).tolist()))

    model = CNN()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr)
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(n); tot = 0.0; corr = 0
        for i in range(0, n, a.bs):
            idx = perm[i:i+a.bs]
            opt.zero_grad()
            out = model(Xtr[idx])
            loss = lossf(out, ytr[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx); corr += (out.argmax(1) == ytr[idx]).sum().item()
        sch.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            print('[ep %3d] loss=%.4f acc=%.4f' % (ep + 1, tot / n, corr / n))

    os.makedirs(a.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(a.out, 'model.pt'))
    print('[save] model.pt')

    # real-valued test accuracy
    model.eval()
    with torch.no_grad():
        te = sum((model(Xte[i:i+256]).argmax(1) == yte[i:i+256]).sum().item() for i in range(0, len(Xte), 256))
    print('[eval] real-valued test acc = %.4f' % (te / len(Xte)))

    print('[calib] grid search thresholds...')
    taus = calibrate(model, Xtr[:2048], ytr[:2048])
    bacc = evaluate(model, Xte, yte, taus)
    print('[eval] BINARIZED test acc = %.4f  (taus=%s)' % (bacc, ['%.3f' % t for t in taus]))

    words = export(model, taus, a.out)
    print('[export] %d words -> %s' % (len(words), a.out))

if __name__ == '__main__':
    main()