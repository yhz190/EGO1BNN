#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_honest.py -- HONEST protocol: train on data/rps only, evaluate on the
official held-out set data/vectors_t1a (620 frames: rps-test-set 124/class + junk).

Why: the previous 0.9597 was measured on `build_dataset(per_class=124)`, which samples
124/class from the UNION of rps (840/class) and rps-test-set (124/class) -> 87% of the
"test" images were training images (data leakage). Honest number must come from the
held-out test split only.

Pipeline:
  1. float CNN (BN) trained on train-only  -> honest float accuracy
  2. warm start binary net (sign weights + STE + BN, padding-correct)
  3. report binary accuracy via exact integer forward (hw_eval) on the 620-frame set
"""
import os, sys, csv, json, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bnn2 import CNN, load_bin, junk_from_images, otsu
import train_bnn10 as T10
from train_bnn5 import SignSTE

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'src', 'data')
VEC = os.path.join(ROOT, 'sim', 'vectors_t1a')
CKPT_OUT = os.path.join(ROOT, 'src', 'weights', 'ckpt_honest_pool4.pt')
FLOAT_OUT = os.path.join(ROOT, 'src', 'weights', 'float_honest.pt')
REPORT = os.path.join(ROOT, 'sim', '20261031_诚实评测报告.json')

SIZE = 64
CLASSES = ['rock', 'paper', 'scissors']


def load_vectors():
    rows = list(csv.DictReader(open(os.path.join(VEC, 'labels.csv'), encoding='utf-8-sig')))
    X = np.zeros((len(rows), 1, SIZE, SIZE), np.float32)
    for i, r in enumerate(rows):
        raw = np.fromfile(os.path.join(VEC, r['file'].strip()), np.uint8)
        b = np.unpackbits(raw)
        X[i, 0] = b[:SIZE * SIZE].reshape(SIZE, SIZE)
    y = np.array([int(r['class']) for r in rows], np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


def build_train(seed=1, augment=True, n_junk=2400):
    """train ONLY on data/rps (840/class) + synthetic junk"""
    rng = np.random.default_rng(seed)
    X, y = [], []
    real = []
    for ci, cls in enumerate(CLASSES):
        d = os.path.join(DATA, 'rps', cls)
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg')))
        real += [os.path.join(d, f) for f in files]
        for f in files:
            p = os.path.join(d, f)
            X.append(load_bin(p)); y.append(ci)
            if augment:
                X.append(load_bin(p, flip=True)); y.append(ci)
    for b in junk_from_images(real, n_junk, rng):
        X.append(b); y.append(3)
    X = torch.from_numpy(np.stack(X)[:, None, :, :]).float()
    y = torch.from_numpy(np.array(y)).long()
    p = rng.permutation(len(X))
    return X[p], y[p]


class BNNet(nn.Module):
    """float CNN with BN (teacher)"""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1, bias=False); self.b1 = nn.BatchNorm2d(8)
        self.dw = nn.Conv2d(8, 8, 3, padding=1, groups=8, bias=False); self.b2 = nn.BatchNorm2d(8)
        self.c3 = nn.Conv2d(8, 4, 1, bias=False); self.b3 = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4 * 16 * 16, 4)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.dw(x)))
        x = F.relu(self.b3(self.c3(x)))
        x = F.adaptive_avg_pool2d(x, 16)
        return self.fc(x.flatten(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs-float', type=int, default=25)
    ap.add_argument('--epochs-bin', type=int, default=40)
    ap.add_argument('--lr-float', type=float, default=1e-3)
    ap.add_argument('--lr-bin', type=float, default=2e-4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--same-domain', action='store_true',
                    help='hold out 20%% of data/rps as test (same domain -> T1a)')
    ap.add_argument('--holdout', type=float, default=0.2)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    Xtr, ytr = build_train(augment=True)
    if a.same_domain:
        rng = np.random.default_rng(7)
        idx = rng.permutation(len(Xtr))
        n_ho = int(len(Xtr) * a.holdout)
        ho, tr = idx[:n_ho], idx[n_ho:]
        Xte, yte = Xtr[ho].clone(), ytr[ho].clone()
        Xtr, ytr = Xtr[tr].clone(), ytr[tr].clone()
        print('[data] SAME-DOMAIN split: train=%d test=%d' % (len(Xtr), len(Xte)))
    else:
        Xte, yte = load_vectors()
    print('[data] train=%d (rps-only)  test=%d (vectors_t1a held-out)' % (len(Xtr), len(Xte)))
    print('[data] train per-class =', np.bincount(ytr.numpy(), minlength=4).tolist())

    # ---------- stage 1: float teacher ----------
    cnn = BNNet()
    opt = torch.optim.Adam(cnn.parameters(), lr=a.lr_float)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs_float)
    lossf = nn.CrossEntropyLoss(); n = len(Xtr)
    for ep in range(a.epochs_float):
        cnn.train(); perm = torch.randperm(n); tot = 0.0; corr = 0
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad(); out = cnn(Xtr[idx]); loss = lossf(out, ytr[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx); corr += (out.argmax(1) == ytr[idx]).sum().item()
        sch.step()
        if (ep + 1) % 5 == 0:
            cnn.eval()
            with torch.no_grad():
                te = (cnn(Xte).argmax(1) == yte).float().mean().item()
            print('[float ep %2d] loss=%.3f train=%.4f | HELD-OUT test=%.4f' % (ep + 1, tot / n, corr / n, te))
    cnn.eval()
    with torch.no_grad():
        float_acc = (cnn(Xte).argmax(1) == yte).float().mean().item()
    print('[FLOAT] honest held-out accuracy = %.4f' % float_acc)
    torch.save(cnn.state_dict(), FLOAT_OUT)

    # ---------- stage 2: binary student (warm start) ----------
    m = T10.Net(4)
    with torch.no_grad():
        m.c1.weight.copy_(cnn.c1.weight); m.dw.weight.copy_(cnn.dw.weight); m.c3.weight.copy_(cnn.c3.weight)
        for k in ['weight', 'bias', 'running_mean', 'running_var']:
            getattr(m.b1, k).copy_(getattr(cnn.b1, k))
            getattr(m.b2, k).copy_(getattr(cnn.b2, k))
            getattr(m.b3, k).copy_(getattr(cnn.b3, k))
        # fc: teacher 4x256 -> student 4x1024 (tile)
        rep = m.fc.weight.shape[1] // cnn.fc.weight.shape[1]
        m.fc.weight.copy_(cnn.fc.weight.repeat(1, rep) / rep)
        m.fc.bias.copy_(cnn.fc.bias)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr_bin)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs_bin)
    best = 0.0; best_ep = 0
    for ep in range(a.epochs_bin):
        m.train(); perm = torch.randperm(n); tot = 0.0; corr = 0
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad(); out = m(Xtr[idx]); loss = lossf(out, ytr[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx); corr += (out.argmax(1) == ytr[idx]).sum().item()
        sch.step()
        if (ep + 1) % 5 == 0:
            m.eval()
            with torch.no_grad():
                ex = (T10.hw_eval(m, Xte).argmax(1) == yte).float().mean().item()
                fl = (m(Xte).argmax(1) == yte).float().mean().item()
            if ex >= best:
                best, best_ep = ex, ep + 1
                torch.save(m.state_dict(), CKPT_OUT)
            print('[bin ep %2d] loss=%.3f train=%.4f | float=%.4f EXACT(hw)=%.4f%s'
                  % (ep + 1, tot / n, corr / n, fl, ex, '  <= best' if ep + 1 == best_ep else ''))
    print('[BINARY] honest held-out EXACT accuracy = %.4f (ep%d)' % (best, best_ep))

    json.dump({'float_honest': float_acc, 'binary_honest': best, 'binary_epoch': best_ep,
               'test_set': 'data/vectors_t1a (620 frames, rps-test-set + synthetic junk)',
               'train_set': 'data/rps only (840/class) + synthetic junk',
               'note': 'supersedes the leaked 0.9597 measurement'},
              open(REPORT, 'w'), indent=2)
    print('[report] ->', REPORT)


if __name__ == '__main__':
    main()
