#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_bnn5.py -- EXPLORATION: the 4-item recipe bundle (does NOT change the plan).

Bundle: identity STE (no |x|<=1 gate) + BatchNorm BEFORE sign + w.clamp(-1,1) + lr 2e-4 cosine.
Evaluation uses the EXACT integer hardware forward (popcount, 0/1, folded per-channel theta).

Gate: binary-exact test accuracy >= 95% before any RTL change.
"""
import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bnn2 import CNN, build_dataset
from hwsim_gate import matches, fold_theta, hw_forward

class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x.sign()
    @staticmethod
    def backward(ctx, g): return g          # identity STE (no magnitude gate)

class BNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(8)
        self.dw = nn.Conv2d(8, 8, 3, padding=1, groups=8, bias=False)
        self.b2 = nn.BatchNorm2d(8)
        self.c3 = nn.Conv2d(8, 4, 1, bias=False)
        self.b3 = nn.BatchNorm2d(4)
        self.b4 = nn.BatchNorm1d(4 * 8 * 8)
        self.fc = nn.Linear(4 * 8 * 8, 4, bias=True)
    def forward(self, x):
        x = SignSTE.apply(self.b1(self.c1(x)))
        x = SignSTE.apply(self.b2(self.dw(x)))
        x = SignSTE.apply(self.b3(self.c3(x)))
        x = F.avg_pool2d(x, 8).flatten(1)
        x = SignSTE.apply(self.b4(x))
        return self.fc(x)

def clamp_w(m):
    with torch.no_grad():
        for c in [m.c1, m.dw, m.c3]:
            c.weight.clamp_(-1, 1)
        m.fc.weight.clamp_(-1, 1)

def eval_exact(m, X, y, perchan=True):
    """exact integer hardware forward using folded theta"""
    class Wrap:  # hw_forward expects modules named c1/dw/c3/fc and BN stats
        pass
    w = Wrap()
    w.c1, w.dw, w.c3, w.fc = m.c1, m.dw, m.c3, m.fc
    thetas = {
        't1': fold_theta(m.b1, m.c1.weight, 9),
        't2': fold_theta(m.b2, m.dw.weight, 9),
        't3': fold_theta(m.b3, m.c3.weight, 8),
    }
    mode = 'perchan' if perchan else 'global'
    with torch.no_grad():
        out = hw_forward(w, X, thetas, mode)
    return (out.argmax(1) == y).float().mean().item()

def main():
    DATA = os.path.join(ROOT, r'data')
    roots = [os.path.join(DATA, s) for s in ['rps', 'rps-test-set'] if os.path.isdir(os.path.join(DATA, s))]
    Xtr, ytr = build_dataset(roots, 0, seed=1, augment=True)
    Xte, yte = build_dataset(roots, 124, seed=99, augment=False)
    print('[data] train=%d test=%d' % (len(Xtr), len(Xte)))

    cnn = CNN(); cnn.load_state_dict(torch.load(os.path.join(ROOT, r'board/weights/model.pt'), map_location='cpu'))
    m = BNNet()
    # warm start: copy real weights + BN stats
    with torch.no_grad():
        m.c1.weight.copy_(cnn.c1.weight); m.dw.weight.copy_(cnn.dw.weight); m.c3.weight.copy_(cnn.c3.weight)
        m.b1.load_state_dict(cnn.b1.state_dict()); m.b2.load_state_dict(cnn.b2.state_dict()); m.b3.load_state_dict(cnn.b3.state_dict())
        m.fc.weight.copy_(cnn.fc.weight); m.fc.bias.copy_(cnn.fc.bias)
        m.b4.weight.fill_(1.0); m.b4.bias.fill_(0.0)

    opt = torch.optim.Adam(m.parameters(), lr=2e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr); hist = []
    print('[init] exact binary acc = %.4f / %.4f' % (eval_exact(m, Xte, yte), eval_exact(m, Xte, yte, False)))
    for ep in range(40):
        m.train(); perm = torch.randperm(n); tot = 0.0; corr = 0
        for i in range(0, n, 128):
            idx = perm[i:i+128]
            opt.zero_grad()
            out = m(Xtr[idx]); loss = lossf(out, ytr[idx])
            loss.backward(); opt.step(); clamp_w(m)
            tot += loss.item() * len(idx); corr += (out.argmax(1) == ytr[idx]).sum().item()
        sch.step()
        tr = tot / n; ac = corr / n; hist.append(tr)
        if (ep + 1) % 2 == 0:
            print('[ep %2d] loss=%.3f train=%.4f | exact perchan=%.4f global=%.4f'
                  % (ep + 1, tr, ac, eval_exact(m, Xte, yte), eval_exact(m, Xte, yte, False)))
        # fast-fail rule: if loss oscillates badly in first 8 epochs -> stop
        if ep >= 4 and max(hist[1:]) > 3 * min(hist[1:]) + 1.0:
            print('[abort] loss oscillating -> check lr (fast-fail rule)'); break
    print('[FINAL] exact perchan=%.4f global=%.4f' % (eval_exact(m, Xte, yte), eval_exact(m, Xte, yte, False)))
    torch.save(m.state_dict(), os.path.join(ROOT, r'board/weights/bnn5.pt'))

if __name__ == '__main__':
    main()