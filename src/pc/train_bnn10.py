#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_bnn10.py -- three experiments with attribution discipline (ONE variable each).

modes:
  scratch : from-scratch long training   (variable: initialisation = scratch)
  warm    : teacher-weight warm start    (variable: initialisation = teacher)
  distill : KD from float teacher        (variable: KD loss, T=3.0 alpha=0.7 fixed)

common constants (identical across all runs):
  padding: L1 bit-domain pad 0, L2/L3 pm1-domain pad -1
  sign weights + identity STE, BN before sign, lr 2e-4 cosine
  per-channel theta folded from BN statistics
acceptance: exact integer forward must equal float forward point-by-point.
"""
import os, sys, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bnn2 import CNN, build_dataset
from train_bnn5 import SignSTE

KD_T = 3.0
KD_ALPHA = 0.7


def matches_fixed(x, w, pad=1, groups=1):
    m1 = F.conv2d(F.pad(x, (pad,) * 4, value=0.0), w, None, 1, 0, groups=groups)
    m2 = F.conv2d(F.pad(1 - x, (pad,) * 4, value=1.0), 1 - w, None, 1, 0, groups=groups)
    return m1 + m2


def bn_c(bn):
    g = bn.weight.detach().clamp(min=1e-6)
    b = bn.bias.detach()
    mu = bn.running_mean.detach()
    sd = (bn.running_var.detach() + bn.eps).sqrt()
    return mu - b * sd / g


def theta_of(bn, w, K, pm1):
    c = bn_c(bn)
    if pm1:
        return (K + c) / 2.0
    s = w.detach().sign().sum(dim=(1, 2, 3))
    return K / 2.0 - s / 2.0 + c


class Net(nn.Module):
    def __init__(self, pool):
        super().__init__()
        self.pool = pool
        self.feat = 4 * (64 // pool) ** 2
        self.c1 = nn.Conv2d(1, 8, 3, padding=0, bias=False)
        self.b1 = nn.BatchNorm2d(8)
        self.dw = nn.Conv2d(8, 8, 3, padding=0, groups=8, bias=False)
        self.b2 = nn.BatchNorm2d(8)
        self.c3 = nn.Conv2d(8, 4, 1, bias=False)
        self.b3 = nn.BatchNorm2d(4)
        self.fc = nn.Linear(self.feat, 4, bias=True)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), value=0.0)
        x = SignSTE.apply(self.b1(F.conv2d(x, SignSTE.apply(self.c1.weight))))
        x = F.pad(x, (1, 1, 1, 1), value=-1.0)
        x = SignSTE.apply(self.b2(F.conv2d(x, SignSTE.apply(self.dw.weight), groups=8)))
        x = SignSTE.apply(self.b3(F.conv2d(x, SignSTE.apply(self.c3.weight))))
        s = F.avg_pool2d(x, self.pool) * (self.pool ** 2)
        x = SignSTE.apply(s + 1e-6)
        return F.linear(x.flatten(1), SignSTE.apply(self.fc.weight), self.fc.bias)


def hw_eval_q6(m, X, fc_scale=64):
    """Hardware-exact integer forward with Q6 fixed-point FC bias (E6 定案).

    theta is floor-equivalent automatically: popcount is integer, so
    (count > theta_float) == (count > floor(theta_float)).
    Returns integer scores [N,4]: fc_scale*(2*matches - feat) + round(bias*fc_scale).
    """
    w1 = (m.c1.weight > 0).float()
    w2 = (m.dw.weight > 0).float()
    w3 = (m.c3.weight > 0).float()
    wf = (m.fc.weight > 0).float()
    t1 = theta_of(m.b1, m.c1.weight, 9, False).floor()
    t2 = theta_of(m.b2, m.dw.weight, 9, True).floor()
    t3 = theta_of(m.b3, m.c3.weight, 8, True).floor()
    x1 = (matches_fixed(X, w1, 1) > t1.view(1, -1, 1, 1)).float()
    x2 = (matches_fixed(x1, w2, 1, groups=8) > t2.view(1, -1, 1, 1)).float()
    x3 = (matches_fixed(x2, w3, 0) > t3.view(1, -1, 1, 1)).float()
    s = F.avg_pool2d(x3, m.pool) * (m.pool ** 2)
    fh = (s >= m.pool ** 2 / 2.0).float().flatten(1)
    mf = matches_fixed(fh.reshape(-1, m.feat, 1, 1), wf.reshape(4, m.feat, 1, 1), 0).reshape(-1, 4)
    bias_q = (m.fc.bias.detach() * fc_scale).round().long()
    return (2 * mf - m.feat).long() * fc_scale + bias_q.view(1, 4)


def hw_eval(m, X, mode='perchan'):
    w1 = (m.c1.weight > 0).float()
    w2 = (m.dw.weight > 0).float()
    w3 = (m.c3.weight > 0).float()
    wf = (m.fc.weight > 0).float()
    t1 = theta_of(m.b1, m.c1.weight, 9, False)
    t2 = theta_of(m.b2, m.dw.weight, 9, True)
    t3 = theta_of(m.b3, m.c3.weight, 8, True)
    if mode == 'global':
        t1, t2, t3 = [x.median().expand_as(x) for x in (t1, t2, t3)]
    x1 = (matches_fixed(X, w1, 1) > t1.view(1, -1, 1, 1)).float()
    x2 = (matches_fixed(x1, w2, 1, groups=8) > t2.view(1, -1, 1, 1)).float()
    x3 = (matches_fixed(x2, w3, 0) > t3.view(1, -1, 1, 1)).float()
    s = F.avg_pool2d(x3, m.pool) * (m.pool ** 2)
    fh = (s >= m.pool ** 2 / 2.0).float().flatten(1)
    mf = matches_fixed(fh.reshape(-1, m.feat, 1, 1), wf.reshape(4, m.feat, 1, 1), 0)
    return (2.0 * mf.reshape(-1, 4) - m.feat) + m.fc.bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['scratch', 'warm', 'distill'])
    ap.add_argument('--pool', type=int, default=8, choices=[8, 4])
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    DATA = os.path.join(ROOT, r'data')
    W = os.path.join(ROOT, r'board/weights')
    roots = [os.path.join(DATA, s) for s in ['rps', 'rps-test-set'] if os.path.isdir(os.path.join(DATA, s))]
    Xtr, ytr = build_dataset(roots, 0, seed=1, augment=True)
    Xte, yte = build_dataset(roots, 124, seed=99, augment=False)
    tag = '%s_pool%d_s%d' % (a.mode, a.pool, a.seed)
    print('[exp] %s train=%d test=%d feat=%d KD_T=%.1f alpha=%.1f'
          % (tag, len(Xtr), len(Xte), 4 * (64 // a.pool) ** 2, KD_T, KD_ALPHA))

    teacher = CNN()
    teacher.load_state_dict(torch.load(os.path.join(W, 'model.pt'), map_location='cpu'))
    teacher.eval()

    m = Net(a.pool)
    if a.mode in ('warm', 'distill'):
        tsd = teacher.state_dict()
        with torch.no_grad():
            m.c1.weight.copy_(tsd['c1.weight'])
            m.dw.weight.copy_(tsd['dw.weight'])
            m.c3.weight.copy_(tsd['c3.weight'])
            for k in ['weight', 'bias', 'running_mean', 'running_var']:
                getattr(m.b1, k).copy_(tsd['b1.' + k])
                getattr(m.b2, k).copy_(tsd['b2.' + k])
                getattr(m.b3, k).copy_(tsd['b3.' + k])
            if a.pool == 8:
                m.fc.weight.copy_(tsd['fc.weight'])
                m.fc.bias.copy_(tsd['fc.bias'])
        print('[init] warm start from teacher (fc %s)' % ('copied' if a.pool == 8 else 'random'))

    opt = torch.optim.Adam(m.parameters(), lr=2e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr)
    best = 0.0
    for ep in range(a.epochs):
        m.train()
        perm = torch.randperm(n)
        tot = 0.0
        corr = 0
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            out = m(Xtr[idx])
            loss = lossf(out, ytr[idx])
            if a.mode == 'distill':
                with torch.no_grad():
                    tl = teacher(Xtr[idx])
                loss = (1 - KD_ALPHA) * loss + KD_ALPHA * F.kl_div(
                    F.log_softmax(out / KD_T, dim=1),
                    F.softmax(tl / KD_T, dim=1),
                    reduction='batchmean') * (KD_T ** 2)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            corr += (out.argmax(1) == ytr[idx]).sum().item()
        sch.step()
        if (ep + 1) % 5 == 0:
            m.eval()
            with torch.no_grad():
                fl = (m(Xte).argmax(1) == yte).float().mean().item()
                ex = (hw_eval(m, Xte).argmax(1) == yte).float().mean().item()
            better = ex >= best
            best = max(best, ex)
            print('[%s ep %02d] loss=%.3f train=%.4f | float=%.4f exact=%.4f%s'
                  % (tag, ep + 1, tot / n, corr / n, fl, ex, '  <= best' if better else ''))
            sys.stdout.flush()
            torch.save(m.state_dict(), os.path.join(W, 'ckpt_%s_ep%02d.pt' % (tag, ep + 1)))
            if ep == 19 and ex < 0.60:
                print('[abort] rule: exact<60%% at ep20 -> check lr')
                break
    print('[DONE] %s best_exact=%.4f' % (tag, best))


if __name__ == '__main__':
    main()
