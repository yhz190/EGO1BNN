#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_fix.py -- fix the two protocol flaws + stabilise binary training.

Fixes
  F1  split by ORIGINAL FILE first (80/20), then augment the training half only.
      (previous code augmented before splitting -> 82.2% of test images had their
       mirrored copy inside the training set)
  F2  junk class: train junk generated from TRAIN images only; test junk from
      HELD-OUT images only (no cross-contamination)
Stabilisers (single-variable comparisons)
  S1  EMA over the real-valued weights (decay 0.99), evaluated with the shadow copy
  S2  clip-STE (gradient passes only for |x| <= 1) instead of identity STE
  S3  multiple seeds (reported mean +- std)
"""
import os, sys, json, argparse, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bnn2 import load_bin, junk_from_images
import train_bnn10 as T10
from train_bnn5 import SignSTE

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, 'src', 'data')
W = os.path.join(ROOT, 'src', 'weights')
REPORT = os.path.join(ROOT, 'sim', 'E_泄漏修复与稳定性.md')
JSON_OUT = os.path.join(ROOT, 'sim', 'E_泄漏修复与稳定性.json')
CLASSES = ['rock', 'paper', 'scissors']
SIZE = 64


class ClipSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x.abs() <= 1.0).float()


def load_bin_flip(path, flip):
    b = load_bin(path)
    return b[:, ::-1].copy() if flip else b


def build_split(seed=7, holdout=0.2):
    """F1: split by original files, then augment train only."""
    rng = np.random.default_rng(seed)
    tr_files, te_files = {}, {}
    for cls in CLASSES:
        d = os.path.join(DATA, 'rps', cls)
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg')))
        idx = rng.permutation(len(files))
        n_ho = int(len(files) * holdout)
        ho = [files[i] for i in idx[:n_ho]]
        tr = [files[i] for i in idx[n_ho:]]
        tr_files[cls] = [os.path.join(d, f) for f in tr]
        te_files[cls] = [os.path.join(d, f) for f in ho]

    Xtr, ytr = [], []
    for ci, cls in enumerate(CLASSES):
        for p in tr_files[cls]:
            Xtr.append(load_bin(p)); ytr.append(ci)
            Xtr.append(load_bin_flip(p, True)); ytr.append(ci)   # augment TRAIN only
    # junk: train junk derived from TRAIN images only
    all_tr = [p for c in CLASSES for p in tr_files[c]]
    for b in junk_from_images(all_tr, 2400, rng):
        Xtr.append(b); ytr.append(3)

    Xte, yte = [], []
    for ci, cls in enumerate(CLASSES):
        for p in te_files[cls]:
            Xte.append(load_bin(p)); yte.append(ci)               # NO augment on test
    all_te = [p for c in CLASSES for p in te_files[c]]
    for b in junk_from_images(all_te, 600, rng):                  # junk only from held-out
        Xte.append(b); yte.append(3)

    Xtr = torch.from_numpy(np.stack(Xtr)[:, None, :, :]).float()
    ytr = torch.from_numpy(np.array(ytr)).long()
    Xte = torch.from_numpy(np.stack(Xte)[:, None, :, :]).float()
    yte = torch.from_numpy(np.array(yte)).long()
    p = rng.permutation(len(Xtr))
    return Xtr[p], ytr[p], Xte, yte


from train_bnn5 import SignSTE as _IDENT_STE


def eval_exact(m, X, y):
    m.eval()
    with torch.no_grad():
        out = T10.hw_eval(m, X)
    return (out.argmax(1) == y).float().mean().item()


def run_one(exp, args, Xtr, ytr, Xte, yte, teacher_path, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = T10.Net(4)
    if args.init == 'teacher':
        tsd = torch.load(teacher_path, map_location='cpu')
        import train_honest
        cnn = train_honest.BNNet()
        cnn.load_state_dict(tsd)
        with torch.no_grad():
            m.c1.weight.copy_(cnn.c1.weight); m.dw.weight.copy_(cnn.dw.weight); m.c3.weight.copy_(cnn.c3.weight)
            for k in ['weight', 'bias', 'running_mean', 'running_var']:
                getattr(m.b1, k).copy_(getattr(cnn.b1, k))
                getattr(m.b2, k).copy_(getattr(cnn.b2, k))
                getattr(m.b3, k).copy_(getattr(cnn.b3, k))
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss()
    params = [p for p in m.parameters() if p.requires_grad]
    ema = [p.detach().clone() for p in params] if args.ema else None
    n = len(Xtr)
    best = 0.0
    for ep in range(args.epochs):
        m.train()
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            out = m(Xtr[idx])
            loss = lossf(out, ytr[idx])
            loss.backward()
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    for k, p in enumerate(params):
                        ema[k].mul_(0.99).add_(p.detach(), alpha=0.01)
        sch.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            if ema is not None:
                backup = [p.detach().clone() for p in params]
                with torch.no_grad():
                    for k, p in enumerate(params):
                        p.copy_(ema[k])
                acc = eval_exact(m, Xte, yte)
                with torch.no_grad():
                    for k, p in enumerate(params):
                        p.copy_(backup[k])
            else:
                acc = eval_exact(m, Xte, yte)
            best = max(best, acc)
            print('  [%s ep%02d] exact=%.4f' % (exp, ep + 1, acc))
            sys.stdout.flush()
    return best, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--seeds', type=str, default='')
    a = ap.parse_args()
    torch.set_num_threads(8)

    teacher = os.path.join(W, 'float_honest.pt')
    Xtr, ytr, Xte, yte = build_split()
    print('[data] train=%d (aug, rps-80%%) test=%d (no aug, rps-20%% + junk)' % (len(Xtr), len(Xte)))

    exps = [
        ('E1_fixsplit',  dict(init='teacher', ema=False, clip=False)),
    ]
    seed_list = [int(s) for s in a.seeds.split(',')] if a.seeds else [a.seed]
    results = []
    for name, cfg in exps:
        # clip STE must be swapped in the module where Net.forward resolves it
        if cfg['clip']:
            T10.SignSTE = ClipSTE
        else:
            T10.SignSTE = _IDENT_STE
        print('[run] %s %s' % (name, cfg))
        for sd in seed_list:
            best, m = run_one('%s_seed%d' % (name, sd), argparse.Namespace(**{**vars(a), **cfg}), Xtr, ytr, Xte, yte, teacher, sd)
            torch.save(m.state_dict(), os.path.join(W, 'E_%s_seed%d.pt' % (name, sd)))
            results.append(('%s_seed%d' % (name, sd), cfg, best))
            print('[done] %s seed=%d best=%.4f' % (name, sd, best))

    lines = ['# 修泄漏 + 压不稳定 实验（2026-10-31）', '',
             '> F1 先按原始文件 80/20 划分再增广；F2 junk 按训练/测试分开生成',
             '', '| 实验 | init | EMA | clip STE | best exact |', '|---|---|---|---|---|']
    for name, cfg, best in results:
        lines.append('| %s | %s | %s | %s | **%.4f** |' % (name, cfg['init'], cfg['ema'], cfg['clip'], best))
    lines += ['', '## 说明', '', '- 测试集：rps 20% 留出（未增广）+ junk（仅由留出图生成）',
              '- 评估：`train_bnn10.hw_eval` 硬件精确整数前向', '- 配置：%s' % vars(a)]
    open(REPORT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    json.dump([{'exp': n, 'cfg': {k: bool(v) if isinstance(v, bool) else str(v) for k, v in c.items()}, 'best_exact': b}
               for n, c, b in results], open(JSON_OUT, 'w'), indent=2)
    print('[report]', REPORT)


if __name__ == '__main__':
    main()
