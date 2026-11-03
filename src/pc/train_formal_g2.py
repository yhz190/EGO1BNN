#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_formal_g2.py -- patch2 formal measurement protocol (G2).

G2 rules:
  * epoch selection uses ONLY a validation split carved out of the TRAIN files (10%);
  * the test set is evaluated ONE time, for the finally selected checkpoint;
  * "best over test epochs" is forbidden for delivery numbers.

3 seeds x 80 epochs, 8-channel model, teacher init (conv+BN, c3/fc random).
"""
import os, sys, json, time
import numpy as np
import torch, torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'scripts'))
torch.set_num_threads(4)
import train_bnn10 as T10
import train_l3x8 as L8
import train_honest

W = os.path.join(ROOT, 'src', 'weights')
OUT = os.path.join(ROOT, 'sim', '20261101_formal_g2.json')

# ---------------- data: file-level split FIRST, then augment ----------------
import os as _os
DATA = os.path.join(ROOT, 'src', 'data')
CLASSES = ['rock', 'paper', 'scissors']


def build_split_g2(seed=7, holdout=0.2, val_frac=0.1):
    """train/val/test, all by ORIGINAL FILE; augment train only; junk split by source."""
    import train_bnn2 as T2
    rng = np.random.default_rng(seed)
    tr_files, te_files = {}, {}
    for cls in CLASSES:
        d = os.path.join(DATA, 'rps', cls)
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg')))
        idx = rng.permutation(len(files))
        n_ho = int(len(files) * holdout)
        te_files[cls] = [os.path.join(d, files[i]) for i in idx[:n_ho]]
        tr_files[cls] = [os.path.join(d, files[i]) for i in idx[n_ho:]]
    # carve val out of train files (before augmentation)
    va_files, trf2 = {}, {}
    for cls in CLASSES:
        fs = tr_files[cls]
        idx = rng.permutation(len(fs))
        n_va = max(1, int(len(fs) * val_frac))
        va_files[cls] = [fs[i] for i in idx[:n_va]]
        trf2[cls] = [fs[i] for i in idx[n_va:]]

    def frames(files, augment):
        X, y = [], []
        for ci, cls in enumerate(CLASSES):
            for p in files[cls]:
                b = T2.load_bin(p)
                X.append(b); y.append(ci)
                if augment:
                    X.append(b[:, ::-1].copy()); y.append(ci)
        return X, y

    Xtr, ytr = frames(trf2, True)
    Xva, yva = frames(va_files, False)
    Xte, yte = frames(te_files, False)
    all_tr = [p for cls in CLASSES for p in trf2[cls]]
    all_va = [p for cls in CLASSES for p in va_files[cls]]
    all_te = [p for cls in CLASSES for p in te_files[cls]]
    for b in T2.junk_from_images(all_tr, 2400, rng):
        Xtr.append(b); ytr.append(3)
    for b in T2.junk_from_images(all_va, 300, rng):
        Xva.append(b); yva.append(3)
    for b in T2.junk_from_images(all_te, 600, rng):
        Xte.append(b); yte.append(3)
    st = lambda X, y: (torch.from_numpy(np.stack(X)[:, None]).float(), torch.from_numpy(np.array(y)).long())
    return st(Xtr, ytr), st(Xva, yva), st(Xte, yte)


(Xtr, ytr), (Xva, yva), (Xte, yte) = build_split_g2()
print('[data] train=%d val=%d test=%d (file-level, augment train only)' % (len(Xtr), len(Xva), len(Xte)), flush=True)

teacher = train_honest.BNNet()
teacher.load_state_dict(torch.load(os.path.join(W, 'float_honest.pt'), map_location='cpu'))
teacher.eval()


def warm(m):
    with torch.no_grad():
        m.c1.weight.copy_(teacher.c1.weight)
        m.dw.weight.copy_(teacher.dw.weight)
        for k in ['weight', 'bias', 'running_mean', 'running_var']:
            getattr(m.b1, k).copy_(getattr(teacher.b1, k))
            getattr(m.b2, k).copy_(getattr(teacher.b2, k))


def run(seed, epochs=80, lr=2e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    m = L8.Net8(); warm(m)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    n = len(Xtr)
    best_val = -1.0; best_state = None; best_ep = 0
    t0 = time.time()
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            loss = ce(m(Xtr[idx]), ytr[idx])
            loss.backward(); opt.step()
        sch.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            m.eval()
            try:
                with torch.no_grad():
                    val = L8.evaluate(m, Xva, yva, tag='val', ep=ep + 1, strict=True)['exact']
            except AssertionError as e:
                # G4: float!=exact -> this epoch is not eligible for selection
                print('[s%d ep%02d] VAL INVALID (float!=exact), skip selection: %s'
                      % (seed, ep + 1, str(e)[:80]), flush=True)
                continue
            if val > best_val:
                best_val = val; best_ep = ep + 1
                best_state = {k: v.clone() for k, v in m.state_dict().items()}
            print('[s%d ep%02d] val=%.4f%s' % (seed, ep + 1, val, '  <= best' if ep + 1 == best_ep else ''), flush=True)
    # ONE-SHOT test evaluation of the selected checkpoint
    m.load_state_dict(best_state)
    m.eval()
    try:
        test = L8.evaluate(m, Xte, yte, tag='test', ep=best_ep, strict=True)['exact']
        valid = True
    except AssertionError as e:
        test = float('nan')
        valid = False
        print('[s%d] TEST ckpt INVALID per G4 (float!=exact): %s' % (seed, str(e)[:90]), flush=True)
    torch.save(best_state, os.path.join(W, 'FORMAL_l3x8_s%d.pt' % seed))
    print('[s%d] best_val=%.4f @ep%d  ONE-SHOT test=%.4f  (%.0fs)'
          % (seed, best_val, best_ep, test, time.time() - t0), flush=True)
    return {'seed': seed, 'best_val': best_val, 'best_ep': best_ep, 'test_exact': test, 'g4_valid': valid}


import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--seeds', default='1,2,3')
a = ap.parse_args()
res = []
for s in [int(x) for x in a.seeds.split(',')]:
    res.append(run(s))
    json.dump(res, open(OUT, 'w'), indent=2)

tests = [r['test_exact'] for r in res if r.get('g4_valid')]
print('\n===== G2 FORMAL RESULT (val-selected, test evaluated once) =====')
for r in res:
    print('  seed %d: val=%.4f @ep%-3d  test=%.4f' % (r['seed'], r['best_val'], r['best_ep'], r['test_exact']))
print('  mean test = %.4f ± %.4f' % (float(np.mean(tests)), float(np.std(tests))))
json.dump({'runs': res, 'mean_test': float(np.mean(tests)), 'std_test': float(np.std(tests))},
          open(OUT, 'w'), indent=2)
print('[out]', OUT)
