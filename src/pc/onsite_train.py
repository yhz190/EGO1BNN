#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""onsite_train.py -- 10-minute on-site calibration for an unreproducible environment.

Why: the capture environment changes between sessions (measured: cross-session binary frames
differ 26-49%, far beyond the model's ~10% noise budget). B6's environment cannot be restored,
so the deployable model must be calibrated in whatever environment is current.

Input: a short capture session (recommended: 4 classes x 2 distances x 30 frames + junk 80,
~10 min with capture_live.py). Frames are selected by capture timestamp.

Pipeline
  1. load frames (timestamp range) with the SAME preprocess as deployment
  2. per-class 70/10/20 split (val = epoch selection, test = one-shot report)
  3. DA training from the RPS baseline with RAW-domain augmentation:
       scale x[0.85,1.20] (distance), shift +-2%, brightness x[0.90,1.10], flip
     (raw-domain: perturb the gray image THEN binarise -- binary-domain flip/shift is useless)
  4. G4 gate: float==exact on test / RPS / vectors, else reject
  5. export weights blob v5 (gen_weights --ckpt) + board test bins + a deploy command
"""
import os, sys, csv, time, random, json, subprocess
import numpy as np
from PIL import Image
import torch, torch.nn as nn
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'scripts'))
torch.set_num_threads(4)
import train_bnn10 as T10
import train_l3x8 as L8
import train_fix as F
import preprocess as P1
from class_map import CLASS_MAP, CLASS_ID

LIVE = os.path.join(ROOT, 'src', 'data', 'live')
W = os.path.join(ROOT, 'src', 'weights')
BASE = os.path.join(W, 'AB_A_l3x8_80ep_s2.pt')
SPLIT_SEED = 7
OVERSAMPLE = 4
LR = 3e-5
EPOCHS = 18


def parse_ts(hhmm):
    return time.mktime(time.strptime('2026-11-01 ' + hhmm, '%Y-%m-%d %H:%M'))


def load_session(t_from, t_to, max_per_class=400):
    man = {r['file']: int(r['ts']) for r in csv.DictReader(open(os.path.join(LIVE, '_manifest.csv'), encoding='utf-8'))}
    items = []
    for cls in CLASS_MAP:
        d = os.path.join(LIVE, cls)
        if not os.path.isdir(d):
            continue
        paths = []
        for f in sorted(x for x in os.listdir(d) if x.endswith('.png')):
            n = f[:-4]
            ts = man.get(n, 0)
            if t_from <= ts < t_to:
                paths.append(os.path.join(d, f))
        paths = paths[:max_per_class]
        for p in paths:
            items.append({'path': p, 'y': CLASS_ID[cls], 'cls': cls})
    return items


AUG = {'scale': (0.85, 1.20), 'shift': 0.02, 'bright': (0.90, 1.10)}


def rand_transform(gray, rng):
    """raw-domain geometric+photometric augmentation (simulates distance/position/light drift)"""
    g = gray
    k = rng.uniform(*AUG['scale'])
    if abs(k - 1.0) > 0.01:
        h, w = g.shape
        g = cv2.resize(g, (int(w * k), int(h * k)), interpolation=cv2.INTER_LINEAR)
    b = rng.uniform(*AUG['bright'])
    g = np.clip(g.astype(np.float32) * b, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        g = g[:, ::-1]
    if rng.random() < 0.7:
        h, w = g.shape
        dx = int(w * rng.uniform(-AUG['shift'], AUG['shift']))
        dy = int(h * rng.uniform(-AUG['shift'], AUG['shift']))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        g = cv2.warpAffine(g, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return g


def frames_of(item, rng=None):
    g = np.asarray(Image.open(item['path']).convert('L'), np.uint8)
    if rng is not None:
        g = rand_transform(g, rng)
    return P1.to_frame(Image.fromarray(g))[1].astype(np.float32)


def acc_q6(m, X, y):
    with torch.no_grad():
        p = T10.hw_eval_q6(m, X).argmax(1)
    rec = {}
    for c in range(4):
        mask = (y == c)
        if int(mask.sum()) > 0:
            rec[CLASS_MAP[c]] = float((p[mask] == c).float().mean())
    return float((p == y).float().mean()), rec


def g4(m, X, y):
    with torch.no_grad():
        q6 = T10.hw_eval_q6(m, X).argmax(1)
        fl = m(X).argmax(1)
    return int((q6 == fl).sum()), len(y)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='t_from', default='')
    ap.add_argument('--to', dest='t_to', default='')
    ap.add_argument('--ckpt-out', default=os.path.join(W, 'ONSITE_DA.pt'))
    ap.add_argument('--blob-out', default=os.path.join(W, 'hw_onsite'))
    ap.add_argument('--skip-export', action='store_true')
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--scale-min', type=float, default=None)
    ap.add_argument('--scale-max', type=float, default=None)
    ap.add_argument('--shift-frac', type=float, default=None)
    ap.add_argument('--bright-min', type=float, default=None)
    ap.add_argument('--bright-max', type=float, default=None)
    a = ap.parse_args()
    global LR, EPOCHS
    if a.lr: LR = a.lr
    if a.epochs: EPOCHS = a.epochs
    if a.scale_min: AUG['scale'] = (a.scale_min, AUG['scale'][1])
    if a.scale_max: AUG['scale'] = (AUG['scale'][0], a.scale_max)
    if a.shift_frac: AUG['shift'] = a.shift_frac
    if a.bright_min: AUG['bright'] = (a.bright_min, AUG['bright'][1])
    if a.bright_max: AUG['bright'] = (AUG['bright'][0], a.bright_max)
    print('  params: lr=%g epochs=%d aug_scale=%s shift=%.3f bright=%s'
          % (LR, EPOCHS, AUG['scale'], AUG['shift'], AUG['bright']))

    if a.t_from:
        t_from = parse_ts(a.t_from)
        t_to = parse_ts(a.t_to) if a.t_to else time.mktime(time.strptime('2026-11-02 00:00', '%Y-%m-%d %H:%M'))
    else:
        # default: everything after the last B6 window
        t_from = time.mktime(time.strptime('2026-11-01 16:30', '%Y-%m-%d %H:%M'))
        t_to = time.mktime(time.strptime('2026-11-02 00:00', '%Y-%m-%d %H:%M'))
    items = load_session(t_from, t_to)
    from collections import Counter
    print('  session frames=%d per-class=%s' % (len(items), dict(Counter(it['cls'] for it in items))))
    if len(items) < 60:
        print('[FAIL] session too small (%d). capture at least 4 classes x 20 + junk 40 first.' % len(items))
        return 1

    # split per class
    rng = random.Random(SPLIT_SEED)
    by = {}
    for it in items:
        by.setdefault(it['y'], []).append(it)
    tr, va, te = [], [], []
    for c, lst in by.items():
        idx = list(range(len(lst)))
        rng.shuffle(idx)
        n = len(lst)
        n_te = max(1, int(round(n * 0.20)))
        n_va = max(1, int(round(n * 0.10)))
        te += [lst[i] for i in idx[:n_te]]
        va += [lst[i] for i in idx[n_te:n_te + n_va]]
        tr += [lst[i] for i in idx[n_te + n_va:]]
    print('  split: train=%d val=%d test=%d' % (len(tr), len(va), len(te)))

    Xva = torch.from_numpy(np.stack([frames_of(it) for it in va]))[:, None]
    yva = torch.tensor([it['y'] for it in va])
    Xte = torch.from_numpy(np.stack([frames_of(it) for it in te]))[:, None]
    yte = torch.tensor([it['y'] for it in te])
    Xrps, yrps, Xsd, ysd = F.build_split()

    # vectors (diagnostic only)
    VEC = os.path.join(ROOT, 'sim', 'vectors_t1a')
    rows = list(csv.DictReader(open(os.path.join(VEC, 'labels.csv'), encoding='utf-8-sig')))
    Xv = np.zeros((len(rows), 1, 64, 64), np.float32)
    for i, r in enumerate(rows):
        Xv[i, 0] = np.unpackbits(np.fromfile(os.path.join(VEC, r['file'].strip()), np.uint8))[:4096].reshape(64, 64)
    Xv = torch.from_numpy(Xv); yv = torch.tensor([int(r['class']) for r in rows])

    torch.manual_seed(1); np.random.seed(1); random.seed(1)
    m = L8.Net8()
    if os.path.exists(BASE):
        m.load_state_dict(torch.load(BASE, map_location='cpu'))
    else:
        print('[WARN] base ckpt not shipped (%s) -> training from random init' % BASE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()

    # live frames are regenerated with augmentation every epoch (raw-domain aug)
    Xrps_rep, yrps_rep = Xrps, yrps
    n = None
    best = None
    for ep in range(1, EPOCHS + 1):
        aug_rng = random.Random(1000 + ep)
        Xl = torch.from_numpy(np.stack([frames_of(it, aug_rng) for it in tr]))[:, None]
        Xl = Xl.repeat(OVERSAMPLE, 1, 1, 1)
        yl = torch.tensor([it['y'] for it in tr]).repeat(OVERSAMPLE)
        Xmix = torch.cat([Xl, Xrps_rep], 0)
        ymix = torch.cat([yl, yrps_rep], 0)
        n = len(Xmix)
        m.train()
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            ce(m(Xmix[idx]), ymix[idx]).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        if ep % 3 == 0:
            m.eval()
            val, _ = acc_q6(m, Xva, yva)
            if best is None or val > best['val']:
                best = {'ep': ep, 'val': val, 'state': {k: v.clone() for k, v in m.state_dict().items()}}
            print('  [ep%02d] val=%.3f' % (ep, val), flush=True)

    m.load_state_dict(best['state']); m.eval()
    test, rec = acc_q6(m, Xte, yte)
    print('\n  ==== ONSITE RESULT ====')
    print('  best val=%.3f @ep%d' % (best['val'], best['ep']))
    print('  **T1b onsite held-out = %.4f**  recall=%s' % (test, {k: round(v, 2) for k, v in rec.items()}))
    print('  RPS same-domain = %.4f | RPS-vectors = %.4f'
          % (acc_q6(m, Xsd, ysd)[0], acc_q6(m, Xv, yv)[0]))

    # G4 gate
    gates = {}
    for tag, X, y in (('test', Xte, yte), ('rps', Xsd, ysd), ('vectors', Xv, yv)):
        ok, tot = g4(m, X, y)
        gates[tag] = '%d/%d' % (ok, tot)
    print('  G4 float==exact:', gates)
    ok_g4 = all(int(gates[t].split('/')[0]) == int(gates[t].split('/')[1]) for t in gates)
    torch.save(best['state'], a.ckpt_out)
    print('  ckpt ->', a.ckpt_out)

    # export blob + board test bins + deploy command
    if not a.skip_export and ok_g4:
        os.makedirs(a.blob_out, exist_ok=True)
        r = subprocess.run([sys.executable, os.path.join(ROOT, 'src', 'scripts', 'gen_weights.py'),
                            '--ckpt', a.ckpt_out, '--outdir', a.blob_out],
                           capture_output=True, text=True, encoding='utf-8', errors='ignore')
        print('  [gen_weights]', (r.stdout or '').strip().splitlines()[-3:] if r.stdout else r.stderr[-300:])
        # board test bins
        bins = os.path.join(ROOT, 'sim', 'onsite_test_bins')
        os.makedirs(bins, exist_ok=True)
        for f in os.listdir(bins):
            os.remove(os.path.join(bins, f))
        out_rows = []
        for k, it in enumerate(te):
            _, bits, _ = P1.to_frame(Image.open(it['path']))
            name = '%s_%04d.bin' % (it['cls'], k)
            open(os.path.join(bins, name), 'wb').write(np.packbits(bits.reshape(-1)).tobytes())
            out_rows.append((name, it['cls']))
        with open(os.path.join(bins, 'labels.csv'), 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh); w.writerow(['file', 'class']); w.writerows(out_rows)
        print('  board test bins ->', bins)
        print('\n  ==== DEPLOY (Lead runs) ====')
        print('  1) powershell -File src/pc/deploy_onsite.ps1 -BlobDir %s -Ckpt %s' % (a.blob_out, a.ckpt_out))
    elif not ok_g4:
        print('  [WARN] G4 failed -> model NOT exported (per G4 rule)')

    json.dump({'val': best['val'], 'ep': best['ep'], 'test': test, 'recall': rec,
               'g4': gates, 'n_train': len(tr), 'n_test': len(te)},
              open(os.path.join(ROOT, 'sim', '20261101_onsite_calib.json'), 'w'), indent=2, default=float)
    return 0 if ok_g4 else 1


if __name__ == '__main__':
    sys.exit(main())
