#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""onsite_train_aug.py -- 现场标定训练（带"手部尺度/位置增广"，2026-11-02 新增，用户已批准）

为什么需要它（2026-11-01 23:00 诊断）：
  * 板端与 PC 模型完全一致（40/40），失败只来自"实时画面的手"与标定不同；
  * 串口实测：手线性 ×1.2 → 82.5%、×1.4 → 50%、×1.6 → 35%；位置偏移 10% → 72.5%；
  * 录屏反推真实画面：手面积波动 ±57%（标定会话仅 ±12%）；
  * 离线验证：加入"手部缩放合成样本"后，×1.2/×1.4/×1.6 的准确率从 34%/8%/9% 提升到 94%/95%/96%。

做法：
  1. 用会话帧构造背景底板（逐像素中位数）；
  2. 对每个训练帧，额外合成若干"手放大/缩小/微移"的样本（手部区域缩放后贴回底板）；
  3. 与原帧 + RPS 数据一起训练；
  4. 验证：原帧留出集（选 epoch）+ 合成尺度/位置测试集（报告鲁棒性）；
  5. G4 门禁（float==exact）→ 导出 v5 blob + 板端测试帧。
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
# (k, dx, dy) at 1280x720 resolution: 手部缩放 + 大范围平移增广
AUG_SPECS = [(1.15, 0, 0), (1.35, 0, 0), (0.85, 0, 0),
             (1.0, 96, 0), (1.0, -96, 0), (1.0, 0, 72), (1.0, 0, -72),
             (1.0, 160, 0), (1.0, -160, 0)]
SPLIT_SEED = 7
LR = 6e-5
EPOCHS = 22
OVERSAMPLE = 3


DATE = '2026-11-01'


def ts(hhmm):
    return time.mktime(time.strptime(DATE + ' ' + hhmm, '%Y-%m-%d %H:%M'))


def otsu(u8):
    hist = np.bincount(u8.reshape(-1), minlength=256).astype(np.float64)
    tot = u8.size
    sa = np.dot(np.arange(256), hist)
    wB = np.cumsum(hist); sB = np.cumsum(np.arange(256) * hist); wF = tot - wB
    ok = (wB > 0) & (wF > 0)
    if not ok.any():
        return 128
    mB = np.zeros(256); mF = np.zeros(256)
    mB[ok] = sB[ok] / wB[ok]; mF[ok] = (sa - sB[ok]) / wF[ok]
    var = np.zeros(256); var[ok] = wB[ok] * wF[ok] * (mB[ok] - mF[ok]) ** 2
    return int(np.argmax(var))


def load_session(lo, hi):
    man = {r['file']: int(r['ts']) for r in csv.DictReader(open(os.path.join(LIVE, '_manifest.csv'), encoding='utf-8'))}
    items = []
    for cls in CLASS_MAP:
        d = os.path.join(LIVE, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith('.png') and lo <= man.get(f[:-4], 0) < hi:
                items.append({'cls': cls, 'y': CLASS_ID[cls], 'path': os.path.join(d, f)})
    return items


def gray(path):
    return np.asarray(Image.open(path).convert('L'), np.uint8)


def make_plate(items):
    gs = [gray(it['path']) for it in items[::3]]
    h = min(g.shape[0] for g in gs); w = min(g.shape[1] for g in gs)
    gs = [g[:h, :w] for g in gs]
    return np.median(np.stack(gs), axis=0).astype(np.uint8)


def bits_of_gray(g):
    _, b, _ = P1.to_frame(Image.fromarray(g))
    return b


def synth_hand(g, plate, k=1.0, dx=0, dy=0):
    """hand region scaled/shifted on a fixed background plate (模拟手离镜头远近/位置移动)"""
    h, w = g.shape
    t = otsu(g)
    mask = (g < t).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    if len(ys) < 30:
        return g
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    cy, cx = ys.mean(), xs.mean()
    patch = g[y0:y1, x0:x1]
    pm = mask[y0:y1, x0:x1]
    hp, wp = patch.shape
    nh, nw = max(4, int(round(hp * k))), max(4, int(round(wp * k)))
    patch2 = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pm2 = cv2.resize(pm.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
    out = plate.copy()
    tcy, tcx = int(round(cy + dy)), int(round(cx + dx))
    py0 = tcy - nh // 2; px0 = tcx - nw // 2
    sy0 = max(0, -py0); sx0 = max(0, -px0)
    sy1 = min(nh, h - py0); sx1 = min(nw, w - px0)
    if sy1 <= sy0 or sx1 <= sx0:
        return g
    dst = out[py0 + sy0:py0 + sy1, px0 + sx0:px0 + sx1]
    m = pm2[sy0:sy1, sx0:sx1]
    src = patch2[sy0:sy1, sx0:sx1]
    dst[m] = src[m]
    return out


def samples_of(items, plate, mode, epochs=1):
    out = []
    for it in items:
        g = gray(it['path'])
        out.append((bits_of_gray(g), 'raw'))
        if mode == 'aug':
            for k, dx, dy in AUG_SPECS:
                try:
                    out.append((bits_of_gray(synth_hand(g, plate, k, dx, dy)), 'aug'))
                except Exception:
                    pass
    return out


def acc_q6(m, X, y):
    with torch.no_grad():
        p = T10.hw_eval_q6(m, X).argmax(1)
    return float((p == y).float().mean()), p


def g4(m, X, y):
    with torch.no_grad():
        q6 = T10.hw_eval_q6(m, X).argmax(1)
        fl = m(X).argmax(1)
    return int((q6 == fl).sum()), len(y)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default='2026-11-01', help='采集日期 YYYY-MM-DD')
    ap.add_argument('--from', dest='t_from', default='')
    ap.add_argument('--to', dest='t_to', default='')
    ap.add_argument('--ckpt-out', default=os.path.join(W, 'ONSITE_AUG.pt'))
    ap.add_argument('--blob-out', default=os.path.join(W, 'hw_onsite_aug'))
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--lr', type=float, default=LR)
    ap.add_argument('--skip-export', action='store_true')
    a = ap.parse_args()
    global DATE
    DATE = a.date
    if not a.t_from:
        a.t_from, a.t_to = '21:20', '21:40'
    items = load_session(ts(a.t_from), ts(a.t_to) if a.t_to else ts('23:59'))
    from collections import Counter
    print('  session frames=%d per-class=%s' % (len(items), dict(Counter(it['cls'] for it in items))))
    if len(items) < 80:
        print('[FAIL] session too small'); return 1
    plate = make_plate(items)
    print('  background plate mean=%.1f' % plate.mean())
    # split per class 70/10/20
    rng = random.Random(SPLIT_SEED)
    by = {}
    for it in items:
        by.setdefault(it['y'], []).append(it)
    tr, va, te = [], [], []
    for c, lst in by.items():
        lst = list(lst); rng.shuffle(lst)
        n = len(lst); nte = max(1, int(round(n * 0.2))); nva = max(1, int(round(n * 0.1)))
        te += lst[:nte]; va += lst[nte:nte + nva]; tr += lst[nte + nva:]
    print('  split train=%d val=%d test=%d' % (len(tr), len(va), len(te)))
    Xva = torch.from_numpy(np.stack([bits_of_gray(gray(it['path'])) for it in va]))[:, None].float()
    yva = torch.tensor([it['y'] for it in va])
    Xte = torch.from_numpy(np.stack([bits_of_gray(gray(it['path'])) for it in te]))[:, None].float()
    yte = torch.tensor([it['y'] for it in te])
    # synthetic robustness tests
    tests = {}
    for tag, k, dx, dy in (('手×1.2', 1.2, 0, 0), ('手×1.4', 1.4, 0, 0), ('手×1.6', 1.6, 0, 0),
                           ('位偏5%', 1.0, 64, 0), ('位偏10%', 1.0, 128, 0), ('位偏15%', 1.0, 192, 0)):
        tests[tag] = (torch.from_numpy(np.stack([bits_of_gray(synth_hand(gray(it['path']), plate, k, dx, dy)) for it in te]))[:, None].float(), yte)
    Xr, yr, Xsd, ysd = F.build_split()
    torch.manual_seed(1); np.random.seed(1); random.seed(1)
    m = L8.Net8()
    if os.path.exists(BASE):
        m.load_state_dict(torch.load(BASE, map_location='cpu'))
    else:
        print('[WARN] base ckpt not shipped (%s) -> training from random init' % BASE)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr); ce = nn.CrossEntropyLoss()
    tr_samples = samples_of(tr, plate, 'aug')
    X = torch.from_numpy(np.stack([s[0] for s in tr_samples]))[:, None].float().repeat(OVERSAMPLE, 1, 1, 1)
    # 构造与 samples 对应的标签
    ys = []
    for it in tr:
        ys.append(it['y'])
        for k, dx, dy in AUG_SPECS:
            ys.append(it['y'])
    y = torch.tensor(ys).repeat(OVERSAMPLE)
    Xall = torch.cat([X, Xr], 0); yall = torch.cat([y, yr], 0)
    n = len(Xall)
    best = None
    for ep in range(1, a.epochs + 1):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad(); ce(m(Xall[idx]), yall[idx]).backward(); opt.step()
        if ep % 3 == 0 or ep == a.epochs:
            m.eval()
            v = acc_q6(m, Xva, yva)[0]
            if best is None or v > best['val']:
                best = {'ep': ep, 'val': v, 'state': {k: t.clone() for k, t in m.state_dict().items()}}
            print('  [ep%02d] val=%.3f' % (ep, v), flush=True)
    m.load_state_dict(best['state']); m.eval()
    raw_test = acc_q6(m, Xte, yte)[0]
    print('\n  ==== ONSITE-AUG RESULT ====')
    print('  best val=%.3f @ep%d' % (best['val'], best['ep']))
    print('  **T1b onsite(原帧 test) = %.4f**' % raw_test)
    for tag, (Xt, yt) in tests.items():
        print('    %-8s : %.4f' % (tag, acc_q6(m, Xt, yt)[0]))
    g = {}
    for tag, Xx, yy in (('test', Xte, yte), ('rps', Xsd, ysd)):
        g[tag] = g4(m, Xx, yy)
    print('  G4:', {k: '%d/%d' % v for k, v in g.items()})
    ok = all(v[0] == v[1] for v in g.values())
    torch.save(best['state'], a.ckpt_out)
    print('  ckpt ->', a.ckpt_out)
    if not a.skip_export and ok:
        os.makedirs(a.blob_out, exist_ok=True)
        r = subprocess.run([sys.executable, os.path.join(ROOT, 'src', 'scripts', 'gen_weights.py'),
                            '--ckpt', a.ckpt_out, '--outdir', a.blob_out],
                           capture_output=True, text=True, encoding='utf-8', errors='ignore')
        print('  [gen_weights]', (r.stdout or '').strip().splitlines()[-1:] if r.stdout else r.stderr[-200:])
        hexp = os.path.join(a.blob_out, 'weights.hex')
        if r.returncode != 0 or ('PASS' not in (r.stdout or '')) or not os.path.exists(hexp):
            print('[FAIL] gen_weights failed -> blob not usable'); return 1
        if len(open(hexp, encoding='utf-8').read().split()) != 271:
            print('[FAIL] weights.hex is not 271 words -> blob not usable'); return 1
        bins = os.path.join(ROOT, 'sim', 'onsite_aug_test_bins')
        os.makedirs(bins, exist_ok=True)
        for f in os.listdir(bins):
            os.remove(os.path.join(bins, f))
        rows = []
        for k, it in enumerate(te):
            _, b, _ = P1.to_frame(Image.open(it['path']))
            nm = '%s_%04d.bin' % (it['cls'], k)
            open(os.path.join(bins, nm), 'wb').write(np.packbits(b.reshape(-1)).tobytes())
            rows.append((nm, it['cls']))
        with open(os.path.join(bins, 'labels.csv'), 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh); w.writerow(['file', 'class']); w.writerows(rows)
        print('  board bins ->', bins)
    json.dump({'val': best['val'], 'ep': best['ep'], 'raw_test': raw_test,
               'robust': {k: acc_q6(m, X, y)[0] for k, (X, y) in tests.items()}, 'g4_ok': ok},
              open(os.path.join(ROOT, 'sim', '20261102_onsite_aug.json'), 'w'), indent=2, ensure_ascii=False)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
