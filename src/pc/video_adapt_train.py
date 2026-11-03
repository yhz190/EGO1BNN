#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""video_adapt_train.py -- 用"演示录屏里的石头片段"做域适配微调（新增脚本，2026-11-02）

背景：用户录制 27 s 视频（仅做石头）→ 6~22 s 连续握拳，板端却判 JUNK。
做法：
  1. 从录屏提取 6~22 s 的帧，反推相机画面，**去除叠加层**（标签/目标框/分数条/fps）；
  2. 这些帧伪标签为 rock；另取 1.5~2.5 s / 23~27 s 的空画面伪标签 junk；
  3. 与 12:13 会话的 371 帧（手工标签）一起，从 ONSITE_AUG_20261102.pt 微调（小 lr、少 epoch）；
  4. 验证：会话留出准确率（不能退化）+ 录屏石头召回率（应显著上升）；
  5. 通过则导出 v5 blob（供烧录）。
产物只写 PC 端工作目录（EGO1_LIVE_DIAG_DIR，默认仓库上级目录/live_diag；不碰项目数据）。
"""
import os, sys, csv, time, random, json, subprocess
import numpy as np
import cv2
from PIL import Image
import torch, torch.nn as nn

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
OUT = os.environ.get('EGO1_LIVE_DIAG_DIR', os.path.join(os.path.dirname(ROOT), 'live_diag'))
os.makedirs(OUT, exist_ok=True)
V = os.environ.get('EGO1_DEMO_VIDEO', '')
_BASE_CANDS = [os.path.join(W, 'ONSITE_ADAPT_20261102.pt'),
               os.path.join(W, 'ONSITE_AUG_20261102.pt'),
               os.path.join(W, 'AB_A_l3x8_80ep_s2.pt')]
BASE_CKPT = next((p for p in _BASE_CANDS if os.path.exists(p)), _BASE_CANDS[-1])


def read_session():
    man = {r['file']: int(r['ts']) for r in csv.DictReader(open(os.path.join(LIVE, '_manifest.csv'), encoding='utf-8'))}
    lo = time.mktime(time.strptime('2026-11-02 12:10', '%Y-%m-%d %H:%M'))
    items = []
    for cls in CLASS_MAP:
        d = os.path.join(LIVE, cls)
        for f in sorted(os.listdir(d)):
            if f.endswith('.png') and man.get(f[:-4], 0) >= lo:
                items.append({'cls': cls, 'y': CLASS_ID[cls], 'path': os.path.join(d, f)})
    return items


# ---------------- 录屏 → 相机画面（去叠加层） ----------------
def open_video():
    cap = cv2.VideoCapture(V)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, fr = cap.read()
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    rows = np.where(g.mean(axis=1) > 5)[0]
    cols = np.where(g.mean(axis=0) > 5)[0]
    cap.release()
    return fps, n, (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)


def canvas_of(fr, box):
    x0, y0, x1, y1 = box
    c = fr[y0:y1, x0:x1]
    c = cv2.resize(c, (1920, 1080), interpolation=cv2.INTER_AREA)
    return cv2.resize(c, (1280, 720), interpolation=cv2.INTER_AREA)


def strip_overlays(cam):
    """去掉 demo 叠加层：顶栏、左下分数条、右下 fps、目标框黄线（inpaint）"""
    img = cam.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    # 顶栏（标签+副标题）
    mask[0:int(h * 0.17), 0:int(w * 0.60)] = 255
    # 左下分数条（含 4 行文字）
    mask[int(h * 0.68):int(h * 0.98), 0:int(w * 0.55)] = 255
    # 右下 fps
    mask[int(h * 0.85):int(h * 0.99), int(w * 0.83):] = 255
    # 目标框黄线
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow = ((hsv[..., 0] >= 22) & (hsv[..., 0] <= 38) & (hsv[..., 1] > 120) & (hsv[..., 2] > 120)).astype(np.uint8) * 255
    mask = np.maximum(mask, yellow)
    out = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    return out


def video_frames(box, fps, t0, t1, step=2):
    cap = cv2.VideoCapture(V)
    out = []
    for f in range(int(t0 * fps), int(t1 * fps), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            continue
        cam = canvas_of(fr, box)
        cam = strip_overlays(cam)
        out.append(cam)
    cap.release()
    return out


def bits_of_bgr(cam):
    rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    _, b, _ = P1.to_frame(Image.fromarray(rgb))
    return b


def aug_bits(bits, rng):
    b = bits.copy()
    if rng.random() < 0.5:
        k = rng.choice([1.12, 1.25, 0.88])
        h, w = b.shape
        m = (1 - b).astype(np.float32)
        ys, xs = np.nonzero(m)
        if len(ys) > 10:
            cy, cx = ys.mean(), xs.mean()
            yy, xx = np.mgrid[0:h, 0:w]
            iy = np.clip(((yy - cy) / k + cy).round().astype(int), 0, h - 1)
            ix = np.clip(((xx - cx) / k + cx).round().astype(int), 0, w - 1)
            b = b[iy, ix]
    if rng.random() < 0.6:
        dx, dy = rng.choice([-4, -2, 0, 2, 4]), rng.choice([-3, 0, 3])
        b = np.roll(np.roll(b, dy, axis=0), dx, axis=1)
    return b


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--ckpt-out', default=os.path.join(OUT, 'ADAPT.pt'))
    ap.add_argument('--blob-out', default=os.path.join(OUT, 'hw_adapt'))
    a = ap.parse_args()

    fps, n, box = open_video()
    print('  video fps=%.1f frames=%d box=%s' % (fps, n, box))
    # 录屏数据：rock = 6~22s（全部握拳）；junk = 1.5~2.5s 与 24~27s（空画面）
    rock_cams = video_frames(box, fps, 6.0, 22.0, step=2)
    junk_cams = video_frames(box, fps, 1.5, 2.5, step=3) + video_frames(box, fps, 24.0, 27.0, step=3)
    print('  video frames: rock=%d junk=%d' % (len(rock_cams), len(junk_cams)))
    # 视觉证据：存 8 张清洗后的帧
    for i, c in enumerate(rock_cams[:4] + junk_cams[:2]):
        cv2.imwrite(os.path.join(OUT, 'clean_%02d.png' % i), c)
    # 会话数据
    items = read_session()
    if len(items) < 40:
        print('[FAIL] no live session found (data/live 为空)。')
        print('       请先用 start_capture.cmd 采集一组会话（4 类各 30~40 帧），再运行本脚本。')
        return 2
    rng = random.Random(7)
    by = {}
    for it in items:
        by.setdefault(it['y'], []).append(it)
    tr, va = [], []
    for c, lst in by.items():
        lst = list(lst); rng.shuffle(lst)
        nte = max(1, int(round(len(lst) * 0.2)))
        va += lst[:nte]; tr += lst[nte:]
    print('  session train=%d val=%d' % (len(tr), len(va)))

    Xva = torch.from_numpy(np.stack([bits_of_bgr(cv2.cvtColor(np.asarray(Image.open(it['path']).convert('RGB')), cv2.COLOR_RGB2BGR)) for it in va]))[:, None].float()
    yva = torch.tensor([it['y'] for it in va])
    Xrock = torch.from_numpy(np.stack([bits_of_bgr(c) for c in rock_cams]))[:, None].float()
    Xjunk = torch.from_numpy(np.stack([bits_of_bgr(c) for c in junk_cams]))[:, None].float()

    m = L8.Net8(); m.load_state_dict(torch.load(BASE_CKPT, map_location='cpu'))
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    ce = nn.CrossEntropyLoss()
    # 训练集：会话帧 + 录屏帧（含增广）
    rng2 = random.Random(11)
    samples = []
    for it in tr:
        rgb = np.asarray(Image.open(it['path']).convert('RGB'))
        b = bits_of_bgr(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        samples.append((b, it['y']))
        samples.append((aug_bits(b, rng2), it['y']))
    for b in [bits_of_bgr(c) for c in rock_cams]:
        samples.append((b, CLASS_ID['rock']))
        samples.append((aug_bits(b, rng2), CLASS_ID['rock']))
    for b in [bits_of_bgr(c) for c in junk_cams]:
        samples.append((b, CLASS_ID['junk']))
        samples.append((aug_bits(b, rng2), CLASS_ID['junk']))
    X = torch.from_numpy(np.stack([s[0] for s in samples]))[:, None].float()
    y = torch.tensor([s[1] for s in samples])
    Xr, yr, _, _ = F.build_split()
    Xall = torch.cat([X, Xr], 0); yall = torch.cat([y, yr], 0)
    print('  total train samples=%d (+RPS)' % len(Xall))
    best = None
    for ep in range(1, a.epochs + 1):
        m.train(); perm = torch.randperm(len(Xall))
        for i in range(0, len(Xall), 128):
            idx = perm[i:i + 128]
            opt.zero_grad(); ce(m(Xall[idx]), yall[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = float((T10.hw_eval_q6(m, Xva).argmax(1) == yva).float().mean())
            pr = T10.hw_eval_q6(m, Xrock).argmax(1)
            rock_recall = float((pr == CLASS_ID['rock']).float().mean())
            pj = T10.hw_eval_q6(m, Xjunk).argmax(1)
            junk_recall = float((pj == CLASS_ID['junk']).float().mean())
        score = 0.5 * v + 0.5 * rock_recall
        if best is None or score > best[0]:
            best = (score, ep, {k: val.clone() for k, val in m.state_dict().items()}, v, rock_recall, junk_recall)
        print('  [ep%02d] 会话留出=%.3f  录屏石头召回=%.3f  录屏空画面(junk)召回=%.3f  score=%.3f%s'
              % (ep, v, rock_recall, junk_recall, score, '  <best' if best[1] == ep else ''), flush=True)
    m.load_state_dict(best[2]); m.eval()
    print('  BEST ep%d: 会话留出=%.3f 石头召回=%.3f' % (best[1], best[3], best[4]))
    # G4
    g = {}
    with torch.no_grad():
        q6 = T10.hw_eval_q6(m, Xva).argmax(1); fl = m(Xva).argmax(1)
        g['val'] = (int((q6 == fl).sum()), len(yva))
    print('  G4:', {k: '%d/%d' % v for k, v in g.items()})
    ok = g['val'][0] == g['val'][1]
    torch.save(m.state_dict(), a.ckpt_out)
    print('  ckpt ->', a.ckpt_out)
    if ok:
        os.makedirs(a.blob_out, exist_ok=True)
        r = subprocess.run([sys.executable, os.path.join(ROOT, 'src', 'scripts', 'gen_weights.py'),
                            '--ckpt', a.ckpt_out, '--outdir', a.blob_out],
                           capture_output=True, text=True, encoding='utf-8', errors='ignore')
        print('  [gen_weights]', (r.stdout or '').strip().splitlines()[-1:] if r.stdout else r.stderr[-200:])
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
