#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""preprocess3.py -- background-subtraction + hand-centred scale normalisation (PC side only).

Motivation (diagnosis 2026-11-01): the binary frame is extremely sensitive to hand geometry;
a "small" camera/distance change flips 26-49% of the 64x64 pixels, far beyond the model's
~10% noise budget. The hardware contract is only "512-byte packed 1-bit frame", so the PC-side
preprocessing can normalise the hand before binarisation.

Pipeline
  1. background reference = median of N empty-scene frames (captured once per environment)
  2. diff = |gray - bg|, blurred; threshold = max(16, otsu(diff)) -> hand mask
  3. largest connected component, centroid; side = sqrt(area / TARGET_FRAC) (scale norm)
  4. crop that square from the ORIGINAL gray frame -> resize 64 -> Otsu -> packbits

Fallback: legacy centre-crop `preprocess.to_frame` when no hand is found.
"""
import os
import numpy as np
from PIL import Image
import cv2

SIZE = 64
TARGET_FRAC = 0.30          # hand should occupy ~30% of the 64x64 crop (RPS-like)
MIN_AREA_FRAC = 0.004


def read_gray(path):
    """Unicode-safe grayscale read (cv2.imread fails on non-ASCII paths)."""
    return np.asarray(Image.open(path).convert('L'), np.uint8)


def otsu_u8(a):
    hist = np.bincount(a.reshape(-1), minlength=256).astype(np.float64)
    total = a.size
    sum_all = np.dot(np.arange(256), hist)
    wB = np.cumsum(hist); sumB = np.cumsum(np.arange(256) * hist); wF = total - wB
    valid = (wB > 0) & (wF > 0)
    if not valid.any():
        return 128
    mB = np.zeros(256); mF = np.zeros(256)
    mB[valid] = sumB[valid] / wB[valid]
    mF[valid] = (sum_all - sumB[valid]) / wF[valid]
    var = np.zeros(256); var[valid] = wB[valid] * wF[valid] * (mB[valid] - mF[valid]) ** 2
    return int(np.argmax(var))


def estimate_background(paths):
    """median of several empty-scene frames -> float32 background"""
    arrs = [read_gray(p).astype(np.float32) for p in paths]
    h = min(a.shape[0] for a in arrs)
    w = min(a.shape[1] for a in arrs)
    arrs = [a[:h, :w] for a in arrs]
    return np.median(np.stack(arrs), axis=0)


def to_frame_v3(gray_u8, bg):
    """returns (packed512, bits64x64, info dict)"""
    g = gray_u8.astype(np.float32)
    h, w = g.shape
    bgh, bgw = bg.shape
    if (bgh, bgw) != (h, w):
        bg2 = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        bg2 = bg
    diff = np.abs(g - bg2)
    d8 = np.clip(diff, 0, 255).astype(np.uint8)
    d8 = cv2.GaussianBlur(d8, (5, 5), 0)
    t = max(16, otsu_u8(d8))
    mask = (d8 > t).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    info = {'mode': 'legacy', 'area_frac': 0.0, 'side': 0}
    if n <= 1:
        raise RuntimeError('no hand component')
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[idx, cv2.CC_STAT_AREA])
    area_frac = area / float(mask.size)
    if area < MIN_AREA_FRAC * mask.size:
        raise RuntimeError('hand too small')
    cx, cy = float(cent[idx][0]), float(cent[idx][1])
    side = int(round((area / TARGET_FRAC) ** 0.5))
    side = max(48, min(side, min(h, w)))
    x0 = int(max(0, min(w - side, cx - side / 2)))
    y0 = int(max(0, min(h - side, cy - side / 2)))
    crop = gray_u8[y0:y0 + side, x0:x0 + side]
    crop = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    t2 = otsu_u8(crop)
    bits = (crop > t2).astype(np.uint8)
    info = {'mode': 'bgsub', 'area_frac': area_frac, 'side': side, 'thr': t2}
    return np.packbits(bits.reshape(-1)).tobytes(), bits, info


def load_background_from_session(junk_dir, after_ts=None, n=20):
    """pick empty-scene frames from a junk directory (flattest n frames)"""
    import csv, time
    man_p = os.path.join(os.path.dirname(junk_dir), '_manifest.csv')
    man = {}
    if os.path.exists(man_p):
        man = {r['file']: int(r['ts']) for r in csv.DictReader(open(man_p, encoding='utf-8'))}
    cand = []
    for f in sorted(x for x in os.listdir(junk_dir) if x.endswith('.png')):
        nme = f[:-4]
        ts = man.get(nme, 0)
        if after_ts is not None and ts < after_ts:
            continue
        g = read_gray(os.path.join(junk_dir, f))
        cand.append((float(g.std()), os.path.join(junk_dir, f)))
    cand.sort(key=lambda x: x[0])
    return [p for _, p in cand[:n]]
