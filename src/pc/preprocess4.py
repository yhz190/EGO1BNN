#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""preprocess4.py -- exposure-normalised binarisation (fixes webcam AE drift).

Why: even with CAP_PROP_AUTO_EXPOSURE "locked", the built-in webcam still drifts (measured
bg_level 151 -> 128 -> 132 inside two minutes). Because the binary frame depends on the Otsu
threshold, AE drift changes the pattern (that is what turned one session into two domains).

Fix: normalise the grayscale so the background level is constant BEFORE Otsu:
    bg   = p90(gray)                      # bright pixels ~ background
    g'   = gray * (TARGET_BG / bg)        # exposure-invariant
    bits = Otsu(g') > thr(g')             # same downstream contract (512 B)
The trained model must use the SAME pipeline (train and inference consistent).

Legacy `preprocess.to_frame` stays untouched for existing models.
"""
import numpy as np
from PIL import Image

SIZE = 64
TARGET_BG = 150.0


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


def normalize_gray(gray_u8, target_bg=TARGET_BG):
    """scale so the background (p90) becomes target_bg; returns uint8 image and the gain"""
    g = gray_u8.astype(np.float32)
    bg = float(np.percentile(g, 90))
    if bg < 1.0:
        bg = 1.0
    gain = target_bg / bg
    return np.clip(g * gain, 0, 255).astype(np.uint8), gain


def to_frame_norm(img, size=SIZE, target_bg=TARGET_BG):
    """same contract as preprocess.to_frame: (packed512, bits[size,size] uint8, thr)"""
    g = img.convert('L')
    w, h = g.size
    s = min(w, h)
    g = g.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((size, size), Image.BILINEAR)
    gray = np.asarray(g, np.uint8)
    norm, gain = normalize_gray(gray, target_bg)
    thr = otsu_u8(norm)
    bits = (norm > thr).astype(np.uint8)
    return np.packbits(bits.reshape(-1)).tobytes(), bits, thr
