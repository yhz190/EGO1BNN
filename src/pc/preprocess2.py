#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""preprocess2.py -- position/scale-normalised preprocessing (PC side only).

v2.1: use the LARGEST CONNECTED DARK COMPONENT (the hand) to build the crop box.
      Global row/col projections were too permissive: with background clutter the bbox
      degenerated to the whole frame and v2 == legacy.

Algorithm
  1. grey -> 1/4 size (analysis image)
  2. Otsu -> dark mask (hand is darker than the light background in this project's data)
  3. connected components (BFS), pick the largest dark component
  4. bbox of that component, expand to square side = 1.30 x max(w,h), clamp to image
  5. crop -> resize 64x64 -> Otsu again -> packbits (MSB first; bit=1 bright, same as load_bin)
Fallback: legacy centre crop when no dark component of reasonable size exists.
"""
import numpy as np
from PIL import Image

SIZE = 64
TARGET_HAND_FRAC = 0.29
TARGET_BIT1 = 0.71   # RPS training distribution mean   # RPS training data: hand ~29% of pixels (bit1 ~0.71)


def otsu_uint8(a):
    hist = np.bincount(a.reshape(-1), minlength=256).astype(np.float64)
    total = a.size
    sum_all = np.dot(np.arange(256), hist)
    wB = np.cumsum(hist)
    sumB = np.cumsum(np.arange(256) * hist)
    wF = total - wB
    valid = (wB > 0) & (wF > 0)
    if not valid.any():
        return 128
    mB = np.zeros(256)
    mF = np.zeros(256)
    mB[valid] = sumB[valid] / wB[valid]
    mF[valid] = (sum_all - sumB[valid]) / wF[valid]
    var = np.zeros(256)
    var[valid] = wB[valid] * wF[valid] * (mB[valid] - mF[valid]) ** 2
    return int(np.argmax(var))


def largest_component_bbox(dark):
    """BFS over the dark mask; return (x0, y0, x1, y1, area) of the largest component."""
    h, w = dark.shape
    seen = np.zeros_like(dark, dtype=bool)
    best = None
    for sy in range(h):
        for sx in range(w):
            if not dark[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            area = 0
            x0 = x1 = sx
            y0 = y1 = sy
            while stack:
                y, x = stack.pop()
                area += 1
                if x < x0: x0 = x
                if x > x1: x1 = x
                if y < y0: y0 = y
                if y > y1: y1 = y
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and dark[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if best is None or area > best[4]:
                best = (x0, y0, x1, y1, area)
    return best


def to_frame_v2(img: Image.Image, min_area_frac=0.02, expand=1.30):
    g = img.convert('L')
    w, h = g.size
    scale = 4
    aw, ah = max(8, w // scale), max(8, h // scale)
    a = np.asarray(g.resize((aw, ah), Image.BILINEAR), dtype=np.uint8)
    t = otsu_uint8(a)
    dark = (a <= t)
    info = {'mode': 'legacy', 'bbox': None, 'area_frac': 0.0}
    best = largest_component_bbox(dark) if dark.any() else None
    if best is not None:
        x0, y0, x1, y1, area = best
        area_frac = area / float(dark.size)
        if area_frac >= min_area_frac:
            sx, sy = w / float(aw), h / float(ah)
            X0, Y0 = int(x0 * sx), int(y0 * sy)
            X1, Y1 = int((x1 + 1) * sx), int((y1 + 1) * sy)
            cx, cy = (X0 + X1) // 2, (Y0 + Y1) // 2
            # scale-normalisation: make the hand occupy TARGET_HAND_FRAC of the crop
            # (RPS training data: bit1 ~0.71 -> hand ~29% of pixels)
            hand_area_full = area * (w / float(aw)) * (h / float(ah))
            side = int(round((hand_area_full / TARGET_HAND_FRAC) ** 0.5))
            side = max(16, min(side, min(w, h)))
            # adaptive refinement: iterate crop side so that bit1 ~= TARGET_BIT1 (RPS 0.71)
            crop = None
            xa = ya = 0
            cur = side
            for _it in range(4):
                xa = max(0, min(w - cur, cx - cur // 2))
                ya = max(0, min(h - cur, cy - cur // 2))
                cand = g.crop((xa, ya, xa + cur, ya + cur)).resize((SIZE, SIZE), Image.BILINEAR)
                arrc = np.asarray(cand, dtype=np.uint8)
                tc = otsu_uint8(arrc)
                b1 = float((arrc > tc).mean())
                crop = cand
                if abs(b1 - TARGET_BIT1) <= 0.04:
                    break
                factor = (TARGET_BIT1 / max(b1, 1e-3)) ** 0.5 if b1 > 0 else 1.3
                # bit1 too low (too much dark) -> enlarge; too high -> shrink
                nxt = int(round(cur * (factor if b1 < TARGET_BIT1 else 1.0 / 1.25)))
                nxt = max(16, min(nxt, min(w, h)))
                if nxt == cur:
                    break
                cur = nxt
            info = {'mode': 'bbox', 'bbox': (xa, ya, xa + cur, ya + cur), 'area_frac': area_frac}
        else:
            crop = None
    else:
        crop = None
    if crop is None:
        s = min(w, h)
        crop = g.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((SIZE, SIZE), Image.BILINEAR)
    arr = np.asarray(crop, dtype=np.uint8)
    t2 = otsu_uint8(arr)
    bits = (arr > t2).astype(np.uint8)
    return np.packbits(bits.reshape(-1)).tobytes(), bits, t2, info
