#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""preprocess.py -- convert an image into the 64x64 binary frame used by the FPGA.

Pipeline: load -> grayscale -> center-crop (optional) -> resize 64x64 (area)
          -> Otsu binarize -> pack to 512 bytes (row-major, MSB first)

usage:
  python preprocess.py --image hand.jpg --out frame.bin [--preview frame.png]
  python preprocess.py --folder data/rock --outdir frames/
"""
import argparse, os, sys
import numpy as np
from PIL import Image

SIZE = 64
FRAME_BYTES = SIZE * SIZE // 8

def otsu_threshold(gray: np.ndarray) -> int:
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)
    w_b = 0.0; sum_b = 0.0; best_t = 127; best_var = -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0: continue
        w_f = total - w_b
        if w_f == 0: break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var = var; best_t = t
    return int(best_t)

def to_frame(img: Image.Image, crop: bool = True) -> tuple:
    g = img.convert('L')
    if crop:
        w, h = g.size
        s = min(w, h)
        g = g.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    g = g.resize((SIZE, SIZE), Image.BILINEAR)
    a = np.asarray(g, dtype=np.uint8)
    t = otsu_threshold(a)
    b = (a > t).astype(np.uint8)          # 1 = bright (hand)
    packed = np.packbits(b.reshape(-1))   # row-major, MSB first
    return packed.tobytes(), b, t

def synth_pattern(name: str):
    """Generate a synthetic 64x64 binary gesture-like pattern (returns bits 0/1)."""
    a = np.zeros((SIZE, SIZE), dtype=np.uint8)
    if name == 'circle':
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        a[((xx - 32) ** 2 + (yy - 32) ** 2) <= 400] = 1
    elif name == 'cross':
        a[28:36, 8:56] = 1
        a[8:56, 28:36] = 1
    elif name == 'triangle':
        for y in range(12, 52):
            half = (y - 12) * 20 // 40
            a[y, 32 - half:32 + half + 1] = 1
    else:  # checker
        a = (((np.mgrid[0:SIZE, 0:SIZE][0] // 8) + (np.mgrid[0:SIZE, 0:SIZE][1] // 8)) & 1).astype(np.uint8)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image')
    ap.add_argument('--folder')
    ap.add_argument('--out')
    ap.add_argument('--outdir')
    ap.add_argument('--preview')
    ap.add_argument('--no-crop', action='store_true')
    ap.add_argument('--pattern', choices=['circle', 'cross', 'triangle', 'checker'],
                    help='生成合成测试图案（不需要 --image）')
    ap.add_argument('--count', type=int, default=10, help='--pattern 时生成的帧数')

    a = ap.parse_args()

    if a.pattern:
        outdir = a.out or 'frames'
        os.makedirs(outdir, exist_ok=True)
        bits = synth_pattern(a.pattern)
        packed = np.packbits(bits.reshape(-1)).tobytes()
        assert len(packed) == FRAME_BYTES
        for i in range(a.count):
            open(os.path.join(outdir, '%s_%03d.bin' % (a.pattern, i + 1)), 'wb').write(packed)
        print('[ok] %d frames (%s) -> %s  (%d B each)' % (a.count, a.pattern, outdir, FRAME_BYTES))
        return 0

    if a.image:
        frame, bits, th = to_frame(Image.open(a.image), crop=not a.no_crop)
        assert len(frame) == FRAME_BYTES, len(frame)
        out = a.out or 'frame.bin'
        open(out, 'wb').write(frame)
        print('[ok] %s -> %s  (%d bytes, otsu=%d, ones=%d/4096)'
              % (a.image, out, len(frame), th, int(bits.sum())))
        if a.preview:
            Image.fromarray((bits * 255).astype(np.uint8)).save(a.preview)
            print('[ok] preview -> %s' % a.preview)

    if a.folder:
        outdir = a.outdir or 'frames'
        os.makedirs(outdir, exist_ok=True)
        n = 0
        for name in sorted(os.listdir(a.folder)):
            if not name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            frame, bits, th = to_frame(Image.open(os.path.join(a.folder, name)), crop=not a.no_crop)
            base = os.path.splitext(name)[0]
            open(os.path.join(outdir, base + '.bin'), 'wb').write(frame)
            n += 1
        print('[ok] %d images -> %s' % (n, outdir))
    return 0

if __name__ == '__main__':
    sys.exit(main())