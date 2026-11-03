#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_b5_test_bins.py -- export the B5 held-out test frames as 512 B .bin for board injection.

Same split as train_b5.py (per-class 70/10/20, seed 7) -> test set only.
Filenames keep the class prefix so the board tester can derive the true label.
"""
import os, sys, csv, time, random, glob
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'scripts'))
import preprocess as P1
from class_map import CLASS_MAP, CLASS_ID

LIVE = os.path.join(ROOT, 'src', 'data', 'live')
OUT = os.path.join(ROOT, 'sim', '20261101_b5_test_bins')
T5 = time.mktime(time.strptime('2026-11-01 15:00:00', '%Y-%m-%d %H:%M:%S'))
SPLIT_SEED = 7


def main():
    os.makedirs(OUT, exist_ok=True)
    man = {r['file']: r for r in csv.DictReader(open(os.path.join(LIVE, '_manifest.csv'), encoding='utf-8'))}
    by = {}
    for cls in CLASS_MAP:
        d = os.path.join(LIVE, cls)
        for f in sorted(x for x in os.listdir(d) if x.endswith('.png')):
            n = f[:-4]
            if int(man.get(n, {'ts': 0})['ts']) < T5:
                continue
            by.setdefault(CLASS_ID[cls], []).append(os.path.join(d, f))
    rng = random.Random(SPLIT_SEED)
    rows = []
    n_written = 0
    for c in sorted(by):
        lst = by[c]
        idx = list(range(len(lst)))
        rng.shuffle(idx)
        n = len(lst)
        n_te = max(1, int(round(n * 0.20)))
        for k, i in enumerate(idx[:n_te]):
            src = lst[i]
            _, bits, _ = P1.to_frame(Image.open(src))
            packed = np.packbits(bits.reshape(-1)).tobytes()
            assert len(packed) == 512
            name = '%s_%04d' % (CLASS_MAP[c], n_written)
            open(os.path.join(OUT, name + '.bin'), 'wb').write(packed)
            rows.append((name + '.bin', CLASS_MAP[c]))
            n_written += 1
    with open(os.path.join(OUT, 'labels.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['file', 'class'])
        w.writerows(rows)
    print('  wrote %d test frames -> %s' % (n_written, OUT))
    print('  per class:', {c: sum(1 for r in rows if r[1] == c) for c in CLASS_MAP})


if __name__ == '__main__':
    main()
