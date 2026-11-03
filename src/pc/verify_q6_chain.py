#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""verify_q6_chain.py -- prove the Q6 integer path == float reference argmax, 100%."""
import os, sys, csv
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'scripts'))
torch.set_num_threads(4)
import train_bnn10 as T10
import train_fix as F
import train_l3x8 as L8

Xtr, ytr, Xsd, ysd = F.build_split()

VEC = os.path.join(ROOT, 'sim', 'vectors_t1a')
rows = list(csv.DictReader(open(os.path.join(VEC, 'labels.csv'), encoding='utf-8-sig')))
Xc = np.zeros((len(rows), 1, 64, 64), np.float32)
for i, r in enumerate(rows):
    Xc[i, 0] = np.unpackbits(np.fromfile(os.path.join(VEC, r['file'].strip()), np.uint8))[:4096].reshape(64, 64)
Xc = torch.from_numpy(Xc)
yc = torch.tensor([int(r['class']) for r in rows])

W = os.path.join(ROOT, 'src', 'weights')
CANDS = ['AB_A_l3x8_80ep_s1.pt', 'AB_A_l3x8_80ep_s2.pt', 'AB_A_l3x8_80ep_s3.pt',
         'C_l3x8_40ep_s2.pt', 'C_l3x8_30ep_s1.pt',
         'AB_B_4ch_80ep_s1.pt', 'C_base4ch_80ep_s1.pt']

print('  %-24s %-22s %-22s' % ('ckpt', 'same-domain agree', 'cross-domain agree'))
print('  ' + '-' * 70)
all_ok = True
for name in CANDS:
    p = os.path.join(W, name)
    if not os.path.exists(p):
        print('  %-24s (missing)' % name)
        continue
    sd = torch.load(p, map_location='cpu')
    m = L8.Net8() if sd['c3.weight'].shape[0] == 8 else T10.Net(4)
    m.load_state_dict(sd)
    m.eval()
    with torch.no_grad():
        for tag, X, y in [('same', Xsd, ysd), ('cross', Xc, yc)]:
            ref = T10.hw_eval(m, X).argmax(1)        # float reference
            q6 = T10.hw_eval_q6(m, X).argmax(1)      # Q6 integer path
            agree = int((ref == q6).sum())
            if tag == 'same':
                s1 = '%d/%d' % (agree, len(y))
                ok_s = agree == len(y)
            else:
                s2 = '%d/%d' % (agree, len(y))
                ok_c = agree == len(y)
        all_ok = all_ok and ok_s and ok_c
        print('  %-24s %-22s %-22s %s' % (name, s1, s2, 'PASS' if (ok_s and ok_c) else 'FAIL'))

if not any(os.path.exists(os.path.join(W, n)) for n in CANDS):
    print('\n  [SKIP] no checkpoint available (training ckpts are intentionally not shipped)')
    sys.exit(2)
print('\n  [RESULT] %s' % ('ALL PASS — Q6 path matches float reference on every frame' if all_ok else 'FAILURES PRESENT'))
sys.exit(0 if all_ok else 1)
