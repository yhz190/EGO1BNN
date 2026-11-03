#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gen_golden_t8_v2.py -- T8 golden vectors for the PATCH2 architecture (8ch + Q6).

Semantics (single source = src/weights_layout_spec.md v4):
  L1: x1 = (matches_fixed(X, w1, pad=1) > floor(theta1))        # 8 planes, 64x64
  L2: x2 = (matches_fixed(x1, w2, pad=1, groups=8) > floor(theta2))
  L3: x3 = (matches_fixed(x2, w3, pad=0) > floor(theta3))       # 8 in -> 8 out
  L4: 4x4 majority (sum >= 8) over x3 -> 8 x 16 x 16 = 2048 bit
  L5: mf = matches_fixed(feat, wf, pad=0); score_q = 64*(2*mf - 2048) + round(bias*64)   # Q6
Output: data/t8_vectors_v2/  (golden.npz + report.json + 3 sample .hex + README)
Self-checks: replay from dumped L4 -> argmax 620/620; Q6 vs float reference argmax 620/620.
"""
import os, sys, csv, json
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'scripts'))
import train_bnn10 as T10
import train_l3x8 as L8
from class_map import assert_class_map

assert_class_map()
torch.set_num_threads(4)
FC_SCALE = 64

VEC = os.path.join(ROOT, 'sim', 'vectors_t1a')
CKPT = os.path.join(ROOT, 'src', 'weights', 'FORMAL_l3x8_s4.pt')
OUT = os.path.join(ROOT, 'sim', '20261101_t8_vectors_v2')
os.makedirs(OUT, exist_ok=True)


def load_frames():
    rows = list(csv.DictReader(open(os.path.join(VEC, 'labels.csv'), encoding='utf-8-sig')))
    X = np.zeros((len(rows), 1, 64, 64), np.float32)
    for i, r in enumerate(rows):
        raw = np.fromfile(os.path.join(VEC, r['file'].strip()), np.uint8)
        X[i, 0] = np.unpackbits(raw)[:4096].reshape(64, 64)
    y = np.array([int(r['class']) for r in rows], np.int64)
    return [r['file'].strip() for r in rows], torch.from_numpy(X), y


def packbits_t(t):
    a = t.detach().cpu().numpy().astype(np.uint8).reshape(t.shape[0], t.shape[1], -1)
    return np.packbits(a, axis=2)


def main():
    files, X, y = load_frames()
    sd = torch.load(CKPT, map_location='cpu')
    m = L8.Net8()
    m.load_state_dict(sd)
    m.eval()
    feat = m.feat
    print('[model] %s  feat=%d  L3 out=%d' % (os.path.basename(CKPT), feat, m.c3.weight.shape[0]))

    w1 = (m.c1.weight > 0).float()
    w2 = (m.dw.weight > 0).float()
    w3 = (m.c3.weight > 0).float()
    wf = (m.fc.weight > 0).float()
    t1i = T10.theta_of(m.b1, m.c1.weight, 9, False).floor().long()
    t2i = T10.theta_of(m.b2, m.dw.weight, 9, True).floor().long()
    t3i = T10.theta_of(m.b3, m.c3.weight, 8, True).floor().long()
    bias_q = (m.fc.bias.detach() * FC_SCALE).round().long()
    print('[theta floor] L1', t1i.tolist())
    print('[theta floor] L2', t2i.tolist())
    print('[theta floor] L3', t3i.tolist())
    print('[bias_q Q6] ', bias_q.tolist())

    with torch.no_grad():
        c1 = T10.matches_fixed(X, w1, 1)
        x1 = (c1 > t1i.view(1, -1, 1, 1)).float()
        c2 = T10.matches_fixed(x1, w2, 1, groups=8)
        x2 = (c2 > t2i.view(1, -1, 1, 1)).float()
        c3 = T10.matches_fixed(x2, w3, 0)
        x3 = (c3 > t3i.view(1, -1, 1, 1)).float()
        s = F.avg_pool2d(x3, 4) * 16.0
        fh = (s >= 8.0).float().flatten(1)                      # [N, 2048]
        mf = T10.matches_fixed(fh.reshape(-1, feat, 1, 1), wf.reshape(4, feat, 1, 1), 0).reshape(-1, 4)
        score_q = (2 * mf - feat).long() * FC_SCALE + bias_q.view(1, 4)
        pred_q = score_q.argmax(1)
        ref = T10.hw_eval(m, X)                                 # float reference
        pred_ref = ref.argmax(1)

    agree = int((pred_q == pred_ref).sum())
    acc_q = float((pred_q == torch.from_numpy(y)).float().mean())
    acc_ref = float((pred_ref == torch.from_numpy(y)).float().mean())
    print('[check] Q6 vs float-argmax : %d/%d' % (agree, len(y)))
    print('[acc]   Q6 = %.4f   float-ref = %.4f' % (acc_q, acc_ref))

    np.savez_compressed(
        os.path.join(OUT, 'golden.npz'),
        files=np.array(files), labels=y,
        L1=packbits_t(x1), L2=packbits_t(x2), L3=packbits_t(x3),
        c1=c1.numpy().astype(np.uint8), c2=c2.numpy().astype(np.uint8), c3=c3.numpy().astype(np.uint8),
        L4=np.packbits(fh.numpy().astype(np.uint8), axis=1),
        L5=score_q.numpy().astype(np.int32),
        pred=pred_q.numpy().astype(np.uint8),
    )
    meta = {
        'arch': 'patch2: L3=8ch, feat=2048, FC=4x2048, Q6 bias',
        'ckpt': os.path.basename(CKPT),
        'fc_scale': FC_SCALE,
        'theta_int_floor': {'L1': t1i.tolist(), 'L2': t2i.tolist(), 'L3': t3i.tolist()},
        'bias_q': bias_q.tolist(),
        'frames': len(files),
        'acc_q6': acc_q,
        'acc_float_ref': acc_ref,
        'q6_vs_float_agree': agree,
        'layout': 'L1/L2 [N,8,64,64] bit; L3 [N,8,64,64] bit; L4 2048 bit; L5 int32 Q6 scores',
        'note': 'bit-domain pad 0 == -1 in pm1 domain; theta floor; Q6 FC bias',
    }

    # replay self-check from the dumped L4
    z = np.load(os.path.join(OUT, 'golden.npz'))
    L4 = np.unpackbits(z['L4'], axis=1)[:, :feat].astype(np.float32)
    with torch.no_grad():
        mf2 = T10.matches_fixed(torch.from_numpy(L4).reshape(-1, feat, 1, 1),
                                wf.reshape(4, feat, 1, 1), 0).reshape(-1, 4)
        sc2 = (2 * mf2 - feat).long() * FC_SCALE + bias_q.view(1, 4)
    replay_ok = int((sc2.argmax(1) == pred_q).sum())
    meta['replay_argmax_match'] = replay_ok
    print('[replay] %d/%d' % (replay_ok, len(files)))
    json.dump(meta, open(os.path.join(OUT, 'report.json'), 'w'), indent=2)

    # samples for module-level TB
    for k in range(3):
        lines = ['# frame %d file=%s label=%d pred_q6=%d' % (k, files[k], y[k], pred_q[k])]
        for name, t in (('L1', x1), ('L2', x2), ('L3', x3)):
            lines.append('[%s]' % name)
            arr = t[k].numpy().astype(np.uint8)
            for c in range(arr.shape[0]):
                for yy in range(64):
                    lines.append(name + ' ' + ''.join('%02x' % b for b in np.packbits(arr[c, yy])))
        fb = fh[k].numpy().astype(np.uint8)
        lines.append('[L4]')
        for yy in range(16):
            lines.append('L4 ' + ''.join('%02x' % b for b in np.packbits(fb[yy * 128:(yy + 1) * 128])))
        lines.append('[L5] scores_q6 = %s  pred=%d' % (score_q[k].tolist(), pred_q[k]))
        open(os.path.join(OUT, 'sample_frame_%d.hex' % k), 'w').write('\n'.join(lines) + '\n')

    print('[out]', OUT)
    ok = (agree == len(y)) and (replay_ok == len(y))
    print('[RESULT] %s' % ('ALL PASS — vectors valid for RTL sign-off' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
